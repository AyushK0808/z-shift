from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from spatial_ingestion.reconstruction._io import write_ply
from spatial_ingestion.reconstruction.device import reproducibility_metadata
from spatial_ingestion.reconstruction.models import SyncViewGroup

logger = logging.getLogger(__name__)

_SUPPORTED_FORMATS = {".obj", ".glb", ".ply"}
_tsdf_cuda_hardcode_patched = False
_tsdf_sample_budget_patched = False

# `TSDFPostProcess._refine_depths_with_TSDF` allocates H*W*nsamples float
# tensors per image *before* `TSDF_batchsize` chunking ever applies (its
# vendor default is nsamples=1000, not exposed through __init__). At
# image_size=512 that alone needs several GiB, on top of whatever the
# alignment stage already holds -- on a 16GB T4 with MASt3R loaded, B7/B2
# runs measured as little as ~1.7GiB free at that point, so TSDF fusion
# OOM'd and fell back to the unfused per-view path on every single run.
# These two constants are the levers that keep it inside that budget:
# nsamples bounds the per-image allocation, TSDF_batchsize bounds the
# later per-point query (which also multiplies by n_imgs internally).
_TSDF_SAMPLE_BUDGET = 150
_TSDF_QUERY_BATCHSIZE = 200_000


def build_run_manifest(
    *,
    image_paths: list[Path],
    output_dir: Path,
    output_path: Path,
    model_name: str,
    device: str,
    image_size: int,
    pairing_strategy: str,
    tsdf_thresh: float = 0,
    min_conf_thr: float = 1.5,
    seed: int | None = None,
    dry_run: bool,
    sync_view_groups: list[SyncViewGroup] | None = None,
    outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record the configuration *and* the measured shape of a reconstruction run.

    `outputs` carries the output-side counters (frames used, pairs built,
    vertices/faces produced, confidence-mask retention) that a results table
    needs; without them a manifest describes only what was asked for, never
    what came back.
    """
    manifest: dict[str, Any] = {
        "backend": "mast3r",
        "model_name": model_name,
        "device": device,
        "image_size": image_size,
        "pairing_strategy": pairing_strategy,
        "tsdf_thresh": tsdf_thresh,
        "min_conf_thr": min_conf_thr,
        "seed": seed,
        "dry_run": dry_run,
        "image_paths": [str(path) for path in image_paths],
        "output_dir": str(output_dir),
        "output_path": str(output_path),
    }
    if sync_view_groups:
        manifest["sync_pairing_enabled"] = True
        manifest["sync_group_count"] = len(sync_view_groups)
    manifest["outputs"] = {"n_frames_used": len(image_paths), **(outputs or {})}
    manifest["reproducibility"] = reproducibility_metadata()
    return manifest


def _dense_points_to_mesh(
    imgs: list[np.ndarray],
    pts3d: list[np.ndarray],
    confs: list[np.ndarray],
    min_conf_thr: float,
) -> Any:
    import trimesh
    from dust3r.viz import cat_meshes, pts3d_to_trimesh

    mask = [c > min_conf_thr for c in confs]
    meshes = []
    per_view_vertex_colors: list[np.ndarray] = []
    for i in range(len(imgs)):
        pts3d_i = pts3d[i].reshape(imgs[i].shape)
        msk_i = mask[i] & np.isfinite(pts3d_i.sum(axis=-1))
        meshes.append(pts3d_to_trimesh(imgs[i], pts3d_i, msk_i))
        per_view_vertex_colors.append(imgs[i].reshape(-1, 3))

    combined = cat_meshes(meshes)
    vertex_colors = np.concatenate(per_view_vertex_colors)
    vertex_colors_uint8 = (np.clip(vertex_colors, 0, 1) * 255).astype(np.uint8)
    return trimesh.Trimesh(
        vertices=combined["vertices"],
        faces=combined["faces"],
        vertex_colors=vertex_colors_uint8,
    )


def _dense_points_xyz_rgb(
    imgs: list[np.ndarray],
    pts3d: list[np.ndarray],
    confs: list[np.ndarray],
    min_conf_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten confidence-masked dense points and their image colors."""
    masks = [c > min_conf_thr for c in confs]
    xyz_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    for i in range(len(imgs)):
        pts3d_i = pts3d[i].reshape(imgs[i].shape)
        msk_i = masks[i] & np.isfinite(pts3d_i.sum(axis=-1))
        xyz_parts.append(pts3d_i[msk_i].reshape(-1, 3))
        rgb_parts.append(imgs[i].reshape(-1, 3)[msk_i.ravel()])
    return np.concatenate(xyz_parts, axis=0), np.concatenate(rgb_parts, axis=0)


def _patch_tsdf_cuda_hardcode() -> None:
    """Work around a MASt3R vendor bug, not a design choice of ours.

    `TSDFPostProcess._get_pixel_depths` calls `conf.cuda()` unconditionally
    (mast3r/cloud_opt/tsdf_optimizer.py), even though every other tensor in
    that method already follows `image_coords.device`. On a CPU-only torch
    build this raises `AssertionError: Torch not compiled with CUDA
    enabled` and crashes reconstruction outright. Only patch when CUDA is
    genuinely unavailable, so `.cuda()` becomes a no-op returning the same
    (already-CPU) tensor instead of raising -- a real CUDA machine is
    untouched.
    """
    global _tsdf_cuda_hardcode_patched
    if _tsdf_cuda_hardcode_patched:
        return

    import torch

    if torch.cuda.is_available():
        return

    def _cuda_noop(self: Any, *args: Any, **kwargs: Any) -> Any:
        return self

    torch.Tensor.cuda = _cuda_noop  # type: ignore[method-assign]
    _tsdf_cuda_hardcode_patched = True


def _patch_tsdf_sample_budget(nsamples: int = _TSDF_SAMPLE_BUDGET) -> None:
    """Lower the vendor's per-pixel Monte Carlo sample count for TSDF refinement.

    `nsamples` isn't a `TSDFPostProcess.__init__` parameter -- it's a default
    baked into `_refine_depths_with_TSDF`, which `torch.no_grad()` wraps.
    Assigning `__defaults__` on the decorated method is a no-op (the wrapper
    calls through a closure over the original function, not its own
    attributes); the original is reachable via `__wrapped__`, which is what
    actually needs patching. Fewer samples means a coarser Monte Carlo depth
    search, not a biased one -- an acceptable trade for a run that otherwise
    always falls back to the unfused path (see `_TSDF_SAMPLE_BUDGET` above).
    """
    global _tsdf_sample_budget_patched
    if _tsdf_sample_budget_patched:
        return

    from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess

    TSDFPostProcess._refine_depths_with_TSDF.__wrapped__.__defaults__ = (1, nsamples)
    _tsdf_sample_budget_patched = True


def export_scene_to_mesh(
    scene: Any,
    output_path: Path,
    output_dir: Path,
    tsdf_thresh: float = 0,
    min_conf_thr: float = 1.5,
    stats: dict[str, Any] | None = None,
) -> bool:
    """Fuse a scene into a mesh + point cloud; return whether TSDF fell back.

    `stats`, when given, is populated in place with the output-side counters
    (vertex/face counts, points before and after confidence masking) so the
    caller can fold them into its manifest. Kept as an out-parameter rather
    than a widened return type so existing callers keep working unchanged.
    """
    try:
        from dust3r.utils.device import to_numpy
        from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
    except ImportError as exc:
        raise RuntimeError("MASt3R mesh export dependencies are not installed.") from exc

    tsdf_fell_back = False
    imgs = to_numpy(scene.imgs)
    if tsdf_thresh > 0:
        _patch_tsdf_cuda_hardcode()
        _patch_tsdf_sample_budget()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            tsdf = TSDFPostProcess(
                scene, TSDF_thresh=tsdf_thresh, TSDF_batchsize=_TSDF_QUERY_BATCHSIZE
            )
            pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=True))
        except (MemoryError, RuntimeError, AssertionError) as exc:
            logger.warning("TSDF fusion failed (%s), falling back to non-TSDF mode", exc)
            tsdf_fell_back = True
            pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=True))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=True))

    mesh = _dense_points_to_mesh(imgs, pts3d, confs, min_conf_thr)

    fmt = output_path.suffix.lower()
    if fmt not in _SUPPORTED_FORMATS:
        logger.warning(
            "Unsupported format '%s', falling back to .glb. Supported: .obj, .glb, .ply",
            fmt,
        )
        output_path = output_path.with_suffix(".glb")
    mesh.export(str(output_path))

    xyz, rgb = _dense_points_xyz_rgb(imgs, pts3d, confs, min_conf_thr)
    write_ply(output_dir / "point_cloud.ply", xyz, rgb)

    if stats is not None:
        total_points = int(sum(int(np.prod(img.shape[:2])) for img in imgs))
        masked_points = int(len(xyz))
        stats.update(
            {
                "n_vertices": int(len(mesh.vertices)),
                "n_faces": int(len(mesh.faces)),
                "n_points_pre_mask": total_points,
                "n_points_post_mask": masked_points,
                "conf_retention": round(masked_points / max(total_points, 1), 4),
                "tsdf_fallback": tsdf_fell_back,
            }
        )

    logger.info(
        "Exported %s (vertex colors: %s)", output_path, mesh.visual.vertex_colors is not None
    )
    return tsdf_fell_back
