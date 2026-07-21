from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from spatial_ingestion.reconstruction._io import write_json
from spatial_ingestion.reconstruction.device import reproducibility_metadata
from spatial_ingestion.reconstruction.models import SyncViewGroup

logger = logging.getLogger(__name__)

_SUPPORTED_FORMATS = {".obj", ".glb", ".ply"}


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
) -> dict[str, Any]:
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
    manifest["reproducibility"] = reproducibility_metadata()
    return manifest


def _samples_to_mesh(
    imgs: list[np.ndarray],
    pts3d: list[np.ndarray],
    confs: list[np.ndarray],
    min_conf_thr: float,
) -> Any:
    import trimesh
    from dust3r.viz import cat_meshes, pts3d_to_trimesh

    mask = [c > min_conf_thr for c in confs]
    meshes = []
    for i in range(len(imgs)):
        pts3d_i = pts3d[i].reshape(imgs[i].shape)
        msk_i = mask[i] & np.isfinite(pts3d_i.sum(axis=-1))
        meshes.append(pts3d_to_trimesh(imgs[i], pts3d_i, msk_i))

    combined = cat_meshes(meshes)
    vertex_colors = (np.clip(combined["colors"], 0, 1) * 255).astype(np.uint8)
    return trimesh.Trimesh(
        vertices=combined["vertices"],
        faces=combined["faces"],
        vertex_colors=vertex_colors,
    )


def export_scene_to_mesh(
    scene: Any,
    output_path: Path,
    output_dir: Path,
    tsdf_thresh: float = 0,
    min_conf_thr: float = 1.5,
) -> bool:
    try:
        from dust3r.utils.device import to_numpy
        from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
    except ImportError as exc:
        raise RuntimeError("MASt3R mesh export dependencies are not installed.") from exc

    tsdf_fell_back = False
    imgs = to_numpy(scene.imgs)
    if tsdf_thresh > 0:
        try:
            tsdf = TSDFPostProcess(scene, TSDF_thresh=tsdf_thresh)
            pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=True))
        except (MemoryError, RuntimeError) as exc:
            logger.warning("TSDF fusion failed (%s), falling back to non-TSDF mode", exc)
            tsdf_fell_back = True
            pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=True))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=True))

    conf_mask = [c > min_conf_thr for c in confs]
    mesh = _samples_to_mesh(imgs, pts3d, conf_mask, min_conf_thr)

    fmt = output_path.suffix.lower()
    if fmt not in _SUPPORTED_FORMATS:
        logger.warning(
            "Unsupported format '%s', falling back to .obj. "
            "Supported: .obj, .glb, .ply",
            fmt,
        )
        output_path = output_path.with_suffix(".obj")
        fmt = ".obj"

    if fmt == ".ply":
        mesh.export(str(output_path), encoding="ascii")
    else:
        mesh.export(str(output_path))

    logger.info("Exported %s (vertex colors: %s)", output_path, mesh.visual.vertex_colors is not None)
    return tsdf_fell_back
