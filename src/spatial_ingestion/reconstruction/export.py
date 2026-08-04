from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from spatial_ingestion.reconstruction._deps import mast3r_dependency_error
from spatial_ingestion.reconstruction.config import POINT_CLOUD_FILENAME
from spatial_ingestion.reconstruction.device import reproducibility_metadata
from spatial_ingestion.reconstruction.io import write_ply
from spatial_ingestion.reconstruction.models import SyncViewGroup

logger = logging.getLogger(__name__)

SUPPORTED_MESH_FORMATS = {".obj", ".glb", ".ply"}


def unsupported_mesh_format_message(suffix: str) -> str:
    return (
        f"Unsupported Phase 2 mesh format '{suffix}'. "
        f"Phase 2 can only write {', '.join(sorted(SUPPORTED_MESH_FORMATS))}."
    )


def validate_mesh_format(output_path: Path) -> None:
    """Reject mesh output formats Phase 2 cannot produce."""
    if output_path.suffix.lower() not in SUPPORTED_MESH_FORMATS:
        raise ValueError(unsupported_mesh_format_message(output_path.suffix))


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
        raise mast3r_dependency_error("MASt3R mesh export dependencies") from exc

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

    mesh = _dense_points_to_mesh(imgs, pts3d, confs, min_conf_thr)

    validate_mesh_format(output_path)
    mesh.export(str(output_path))

    xyz, rgb = _dense_points_xyz_rgb(imgs, pts3d, confs, min_conf_thr)
    write_ply(output_dir / POINT_CLOUD_FILENAME, xyz, rgb)

    logger.info(
        "Exported %s (vertex colors: %s)", output_path, mesh.visual.vertex_colors is not None
    )
    return tsdf_fell_back
