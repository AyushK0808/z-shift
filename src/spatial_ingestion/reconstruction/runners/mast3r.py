from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np

from spatial_ingestion.config import MAST3R_ROOT


DEFAULT_MODEL_NAME = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MASt3R reconstruction")
    parser.add_argument("images", nargs="+", help="Normalized image URIs or paths")
    parser.add_argument("--output-dir", required=True, help="Artifact output directory")
    parser.add_argument("--output-path", help="Explicit OBJ output path")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Model id or local checkpoint")
    parser.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    parser.add_argument("--image-size", type=int, default=512, help="Resize used for MASt3R loading")
    parser.add_argument(
        "--pairing-strategy",
        default="complete",
        choices=["complete", "swin"],
        help="Pair construction strategy",
    )
    parser.add_argument("--tsdf-thresh", type=float, default=0,
                        help="TSDF fusion threshold (0=disabled, 0.1-0.5 recommended, expensive)")
    parser.add_argument("--dry-run", action="store_true", help="Write manifest only")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path
        else output_dir / "mesh.obj"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = [resolve_image_input(image) for image in args.images]
    if len(image_paths) < 2:
        raise ValueError("MASt3R reconstruction requires at least two images")

    device = resolve_device(args.device)
    manifest = build_run_manifest(
        image_paths=image_paths,
        output_dir=output_dir,
        output_path=output_path,
        model_name=args.model_name,
        device=device,
        image_size=args.image_size,
        pairing_strategy=args.pairing_strategy,
        tsdf_thresh=args.tsdf_thresh,
        dry_run=args.dry_run,
    )
    write_json(output_dir / "run_manifest.json", manifest)

    if args.dry_run:
        return 0

    sparse_scene = run_sparse_alignment(
        image_paths=image_paths,
        output_dir=output_dir,
        model_name=args.model_name,
        device=device,
        image_size=args.image_size,
        pairing_strategy=args.pairing_strategy,
    )
    export_sparse_scene_to_path(sparse_scene, output_path, output_dir, tsdf_thresh=args.tsdf_thresh)
    return 0


def resolve_image_input(raw: str) -> Path:
    if _looks_like_windows_drive_path(raw):
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {raw}")
        return path

    parsed = urlparse(raw)
    if parsed.scheme in {"", "file"}:
        candidate = unquote(parsed.path if parsed.scheme == "file" else raw)
        if parsed.scheme == "file" and _looks_like_windows_file_uri_path(candidate):
            candidate = candidate.lstrip("/")
        path = Path(candidate).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {raw}")
        return path
    raise ValueError(f"Unsupported image URI scheme for MASt3R input: {raw}")


def _looks_like_windows_drive_path(raw: str) -> bool:
    return len(raw) >= 3 and raw[1] == ":" and raw[2] in {"\\", "/"} and raw[0].isalpha()


def _looks_like_windows_file_uri_path(raw: str) -> bool:
    return len(raw) >= 4 and raw[0] == "/" and raw[2] == ":" and raw[1].isalpha()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested

    try:
        import torch
    except ImportError:
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"


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
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "backend": "mast3r",
        "model_name": model_name,
        "device": device,
        "image_size": image_size,
        "pairing_strategy": pairing_strategy,
        "tsdf_thresh": tsdf_thresh,
        "dry_run": dry_run,
        "image_paths": [str(path) for path in image_paths],
        "artifacts": {
            "point_cloud": str((output_dir / "point_cloud.ply").resolve()),
            "poses": str((output_dir / "camera_poses.json").resolve()),
            "mesh": str(output_path),
        },
    }


def run_sparse_alignment(
    *,
    image_paths: list[Path],
    output_dir: Path,
    model_name: str,
    device: str,
    image_size: int,
    pairing_strategy: str,
) -> Any:
    configure_local_mast3r_imports()

    try:
        import mast3r.utils.path_to_dust3r  # noqa: F401
        from dust3r.image_pairs import make_pairs
        from dust3r.utils.image import load_images
        from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
        from mast3r.model import AsymmetricMASt3R
    except ImportError as exc:
        raise RuntimeError(
            "MASt3R runtime dependencies are not installed. Add the upstream repo under third_party/mast3r "
            "or install the MASt3R environment in the current Python environment."
        ) from exc

    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    images = load_images([str(path) for path in image_paths], size=image_size)
    pairs = make_pairs(images, scene_graph=pairing_strategy, symmetrize=True)
    cache_path = str((output_dir / "cache").resolve())
    return sparse_global_alignment(
        [str(path) for path in image_paths],
        pairs,
        cache_path=cache_path,
        model=model,
        device=device,
    )


def configure_local_mast3r_imports(root: Path = MAST3R_ROOT) -> None:
    if not root.exists():
        return

    candidates = [root]
    dust3r_root = root / "dust3r"
    if dust3r_root.exists():
        candidates.append(dust3r_root)

    for candidate in reversed(candidates):
        candidate_str = str(candidate.resolve())
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def export_sparse_scene(scene: Any, output_dir: Path) -> None:
    export_sparse_scene_to_path(scene, output_dir / "mesh", output_dir)


def export_sparse_scene_to_path(
    scene: Any,
    output_path: Path,
    output_dir: Path,
    tsdf_thresh: float = 0,
    min_conf_thr: float = 2.0,
) -> None:
    configure_local_mast3r_imports()

    try:
        import trimesh
        from dust3r.utils.device import to_numpy
        from dust3r.viz import cat_meshes, pts3d_to_trimesh
        from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
    except ImportError as exc:
        raise RuntimeError("MASt3R mesh export dependencies are not installed.") from exc

    rgbimg = scene.imgs
    imgs = to_numpy(rgbimg)
    focals = to_numpy(scene.get_focals().cpu())
    cams2world = to_numpy(scene.get_im_poses().cpu())

    if tsdf_thresh > 0:
        try:
            tsdf = TSDFPostProcess(scene, TSDF_thresh=tsdf_thresh)
            pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=True))
        except (MemoryError, RuntimeError) as exc:
            print(f"TSDF fusion failed ({exc}), falling back to non-TSDF mode")
            pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=True))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=True))

    mask = to_numpy([c > min_conf_thr for c in confs])

    # Build per-view meshes with TSDF-refined points
    meshes = []
    for i in range(len(imgs)):
        pts3d_i = pts3d[i].reshape(imgs[i].shape)
        msk_i = mask[i] & np.isfinite(pts3d_i.sum(axis=-1))
        meshes.append(pts3d_to_trimesh(imgs[i], pts3d_i, msk_i))

    combined = cat_meshes(meshes)
    vertex_colors = np.concatenate([img.reshape(-1, 3) for img in imgs])
    vertex_colors = (np.clip(vertex_colors, 0, 1) * 255).astype(np.uint8)
    mesh = trimesh.Trimesh(
        vertices=combined['vertices'],
        faces=combined['faces'],
        vertex_colors=vertex_colors,
    )

    # Export GLB (preserves vertex color)
    glb_path = output_path.with_suffix('.glb')
    mesh.export(str(glb_path))
    print(f"Exported {glb_path}")

    # Export OBJ
    obj_path = output_path.with_suffix('.obj')
    mesh.export(str(obj_path))
    print(f"Exported {obj_path}")

    # JSON metadata
    poses = to_serializable_array(scene.get_im_poses())
    focals_list = to_serializable_array(scene.get_focals())
    principal_points = to_serializable_array(scene.get_principal_points())
    write_json(
        output_dir / "camera_poses.json",
        {
            "camera_count": len(poses),
            "poses": poses,
            "focals": focals_list,
            "principal_points": principal_points,
        },
    )

    # PLY sparse point cloud
    sparse_points = scene.get_sparse_pts3d()
    sparse_colors = scene.get_pts3d_colors()
    write_ply(output_dir / "point_cloud.ply", sparse_points, sparse_colors)


def to_serializable_array(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [to_serializable_array(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_ply(path: Path, points: Any, colors: Any) -> None:
    xyz_rows = flatten_rows(points)
    rgb_rows = flatten_rows(colors)
    row_count = min(len(xyz_rows), len(rgb_rows))

    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {row_count}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for xyz, rgb in zip(xyz_rows[:row_count], rgb_rows[:row_count], strict=False):
        red, green, blue = normalize_rgb(rgb)
        lines.append(f"{float(xyz[0])} {float(xyz[1])} {float(xyz[2])} {red} {green} {blue}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def flatten_rows(value: Any) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, (list, tuple)):
        parts = [np.asarray(v, dtype=float) for v in value]
        if parts and parts[0].ndim == 2:
            array = np.concatenate(parts, axis=0)
        elif parts:
            array = np.concatenate(parts)
        else:
            return []
    else:
        array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        return [array.tolist()]
    if array.ndim == 2:
        return array.tolist()
    if array.ndim >= 3:
        return array.reshape(-1, array.shape[-1]).tolist()
    return []


def normalize_rgb(values: list[float]) -> tuple[int, int, int]:
    clipped = np.clip(np.asarray(values[:3], dtype=float), 0.0, 1.0)
    scaled = np.rint(clipped * 255.0).astype(int)
    return int(scaled[0]), int(scaled[1]), int(scaled[2])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
