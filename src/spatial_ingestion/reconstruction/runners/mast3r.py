from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np

from spatial_ingestion.config import MAST3R_ROOT
from spatial_ingestion.reconstruction.runners._io import write_json


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
    image_paths = [resolve_image_input(img) for img in args.images]
    return run(
        image_paths=image_paths,
        output_dir=Path(args.output_dir).resolve(),
        output_path=Path(args.output_path).expanduser().resolve() if args.output_path else None,
        model_name=args.model_name,
        device=resolve_device(args.device),
        image_size=args.image_size,
        pairing_strategy=args.pairing_strategy,
        tsdf_thresh=args.tsdf_thresh,
        dry_run=args.dry_run,
    )


def run(
    *,
    image_paths: list[Path],
    output_dir: Path,
    output_path: Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cpu",
    image_size: int = 512,
    pairing_strategy: str = "complete",
    tsdf_thresh: float = 0,
    dry_run: bool = False,
) -> int:
    configure_local_mast3r_imports()
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = output_dir / "mesh.obj"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(image_paths) < 2:
        raise ValueError("MASt3R reconstruction requires at least two images")

    manifest = build_run_manifest(
        image_paths=image_paths,
        output_dir=output_dir,
        output_path=output_path,
        model_name=model_name,
        device=device,
        image_size=image_size,
        pairing_strategy=pairing_strategy,
        tsdf_thresh=tsdf_thresh,
        dry_run=dry_run,
    )
    write_json(output_dir / "run_manifest.json", manifest)

    if dry_run:
        return 0

    sparse_scene = run_sparse_alignment(
        image_paths=image_paths,
        output_dir=output_dir,
        model_name=model_name,
        device=device,
        image_size=image_size,
        pairing_strategy=pairing_strategy,
    )
    export_sparse_scene_to_path(sparse_scene, output_path, output_dir, tsdf_thresh=tsdf_thresh)
    return 0


def resolve_image_input(raw: str) -> Path:
    parsed = urlparse(raw)
    if parsed.scheme in {"", "file"}:
        candidate = unquote(parsed.path if parsed.scheme == "file" else raw)
        path = Path(candidate).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {raw}")
        return path
    raise ValueError(f"Unsupported image URI scheme for MASt3R input: {raw}")


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
        "output_path": str(output_path),
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


def export_sparse_scene_to_path(
    scene: Any,
    output_path: Path,
    output_dir: Path,
    tsdf_thresh: float = 0,
    min_conf_thr: float = 2.0,
) -> None:
    try:
        import trimesh
        from dust3r.utils.device import to_numpy
        from dust3r.viz import cat_meshes, pts3d_to_trimesh
        from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
    except ImportError as exc:
        raise RuntimeError("MASt3R mesh export dependencies are not installed.") from exc

    rgbimg = scene.imgs
    imgs = to_numpy(rgbimg)
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

    if output_path.suffix not in ('.obj', '.glb'):
        output_path = output_path.with_suffix('.obj')
    mesh.export(str(output_path))
    print(f"Exported {output_path}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
