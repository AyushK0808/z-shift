from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np

from spatial_ingestion.config import MAST3R_ROOT
from spatial_ingestion.metadata.schema import CameraIntrinsics
from spatial_ingestion.reconstruction.models import HandoffFrame, SyncViewGroup
from spatial_ingestion.reconstruction.runners._io import uri_to_path, write_json

logger = logging.getLogger(__name__)

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
    parser.add_argument("--min-conf-thr", type=float, default=2.0,
                        help="Minimum confidence threshold for point filtering")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
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
        min_conf_thr=args.min_conf_thr,
        seed=args.seed,
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
    min_conf_thr: float = 2.0,
    seed: int | None = None,
    sync_view_groups: list[SyncViewGroup] | None = None,
    dry_run: bool = False,
    frames: list[HandoffFrame] | None = None,
) -> int:
    configure_local_mast3r_imports()
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = output_dir / "mesh.obj"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(device)

    if seed is not None:
        _set_seed(seed)

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
        min_conf_thr=min_conf_thr,
        seed=seed,
        dry_run=dry_run,
        sync_view_groups=sync_view_groups,
    )

    if dry_run:
        write_json(output_dir / "run_manifest.json", manifest)
        return 0

    sparse_scene = run_sparse_alignment(
        image_paths=image_paths,
        output_dir=output_dir,
        model_name=model_name,
        device=device,
        image_size=image_size,
        pairing_strategy=pairing_strategy,
        sync_view_groups=sync_view_groups,
        frames=frames,
    )
    tsdf_fell_back = export_sparse_scene_to_path(
        sparse_scene, output_path, output_dir,
        tsdf_thresh=tsdf_thresh, min_conf_thr=min_conf_thr,
    )
    manifest["tsdf_fallback"] = tsdf_fell_back
    write_json(output_dir / "run_manifest.json", manifest)
    return 0


def resolve_image_input(raw: str) -> Path:
    if _looks_like_windows_drive_path(raw):
        candidate = unquote(raw)
        path = Path(candidate).expanduser().resolve()
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


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _reproducibility_metadata() -> dict[str, Any]:
    meta: dict[str, Any] = {}
    try:
        import torch
        meta["torch_version"] = torch.__version__
        meta["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            meta["cuda_version"] = torch.version.cuda
            meta["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        meta["torch_version"] = None
    try:
        meta["numpy_version"] = np.__version__
    except AttributeError:
        pass
    return meta


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
    min_conf_thr: float = 2.0,
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
    manifest["reproducibility"] = _reproducibility_metadata()
    return manifest


def _build_sync_pairs(
    image_paths: list[Path],
    sync_view_groups: list[SyncViewGroup],
) -> list[tuple[int, int]]:
    """Build cross-camera pairs within each sync group."""
    stem_to_idx: dict[str, int] = {}
    for i, p in enumerate(image_paths):
        s = p.stem
        if s in stem_to_idx:
            logger.warning("Duplicate stem '%s' at index %d conflicts with index %d", s, i, stem_to_idx[s])
        stem_to_idx.setdefault(s, i)

    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    for group in sync_view_groups:
        cameras = sorted(group.frames_by_source.keys())
        indices: list[int] = []
        for source_id in cameras:
            handoff = group.frames_by_source[source_id]
            path_stem = uri_to_path(handoff.uri).stem
            idx = stem_to_idx.get(path_stem)
            if idx is not None:
                indices.append(idx)

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a, b = indices[i], indices[j]
                if (a, b) not in seen:
                    pairs.append((a, b))
                    pairs.append((b, a))
                    seen.add((a, b))
                    seen.add((b, a))

    return pairs


def _intrinsics_to_k_matrix(intrinsics: CameraIntrinsics, img_path: Path) -> Any | None:
    """Convert EXIF-derived CameraIntrinsics into a 3x3 K-matrix prior for MASt3R."""
    if intrinsics.focal_length_35mm is None:
        return None
    try:
        from PIL import Image
        with Image.open(img_path) as pil_img:
            w, h = pil_img.size
    except Exception:
        return None
    # 35 mm film frame is 36 × 24 mm
    focal_px = intrinsics.focal_length_35mm / 36.0 * max(w, h)
    cx, cy = w / 2, h / 2
    import torch
    K = torch.eye(3)
    K[0, 0] = K[1, 1] = focal_px
    K[0, 2] = cx
    K[1, 2] = cy
    return K


def run_sparse_alignment(
    *,
    image_paths: list[Path],
    output_dir: Path,
    model_name: str,
    device: str,
    image_size: int,
    pairing_strategy: str,
    sync_view_groups: list[SyncViewGroup] | None = None,
    frames: list[HandoffFrame] | None = None,
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

    if sync_view_groups:
        idx_pairs = _build_sync_pairs(image_paths, sync_view_groups)
        if idx_pairs:
            pairs = [(images[a], images[b]) for a, b in idx_pairs]
        else:
            logger.warning("Sync-aware pairing produced no pairs, falling back to %s", pairing_strategy)
            pairs = make_pairs(images, scene_graph=pairing_strategy, symmetrize=True)
    else:
        pairs = make_pairs(images, scene_graph=pairing_strategy, symmetrize=True)

    cache_path = str((output_dir / "cache").resolve())
    str_paths = [str(path) for path in image_paths]
    init: dict[str, dict[str, Any]] = {}
    if frames:
        for img_path, frame in zip(image_paths, frames):
            if frame.camera_intrinsics:
                K = _intrinsics_to_k_matrix(frame.camera_intrinsics, img_path)
                if K is not None:
                    init[str(img_path)] = {"intrinsics": K}
    return sparse_global_alignment(
        str_paths,
        pairs,
        cache_path=cache_path,
        model=model,
        device=device,
        init=init,
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
) -> bool:
    """Export scene to mesh. Returns True if TSDF fell back to non-TSDF mode."""
    try:
        import trimesh
        from dust3r.utils.device import to_numpy
        from dust3r.viz import cat_meshes, pts3d_to_trimesh
        from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
    except ImportError as exc:
        raise RuntimeError("MASt3R mesh export dependencies are not installed.") from exc

    tsdf_fell_back = False
    rgbimg = scene.imgs
    imgs = to_numpy(rgbimg)
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

    mask = to_numpy([c > min_conf_thr for c in confs])

    meshes = []
    for i in range(len(imgs)):
        pts3d_i = pts3d[i].reshape(imgs[i].shape)
        msk_i = mask[i] & np.isfinite(pts3d_i.sum(axis=-1))
        meshes.append(pts3d_to_trimesh(imgs[i], pts3d_i, msk_i))

    combined = cat_meshes(meshes)
    vertex_colors = (np.clip(combined['colors'], 0, 1) * 255).astype(np.uint8)
    mesh = trimesh.Trimesh(
        vertices=combined['vertices'],
        faces=combined['faces'],
        vertex_colors=vertex_colors,
    )

    if output_path.suffix not in ('.obj', '.glb'):
        output_path = output_path.with_suffix('.obj')
    mesh.export(str(output_path))
    logger.info("Exported %s", output_path)
    return tsdf_fell_back


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
