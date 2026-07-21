from __future__ import annotations

import argparse
import logging
from pathlib import Path
from uuid import uuid4

from spatial_ingestion.config import RECONSTRUCTION_OUTPUT_ROOT
from spatial_ingestion.reconstruction.models import ReconstructionJob, ReconstructionMode
from spatial_ingestion.reconstruction.registry import ReconstructionBackendRegistry

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a folder of multi-view images to OBJ")
    parser.add_argument("input", help="Folder containing at least two views of the same subject")
    parser.add_argument("-o", "--output", help="Output .obj or .glb path")
    parser.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    parser.add_argument(
        "--model",
        default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        help="MASt3R model id or local checkpoint path",
    )
    parser.add_argument(
        "--pairing-strategy",
        default="complete",
        choices=["complete", "swin"],
        help="MASt3R pairing strategy",
    )
    parser.add_argument("--image-size", type=int, default=512, help="MASt3R image size")
    parser.add_argument("--tsdf-thresh", type=float, default=0,
                        help="TSDF fusion threshold (0=disabled, 0.1-0.5 recommended, expensive)")
    parser.add_argument("--min-conf-thr", type=float, default=2.0,
                        help="Minimum confidence threshold for point filtering")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--dry-run", action="store_true", help="Validate routing without running models")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input).expanduser().resolve()
    image_paths = collect_input_images(input_path)

    if len(image_paths) < 2:
        raise ValueError("MASt3R requires a folder containing at least two views of the same subject")

    from spatial_ingestion.reconstruction.runners.mast3r import resolve_device

    output_path = resolve_output_path(input_path, args.output)

    metadata: dict[str, object] = {
        "model_name": args.model,
        "device": resolve_device(args.device),
        "image_size": args.image_size,
        "pairing_strategy": args.pairing_strategy,
        "tsdf_thresh": args.tsdf_thresh,
        "min_conf_thr": args.min_conf_thr,
        "seed": args.seed,
        "dry_run": args.dry_run,
    }

    job = ReconstructionJob(
        mode=ReconstructionMode.MULTI_VIEW,
        backend_name="mast3r",
        image_uris=[str(p) for p in image_paths],
        output_path=str(output_path),
        metadata=metadata,
    )

    backend = ReconstructionBackendRegistry().resolve_for_job(job)
    return backend.execute(job)


def collect_input_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        raise ValueError("MASt3R requires a folder containing at least two views of the same subject")

    if input_path.is_dir():
        image_paths = sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise ValueError(f"No supported images found in directory: {input_path}")
        return image_paths

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def resolve_output_path(input_path: Path, explicit_output: str | None) -> Path:
    job_id = uuid4().hex[:12]

    if explicit_output:
        output_path = Path(explicit_output).expanduser().resolve()
        if output_path.suffix.lower() in {".obj", ".glb"}:
            stem = output_path.stem
            return output_path.parent / f"{stem}_{job_id}" / output_path.name
        return output_path / f"mesh_{job_id}.obj"

    stem = input_path.stem if input_path.is_file() else input_path.name
    return RECONSTRUCTION_OUTPUT_ROOT / f"{stem}_{job_id}" / f"{stem}.obj"
