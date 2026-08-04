from __future__ import annotations

import argparse
import logging
from pathlib import Path

from spatial_ingestion.reconstruction.config import DEFAULT_MODEL_NAME
from spatial_ingestion.reconstruction.input import collect_input_images
from spatial_ingestion.reconstruction.models import (
    Mast3rRunParams,
    ReconstructionJob,
    ReconstructionMode,
)
from spatial_ingestion.reconstruction.paths import resolve_output_path
from spatial_ingestion.reconstruction.pipeline import run as pipeline_run

logger = logging.getLogger(__name__)

__all__ = [
    "build_parser",
    "collect_input_images",
    "main",
    "resolve_output_path",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct 3D mesh from multi-view images using MASt3R"
    )
    parser.add_argument("input", help="Folder containing at least two views of the same subject")
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Output path (.obj, .glb, .ply); defaults to .glb. "
            ".glb and .ply have proper vertex color support."
        ),
    )
    parser.add_argument("--device", default="auto", help="cuda, cpu, mps, or auto")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="MASt3R model id or local checkpoint path",
    )
    parser.add_argument(
        "--pairing-strategy",
        default="complete",
        choices=["complete", "swin"],
        help="MASt3R pairing strategy",
    )
    parser.add_argument("--image-size", type=int, default=512, help="MASt3R image size")
    parser.add_argument(
        "--tsdf-thresh",
        type=float,
        default=0,
        help="TSDF fusion threshold (0=disabled, 0.1-0.5 recommended)",
    )
    parser.add_argument(
        "--min-conf-thr",
        type=float,
        default=1.5,
        help="Minimum confidence threshold for point filtering",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate routing without running models"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input).expanduser().resolve()
    image_paths = collect_input_images(input_path)

    if len(image_paths) < 2:
        raise ValueError("Need a folder containing at least two views of the same subject")

    job = ReconstructionJob(
        mode=ReconstructionMode.MULTI_VIEW,
        label=input_path.name,
        image_uris=[str(p) for p in image_paths],
        params=Mast3rRunParams(
            model_name=args.model,
            device=args.device,
            image_size=args.image_size,
            pairing_strategy=args.pairing_strategy,
            tsdf_thresh=args.tsdf_thresh,
            min_conf_thr=args.min_conf_thr,
            seed=args.seed,
            dry_run=args.dry_run,
        ),
    )
    job.output_path = str(resolve_output_path(input_path, args.output, job_id=job.job_id))

    pipeline_run(job)
    return 0
