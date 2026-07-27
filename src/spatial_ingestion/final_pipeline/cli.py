from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from spatial_ingestion.final_pipeline.core import result_to_dict, run_phase2_phase3_pipeline
from spatial_ingestion.reconstruction.cli import DEFAULT_MODEL, collect_input_images, resolve_output_path
from spatial_ingestion.reconstruction.models import Mast3rRunParams, ReconstructionJob, ReconstructionMode
from spatial_ingestion.refinement import MeshCleaningConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 2 reconstruction followed by Phase 3 mesh refinement.")
    parser.add_argument("input", help="Folder containing at least two views of the same subject")
    parser.add_argument("-o", "--output", help="Raw Phase 2 mesh output path or output directory")
    parser.add_argument("--refined-output", type=Path, help="Where to write the Phase 3 refined mesh")

    parser.add_argument("--device", default="auto", help="cuda, cpu, mps, or auto")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="MASt3R model id or local checkpoint path")
    parser.add_argument("--pairing-strategy", default="complete", choices=["complete", "swin"])
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--tsdf-thresh", type=float, default=0)
    parser.add_argument("--min-conf-thr", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--refinement-mode", choices=("object", "room"), default="object")
    parser.add_argument("--smoothing-iters", type=int, default=15)
    parser.add_argument("--pass-band", type=float, default=0.1)
    parser.add_argument("--hole-size", type=float)
    parser.add_argument("--min-cell-count", type=int, default=500)
    parser.add_argument("--feature-angle", type=float, default=45.0)
    parser.add_argument("--merge-tolerance", type=float, default=1e-5)
    parser.add_argument("--decimate-target-reduction", type=float)
    parser.add_argument("--no-watertight-check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input).expanduser().resolve()
    image_paths = collect_input_images(input_path)
    if len(image_paths) < 2:
        raise ValueError("Need a folder containing at least two views of the same subject")

    job = ReconstructionJob(
        mode=ReconstructionMode.MULTI_VIEW,
        label=input_path.name,
        image_uris=[str(path) for path in image_paths],
        output_path=str(resolve_output_path(input_path, args.output)),
        metadata=Mast3rRunParams(
            model_name=args.model,
            device=args.device,
            image_size=args.image_size,
            pairing_strategy=args.pairing_strategy,
            tsdf_thresh=args.tsdf_thresh,
            min_conf_thr=args.min_conf_thr,
            seed=args.seed,
            dry_run=False,
        ).model_dump(),
    )
    refinement_config = MeshCleaningConfig(
        mode=args.refinement_mode,
        smoothing_iters=args.smoothing_iters,
        pass_band=args.pass_band,
        hole_size=args.hole_size,
        min_cell_count=args.min_cell_count,
        feature_angle=args.feature_angle,
        merge_tolerance=args.merge_tolerance,
        decimate_target_reduction=args.decimate_target_reduction,
        verify_watertight=not args.no_watertight_check,
    )

    result = run_phase2_phase3_pipeline(job, refinement_config, refined_output_path=args.refined_output)
    print(json.dumps(result_to_dict(result), indent=2, sort_keys=True))
    return 0
