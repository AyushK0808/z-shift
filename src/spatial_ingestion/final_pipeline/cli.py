from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from spatial_ingestion.final_pipeline.core import (
    full_result_to_dict,
    result_to_dict,
    run_full_pipeline,
    run_phase2_phase3_pipeline,
)
from spatial_ingestion.final_pipeline.handoff import (
    build_job,
    ingest_batch,
    load_schema,
)
from spatial_ingestion.metadata.schema import SourceType
from spatial_ingestion.reconstruction.cli import (
    DEFAULT_MODEL,
    collect_input_images,
)
from spatial_ingestion.reconstruction.models import Mast3rRunParams
from spatial_ingestion.refinement import MeshCleaningConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase 2 reconstruction followed by Phase 3 mesh refinement."
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Folder containing at least two views of the same subject "
        "(optional when --from-schema is set)",
    )
    parser.add_argument(
        "--from-schema",
        type=Path,
        help="Use a Phase 1 payload JSON (as returned by the ingestion gateway) instead of "
        "ingesting the input folder",
    )
    parser.add_argument("-o", "--output", help="Raw Phase 2 mesh output path or output directory")
    parser.add_argument(
        "--refined-output", type=Path, help="Where to write the Phase 3 refined mesh"
    )
    parser.add_argument(
        "--use-case",
        choices=("editing", "viewing", "live"),
        help="Run Phase 4 too, routing to this use case",
    )
    parser.add_argument(
        "--source-type",
        "--input-type",
        dest="source_type",
        type=SourceType,
        help="SourceType for Phase 4 routing (e.g. image_folder). Defaults to the "
        "Phase 1 classified source type.",
    )
    parser.add_argument("--device", default="auto", help="cuda, cpu, mps, or auto")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="MASt3R model id or local checkpoint path"
    )
    parser.add_argument(
        "--pairing-strategy",
        choices=["complete", "swin"],
        help="MASt3R pairing strategy (default: auto; swin is auto-selected for videos "
        "and large image sets)",
    )
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

    if args.use_case == "live":
        parser.error(
            "live delivery (real-time WebRTC/WebSocket) is not implemented yet; "
            "use --use-case editing or viewing"
        )

    if args.from_schema:
        label = None
        payload = load_schema(args.from_schema)
        if payload.source_type == SourceType.SINGLE_IMAGE:
            parser.error(
                "single-image reconstruction is not supported; the payload must contain "
                "at least two views (image_folder) or a video capture"
            )
    else:
        if not args.input:
            parser.error("an input folder or --from-schema is required")
        folder = Path(args.input).expanduser().resolve()
        image_paths = collect_input_images(folder)
        if len(image_paths) < 2:
            parser.error("Need a folder containing at least two views of the same subject")
        payload = ingest_batch(image_paths)
        label = image_paths[0].parent.name

    mast3r_params = Mast3rRunParams(
        model_name=args.model,
        device=args.device,
        image_size=args.image_size,
        tsdf_thresh=args.tsdf_thresh,
        min_conf_thr=args.min_conf_thr,
        seed=args.seed,
        dry_run=False,
    )
    if args.pairing_strategy:
        mast3r_params.pairing_strategy = args.pairing_strategy

    job = build_job(
        payload,
        mast3r_params=mast3r_params,
        output_path=args.output,
        label=label,
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

    if args.use_case:
        full_result = run_full_pipeline(
            job,
            use_case=args.use_case,
            source_type=args.source_type or payload.source_type,
            refinement_config=refinement_config,
            refined_output_path=args.refined_output,
        )
        print(json.dumps(full_result_to_dict(full_result), indent=2, sort_keys=True))
        return 0

    result = run_phase2_phase3_pipeline(
        job, refinement_config, refined_output_path=args.refined_output
    )
    print(json.dumps(result_to_dict(result), indent=2, sort_keys=True))
    return 0
