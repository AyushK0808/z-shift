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
from spatial_ingestion.reconstruction.config import DEFAULT_MODEL_NAME
from spatial_ingestion.reconstruction.input import collect_input_images
from spatial_ingestion.reconstruction.models import Mast3rRunParams
from spatial_ingestion.refinement.options import (
    add_mesh_cleaning_args,
    mesh_cleaning_config_from_args,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 1-4 end-to-end pipeline: ingest -> reconstruct (MASt3R) -> "
        "refine -> deliverable."
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
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="MASt3R model id or local checkpoint path",
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

    add_mesh_cleaning_args(parser, mode_flag="--refinement-mode")
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
    refinement_config = mesh_cleaning_config_from_args(args)

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
