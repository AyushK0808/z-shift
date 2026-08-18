from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from spatial_ingestion.auto_rigging.models import ArticulationType, AutoRigConfig
from spatial_ingestion.final_pipeline.core import (
    full_result_to_dict,
    result_to_dict,
    run_full_pipeline,
    run_phase2_phase3_phase5_pipeline,
    run_phase2_phase3_pipeline,
)
from spatial_ingestion.final_pipeline.handoff import (
    build_job,
    ingest_batch,
    load_schema,
)
from spatial_ingestion.reconstruction.cli import (
    DEFAULT_MODEL,
    collect_input_images,
    resolve_output_path,
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
        "--input-type",
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

    parser.add_argument(
        "--rig",
        action="store_true",
        help="Run Phase 5 after refinement and export a skinned GLB plus rig metadata",
    )
    parser.add_argument(
        "--articulation",
        choices=[item.value for item in ArticulationType],
        default=ArticulationType.STATIC.value,
        help="Phase 5 template articulation type",
    )
    parser.add_argument("--rig-max-influences", type=int, default=4)
    parser.add_argument(
        "--no-rig-normalize",
        action="store_true",
        help="Keep refined mesh scale/orientation for Phase 5 instead of unit-box normalization",
    )
    parser.add_argument("--rig-output-dir", type=Path, help="Directory for Phase 5 rig metadata")
    parser.add_argument("--rigged-output", type=Path, help="Exact Phase 5 skinned GLB path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.from_schema:
        input_path: Path | None = None
        label = None
        payload = load_schema(args.from_schema)
    else:
        if not args.input:
            parser.error("an input folder or --from-schema is required")
        input_path = Path(args.input).expanduser().resolve()
        image_paths = collect_input_images(input_path)
        if len(image_paths) < 2:
            raise ValueError("Need a folder containing at least two views of the same subject")
        payload = ingest_batch(image_paths)
        input_path = image_paths[0].parent
        label = input_path.name

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

    output_path = resolve_output_path(
        input_path,
        args.output,
        label=label or (payload.sync_group_id or payload.source_type.value),
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
    rigging_config = AutoRigConfig(
        articulation_type=ArticulationType(args.articulation),
        max_skinning_influences=args.rig_max_influences,
        normalize_mesh=not args.no_rig_normalize,
        output_dir=args.rig_output_dir,
        rigged_output_path=args.rigged_output,
    )

    job = build_job(
        payload,
        mast3r_params=mast3r_params,
        output_path=output_path,
        label=label,
    )
    if args.use_case:
        if args.rig:
            parser.error(
                "--rig cannot be combined with --use-case yet; "
                "run Phase 5 on the refined mesh output"
            )
        full_result = run_full_pipeline(
            job,
            use_case=args.use_case,
            input_type=args.input_type or payload.source_type.value,
            refinement_config=refinement_config,
            refined_output_path=args.refined_output,
        )
        print(json.dumps(full_result_to_dict(full_result), indent=2, sort_keys=True))
        return 0

    if args.rig:
        result = run_phase2_phase3_phase5_pipeline(
            job,
            refinement_config,
            rigging_config=rigging_config,
            refined_output_path=args.refined_output,
            rigged_output_path=args.rigged_output,
            rig_output_dir=args.rig_output_dir,
        )
        print(json.dumps(result_to_dict(result), indent=2, sort_keys=True))
        return 0

    result = run_phase2_phase3_pipeline(
        job, refinement_config, refined_output_path=args.refined_output
    )
    print(json.dumps(result_to_dict(result), indent=2, sort_keys=True))
    return 0
