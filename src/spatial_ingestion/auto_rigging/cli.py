from __future__ import annotations

import argparse
from pathlib import Path

from spatial_ingestion.auto_rigging.models import ArticulationType, AutoRigConfig
from spatial_ingestion.auto_rigging.pipeline import AutoRiggingPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 5 auto-rigging MVP")
    parser.add_argument("mesh", help="Input mesh path")
    parser.add_argument(
        "--articulation",
        choices=[item.value for item in ArticulationType],
        default=ArticulationType.STATIC.value,
        help="Template articulation type to fit",
    )
    parser.add_argument(
        "--max-influences",
        type=int,
        default=4,
        help="Maximum non-zero joint influences per vertex",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not normalize the mesh to a unit-scale bounding box before fitting",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Return rig data without writing skeleton/weight JSON files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AutoRigConfig(
        articulation_type=ArticulationType(args.articulation),
        max_skinning_influences=args.max_influences,
        normalize_mesh=not args.no_normalize,
    )
    result = AutoRiggingPipeline().rig_mesh_file(
        Path(args.mesh),
        config=config,
        export_metadata=not args.no_export,
    )
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

