from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .core import MeshProcessingError, MeshValidationError, refine_mesh_file
from .options import add_mesh_cleaning_args, mesh_cleaning_config_from_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean a mesh file with the refinement pipeline.")
    parser.add_argument(
        "mesh",
        nargs="?",
        type=Path,
        help="Input mesh file to clean (.glb, .obj, .ply, .stl, .vtk)",
    )
    # Deprecated alias: the input used to be `--refine <path>`.
    parser.add_argument("--refine", type=Path, dest="mesh", help=argparse.SUPPRESS)
    parser.add_argument(
        "--output",
        type=Path,
        help="Where to write the cleaned mesh (default: <stem>_refined<suffix>)",
    )
    add_mesh_cleaning_args(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mesh is None:
        parser.error("an input mesh file is required")

    config = mesh_cleaning_config_from_args(args)

    try:
        result = refine_mesh_file(args.mesh, args.output, config)
    except (MeshValidationError, MeshProcessingError) as exc:
        parser.exit(1, f"{parser.prog}: error: {exc}\n")

    summary = {
        "input_path": result["input_path"],
        "output_path": result["output_path"],
        "mode": result["mode"],
        "is_watertight": result["is_watertight"],
        "output_point_count": result["output_point_count"],
        "output_cell_count": result["output_cell_count"],
        "warnings": result["warnings"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
