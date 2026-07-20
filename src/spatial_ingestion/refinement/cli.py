from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pyvista as pv

from .core import MeshCleaningConfig, MeshProcessingError, MeshValidationError, clean_mesh


def _default_output_path(input_path: Path) -> Path:
    if input_path.suffix:
        return input_path.with_name(f"{input_path.stem}.cleaned{input_path.suffix}")
    return input_path.with_name(f"{input_path.name}.cleaned.vtk")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean a mesh file with the refinement pipeline.")
    parser.add_argument("--refine", dest="input_path", type=Path, required=True, help="Input mesh file to clean.")
    parser.add_argument("--output", type=Path, help="Where to write the cleaned mesh.")
    parser.add_argument("--mode", choices=("object", "room"), default="object")
    parser.add_argument("--smoothing-iters", type=int, default=15)
    parser.add_argument("--pass-band", type=float, default=0.1)
    parser.add_argument("--hole-size", type=float)
    parser.add_argument("--min-cell-count", type=int, default=500)
    parser.add_argument("--feature-angle", type=float, default=45.0)
    parser.add_argument("--merge-tolerance", type=float, default=1e-5)
    parser.add_argument("--decimate-target-reduction", type=float)
    parser.add_argument("--no-watertight-check", action="store_true")
    return parser


def refine_mesh(input_path: Path, output_path: Path | None = None, config: MeshCleaningConfig | None = None) -> dict:
    mesh = pv.read(str(input_path))
    result = clean_mesh(mesh, config)
    cleaned_mesh = result["mesh"]
    destination = output_path or _default_output_path(input_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cleaned_mesh.save(str(destination))
    result = dict(result)
    result["input_path"] = str(input_path)
    result["output_path"] = str(destination)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = MeshCleaningConfig(
        mode=args.mode,
        smoothing_iters=args.smoothing_iters,
        pass_band=args.pass_band,
        hole_size=args.hole_size,
        min_cell_count=args.min_cell_count,
        feature_angle=args.feature_angle,
        merge_tolerance=args.merge_tolerance,
        decimate_target_reduction=args.decimate_target_reduction,
        verify_watertight=not args.no_watertight_check,
    )

    try:
        result = refine_mesh(args.input_path, args.output, config)
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