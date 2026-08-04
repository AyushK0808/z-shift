"""Shared CLI wiring for the Phase 3 mesh-cleaning configuration.

Both the standalone refinement CLI and the final pipeline CLI expose the same
cleaning knobs; defining them once keeps the two entry points in sync.
"""

from __future__ import annotations

import argparse

from .core import MeshCleaningConfig

_DEFAULTS = MeshCleaningConfig()


def add_mesh_cleaning_args(parser: argparse.ArgumentParser, *, mode_flag: str = "--mode") -> None:
    """Add every MeshCleaningConfig knob as argparse options."""
    parser.add_argument(
        mode_flag,
        dest="mode",
        choices=("object", "room"),
        default=_DEFAULTS.mode,
        help="Cleaning mode (default: %(default)s)",
    )
    parser.add_argument(
        "--smoothing-iters",
        type=int,
        default=_DEFAULTS.smoothing_iters,
        help="Taubin smoothing iterations; 0 disables smoothing",
    )
    parser.add_argument(
        "--pass-band", type=float, default=_DEFAULTS.pass_band, help="Taubin pass-band"
    )
    parser.add_argument(
        "--hole-size",
        type=float,
        default=_DEFAULTS.hole_size,
        help="Max hole size to fill (default: auto-sized to model scale)",
    )
    parser.add_argument(
        "--min-cell-count",
        type=int,
        default=_DEFAULTS.min_cell_count,
        help="Room mode: drop components at or below this size",
    )
    parser.add_argument(
        "--feature-angle",
        type=float,
        default=_DEFAULTS.feature_angle,
        help="Room mode: sharp-edge preservation threshold",
    )
    parser.add_argument(
        "--merge-tolerance",
        type=float,
        default=_DEFAULTS.merge_tolerance,
        help="Relative tolerance for duplicate-point merging",
    )
    parser.add_argument(
        "--decimate-target-reduction",
        type=float,
        default=_DEFAULTS.decimate_target_reduction,
        help="e.g. 0.5 drops ~50%% of triangles; unset keeps all",
    )
    parser.add_argument(
        "--no-watertight-check",
        action="store_true",
        help="Skip the open-edge watertight check",
    )


def mesh_cleaning_config_from_args(args: argparse.Namespace) -> MeshCleaningConfig:
    return MeshCleaningConfig(
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
