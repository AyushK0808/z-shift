"""A1 - refinement cost scaling. Replaces Table I.

Table I currently reports one remembered wall-clock run (~2.6M triangles,
~7 minutes, smoothing off) with no machine attached to it. This measures the
cost surface instead, on two axes:

  ladder="triangles"     size sweep at a fixed component count
  ladder="components"    component sweep at a fixed triangle count
  ladder="interaction"   both raised together, in exactly Table I's
                         configuration (smoothing off, watertightness skipped)

The second axis exists because a pilot run showed triangle count alone cannot
reproduce the reported cost: 500k triangles in one component refines in about
3 s on this machine. A fused MASt3R pointmap is not one connected surface, and
component-filtering's `split_bodies()` walks every piece regardless of how
many survive filtering, so component count is an independent -- and far
steeper -- cost driver.

The "components" and "interaction" ladders run in room mode with
`min_cell_count=0`, not object mode: object mode now keeps only the largest
component (bench/FINDINGS.md #2), which would shrink the mesh handed to
smoothing and finalization as the fragment count rose, confounding the very
axis this experiment holds fixed. Room mode with no minimum keeps every
fragment, same as object mode used to before that fix.

The interaction ladder exists because neither axis alone reaches the reported
cost, but together they are strongly super-additive: 2.5M triangles in one
component and 100k triangles in 1000 components are each about 10 s, while
2.5M triangles in 1000 components is roughly ten times that.

Two mesh sources, so a result is not a property of one shape: a closed
icosphere and a displaced grid standing in for a fused per-view pointmap.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pyvista as pv

from bench.csvio import ResultWriter, mean_std
from bench.harness import experiment_parser, finish
from bench.meshes import SOURCES, fragmented_mesh, ladder_mesh, to_pyvista
from spatial_ingestion.instrumentation import peak_rss_mb
from spatial_ingestion.refinement import MeshCleaningConfig, clean_mesh
from spatial_ingestion.refinement.core import Mode, is_sheet_like

EXP_ID = "a1_refine_scaling"

TRIANGLE_LADDER: tuple[int, ...] = (50_000, 100_000, 250_000, 500_000, 1_000_000, 2_500_000)
COMPONENT_LADDER: tuple[int, ...] = (1, 10, 50, 100, 250, 500, 1000)
COMPONENT_LADDER_TRIANGLES = 100_000

# Table I reports ~2.6M triangles with smoothing disabled and the
# watertightness check skipped, so the interaction ladder runs exactly that
# configuration -- the point is to be directly comparable to the number the
# paper already prints, not to explore the config space again.
INTERACTION_CELLS: tuple[tuple[int, int], ...] = (
    (500_000, 500),
    (1_000_000, 1000),
    (2_500_000, 1000),
)
INTERACTION_SMOOTHING = 0
INTERACTION_WATERTIGHT = False

SMOOTHING_ITERS: tuple[int, ...] = (0, 15)
WATERTIGHT: tuple[bool, ...] = (False, True)
REPEATS = 3

# A cell that exceeds this is recorded, and larger rungs of the same ladder are
# then skipped for that configuration and written out as budget_exceeded rather
# than silently left blank. The protocol's rule: a ceiling is itself a result.
DEFAULT_CELL_BUDGET_S = 240.0

logger = logging.getLogger(__name__)


def _measure(
    mesh: pv.PolyData,
    *,
    smoothing: int,
    watertight: bool,
    mode: Mode = "object",
) -> dict[str, Any]:
    """One timed `clean_mesh` call. Cleaning is timed; mesh building is not."""
    config = MeshCleaningConfig(
        mode=mode,
        smoothing_iters=smoothing,
        verify_watertight=watertight,
        # Room mode with min_cell_count=0 keeps every fragment, same as object
        # mode used to before it was fixed to keep only the largest component
        # (bench/FINDINGS.md #2). The "components" and "interaction" ladders
        # need every fragment kept so component count stays the cost driver
        # being measured, rather than being confounded by how much geometry
        # object mode now discards.
        min_cell_count=0 if mode == "room" else 500,
    )
    start = time.perf_counter()
    result = clean_mesh(mesh, config)
    elapsed = time.perf_counter() - start

    stage_seconds = {
        f"stage_{record['stage']}_s": record["seconds"] for record in result["stage_timings"]
    }
    return {
        "seconds": round(elapsed, 4),
        "peak_rss_mb": peak_rss_mb(),
        "in_points": result["input_point_count"],
        "in_cells": result["input_cell_count"],
        "out_points": result["output_point_count"],
        "out_cells": result["output_cell_count"],
        "cell_ratio": round(result["output_cell_count"] / max(result["input_cell_count"], 1), 4),
        "is_watertight": result["is_watertight"],
        "total_stage_seconds": result["total_stage_seconds"],
        "status": "ok",
        **stage_seconds,
    }


def _build(ladder: str, source: str, rung: int, seed: int) -> pv.PolyData:
    if ladder == "triangles":
        return to_pyvista(ladder_mesh(source, rung, seed=seed))
    return to_pyvista(fragmented_mesh(COMPONENT_LADDER_TRIANGLES, rung, seed=seed))


def run(
    results_dir: Path | None = None,
    *,
    quick: bool = False,
    seed: int = 0,
    repeats: int = REPEATS,
    cell_budget_s: float = DEFAULT_CELL_BUDGET_S,
) -> ResultWriter:
    writer = ResultWriter(EXP_ID, results_dir)

    triangle_rungs = TRIANGLE_LADDER[:2] if quick else TRIANGLE_LADDER
    component_rungs = COMPONENT_LADDER[:3] if quick else COMPONENT_LADDER
    sources = SOURCES[:1] if quick else SOURCES
    smoothings = SMOOTHING_ITERS[:1] if quick else SMOOTHING_ITERS
    watertights = WATERTIGHT[:1] if quick else WATERTIGHT
    n_repeats = 1 if quick else repeats

    plans: list[tuple[str, str, tuple[int, ...]]] = [
        ("triangles", source, triangle_rungs) for source in sources
    ]
    # The component ladder builds its own multi-piece geometry, so it has one
    # nominal source rather than one per shape.
    plans.append(("components", "fragmented_icosphere", component_rungs))

    for ladder, source, rungs in plans:
        for smoothing in smoothings:
            for watertight in watertights:
                over_budget = False
                for rung in rungs:
                    common = {
                        "ladder": ladder,
                        "source": source,
                        "rung": rung,
                        "n_tri_nominal": (
                            rung if ladder == "triangles" else COMPONENT_LADDER_TRIANGLES
                        ),
                        "n_components_nominal": 1 if ladder == "triangles" else rung,
                        "smoothing_iters": smoothing,
                        "verify_watertight": watertight,
                        "seed": seed,
                    }
                    if over_budget:
                        for repeat in range(n_repeats):
                            writer.add(
                                **common,
                                repeat=repeat,
                                status="skipped_budget_exceeded",
                                cell_budget_s=cell_budget_s,
                            )
                        continue

                    mesh = _build(ladder, source, rung, seed)
                    sheet_like = bool(is_sheet_like(mesh))
                    ladder_mode = "room" if ladder == "components" else "object"
                    for repeat in range(n_repeats):
                        try:
                            measured = _measure(
                                mesh, smoothing=smoothing, watertight=watertight, mode=ladder_mode
                            )
                        except MemoryError as exc:
                            measured = {"status": f"MemoryError: {exc}"}
                        writer.add(
                            **common,
                            repeat=repeat,
                            input_is_sheet_like=sheet_like,
                            cell_budget_s=cell_budget_s,
                            **measured,
                        )
                        logger.info(
                            "%s %s rung=%s smooth=%d wt=%s rep=%d -> %s s",
                            ladder,
                            source,
                            rung,
                            smoothing,
                            watertight,
                            repeat,
                            measured.get("seconds"),
                        )
                        if float(measured.get("seconds") or 0.0) > cell_budget_s:
                            over_budget = True

    interaction_cells = INTERACTION_CELLS[:1] if quick else INTERACTION_CELLS
    for n_tri, n_components in interaction_cells:
        mesh = to_pyvista(fragmented_mesh(n_tri, n_components, seed=seed))
        for repeat in range(n_repeats):
            try:
                measured = _measure(
                    mesh,
                    smoothing=INTERACTION_SMOOTHING,
                    watertight=INTERACTION_WATERTIGHT,
                    mode="room",
                )
            except MemoryError as exc:
                measured = {"status": f"MemoryError: {exc}"}
            writer.add(
                ladder="interaction",
                source="fragmented_icosphere",
                rung=n_components,
                n_tri_nominal=n_tri,
                n_components_nominal=n_components,
                smoothing_iters=INTERACTION_SMOOTHING,
                verify_watertight=INTERACTION_WATERTIGHT,
                seed=seed,
                repeat=repeat,
                cell_budget_s=cell_budget_s,
                **measured,
            )
            logger.info(
                "interaction n_tri=%d components=%d rep=%d -> %s s",
                n_tri,
                n_components,
                repeat,
                measured.get("seconds"),
            )
    return writer


def _summarise(writer: ResultWriter) -> None:
    print(
        f"  {'ladder':<11}{'source':<22}{'rung':>9}{'sm':>4}"
        f"{'wt':>4}{'mean_s':>9}{'sd':>7}{'MB':>8}"
    )
    seen: list[tuple[Any, ...]] = []
    for row in writer.rows:
        key = (
            row["ladder"],
            row["source"],
            row["rung"],
            row["smoothing_iters"],
            row["verify_watertight"],
        )
        if key in seen:
            continue
        seen.append(key)
        group = [
            r
            for r in writer.rows
            if (r["ladder"], r["source"], r["rung"], r["smoothing_iters"], r["verify_watertight"])
            == key
        ]
        values = [r["seconds"] for r in group if r.get("status") == "ok"]
        if not values:
            print(f"  {key[0]:<11}{key[1]:<22}{key[2]:>9}{key[3]:>4}{str(key[4]):>4}{'skipped':>9}")
            continue
        mean, std, _ = mean_std(values)
        peak = max(r.get("peak_rss_mb") or 0 for r in group)
        print(
            f"  {key[0]:<11}{key[1]:<22}{key[2]:>9}{key[3]:>4}{str(key[4]):>4}"
            f"{mean:>9.2f}{std:>7.2f}{peak:>8.0f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = experiment_parser(EXP_ID, __doc__ or "")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--cell-budget-s", type=float, default=DEFAULT_CELL_BUDGET_S)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    writer = run(
        results_dir=args.results_dir,
        quick=args.quick,
        seed=args.seed,
        repeats=args.repeats,
        cell_budget_s=args.cell_budget_s,
    )
    finish(writer)
    _summarise(writer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
