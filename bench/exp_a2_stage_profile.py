"""A2 - per-stage refinement profile.

The README attributes refinement cost to `split_bodies()`; the paper drops the
attribution entirely because nothing measured it. This reports seconds and
percentage of total per stage, and saves a cProfile so the attribution can be
pushed down to the function that actually spends the time.

Two configurations, because A1 shows they have different dominant stages:

  big_triangles   1M triangles, one component  -- the size the paper describes
  many_components 100k triangles, 1000 pieces  -- the shape a fused MASt3R
                  pointmap actually has

If `component_filter` dominates, that is `split_bodies()` plus
`merge_components`, and the cProfile distinguishes them. If `finalize` or
`transfer_data` dominates, the optimisation target is different, and the
paper's future-work item should say so.
"""

from __future__ import annotations

import cProfile
import io
import logging
import pstats
import time
from pathlib import Path
from typing import Any

import pyvista as pv

from bench import RESULTS_DIR
from bench.csvio import ResultWriter, mean_std
from bench.harness import experiment_parser, finish
from bench.meshes import fragmented_mesh, ladder_mesh, to_pyvista
from spatial_ingestion.instrumentation import peak_rss_mb
from spatial_ingestion.refinement import MeshCleaningConfig, clean_mesh

EXP_ID = "a2_stage_profile"
REPEATS = 5
PROFILE_TOP_N = 25

CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "big_triangles": {"kind": "triangles", "n_tri": 1_000_000, "n_components": 1},
    "many_components": {"kind": "components", "n_tri": 100_000, "n_components": 1000},
}

logger = logging.getLogger(__name__)


def _build(spec: dict[str, Any], seed: int) -> pv.PolyData:
    if spec["kind"] == "triangles":
        return to_pyvista(ladder_mesh("icosphere", int(spec["n_tri"]), seed=seed))
    return to_pyvista(fragmented_mesh(int(spec["n_tri"]), int(spec["n_components"]), seed=seed))


def _profile_to_text(mesh: pv.PolyData, config: MeshCleaningConfig, prof_path: Path) -> str:
    profiler = cProfile.Profile()
    profiler.enable()
    clean_mesh(mesh, config)
    profiler.disable()
    prof_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(prof_path))

    buffer = io.StringIO()
    pstats.Stats(profiler, stream=buffer).sort_stats("cumtime").print_stats(PROFILE_TOP_N)
    text = buffer.getvalue()
    prof_path.with_suffix(".txt").write_text(text, encoding="utf-8")
    return text


def run(
    results_dir: Path | None = None,
    *,
    quick: bool = False,
    seed: int = 0,
    repeats: int = REPEATS,
    profile: bool = True,
) -> ResultWriter:
    writer = ResultWriter(EXP_ID, results_dir)
    out_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    configurations = (
        {"many_components": CONFIGURATIONS["many_components"]} if quick else CONFIGURATIONS
    )
    n_repeats = 1 if quick else repeats

    for name, spec in configurations.items():
        mesh = _build(spec, seed)
        config = MeshCleaningConfig(mode="object", smoothing_iters=15, verify_watertight=True)

        for repeat in range(n_repeats):
            start = time.perf_counter()
            result = clean_mesh(mesh, config)
            total = time.perf_counter() - start
            stage_total = max(result["total_stage_seconds"], 1e-9)

            for record in result["stage_timings"]:
                writer.add(
                    configuration=name,
                    n_tri=int(mesh.n_cells),
                    n_components_nominal=int(spec["n_components"]),
                    repeat=repeat,
                    seed=seed,
                    stage=record["stage"],
                    seconds=record["seconds"],
                    pct_of_stage_total=round(100.0 * record["seconds"] / stage_total, 2),
                    stage_peak_rss_mb=record["peak_rss_mb"],
                    stage_rss_delta_mb=record["rss_delta_mb"],
                    wall_total_seconds=round(total, 4),
                    stage_total_seconds=result["total_stage_seconds"],
                    unaccounted_seconds=round(total - result["total_stage_seconds"], 4),
                    peak_rss_mb=peak_rss_mb(),
                )
            logger.info("%s repeat %d: %.2f s", name, repeat, total)

        if profile:
            prof_path = out_dir / f"{EXP_ID}_{name}.prof"
            text = _profile_to_text(mesh, config, prof_path)
            logger.info("profile written: %s", prof_path)
            print(f"\n--- cProfile top {PROFILE_TOP_N} ({name}) ---")
            print("\n".join(text.splitlines()[:12]))
    return writer


def _summarise(writer: ResultWriter) -> None:
    print(f"\n  {'configuration':<17}{'stage':<20}{'mean_s':>9}{'sd':>7}{'pct':>7}")
    for configuration in dict.fromkeys(r["configuration"] for r in writer.rows):
        rows = [r for r in writer.rows if r["configuration"] == configuration]
        for stage in dict.fromkeys(r["stage"] for r in rows):
            group = [r for r in rows if r["stage"] == stage]
            mean, std, _ = mean_std([r["seconds"] for r in group])
            pct, _, _ = mean_std([r["pct_of_stage_total"] for r in group])
            print(f"  {configuration:<17}{stage:<20}{mean:>9.3f}{std:>7.3f}{pct:>7.1f}")


def main(argv: list[str] | None = None) -> int:
    parser = experiment_parser(EXP_ID, __doc__ or "")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--no-profile", action="store_true", help="skip the cProfile pass")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    writer = run(
        results_dir=args.results_dir,
        quick=args.quick,
        seed=args.seed,
        repeats=args.repeats,
        profile=not args.no_profile,
    )
    finish(writer)
    _summarise(writer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
