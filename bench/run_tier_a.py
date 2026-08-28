"""Run every Tier A experiment in the protocol's suggested order.

Tier A is CPU-only and needs no MASt3R checkpoint, so this is the whole
laptop-runnable results set. Ordering follows the protocol: findings that may
change the paper (A9, A5, A6) run before the measurement experiments, so a
result that invalidates a claim is known before any of Section V is written.

Each experiment writes its own CSV and is independent; a failure is reported
and the run continues, so one broken cell cannot cost a whole night.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import logging
import time
import traceback
from pathlib import Path

from bench import RESULTS_DIR

# (exp_id, module) in run order.
TIER_A: tuple[tuple[str, str], ...] = (
    ("a9_routing_matrix", "bench.exp_a9_routing_matrix"),
    ("a5_frame_budget", "bench.exp_a5_frame_budget"),
    ("a6_sync_offset", "bench.exp_a6_sync_offset"),
    ("a4_color_transfer", "bench.exp_a4_color_transfer"),
    ("a7_motion_sampling", "bench.exp_a7_motion_sampling"),
    ("a8_rigging_quality", "bench.exp_a8_rigging_quality"),
    ("a3_refine_quality", "bench.exp_a3_refine_quality"),
    ("a1_refine_scaling", "bench.exp_a1_refine_scaling"),
    ("a2_stage_profile", "bench.exp_a2_stage_profile"),
)

logger = logging.getLogger("bench.tier_a")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.run_tier_a", description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="reduced grids, for smoke-testing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        metavar="EXP_ID",
        help="run only these experiment ids (e.g. --only a1_refine_scaling a2_stage_profile)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    results_dir = args.results_dir or RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    selected = [entry for entry in TIER_A if args.only is None or entry[0] in set(args.only)]
    if args.only:
        unknown = set(args.only) - {exp_id for exp_id, _ in TIER_A}
        if unknown:
            parser.error(f"unknown experiment id(s): {sorted(unknown)}")

    failures: list[tuple[str, str]] = []
    overall_start = time.perf_counter()

    for exp_id, module_name in selected:
        logger.info("=== %s starting ===", exp_id)
        start = time.perf_counter()
        try:
            module = importlib.import_module(module_name)
            # Experiments share a signature, but accept only what each declares
            # so adding an experiment cannot break the whole night's run.
            accepted = inspect.signature(module.run).parameters
            kwargs = {"results_dir": results_dir}
            if "quick" in accepted:
                kwargs["quick"] = args.quick
            if "seed" in accepted:
                kwargs["seed"] = args.seed
            writer = module.run(**kwargs)
            path = writer.write()
            logger.info(
                "=== %s done in %.1f s: %d rows -> %s ===",
                exp_id,
                time.perf_counter() - start,
                len(writer),
                path,
            )
        except Exception:  # noqa: BLE001 - one failed experiment must not end the night
            failures.append((exp_id, traceback.format_exc()))
            logger.exception("=== %s FAILED after %.1f s ===", exp_id, time.perf_counter() - start)

    elapsed = time.perf_counter() - overall_start
    print(
        f"\nTier A finished in {elapsed / 60:.1f} min; {len(selected) - len(failures)}"
        f"/{len(selected)} experiments produced CSVs in {results_dir}"
    )
    for exp_id, trace in failures:
        print(f"\n--- {exp_id} failed ---\n{trace}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
