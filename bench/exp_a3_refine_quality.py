"""A3 - refinement geometric correctness on controlled corruption.

Supports the four assertions in SV-B: "component filtering removes isolated
fragments, hole filling improves surface completeness, Taubin smoothing
reduces local irregularity, and normal recomputation improves surface
consistency". None of them currently has a number behind it.

The trick that makes this cheap: damage a known-good mesh by a known amount,
then score the refined output against the *original*. No reconstruction ground
truth is needed, so this runs on a laptop.

Two defects were found here and have since been fixed in the code
(bench/FINDINGS.md #2 and #3); this experiment still records both signals so
a regression shows up as a result, not a silent behaviour change:

  fragments   `keep_object_components` used to merge every piece and drop
              none. It now keeps only the single largest piece by cell count,
              so object mode should show components-out == 1 regardless of
              how many fragments were injected.
  hole filling `fill_mesh_holes` returns early when `hole_size is None and
              is_sheet_like(...)`. The check now runs against the raw,
              pre-component-filter mesh rather than the filtered one, so the
              guard reflects whether the *capture* is inherently thin and
              open rather than an artifact of what filtering happened to
              leave behind. The trigger rate is recorded per cell either way.

part="refinement"        the corruption grid
part="smoothing_baseline" Taubin vs Laplacian at matched iterations, which is
                         what substantiates the shrink-free claim
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import trimesh

from bench.csvio import ResultWriter, aggregate
from bench.fixtures import corrupt_mesh
from bench.harness import experiment_parser, finish
from bench.meshes import ladder_mesh, to_pyvista
from bench.metrics import (
    chamfer_l1,
    hausdorff_95,
    normal_consistency,
    precision_recall_f,
    sample_points,
    unit_normalize,
)
from spatial_ingestion.refinement import MeshCleaningConfig, clean_mesh
from spatial_ingestion.refinement.core import (
    Mode,
    count_topology_issues,
    filter_room_components,
    is_sheet_like,
    keep_object_components,
)

EXP_ID = "a3_refine_quality"

BASES: tuple[str, ...] = ("icosphere", "torus", "pointmap_sheet")
NOISE_SIGMAS: tuple[float, ...] = (0.0, 0.002, 0.005, 0.01)
HOLE_COUNTS: tuple[int, ...] = (0, 5, 20)
FRAGMENT_COUNTS: tuple[int, ...] = (0, 5, 20)
SEEDS: tuple[int, ...] = (0, 1, 2)
MODES: tuple[Mode, ...] = ("object", "room")

MESH_TRIANGLES = 20_000

# 100k samples put the Chamfer noise floor -- two independent samplings of the
# *same* surface -- at 0.0016 of the bounding diagonal. At 30k it is 0.0029,
# which would swamp the smallest corruption level (sigma = 0.002).
SAMPLE_POINTS = 100_000
CHAMFER_NOISE_FLOOR = 0.0016
SMOOTHING_ITERS = 15

# Laplacian shrinkage compounds with iteration count; at 15 iterations it is
# below the measurement floor, so the Taubin claim is untestable there.
BASELINE_ITERATIONS: tuple[int, ...] = (15, 50, 100)

# Inputs are normalised to a unit bounding diagonal, so tau is directly a
# fraction of that diagonal. Stated on every row, per the reporting checklist.
TAU = 0.01

logger = logging.getLogger(__name__)


def _n_components(mesh: pv.DataSet) -> int:
    try:
        return int(len(mesh.split_bodies()))
    except Exception:  # noqa: BLE001 - VTK can refuse degenerate input
        return -1


def _pv_to_trimesh(mesh: pv.DataSet) -> trimesh.Trimesh:
    surface = mesh.extract_surface(algorithm=None).triangulate()
    faces = np.asarray(surface.faces).reshape(-1, 4)[:, 1:]
    return trimesh.Trimesh(
        vertices=np.asarray(surface.points, dtype=float), faces=faces, process=False
    )


# Candidates are sampled with a different seed from the reference. Sharing one
# would make the undamaged cell (sigma=0, no holes, no fragments) report a
# Chamfer of exactly 0 -- the identical point set compared with itself --
# instead of the real sampling noise floor, biasing the sigma=0 row downwards.
CANDIDATE_SEED_OFFSET = 1000


def _score(
    candidate: trimesh.Trimesh,
    reference_points: np.ndarray,
    reference: trimesh.Trimesh,
    seed: int,
) -> dict[str, float]:
    points = sample_points(candidate, SAMPLE_POINTS, seed + CANDIDATE_SEED_OFFSET)
    precision, recall, f_score = precision_recall_f(points, reference_points, TAU)
    return {
        "chamfer_l1": round(chamfer_l1(points, reference_points), 6),
        "hausdorff_95": round(hausdorff_95(points, reference_points), 6),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f_score": round(f_score, 4),
        "normal_consistency": round(
            normal_consistency(candidate, reference, 20_000, seed + CANDIDATE_SEED_OFFSET),
            5,
        ),
    }


def _guard_probe(poly: pv.PolyData, mode: Mode, min_cell_count: int) -> dict[str, Any]:
    """Reproduce the component-filter step so the hole-fill guard can be observed.

    `fill_mesh_holes` now tests `is_sheet_like` on the raw, pre-filter mesh
    (`poly`), not the filtered one -- so the guard result is independent of
    `mode` and of `filtered`, which is only computed here to also report
    `filtered_components`.
    """
    guard_fires = bool(is_sheet_like(poly))
    try:
        filtered = (
            keep_object_components(poly)
            if mode == "object"
            else filter_room_components(poly, min_cell_count)
        )
    except Exception as exc:  # noqa: BLE001 - room mode legitimately refuses
        # `filter_room_components` raises when no component clears
        # min_cell_count. That is a real outcome for a heavily fragmented
        # input, not a harness bug, so record it and move on; the guard
        # result itself is unaffected since it no longer depends on filtering.
        return {
            "holefill_guard_fires": guard_fires,
            "filtered_components": None,
            "guard_probe_error": str(exc)[:120],
        }
    return {
        "holefill_guard_fires": guard_fires,
        "filtered_components": _n_components(filtered),
    }


def _trial(
    base: str,
    noise_sigma: float,
    n_holes: int,
    n_fragments: int,
    seed: int,
    mode: Mode,
) -> dict[str, Any]:
    clean = unit_normalize(ladder_mesh(base, MESH_TRIANGLES, seed=0))
    clean_points = sample_points(clean, SAMPLE_POINTS, seed)

    damaged = corrupt_mesh(
        clean,
        noise_sigma=noise_sigma,
        n_holes=n_holes,
        n_fragments=n_fragments,
        seed=seed,
    )
    damaged_poly = to_pyvista(damaged)

    row: dict[str, Any] = {
        "part": "refinement",
        "base": base,
        "mode": mode,
        "noise_sigma": noise_sigma,
        "n_holes": n_holes,
        "n_fragments_injected": n_fragments,
        "seed": seed,
        "tau": TAU,
        "distance_unit": "fraction_of_unit_bbox_diagonal",
        "sample_points": SAMPLE_POINTS,
        "chamfer_noise_floor": CHAMFER_NOISE_FLOOR,
        "smoothing_iters": SMOOTHING_ITERS,
        "components_in": _n_components(damaged_poly),
        "in_cells": int(damaged_poly.n_cells),
    }
    row.update(count_topology_issues(damaged_poly.extract_surface(algorithm=None)))
    row["boundary_edges_in"] = row.pop("boundary_edge_count")
    row["non_manifold_edges_in"] = row.pop("non_manifold_edge_count")

    before = _score(damaged, clean_points, clean, seed)
    row.update({f"before_{key}": value for key, value in before.items()})

    config = MeshCleaningConfig(mode=mode, smoothing_iters=SMOOTHING_ITERS, verify_watertight=True)
    row.update(_guard_probe(damaged_poly, mode, config.min_cell_count))

    try:
        result = clean_mesh(damaged_poly, config)
    except Exception as exc:  # noqa: BLE001 - a refusal is a result, not a crash
        row["status"] = f"{type(exc).__name__}: {exc}"[:200]
        return row

    refined_poly = result["mesh"]
    refined = _pv_to_trimesh(refined_poly)
    after = _score(refined, clean_points, clean, seed)

    row.update(
        {
            "status": "ok",
            "out_cells": result["output_cell_count"],
            "components_out": _n_components(refined_poly),
            "boundary_edges_out": result["boundary_edge_count"],
            "non_manifold_edges_out": result["non_manifold_edge_count"],
            "is_watertight_out": result["is_watertight"],
            **{f"after_{key}": value for key, value in after.items()},
            "chamfer_delta": round(after["chamfer_l1"] - before["chamfer_l1"], 6),
            "chamfer_improved": bool(after["chamfer_l1"] < before["chamfer_l1"]),
            "normal_consistency_delta": round(
                after["normal_consistency"] - before["normal_consistency"], 5
            ),
            "fragments_removed": row["components_in"] - _n_components(refined_poly),
        }
    )
    return row


def _smoothing_baseline(base: str, noise_sigma: float, seed: int) -> list[dict[str, Any]]:
    """Taubin vs Laplacian at matched iterations.

    The paper's claim about Taubin is specifically that it avoids shrinkage, so
    volume ratio against the clean original is the measurement that matters;
    Chamfer alone would not distinguish them. Iterations are swept because
    Laplacian shrinkage compounds and is invisible at 15.
    """
    clean = unit_normalize(ladder_mesh(base, MESH_TRIANGLES, seed=0))
    clean_points = sample_points(clean, SAMPLE_POINTS, seed)
    clean_volume = float(clean.volume) if clean.is_watertight else float("nan")

    damaged = corrupt_mesh(clean, noise_sigma=noise_sigma, seed=seed)
    damaged_poly = to_pyvista(damaged).triangulate()

    rows: list[dict[str, Any]] = []
    for iterations in BASELINE_ITERATIONS:
        variants = {
            "taubin": damaged_poly.smooth_taubin(n_iter=iterations, pass_band=0.1),
            "laplacian": damaged_poly.smooth(n_iter=iterations),
            "none": damaged_poly,
        }
        for name, smoothed in variants.items():
            candidate = _pv_to_trimesh(smoothed)
            volume = float(candidate.volume) if candidate.is_watertight else float("nan")
            rows.append(
                {
                    "part": "smoothing_baseline",
                    "base": base,
                    "smoother": name,
                    "noise_sigma": noise_sigma,
                    "seed": seed,
                    "smoothing_iters": iterations,
                    "tau": TAU,
                    "distance_unit": "fraction_of_unit_bbox_diagonal",
                    "chamfer_noise_floor": CHAMFER_NOISE_FLOOR,
                    "status": "ok",
                    "volume_ratio_vs_clean": (
                        round(volume / clean_volume, 6)
                        if clean_volume and not np.isnan(clean_volume) and not np.isnan(volume)
                        else float("nan")
                    ),
                    **{
                        f"after_{key}": value
                        for key, value in _score(candidate, clean_points, clean, seed).items()
                    },
                }
            )
    return rows


def run(results_dir: Path | None = None, *, quick: bool = False, seed: int = 0) -> ResultWriter:
    writer = ResultWriter(EXP_ID, results_dir)
    bases = BASES[:1] if quick else BASES
    sigmas = NOISE_SIGMAS[:2] if quick else NOISE_SIGMAS
    holes = HOLE_COUNTS[:2] if quick else HOLE_COUNTS
    fragments = FRAGMENT_COUNTS[:2] if quick else FRAGMENT_COUNTS
    seeds = SEEDS[:1] if quick else SEEDS
    modes = MODES[:1] if quick else MODES

    for base in bases:
        for sigma in sigmas:
            for n_holes in holes:
                for n_fragments in fragments:
                    for trial_seed in seeds:
                        for mode in modes:
                            writer.add(
                                **_trial(base, sigma, n_holes, n_fragments, trial_seed, mode)
                            )
        logger.info("%s refinement grid done (%d rows)", base, len(writer))

        for sigma in sigmas:
            for trial_seed in seeds:
                for row in _smoothing_baseline(base, sigma, trial_seed):
                    writer.add(**row)
    return writer


def _summarise(writer: ResultWriter) -> None:
    grid = [r for r in writer.rows if r["part"] == "refinement" and r.get("status") == "ok"]

    print("\n  Chamfer before -> after (mean over seeds/holes/fragments)")
    print(f"  {'base':<16}{'mode':<8}{'sigma':>7}{'before':>10}{'after':>10}{'improved':>10}")
    for key, (mean_delta, _, n) in aggregate(
        grid, ["base", "mode", "noise_sigma"], "chamfer_delta"
    ).items():
        subset = [r for r in grid if (r["base"], r["mode"], r["noise_sigma"]) == key]
        before = float(np.mean([r["before_chamfer_l1"] for r in subset]))
        after = float(np.mean([r["after_chamfer_l1"] for r in subset]))
        improved = float(np.mean([r["chamfer_improved"] for r in subset]))
        print(
            f"  {key[0]:<16}{key[1]:<8}{key[2]:>7}{before:>10.5f}{after:>10.5f}"
            f"{improved:>10.2f}  (n={n}, delta={mean_delta:+.5f})"
        )

    print("\n  Fragment removal (object mode keeps only the largest component)")
    print(f"  {'base':<16}{'mode':<8}{'injected':>9}{'comp_in':>9}{'comp_out':>10}")
    for base in dict.fromkeys(r["base"] for r in grid):
        for mode in dict.fromkeys(r["mode"] for r in grid):
            for injected in FRAGMENT_COUNTS:
                subset = [
                    r
                    for r in grid
                    if r["base"] == base
                    and r["mode"] == mode
                    and r["n_fragments_injected"] == injected
                ]
                if not subset:
                    continue
                print(
                    f"  {base:<16}{mode:<8}{injected:>9}"
                    f"{np.mean([r['components_in'] for r in subset]):>9.1f}"
                    f"{np.mean([r['components_out'] for r in subset]):>10.1f}"
                )

    print("\n  Hole-fill guard trigger rate (is_sheet_like on the raw, pre-filter mesh)")
    for base in dict.fromkeys(r["base"] for r in grid):
        for mode in dict.fromkeys(r["mode"] for r in grid):
            subset = [
                r
                for r in grid
                if r["base"] == base and r["mode"] == mode and r["holefill_guard_fires"] is not None
            ]
            if subset:
                rate = float(np.mean([r["holefill_guard_fires"] for r in subset]))
                print(f"  {base:<16}{mode:<8}fires in {rate:.0%} of {len(subset)} cells")

    baseline = [r for r in writer.rows if r["part"] == "smoothing_baseline"]
    if baseline:
        print("\n  Shrinkage: volume relative to the clean original (1.0 = no shrink)")
        print(f"  {'base':<16}{'smoother':<12}{'iters':>6}{'vol_ratio':>11}{'chamfer':>10}")
        for base in dict.fromkeys(r["base"] for r in baseline):
            for iterations in dict.fromkeys(r["smoothing_iters"] for r in baseline):
                for smoother in dict.fromkeys(r["smoother"] for r in baseline):
                    subset = [
                        r
                        for r in baseline
                        if r["base"] == base
                        and r["smoother"] == smoother
                        and r["smoothing_iters"] == iterations
                    ]
                    if not subset:
                        continue
                    ratios = [
                        r["volume_ratio_vs_clean"]
                        for r in subset
                        if not np.isnan(r["volume_ratio_vs_clean"])
                    ]
                    chamfer = float(np.mean([r["after_chamfer_l1"] for r in subset]))
                    ratio_text = f"{np.mean(ratios):.5f}" if ratios else "n/a (open)"
                    print(
                        f"  {base:<16}{smoother:<12}{iterations:>6}{ratio_text:>11}{chamfer:>10.5f}"
                    )


def main(argv: list[str] | None = None) -> int:
    args = experiment_parser(EXP_ID, __doc__ or "").parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    writer = run(results_dir=args.results_dir, quick=args.quick, seed=args.seed)
    finish(writer)
    _summarise(writer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
