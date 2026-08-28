"""A4 - vertex-colour transfer fidelity.

Supports the SV-B claim that "vertex-color transfer allows the refined mesh to
retain the visual appearance of the reconstructed geometry", which is
currently backed only by a boolean round-trip test.

The input carries an *analytic* colour field (RGB = normalised XYZ position),
so the ground-truth colour of any output vertex is a closed-form function of
where that vertex ended up. Transfer error is therefore measurable without any
vertex correspondence between input and output, which smoothing and decimation
destroy.

`preserve_data_arrays` does a nearest-neighbour KD-tree lookup into the source
points, so error should scale with how far output vertices move from their
nearest input vertex; that distance is recorded alongside so the relationship
can be checked rather than assumed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import KDTree

from bench.csvio import ResultWriter, mean_std
from bench.harness import experiment_parser, finish
from bench.meshes import colored_pyvista, ladder_mesh
from bench.metrics import unit_normalize
from spatial_ingestion.refinement import MeshCleaningConfig, clean_mesh

EXP_ID = "a4_color_transfer"

SOURCES: tuple[str, ...] = ("icosphere", "pointmap_sheet")
SMOOTHING_ITERS: tuple[int, ...] = (0, 5, 15, 30)
DECIMATIONS: tuple[float | None, ...] = (None, 0.3, 0.5, 0.7)
SEEDS: tuple[int, ...] = (0, 1, 2)
MESH_TRIANGLES = 40_000

# Roughly the just-noticeable difference for an 8-bit channel; the fraction of
# vertices above it is the number a reader actually cares about.
JND_THRESHOLD = 8.0

logger = logging.getLogger(__name__)


def _trial(source: str, smoothing: int, decimation: float | None, seed: int) -> dict[str, Any]:
    mesh = unit_normalize(ladder_mesh(source, MESH_TRIANGLES, seed=seed))
    poly = colored_pyvista(mesh)
    source_points = np.asarray(poly.points, dtype=float)

    result = clean_mesh(
        poly,
        MeshCleaningConfig(
            mode="object",
            smoothing_iters=smoothing,
            decimate_target_reduction=decimation,
            verify_watertight=False,
        ),
    )
    cleaned = result["mesh"]
    transferred = cleaned.point_data.get("RGB")
    if transferred is None:
        return {
            "status": "no_color_array_on_output",
            "out_points": int(cleaned.n_points),
        }

    out_points = np.asarray(cleaned.points, dtype=float)
    # Ground truth is evaluated on the *source* bounding box so the analytic
    # field is the same function for input and output, not a rescaled one.
    low = source_points.min(axis=0)
    span = np.maximum(source_points.max(axis=0) - low, 1e-12)
    expected = np.clip(((out_points - low) / span) * 255.0, 0, 255)

    error = np.abs(np.asarray(transferred, dtype=float)[:, :3] - expected)
    per_vertex_max = error.max(axis=1)
    nn_distance = KDTree(source_points).query(out_points, k=1)[0]

    return {
        "status": "ok",
        "out_points": int(cleaned.n_points),
        "out_cells": int(cleaned.n_cells),
        "point_ratio": round(cleaned.n_points / max(poly.n_points, 1), 4),
        "mean_abs_error": round(float(error.mean()), 4),
        "p95_abs_error": round(float(np.percentile(error, 95)), 4),
        "max_abs_error": round(float(error.max()), 4),
        "frac_vertices_over_jnd": round(float((per_vertex_max > JND_THRESHOLD).mean()), 4),
        "mean_nn_distance": round(float(nn_distance.mean()), 6),
        "p95_nn_distance": round(float(np.percentile(nn_distance, 95)), 6),
    }


def run(results_dir: Path | None = None, *, quick: bool = False, seed: int = 0) -> ResultWriter:
    writer = ResultWriter(EXP_ID, results_dir)
    sources = SOURCES[:1] if quick else SOURCES
    smoothings = SMOOTHING_ITERS[:2] if quick else SMOOTHING_ITERS
    decimations = DECIMATIONS[:2] if quick else DECIMATIONS
    seeds = SEEDS[:1] if quick else SEEDS

    for source in sources:
        for smoothing in smoothings:
            for decimation in decimations:
                for trial_seed in seeds:
                    row = _trial(source, smoothing, decimation, trial_seed)
                    writer.add(
                        source=source,
                        smoothing_iters=smoothing,
                        decimate_target_reduction=(decimation if decimation is not None else ""),
                        seed=trial_seed,
                        jnd_threshold=JND_THRESHOLD,
                        n_tri_nominal=MESH_TRIANGLES,
                        **row,
                    )
        logger.info("%s done (%d rows)", source, len(writer))
    return writer


def _summarise(writer: ResultWriter) -> None:
    print(
        f"  {'source':<16}{'smooth':>7}{'decim':>7}{'mean_err':>10}{'p95':>8}{'>JND':>8}{'nn_d':>9}"
    )
    keys: list[tuple[Any, ...]] = []
    for row in writer.rows:
        key = (row["source"], row["smoothing_iters"], row["decimate_target_reduction"])
        if key in keys:
            continue
        keys.append(key)
        group = [
            r
            for r in writer.rows
            if (r["source"], r["smoothing_iters"], r["decimate_target_reduction"]) == key
            and r.get("status") == "ok"
        ]
        if not group:
            continue
        mean_err, _, _ = mean_std([r["mean_abs_error"] for r in group])
        p95, _, _ = mean_std([r["p95_abs_error"] for r in group])
        over, _, _ = mean_std([r["frac_vertices_over_jnd"] for r in group])
        nn_d, _, _ = mean_std([r["mean_nn_distance"] for r in group])
        print(
            f"  {key[0]:<16}{key[1]:>7}{str(key[2] or '-'):>7}"
            f"{mean_err:>10.2f}{p95:>8.2f}{over:>8.3f}{nn_d:>9.5f}"
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
