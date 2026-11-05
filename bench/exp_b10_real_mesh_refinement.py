"""B10 - refinement on the real DTU meshes B6 saved. Needs B6's outputs; no new reconstruction.

Answers the reviewer's request to rerun the batched component filter on several
real meshes, with variance, and to justify how much of a mesh object mode drops.

Two parts, one scene at a time (scan24, 37, 40, 55 and 63, the same five as B2
and B6), both on the raw mesh B6 wrote:

- `part=accuracy`: raw, and refined in object, room and object-with-isolation
  modes, each scored against DTU's ground truth as B6 scores it, plus the share
  of faces kept and the mean |cos| between mesh normals and the ground truth's
  (bench.dtu_eval.mesh_normal_consistency). B6 ran with isolation off; here it
  is a row of its own, so the cost of "keep the largest component" and what an
  isolating step recovers are side by side.
- `part=timing`: the component filters, `labels` (batched, on the GPU) and
  `split_bodies` (the submitted per-component VTK path), repeated, with the
  output size each produced. Refinement is deterministic, so the variance here
  is timing only. `split_bodies` runs in a child process under a time budget: on
  the CO3D teddybear it took half an hour on one mesh, and a hung cell should be
  a recorded timeout, not a stalled run.

    uv run python -m bench.exp_b10_real_mesh_refinement \\
        --manifest ~/zshift/dtu/manifest.json --outputs ~/zshift/tier_b/full/outputs/b6
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from bench.csvio import ResultWriter
from bench.dtu_eval import mesh_normal_consistency
from bench.exp_b6_refinement_effect import _mesh_points
from bench.harness import experiment_parser, finish
from bench.tier_b_common import (
    SceneSet,
    iter_scenes,
    reconstruction_scale_for,
    score_against_gt,
)

EXP_ID = "b10_real_mesh_refinement"
SMOOTHING_ITERS = 15
# (label, mode, isolate_object)
ACCURACY_VARIANTS: tuple[tuple[str, str, bool], ...] = (
    ("object", "object", False),
    ("room", "room", False),
    ("object_isolated", "object", True),
)
FILTERS: tuple[str, ...] = ("labels", "split_bodies")
DEFAULT_REPEATS = 5
DEFAULT_SPLIT_BUDGET_S = 1800.0
# A split_bodies cell slower than this is not repeated: its variance is
# already small next to its mean, and repeats would multiply an hour of compute.
_SLOW_CELL_S = 120.0

logger = logging.getLogger(__name__)


def _config(mode: str, isolate: bool, component_filter: str, device: str = "auto") -> Any:
    from spatial_ingestion.refinement import MeshCleaningConfig

    return MeshCleaningConfig(
        mode=mode,  # ty: ignore[invalid-argument-type]
        smoothing_iters=SMOOTHING_ITERS,
        verify_watertight=True,
        isolate_object=isolate,
        component_filter=component_filter,  # ty: ignore[invalid-argument-type]
        device=device,  # ty: ignore[invalid-argument-type]
    )


def _timing_child(path: str, mode: str, component_filter: str, queue: Any) -> None:
    """Refine `path` once and report the result; runs in a child so it can be killed."""
    from spatial_ingestion.refinement import clean_mesh, load_mesh_file

    try:
        mesh = load_mesh_file(path)
        start = time.perf_counter()
        result = clean_mesh(mesh, _config(mode, False, component_filter, "cpu"))
        queue.put(
            {
                "seconds": round(time.perf_counter() - start, 3),
                "out_cells": int(result["output_cell_count"]),
                "out_points": int(result["output_point_count"]),
                "status": "ok",
            }
        )
    except Exception as exc:  # noqa: BLE001 - a refusal is a result
        queue.put({"status": f"{type(exc).__name__}: {exc}"[:300]})


def _time_in_child(path: Path, mode: str, component_filter: str, budget_s: float) -> dict[str, Any]:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_timing_child, args=(str(path), mode, component_filter, queue), daemon=True
    )
    process.start()
    logger.info(
        "%s: started (pid %s, budget %.0f s); nothing prints until it finishes, "
        "so a line is logged every minute",
        component_filter,
        process.pid,
        budget_s,
    )
    started = time.perf_counter()
    while process.is_alive() and time.perf_counter() - started < budget_s:
        process.join(min(60.0, budget_s - (time.perf_counter() - started)))
        if process.is_alive():
            logger.info(
                "%s: still running, %.0f of %.0f s",
                component_filter,
                time.perf_counter() - started,
                budget_s,
            )
    if process.is_alive():
        process.kill()
        process.join()
        return {"status": "timeout", "seconds": float("nan"), "budget_s": budget_s}
    return queue.get() if not queue.empty() else {"status": "child exited without a result"}


def _time_in_process(raw_poly: Any, mode: str, component_filter: str) -> dict[str, Any]:
    from spatial_ingestion.refinement import clean_mesh

    start = time.perf_counter()
    result = clean_mesh(raw_poly, _config(mode, False, component_filter))
    return {
        "seconds": round(time.perf_counter() - start, 3),
        "out_cells": int(result["output_cell_count"]),
        "out_points": int(result["output_point_count"]),
        "refine_device": result["compute_device"],
        "status": "ok",
    }


def _accuracy_rows(
    writer: ResultWriter, scene: Any, raw_poly: Any, output_dir: Path, seed: int
) -> None:
    from spatial_ingestion.refinement import clean_mesh, to_trimesh

    reference = scene.dtu_reference()
    scale = reconstruction_scale_for(scene.units)
    raw_faces = int(raw_poly.n_cells)

    def score(mesh: Any) -> dict[str, Any]:
        scores = score_against_gt(
            _mesh_points(mesh, seed),
            np.empty((0, 3)),
            tau=scene.tau,
            seed=seed,
            reconstruction_scale=scale,
            reference=reference,
            output_dir=output_dir,
        )
        return {**scores, **mesh_normal_consistency(mesh, reference, output_dir, seed=seed)}

    raw = score(to_trimesh(raw_poly))
    writer.add(
        scene=scene.name,
        part="accuracy",
        variant="raw",
        status="ok",
        n_faces=raw_faces,
        face_fraction_kept=1.0,
        **raw,
    )
    for label, mode, isolate in ACCURACY_VARIANTS:
        try:
            result = clean_mesh(raw_poly, _config(mode, isolate, "labels"))
        except Exception as exc:  # noqa: BLE001 - a refusal is a result
            writer.add(
                scene=scene.name,
                part="accuracy",
                variant=label,
                status=f"{type(exc).__name__}: {exc}"[:300],
            )
            continue
        refined = to_trimesh(result["mesh"])
        scores = score(refined)
        support = {
            key: result[key]
            for key in ("support_plane_removed", "support_plane_face_frac")
            if key in result
        }
        writer.add(
            scene=scene.name,
            part="accuracy",
            variant=label,
            status="ok",
            n_faces=int(len(refined.faces)),
            face_fraction_kept=round(len(refined.faces) / max(raw_faces, 1), 4),
            is_watertight=result["is_watertight"],
            f_score_delta_vs_raw=round(scores["f_score"] - raw["f_score"], 4),
            chamfer_delta_vs_raw=round(scores["chamfer_l1"] - raw["chamfer_l1"], 6),
            normal_consistency_delta_vs_raw=round(
                scores["normal_consistency"] - raw["normal_consistency"], 5
            ),
            **support,
            **scores,
        )
        logger.info(
            "%s %s: kept %.0f%% of faces, F %.3f -> %.3f, NC %.3f -> %.3f",
            scene.name,
            label,
            100 * len(refined.faces) / max(raw_faces, 1),
            raw["f_score"],
            scores["f_score"],
            raw["normal_consistency"],
            scores["normal_consistency"],
        )


def _timing_rows(
    writer: ResultWriter,
    scene: Any,
    raw_path: Path,
    raw_poly: Any,
    repeats: int,
    split_budget_s: float,
) -> None:
    for component_filter in FILTERS:
        seconds: list[float] = []
        cells: set[int] = set()
        points: set[int] = set()
        status = "ok"
        # The first labels call pays for CUDA context creation; it is a warm-up.
        if component_filter == "labels":
            _time_in_process(raw_poly, "object", "labels")
        for _ in range(repeats):
            result = (
                _time_in_process(raw_poly, "object", "labels")
                if component_filter == "labels"
                else _time_in_child(raw_path, "object", component_filter, split_budget_s)
            )
            if result["status"] != "ok":
                status = result["status"]
                break
            seconds.append(result["seconds"])
            cells.add(result["out_cells"])
            points.add(result["out_points"])
            if seconds[-1] > _SLOW_CELL_S:
                break
        writer.add(
            scene=scene.name,
            part="timing",
            filter_method=component_filter,
            status=status,
            n_repeats=len(seconds),
            seconds_mean=round(statistics.fmean(seconds), 3) if seconds else float("nan"),
            seconds_sd=round(statistics.stdev(seconds), 3) if len(seconds) > 1 else 0.0,
            seconds_min=min(seconds) if seconds else float("nan"),
            seconds_max=max(seconds) if seconds else float("nan"),
            out_cells=min(cells) if cells else "",
            out_points=min(points) if points else "",
            output_size_varies=len(cells) > 1 or len(points) > 1,
            in_cells=int(raw_poly.n_cells),
            in_points=int(raw_poly.n_points),
        )
        logger.info(
            "%s %s: %s over %d repeat(s)", scene.name, component_filter, status, len(seconds)
        )


def run(
    results_dir: Path | None = None,
    *,
    manifest: Path,
    outputs: Path,
    scene_names: list[str] | None = None,
    seed: int = 0,
    repeats: int = DEFAULT_REPEATS,
    split_budget_s: float = DEFAULT_SPLIT_BUDGET_S,
    quick: bool = False,
) -> ResultWriter:
    from spatial_ingestion.refinement import load_mesh_file

    writer = ResultWriter(EXP_ID, results_dir)
    scenes = iter_scenes(SceneSet.from_manifest(manifest), scene_names)
    if quick:
        scenes, repeats = scenes[:1], 2
    for scene in scenes:
        output_dir = outputs / scene.name
        raw_path = output_dir / f"{scene.name}.glb"
        if not raw_path.exists():
            logger.warning("%s: no saved mesh at %s; skipping", scene.name, raw_path)
            continue
        raw_poly = load_mesh_file(raw_path)
        _accuracy_rows(writer, scene, raw_poly, output_dir, seed)
        _timing_rows(writer, scene, raw_path, raw_poly, repeats, split_budget_s)
    return writer


def main(argv: list[str] | None = None) -> int:
    parser = experiment_parser(EXP_ID, __doc__ or "")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--outputs", type=Path, required=True, help="B6's --output-root (holds <scene>/<scene>.glb)"
    )
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--split-budget-s", type=float, default=DEFAULT_SPLIT_BUDGET_S)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    finish(
        run(
            args.results_dir,
            manifest=args.manifest,
            outputs=args.outputs,
            scene_names=args.scenes,
            seed=args.seed,
            repeats=args.repeats,
            split_budget_s=args.split_budget_s,
            quick=args.quick,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
