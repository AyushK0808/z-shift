"""B6 - refinement effect on reconstruction quality. Requires MASt3R and a GPU.

Nobody has checked whether refinement helps or hurts against ground truth. It
runs B2's scenes through `clean_mesh` and scores raw vs refined.

Report the sign honestly. Smoothing an already-accurate surface usually
*increases* Chamfer slightly while improving normal consistency and visual
quality; A3 shows exactly that on synthetic data. Reporting the trade-off is
more valuable than pretending refinement is free improvement, and it supports
SV-B's framing about downstream asset usability rather than geometric accuracy.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from bench.csvio import ResultWriter
from bench.harness import experiment_parser, finish
from bench.metrics import normal_consistency, sample_points
from bench.tier_b_common import (
    SceneSet,
    build_job,
    iter_scenes,
    load_gt_object,
    load_gt_points,
    reconstruction_scale_for,
    require_cuda_device,
    reset_gpu_peak,
    run_reconstruction,
    score_against_gt,
)
from spatial_ingestion.reconstruction.models import Mast3rRunParams
from spatial_ingestion.refinement import (
    MeshCleaningConfig,
    clean_mesh,
    load_mesh_file,
    to_trimesh,
)
from spatial_ingestion.refinement.core import Mode

EXP_ID = "b6_refinement_effect"
SMOOTHING_ITERS: tuple[int, ...] = (0, 15)
MODES: tuple[Mode, ...] = ("object", "room")
SAMPLE_POINTS = 200_000

logger = logging.getLogger(__name__)


def _mesh_points(mesh: trimesh.Trimesh, seed: int) -> np.ndarray:
    if len(mesh.faces) == 0:
        return np.asarray(mesh.vertices, dtype=float)
    return sample_points(mesh, SAMPLE_POINTS, seed)


def run(
    results_dir: Path | None = None,
    *,
    manifest: Path,
    scene_names: list[str] | None = None,
    seed: int = 0,
    image_size: int = 512,
    tsdf_thresh: float = 0.2,
    n_images: int = 10,
    with_scale: bool = False,
    output_root: Path | None = None,
) -> ResultWriter:
    writer = ResultWriter(EXP_ID, results_dir)
    scenes = iter_scenes(SceneSet.from_manifest(manifest), scene_names)
    root = output_root or (Path("data") / "bench" / EXP_ID)

    for scene in scenes:
        if scene.gt_path is None:
            logger.warning("scene %s has no ground truth; skipping", scene.name)
            continue
        gt_points = load_gt_points(scene.gt_path, seed=seed)
        gt_mesh = load_gt_object(scene.gt_path)
        gt_is_mesh = isinstance(gt_mesh, trimesh.Trimesh)

        image_paths = scene.image_paths(n_images)
        reconstruction_scale = reconstruction_scale_for(scene.units)
        params = Mast3rRunParams(
            image_size=image_size, tsdf_thresh=tsdf_thresh, seed=seed, device=require_cuda_device()
        )
        output_path = root / scene.name / f"{scene.name}.glb"
        job = build_job(image_paths, output_path, params=params, label=scene.name)

        reset_gpu_peak()
        run_info = run_reconstruction(job)
        raw_poly = load_mesh_file(run_info["output_path"])
        raw_mesh = to_trimesh(raw_poly)

        base_row: dict[str, Any] = {
            "scene": scene.name,
            "n_images": len(image_paths),
            "image_size": image_size,
            "tsdf_thresh": tsdf_thresh,
            "units": scene.units,
            "sample_points": SAMPLE_POINTS,
        }

        raw_scores = score_against_gt(
            _mesh_points(raw_mesh, seed),
            gt_points,
            tau=scene.tau,
            with_scale=with_scale,
            seed=seed,
            reconstruction_scale=reconstruction_scale,
        )
        writer.add(
            **base_row,
            stage="raw",
            mode="",
            smoothing_iters="",
            refine_seconds=0.0,
            n_vertices=int(len(raw_mesh.vertices)),
            n_faces=int(len(raw_mesh.faces)),
            normal_consistency=(
                round(normal_consistency(raw_mesh, gt_mesh, 50_000, seed), 5)
                if gt_is_mesh and len(raw_mesh.faces)
                else ""
            ),
            **raw_scores,
        )

        for mode in MODES:
            for smoothing in SMOOTHING_ITERS:
                start = time.perf_counter()
                try:
                    result = clean_mesh(
                        raw_poly,
                        MeshCleaningConfig(
                            mode=mode, smoothing_iters=smoothing, verify_watertight=True
                        ),
                    )
                except Exception as exc:  # noqa: BLE001 - a refusal is a result
                    writer.add(
                        **base_row,
                        stage="refined",
                        mode=mode,
                        smoothing_iters=smoothing,
                        status=f"{type(exc).__name__}: {exc}"[:300],
                    )
                    continue
                refine_seconds = round(time.perf_counter() - start, 3)
                refined = to_trimesh(result["mesh"])
                refined_scores = score_against_gt(
                    _mesh_points(refined, seed),
                    gt_points,
                    tau=scene.tau,
                    with_scale=with_scale,
                    seed=seed,
                    reconstruction_scale=reconstruction_scale,
                )
                writer.add(
                    **base_row,
                    stage="refined",
                    mode=mode,
                    smoothing_iters=smoothing,
                    status="ok",
                    refine_seconds=refine_seconds,
                    n_vertices=int(len(refined.vertices)),
                    n_faces=int(len(refined.faces)),
                    is_watertight=result["is_watertight"],
                    open_edge_count=result["open_edge_count"],
                    normal_consistency=(
                        round(normal_consistency(refined, gt_mesh, 50_000, seed), 5)
                        if gt_is_mesh and len(refined.faces)
                        else ""
                    ),
                    chamfer_delta_vs_raw=round(
                        refined_scores["chamfer_l1"] - raw_scores["chamfer_l1"], 6
                    ),
                    f_score_delta_vs_raw=round(
                        refined_scores["f_score"] - raw_scores["f_score"], 4
                    ),
                    **{
                        f"stage_{record['stage']}_s": record["seconds"]
                        for record in result["stage_timings"]
                    },
                    **refined_scores,
                )
    return writer


def main(argv: list[str] | None = None) -> int:
    parser = experiment_parser(EXP_ID, __doc__ or "")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--tsdf-thresh", type=float, default=0.2)
    parser.add_argument("--n-images", type=int, default=10)
    parser.add_argument("--with-scale", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    writer = run(
        results_dir=args.results_dir,
        manifest=args.manifest,
        scene_names=args.scenes,
        seed=args.seed,
        image_size=args.image_size,
        tsdf_thresh=args.tsdf_thresh,
        n_images=args.n_images,
        with_scale=args.with_scale,
        output_root=args.output_root,
    )
    finish(writer)
    for row in writer.rows:
        if row.get("status", "ok") != "ok":
            continue
        print(
            f"  {row['scene']:<12}{row['stage']:<9}{str(row['mode']):<8}"
            f"sm={str(row['smoothing_iters']):<4} F={row.get('f_score', 0):.3f} "
            f"chamfer={row.get('chamfer_l1', 0):.4f} nc={row.get('normal_consistency')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
