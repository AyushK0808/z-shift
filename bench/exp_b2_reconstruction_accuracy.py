"""B2 - reconstruction accuracy baseline. Requires MASt3R weights and a GPU.

Closes SV-E, which currently concedes that no standardised geometric metrics
are reported. Frame this explicitly as "MASt3R-as-integrated, for reference":
the paper's contribution is the orchestration, not the network, and claiming
otherwise invites a comparison against the MASt3R authors' own numbers that
this system will lose.

Not run on the authoring machine (CPU-only). Point it at a scene manifest:

    uv run python -m bench.exp_b2_reconstruction_accuracy \\
        --manifest bench/scenes/dtu.json --n-images 10 49 --seed 0

Manifest format (paths relative to the manifest file):

    {"tau": 2.0, "units": "mm",
     "scenes": [{"name": "scan24",
                 "image_dir": "dtu/scan24/images",
                 "gt_path":   "dtu/scan24/gt_points.ply"}]}
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

from bench.csvio import ResultWriter
from bench.harness import experiment_parser, finish
from bench.tier_b_common import (
    SceneSet,
    build_job,
    iter_scenes,
    load_gt_points,
    point_cloud_from_output,
    reconstruction_scale_for,
    require_cuda_device,
    reset_gpu_peak,
    run_reconstruction,
    score_against_gt,
    timed,
)
from spatial_ingestion.reconstruction.models import Mast3rRunParams

EXP_ID = "b2_reconstruction_accuracy"
# Ten views and the whole rig. Views are spread over the capture rather than
# taken first-N: DTU numbers them along the arm's path, so its first ten are
# one short arc and score as a narrow-baseline capture.
DEFAULT_VIEW_COUNTS: tuple[int, ...] = (10, 49)

logger = logging.getLogger(__name__)


def _out_of_memory_errors() -> tuple[type[BaseException], ...]:
    try:
        import torch

        return (torch.cuda.OutOfMemoryError, MemoryError)
    except (ImportError, AttributeError):
        return (MemoryError,)


def _release_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def run(
    results_dir: Path | None = None,
    *,
    manifest: Path,
    scene_names: list[str] | None = None,
    view_counts: tuple[int, ...] = DEFAULT_VIEW_COUNTS,
    seed: int = 0,
    image_size: int = 512,
    tsdf_thresh: float = 0.2,
    with_scale: bool = False,
    output_root: Path | None = None,
    discard_first: bool = True,
) -> ResultWriter:
    writer = ResultWriter(EXP_ID, results_dir)
    scenes = iter_scenes(SceneSet.from_manifest(manifest), scene_names)
    root = output_root or (Path("data") / "bench" / EXP_ID)
    params = Mast3rRunParams(
        image_size=image_size, tsdf_thresh=tsdf_thresh, seed=seed, device=require_cuda_device()
    )

    for index, scene in enumerate(scenes):
        if scene.gt_path is None:
            logger.warning("scene %s has no ground truth; skipping", scene.name)
            continue
        gt_points = None

        for count_index, n_images in enumerate(view_counts):
            image_paths = scene.image_paths(n_images, spread=True)
            base_row = {
                "scene": scene.name,
                "units": scene.units,
                "n_images": len(image_paths),
                "view_selection": "spread",
                "image_size": image_size,
                "tsdf_thresh": tsdf_thresh,
                "pairing_strategy": params.pairing_strategy,
                "model_name": params.model_name,
            }
            output_path = root / scene.name / f"views{len(image_paths)}" / f"{scene.name}.glb"
            job = build_job(image_paths, output_path, params=params, label=scene.name)

            try:
                reset_gpu_peak()
                run_info = run_reconstruction(job)
                # The protocol's rule: the first MASt3R run of a session pays
                # for the checkpoint download and cache warm, so its timing is
                # discarded.
                if discard_first and index == 0 and count_index == 0 and len(scenes) > 1:
                    logger.info("discarding warm-up run for %s", scene.name)
                    reset_gpu_peak()
                    run_info = run_reconstruction(job)
            except _out_of_memory_errors() as exc:
                # A view count the GPU cannot hold is a result, and must not
                # cost the scene its other rows.
                logger.error("%s with %d views: %s", scene.name, len(image_paths), exc)
                writer.add(**base_row, status=f"{type(exc).__name__}: {exc}"[:300])
                _release_gpu_memory()
                continue

            outputs = run_info["manifest"].get("outputs", {})
            stage_seconds = {
                f"stage_{record['stage']}_s": record["seconds"]
                for record in run_info["manifest"].get("stage_timings", [])
            }

            if gt_points is None:
                with timed(f"{scene.name}/load_gt_points"):
                    gt_points = load_gt_points(scene.gt_path, seed=seed)
            reconstruction = point_cloud_from_output(run_info["output_dir"], seed=seed)
            with timed(f"{scene.name}/score_against_gt"):
                scores = score_against_gt(
                    reconstruction,
                    gt_points,
                    tau=scene.tau,
                    with_scale=with_scale,
                    seed=seed,
                    reconstruction_scale=reconstruction_scale_for(scene.units),
                    reference=scene.dtu_reference(),
                    output_dir=run_info["output_dir"],
                )

            writer.add(
                **base_row,
                status="ok",
                device=run_info["manifest"].get("device"),
                tsdf_fallback=run_info["manifest"].get("tsdf_fallback"),
                wall_seconds=run_info["wall_seconds"],
                peak_rss_mb=run_info["peak_rss_mb"],
                gpu_peak_mb=run_info["gpu_peak_mb"],
                **{f"out_{key}": value for key, value in outputs.items()},
                **stage_seconds,
                **scores,
            )
            logger.info(
                "%s/%d views: F=%.3f chamfer=%.4f",
                scene.name,
                len(image_paths),
                scores["f_score"],
                scores["chamfer_l1"],
            )
    return writer


def main(argv: list[str] | None = None) -> int:
    parser = experiment_parser(EXP_ID, __doc__ or "")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument(
        "--n-images",
        type=int,
        nargs="+",
        default=list(DEFAULT_VIEW_COUNTS),
        help="view counts to run, each spread evenly over the capture",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--tsdf-thresh", type=float, default=0.2)
    parser.add_argument(
        "--with-scale",
        action="store_true",
        help="fit scale during alignment; off by default because the default "
        "checkpoint is the metric variant and scale is part of what is tested",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    writer = run(
        results_dir=args.results_dir,
        manifest=args.manifest,
        scene_names=args.scenes,
        view_counts=tuple(args.n_images),
        seed=args.seed,
        image_size=args.image_size,
        tsdf_thresh=args.tsdf_thresh,
        with_scale=args.with_scale,
        output_root=args.output_root,
    )
    finish(writer)
    for row in writer.rows:
        if row.get("status") != "ok":
            print(f"  {row['scene']:<12} views={row['n_images']:<3} {row['status']}")
            continue
        print(
            f"  {row['scene']:<12} views={row['n_images']:<3} F={row['f_score']:.3f} "
            f"(tau={row['tau']}) chamfer={row['chamfer_l1']:.4f} "
            f"acc={row['accuracy_mean']:.4f} comp={row['completeness_mean']:.4f} "
            f"align_rmse={row['align_rmse']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
