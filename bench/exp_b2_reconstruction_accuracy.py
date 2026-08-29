"""B2 - reconstruction accuracy baseline. Requires MASt3R weights and a GPU.

Closes SV-E, which currently concedes that no standardised geometric metrics
are reported. Frame this explicitly as "MASt3R-as-integrated, for reference":
the paper's contribution is the orchestration, not the network, and claiming
otherwise invites a comparison against the MASt3R authors' own numbers that
this system will lose.

Not run on the authoring machine (CPU-only). Point it at a scene manifest:

    uv run python -m bench.exp_b2_reconstruction_accuracy \\
        --manifest bench/scenes/dtu.json --n-images 10 --seed 0

Manifest format (paths relative to the manifest file):

    {"tau": 2.0, "units": "mm",
     "scenes": [{"name": "scan24",
                 "image_dir": "dtu/scan24/images",
                 "gt_path":   "dtu/scan24/gt_points.ply"}]}
"""

from __future__ import annotations

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
    require_cuda_device,
    reset_gpu_peak,
    run_reconstruction,
    score_against_gt,
)
from spatial_ingestion.reconstruction.models import Mast3rRunParams

EXP_ID = "b2_reconstruction_accuracy"
DEFAULT_N_IMAGES = 10

logger = logging.getLogger(__name__)


def run(
    results_dir: Path | None = None,
    *,
    manifest: Path,
    scene_names: list[str] | None = None,
    n_images: int = DEFAULT_N_IMAGES,
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

    for index, scene in enumerate(scenes):
        if scene.gt_path is None:
            logger.warning("scene %s has no ground truth; skipping", scene.name)
            continue

        image_paths = scene.image_paths(n_images)
        params = Mast3rRunParams(
            image_size=image_size, tsdf_thresh=tsdf_thresh, seed=seed, device=require_cuda_device()
        )
        output_path = root / scene.name / f"{scene.name}.glb"
        job = build_job(image_paths, output_path, params=params, label=scene.name)

        reset_gpu_peak()
        run_info = run_reconstruction(job)

        # The protocol's rule: the first MASt3R run of a session pays for the
        # checkpoint download and cache warm, so its timing is discarded.
        if discard_first and index == 0 and len(scenes) > 1:
            logger.info("discarding warm-up run for %s", scene.name)
            reset_gpu_peak()
            run_info = run_reconstruction(job)

        outputs = run_info["manifest"].get("outputs", {})
        stage_seconds = {
            f"stage_{record['stage']}_s": record["seconds"]
            for record in run_info["manifest"].get("stage_timings", [])
        }

        gt_points = load_gt_points(scene.gt_path, seed=seed)
        reconstruction = point_cloud_from_output(run_info["output_dir"], seed=seed)
        scores = score_against_gt(
            reconstruction, gt_points, tau=scene.tau, with_scale=with_scale, seed=seed
        )

        writer.add(
            scene=scene.name,
            units=scene.units,
            n_images=len(image_paths),
            image_size=image_size,
            tsdf_thresh=tsdf_thresh,
            pairing_strategy=params.pairing_strategy,
            model_name=params.model_name,
            device=run_info["manifest"].get("device"),
            tsdf_fallback=run_info["manifest"].get("tsdf_fallback"),
            wall_seconds=run_info["wall_seconds"],
            peak_rss_mb=run_info["peak_rss_mb"],
            gpu_peak_mb=run_info["gpu_peak_mb"],
            **{f"out_{key}": value for key, value in outputs.items()},
            **stage_seconds,
            **scores,
        )
        logger.info("%s: F=%.3f chamfer=%.4f", scene.name, scores["f_score"], scores["chamfer_l1"])
    return writer


def main(argv: list[str] | None = None) -> int:
    parser = experiment_parser(EXP_ID, __doc__ or "")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--n-images", type=int, default=DEFAULT_N_IMAGES)
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
        n_images=args.n_images,
        seed=args.seed,
        image_size=args.image_size,
        tsdf_thresh=args.tsdf_thresh,
        with_scale=args.with_scale,
        output_root=args.output_root,
    )
    finish(writer)
    for row in writer.rows:
        print(
            f"  {row['scene']:<12} F={row['f_score']:.3f} (tau={row['tau']}) "
            f"chamfer={row['chamfer_l1']:.4f} acc={row['accuracy_mean']:.4f} "
            f"comp={row['completeness_mean']:.4f} align_rmse={row['align_rmse']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
