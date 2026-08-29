"""B3 - pairing strategy ablation. Requires MASt3R and a GPU.

`complete` vs `swin` on the same scenes at matched frame counts. The number
that matters is not either strategy's F-score in isolation but the cost/quality
slope: how much F-score is given up per unit of pair-count reduction. That is a
systems result, and it is what justifies SWIN_PAIRING_THRESHOLD = 20 -- a
constant currently chosen without evidence.

Frame count is swept as well, because the two strategies' pair counts diverge
quadratically: `complete` is O(n^2) and `swin` is O(n * winsize), so the slope
is only meaningful as a function of n.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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
from spatial_ingestion.config import SWIN_PAIRING_THRESHOLD
from spatial_ingestion.reconstruction.models import Mast3rRunParams

EXP_ID = "b3_pairing_ablation"
STRATEGIES: tuple[str, ...] = ("complete", "swin")
FRAME_COUNTS: tuple[int, ...] = (8, 12, 20, 30)

logger = logging.getLogger(__name__)


def run(
    results_dir: Path | None = None,
    *,
    manifest: Path,
    scene_names: list[str] | None = None,
    seed: int = 0,
    image_size: int = 512,
    tsdf_thresh: float = 0.2,
    frame_counts: tuple[int, ...] = FRAME_COUNTS,
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

        for n_images in frame_counts:
            available = scene.image_paths()
            if len(available) < n_images:
                logger.info(
                    "scene %s has %d images; skipping the %d-image cell",
                    scene.name,
                    len(available),
                    n_images,
                )
                continue
            image_paths = scene.image_paths(n_images)

            for strategy in STRATEGIES:
                params = Mast3rRunParams(
                    image_size=image_size,
                    tsdf_thresh=tsdf_thresh,
                    seed=seed,
                    pairing_strategy=strategy,
                    device=require_cuda_device(),
                )
                output_path = root / scene.name / f"n{n_images}" / strategy / "mesh.glb"
                job = build_job(
                    image_paths,
                    output_path,
                    params=params,
                    label=f"{scene.name}_{strategy}_{n_images}",
                )

                reset_gpu_peak()
                run_info = run_reconstruction(job)
                outputs = run_info["manifest"].get("outputs", {})
                stage_seconds = {
                    f"stage_{record['stage']}_s": record["seconds"]
                    for record in run_info["manifest"].get("stage_timings", [])
                }

                reconstruction = point_cloud_from_output(run_info["output_dir"], seed=seed)
                scores = score_against_gt(
                    reconstruction,
                    gt_points,
                    tau=scene.tau,
                    with_scale=with_scale,
                    seed=seed,
                )

                row: dict[str, Any] = {
                    "scene": scene.name,
                    "pairing_strategy": strategy,
                    "n_images": n_images,
                    "swin_pairing_threshold": SWIN_PAIRING_THRESHOLD,
                    "above_threshold": n_images > SWIN_PAIRING_THRESHOLD,
                    "image_size": image_size,
                    "tsdf_thresh": tsdf_thresh,
                    "wall_seconds": run_info["wall_seconds"],
                    "peak_rss_mb": run_info["peak_rss_mb"],
                    "gpu_peak_mb": run_info["gpu_peak_mb"],
                    "tsdf_fallback": run_info["manifest"].get("tsdf_fallback"),
                    **{f"out_{key}": value for key, value in outputs.items()},
                    **stage_seconds,
                    **scores,
                }
                writer.add(**row)
                logger.info(
                    "%s n=%d %s: pairs=%s F=%.3f align=%.1fs",
                    scene.name,
                    n_images,
                    strategy,
                    outputs.get("n_pairs"),
                    scores["f_score"],
                    stage_seconds.get("stage_sparse_alignment_s", 0.0),
                )
    return writer


def main(argv: list[str] | None = None) -> int:
    parser = experiment_parser(EXP_ID, __doc__ or "")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--tsdf-thresh", type=float, default=0.2)
    parser.add_argument("--frame-counts", type=int, nargs="*", default=list(FRAME_COUNTS))
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
        frame_counts=tuple(args.frame_counts),
        with_scale=args.with_scale,
        output_root=args.output_root,
    )
    finish(writer)
    for row in writer.rows:
        print(
            f"  {row['scene']:<12}n={row['n_images']:<4}{row['pairing_strategy']:<10}"
            f"pairs={row.get('out_n_pairs', '?'):<6} F={row['f_score']:.3f} "
            f"align={row.get('stage_sparse_alignment_s', 0):.1f}s "
            f"gpu={row.get('gpu_peak_mb')}MB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
