"""B4 - frame-budget ablation. The headline result. Requires MASt3R and a GPU.

Closes the loop on A5, which showed on CPU that `_cap_frames` hands pairing a
motion-rank-ordered list and that swin pairing (the default for video) then
builds pairs between temporally distant frames. A5 measured pair adjacency;
this measures what that does to the reconstruction.

    V1 motion-cap, as currently implemented (motion-rank order into pairing)
    V2 motion-cap + temporal re-sort before pairing
    V3 uniform subsample, temporal order

All three are matched at the same frame count, so this compares selection
strategy rather than budget. Every cell runs under three seeds: MASt3R's TSDF
depth refinement samples with torch.randn, so the seed is a genuine source of
run-to-run variation, and one run per cell cannot separate a variant effect
from it.

Either outcome is publishable, and the paper should say which happened:
V2 > V1 means a real integration bug was found, fixed and quantified. V1 ~ V2
means swin pairing is more robust to ordering than expected, which is itself a
finding no component-level benchmark would surface.

A second sweep varies MAX_RECONSTRUCTION_FRAMES over {10, 20, 40, 60} on one
scene, which is what turns "bounds reconstruction cost" into a curve. Its
budget-40 point is the ablation's V3 run, reused rather than reconstructed
again; a budget above the frame count takes every frame (`n_selected`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from bench.csvio import ResultWriter
from bench.dtu_eval import DtuReference
from bench.exp_a5_frame_budget import VARIANTS, select_variant
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
)
from spatial_ingestion.metadata.schema import FrameReference
from spatial_ingestion.reconstruction.models import (
    Mast3rRunParams,
    ReconstructionMode,
)

EXP_ID = "b4_frame_budget_ablation"
BUDGET_SWEEP: tuple[int, ...] = (10, 20, 40, 60)
DEFAULT_BUDGET = 40
N_SEEDS = 3

logger = logging.getLogger(__name__)


def _frames_from_image_dir(image_paths: list[Path]) -> list[FrameReference]:
    """Build FrameReferences for an extracted video sequence.

    Motion scores come from consecutive-frame difference, matching
    `MotionAdaptiveFrameSampler._motion_score`, so `_cap_frames` sees the same
    ranking signal it would in production.
    """
    import cv2

    logger.info("computing motion scores for %d frames", len(image_paths))
    frames: list[FrameReference] = []
    previous: Any = None
    for index, path in enumerate(image_paths):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"could not read {path}")
        small = cv2.resize(image, (160, 90))
        if previous is None:
            motion = 1.0
        else:
            diff = cv2.absdiff(previous, small)
            motion = round(
                min(1.0, float(np.mean(diff)) / 255.0 * 1.8 + float(np.mean(diff > 18)) * 0.8),
                4,
            )
        previous = small
        frames.append(
            FrameReference(
                frame_id=f"frame_{index:06d}",
                uri=path.resolve().as_uri(),
                index=index,
                timestamp_ms=index * (1000.0 / 24.0),
                source_id="cam_a",
                motion_score=motion,
            )
        )
    return frames


def _run_variant(
    scene_name: str,
    variant: str,
    frames: list[FrameReference],
    budget: int,
    *,
    params: Mast3rRunParams,
    output_root: Path,
    gt_points: np.ndarray,
    tau: float,
    seed: int,
    with_scale: bool,
    reconstruction_scale: float,
    reference: DtuReference | None = None,
) -> dict[str, Any]:
    from spatial_ingestion.reconstruction._io import uri_to_path

    selected = select_variant(variant, frames, budget=budget)
    image_paths = [uri_to_path(frame.uri) for frame in selected]
    logger.info(
        "%s/%s: running reconstruction (budget=%d, %d frames selected)",
        scene_name,
        variant,
        budget,
        len(selected),
    )

    output_path = output_root / scene_name / variant / f"{variant}.glb"
    job = build_job(
        image_paths,
        output_path,
        params=params,
        mode=ReconstructionMode.VIDEO_SEQUENCE,
        label=f"{scene_name}_{variant}",
    )
    job.frames = selected

    reset_gpu_peak()
    run_info = run_reconstruction(job)
    outputs = run_info["manifest"].get("outputs", {})
    stage_seconds = {
        f"stage_{record['stage']}_s": record["seconds"]
        for record in run_info["manifest"].get("stage_timings", [])
    }

    capture_index = np.array([frame.index for frame in selected], dtype=float)
    reconstruction = point_cloud_from_output(run_info["output_dir"], seed=seed)
    scores = score_against_gt(
        reconstruction,
        gt_points,
        tau=tau,
        with_scale=with_scale,
        seed=seed,
        reconstruction_scale=reconstruction_scale,
        reference=reference,
        output_dir=run_info["output_dir"],
    )
    logger.info(
        "%s/%s (budget=%d): F=%.3f chamfer=%.4f",
        scene_name,
        variant,
        budget,
        scores["f_score"],
        scores["chamfer_l1"],
    )

    return {
        "scene": scene_name,
        "variant": variant,
        "budget": budget,
        "n_selected": len(selected),
        "in_capture_order": bool(np.all(np.diff(capture_index) > 0)),
        "wall_seconds": run_info["wall_seconds"],
        "peak_rss_mb": run_info["peak_rss_mb"],
        "gpu_peak_mb": run_info["gpu_peak_mb"],
        "tsdf_fallback": run_info["manifest"].get("tsdf_fallback"),
        **{f"out_{key}": value for key, value in outputs.items()},
        **stage_seconds,
        **scores,
    }


def run(
    results_dir: Path | None = None,
    *,
    manifest: Path,
    scene_names: list[str] | None = None,
    seeds: tuple[int, ...] = tuple(range(N_SEEDS)),
    image_size: int = 512,
    tsdf_thresh: float = 0.2,
    with_scale: bool = False,
    output_root: Path | None = None,
    sweep_budgets: bool = True,
) -> ResultWriter:
    writer = ResultWriter(EXP_ID, results_dir)
    scenes = iter_scenes(SceneSet.from_manifest(manifest), scene_names)
    root = output_root or (Path("data") / "bench" / EXP_ID)
    device = require_cuda_device()

    def params_for(seed: int) -> Mast3rRunParams:
        return Mast3rRunParams(
            image_size=image_size,
            tsdf_thresh=tsdf_thresh,
            seed=seed,
            pairing_strategy="swin",
            device=device,
        )

    for scene_index, scene in enumerate(scenes):
        if scene.gt_path is None:
            logger.warning("scene %s has no ground truth; skipping", scene.name)
            continue
        image_paths = scene.image_paths()
        if len(image_paths) <= DEFAULT_BUDGET:
            logger.warning(
                "scene %s has only %d frames; the budget never fires and the "
                "variants are identical",
                scene.name,
                len(image_paths),
            )
        frames = _frames_from_image_dir(image_paths)
        gt_points = load_gt_points(scene.gt_path, seed=seeds[0])
        common = {
            "gt_points": gt_points,
            "tau": scene.tau,
            "with_scale": with_scale,
            "reconstruction_scale": reconstruction_scale_for(scene.units),
            "reference": scene.dtu_reference(),
        }
        sweep_here = sweep_budgets and scene_index == 0

        for seed in seeds:
            params = params_for(seed)
            seed_root = root / f"seed{seed}"
            for variant in VARIANTS:
                result = _run_variant(
                    scene.name,
                    variant,
                    frames,
                    DEFAULT_BUDGET,
                    params=params,
                    output_root=seed_root,
                    seed=seed,
                    **common,
                )
                writer.add(
                    part="variant_ablation",
                    n_frames_available=len(frames),
                    pairing_strategy=params.pairing_strategy,
                    **result,
                )
                if sweep_here and variant == "V3_uniform" and DEFAULT_BUDGET in BUDGET_SWEEP:
                    writer.add(
                        part="budget_sweep",
                        n_frames_available=len(frames),
                        pairing_strategy=params.pairing_strategy,
                        reused_from="variant_ablation",
                        **result,
                    )

            # The quality-vs-budget curve only needs one scene to be informative.
            if not sweep_here:
                continue
            for budget in BUDGET_SWEEP:
                if budget == DEFAULT_BUDGET:
                    continue
                writer.add(
                    part="budget_sweep",
                    n_frames_available=len(frames),
                    pairing_strategy=params.pairing_strategy,
                    **_run_variant(
                        scene.name,
                        "V3_uniform",
                        frames,
                        budget,
                        params=params,
                        output_root=seed_root / f"budget_{budget}",
                        seed=seed,
                        **common,
                    ),
                )
    return writer


def main(argv: list[str] | None = None) -> int:
    parser = experiment_parser(EXP_ID, __doc__ or "")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--tsdf-thresh", type=float, default=0.2)
    parser.add_argument("--with-scale", action="store_true")
    parser.add_argument("--no-budget-sweep", action="store_true")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=f"seeds to repeat every cell under (default: --seed and the next {N_SEEDS - 1})",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    writer = run(
        results_dir=args.results_dir,
        manifest=args.manifest,
        scene_names=args.scenes,
        seeds=tuple(args.seeds or range(args.seed, args.seed + N_SEEDS)),
        image_size=args.image_size,
        tsdf_thresh=args.tsdf_thresh,
        with_scale=args.with_scale,
        output_root=args.output_root,
        sweep_budgets=not args.no_budget_sweep,
    )
    finish(writer)
    for row in writer.rows:
        print(
            f"  {row['part']:<17}{row['scene']:<12}{row['variant']:<20}"
            f"budget={row['budget']:<4} n={row['n_selected']:<3} seed={row['seed']} "
            f"F={row['f_score']:.3f} pose_failure={row.get('pose_failure')} "
            f"chamfer={row['chamfer_l1']:.4f} align_s={row.get('stage_sparse_alignment_s')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
