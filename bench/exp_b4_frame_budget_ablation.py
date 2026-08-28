"""B4 - frame-budget ablation. The headline result. Requires MASt3R and a GPU.

Closes the loop on A5, which showed on CPU that `_cap_frames` hands pairing a
motion-rank-ordered list and that swin pairing (the default for video) then
builds pairs between temporally distant frames. A5 measured pair adjacency;
this measures what that does to the reconstruction.

    V1 motion-cap, as currently implemented (motion-rank order into pairing)
    V2 motion-cap + temporal re-sort before pairing
    V3 uniform subsample, temporal order

All three are matched at the same frame count, so this compares selection
strategy rather than budget.

Either outcome is publishable, and the paper should say which happened:
V2 > V1 means a real integration bug was found, fixed and quantified. V1 ~ V2
means swin pairing is more robust to ordering than expected, which is itself a
finding no component-level benchmark would surface.

A second sweep varies MAX_RECONSTRUCTION_FRAMES over {10, 20, 40, 60} on one
scene, which is what turns "bounds reconstruction cost" into a curve.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from bench.csvio import ResultWriter
from bench.exp_a5_frame_budget import VARIANTS, select_variant
from bench.harness import experiment_parser, finish
from bench.tier_b_common import (
    SceneSet,
    build_job,
    iter_scenes,
    load_gt_points,
    point_cloud_from_output,
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

logger = logging.getLogger(__name__)


def _frames_from_image_dir(image_paths: list[Path]) -> list[FrameReference]:
    """Build FrameReferences for an extracted video sequence.

    Motion scores come from consecutive-frame difference, matching
    `MotionAdaptiveFrameSampler._motion_score`, so `_cap_frames` sees the same
    ranking signal it would in production.
    """
    import cv2

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
) -> dict[str, Any]:
    from spatial_ingestion.reconstruction._io import uri_to_path

    selected = select_variant(variant, frames, budget=budget)
    image_paths = [uri_to_path(frame.uri) for frame in selected]

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
        reconstruction, gt_points, tau=tau, with_scale=with_scale, seed=seed
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
    seed: int = 0,
    image_size: int = 512,
    tsdf_thresh: float = 0.2,
    with_scale: bool = False,
    output_root: Path | None = None,
    sweep_budgets: bool = True,
) -> ResultWriter:
    writer = ResultWriter(EXP_ID, results_dir)
    scenes = iter_scenes(SceneSet.from_manifest(manifest), scene_names)
    root = output_root or (Path("data") / "bench" / EXP_ID)
    params = Mast3rRunParams(
        image_size=image_size, tsdf_thresh=tsdf_thresh, seed=seed, pairing_strategy="swin"
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
        gt_points = load_gt_points(scene.gt_path, seed=seed)

        for variant in VARIANTS:
            writer.add(
                part="variant_ablation",
                n_frames_available=len(frames),
                pairing_strategy=params.pairing_strategy,
                **_run_variant(
                    scene.name,
                    variant,
                    frames,
                    DEFAULT_BUDGET,
                    params=params,
                    output_root=root,
                    gt_points=gt_points,
                    tau=scene.tau,
                    seed=seed,
                    with_scale=with_scale,
                ),
            )

        # The quality-vs-budget curve only needs one scene to be informative.
        if sweep_budgets and scene_index == 0:
            for budget in BUDGET_SWEEP:
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
                        output_root=root / f"budget_{budget}",
                        gt_points=gt_points,
                        tau=scene.tau,
                        seed=seed,
                        with_scale=with_scale,
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
        with_scale=args.with_scale,
        output_root=args.output_root,
        sweep_budgets=not args.no_budget_sweep,
    )
    finish(writer)
    for row in writer.rows:
        print(
            f"  {row['part']:<17}{row['scene']:<12}{row['variant']:<20}"
            f"budget={row['budget']:<4} F={row['f_score']:.3f} "
            f"chamfer={row['chamfer_l1']:.4f} align_s={row.get('stage_sparse_alignment_s')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
