"""B5 - TSDF fusion and fallback characterisation. Requires MASt3R and a GPU.

SV-C/SV-D present the TSDF-to-pointmap fallback as a robustness feature with no
data on when it fires. This measures the trade-off: how often it triggers, at
what memory ceiling, and what the graceful degradation actually costs in
reconstruction quality.

The failure path is forced deliberately rather than waited for. On Linux that
is `resource.setrlimit(RLIMIT_AS, ...)` in a child process; the helper below
refuses on Windows, where there is no equivalent per-process address-space cap,
and says so instead of silently reporting a fallback rate of zero.

Also verifies `_patch_tsdf_cuda_hardcode` on a CPU-only build. That patch
replaces `torch.Tensor.cuda` process-wide, which is a global side effect and
belongs in SVII; the row records whether it was applied.
"""

from __future__ import annotations

import logging
import sys
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
    reset_gpu_peak,
    run_reconstruction,
    score_against_gt,
)
from spatial_ingestion.reconstruction.models import Mast3rRunParams

EXP_ID = "b5_tsdf_fallback"
TSDF_THRESHOLDS: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5)
FRAME_COUNTS: tuple[int, ...] = (8, 16)
MEMORY_CEILINGS_GB: tuple[float, ...] = (2.0, 4.0, 8.0)

logger = logging.getLogger(__name__)


def set_address_space_limit(limit_gb: float) -> bool:
    """Cap this process's address space. Returns False where unsupported.

    Windows has no `RLIMIT_AS` equivalent that applies to the current process,
    so the memory-ceiling sweep is reported as unsupported there rather than
    producing a fallback rate of zero that means "never tested".
    """
    if sys.platform == "win32":
        return False
    try:
        import resource

        limit_bytes = int(limit_gb * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        return True
    except (ImportError, ValueError, OSError) as exc:
        logger.warning("could not set address-space limit: %s", exc)
        return False


def _cuda_patch_state() -> dict[str, Any]:
    from spatial_ingestion.reconstruction import export

    return {
        "tsdf_cuda_hardcode_patched": export._tsdf_cuda_hardcode_patched,
        "torch_tensor_cuda_is_patched": (
            getattr(__import__("torch").Tensor.cuda, "__name__", "") == "_cuda_noop"
        ),
    }


def run(
    results_dir: Path | None = None,
    *,
    manifest: Path,
    scene_names: list[str] | None = None,
    seed: int = 0,
    image_size: int = 512,
    frame_counts: tuple[int, ...] = FRAME_COUNTS,
    thresholds: tuple[float, ...] = TSDF_THRESHOLDS,
    memory_ceiling_gb: float | None = None,
    with_scale: bool = False,
    output_root: Path | None = None,
) -> ResultWriter:
    writer = ResultWriter(EXP_ID, results_dir)
    scenes = iter_scenes(SceneSet.from_manifest(manifest), scene_names)
    root = output_root or (Path("data") / "bench" / EXP_ID)

    ceiling_applied = set_address_space_limit(memory_ceiling_gb) if memory_ceiling_gb else False
    if memory_ceiling_gb and not ceiling_applied:
        logger.warning(
            "memory ceiling of %.1f GB requested but unsupported on %s; "
            "rows are marked memory_ceiling_applied=False",
            memory_ceiling_gb,
            sys.platform,
        )

    for scene in scenes:
        gt_points = load_gt_points(scene.gt_path, seed=seed) if scene.gt_path else None
        for n_images in frame_counts:
            if len(scene.image_paths()) < n_images:
                continue
            image_paths = scene.image_paths(n_images)
            for threshold in thresholds:
                params = Mast3rRunParams(image_size=image_size, tsdf_thresh=threshold, seed=seed)
                output_path = root / scene.name / f"n{n_images}_t{threshold}" / "mesh.glb"
                job = build_job(
                    image_paths,
                    output_path,
                    params=params,
                    label=f"{scene.name}_t{threshold}",
                )

                row: dict[str, Any] = {
                    "scene": scene.name,
                    "n_images": n_images,
                    "tsdf_thresh": threshold,
                    "tsdf_enabled": threshold > 0,
                    "image_size": image_size,
                    "memory_ceiling_gb": memory_ceiling_gb,
                    "memory_ceiling_applied": ceiling_applied,
                    "platform": sys.platform,
                    **_cuda_patch_state(),
                }
                try:
                    reset_gpu_peak()
                    run_info = run_reconstruction(job)
                    outputs = run_info["manifest"].get("outputs", {})
                    row.update(
                        status="ok",
                        tsdf_fallback=run_info["manifest"].get("tsdf_fallback"),
                        wall_seconds=run_info["wall_seconds"],
                        peak_rss_mb=run_info["peak_rss_mb"],
                        gpu_peak_mb=run_info["gpu_peak_mb"],
                        **{f"out_{key}": value for key, value in outputs.items()},
                        **{
                            f"stage_{record['stage']}_s": record["seconds"]
                            for record in run_info["manifest"].get("stage_timings", [])
                        },
                    )
                    if gt_points is not None:
                        reconstruction = point_cloud_from_output(run_info["output_dir"], seed=seed)
                        row.update(
                            score_against_gt(
                                reconstruction,
                                gt_points,
                                tau=scene.tau,
                                with_scale=with_scale,
                                seed=seed,
                            )
                        )
                except (MemoryError, RuntimeError) as exc:
                    # Distinct from the in-pipeline fallback: this is the run
                    # dying outright, which is the outcome the fallback exists
                    # to prevent.
                    row.update(status=f"{type(exc).__name__}: {exc}"[:300], tsdf_fallback=None)
                    logger.warning("%s t=%s failed outright: %s", scene.name, threshold, exc)
                writer.add(**row)
    return writer


def main(argv: list[str] | None = None) -> int:
    parser = experiment_parser(EXP_ID, __doc__ or "")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--frame-counts", type=int, nargs="*", default=list(FRAME_COUNTS))
    parser.add_argument("--thresholds", type=float, nargs="*", default=list(TSDF_THRESHOLDS))
    parser.add_argument(
        "--memory-ceiling-gb",
        type=float,
        default=None,
        help="cap the address space to force the fallback path (Linux/macOS only)",
    )
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
        frame_counts=tuple(args.frame_counts),
        thresholds=tuple(args.thresholds),
        memory_ceiling_gb=args.memory_ceiling_gb,
        with_scale=args.with_scale,
        output_root=args.output_root,
    )
    finish(writer)

    fired = [r for r in writer.rows if r.get("tsdf_fallback")]
    enabled = [r for r in writer.rows if r.get("tsdf_enabled")]
    print(
        f"  fallback fired in {len(fired)}/{len(enabled)} TSDF-enabled runs"
        f" (ceiling={args.memory_ceiling_gb} GB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
