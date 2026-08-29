"""B7 - determinism bound. Requires MASt3R; run on both CPU and GPU.

SV-D claims run manifests support "reproducible re-execution of a given job".
That is an untested assertion. This turns it into a measured bound: run the
same job twice with the same seed and report how far the two outputs diverge.

The load-bearing detail: `run_sparse_alignment` reuses
`<output_dir>/cache`, so a naive repeat run replays the first alignment and the
test is vacuous. `tier_b_common.clear_alignment_cache` is called between runs,
and `cache_cleared` is recorded on every row so a reader can tell a real
measurement from a cached one.

Also sweeps `torch.use_deterministic_algorithms(True, warn_only=True)` on and
off, since `set_seed` now requests it and the cost of that request is worth
knowing.

Two more failure modes a naive version of this test has, both fixed here:

- The first MASt3R run of a process pays for the checkpoint download and
  CUDA/cuDNN kernel warm-up (the same cost B2's `discard_first` exists to
  absorb), and that warm-up can itself pick a different, non-deterministic
  kernel than every later call. Without a warm-up run, that cold-start cost
  lands entirely on row 0 of the first (scene, mode) pair and is
  indistinguishable from a genuine determinism failure. `discard_first` here
  runs one throwaway job before the measured sweep so the cost is paid once,
  off the books.
- Comparing two point clouds index-by-index (`points_a - points_b`) assumes
  run 1's point *i* is run 2's point *i*. It isn't: `point_cloud_from_output`
  subsamples with a fixed seed, but indexes into two independently-written
  PLYs, so any run-to-run change in vertex *write order* turns into a
  spurious multi-unit "displacement" even when the two clouds are
  geometrically identical. `hausdorff95_between_runs` uses nearest-neighbour
  correspondence instead (`bench.metrics.hausdorff_95`), so it measures real
  divergence and, being a 95th percentile, is not dominated by one reordered
  or duplicated point.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from bench.csvio import ResultWriter
from bench.harness import experiment_parser, finish
from bench.metrics import chamfer_l1, hausdorff_95
from bench.tier_b_common import (
    SceneSet,
    build_job,
    clear_alignment_cache,
    iter_scenes,
    point_cloud_from_output,
    reset_gpu_peak,
    run_reconstruction,
)
from spatial_ingestion.reconstruction.models import Mast3rRunParams

EXP_ID = "b7_determinism"
REPEATS = 2

logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_determinism(enabled: bool) -> bool:
    try:
        import torch

        torch.use_deterministic_algorithms(enabled, warn_only=True)
        return True
    except (ImportError, RuntimeError, AttributeError) as exc:
        logger.warning("could not set deterministic algorithms=%s: %s", enabled, exc)
        return False


def run(
    results_dir: Path | None = None,
    *,
    manifest: Path,
    scene_names: list[str] | None = None,
    seed: int = 0,
    image_size: int = 512,
    tsdf_thresh: float = 0.2,
    n_images: int = 8,
    device: str = "cuda",
    deterministic_modes: tuple[bool, ...] = (True, False),
    clear_cache: bool = True,
    output_root: Path | None = None,
    discard_first: bool = True,
) -> ResultWriter:
    writer = ResultWriter(EXP_ID, results_dir)
    scenes = iter_scenes(SceneSet.from_manifest(manifest), scene_names)
    root = output_root or (Path("data") / "bench" / EXP_ID)

    if discard_first and scenes and len(scenes) * len(deterministic_modes) > 1:
        warmup_scene = scenes[0]
        warmup_mode = deterministic_modes[0]
        logger.info("discarding warm-up run for %s", warmup_scene.name)
        _set_determinism(warmup_mode)
        warmup_params = Mast3rRunParams(
            image_size=image_size,
            tsdf_thresh=tsdf_thresh,
            seed=seed,
            device=device,
            deterministic=warmup_mode,
        )
        warmup_path = root / warmup_scene.name / "_warmup" / "mesh.glb"
        warmup_job = build_job(
            warmup_scene.image_paths(n_images),
            warmup_path,
            params=warmup_params,
            label=f"{warmup_scene.name}_warmup",
        )
        if clear_cache:
            clear_alignment_cache(warmup_path.parent)
        reset_gpu_peak()
        run_reconstruction(warmup_job, clear_cache=clear_cache)

    for scene in scenes:
        image_paths = scene.image_paths(n_images)
        for deterministic in deterministic_modes:
            applied = _set_determinism(deterministic)
            runs: list[dict[str, Any]] = []

            for repeat in range(REPEATS):
                params = Mast3rRunParams(
                    image_size=image_size,
                    tsdf_thresh=tsdf_thresh,
                    seed=seed,
                    device=device,
                    deterministic=deterministic,
                )
                output_path = (
                    root / scene.name / f"det{int(deterministic)}" / f"run{repeat}" / "mesh.glb"
                )
                job = build_job(
                    image_paths,
                    output_path,
                    params=params,
                    label=f"{scene.name}_det{int(deterministic)}_{repeat}",
                )
                if clear_cache:
                    clear_alignment_cache(output_path.parent)
                reset_gpu_peak()
                run_info = run_reconstruction(job, clear_cache=clear_cache)
                runs.append(
                    {
                        "points": point_cloud_from_output(
                            run_info["output_dir"], max_points=300_000, seed=seed
                        ),
                        "mesh_sha256": _sha256(run_info["output_path"]),
                        "mesh_bytes": run_info["output_path"].stat().st_size,
                        "wall_seconds": run_info["wall_seconds"],
                        "manifest": run_info["manifest"],
                    }
                )

            first, second = runs[0], runs[1]
            first_points, second_points = first["points"], second["points"]
            same_count = len(first_points) == len(second_points)

            writer.add(
                scene=scene.name,
                n_images=len(image_paths),
                device=first["manifest"].get("device"),
                deterministic_requested=deterministic,
                deterministic_applied=applied,
                deterministic_recorded=first["manifest"]
                .get("reproducibility", {})
                .get("deterministic_algorithms"),
                cache_cleared=clear_cache,
                seed=seed,
                image_size=image_size,
                tsdf_thresh=tsdf_thresh,
                n_points_run1=len(first_points),
                n_points_run2=len(second_points),
                point_count_delta=len(first_points) - len(second_points),
                identical_point_count=same_count,
                hausdorff95_between_runs=round(hausdorff_95(first_points, second_points), 9),
                chamfer_between_runs=round(chamfer_l1(first_points, second_points), 9),
                glb_bytes_run1=first["mesh_bytes"],
                glb_bytes_run2=second["mesh_bytes"],
                glb_byte_identical=first["mesh_sha256"] == second["mesh_sha256"],
                wall_seconds_run1=first["wall_seconds"],
                wall_seconds_run2=second["wall_seconds"],
            )
    return writer


def main(argv: list[str] | None = None) -> int:
    parser = experiment_parser(EXP_ID, __doc__ or "")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--tsdf-thresh", type=float, default=0.2)
    parser.add_argument("--n-images", type=int, default=8)
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["auto", "cpu", "cuda", "mps"],
        help="defaults to cuda, since B1-B6/B8 are CUDA-only; override to cpu/mps "
        "for this experiment's documented cross-device determinism comparison",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="do NOT clear the alignment cache between runs; use only to "
        "demonstrate that the cached comparison is vacuous",
    )
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
        device=args.device,
        clear_cache=not args.keep_cache,
        output_root=args.output_root,
    )
    finish(writer)
    for row in writer.rows:
        print(
            f"  {row['scene']:<12}{row['device']:<6}det={row['deterministic_requested']!s:<6}"
            f"chamfer={row['chamfer_between_runs']:.3e} "
            f"hausdorff95={row['hausdorff95_between_runs']:.3e} "
            f"byte_identical={row['glb_byte_identical']} "
            f"cache_cleared={row['cache_cleared']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
