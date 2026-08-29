"""B8 - EXIF intrinsics initialisation. Requires MASt3R and a GPU.

`alignment.run_sparse_alignment` builds a per-image K from
`focal_length_35mm` when EXIF provides it and passes it to
`sparse_global_alignment` as `init`. Nothing measures whether that helps.

Runs each scene with EXIF-derived `init` on and off (the "off" arm drops the
init dict entirely) and reports the delta in reconstruction metrics and in
alignment time. If it is negligible, say so and simplify the SIV-C claim --
a null result here costs nothing and removes an unsupported sentence.

Only scenes whose images actually carry a usable focal length are eligible;
`eligible_frames` records how many did, so a null result cannot be confused
with "the prior was never populated".
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
    reconstruction_scale_for,
    require_cuda_device,
    reset_gpu_peak,
    run_reconstruction,
    score_against_gt,
)
from spatial_ingestion.batch_normalization.exif import ExifExtractor
from spatial_ingestion.metadata.schema import CameraIntrinsics
from spatial_ingestion.reconstruction.models import HandoffFrame, Mast3rRunParams

EXP_ID = "b8_exif_intrinsics"

logger = logging.getLogger(__name__)


def _frames_with_intrinsics(image_paths: list[Path]) -> tuple[list[HandoffFrame], int]:
    """Build handoff frames carrying EXIF intrinsics; also count usable ones."""
    extractor = ExifExtractor()
    frames: list[HandoffFrame] = []
    eligible = 0
    for index, path in enumerate(image_paths):
        # ExifExtractor already swallows unreadable EXIF and returns an empty
        # CameraIntrinsics, so a missing tag is data, not an error path.
        intrinsics: CameraIntrinsics = extractor.extract(path)
        if intrinsics.focal_length_35mm is not None:
            eligible += 1
        frames.append(
            HandoffFrame(
                frame_id=f"frame_{index:06d}",
                uri=path.resolve().as_uri(),
                index=index,
                camera_intrinsics=intrinsics,
            )
        )
    return frames, eligible


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
        image_paths = scene.image_paths(n_images)
        frames, eligible = _frames_with_intrinsics(image_paths)
        if eligible == 0:
            logger.warning(
                "scene %s: no image carries focal_length_35mm; both arms are "
                "identical and the comparison is uninformative",
                scene.name,
            )
        gt_points = load_gt_points(scene.gt_path, seed=seed) if scene.gt_path else None

        for exif_init in (True, False):
            params = Mast3rRunParams(
                image_size=image_size,
                tsdf_thresh=tsdf_thresh,
                seed=seed,
                device=require_cuda_device(),
            )
            output_path = root / scene.name / f"exif{int(exif_init)}" / "mesh.glb"
            job = build_job(
                image_paths,
                output_path,
                params=params,
                label=f"{scene.name}_exif{int(exif_init)}",
            )
            # `run_sparse_alignment` only builds the init dict when job.frames
            # carry intrinsics, so the "off" arm simply withholds them.
            job.frames = frames if exif_init else []

            reset_gpu_peak()
            run_info = run_reconstruction(job)
            outputs = run_info["manifest"].get("outputs", {})
            stage_seconds = {
                f"stage_{record['stage']}_s": record["seconds"]
                for record in run_info["manifest"].get("stage_timings", [])
            }

            row: dict[str, Any] = {
                "scene": scene.name,
                "exif_init": exif_init,
                "n_images": len(image_paths),
                "eligible_frames": eligible,
                "eligible_fraction": round(eligible / max(len(image_paths), 1), 3),
                "image_size": image_size,
                "tsdf_thresh": tsdf_thresh,
                "wall_seconds": run_info["wall_seconds"],
                "peak_rss_mb": run_info["peak_rss_mb"],
                "gpu_peak_mb": run_info["gpu_peak_mb"],
                **{f"out_{key}": value for key, value in outputs.items()},
                **stage_seconds,
            }
            if gt_points is not None:
                reconstruction = point_cloud_from_output(run_info["output_dir"], seed=seed)
                row.update(
                    score_against_gt(
                        reconstruction,
                        gt_points,
                        tau=scene.tau,
                        with_scale=with_scale,
                        seed=seed,
                        reconstruction_scale=reconstruction_scale_for(scene.units),
                    )
                )
            writer.add(**row)
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
        print(
            f"  {row['scene']:<12}exif={row['exif_init']!s:<6}"
            f"eligible={row['eligible_frames']}/{row['n_images']} "
            f"F={row.get('f_score', float('nan')):.3f} "
            f"align={row.get('stage_sparse_alignment_s', 0):.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
