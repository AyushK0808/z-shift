"""B12 - multi-camera synchronisation on the real DTU rig images. CPU only.

Answers the reviewer's request to validate synchronisation on real multi-camera
data rather than a synthetic video (A6). The data are DTU's scans -- scan24, 37,
40, 55 and 63 -- each a real 49-camera capture of a real object.

What is real and what is injected, which the paper has to say plainly:

- real: the images, and the motion signal. Each scan's 49 views, taken in
  capture order at 25 fps, are scored by the production sampler's own
  `_motion_score` on consecutive real frames, so the signal has a real scene's
  structure (the arm's uneven motion, the object's texture) instead of A6's
  constructed pan windows.
- real: the per-camera disagreement. A second camera sees the same event with
  its own gain, offset and sensor noise, so its motion scores are computed from
  perturbed pixels, not copied with added noise as in A6.
- injected: the clocks. DTU's cameras have no timestamps, so the second
  camera's clock gets a known offset and jitter, and recovery is scored against
  that. This is a validation of the synchroniser on a real scene's signal, not
  a phone-rig experiment.

    uv run python -m bench.exp_b12_dtu_sync --manifest ~/zshift/dtu/manifest.json
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bench.csvio import ResultWriter
from bench.exp_a6_sync_offset import _median_interval_ms, _signal_variance
from bench.harness import experiment_parser, finish
from bench.tier_b_common import SceneSet, iter_scenes
from spatial_ingestion.batch_normalization.video_sampler import MotionAdaptiveFrameSampler
from spatial_ingestion.metadata.schema import FrameReference
from spatial_ingestion.sync.multi_source import (
    MIN_MOTION_VARIANCE,
    MOTION_MATCH_TOLERANCE,
    OFFSET_BUCKET_MS,
    MultiSourceSyncer,
)

EXP_ID = "b12_dtu_sync"

FPS = 25
TRUE_OFFSETS_MS: tuple[float, ...] = (
    0.0,
    40.0,
    -40.0,
    80.0,
    -80.0,
    160.0,
    -160.0,
    320.0,
    -320.0,
    640.0,
    -640.0,
)
JITTERS_MS: tuple[float, ...] = (0.0, 20.0)
SOURCE_COUNTS: tuple[int, ...] = (2, 3)
TOLERANCE_MS = 120.0
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
# Working resolution for the perturbation; _motion_score downsizes further.
_FRAME_SIZE = (640, 360)

logger = logging.getLogger(__name__)


def _load_gray(paths: list[Path]) -> list[np.ndarray]:
    frames = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"cannot read {path}")
        frames.append(cv2.resize(image, _FRAME_SIZE, interpolation=cv2.INTER_AREA))
    return frames


def _scores(frames: list[np.ndarray]) -> list[float]:
    """The shipped sampler's motion score on each consecutive pair of real frames."""
    score = MotionAdaptiveFrameSampler._motion_score
    return [score(frames[i - 1] if i else None, frame) for i, frame in enumerate(frames)]


def _another_camera(frames: list[np.ndarray], rng: np.random.Generator) -> list[np.ndarray]:
    """What a second camera of the rig records: its own gain, offset, blur and sensor noise."""
    gain, bias = rng.uniform(0.85, 1.15), rng.uniform(-8.0, 8.0)
    out = []
    for frame in frames:
        view = frame.astype(np.float32) * gain + bias
        view = cv2.GaussianBlur(view, (3, 3), 0) + rng.normal(0.0, 4.0, view.shape)
        out.append(np.clip(view, 0, 255).astype(np.uint8))
    return out


def _stream(
    scores: list[float],
    source_id: str,
    offset_ms: float,
    jitter_ms: float,
    rng: np.random.Generator,
) -> list[FrameReference]:
    interval = 1000.0 / FPS
    return [
        FrameReference(
            frame_id=f"{source_id}_{index:06d}",
            uri=f"file:///dtu/{source_id}#frame={index}",
            index=index,
            timestamp_ms=index * interval
            + offset_ms
            + (rng.normal(0.0, jitter_ms) if jitter_ms else 0.0),
            source_id=source_id,
            motion_score=float(score),
            resolution=_FRAME_SIZE,
        )
        for index, score in enumerate(scores)
    ]


def _trial(
    anchor: list[FrameReference],
    cameras: list[list[float]],
    *,
    true_offset_ms: float,
    jitter_ms: float,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    streams = {"cam_0": anchor}
    for index, scores in enumerate(cameras, start=1):
        streams[f"cam_{index}"] = _stream(scores, f"cam_{index}", true_offset_ms, jitter_ms, rng)
    entries = MultiSourceSyncer().build_sync_map(
        streams, sync_group_id="b12", tolerance_ms=TOLERANCE_MS
    )
    estimates = [
        -value
        for entry in entries
        for source, value in entry.offsets_ms.items()
        if source != "cam_0"
    ]
    recovered = float(np.mean(estimates)) if estimates else float("nan")
    error = abs(recovered - true_offset_ms) if estimates else float("nan")
    interval = _median_interval_ms(anchor)
    if math.isnan(recovered):
        sign_correct = False
    elif true_offset_ms == 0.0:
        sign_correct = abs(recovered) < interval
    else:
        sign_correct = (recovered > 0) == (true_offset_ms > 0)
    return {
        "true_offset_ms": true_offset_ms,
        "jitter_ms": jitter_ms,
        "n_sources": len(streams),
        "seed": seed,
        "n_frames": len(anchor),
        "median_sample_interval_ms": round(interval, 3),
        "anchor_motion_variance": round(_signal_variance(anchor), 6),
        "motion_variance_guard_fires": _signal_variance(anchor) < MIN_MOTION_VARIANCE,
        "n_entries": len(entries),
        "group_formation_rate": round(len(entries) / max(len(anchor), 1), 4),
        "recovered_offset_ms": round(recovered, 3) if estimates else float("nan"),
        "offset_abs_error_ms": round(error, 3) if estimates else float("nan"),
        "sign_correct": sign_correct,
        "error_exceeds_frame_interval": (
            bool(error > interval) if estimates and not math.isnan(interval) else True
        ),
        "tolerance_ms": TOLERANCE_MS,
        "motion_match_tolerance": MOTION_MATCH_TOLERANCE,
        "offset_bucket_ms": OFFSET_BUCKET_MS,
        "min_motion_variance": MIN_MOTION_VARIANCE,
    }


def run(
    results_dir: Path | None = None,
    *,
    manifest: Path,
    scene_names: list[str] | None = None,
    quick: bool = False,
) -> ResultWriter:
    writer = ResultWriter(EXP_ID, results_dir)
    scenes = iter_scenes(SceneSet.from_manifest(manifest), scene_names)
    offsets = TRUE_OFFSETS_MS[:3] if quick else TRUE_OFFSETS_MS
    jitters = JITTERS_MS[:1] if quick else JITTERS_MS
    seeds = SEEDS[:1] if quick else SEEDS
    if quick:
        scenes = scenes[:1]

    for scene in scenes:
        frames = _load_gray(scene.image_paths())
        anchor_scores = _scores(frames)
        anchor = _stream(anchor_scores, "cam_0", 0.0, 0.0, np.random.default_rng(0))
        logger.info(
            "%s: %d real frames, motion variance %.4f",
            scene.name,
            len(frames),
            _signal_variance(anchor),
        )
        for seed in seeds:
            # Other cameras' scores depend on the seed (their gain, bias, noise)
            # but not on the clock, so they are computed once per seed.
            rng = np.random.default_rng(1000 + seed)
            cameras = [_scores(_another_camera(frames, rng)) for _ in range(max(SOURCE_COUNTS) - 1)]
            for n_sources in SOURCE_COUNTS:
                for offset in offsets:
                    for jitter in jitters:
                        writer.add(
                            scene=scene.name,
                            data_source="dtu_real_images_injected_clock",
                            **_trial(
                                anchor,
                                cameras[: n_sources - 1],
                                true_offset_ms=offset,
                                jitter_ms=jitter,
                                seed=seed,
                            ),
                        )
        logger.info("%s done (%d rows)", scene.name, len(writer))
    return writer


def main(argv: list[str] | None = None) -> int:
    parser = experiment_parser(EXP_ID, __doc__ or "")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    finish(run(args.results_dir, manifest=args.manifest, scene_names=args.scenes, quick=args.quick))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
