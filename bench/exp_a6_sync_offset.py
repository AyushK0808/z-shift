"""A6 - multi-source synchroniser offset recovery.

Supports the SIV-A / SV-B claim that "synchronized multi-camera inputs can
also be converted into cross-camera view groups using timestamp information",
and puts numbers behind four constants that are currently set by intuition:
MOTION_MATCH_TOLERANCE, OFFSET_BUCKET_MS, MIN_MOTION_VARIANCE and the
tolerance_ms default of 120.

This has genuine ground truth: derive_offset_stream injects a known clock
offset, so recovery error is measurable rather than inferred.

Sign convention: MultiSourceSyncer estimates a *correction* to add to a
source's clock, i.e. offsets_ms[src] ~= -injected_offset. The recovered
estimate reported here negates it back into injected-offset units, so
recovered_offset_ms is directly comparable to true_offset_ms.
"""

from __future__ import annotations

import logging
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from bench.csvio import ResultWriter
from bench.fixtures import derive_offset_stream, frames_from_video, make_motion_video
from bench.harness import experiment_parser, finish
from spatial_ingestion.metadata.schema import FrameReference
from spatial_ingestion.sync.multi_source import (
    MIN_MOTION_VARIANCE,
    MOTION_MATCH_TOLERANCE,
    OFFSET_BUCKET_MS,
    MultiSourceSyncer,
)

EXP_ID = "a6_sync_offset"

TRUE_OFFSETS_MS: tuple[float, ...] = (
    0.0, 40.0, -40.0, 80.0, -80.0, 160.0, -160.0, 320.0, -320.0, 640.0, -640.0,
)
JITTERS_MS: tuple[float, ...] = (0.0, 5.0, 20.0)
SOURCE_COUNTS: tuple[int, ...] = (2, 3, 4)
TOLERANCES_MS: tuple[float, ...] = (60.0, 120.0, 240.0)
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

# Per-camera motion-score disagreement. 0.0 is the idealised case where every
# camera scores the same event identically; 0.05 exceeds the syncer's
# MOTION_MATCH_TOLERANCE of 0.03, so this axis is what makes that constant
# measurable rather than assumed.
MOTION_NOISES: tuple[float, ...] = (0.0, 0.01, 0.05)

VIDEO_FPS = 24
VIDEO_SECONDS = 20
PAN_SPEED_PX = 12
MOTION_WINDOWS: tuple[tuple[float, float], ...] = ((2.0, 7.0), (10.0, 15.0), (16.0, 19.0))

logger = logging.getLogger(__name__)


def _signal_variance(frames: list[FrameReference]) -> float:
    values = [f.motion_score or 0.0 for f in frames]
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _median_interval_ms(frames: list[FrameReference]) -> float:
    stamps = sorted(f.timestamp_ms or 0.0 for f in frames)
    if len(stamps) < 2:
        return float("nan")
    return float(np.median(np.diff(stamps)))


def _trial(
    anchor_frames: list[FrameReference],
    *,
    true_offset_ms: float,
    jitter_ms: float,
    n_sources: int,
    tolerance_ms: float,
    seed: int,
    motion_noise: float = 0.0,
) -> dict[str, Any]:
    frames_by_source: dict[str, list[FrameReference]] = {"cam_0": anchor_frames}
    for source_index in range(1, n_sources):
        source_id = f"cam_{source_index}"
        frames_by_source[source_id] = derive_offset_stream(
            anchor_frames,
            true_offset_ms,
            jitter_ms=jitter_ms,
            seed=seed * 100 + source_index,
            source_id=source_id,
            motion_noise=motion_noise,
        )

    entries = MultiSourceSyncer().build_sync_map(
        frames_by_source, sync_group_id="a6", tolerance_ms=tolerance_ms
    )

    estimates = [
        -value
        for entry in entries
        for source, value in entry.offsets_ms.items()
        if source != "cam_0"
    ]
    recovered = float(np.mean(estimates)) if estimates else float("nan")
    error = abs(recovered - true_offset_ms) if estimates else float("nan")

    interval = _median_interval_ms(anchor_frames)
    if math.isnan(recovered):
        sign_correct = False
    elif true_offset_ms == 0.0:
        sign_correct = abs(recovered) < interval
    else:
        sign_correct = (recovered > 0) == (true_offset_ms > 0)

    complete = [
        entry for entry in entries if len(entry.aligned_frames) == len(frames_by_source)
    ]
    return {
        "true_offset_ms": true_offset_ms,
        "jitter_ms": jitter_ms,
        "motion_noise": motion_noise,
        "n_sources": n_sources,
        "tolerance_ms": tolerance_ms,
        "seed": seed,
        "n_anchor_frames": len(anchor_frames),
        "median_sample_interval_ms": round(interval, 3),
        "anchor_motion_variance": round(_signal_variance(anchor_frames), 6),
        "motion_variance_guard_fires": _signal_variance(anchor_frames) < MIN_MOTION_VARIANCE,
        "n_entries": len(entries),
        "group_formation_rate": round(len(entries) / max(len(anchor_frames), 1), 4),
        "group_completeness": round(len(complete) / len(entries), 4) if entries else 0.0,
        "recovered_offset_ms": round(recovered, 3) if estimates else float("nan"),
        "offset_abs_error_ms": round(error, 3) if estimates else float("nan"),
        "sign_correct": sign_correct,
        "error_exceeds_frame_interval": (
            bool(error > interval) if estimates and not math.isnan(interval) else True
        ),
        "motion_match_tolerance": MOTION_MATCH_TOLERANCE,
        "offset_bucket_ms": OFFSET_BUCKET_MS,
        "min_motion_variance": MIN_MOTION_VARIANCE,
    }


def run(results_dir: Path | None = None, *, quick: bool = False, seed: int = 0) -> ResultWriter:
    writer = ResultWriter(EXP_ID, results_dir)
    offsets = TRUE_OFFSETS_MS[:3] if quick else TRUE_OFFSETS_MS
    jitters = JITTERS_MS[:1] if quick else JITTERS_MS
    source_counts = SOURCE_COUNTS[:1] if quick else SOURCE_COUNTS
    tolerances = TOLERANCES_MS[1:2] if quick else TOLERANCES_MS
    seeds = SEEDS[:1] if quick else SEEDS
    motion_noises = MOTION_NOISES[:1] if quick else MOTION_NOISES

    with tempfile.TemporaryDirectory(prefix="a6_") as tmp:
        video = Path(tmp) / "anchor.mp4"
        make_motion_video(
            video,
            fps=VIDEO_FPS,
            seconds=VIDEO_SECONDS,
            motion_windows=MOTION_WINDOWS,
            speed_px_per_frame=PAN_SPEED_PX,
            seed=seed,
        )
        anchor_frames = frames_from_video(video, source_id="cam_0")
        logger.info("anchor stream: %d frames", len(anchor_frames))

        for true_offset in offsets:
            for jitter in jitters:
                for n_sources in source_counts:
                    for tolerance in tolerances:
                        for motion_noise in motion_noises:
                            for trial_seed in seeds:
                                writer.add(
                                    **_trial(
                                        anchor_frames,
                                        true_offset_ms=true_offset,
                                        jitter_ms=jitter,
                                        n_sources=n_sources,
                                        tolerance_ms=tolerance,
                                        seed=trial_seed,
                                        motion_noise=motion_noise,
                                    )
                                )
            logger.info("offset %+.0f ms done (%d rows)", true_offset, len(writer))
    return writer


def main(argv: list[str] | None = None) -> int:
    args = experiment_parser(EXP_ID, __doc__ or "").parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    writer = run(results_dir=args.results_dir, quick=args.quick, seed=args.seed)
    finish(writer)

    header = "  {:>9}{:>8}{:>9}{:>9}{:>7}{:>6}".format(
        "|offset|", "mnoise", "err_ms", "sign_ok", "form", "n"
    )
    print(header)
    for offset in sorted({abs(row["true_offset_ms"]) for row in writer.rows}):
        for noise in sorted({row["motion_noise"] for row in writer.rows}):
            subset = [
                row
                for row in writer.rows
                if abs(row["true_offset_ms"]) == offset and row["motion_noise"] == noise
            ]
            errors = [
                r["offset_abs_error_ms"]
                for r in subset
                if not math.isnan(r["offset_abs_error_ms"])
            ]
            mean_error = float(np.mean(errors)) if errors else float("nan")
            print(
                "  {:>9.0f}{:>8.2f}{:>9.1f}{:>9.2f}{:>7.2f}{:>6d}".format(
                    offset,
                    noise,
                    mean_error,
                    float(np.mean([r["sign_correct"] for r in subset])),
                    float(np.mean([r["group_formation_rate"] for r in subset])),
                    len(subset),
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
