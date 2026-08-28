"""A7 - motion-adaptive sampling vs uniform.

Supports the SV-B claim that the frame-selection strategy "provides a
practical mechanism for controlling computational cost" and "preferentially
retains high-motion observations".

The uniform baseline is matched to the adaptive sampler's *own* frame count on
each video. Without that matching the comparison measures budget rather than
strategy, and the adaptive sampler would win trivially by taking more frames.

Headline metric is budget concentration: the fraction of selected frames that
land inside the labelled motion windows. If it is not above the uniform
baseline, the sampler is not doing what the paper says it does, and that is a
result too.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bench.csvio import ResultWriter, mean_std
from bench.fixtures import frames_from_video, make_motion_video
from bench.harness import experiment_parser, finish
from spatial_ingestion.batch_normalization.video_sampler import MotionAdaptiveFrameSampler

EXP_ID = "a7_motion_sampling"

VIDEO_FPS = 24
VIDEO_SECONDS = 30
SEEDS: tuple[int, ...] = (0, 1, 2)

# Motion fraction of the clip, and how hard the camera pans while moving.
MOTION_FRACTIONS: tuple[float, ...] = (0.1, 0.3, 0.6)
MAGNITUDES: dict[str, int] = {"slow": 5, "fast": 14}

# The sampler's hardcoded decision thresholds and intervals. Swept so the
# reported behaviour can be attributed to them rather than to the video.
THRESHOLD_VARIANTS: dict[str, dict[str, Any]] = {
    "shipped": {},
    "intervals_2x": {
        "low_motion_interval_frames": 48,
        "medium_motion_interval_frames": 24,
        "high_motion_interval_frames": 8,
    },
    "intervals_half": {
        "low_motion_interval_frames": 12,
        "medium_motion_interval_frames": 6,
        "high_motion_interval_frames": 2,
    },
}

logger = logging.getLogger(__name__)


def _windows_for(fraction: float, seconds: int) -> tuple[tuple[float, float], ...]:
    """Two equal motion windows covering `fraction` of the clip, centred apart."""
    total = fraction * seconds
    half = total / 2.0
    first_start = seconds * 0.15
    second_start = seconds * 0.60
    return ((first_start, first_start + half), (second_start, second_start + half))


def _inside(timestamp_ms: float, windows: list[tuple[float, float]]) -> bool:
    return any(start <= timestamp_ms < end for start, end in windows)


def _redundancy(video_path: Path, indices: list[int]) -> float:
    """Mean pairwise cosine distance of downscaled grayscale selected frames.

    Higher means the selected set is less redundant, which is the actual goal
    of adaptive sampling -- fewer near-duplicate views for the same budget.
    """
    capture = cv2.VideoCapture(str(video_path))
    wanted = set(indices)
    vectors: list[np.ndarray] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in wanted:
                small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 24))
                vector = small.astype(float).ravel()
                norm = float(np.linalg.norm(vector))
                vectors.append(vector / norm if norm > 0 else vector)
            frame_index += 1
    finally:
        capture.release()

    if len(vectors) < 2:
        return float("nan")
    matrix = np.asarray(vectors)
    similarity = matrix @ matrix.T
    upper = similarity[np.triu_indices(len(vectors), k=1)]
    return float(1.0 - upper.mean())


def _stats(
    label: str,
    video: Path,
    indices: list[int],
    timestamps: list[float],
    windows: list[tuple[float, float]],
    seconds: int,
) -> dict[str, Any]:
    inside_flags = [_inside(t, windows) for t in timestamps]
    moving_seconds = sum(end - start for start, end in windows) / 1000.0
    static_seconds = max(seconds - moving_seconds, 1e-9)
    n_inside = int(sum(inside_flags))

    ordered = np.sort(np.asarray(timestamps, dtype=float))
    largest_gap = float(np.diff(ordered).max()) / 1000.0 if ordered.size > 1 else float("nan")

    return {
        "strategy": label,
        "n_selected": len(indices),
        "budget_concentration": round(n_inside / max(len(indices), 1), 4),
        "rate_inside_fps": round(n_inside / moving_seconds, 4) if moving_seconds else float("nan"),
        "rate_outside_fps": round((len(indices) - n_inside) / static_seconds, 4),
        "largest_unsampled_gap_s": round(largest_gap, 4),
        "redundancy_distance": round(_redundancy(video, indices), 6),
    }


def _trial(
    video: Path,
    windows: list[tuple[float, float]],
    variant: str,
    seconds: int,
) -> list[dict[str, Any]]:
    sampler = MotionAdaptiveFrameSampler(**THRESHOLD_VARIANTS[variant])
    adaptive = frames_from_video(video, source_id="cam_a", sampler=sampler)
    adaptive_indices = [f.index for f in adaptive]
    adaptive_timestamps = [f.timestamp_ms or 0.0 for f in adaptive]

    # Uniform baseline matched to the adaptive sampler's own frame count.
    total_frames = int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT))
    budget = max(len(adaptive_indices), 1)
    uniform_indices = [
        int(round(position)) for position in np.linspace(0, max(total_frames - 1, 0), budget)
    ]
    uniform_timestamps = [index / VIDEO_FPS * 1000.0 for index in uniform_indices]

    return [
        _stats("adaptive", video, adaptive_indices, adaptive_timestamps, windows, seconds),
        _stats("uniform_matched", video, uniform_indices, uniform_timestamps, windows, seconds),
    ]


def run(results_dir: Path | None = None, *, quick: bool = False, seed: int = 0) -> ResultWriter:
    writer = ResultWriter(EXP_ID, results_dir)
    fractions = MOTION_FRACTIONS[:1] if quick else MOTION_FRACTIONS
    magnitudes = dict(list(MAGNITUDES.items())[:1]) if quick else MAGNITUDES
    variants = ("shipped",) if quick else tuple(THRESHOLD_VARIANTS)
    seeds = SEEDS[:1] if quick else SEEDS

    with tempfile.TemporaryDirectory(prefix="a7_") as tmp:
        tmp_dir = Path(tmp)
        for fraction in fractions:
            for magnitude_name, speed in magnitudes.items():
                for trial_seed in seeds:
                    video = tmp_dir / f"f{fraction}_{magnitude_name}_{trial_seed}.mp4"
                    schedule = _windows_for(fraction, VIDEO_SECONDS)
                    windows_ms = make_motion_video(
                        video,
                        fps=VIDEO_FPS,
                        seconds=VIDEO_SECONDS,
                        motion_windows=schedule,
                        speed_px_per_frame=speed,
                        seed=trial_seed,
                    )
                    for variant in variants:
                        for row in _trial(video, windows_ms, variant, VIDEO_SECONDS):
                            writer.add(
                                motion_fraction=fraction,
                                magnitude=magnitude_name,
                                pan_speed_px=speed,
                                threshold_variant=variant,
                                seed=trial_seed,
                                video_seconds=VIDEO_SECONDS,
                                video_fps=VIDEO_FPS,
                                **row,
                            )
                logger.info(
                    "fraction=%.1f %s done (%d rows)", fraction, magnitude_name, len(writer)
                )
    return writer


def _summarise(writer: ResultWriter) -> None:
    print(
        f"  {'frac':>5}{'mag':>6}{'variant':>15}{'strategy':>17}"
        f"{'n':>5}{'concentr':>10}{'in_fps':>8}{'out_fps':>8}{'redund':>8}"
    )
    keys: list[tuple[Any, ...]] = []
    for row in writer.rows:
        key = (row["motion_fraction"], row["magnitude"], row["threshold_variant"], row["strategy"])
        if key in keys:
            continue
        keys.append(key)
        group = [
            r
            for r in writer.rows
            if (r["motion_fraction"], r["magnitude"], r["threshold_variant"], r["strategy"]) == key
        ]
        conc, _, _ = mean_std([r["budget_concentration"] for r in group])
        n_sel, _, _ = mean_std([r["n_selected"] for r in group])
        inside, _, _ = mean_std([r["rate_inside_fps"] for r in group])
        outside, _, _ = mean_std([r["rate_outside_fps"] for r in group])
        redundancy, _, _ = mean_std([r["redundancy_distance"] for r in group])
        print(
            f"  {key[0]:>5.1f}{key[1]:>6}{key[2]:>15}{key[3]:>17}"
            f"{n_sel:>5.0f}{conc:>10.3f}{inside:>8.2f}{outside:>8.2f}{redundancy:>8.4f}"
        )


def main(argv: list[str] | None = None) -> int:
    args = experiment_parser(EXP_ID, __doc__ or "").parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    writer = run(results_dir=args.results_dir, quick=args.quick, seed=args.seed)
    finish(writer)
    _summarise(writer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
