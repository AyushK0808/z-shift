from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class SampledFrame:
    image: np.ndarray
    index: int
    timestamp_ms: float
    motion_score: float


class MotionAdaptiveFrameSampler:
    """Samples densely during motion and sparsely during static spans."""

    def __init__(
        self,
        low_motion_interval_frames: int = 24,
        medium_motion_interval_frames: int = 12,
        high_motion_interval_frames: int = 4,
        min_seconds_between_samples: float = 0.08,
    ) -> None:
        self._low_interval = low_motion_interval_frames
        self._medium_interval = medium_motion_interval_frames
        self._high_interval = high_motion_interval_frames
        self._min_seconds_between = min_seconds_between_samples

    def sample(self, video_path: Path) -> list[SampledFrame]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        min_frame_gap = max(1, int(round(fps * self._min_seconds_between)))
        samples: list[SampledFrame] = []
        previous_gray: np.ndarray | None = None
        last_sample_index = -10**9
        frame_index = 0

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                motion_score = self._motion_score(previous_gray, gray)
                dynamic_interval = self._interval_for_motion(motion_score)
                should_sample = (
                    frame_index == 0
                    or (
                        frame_index - last_sample_index >= min_frame_gap
                        and frame_index - last_sample_index >= dynamic_interval
                    )
                )

                if should_sample:
                    timestamp_ms = (frame_index / fps) * 1000.0
                    samples.append(
                        SampledFrame(
                            image=frame.copy(),
                            index=frame_index,
                            timestamp_ms=timestamp_ms,
                            motion_score=motion_score,
                        )
                    )
                    last_sample_index = frame_index

                previous_gray = gray
                frame_index += 1
        finally:
            capture.release()

        return samples

    @staticmethod
    def _motion_score(previous: np.ndarray | None, current: np.ndarray) -> float:
        if previous is None:
            return 1.0
        previous_small = cv2.resize(previous, (160, 90))
        current_small = cv2.resize(current, (160, 90))
        diff = cv2.absdiff(previous_small, current_small)
        mean_diff = float(np.mean(diff)) / 255.0
        changed_pixels = float(np.mean(diff > 18))
        return round(min(1.0, (mean_diff * 1.8) + (changed_pixels * 0.8)), 4)

    def _interval_for_motion(self, motion_score: float) -> int:
        if motion_score >= 0.18:
            return self._high_interval
        if motion_score >= 0.055:
            return self._medium_interval
        return self._low_interval

