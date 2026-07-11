from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

import cv2
import numpy as np


@dataclass(frozen=True)
class LiveFrame:
    frame_id: str
    image: np.ndarray
    timestamp: datetime
    sequence: int


@dataclass(frozen=True)
class BackpressureDecision:
    accepted: bool
    action: str
    dropped_frames: int = 0


class LiveStreamBuffer:
    """Low-latency in-memory ring buffer for live frames."""

    def __init__(self, stream_id: str, max_frames: int = 64) -> None:
        self.stream_id = stream_id
        self._buffer: deque[LiveFrame] = deque(maxlen=max_frames)
        self._sequence = 0
        self._dropped = 0

    def push_encoded_frame(self, payload: bytes) -> BackpressureDecision:
        arr = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            return BackpressureDecision(accepted=False, action="invalid_frame")
        return self.push_frame(image)

    def push_frame(self, image: np.ndarray) -> BackpressureDecision:
        dropped = 1 if len(self._buffer) == self._buffer.maxlen else 0
        if dropped:
            self._dropped += 1

        self._buffer.append(
            LiveFrame(
                frame_id=f"live_{uuid4().hex}",
                image=image,
                timestamp=datetime.now(timezone.utc),
                sequence=self._sequence,
            )
        )
        self._sequence += 1
        return BackpressureDecision(
            accepted=True,
            action="drop_oldest" if dropped else "accepted",
            dropped_frames=dropped,
        )

    def latest(self) -> LiveFrame | None:
        return self._buffer[-1] if self._buffer else None

    def snapshot(self) -> list[LiveFrame]:
        return list(self._buffer)

    @property
    def dropped_frames(self) -> int:
        return self._dropped

