from __future__ import annotations

from uuid import uuid4

import numpy as np

from spatial_ingestion.config import LIVE_BUFFER_SIZE
from spatial_ingestion.live_stream.buffer import BackpressureDecision, LiveStreamBuffer
from spatial_ingestion.metadata.schema import (
    SourceType,
    Track,
    UnifiedSpatialIngestionSchema,
)
from spatial_ingestion.resource_tagging.priority import LatencyAwareResourceTagger


class LiveStreamManager:
    def __init__(self, tagger: LatencyAwareResourceTagger | None = None) -> None:
        self._streams: dict[str, LiveStreamBuffer] = {}
        self._tagger = tagger or LatencyAwareResourceTagger()

    def open_stream(self, stream_id: str | None = None) -> UnifiedSpatialIngestionSchema:
        handle = stream_id or f"stream_{uuid4().hex}"
        self._streams[handle] = LiveStreamBuffer(handle, LIVE_BUFFER_SIZE)
        return UnifiedSpatialIngestionSchema(
            source_type=SourceType.LIVE_STREAM,
            track=Track.LIVE,
            resolution=None,
            frame_count=None,
            is_stream=True,
            compute_priority_score=self._tagger.score(SourceType.LIVE_STREAM, Track.LIVE),
            live_stream_handle=handle,
            metadata={"buffer_size": LIVE_BUFFER_SIZE},
        )

    def push_frame(self, handle: str, image: np.ndarray) -> BackpressureDecision:
        return self._streams[handle].push_frame(image)

    def push_encoded_frame(self, handle: str, payload: bytes) -> BackpressureDecision:
        return self._streams[handle].push_encoded_frame(payload)

    def get_buffer(self, handle: str) -> LiveStreamBuffer:
        return self._streams[handle]

