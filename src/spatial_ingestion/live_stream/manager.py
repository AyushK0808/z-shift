from __future__ import annotations

from uuid import uuid4

import numpy as np

from spatial_ingestion.config import LIVE_BUFFER_SIZE, MAX_LIVE_STREAMS
from spatial_ingestion.live_stream.buffer import BackpressureDecision, LiveStreamBuffer
from spatial_ingestion.metadata.schema import (
    SourceType,
    Track,
    UnifiedSpatialIngestionSchema,
)
from spatial_ingestion.resource_tagging.priority import LatencyAwareResourceTagger


class LiveStreamManager:
    def __init__(
        self,
        tagger: LatencyAwareResourceTagger | None = None,
        max_streams: int = MAX_LIVE_STREAMS,
    ) -> None:
        self._streams: dict[str, LiveStreamBuffer] = {}
        self._owners: dict[str, str] = {}
        self._tagger = tagger or LatencyAwareResourceTagger()
        self._max_streams = max_streams

    def open_stream(
        self,
        stream_id: str | None = None,
        owner_subject: str = "anonymous",
    ) -> UnifiedSpatialIngestionSchema:
        handle = stream_id or f"stream_{uuid4().hex}"
        if handle in self._streams and not self.is_owner(handle, owner_subject):
            raise PermissionError("stream id is already owned by another client")
        if handle not in self._streams and len(self._streams) >= self._max_streams:
            raise RuntimeError("maximum live stream count reached")

        self._streams[handle] = LiveStreamBuffer(handle, LIVE_BUFFER_SIZE)
        self._owners[handle] = owner_subject
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

    def has_stream(self, handle: str) -> bool:
        return handle in self._streams

    def is_owner(self, handle: str, subject: str) -> bool:
        return self._owners.get(handle) == subject

    def close_stream(self, handle: str, subject: str | None = None) -> bool:
        if handle not in self._streams:
            return False
        if subject is not None and not self.is_owner(handle, subject):
            return False
        self._streams.pop(handle, None)
        self._owners.pop(handle, None)
        return True
