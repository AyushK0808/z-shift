from __future__ import annotations

from uuid import uuid4

import numpy as np

from spatial_ingestion.config import (
    LIVE_BUFFER_SIZE,
    MAX_LIVE_FRAME_BYTES,
    MAX_LIVE_STREAMS,
    MAX_LIVE_STREAMS_PER_SUBJECT,
)
from spatial_ingestion.live_stream.buffer import BackpressureDecision, LiveStreamBuffer
from spatial_ingestion.metadata.schema import (
    SourceType,
    Track,
    UnifiedSpatialIngestionSchema,
)
from spatial_ingestion.resource_tagging.priority import LatencyAwareResourceTagger


class StreamLimitExceeded(Exception):
    """Raised when a stream creation request exceeds configured limits."""


class StreamOwnershipError(Exception):
    """Raised when a caller tries to reuse or access another subject's stream."""


class LiveStreamManager:
    def __init__(
        self,
        tagger: LatencyAwareResourceTagger | None = None,
        max_streams: int = MAX_LIVE_STREAMS,
        max_streams_per_subject: int = MAX_LIVE_STREAMS_PER_SUBJECT,
        max_frame_payload_bytes: int = MAX_LIVE_FRAME_BYTES,
    ) -> None:
        self._streams: dict[str, LiveStreamBuffer] = {}
        self._owners: dict[str, str] = {}
        self._tagger = tagger or LatencyAwareResourceTagger()
        self._max_streams = max_streams
        self._max_streams_per_subject = max_streams_per_subject
        self._max_frame_payload_bytes = max_frame_payload_bytes

    def open_stream(
        self,
        stream_id: str | None = None,
        owner_subject: str = "anonymous",
    ) -> UnifiedSpatialIngestionSchema:
        handle = stream_id or f"stream_{uuid4().hex}"
        if handle in self._streams and not self.is_owner(handle, owner_subject):
            raise StreamOwnershipError("stream id is already owned by another client")

        if handle not in self._streams:
            if len(self._streams) >= self._max_streams:
                raise StreamLimitExceeded("maximum live stream count reached")
            if self._owner_stream_count(owner_subject) >= self._max_streams_per_subject:
                raise StreamLimitExceeded("maximum live stream count reached for subject")
            self._streams[handle] = LiveStreamBuffer(
                handle,
                LIVE_BUFFER_SIZE,
                max_payload_bytes=self._max_frame_payload_bytes,
            )
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

    def _owner_stream_count(self, subject: str) -> int:
        return sum(1 for owner in self._owners.values() if owner == subject)
