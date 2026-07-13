from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    SINGLE_IMAGE = "single_image"
    IMAGE_FOLDER = "image_folder"
    SINGLE_VIDEO = "single_video"
    VIDEO_FOLDER = "video_folder"
    LIVE_STREAM = "live_stream"
    UNKNOWN = "unknown"


class Track(str, Enum):
    BATCH = "track_a_batch"
    LIVE = "track_b_live"


class CameraIntrinsics(BaseModel):
    focal_length_mm: float | None = None
    focal_length_35mm: float | None = None
    make: str | None = None
    model: str | None = None
    lens_model: str | None = None
    raw_exif: dict[str, Any] = Field(default_factory=dict)


class FrameReference(BaseModel):
    frame_id: str
    uri: str | None = None
    original_uri: str | None = None
    index: int
    timestamp_ms: float | None = None
    source_id: str | None = None
    motion_score: float | None = None
    resolution: tuple[int, int] | None = None
    camera_intrinsics: CameraIntrinsics | None = None


class SyncMapEntry(BaseModel):
    sync_group_id: str
    anchor_timestamp_ms: float
    aligned_frames: dict[str, int]
    offsets_ms: dict[str, float] = Field(default_factory=dict)


class UnifiedSpatialIngestionSchema(BaseModel):
    source_type: SourceType
    track: Track
    resolution: tuple[int, int] | None = None
    frame_count: int | None = None
    is_stream: bool
    camera_intrinsics: CameraIntrinsics | None = None
    compute_priority_score: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sync_group_id: str | None = None
    frames: list[FrameReference] = Field(default_factory=list)
    live_stream_handle: str | None = None
    sync_map: list[SyncMapEntry] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def new_sync_group_id(cls) -> str:
        return f"sync_{uuid4().hex}"
