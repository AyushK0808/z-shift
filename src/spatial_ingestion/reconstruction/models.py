from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from spatial_ingestion.generation_handoff import HandoffFrame, SyncViewGroup


class ReconstructionMode(str, Enum):
    SINGLE_VIEW = "single_view"
    MULTI_VIEW = "multi_view"
    VIDEO_SEQUENCE = "video_sequence"
    SYNCHRONIZED_VIEWS = "synchronized_views"


class ReconstructionArtifact(BaseModel):
    kind: str
    uri: str
    metadata: dict[str, object] = Field(default_factory=dict)


class ReconstructionJob(BaseModel):
    mode: ReconstructionMode
    backend_name: str
    image_uris: list[str]
    frames: list[HandoffFrame] = Field(default_factory=list)
    sync_view_groups: list[SyncViewGroup] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
