from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from spatial_ingestion.metadata.schema import CameraIntrinsics


class Mast3rRunParams(BaseModel):
    model_name: str = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    device: str = "auto"
    image_size: int = 512
    pairing_strategy: str = "complete"
    tsdf_thresh: float = 0
    min_conf_thr: float = 2.0
    seed: int | None = None
    dry_run: bool = False


class GenerationMode(str, Enum):
    SINGLE_VIEW = "single_view"
    MULTI_VIEW = "multi_view"
    VIDEO_SEQUENCE = "video_sequence"
    SYNCHRONIZED_VIEWS = "synchronized_views"
    LIVE_STREAM = "live_stream"


class HandoffFrame(BaseModel):
    frame_id: str
    uri: str
    index: int
    source_id: str | None = None
    timestamp_ms: float | None = None
    motion_score: float | None = None
    resolution: tuple[int, int] | None = None
    camera_intrinsics: CameraIntrinsics | None = None


class SyncViewGroup(BaseModel):
    anchor_timestamp_ms: float
    frames_by_source: dict[str, HandoffFrame]
    offsets_ms: dict[str, float] = Field(default_factory=dict)


class ReconstructionMode(str, Enum):
    SINGLE_VIEW = "single_view"
    MULTI_VIEW = "multi_view"
    VIDEO_SEQUENCE = "video_sequence"
    SYNCHRONIZED_VIEWS = "synchronized_views"


class ReconstructionArtifactKind(str, Enum):
    POINT_CLOUD = "point_cloud"
    POSES = "poses"
    RUN_MANIFEST = "run_manifest"
    MESH = "mesh"


class ReconstructionArtifact(BaseModel):
    kind: ReconstructionArtifactKind
    uri: str
    metadata: dict[str, object] = Field(default_factory=dict)


class ReconstructionJob(BaseModel):
    mode: ReconstructionMode
    backend_name: str
    image_uris: list[str]
    frames: list[HandoffFrame] = Field(default_factory=list)
    sync_view_groups: list[SyncViewGroup] = Field(default_factory=list)
    output_path: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
