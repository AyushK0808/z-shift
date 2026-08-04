from __future__ import annotations

from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field

from spatial_ingestion.metadata.schema import FrameReference
from spatial_ingestion.reconstruction.config import DEFAULT_MODEL_NAME


class ReconstructionMode(str, Enum):
    SINGLE_VIEW = "single_view"
    MULTI_VIEW = "multi_view"
    VIDEO_SEQUENCE = "video_sequence"
    SYNCHRONIZED_VIEWS = "synchronized_views"


class SyncViewGroup(BaseModel):
    anchor_timestamp_ms: float
    frames_by_source: dict[str, FrameReference]
    offsets_ms: dict[str, float] = Field(default_factory=dict)


class ReconstructionArtifactKind(str, Enum):
    POINT_CLOUD = "point_cloud"
    POSES = "poses"
    RUN_MANIFEST = "run_manifest"
    MESH = "mesh"
    RIGGED_MESH = "rigged_mesh"
    SKELETON = "skeleton"
    SKINNING_WEIGHTS = "skinning_weights"


class Mast3rRunParams(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_name: str = DEFAULT_MODEL_NAME
    device: str = "auto"
    image_size: int = 512
    pairing_strategy: str = "complete"
    tsdf_thresh: float = 0
    min_conf_thr: float = 1.5
    seed: int | None = None
    dry_run: bool = False


class ReconstructionJob(BaseModel):
    mode: ReconstructionMode
    image_uris: list[str]
    job_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    label: str = ""
    backend_name: str = "mast3r"
    frames: list[FrameReference] = Field(default_factory=list)
    sync_view_groups: list[SyncViewGroup] = Field(default_factory=list)
    output_path: str | None = None
    params: Mast3rRunParams = Field(default_factory=Mast3rRunParams)
    metadata: dict[str, object] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
