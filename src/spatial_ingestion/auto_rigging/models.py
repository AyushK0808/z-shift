from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class ArticulationType(str, Enum):
    STATIC = "static"
    BIPED = "biped"
    QUADRUPED = "quadruped"
    WINGED = "winged"


class Joint(BaseModel):
    name: str
    position: tuple[float, float, float]
    parent: str | None = None


class Bone(BaseModel):
    name: str
    parent_joint: str
    child_joint: str


class Skeleton(BaseModel):
    articulation_type: ArticulationType
    joints: list[Joint]
    bones: list[Bone]
    root_joint: str

    @field_validator("joints")
    @classmethod
    def require_unique_joints(cls, joints: list[Joint]) -> list[Joint]:
        names = [joint.name for joint in joints]
        if len(names) != len(set(names)):
            raise ValueError("skeleton joint names must be unique")
        return joints

    @field_validator("bones")
    @classmethod
    def require_unique_bones(cls, bones: list[Bone]) -> list[Bone]:
        names = [bone.name for bone in bones]
        if len(names) != len(set(names)):
            raise ValueError("skeleton bone names must be unique")
        return bones

    def joint_index(self) -> dict[str, int]:
        return {joint.name: index for index, joint in enumerate(self.joints)}


class SkinningWeights(BaseModel):
    joint_names: list[str]
    weights: list[list[float]]
    max_influences: int

    @field_validator("weights")
    @classmethod
    def require_weight_rows(cls, weights: list[list[float]]) -> list[list[float]]:
        if not weights:
            raise ValueError("skinning weights must contain at least one vertex row")
        return weights


class RiggedMesh(BaseModel):
    mesh_uri: str | None = None
    vertex_count: int
    face_count: int
    skeleton: Skeleton
    skinning: SkinningWeights
    metadata: dict[str, object] = Field(default_factory=dict)


class AutoRigConfig(BaseModel):
    articulation_type: ArticulationType = ArticulationType.STATIC
    max_skinning_influences: int = Field(default=4, ge=1)
    normalize_mesh: bool = True
    output_dir: Path | None = None


class AutoRigResult(BaseModel):
    rigged_mesh: RiggedMesh
    skeleton_uri: str | None = None
    weights_uri: str | None = None
    warnings: list[str] = Field(default_factory=list)

