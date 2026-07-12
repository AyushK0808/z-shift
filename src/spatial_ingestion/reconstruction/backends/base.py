from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from spatial_ingestion.reconstruction.models import ReconstructionArtifact, ReconstructionJob


class BackendExecutionPlan(BaseModel):
    backend_name: str
    command: list[str] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)
    working_directory: str | None = None
    inputs: list[str] = Field(default_factory=list)
    expected_artifacts: list[ReconstructionArtifact] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ReconstructionBackend(ABC):
    name: str

    @abstractmethod
    def supports(self, job: ReconstructionJob) -> bool:
        raise NotImplementedError

    @abstractmethod
    def plan(self, job: ReconstructionJob) -> BackendExecutionPlan:
        raise NotImplementedError
