from __future__ import annotations

from pathlib import Path

from spatial_ingestion.config import RECONSTRUCTION_OUTPUT_ROOT
from spatial_ingestion.reconstruction.backends.base import BackendExecutionPlan, ReconstructionBackend
from spatial_ingestion.reconstruction.models import ReconstructionArtifact, ReconstructionArtifactKind, ReconstructionJob, ReconstructionMode


class Mast3rBackend(ReconstructionBackend):
    name = "mast3r"

    def __init__(self, output_root: Path = RECONSTRUCTION_OUTPUT_ROOT) -> None:
        self._output_root = output_root

    def supports(self, job: ReconstructionJob) -> bool:
        return job.mode in {
            ReconstructionMode.MULTI_VIEW,
            ReconstructionMode.SYNCHRONIZED_VIEWS,
        }

    def plan(self, job: ReconstructionJob) -> BackendExecutionPlan:
        if not self.supports(job):
            raise ValueError(f"{self.name} does not support reconstruction mode {job.mode}")

        output_dir = self._output_dir(job)
        warnings = list(job.warnings)

        expected_artifacts = [
            ReconstructionArtifact(
                kind=ReconstructionArtifactKind.POINT_CLOUD,
                uri=(output_dir / "point_cloud.ply").as_uri(),
            ),
            ReconstructionArtifact(
                kind=ReconstructionArtifactKind.POSES,
                uri=(output_dir / "camera_poses.json").as_uri(),
            ),
            ReconstructionArtifact(
                kind=ReconstructionArtifactKind.RUN_MANIFEST,
                uri=(output_dir / "run_manifest.json").as_uri(),
            ),
            ReconstructionArtifact(
                kind=ReconstructionArtifactKind.MESH,
                uri=(output_dir / "mesh.obj").as_uri(),
            ),
        ]

        return BackendExecutionPlan(
            backend_name=self.name,
            expected_artifacts=expected_artifacts,
            warnings=warnings,
        )

    def _output_dir(self, job: ReconstructionJob) -> Path:
        stem = self._job_stem(job)
        return self._output_root / self.name / stem

    @staticmethod
    def _job_stem(job: ReconstructionJob) -> str:
        if job.frames:
            joined = "_".join(frame.source_id or frame.frame_id for frame in job.frames[:3])
            return joined[:120]
        return "reconstruction_job"
