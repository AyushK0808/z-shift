from __future__ import annotations

from pathlib import Path

from spatial_ingestion.config import RECONSTRUCTION_OUTPUT_ROOT
from spatial_ingestion.reconstruction.backends.base import BackendExecutionPlan, ReconstructionBackend
from spatial_ingestion.reconstruction.models import ReconstructionArtifact, ReconstructionJob, ReconstructionMode


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
        command = [
            "uv",
            "run",
            "python",
            "-m",
            "spatial_ingestion.reconstruction.runners.mast3r",
            "--output-dir",
            str(output_dir),
            "--output-path",
            str(output_dir / "mesh.obj"),
        ]

        warnings = list(job.warnings)

        expected_artifacts = [
            ReconstructionArtifact(
                kind="point_cloud",
                uri=(output_dir / "point_cloud.ply").as_uri(),
            ),
            ReconstructionArtifact(
                kind="poses",
                uri=(output_dir / "camera_poses.json").as_uri(),
            ),
            ReconstructionArtifact(
                kind="run_manifest",
                uri=(output_dir / "run_manifest.json").as_uri(),
            ),
            ReconstructionArtifact(
                kind="mesh",
                uri=(output_dir / "mesh.obj").as_uri(),
            ),
        ]

        environment = {
            "Z_SHIFT_RECONSTRUCTION_BACKEND": self.name,
            "Z_SHIFT_RECONSTRUCTION_MODE": job.mode.value,
        }

        if job.mode == ReconstructionMode.SYNCHRONIZED_VIEWS:
            command.extend(["--pairing-strategy", "swin"])
        command.extend(job.image_uris)

        return BackendExecutionPlan(
            backend_name=self.name,
            command=command,
            environment=environment,
            working_directory=str(output_dir),
            inputs=list(job.image_uris),
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
