from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from spatial_ingestion.config import RECONSTRUCTION_OUTPUT_ROOT
from spatial_ingestion.reconstruction.backends.base import BackendExecutionPlan, ReconstructionBackend
from spatial_ingestion.reconstruction.models import ReconstructionArtifact, ReconstructionArtifactKind, ReconstructionJob, ReconstructionMode
from spatial_ingestion.reconstruction.runners._io import uri_to_path, uri_to_path_or_none

logger = logging.getLogger(__name__)


class Mast3rBackend(ReconstructionBackend):
    name = "mast3r"

    def __init__(self, output_root: Path = RECONSTRUCTION_OUTPUT_ROOT) -> None:
        self._output_root = output_root

    def supports(self, job: ReconstructionJob) -> bool:
        return job.mode in {
            ReconstructionMode.MULTI_VIEW,
            ReconstructionMode.VIDEO_SEQUENCE,
            ReconstructionMode.SYNCHRONIZED_VIEWS,
        }

    def plan(self, job: ReconstructionJob) -> BackendExecutionPlan:
        if not self.supports(job):
            raise ValueError(f"{self.name} does not support reconstruction mode {job.mode}")

        output_dir = self._output_dir(job)
        warnings = list(job.warnings)

        expected_artifacts = [
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

    def execute(self, job: ReconstructionJob) -> int:
        from spatial_ingestion.reconstruction.runners.mast3r import run as mast3r_run

        if not self.supports(job):
            raise ValueError(f"{self.name} does not support reconstruction mode {job.mode}")

        metadata: dict[str, Any] = job.metadata or {}

        explicit_output_path = metadata.get("output_path")
        if explicit_output_path:
            output_path = Path(str(explicit_output_path)).resolve()
            output_dir = output_path.parent
        else:
            output_dir = self._output_dir(job)
            output_path = output_dir / "mesh.obj"

        image_paths = [uri_to_path(uri) for uri in job.image_uris]

        pairing_strategy = metadata.get("pairing_strategy", "complete")
        sync_view_groups = job.sync_view_groups if job.sync_view_groups else None

        exit_code = mast3r_run(
            image_paths=image_paths,
            output_dir=output_dir,
            output_path=output_path,
            model_name=metadata.get("model_name", "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"),
            device=metadata.get("device", "auto"),
            image_size=metadata.get("image_size", 512),
            pairing_strategy=pairing_strategy,
            tsdf_thresh=metadata.get("tsdf_thresh", 0),
            min_conf_thr=metadata.get("min_conf_thr", 2.0),
            seed=metadata.get("seed"),
            sync_view_groups=sync_view_groups,
            dry_run=metadata.get("dry_run", False),
        )

        plan = self.plan(job)
        for artifact in plan.expected_artifacts:
            uri = artifact.uri
            artifact_path = _artifact_path_from_uri(uri)
            if artifact_path and not artifact_path.exists():
                logger.warning("Expected artifact not found: %s", uri)

        return exit_code

    def _output_dir(self, job: ReconstructionJob) -> Path:
        stem = self._job_stem(job)
        return self._output_root / self.name / stem

    @staticmethod
    def _job_stem(job: ReconstructionJob) -> str:
        if not job.frames:
            return "reconstruction_job"
        frame_ids = sorted(f.frame_id for f in job.frames)
        joined = "_".join(frame_ids)
        h = hashlib.sha256(joined.encode()).hexdigest()[:12]
        prefixes = sorted(set(f.source_id or f.frame_id for f in job.frames[:3]))
        return f"{'_'.join(prefixes)}_{h}"[:120]


def _artifact_path_from_uri(uri: str) -> Path | None:
    return uri_to_path_or_none(uri)
