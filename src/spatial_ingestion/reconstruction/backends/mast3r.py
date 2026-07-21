from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, cast

from spatial_ingestion.config import RECONSTRUCTION_OUTPUT_ROOT
from spatial_ingestion.reconstruction.backends.base import BackendExecutionPlan, ReconstructionBackend
from spatial_ingestion.reconstruction.models import Mast3rRunParams, ReconstructionArtifact, ReconstructionArtifactKind, ReconstructionJob, ReconstructionMode
from spatial_ingestion.reconstruction.runners._io import uri_to_path

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

        params = Mast3rRunParams(**cast(dict[str, Any], job.metadata or {}))

        if job.output_path:
            output_path = Path(job.output_path).resolve()
            output_dir = output_path.parent
        else:
            output_dir = self._output_dir(job)
            output_path = output_dir / "mesh.obj"

        image_paths = [uri_to_path(uri) for uri in job.image_uris]

        sync_view_groups = job.sync_view_groups if job.sync_view_groups else None

        exit_code = mast3r_run(
            image_paths=image_paths,
            output_dir=output_dir,
            output_path=output_path,
            model_name=params.model_name,
            device=params.device,
            image_size=params.image_size,
            pairing_strategy=params.pairing_strategy,
            tsdf_thresh=params.tsdf_thresh,
            min_conf_thr=params.min_conf_thr,
            seed=params.seed,
            sync_view_groups=sync_view_groups,
            dry_run=params.dry_run,
            frames=job.frames,
        )

        expected_paths: list[Path] = [output_dir / "run_manifest.json"]
        if output_path is not None:
            expected_paths.append(output_path)
        for path in expected_paths:
            if not path.exists():
                logger.warning("Expected artifact not found: %s", path.as_uri())

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



