from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from spatial_ingestion.auto_rigging.models import AutoRigResult, RiggedMesh
from spatial_ingestion.config import AUTO_RIGGING_OUTPUT_ROOT


class RigMetadataExporter:
    """Writes Phase 5 rig metadata while skinned GLB export is being built."""

    def __init__(self, output_root: Path = AUTO_RIGGING_OUTPUT_ROOT) -> None:
        self._output_root = output_root
        self._output_root.mkdir(parents=True, exist_ok=True)

    def export(self, rigged_mesh: RiggedMesh, job_id: str | None = None) -> tuple[str, str]:
        job = job_id or f"rig_{uuid4().hex}"
        job_dir = self._output_root / job
        job_dir.mkdir(parents=True, exist_ok=True)

        skeleton_path = job_dir / "skeleton.json"
        weights_path = job_dir / "skinning_weights.json"

        skeleton_path.write_text(
            rigged_mesh.skeleton.model_dump_json(indent=2),
            encoding="utf-8",
        )
        weights_path.write_text(
            rigged_mesh.skinning.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return skeleton_path.as_uri(), weights_path.as_uri()

    def export_result(self, result: AutoRigResult, job_id: str | None = None) -> AutoRigResult:
        skeleton_uri, weights_uri = self.export(result.rigged_mesh, job_id=job_id)
        return AutoRigResult(
            rigged_mesh=result.rigged_mesh,
            skeleton_uri=skeleton_uri,
            weights_uri=weights_uri,
            warnings=result.warnings,
        )


def write_debug_rig_json(result: AutoRigResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8")
