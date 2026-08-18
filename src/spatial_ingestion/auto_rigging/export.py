from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import trimesh

from spatial_ingestion.auto_rigging.gltf import SkinnedGlbExporter
from spatial_ingestion.auto_rigging.models import AutoRigResult, RiggedMesh
from spatial_ingestion.config import AUTO_RIGGING_OUTPUT_ROOT


class RigMetadataExporter:
    """Writes Phase 5 rig metadata while skinned GLB export is being built."""

    def __init__(
        self,
        output_root: Path = AUTO_RIGGING_OUTPUT_ROOT,
        glb_exporter: SkinnedGlbExporter | None = None,
    ) -> None:
        self._output_root = output_root
        self._glb_exporter = glb_exporter or SkinnedGlbExporter()
        self._output_root.mkdir(parents=True, exist_ok=True)

    def _job_dir(self, job_id: str | None = None) -> Path:
        job = job_id or f"rig_{uuid4().hex}"
        job_dir = self._output_root / job
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def export(self, rigged_mesh: RiggedMesh, job_id: str | None = None) -> tuple[str, str]:
        job_dir = self._job_dir(job_id)
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
            rigged_mesh_uri=result.rigged_mesh_uri,
            skeleton_uri=skeleton_uri,
            weights_uri=weights_uri,
            manifest_uri=result.manifest_uri,
            warnings=result.warnings,
        )

    def export_bundle(
        self,
        mesh: trimesh.Trimesh,
        result: AutoRigResult,
        job_id: str | None = None,
        rigged_mesh_path: Path | None = None,
    ) -> AutoRigResult:
        job_dir = rigged_mesh_path.parent if rigged_mesh_path is not None else self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        glb_path = rigged_mesh_path or (job_dir / "rigged_mesh.glb")
        rigged_mesh_uri = self._glb_exporter.export(
            mesh,
            result.rigged_mesh,
            glb_path,
        )
        skeleton_path = job_dir / "skeleton.json"
        weights_path = job_dir / "skinning_weights.json"
        manifest_path = job_dir / "rigging_manifest.json"

        skeleton_path.write_text(
            result.rigged_mesh.skeleton.model_dump_json(indent=2),
            encoding="utf-8",
        )
        weights_path.write_text(
            result.rigged_mesh.skinning.model_dump_json(indent=2),
            encoding="utf-8",
        )
        bundle = AutoRigResult(
            rigged_mesh=result.rigged_mesh,
            rigged_mesh_uri=rigged_mesh_uri,
            skeleton_uri=skeleton_path.as_uri(),
            weights_uri=weights_path.as_uri(),
            manifest_uri=manifest_path.as_uri(),
            warnings=result.warnings,
        )
        manifest_path.write_text(
            json.dumps(bundle.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return bundle


def write_debug_rig_json(result: AutoRigResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8")
