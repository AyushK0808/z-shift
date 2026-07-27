from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import pyvista as pv

from spatial_ingestion.reconstruction.models import ReconstructionJob
from spatial_ingestion.reconstruction.pipeline import _resolve_output_paths
from spatial_ingestion.reconstruction.pipeline import run as run_reconstruction
from spatial_ingestion.refinement import MeshCleaningConfig, clean_mesh


class PipelineArtifactError(RuntimeError):
    """Raised when a phase boundary artifact is missing or unusable."""


@dataclass(frozen=True)
class FinalPipelineResult:
    job_id: str
    raw_mesh_path: Path
    refined_mesh_path: Path
    reconstruction_manifest_path: Path
    refinement_manifest_path: Path
    refinement_diagnostics: dict[str, Any]


def run_phase2_phase3_pipeline(
    job: ReconstructionJob,
    refinement_config: MeshCleaningConfig | None = None,
    *,
    refined_output_path: Path | str | None = None,
) -> FinalPipelineResult:
    """Run Phase 2 reconstruction, then clean its mesh with Phase 3 refinement."""
    exit_code = run_reconstruction(job)
    if exit_code != 0:
        raise RuntimeError(f"Phase 2 reconstruction failed with exit code {exit_code}")

    raw_mesh_path, output_dir = _resolve_output_paths(job)
    _validate_raw_mesh(raw_mesh_path)

    raw_mesh = pv.read(str(raw_mesh_path))
    refinement_result = clean_mesh(raw_mesh, refinement_config or MeshCleaningConfig())
    cleaned_mesh = refinement_result["mesh"]

    destination = Path(refined_output_path).expanduser().resolve() if refined_output_path else _default_refined_path(raw_mesh_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cleaned_mesh.save(str(destination))

    manifest_path = output_dir / "refinement_manifest.json"
    diagnostics = _serializable_diagnostics(refinement_result)
    manifest = {
        "job_id": job.job_id,
        "input_mesh": str(raw_mesh_path),
        "output_mesh": str(destination),
        **diagnostics,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return FinalPipelineResult(
        job_id=job.job_id,
        raw_mesh_path=raw_mesh_path,
        refined_mesh_path=destination,
        reconstruction_manifest_path=output_dir / "run_manifest.json",
        refinement_manifest_path=manifest_path,
        refinement_diagnostics=diagnostics,
    )


def result_to_dict(result: FinalPipelineResult) -> dict[str, Any]:
    data = asdict(result)
    for key in (
        "raw_mesh_path",
        "refined_mesh_path",
        "reconstruction_manifest_path",
        "refinement_manifest_path",
    ):
        data[key] = str(data[key])
    return data


def _default_refined_path(raw_mesh_path: Path) -> Path:
    suffix = raw_mesh_path.suffix or ".obj"
    return raw_mesh_path.with_name(f"{raw_mesh_path.stem}_refined{suffix}")


def _validate_raw_mesh(raw_mesh_path: Path) -> None:
    if not raw_mesh_path.exists():
        raise PipelineArtifactError(f"Phase 2 completed but no mesh artifact was produced: {raw_mesh_path}")
    if raw_mesh_path.stat().st_size == 0:
        raise PipelineArtifactError(f"Phase 2 mesh artifact is empty: {raw_mesh_path}")
    if raw_mesh_path.suffix.lower() not in {".obj", ".glb", ".ply", ".stl", ".vtk", ".vtp"}:
        raise PipelineArtifactError(f"Unsupported Phase 2 mesh artifact format: {raw_mesh_path.suffix}")


def _serializable_diagnostics(refinement_result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in refinement_result.items() if key != "mesh"}
