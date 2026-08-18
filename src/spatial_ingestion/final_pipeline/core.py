from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import trimesh

from spatial_ingestion.auto_rigging.models import AutoRigConfig
from spatial_ingestion.auto_rigging.pipeline import AutoRiggingPipeline
from spatial_ingestion.metadata.schema import SourceType
from spatial_ingestion.outcomes_engine.engine import (
    DEFAULT_DELIVERABLES_ROOT,
    DeliverableResult,
    TrackNotImplementedError,
    deliverable_router,
    validate_routing,
)
from spatial_ingestion.reconstruction.models import ReconstructionJob
from spatial_ingestion.reconstruction.pipeline import _resolve_output_paths
from spatial_ingestion.reconstruction.pipeline import run as run_reconstruction
from spatial_ingestion.refinement import (
    MeshCleaningConfig,
    clean_mesh,
    load_mesh_file,
    to_trimesh,
    write_mesh_file,
)


@dataclass(frozen=True)
class FullPipelineResult:
    pipeline_result: FinalPipelineResult
    deliverable: DeliverableResult


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
    rigged_mesh_path: Path | None = None
    rigging_manifest_path: Path | None = None
    skeleton_path: Path | None = None
    skinning_weights_path: Path | None = None
    rigging_warnings: list[str] | None = None


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

    raw_mesh = load_mesh_file(raw_mesh_path)
    refinement_result = clean_mesh(raw_mesh, refinement_config or MeshCleaningConfig())
    cleaned_mesh = refinement_result["mesh"]

    destination = (
        Path(refined_output_path).expanduser().resolve()
        if refined_output_path
        else _default_refined_path(raw_mesh_path)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_mesh_file(cleaned_mesh, destination)

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


def run_phase2_phase3_phase5_pipeline(
    job: ReconstructionJob,
    refinement_config: MeshCleaningConfig | None = None,
    rigging_config: AutoRigConfig | None = None,
    *,
    refined_output_path: Path | str | None = None,
    rigged_output_path: Path | str | None = None,
    rig_output_dir: Path | str | None = None,
) -> FinalPipelineResult:
    """Run Phase 2 reconstruction, Phase 3 cleanup, and Phase 5 skinned GLB export."""
    result = run_phase2_phase3_pipeline(
        job,
        refinement_config,
        refined_output_path=refined_output_path,
    )
    _, output_dir = _resolve_output_paths(job)
    rig_config = (rigging_config or AutoRigConfig()).model_copy()
    if rig_output_dir is not None:
        rig_config.output_dir = Path(rig_output_dir).expanduser().resolve()
    elif rig_config.output_dir is None:
        rig_config.output_dir = output_dir / "rigging"
    if rigged_output_path is not None:
        rig_config.rigged_output_path = Path(rigged_output_path).expanduser().resolve()

    refined_mesh = load_mesh_file(result.refined_mesh_path)
    rig_input_mesh = to_trimesh(refined_mesh)
    rig_result = AutoRiggingPipeline().rig_mesh(
        rig_input_mesh,
        config=rig_config,
        mesh_uri=result.refined_mesh_path.resolve().as_uri(),
    )
    return FinalPipelineResult(
        job_id=result.job_id,
        raw_mesh_path=result.raw_mesh_path,
        refined_mesh_path=result.refined_mesh_path,
        reconstruction_manifest_path=result.reconstruction_manifest_path,
        refinement_manifest_path=result.refinement_manifest_path,
        refinement_diagnostics=result.refinement_diagnostics,
        rigged_mesh_path=_path_from_uri(rig_result.rigged_mesh_uri),
        rigging_manifest_path=_path_from_uri(rig_result.manifest_uri),
        skeleton_path=_path_from_uri(rig_result.skeleton_uri),
        skinning_weights_path=_path_from_uri(rig_result.weights_uri),
        rigging_warnings=rig_result.warnings,
    )


def run_full_pipeline(
    job: ReconstructionJob,
    use_case: str,
    input_type: str | SourceType,
    refinement_config: MeshCleaningConfig | None = None,
    *,
    refined_output_path: Path | str | None = None,
    deliverables_root: Path | str = DEFAULT_DELIVERABLES_ROOT,
) -> FullPipelineResult:
    """Run Phase 2 -> Phase 3 -> Phase 4 as one command."""
    validate_routing(input_type, use_case)
    if use_case == "live":
        raise TrackNotImplementedError(
            f"[{job.job_id}] Track C (real-time WebRTC/WebSocket delivery) is not implemented yet."
        )

    pipeline_result = run_phase2_phase3_pipeline(
        job, refinement_config, refined_output_path=refined_output_path
    )

    if use_case == "editing":
        refined_mesh = load_mesh_file(pipeline_result.refined_mesh_path)
        mesh = to_trimesh(refined_mesh)
        deliverable = deliverable_router(
            input_type=input_type,
            use_case="editing",
            output_root=deliverables_root,
            mesh=mesh,
        )
    elif use_case == "viewing":
        point_cloud_path = pipeline_result.reconstruction_manifest_path.parent / "point_cloud.ply"
        if not point_cloud_path.exists():
            raise PipelineArtifactError(f"Phase 2 point cloud not found: {point_cloud_path}")
        cloud_mesh = trimesh.load(str(point_cloud_path))
        if not isinstance(cloud_mesh, trimesh.PointCloud):
            raise PipelineArtifactError(
                f"Phase 2 point cloud artifact is not a point cloud: {point_cloud_path}"
            )
        deliverable = deliverable_router(
            input_type=input_type,
            use_case="viewing",
            output_root=deliverables_root,
            point_cloud=cloud_mesh,
        )
    else:
        deliverable = deliverable_router(
            input_type=input_type, use_case=use_case, output_root=deliverables_root
        )

    return FullPipelineResult(pipeline_result=pipeline_result, deliverable=deliverable)


def full_result_to_dict(result: FullPipelineResult) -> dict[str, Any]:
    data = result_to_dict(result.pipeline_result)
    data["deliverable"] = asdict(result.deliverable)
    return data


def result_to_dict(result: FinalPipelineResult) -> dict[str, Any]:
    data = asdict(result)
    for key in (
        "raw_mesh_path",
        "refined_mesh_path",
        "reconstruction_manifest_path",
        "refinement_manifest_path",
        "rigged_mesh_path",
        "rigging_manifest_path",
        "skeleton_path",
        "skinning_weights_path",
    ):
        data[key] = str(data[key]) if data[key] is not None else None
    return data


def _default_refined_path(raw_mesh_path: Path) -> Path:
    suffix = raw_mesh_path.suffix or ".glb"
    return raw_mesh_path.with_name(f"{raw_mesh_path.stem}_refined{suffix}")


def _validate_raw_mesh(raw_mesh_path: Path) -> None:
    if not raw_mesh_path.exists():
        raise PipelineArtifactError(
            f"Phase 2 completed but no mesh artifact was produced: {raw_mesh_path}"
        )
    if raw_mesh_path.stat().st_size == 0:
        raise PipelineArtifactError(f"Phase 2 mesh artifact is empty: {raw_mesh_path}")
    if raw_mesh_path.suffix.lower() not in {".obj", ".glb", ".ply", ".stl", ".vtk", ".vtp"}:
        raise PipelineArtifactError(
            f"Unsupported Phase 2 mesh artifact format: {raw_mesh_path.suffix}"
        )


def _serializable_diagnostics(refinement_result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in refinement_result.items() if key != "mesh"}


def _path_from_uri(uri: str | None) -> Path | None:
    if uri is None:
        return None
    from spatial_ingestion.reconstruction._io import uri_to_path

    return uri_to_path(uri)
