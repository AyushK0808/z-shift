"""Phase 4 — Outcomes & Deliverables Engine.

Routes a cleaned Phase 3 artifact (mesh or point cloud) to the correct
export/delivery pipeline based on the declared use case, and returns a
structured result instead of printing to stdout.

NOTE: `get_phase3_cleaned_mesh` / `get_phase3_point_cloud` below are still
in-memory mocks of the real Phase 3 handoff (tracked separately — de-mocking
is blocked on the pipeline executor landing, see ROADMAP §1·X1 and §5). They
let the router, validation, and export logic be developed and tested without
running the upstream models.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from spatial_ingestion.metadata.schema import SourceType

# ---------------------------------------------------------------------------
# 1. Mocked Phase 3 inputs (temporary — see module docstring)
# ---------------------------------------------------------------------------


def get_phase3_cleaned_mesh() -> trimesh.Trimesh:
    """Simulates Phase 3 handing over a cleaned, unformatted 3D mesh."""
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    mesh.visual.vertex_colors = [100, 150, 255, 255]
    return mesh


def get_phase3_point_cloud() -> trimesh.PointCloud:
    """Simulates Phase 3 handing over raw point/splat-center data.

    NOTE: this is a plain colored point cloud, not real 4D Gaussian splats
    (no covariances, no spherical-harmonic coefficients, no opacity, no time
    dimension). See `export_point_cloud` below.
    """
    points = np.random.rand(10000, 3) * 10
    colors = np.random.randint(0, 255, (10000, 4))
    return trimesh.PointCloud(vertices=points, colors=colors)


# ---------------------------------------------------------------------------
# 2. Result types & errors
# ---------------------------------------------------------------------------


class InvalidRoutingError(ValueError):
    """Raised when (input_type, use_case) is not a supported combination."""


class TrackNotImplementedError(NotImplementedError):
    """Raised when a valid routing decision targets an unbuilt track."""


@dataclass(frozen=True)
class DeliverableResult:
    job_id: str
    track: str  # "A" (editing/blender), "B" (viewing/point-cloud), "C" (live)
    input_type: str
    use_case: str
    output_path: Optional[str]
    message: str


# ---------------------------------------------------------------------------
# 3. Output location
# ---------------------------------------------------------------------------

# Deliverables previously landed in the source tree (only `data/` is
# gitignored), which made it easy to accidentally commit binary artifacts.
# Default to <repo_root>/data/deliverables instead; callers (e.g. tests) can
# still override via `output_root`.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DELIVERABLES_ROOT = _REPO_ROOT / "data" / "deliverables"


def _deliverable_dir(output_root: Path, *parts: str) -> Path:
    path = output_root.joinpath(*parts)
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 4. Packaging & export pipelines
# ---------------------------------------------------------------------------


def export_blender_ready(mesh_data: trimesh.Trimesh, job_id: str, output_root: Path) -> str:
    """Converts raw mesh data to standard interchange formats (e.g. .glb)."""
    output_dir = _deliverable_dir(output_root, "blender_ready")
    file_path = output_dir / f"{job_id}_model.glb"
    mesh_data.export(str(file_path))
    return str(file_path)


def export_point_cloud(point_cloud_data: trimesh.PointCloud, job_id: str, output_root: Path) -> str:
    """Bundles point/splat-center data into a `.ply`.

    Renamed from `package_4d_gaussian`: the previous name claimed a 4D
    Gaussian-splat deliverable this function does not produce (no
    covariances, SH coefficients, opacity, or time dimension — just points +
    RGBA). Rename back once real Gaussian-splat export exists.
    """
    output_dir = _deliverable_dir(output_root, "point_clouds")
    file_path = output_dir / f"{job_id}_points.ply"
    point_cloud_data.export(str(file_path))
    return str(file_path)


# ---------------------------------------------------------------------------
# 5. Routing rules
# ---------------------------------------------------------------------------

# Which SourceTypes each use_case is actually valid for. Explicit, so a
# mismatch (e.g. live_stream + editing) is rejected instead of silently
# falling through.
_EDITING_INPUT_TYPES = {
    SourceType.SINGLE_IMAGE,
    SourceType.IMAGE_FOLDER,
    SourceType.SINGLE_VIDEO,
    SourceType.VIDEO_FOLDER,
}
_VIEWING_INPUT_TYPES = {
    SourceType.SINGLE_VIDEO,
    SourceType.VIDEO_FOLDER,
    SourceType.IMAGE_FOLDER,
}
_LIVE_INPUT_TYPES = {SourceType.LIVE_STREAM}


def _coerce_source_type(input_type: str | SourceType) -> SourceType:
    if isinstance(input_type, SourceType):
        return input_type
    try:
        return SourceType(input_type)
    except ValueError as exc:
        valid = ", ".join(t.value for t in SourceType)
        raise InvalidRoutingError(
            f"Unknown input_type '{input_type}'. Expected one of: {valid}."
        ) from exc


# ---------------------------------------------------------------------------
# 6. Deliverable router
# ---------------------------------------------------------------------------


def deliverable_router(
    input_type: str | SourceType,
    use_case: str,
    output_root: Path | str = DEFAULT_DELIVERABLES_ROOT,
) -> DeliverableResult:
    """Routes and packages Phase 3 output based on the declared use case.

    Raises `InvalidRoutingError` for an unsupported (input_type, use_case)
    combination, and `TrackNotImplementedError` for a valid combination that
    targets a track not yet built (currently: Track C / live delivery).
    Returns a `DeliverableResult` on success — nothing is printed.
    """
    source_type = _coerce_source_type(input_type)
    output_root = Path(output_root)
    job_id = f"JOB_{uuid.uuid4().hex[:6].upper()}"

    if use_case == "editing":
        if source_type not in _EDITING_INPUT_TYPES:
            raise InvalidRoutingError(
                f"'editing' is not valid for input_type '{source_type.value}' "
                f"(valid: {sorted(t.value for t in _EDITING_INPUT_TYPES)})."
            )
        raw_mesh = get_phase3_cleaned_mesh()
        final_file = export_blender_ready(raw_mesh, job_id, output_root)
        return DeliverableResult(
            job_id=job_id,
            track="A",
            input_type=source_type.value,
            use_case=use_case,
            output_path=final_file,
            message="Blender-ready export packaged successfully.",
        )

    if use_case == "viewing":
        if source_type not in _VIEWING_INPUT_TYPES:
            raise InvalidRoutingError(
                f"'viewing' is not valid for input_type '{source_type.value}' "
                f"(valid: {sorted(t.value for t in _VIEWING_INPUT_TYPES)})."
            )
        raw_cloud = get_phase3_point_cloud()
        final_file = export_point_cloud(raw_cloud, job_id, output_root)
        return DeliverableResult(
            job_id=job_id,
            track="B",
            input_type=source_type.value,
            use_case=use_case,
            output_path=final_file,
            message="Point-cloud deliverable packaged successfully.",
        )

    if use_case == "live":
        if source_type not in _LIVE_INPUT_TYPES:
            raise InvalidRoutingError(
                f"'live' is not valid for input_type '{source_type.value}' "
                f"(valid: {sorted(t.value for t in _LIVE_INPUT_TYPES)})."
            )
        # Honesty item: no WebRTC/WebSocket delivery layer exists yet. Raise
        # instead of claiming a stream was established.
        raise TrackNotImplementedError(
            f"[{job_id}] Track C (real-time WebRTC/WebSocket delivery) is not "
            "implemented yet."
        )

    raise InvalidRoutingError(
        f"Unknown use_case '{use_case}'. Expected one of: 'editing', 'viewing', 'live'."
    )
