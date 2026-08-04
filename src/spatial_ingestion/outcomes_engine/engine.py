"""Phase 4 — Outcomes & Deliverables Engine.

Routes a cleaned Phase 3 artifact (mesh or point cloud) to the correct
export/delivery pipeline based on the declared use case, and returns a
structured result instead of printing to stdout.

The router expects the actual Phase 3 artifact (a `trimesh.Trimesh` for
`editing`, a `trimesh.PointCloud` for `viewing`) — it no longer falls back to
fabricated geometry. Callers that want to exercise routing without running the
upstream models must supply their own synthetic artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import trimesh

from spatial_ingestion.config import DELIVERABLES_OUTPUT_ROOT
from spatial_ingestion.metadata.schema import SourceType

DEFAULT_DELIVERABLES_ROOT = DELIVERABLES_OUTPUT_ROOT

# ---------------------------------------------------------------------------
# 1. Result types & errors
# ---------------------------------------------------------------------------


class InvalidRoutingError(ValueError):
    """Raised when (source_type, use_case) is not a supported combination."""


class TrackNotImplementedError(NotImplementedError):
    """Raised when a valid routing decision targets an unbuilt track."""


@dataclass(frozen=True)
class DeliverableResult:
    job_id: str
    track: str  # the deliverable lane: "editing", "viewing", or "live"
    source_type: str
    use_case: str
    output_path: str | None
    message: str


# ---------------------------------------------------------------------------
# 2. Output location
# ---------------------------------------------------------------------------

# Deliverables previously landed in the source tree (only `data/` is
# gitignored), which made it easy to accidentally commit binary artifacts.
# The canonical root lives in `spatial_ingestion.config`; callers (e.g. tests)
# can still override via `output_root`.


def _deliverable_dir(output_root: Path, *parts: str) -> Path:
    path = output_root.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 3. Packaging & export pipelines
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
# falling through. SINGLE_IMAGE is deliberately absent from every set: the
# pipeline rejects single-view jobs before Phase 2, so no deliverable can
# ever be produced from one.
_EDITING_SOURCE_TYPES = {
    SourceType.IMAGE_FOLDER,
    SourceType.SINGLE_VIDEO,
    SourceType.VIDEO_FOLDER,
}
_VIEWING_SOURCE_TYPES = {
    SourceType.SINGLE_VIDEO,
    SourceType.VIDEO_FOLDER,
    SourceType.IMAGE_FOLDER,
}
_LIVE_SOURCE_TYPES = {SourceType.LIVE_STREAM}


def _coerce_source_type(source_type: str | SourceType) -> SourceType:
    if isinstance(source_type, SourceType):
        return source_type
    try:
        return SourceType(source_type)
    except ValueError as exc:
        valid = ", ".join(t.value for t in SourceType)
        raise InvalidRoutingError(
            f"Unknown source_type '{source_type}'. Expected one of: {valid}."
        ) from exc


def validate_routing(source_type: str | SourceType, use_case: str) -> SourceType:
    """Validate a Phase 4 routing decision without exporting anything."""
    source_type = _coerce_source_type(source_type)

    if use_case == "editing":
        if source_type not in _EDITING_SOURCE_TYPES:
            raise InvalidRoutingError(
                f"'editing' is not valid for source_type '{source_type.value}' "
                f"(valid: {sorted(t.value for t in _EDITING_SOURCE_TYPES)})."
            )
        return source_type

    if use_case == "viewing":
        if source_type not in _VIEWING_SOURCE_TYPES:
            raise InvalidRoutingError(
                f"'viewing' is not valid for source_type '{source_type.value}' "
                f"(valid: {sorted(t.value for t in _VIEWING_SOURCE_TYPES)})."
            )
        return source_type

    if use_case == "live":
        if source_type not in _LIVE_SOURCE_TYPES:
            raise InvalidRoutingError(
                f"'live' is not valid for source_type '{source_type.value}' "
                f"(valid: {sorted(t.value for t in _LIVE_SOURCE_TYPES)})."
            )
        return source_type

    raise InvalidRoutingError(
        f"Unknown use_case '{use_case}'. Expected one of: 'editing', 'viewing', 'live'."
    )


# ---------------------------------------------------------------------------
# 6. Deliverable router
# ---------------------------------------------------------------------------


def deliverable_router(
    source_type: str | SourceType,
    use_case: str,
    output_root: Path | str = DEFAULT_DELIVERABLES_ROOT,
    job_id: str | None = None,
    mesh: trimesh.Trimesh | None = None,
    point_cloud: trimesh.PointCloud | None = None,
) -> DeliverableResult:
    """Routes and packages a Phase 3 artifact based on the declared use case.

    `job_id` is the reconstruction job id from Phase 2; when omitted a fresh
    value is generated. The packaged deliverable shares this id so folders,
    manifests, and deliverables are traceable to the same run.
    """
    source_type = validate_routing(source_type, use_case)
    output_root = Path(output_root)
    resolved_job_id = job_id or uuid4().hex[:12]

    if use_case == "live":
        raise TrackNotImplementedError(
            f"[{resolved_job_id}] live delivery (real-time WebRTC/WebSocket) "
            "is not implemented yet."
        )

    if use_case == "editing":
        if mesh is None:
            raise ValueError(
                "'editing' requires a Phase 3 cleaned mesh artifact; pass one via `mesh=`."
            )
        final_file = export_blender_ready(mesh, resolved_job_id, output_root)
        return DeliverableResult(
            job_id=resolved_job_id,
            track="editing",
            source_type=source_type.value,
            use_case=use_case,
            output_path=final_file,
            message="Blender-ready export packaged successfully.",
        )

    if use_case == "viewing":
        if point_cloud is None:
            raise ValueError(
                "'viewing' requires a Phase 3 point-cloud artifact; pass one via `point_cloud=`."
            )
        final_file = export_point_cloud(point_cloud, resolved_job_id, output_root)
        return DeliverableResult(
            job_id=resolved_job_id,
            track="viewing",
            source_type=source_type.value,
            use_case=use_case,
            output_path=final_file,
            message="Point-cloud deliverable packaged successfully.",
        )

    # Defensive: `validate_routing` should have raised for unknown use_cases,
    # and all valid use_cases either return a DeliverableResult or raise.
    # Add an explicit raise to make the control flow obvious to static
    # analyzers and satisfy the type checker.
    raise InvalidRoutingError(f"Unhandled use_case '{use_case}'")
