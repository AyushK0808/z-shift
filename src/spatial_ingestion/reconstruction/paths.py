from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from spatial_ingestion.config import RECONSTRUCTION_OUTPUT_ROOT
from spatial_ingestion.reconstruction.export import SUPPORTED_MESH_FORMATS


def resolve_output_path(
    input_path: Path | None,
    explicit_output: str | None,
    label: str | None = None,
    job_id: str | None = None,
) -> Path:
    """Resolve where a reconstruction job's raw mesh should be written.

    The resolved path always embeds the job id in its parent folder, so the
    output folder, run manifest, and reconstruction job share one id.
    """
    resolved_job_id = job_id or uuid4().hex[:12]

    if explicit_output:
        output_path = Path(explicit_output).expanduser().resolve()
        suffix = output_path.suffix.lower()
        if suffix in SUPPORTED_MESH_FORMATS:
            stem = output_path.stem
            return output_path.parent / f"{stem}_{resolved_job_id}" / output_path.name
        if suffix:
            raise ValueError(
                f"Unsupported Phase 2 mesh format '{output_path.suffix}'. "
                f"Phase 2 can only write {', '.join(sorted(SUPPORTED_MESH_FORMATS))}."
            )
        return output_path / f"mesh_{resolved_job_id}.glb"

    if input_path is None:
        stem = label or "reconstruction"
    else:
        stem = label or (input_path.stem if input_path.is_file() else input_path.name)
    return RECONSTRUCTION_OUTPUT_ROOT / f"{stem}_{resolved_job_id}" / f"{stem}.glb"
