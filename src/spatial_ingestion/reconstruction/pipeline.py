from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from spatial_ingestion.reconstruction.alignment import run_sparse_alignment
from spatial_ingestion.reconstruction.config import (
    POINT_CLOUD_FILENAME,
    RUN_MANIFEST_FILENAME,
)
from spatial_ingestion.reconstruction.device import resolve_device, set_seed
from spatial_ingestion.reconstruction.export import (
    build_run_manifest,
    export_scene_to_mesh,
    validate_mesh_format,
)
from spatial_ingestion.reconstruction.io import uri_to_path, write_json
from spatial_ingestion.reconstruction.models import Mast3rRunParams, ReconstructionJob
from spatial_ingestion.reconstruction.paths import resolve_output_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReconstructionRunResult:
    """Where a completed Phase 2 run left its artifacts."""

    job_id: str
    mode: str
    output_dir: Path
    output_path: Path
    point_cloud_path: Path
    manifest_path: Path
    dry_run: bool


def run(job: ReconstructionJob) -> ReconstructionRunResult:
    params = job.params
    image_paths = [uri_to_path(u) for u in job.image_uris]

    if len(image_paths) < 2:
        raise ValueError("MASt3R reconstruction requires at least two images")

    output_path, output_dir = resolve_output_paths(job)
    validate_mesh_format(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(params.device)
    if params.seed is not None:
        set_seed(params.seed)

    manifest = _build_manifest(
        image_paths=image_paths,
        output_dir=output_dir,
        output_path=output_path,
        params=params,
        device=device,
        job=job,
    )
    manifest_path = output_dir / RUN_MANIFEST_FILENAME
    point_cloud_path = output_dir / POINT_CLOUD_FILENAME

    if params.dry_run:
        write_json(manifest_path, manifest)
        return ReconstructionRunResult(
            job_id=job.job_id,
            mode=job.mode.value,
            output_dir=output_dir,
            output_path=output_path,
            point_cloud_path=point_cloud_path,
            manifest_path=manifest_path,
            dry_run=True,
        )

    sparse_scene = run_sparse_alignment(
        image_paths=image_paths,
        output_dir=output_dir,
        model_name=params.model_name,
        device=device,
        image_size=params.image_size,
        pairing_strategy=params.pairing_strategy,
        sync_view_groups=job.sync_view_groups or None,
        frames=job.frames or None,
    )
    tsdf_fell_back = export_scene_to_mesh(
        sparse_scene,
        output_path,
        output_dir,
        tsdf_thresh=params.tsdf_thresh,
        min_conf_thr=params.min_conf_thr,
    )
    manifest["tsdf_fallback"] = tsdf_fell_back
    write_json(manifest_path, manifest)

    if not output_path.exists():
        logger.warning("Expected output artifact not found: %s", output_path)
    return ReconstructionRunResult(
        job_id=job.job_id,
        mode=job.mode.value,
        output_dir=output_dir,
        output_path=output_path,
        point_cloud_path=point_cloud_path,
        manifest_path=manifest_path,
        dry_run=False,
    )


def resolve_output_paths(job: ReconstructionJob) -> tuple[Path, Path]:
    """Resolve (output_path, output_dir) for a job.

    A job with an explicit ``output_path`` (always set by the CLIs and
    ``final_pipeline.build_job``) writes exactly there; otherwise the shared
    job-id-aware resolver picks ``data/reconstruction/<label>_<job_id>/
    <label>.glb``.
    """
    if job.output_path:
        output_path = Path(job.output_path).resolve()
        return output_path, output_path.parent
    output_path = resolve_output_path(
        None, None, label=job.label or "reconstruction", job_id=job.job_id
    )
    return output_path, output_path.parent


def _build_manifest(
    *,
    image_paths: list[Path],
    output_dir: Path,
    output_path: Path,
    params: Mast3rRunParams,
    device: str,
    job: ReconstructionJob,
) -> dict[str, object]:
    manifest = build_run_manifest(
        image_paths=image_paths,
        output_dir=output_dir,
        output_path=output_path,
        model_name=params.model_name,
        device=device,
        image_size=params.image_size,
        pairing_strategy=params.pairing_strategy,
        tsdf_thresh=params.tsdf_thresh,
        min_conf_thr=params.min_conf_thr,
        seed=params.seed,
        dry_run=params.dry_run,
        sync_view_groups=job.sync_view_groups or None,
    )
    manifest["job_id"] = job.job_id
    manifest["mode"] = job.mode.value
    manifest["label"] = job.label
    manifest["provenance"] = dict(job.metadata)
    return manifest
