from __future__ import annotations

import logging
from pathlib import Path

from spatial_ingestion.config import RECONSTRUCTION_OUTPUT_ROOT
from spatial_ingestion.reconstruction._io import write_json
from spatial_ingestion.reconstruction.alignment import run_sparse_alignment
from spatial_ingestion.reconstruction.device import resolve_device, set_seed
from spatial_ingestion.reconstruction.export import (
    build_run_manifest,
    export_scene_to_mesh,
)
from spatial_ingestion.reconstruction.models import Mast3rRunParams, ReconstructionJob

logger = logging.getLogger(__name__)


def run(job: ReconstructionJob) -> int:
    params = _resolve_params(job)
    image_paths = [Path(u).expanduser().resolve() for u in job.image_uris]

    if len(image_paths) < 2:
        raise ValueError("MASt3R reconstruction requires at least two images")

    output_path, output_dir = _resolve_output_paths(job)
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

    if params.dry_run:
        write_json(output_dir / "run_manifest.json", manifest)
        return 0

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
    write_json(output_dir / "run_manifest.json", manifest)

    if not output_path.exists():
        logger.warning("Expected output artifact not found: %s", output_path)
    return 0


def _resolve_params(job: ReconstructionJob) -> Mast3rRunParams:
    return Mast3rRunParams.model_validate(job.metadata or {})


def _resolve_output_paths(job: ReconstructionJob) -> tuple[Path, Path]:
    if job.output_path:
        output_path = Path(job.output_path).resolve()
        output_dir = output_path.parent
    else:
        label = job.label or "reconstruction"
        output_dir = RECONSTRUCTION_OUTPUT_ROOT / f"{label}_{job.job_id}"
        output_path = output_dir / f"{label}.obj"
    return output_path, output_dir


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
    return manifest
