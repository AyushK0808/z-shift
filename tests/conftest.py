"""Shared fixtures for the z-shift test suite."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pytest
import pyvista as pv

from spatial_ingestion.batch_normalization.image_processor import ImageProcessor
from spatial_ingestion.batch_normalization.normalizer import BatchNormalizer
from spatial_ingestion.reconstruction.config import (
    POINT_CLOUD_FILENAME,
    RUN_MANIFEST_FILENAME,
)
from spatial_ingestion.reconstruction.models import ReconstructionJob
from spatial_ingestion.reconstruction.pipeline import ReconstructionRunResult
from spatial_ingestion.test_harness.media_factory import create_sample_image

MeshWriter = Callable[[Path], None]


@pytest.fixture
def ingest_folder(tmp_path: Path) -> Path:
    """A two-view image folder: ``front.jpg`` + ``side.jpg``."""
    folder = tmp_path / "views"
    folder.mkdir()
    create_sample_image(folder / "front.jpg")
    create_sample_image(folder / "side.jpg")
    return folder


@pytest.fixture
def normalizer(tmp_path: Path) -> BatchNormalizer:
    """A Phase 1 batch normalizer writing normalized PNGs under tmp_path."""
    return BatchNormalizer(image_processor=ImageProcessor(output_root=tmp_path / "normalized"))


@pytest.fixture
def fake_reconstruction(monkeypatch: pytest.MonkeyPatch):
    """Patch ``final_pipeline.core.run_reconstruction`` with a fake Phase 2.

    Returns ``install(...)``; options:
      - ``raw_mesh_path``: fixed mesh destination (defaults to ``job.output_path``)
      - ``write_mesh``: callable(raw_path) invoked with the mesh destination,
        or ``False`` to leave the mesh unwritten
      - ``check_image_uris``: assert every ``job.image_uris`` resolves to a file
    """

    def install(
        *,
        raw_mesh_path: Path | None = None,
        write_mesh: MeshWriter | Literal[False] | None = None,
        check_image_uris: bool = False,
    ) -> None:
        if write_mesh is None:

            def write_sphere(path: Path) -> None:
                pv.Sphere(theta_resolution=16, phi_resolution=16).save(str(path))

            write_mesh = write_sphere

        def fake(job: ReconstructionJob) -> ReconstructionRunResult:
            if check_image_uris:
                from spatial_ingestion.reconstruction.io import uri_to_path

                assert all(Path(uri_to_path(uri)).exists() for uri in job.image_uris)
            destination = (
                Path(job.output_path) if job.output_path is not None else Path(raw_mesh_path or ".")
            )
            raw = destination.resolve()
            raw.parent.mkdir(parents=True, exist_ok=True)
            if write_mesh is not False:
                write_mesh(raw)
            return ReconstructionRunResult(
                job_id=job.job_id,
                mode=job.mode.value,
                output_dir=raw.parent,
                output_path=raw,
                point_cloud_path=raw.parent / POINT_CLOUD_FILENAME,
                manifest_path=raw.parent / RUN_MANIFEST_FILENAME,
                dry_run=False,
            )

        monkeypatch.setattr("spatial_ingestion.final_pipeline.core.run_reconstruction", fake)

    return install
