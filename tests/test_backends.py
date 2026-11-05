"""The reconstruction stage's backend seam: any backend ends in the same artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import trimesh

from spatial_ingestion.reconstruction import backends
from spatial_ingestion.reconstruction._io import write_json, write_ply
from spatial_ingestion.reconstruction.backends import (
    DUST3R_MODEL_NAME,
    ReconstructionBackend,
    available_backends,
    get_backend,
    register_backend,
)
from spatial_ingestion.reconstruction.cli import build_parser
from spatial_ingestion.reconstruction.config import DEFAULT_MODEL_NAME
from spatial_ingestion.reconstruction.models import (
    Mast3rRunParams,
    ReconstructionJob,
    ReconstructionMode,
)
from spatial_ingestion.reconstruction.pipeline import run
from spatial_ingestion.refinement import MeshCleaningConfig, clean_mesh, load_mesh_file


class FakeBackend(ReconstructionBackend):
    """No network: writes the stage's artifacts for a sphere."""

    name = "fake"
    default_model = "fake/model"

    def align(self, **kwargs: Any) -> object:
        kwargs["stats"].update(
            {"n_images": len(kwargs["image_paths"]), "n_pairs": 0, "pairing_strategy_used": "fake"}
        )
        return {"model": kwargs["model_name"]}

    def export(
        self,
        scene: object,
        output_path: Path,
        output_dir: Path,
        *,
        tsdf_thresh: float,
        min_conf_thr: float,
        stats: dict[str, Any],
    ) -> bool:
        sphere = trimesh.creation.icosphere(subdivisions=3)
        sphere.export(str(output_path))
        write_ply(
            output_dir / "point_cloud.ply",
            np.asarray(sphere.vertices),
            np.full((len(sphere.vertices), 3), 128, dtype=np.uint8),
        )
        write_json(output_dir / "cameras.json", {"cameras": []})
        stats["scene_model"] = scene["model"]  # ty: ignore[not-subscriptable]
        return False


@pytest.fixture
def fake_backend():
    backend = FakeBackend()
    register_backend(backend)
    yield backend
    backends._BACKENDS.pop(backend.name)
    backends._ALL_DEFAULT_MODELS.discard(backend.default_model)


def _job(tmp_path: Path, backend_name: str, **params: Any) -> ReconstructionJob:
    return ReconstructionJob(
        mode=ReconstructionMode.MULTI_VIEW,
        image_uris=[(tmp_path / f"v{i}.jpg").as_uri() for i in range(2)],
        backend_name=backend_name,
        output_path=str(tmp_path / "out" / "mesh.glb"),
        metadata=Mast3rRunParams(device="cpu", **params).model_dump(),
    )


def _manifest(output_dir: Path) -> dict[str, Any]:
    return json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))


def test_both_backends_are_registered() -> None:
    assert {"mast3r", "dust3r"} <= set(available_backends())
    assert get_backend("dust3r").default_model == DUST3R_MODEL_NAME
    with pytest.raises(ValueError, match="Unknown reconstruction backend"):
        get_backend("nope")


def test_a_backend_gets_its_own_checkpoint_unless_one_is_named() -> None:
    mast3r, dust3r = get_backend("mast3r"), get_backend("dust3r")
    assert dust3r.resolve_model(DEFAULT_MODEL_NAME) == DUST3R_MODEL_NAME
    assert mast3r.resolve_model(DEFAULT_MODEL_NAME) == DEFAULT_MODEL_NAME
    assert dust3r.resolve_model("/local/checkpoint.pth") == "/local/checkpoint.pth"


def test_a_swapped_backend_writes_the_same_artifacts_and_refinement_consumes_them(
    tmp_path: Path, fake_backend: FakeBackend
) -> None:
    assert run(_job(tmp_path, "fake")) == 0

    output_dir = tmp_path / "out"
    for artifact in ("mesh.glb", "point_cloud.ply", "cameras.json", "run_manifest.json"):
        assert (output_dir / artifact).exists(), artifact
    manifest = _manifest(output_dir)
    assert manifest["backend"] == "fake"
    assert manifest["model_name"] == "fake/model"
    assert manifest["outputs"]["scene_model"] == "fake/model"
    assert [stage["stage"] for stage in manifest["stage_timings"]] == [
        "sparse_alignment",
        "export_scene_to_mesh",
    ]

    # Downstream sees only the artifact.
    refined = clean_mesh(load_mesh_file(output_dir / "mesh.glb"), MeshCleaningConfig(device="cpu"))
    assert refined["output_cell_count"] > 0


def test_an_unknown_backend_fails_before_any_work(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown reconstruction backend"):
        run(_job(tmp_path, "nope"))


def test_dry_run_records_the_backend(tmp_path: Path) -> None:
    assert run(_job(tmp_path, "dust3r", dry_run=True)) == 0
    manifest = _manifest(tmp_path / "out")
    assert manifest["backend"] == "dust3r"
    assert manifest["model_name"] == DUST3R_MODEL_NAME


def test_the_cli_selects_a_backend() -> None:
    assert build_parser().parse_args(["images/", "--backend", "dust3r"]).backend == "dust3r"
    assert build_parser().parse_args(["images/"]).backend == "mast3r"
