from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest
import trimesh

from spatial_ingestion.auto_rigging import (
    ArticulationType,
    AutoRigConfig,
    AutoRiggingPipeline,
)
from spatial_ingestion.auto_rigging.export import RigMetadataExporter
from spatial_ingestion.auto_rigging.skeleton.templates import TemplateSkeletonFitter
from spatial_ingestion.auto_rigging.skinning.inverse_distance import InverseDistanceSkinner
from spatial_ingestion.reconstruction.models import ReconstructionArtifactKind


def test_template_skeleton_fitter_creates_static_root() -> None:
    mesh = trimesh.creation.box(extents=(2.0, 1.0, 1.0))
    skeleton = TemplateSkeletonFitter().fit(mesh, ArticulationType.STATIC)

    assert skeleton.root_joint == "root"
    assert len(skeleton.joints) == 1
    assert skeleton.bones == []


def test_template_skeleton_fitter_creates_quadruped_tree() -> None:
    mesh = trimesh.creation.box(extents=(3.0, 1.0, 1.2))
    skeleton = TemplateSkeletonFitter().fit(mesh, ArticulationType.QUADRUPED)

    assert skeleton.root_joint == "body"
    assert len(skeleton.joints) == 7
    assert len(skeleton.bones) == 6
    assert {joint.name for joint in skeleton.joints} >= {
        "front_left_foot",
        "front_right_foot",
        "back_left_foot",
        "back_right_foot",
    }


def test_inverse_distance_skinning_rows_are_normalized_and_sparse() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    skeleton = TemplateSkeletonFitter().fit(mesh, ArticulationType.BIPED)
    skinning = InverseDistanceSkinner().compute(mesh, skeleton, max_influences=4)

    assert skinning.joint_names == [joint.name for joint in skeleton.joints]
    assert len(skinning.weights) == len(mesh.vertices)
    for row in skinning.weights:
        assert sum(weight > 0 for weight in row) <= 4
        assert sum(row) == pytest.approx(1.0)


def test_auto_rigging_pipeline_exports_metadata(tmp_path: Path) -> None:
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    pipeline = AutoRiggingPipeline(
        exporter=RigMetadataExporter(output_root=tmp_path / "rigs")
    )

    result = pipeline.rig_mesh(
        mesh,
        config=AutoRigConfig(articulation_type=ArticulationType.BIPED),
    )

    assert result.rigged_mesh.skeleton.articulation_type == ArticulationType.BIPED
    assert result.rigged_mesh.vertex_count == len(mesh.vertices)
    assert result.skeleton_uri is not None
    assert result.weights_uri is not None
    assert _path_from_file_uri(result.skeleton_uri).exists()
    assert _path_from_file_uri(result.weights_uri).exists()


def test_auto_rigging_pipeline_accepts_mesh_file(tmp_path: Path) -> None:
    mesh_path = tmp_path / "mesh.obj"
    trimesh.creation.box().export(mesh_path)
    pipeline = AutoRiggingPipeline(
        exporter=RigMetadataExporter(output_root=tmp_path / "rigs")
    )

    result = pipeline.rig_mesh_file(
        mesh_path,
        config=AutoRigConfig(articulation_type=ArticulationType.STATIC),
        export_metadata=False,
    )

    assert result.rigged_mesh.mesh_uri == mesh_path.resolve().as_uri()
    assert result.rigged_mesh.skeleton.root_joint == "root"


def test_reconstruction_artifact_kinds_include_phase5_outputs() -> None:
    assert ReconstructionArtifactKind.RIGGED_MESH.value == "rigged_mesh"
    assert ReconstructionArtifactKind.SKELETON.value == "skeleton"
    assert ReconstructionArtifactKind.SKINNING_WEIGHTS.value == "skinning_weights"


def _path_from_file_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    raw = unquote(parsed.path)
    if len(raw) >= 4 and raw[0] == "/" and raw[2] == ":":
        raw = raw.lstrip("/")
    return Path(raw)
