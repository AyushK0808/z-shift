import json
import struct
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pytest
import trimesh
from pydantic import ValidationError

from spatial_ingestion.auto_rigging import (
    ArticulationType,
    AutoRigConfig,
    AutoRiggingPipeline,
    GltfSkinPayloadBuilder,
    Joint,
    RiggedMesh,
    Skeleton,
    SkinningWeights,
)
from spatial_ingestion.auto_rigging.cli import main as auto_rig_main
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
    pipeline = AutoRiggingPipeline(exporter=RigMetadataExporter(output_root=tmp_path / "rigs"))

    result = pipeline.rig_mesh(
        mesh,
        config=AutoRigConfig(articulation_type=ArticulationType.BIPED),
    )

    assert result.rigged_mesh.skeleton.articulation_type == ArticulationType.BIPED
    assert result.rigged_mesh.vertex_count == len(mesh.vertices)
    assert result.skeleton_uri is not None
    assert result.weights_uri is not None
    assert result.rigged_mesh_uri is not None
    assert result.manifest_uri is not None
    assert _path_from_file_uri(result.skeleton_uri).exists()
    assert _path_from_file_uri(result.weights_uri).exists()
    assert _path_from_file_uri(result.rigged_mesh_uri).exists()
    assert _path_from_file_uri(result.manifest_uri).exists()
    assert result.warnings
    assert result.rigged_mesh.metadata["template_axis_convention"] == (
        "Y-up height, X forward/width, Z lateral/depth"
    )


def test_auto_rigging_config_output_dir_controls_export_location(tmp_path: Path) -> None:
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    configured_output = tmp_path / "configured-rigs"

    result = AutoRiggingPipeline().rig_mesh(
        mesh,
        config=AutoRigConfig(
            articulation_type=ArticulationType.STATIC,
            output_dir=configured_output,
        ),
    )

    assert result.skeleton_uri is not None
    assert result.weights_uri is not None
    assert configured_output in _path_from_file_uri(result.skeleton_uri).parents
    assert configured_output in _path_from_file_uri(result.weights_uri).parents


def test_auto_rigging_pipeline_accepts_mesh_file(tmp_path: Path) -> None:
    mesh_path = tmp_path / "mesh.obj"
    trimesh.creation.box().export(mesh_path)
    pipeline = AutoRiggingPipeline(exporter=RigMetadataExporter(output_root=tmp_path / "rigs"))

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


def test_auto_rigging_cli_runs_on_mesh_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mesh_path = tmp_path / "mesh.obj"
    trimesh.creation.box().export(mesh_path)

    exit_code = auto_rig_main([str(mesh_path), "--articulation", "biped", "--no-export"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"articulation_type": "biped"' in captured.out


def test_auto_rigging_cli_honors_output_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mesh_path = tmp_path / "mesh.obj"
    output_dir = tmp_path / "cli-rigs"
    trimesh.creation.box().export(mesh_path)

    exit_code = auto_rig_main(
        [
            str(mesh_path),
            "--articulation",
            "static",
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert output_dir in _path_from_file_uri(payload["skeleton_uri"]).parents
    assert output_dir in _path_from_file_uri(payload["weights_uri"]).parents
    assert output_dir in _path_from_file_uri(payload["rigged_mesh_uri"]).parents


def test_auto_rigging_cli_honors_explicit_rigged_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mesh_path = tmp_path / "mesh.obj"
    rigged_output = tmp_path / "paper" / "rigged.glb"
    trimesh.creation.box().export(mesh_path)

    exit_code = auto_rig_main(
        [
            str(mesh_path),
            "--articulation",
            "static",
            "--rigged-output",
            str(rigged_output),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert _path_from_file_uri(payload["rigged_mesh_uri"]) == rigged_output
    assert rigged_output.exists()


def test_gltf_skin_payload_builder_packs_four_influences() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    result = AutoRiggingPipeline().rig_mesh(
        mesh,
        config=AutoRigConfig(articulation_type=ArticulationType.BIPED),
        export_metadata=False,
    )

    payload = GltfSkinPayloadBuilder().build(result.rigged_mesh)

    assert payload.joint_names == result.rigged_mesh.skinning.joint_names
    assert len(payload.joints_0) == result.rigged_mesh.vertex_count
    assert len(payload.weights_0) == result.rigged_mesh.vertex_count
    assert all(len(row) == 4 for row in payload.joints_0)
    assert all(len(row) == 4 for row in payload.weights_0)
    assert all(sum(row) == pytest.approx(1.0) for row in payload.weights_0)


def test_gltf_skin_payload_builder_rejects_empty_influence_rows() -> None:
    skeleton = Skeleton(
        articulation_type=ArticulationType.STATIC,
        joints=[Joint(name="root", position=(0.0, 0.0, 0.0))],
        bones=[],
        root_joint="root",
    )
    rigged_mesh = RiggedMesh(
        vertex_count=1,
        face_count=0,
        skeleton=skeleton,
        skinning=SkinningWeights(
            joint_names=["root"],
            weights=[[0.0]],
            max_influences=1,
        ),
    )

    with pytest.raises(ValueError, match="at least one positive influence"):
        GltfSkinPayloadBuilder().build(rigged_mesh)


def test_skinned_glb_export_contains_skin_payload(tmp_path: Path) -> None:
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    output_path = tmp_path / "rigged.glb"

    result = AutoRiggingPipeline().rig_mesh(
        mesh,
        config=AutoRigConfig(
            articulation_type=ArticulationType.BIPED,
            rigged_output_path=output_path,
        ),
    )

    gltf = _read_glb_json(output_path)
    primitive = gltf["meshes"][0]["primitives"][0]
    attributes = primitive["attributes"]

    assert result.rigged_mesh_uri == output_path.as_uri()
    assert "skins" in gltf
    assert len(gltf["skins"][0]["joints"]) == len(result.rigged_mesh.skeleton.joints)
    assert "JOINTS_0" in attributes
    assert "WEIGHTS_0" in attributes
    assert gltf["accessors"][attributes["POSITION"]]["count"] == len(mesh.vertices)
    assert gltf["accessors"][primitive["indices"]]["count"] == len(mesh.faces.reshape(-1))


# ---------------------------------------------------------------------------
# Template geometry and the structural invariants every articulation must hold
# ---------------------------------------------------------------------------


def test_template_skeleton_fitter_places_biped_joints_on_the_mesh() -> None:
    """The biped template is fitted, never asserted -- pin where its joints land.

    A wrong up-axis or a swapped width axis still produces a valid Skeleton, so
    only the positions catch it.
    """
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 0.5))
    skeleton = TemplateSkeletonFitter().fit(mesh, ArticulationType.BIPED)
    joints = {joint.name: joint.position for joint in skeleton.joints}
    min_y, max_y = float(mesh.bounds[0][1]), float(mesh.bounds[1][1])

    assert skeleton.root_joint == "hips"
    assert len(skeleton.joints) == 7
    assert len(skeleton.bones) == 6
    assert joints["left_foot"][1] == pytest.approx(min_y)
    assert joints["right_foot"][1] == pytest.approx(min_y)
    assert joints["hips"][1] < joints["spine"][1] < joints["head"][1] < max_y
    assert joints["left_hand"][0] < joints["right_hand"][0]


@pytest.mark.parametrize("articulation", list(ArticulationType))
def test_every_template_is_a_tree_rooted_at_its_root_joint(
    articulation: ArticulationType,
) -> None:
    mesh = trimesh.creation.box(extents=(2.0, 1.5, 1.0))
    skeleton = TemplateSkeletonFitter().fit(mesh, articulation)
    parents = {joint.name: joint.parent for joint in skeleton.joints}
    names = set(parents)

    assert skeleton.root_joint in names
    assert parents[skeleton.root_joint] is None
    for bone in skeleton.bones:
        assert bone.parent_joint in names
        assert bone.child_joint in names
    for name in names:
        walker, seen = name, set()
        while parents[walker] is not None:
            assert walker not in seen, f"cycle through {walker}"
            seen.add(walker)
            walker = parents[walker]
            assert walker in names
        assert walker == skeleton.root_joint


# ---------------------------------------------------------------------------
# Mesh preparation: what the template is actually fitted to
# ---------------------------------------------------------------------------


def test_normalize_mesh_centres_and_rescales_before_the_template_is_fitted() -> None:
    """A reconstruction arrives at arbitrary scale and offset.

    Normalisation puts the longest axis at 1.0 centred on the origin, so a
    4 x 2 x 1 box parked away from the origin becomes 1 x 0.5 x 0.25 and the
    feet land at y = -0.25.
    """
    mesh = trimesh.creation.box(extents=(4.0, 2.0, 1.0))
    mesh.apply_translation((10.0, -3.0, 7.0))

    result = AutoRiggingPipeline().rig_mesh(
        mesh,
        config=AutoRigConfig(articulation_type=ArticulationType.BIPED),
        export_metadata=False,
    )
    joints = {joint.name: joint.position for joint in result.rigged_mesh.skeleton.joints}

    assert joints["left_foot"][1] == pytest.approx(-0.25)
    assert joints["hips"][0] == pytest.approx(0.0)
    assert joints["hips"][2] == pytest.approx(0.0)


def test_normalize_mesh_false_fits_the_template_where_the_mesh_actually_is() -> None:
    mesh = trimesh.creation.box(extents=(4.0, 2.0, 1.0))
    mesh.apply_translation((10.0, -3.0, 7.0))

    result = AutoRiggingPipeline().rig_mesh(
        mesh,
        config=AutoRigConfig(
            articulation_type=ArticulationType.STATIC,
            normalize_mesh=False,
        ),
        export_metadata=False,
    )

    assert result.rigged_mesh.skeleton.joints[0].position == pytest.approx((10.0, -3.0, 7.0))


def test_rigging_an_empty_mesh_is_rejected() -> None:
    with pytest.raises(ValueError, match="must contain vertices and faces"):
        AutoRiggingPipeline().rig_mesh(
            trimesh.Trimesh(vertices=[], faces=[]),
            export_metadata=False,
        )


# ---------------------------------------------------------------------------
# The static template, which is what an object capture (a tank, a bulldozer)
# gets from B1. Every other test here fits a biped or a quadruped.
# ---------------------------------------------------------------------------


def test_static_rig_binds_every_vertex_to_its_single_joint() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)

    result = AutoRiggingPipeline().rig_mesh(
        mesh,
        config=AutoRigConfig(articulation_type=ArticulationType.STATIC),
        export_metadata=False,
    )
    skinning = result.rigged_mesh.skinning

    assert skinning.joint_names == ["root"]
    assert skinning.max_influences == 1
    assert len(skinning.weights) == len(mesh.vertices)
    assert all(row == [pytest.approx(1.0)] for row in skinning.weights)


def test_static_rig_exports_a_glb_carrying_a_one_joint_skin(tmp_path: Path) -> None:
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    output_path = tmp_path / "static_rigged.glb"

    result = AutoRiggingPipeline().rig_mesh(
        mesh,
        config=AutoRigConfig(
            articulation_type=ArticulationType.STATIC,
            output_dir=tmp_path / "rigs",
            rigged_output_path=output_path,
        ),
    )
    gltf = _read_glb_json(output_path)
    attributes = gltf["meshes"][0]["primitives"][0]["attributes"]

    assert result.rigged_mesh_uri == output_path.as_uri()
    assert len(gltf["skins"][0]["joints"]) == 1
    assert "JOINTS_0" in attributes
    assert "WEIGHTS_0" in attributes


# ---------------------------------------------------------------------------
# Skinning knobs and reproducibility
# ---------------------------------------------------------------------------


def test_a_single_influence_gives_one_hot_rows() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    skeleton = TemplateSkeletonFitter().fit(mesh, ArticulationType.BIPED)

    skinning = InverseDistanceSkinner().compute(mesh, skeleton, max_influences=1)

    assert skinning.max_influences == 1
    for row in skinning.weights:
        assert sum(weight > 0 for weight in row) == 1
        assert sum(row) == pytest.approx(1.0)


def test_max_influences_is_clamped_to_the_joint_count_and_reported() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    skeleton = TemplateSkeletonFitter().fit(mesh, ArticulationType.STATIC)

    skinning = InverseDistanceSkinner().compute(mesh, skeleton, max_influences=8)

    assert skinning.max_influences == 1
    assert all(len(row) == 1 for row in skinning.weights)


def test_rigging_the_same_mesh_twice_gives_the_same_rig() -> None:
    """§V-D claims reproducibility, and the rig is part of what has to reproduce."""
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    config = AutoRigConfig(articulation_type=ArticulationType.QUADRUPED)

    first = AutoRiggingPipeline().rig_mesh(mesh, config=config, export_metadata=False)
    second = AutoRiggingPipeline().rig_mesh(mesh, config=config, export_metadata=False)

    assert first.rigged_mesh.skeleton == second.rigged_mesh.skeleton
    assert first.rigged_mesh.skinning.weights == second.rigged_mesh.skinning.weights


def test_the_articulation_choice_survives_into_the_written_manifest(tmp_path: Path) -> None:
    """B1 now passes `--articulation` through; the artifact has to record which
    template was fitted, or the manual Blender check is uninterpretable."""
    mesh = trimesh.creation.box(extents=(3.0, 1.0, 1.2))

    result = AutoRiggingPipeline().rig_mesh(
        mesh,
        config=AutoRigConfig(
            articulation_type=ArticulationType.QUADRUPED,
            output_dir=tmp_path / "rigs",
        ),
    )
    assert result.manifest_uri is not None
    manifest = json.loads(_path_from_file_uri(result.manifest_uri).read_text(encoding="utf-8"))

    assert manifest["rigged_mesh"]["metadata"]["articulation_type"] == "quadruped"
    assert manifest["rigged_mesh"]["skeleton"]["articulation_type"] == "quadruped"
    assert manifest["rigged_mesh"]["skeleton"]["root_joint"] == "body"


def test_skeleton_rejects_duplicate_joint_names() -> None:
    with pytest.raises(ValidationError, match="joint names must be unique"):
        Skeleton(
            articulation_type=ArticulationType.STATIC,
            joints=[
                Joint(name="root", position=(0.0, 0.0, 0.0)),
                Joint(name="root", position=(1.0, 1.0, 1.0)),
            ],
            bones=[],
            root_joint="root",
        )


def _path_from_file_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    raw = unquote(parsed.path)
    if len(raw) >= 4 and raw[0] == "/" and raw[2] == ":":
        raw = raw.lstrip("/")
    return Path(raw)


def _read_glb_json(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    magic, version, _ = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67
    assert version == 2
    json_length, chunk_type = struct.unpack_from("<I4s", data, 12)
    assert chunk_type == b"JSON"
    return json.loads(data[20 : 20 + json_length].decode("utf-8"))
