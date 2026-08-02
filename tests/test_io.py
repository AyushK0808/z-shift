import json
from pathlib import Path

import numpy as np
import pytest

from spatial_ingestion.reconstruction._io import (
    flatten_rows,
    scale_rgb_to_byte,
    to_array,
    to_serializable_array,
    write_json,
    write_ply,
)


def test_to_array_passes_numpy_through() -> None:
    arr = np.array([1.0, 2.0, 3.0])
    assert to_array(arr) is arr


def test_to_array_passes_list_through() -> None:
    result = to_array([1.0, 2.0, 3.0])
    assert result == [1.0, 2.0, 3.0]


def test_to_serializable_array_converts_ndarray() -> None:
    result = to_serializable_array(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert result == [[1.0, 2.0], [3.0, 4.0]]


def test_to_serializable_array_passes_list_through() -> None:
    result = to_serializable_array([1, 2, 3])
    assert result == [1, 2, 3]


def test_flatten_rows_2d_array() -> None:
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert flatten_rows(arr) == [[1.0, 2.0], [3.0, 4.0]]


def test_flatten_rows_3d_array() -> None:
    arr = np.ones((2, 3, 3))
    result = flatten_rows(arr)
    assert len(result) == 6
    assert result[0] == [1.0, 1.0, 1.0]


def test_flatten_rows_1d_array() -> None:
    arr = np.array([1.0, 2.0, 3.0])
    assert flatten_rows(arr) == [[1.0, 2.0, 3.0]]


def test_scale_rgb_to_byte_clamps_and_scales() -> None:
    assert scale_rgb_to_byte([0.0, 0.5, 1.0]) == (0, 128, 255)


def test_scale_rgb_to_byte_clips_out_of_range() -> None:
    assert scale_rgb_to_byte([-0.5, 1.5, 0.0]) == (0, 255, 0)


def test_scale_rgb_to_byte_ignores_extra_channels() -> None:
    assert scale_rgb_to_byte([0.2, 0.4, 0.6, 0.8, 0.9]) == (51, 102, 153)


def test_write_json_writes_indented_with_newline(tmp_path: Path) -> None:
    path = tmp_path / "test.json"
    write_json(path, {"a": 1, "b": 2})
    content = path.read_text(encoding="utf-8")
    assert json.loads(content) == {"a": 1, "b": 2}
    assert content.endswith("\n")


def test_write_ply_writes_correct_header(tmp_path: Path) -> None:
    path = tmp_path / "test.ply"
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    colors = np.array([[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]])
    write_ply(path, points, colors)
    content = path.read_bytes()
    assert content.startswith(b"ply\nformat binary_little_endian 1.0\n")
    assert b"element vertex 2" in content
    assert b"end_header\n" in content

    trimesh = pytest.importorskip("trimesh")
    loaded = trimesh.load(str(path))
    assert len(loaded.vertices) == 2
    assert loaded.vertices[1][0] == pytest.approx(1.0)
