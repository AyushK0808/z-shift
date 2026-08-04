from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np


def to_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return value


def to_serializable_array(value: Any) -> Any:
    value = to_array(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [to_serializable_array(item) for item in value]
    return value


def flatten_rows(value: Any) -> list[list[float]]:
    value = to_array(value)
    if isinstance(value, (list, tuple)):
        parts = [np.asarray(v, dtype=float) for v in value]
        if parts and parts[0].ndim == 2:
            array = np.concatenate(parts, axis=0)
        elif parts:
            array = np.concatenate(parts)
        else:
            return []
    else:
        array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        return [array.tolist()]
    if array.ndim == 2:
        return array.tolist()
    if array.ndim >= 3:
        return array.reshape(-1, array.shape[-1]).tolist()
    return []


def scale_rgb_to_byte(values: list[float]) -> tuple[int, int, int]:
    clipped = np.clip(np.asarray(values[:3], dtype=float), 0.0, 1.0)
    scaled = np.rint(clipped * 255.0).astype(int)
    return int(scaled[0]), int(scaled[1]), int(scaled[2])


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_ply(path: Path, points: Any, colors: Any) -> None:
    xyz_rows = flatten_rows(points)
    rgb_rows = flatten_rows(colors)
    row_count = min(len(xyz_rows), len(rgb_rows))

    xyz = np.asarray(xyz_rows[:row_count], dtype=np.float32).reshape(row_count, 3)
    rgb = np.asarray(rgb_rows[:row_count], dtype=float).reshape(row_count, 3)
    if row_count and np.nanmax(rgb) <= 1.0:
        rgb = np.clip(np.rint(rgb[:, :3] * 255.0), 0.0, 255.0).astype(np.uint8, copy=False)
    else:
        rgb = np.clip(rgb[:, :3], 0.0, 255.0).astype(np.uint8, copy=False)

    vertex_data = np.empty(
        row_count,
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    if row_count:
        vertex_data["x"] = xyz[:, 0]
        vertex_data["y"] = xyz[:, 1]
        vertex_data["z"] = xyz[:, 2]
        vertex_data["red"] = rgb[:, 0]
        vertex_data["green"] = rgb[:, 1]
        vertex_data["blue"] = rgb[:, 2]

    header = (
        "\n".join(
            [
                "ply",
                "format binary_little_endian 1.0",
                f"element vertex {row_count}",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
            ]
        )
        + "\n"
    )

    with path.open("wb") as file_handle:
        file_handle.write(header.encode("ascii"))
        file_handle.write(vertex_data.tobytes())


def uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme in {"", "file"}:
        candidate = unquote(parsed.path if parsed.scheme == "file" else uri)
        if parsed.scheme == "file" and _is_windows_drive_netloc(parsed.netloc):
            candidate = parsed.netloc + candidate
        if parsed.scheme == "file" and _is_windows_drive_path(candidate):
            candidate = candidate[1:]
        return Path(candidate).expanduser().resolve()
    return Path(uri).expanduser().resolve()


def _is_windows_drive_path(path: str) -> bool:
    return len(path) >= 3 and path[0] == "/" and path[1].isalpha() and path[2] == ":"


def _is_windows_drive_netloc(netloc: str) -> bool:
    return len(netloc) == 2 and netloc[0].isalpha() and netloc[1] == ":"
