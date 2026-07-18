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

    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {row_count}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for xyz, rgb in zip(xyz_rows[:row_count], rgb_rows[:row_count], strict=False):
        red, green, blue = scale_rgb_to_byte(rgb)
        lines.append(f"{float(xyz[0])} {float(xyz[1])} {float(xyz[2])} {red} {green} {blue}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme in {"", "file"}:
        candidate = unquote(parsed.path if parsed.scheme == "file" else uri)
        return Path(candidate).expanduser().resolve()
    return Path(uri).expanduser().resolve()


def uri_to_path_or_none(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme in {"", "file"}:
        candidate = unquote(parsed.path if parsed.scheme == "file" else uri)
        return Path(candidate)
    return Path(uri)
