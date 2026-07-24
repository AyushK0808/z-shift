from __future__ import annotations

import logging
from pathlib import Path

import torch

from spatial_ingestion.metadata.schema import CameraIntrinsics
from spatial_ingestion.reconstruction.models import HandoffFrame, SyncViewGroup
from spatial_ingestion.reconstruction._io import uri_to_path

logger = logging.getLogger(__name__)


def build_pairs(
    images: list[dict],
    strategy: str = "complete",
) -> list[tuple[dict, dict]]:
    try:
        from dust3r.image_pairs import make_pairs
    except ImportError as exc:
        raise RuntimeError("MASt3R (dust3r) is not installed.") from exc

    pairs = make_pairs(images, scene_graph=strategy, symmetrize=True)
    return pairs


def build_sync_pairs(
    image_paths: list[Path],
    sync_view_groups: list[SyncViewGroup],
) -> list[tuple[int, int]]:
    stem_to_idx: dict[str, int] = {}
    for i, p in enumerate(image_paths):
        s = p.stem
        if s in stem_to_idx:
            logger.warning(
                "Duplicate stem '%s' at index %d conflicts with index %d",
                s, i, stem_to_idx[s],
            )
        stem_to_idx.setdefault(s, i)

    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    for group in sync_view_groups:
        cameras = sorted(group.frames_by_source.keys())
        indices: list[int] = []
        for source_id in cameras:
            handoff = group.frames_by_source[source_id]
            path_stem = uri_to_path(handoff.uri).stem
            idx = stem_to_idx.get(path_stem)
            if idx is not None:
                indices.append(idx)

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a, b = indices[i], indices[j]
                if (a, b) not in seen:
                    pairs.append((a, b))
                    pairs.append((b, a))
                    seen.add((a, b))
                    seen.add((b, a))

    return pairs


def intrinsics_to_k_matrix(intrinsics: CameraIntrinsics, img_path: Path) -> torch.Tensor | None:
    if intrinsics.focal_length_35mm is None:
        return None
    try:
        from PIL import Image
        with Image.open(img_path) as pil_img:
            w, h = pil_img.size
    except Exception:
        return None
    focal_px = intrinsics.focal_length_35mm / 36.0 * max(w, h)
    cx, cy = w / 2, h / 2
    K = torch.eye(3)
    K[0, 0] = K[1, 1] = focal_px
    K[0, 2] = cx
    K[1, 2] = cy
    return K
