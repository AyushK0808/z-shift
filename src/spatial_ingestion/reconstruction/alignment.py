from __future__ import annotations

import logging
from pathlib import Path

from spatial_ingestion.reconstruction.inference import load_images, load_model
from spatial_ingestion.reconstruction.models import HandoffFrame, SyncViewGroup
from spatial_ingestion.reconstruction.pairing import (
    build_pairs,
    build_sync_pairs,
    intrinsics_to_k_matrix,
)

logger = logging.getLogger(__name__)

try:
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
except ImportError:
    sparse_global_alignment = None


def run_sparse_alignment(
    *,
    image_paths: list[Path],
    output_dir: Path,
    model_name: str,
    device: str,
    image_size: int = 512,
    pairing_strategy: str = "complete",
    sync_view_groups: list[SyncViewGroup] | None = None,
    frames: list[HandoffFrame] | None = None,
) -> object:
    if sparse_global_alignment is None:
        raise RuntimeError(
            "MASt3R is not installed. Run scripts/setup-mast3r.sh or "
            "pip install -e third_party/mast3r && pip install -e third_party/mast3r/dust3r"
        )

    model = load_model(model_name, device)
    images = load_images(image_paths, image_size=image_size)

    if sync_view_groups:
        idx_pairs = build_sync_pairs(image_paths, sync_view_groups)
        if idx_pairs:
            pairs = [(images[a], images[b]) for a, b in idx_pairs]
        else:
            logger.warning(
                "Sync-aware pairing produced no pairs, falling back to %s",
                pairing_strategy,
            )
            pairs = build_pairs(images, strategy=pairing_strategy)
    else:
        pairs = build_pairs(images, strategy=pairing_strategy)

    cache_path = str((output_dir / "cache").resolve())
    str_paths = [str(path) for path in image_paths]
    init: dict[str, dict[str, object]] = {}
    if frames:
        for img_path, frame in zip(image_paths, frames, strict=False):
            if frame.camera_intrinsics:
                K = intrinsics_to_k_matrix(frame.camera_intrinsics, img_path)
                if K is not None:
                    init[str(img_path)] = {"intrinsics": K}

    return sparse_global_alignment(
        str_paths,
        pairs,
        cache_path=cache_path,
        model=model,
        device=device,
        init=init,
    )
