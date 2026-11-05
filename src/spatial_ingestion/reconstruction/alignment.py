from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from spatial_ingestion.reconstruction.inference import load_images, load_model
from spatial_ingestion.reconstruction.models import HandoffFrame, SyncViewGroup
from spatial_ingestion.reconstruction.pairing import (
    build_pairs,
    build_sync_pairs,
    intrinsics_to_k_matrix,
)

logger = logging.getLogger(__name__)

_mast3r_import_error: ImportError | None = None
try:
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
except ImportError as exc:
    sparse_global_alignment: Callable[..., Any] | None = None
    _mast3r_import_error = exc


def select_pairs(
    image_paths: list[Path],
    images: list[dict],
    pairing_strategy: str,
    sync_view_groups: list[SyncViewGroup] | None = None,
    *,
    stats: dict[str, Any] | None = None,
) -> list[tuple[dict, dict]]:
    """The image pairs a backend's network is run over, shared by every backend.

    Sync-aware pairing when view groups are given and yield pairs, otherwise the
    named scene graph. `stats`, when given, is populated in place with the
    pairing counters (`n_images`, `n_pairs`, `pairing_strategy_used`) the run
    manifest needs.
    """
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

    if stats is not None:
        stats.update(
            {
                "n_images": len(images),
                "n_pairs": len(pairs),
                "pairing_strategy_used": (
                    "sync" if sync_view_groups and len(pairs) else pairing_strategy
                ),
            }
        )
    return pairs


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
    stats: dict[str, Any] | None = None,
) -> object:
    """Run MASt3R sparse global alignment over `image_paths`.

    `stats`, when given, is populated in place with the pairing counters
    (`n_images`, `n_pairs`, `pairing_strategy_used`) the run manifest needs;
    an out-parameter keeps the scene the sole return value for callers that
    do not care.
    """
    if sparse_global_alignment is None:
        raise RuntimeError(
            "MASt3R is not installed or failed to import (see the chained "
            "exception above for the real cause). Run scripts/setup-mast3r.sh "
            "or pip install -e third_party/mast3r && "
            "pip install -e third_party/mast3r/dust3r"
        ) from _mast3r_import_error

    model = load_model(model_name, device)
    images = load_images(image_paths, image_size=image_size)
    pairs = select_pairs(image_paths, images, pairing_strategy, sync_view_groups, stats=stats)

    cache_path = str((output_dir / "cache").resolve())
    str_paths = [str(path) for path in image_paths]
    init: dict[str, dict[str, object]] = {}
    if frames:
        for img_path, frame in zip(image_paths, frames, strict=False):
            if frame.camera_intrinsics:
                K = intrinsics_to_k_matrix(frame.camera_intrinsics, img_path)
                if K is not None:
                    init[str(img_path)] = {"intrinsics": K}

    with _model_offloaded_after_forward(model, device):
        return sparse_global_alignment(
            str_paths,
            pairs,
            cache_path=cache_path,
            model=model,
            device=device,
            init=init,
        )


@contextmanager
def _model_offloaded_after_forward(model: Any, device: str) -> Iterator[None]:
    """Move the network off the GPU once every pair has been through it.

    sparse_global_alignment runs the network over all pairs first, caching the
    outputs to disk, and never touches it again. Its scene optimizer holds the
    correspondences of every pair on the GPU (about 6 MB a pair at 512 px)
    while the ~2.7 GB of ViT-L weights sit there unused. Freeing them is the
    difference between fitting ~450 pairs on an 8 GB card and ~900.
    """
    if not str(device).startswith("cuda") or not hasattr(model, "to"):
        yield
        return
    import torch
    from mast3r.cloud_opt import sparse_ga

    original = sparse_ga.forward_mast3r

    def forward_then_offload(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        model.to("cpu")
        torch.cuda.empty_cache()
        return result

    sparse_ga.forward_mast3r = forward_then_offload
    try:
        yield
    finally:
        sparse_ga.forward_mast3r = original
        # load_model caches this instance for the next run.
        model.to(device)
