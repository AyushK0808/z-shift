from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_model_cache: dict[str, object] = {}

try:
    from mast3r.model import AsymmetricMASt3R
except ImportError:
    AsymmetricMASt3R: type[Any] | None = None

try:
    from dust3r.utils.image import load_images as dust3r_load_images
except ImportError:
    dust3r_load_images: Callable[..., Any] | None = None


def load_model(model_name: str, device: str) -> object:
    cached = _model_cache.get(model_name)
    if cached is not None:
        return cached

    if AsymmetricMASt3R is None:
        raise RuntimeError(
            "MASt3R is not installed. Run scripts/setup-mast3r.sh or "
            "pip install -e third_party/mast3r"
        )

    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    model.eval()
    _model_cache[model_name] = model
    return model


def load_images(image_paths: list[Path], image_size: int = 512) -> list[dict]:
    if dust3r_load_images is None:
        raise RuntimeError(
            "MASt3R (dust3r) is not installed. Run scripts/setup-mast3r.sh or "
            "pip install -e third_party/mast3r/dust3r"
        )

    return dust3r_load_images([str(p) for p in image_paths], size=image_size)
