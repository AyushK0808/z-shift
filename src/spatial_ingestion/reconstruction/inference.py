from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_model_cache: dict[str, object] = {}

try:
    from mast3r.model import AsymmetricMASt3R
    _MAST3R_MODEL_AVAILABLE = True
except ImportError:
    AsymmetricMASt3R = None  # type: ignore[assignment]
    _MAST3R_MODEL_AVAILABLE = False

try:
    from dust3r.utils.image import load_images as dust3r_load_images
    _DUST3R_LOAD_AVAILABLE = True
except ImportError:
    dust3r_load_images = None  # type: ignore[assignment]
    _DUST3R_LOAD_AVAILABLE = False

def load_model(model_name: str, device: str) -> object:
    cached = _model_cache.get(model_name)
    if cached is not None:
        return cached

    if not _MAST3R_MODEL_AVAILABLE:
        raise RuntimeError(
            "MASt3R is not installed. Run scripts/setup-mast3r.sh or "
            "pip install -e third_party/mast3r"
        )

    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    model.eval()
    _model_cache[model_name] = model
    return model


def load_images(image_paths: list[Path], image_size: int = 512) -> list[dict]:
    if not _DUST3R_LOAD_AVAILABLE:
        raise RuntimeError(
            "MASt3R (dust3r) is not installed. Run scripts/setup-mast3r.sh or "
            "pip install -e third_party/mast3r/dust3r"
        )

    return dust3r_load_images([str(p) for p in image_paths], size=image_size)
