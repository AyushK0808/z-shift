from __future__ import annotations

import logging

import numpy as np

from spatial_ingestion.reconstruction.config import MAST3R_COMMIT

logger = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def memory_summary(device: str) -> str:
    try:
        import torch
    except ImportError:
        return "unknown"

    if device == "cuda" and torch.cuda.is_available():
        i = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(i)
        free_gb = free / 1024**3
        total_gb = total / 1024**3
        return f"{total_gb:.1f}GB total, {free_gb:.1f}GB free"

    return "n/a"


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def reproducibility_metadata() -> dict[str, object]:
    meta: dict[str, object] = {}
    try:
        import torch
        meta["torch_version"] = torch.__version__
        meta["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            meta["cuda_version"] = torch.version.cuda
            meta["cuda_device"] = torch.cuda.get_device_name(0)
        meta["mps_available"] = torch.backends.mps.is_available()
    except ImportError:
        meta["torch_version"] = None
    try:
        meta["numpy_version"] = np.__version__
    except AttributeError:
        pass
    meta["mast3r_commit"] = MAST3R_COMMIT
    return meta
