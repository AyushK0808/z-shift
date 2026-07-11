from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from spatial_ingestion.config import LOCAL_STORAGE_ROOT


class ObjectStore:
    """S3-compatible storage boundary; local disk implementation for Phase 1."""

    def __init__(self, root: Path = LOCAL_STORAGE_ROOT) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def put_file(self, source_path: Path, namespace: str) -> str:
        target_dir = self._root / namespace
        target_dir.mkdir(parents=True, exist_ok=True)
        key = f"{uuid4().hex}_{source_path.name}"
        target = target_dir / key
        shutil.copy2(source_path, target)
        return target.as_uri()

