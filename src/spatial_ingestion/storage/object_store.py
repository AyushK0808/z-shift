from __future__ import annotations

import shutil
from urllib.parse import unquote, urlparse
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

    def delete_uri(self, uri: str) -> None:
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            return
        path = Path(unquote(parsed.path.lstrip("/")))
        if not path.is_absolute():
            path = Path(parsed.path)
        try:
            resolved = path.resolve()
            root = self._root.resolve()
            if root in resolved.parents or resolved == root:
                resolved.unlink(missing_ok=True)
        except OSError:
            return
