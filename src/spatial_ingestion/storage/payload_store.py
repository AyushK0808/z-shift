"""Persist Phase 1 gateway payloads so downstream phases can consume them.

The ingestion gateway returns the ``UnifiedSpatialIngestionSchema`` as its HTTP
response but, historically, never wrote it anywhere. This store gives the
gateway an automatic handoff point: every accepted upload is persisted as JSON
under ``data/payloads/`` and the payload's ``metadata["payload_uri"]`` points at
the file — so the final pipeline can be run with ``--from-schema`` directly,
and the persisted file is self-describing (it carries its own ``payload_uri``).

Payloads are retained in the gitignored ``data/`` scratch tree for handoff;
there is deliberately no GC/retention policy — they are small JSON files and
cleanup is left to whoever runs the prototype.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from spatial_ingestion.config import PAYLOAD_OUTPUT_ROOT
from spatial_ingestion.metadata.schema import UnifiedSpatialIngestionSchema


class PayloadStore:
    def __init__(self, root: Path = PAYLOAD_OUTPUT_ROOT) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def store(self, payload: UnifiedSpatialIngestionSchema) -> Path:
        """Stamp ``payload_uri``, write the payload as JSON, return the path."""
        path = self._root / f"{payload.source_type.value}_{uuid4().hex}.json"
        payload.metadata["payload_uri"] = path.as_uri()
        path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
        return path
