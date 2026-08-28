"""A9 - routing validation completeness matrix.

Supports the SIV-E claim that "any other combination is rejected with an
explicit routing error rather than silently falling through to a default".
Enumerates every (SourceType, use_case) cell exhaustively, including an
invalid use_case string, so the claim becomes a table instead of an assertion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from bench.csvio import ResultWriter
from bench.harness import experiment_parser, finish
from spatial_ingestion.metadata.schema import SourceType
from spatial_ingestion.outcomes_engine.engine import (
    InvalidRoutingError,
    TrackNotImplementedError,
    deliverable_router,
    validate_routing,
)

EXP_ID = "a9_routing_matrix"
USE_CASES: tuple[str, ...] = ("editing", "viewing", "live", "not_a_use_case")

logger = logging.getLogger(__name__)


def _classify(source_type: SourceType, use_case: str) -> dict[str, Any]:
    """Record what `validate_routing` does, and what the router does after it."""
    row: dict[str, Any] = {
        "source_type": source_type.value,
        "use_case": use_case,
        "validate_outcome": "",
        "validate_error": "",
        "router_outcome": "",
        "router_error": "",
    }
    try:
        validate_routing(source_type, use_case)
        row["validate_outcome"] = "accept"
    except InvalidRoutingError as exc:
        row["validate_outcome"] = "InvalidRoutingError"
        row["validate_error"] = str(exc)
    except Exception as exc:  # noqa: BLE001 - an unexpected type is itself the finding
        row["validate_outcome"] = f"UNEXPECTED:{type(exc).__name__}"
        row["validate_error"] = str(exc)

    # The router is exercised separately because validation passing does not
    # mean a deliverable exists: `live` validates and then raises.
    try:
        deliverable_router(source_type, use_case)
        row["router_outcome"] = "deliverable"
    except InvalidRoutingError as exc:
        row["router_outcome"] = "InvalidRoutingError"
        row["router_error"] = str(exc)
    except TrackNotImplementedError as exc:
        row["router_outcome"] = "TrackNotImplementedError"
        row["router_error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        row["router_outcome"] = f"UNEXPECTED:{type(exc).__name__}"
        row["router_error"] = str(exc)
    return row


def run(results_dir: Path | None = None, *, quick: bool = False, seed: int = 0) -> ResultWriter:
    """Enumerate the full matrix.

    `quick` and `seed` are accepted for a uniform experiment signature but do
    nothing: the matrix is exhaustive and deterministic, so there is no smaller
    grid to fall back to and nothing to seed.
    """
    del quick, seed
    writer = ResultWriter(EXP_ID, results_dir)
    for source_type in SourceType:
        for use_case in USE_CASES:
            writer.add(**_classify(source_type, use_case))
    return writer


def main(argv: list[str] | None = None) -> int:
    args = experiment_parser(EXP_ID, __doc__ or "").parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    writer = run(results_dir=args.results_dir, quick=args.quick, seed=args.seed)
    finish(writer)

    accepted = sum(1 for row in writer.rows if row["router_outcome"] == "deliverable")
    rejected = sum(1 for row in writer.rows if row["router_outcome"].endswith("Error"))
    unexpected = [row for row in writer.rows if "UNEXPECTED" in row["router_outcome"]]
    print(f"  cells={len(writer)} deliverable={accepted} rejected={rejected}")
    if unexpected:
        print(f"  UNEXPECTED outcomes in {len(unexpected)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
