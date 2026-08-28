"""Result-row accumulation and CSV emission.

Every experiment produces exactly one CSV under ``bench/results/``. Rows carry
the environment inline (``env_*`` columns) so a number can never be separated
from the machine, library versions and commit that produced it.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench import RESULTS_DIR
from bench.instrument import env_metadata

__all__ = ["ResultWriter", "aggregate", "mean_std", "read_rows"]

Row = dict[str, Any]


class ResultWriter:
    """Accumulates result rows and writes them to ``results/<exp_id>.csv``.

    Columns are the union of every row's keys, so an experiment can record an
    extra field for a subset of its cells (a timeout reason, a fallback flag)
    without pre-declaring a schema.
    """

    def __init__(
        self,
        exp_id: str,
        results_dir: Path | str | None = None,
        *,
        env: dict[str, Any] | None = None,
    ) -> None:
        self.exp_id = exp_id
        self.results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
        self.env = env if env is not None else env_metadata()
        self.rows: list[Row] = []
        self._started = datetime.now(UTC).isoformat(timespec="seconds")

    def add(self, **row: Any) -> Row:
        record: Row = {"exp_id": self.exp_id, "row_index": len(self.rows), **row}
        self.rows.append(record)
        return record

    def extend(self, rows: Iterable[Row]) -> None:
        for row in rows:
            self.add(**row)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def path(self) -> Path:
        return self.results_dir / f"{self.exp_id}.csv"

    def write(self) -> Path:
        env_columns = {f"env_{key}": value for key, value in self.env.items()}
        env_columns["env_run_started_utc"] = self._started
        full_rows = [{**row, **env_columns} for row in self.rows]

        fieldnames: list[str] = []
        for row in full_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

        self.results_dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in full_rows:
                writer.writerow(row)
        return self.path


def read_rows(exp_id_or_path: str | Path, results_dir: Path | str | None = None) -> list[Row]:
    """Read a results CSV back, coercing numeric-looking cells to floats."""
    path = Path(exp_id_or_path)
    if path.suffix != ".csv":
        base = Path(results_dir) if results_dir is not None else RESULTS_DIR
        path = base / f"{exp_id_or_path}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: _coerce(value) for key, value in row.items()} for row in csv.DictReader(handle)
        ]


def _coerce(value: str | None) -> Any:
    if value is None or value == "":
        return None
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        number = float(value)
    except ValueError:
        return value
    looks_integral = number.is_integer() and "." not in value and "e" not in lowered
    return int(number) if looks_integral else number


def mean_std(values: Sequence[float]) -> tuple[float, float, int]:
    """Sample mean, sample standard deviation (ddof=1), and n.

    ddof=1 because these are repeated measurements of a process, not a
    population census; with n=3 the difference is not cosmetic.
    """
    clean = [float(v) for v in values if v is not None and not _is_nan(v)]
    n = len(clean)
    if n == 0:
        return (float("nan"), float("nan"), 0)
    mean = sum(clean) / n
    if n == 1:
        return (mean, 0.0, 1)
    variance = sum((v - mean) ** 2 for v in clean) / (n - 1)
    return (mean, math.sqrt(variance), n)


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def aggregate(
    rows: Iterable[Row], group_by: Sequence[str], value_key: str
) -> dict[tuple[Any, ...], tuple[float, float, int]]:
    """Group rows by `group_by` and reduce `value_key` to (mean, std, n)."""
    buckets: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        value = row.get(value_key)
        if value is None or _is_nan(value):
            continue
        key = tuple(row.get(column) for column in group_by)
        buckets.setdefault(key, []).append(float(value))
    return {key: mean_std(values) for key, values in sorted(buckets.items(), key=repr)}
