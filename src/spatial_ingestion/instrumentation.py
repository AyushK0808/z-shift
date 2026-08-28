"""Stage timing and peak-memory instrumentation shared by the pipelines.

This lives inside the package rather than in ``bench/`` so ``refinement`` and
``reconstruction`` can emit machine-readable timings into their manifests
without importing the benchmark harness, which is a development-only tree and
is not installed alongside the library.

Peak RSS is read from the OS rather than ``tracemalloc`` because the memory
that matters here is allocated by VTK/numpy/torch in native code, which
``tracemalloc`` cannot see. ``resource.getrusage`` is Unix-only, so Windows
goes through ``GetProcessMemoryInfo``.
"""

from __future__ import annotations

import ctypes
import json
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

__all__ = [
    "StageLog",
    "StageRecord",
    "current_rss_bytes",
    "peak_rss_bytes",
    "peak_rss_mb",
]

StageRecord = dict[str, Any]

_BYTES_PER_MB = 1024.0 * 1024.0


class _ProcessMemoryCounters(ctypes.Structure):
    """Win32 ``PROCESS_MEMORY_COUNTERS`` (psapi.h)."""

    _fields_ = (
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    )


_WINDOWS_PSAPI: Any = None
_WINDOWS_KERNEL32: Any = None
_WINDOWS_PROBED = False


def _load_windows_apis() -> bool:
    """Bind ``GetProcessMemoryInfo`` once, with explicit signatures.

    The explicit ``restype``/``argtypes`` are load-bearing: ctypes defaults a
    return value to ``c_int``, which truncates the 64-bit ``GetCurrentProcess``
    pseudo-handle and makes ``GetProcessMemoryInfo`` fail silently.
    """
    global _WINDOWS_PSAPI, _WINDOWS_KERNEL32, _WINDOWS_PROBED
    if _WINDOWS_PROBED:
        return _WINDOWS_PSAPI is not None
    _WINDOWS_PROBED = True
    # `ctypes.WinDLL` only exists on Windows; look it up dynamically so this
    # module still imports and type-checks on Linux/macOS.
    win_dll = getattr(ctypes, "WinDLL", None)
    if win_dll is None:
        return False
    try:
        kernel32 = win_dll("kernel32")
        psapi = win_dll("psapi")
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.GetCurrentProcess.argtypes = []
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCounters),
            ctypes.c_uint32,
        ]
    except (AttributeError, OSError):
        return False
    _WINDOWS_KERNEL32 = kernel32
    _WINDOWS_PSAPI = psapi
    return True


def _windows_memory_counters() -> _ProcessMemoryCounters | None:
    if not _load_windows_apis():
        return None
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
    try:
        handle = _WINDOWS_KERNEL32.GetCurrentProcess()
        ok = _WINDOWS_PSAPI.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    except OSError:
        return None
    return counters if ok else None


def _posix_peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # ru_maxrss is bytes on macOS and KiB everywhere else that implements it.
    return maxrss if sys.platform == "darwin" else maxrss * 1024


def peak_rss_bytes() -> int:
    """High-water-mark resident set size of this process, in bytes.

    Returns 0 when the platform exposes no usable counter, so callers can
    record a row rather than crash a benchmark run.
    """
    counters = _windows_memory_counters()
    if counters is not None:
        return int(counters.PeakWorkingSetSize)
    posix = _posix_peak_rss_bytes()
    return posix if posix is not None else 0


def current_rss_bytes() -> int:
    """Instantaneous resident set size of this process, in bytes (0 if unknown)."""
    counters = _windows_memory_counters()
    if counters is not None:
        return int(counters.WorkingSetSize)
    try:
        # Linux: field 2 of statm is resident pages.
        import os

        with open("/proc/self/statm", encoding="utf-8") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError, AttributeError):
        return 0


def peak_rss_mb() -> float:
    return round(peak_rss_bytes() / _BYTES_PER_MB, 1)


class StageLog:
    """Collects ``(stage, seconds, memory)`` records for one pipeline run.

    ``peak_rss_mb`` is a process-wide high-water mark, so it never decreases;
    ``rss_delta_mb`` is therefore how much *new* peak this stage caused (0 when
    the stage stayed under an earlier high-water mark), not the stage's own
    allocation. ``current_rss_mb`` is the instantaneous figure at stage exit.
    """

    def __init__(self) -> None:
        self.stages: list[StageRecord] = []

    @contextmanager
    def stage(self, name: str, **meta: Any) -> Iterator[StageRecord]:
        rss0 = peak_rss_bytes()
        record: StageRecord = {"stage": name, **meta}
        start = time.perf_counter()
        try:
            yield record
        except BaseException as exc:  # noqa: BLE001 - re-raised immediately below
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed = time.perf_counter() - start
            rss1 = peak_rss_bytes()
            record.update(
                {
                    "seconds": round(elapsed, 4),
                    "peak_rss_mb": round(rss1 / _BYTES_PER_MB, 1),
                    "rss_delta_mb": round((rss1 - rss0) / _BYTES_PER_MB, 1),
                    "current_rss_mb": round(current_rss_bytes() / _BYTES_PER_MB, 1),
                }
            )
            self.stages.append(record)

    def run_step(self, name: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Time ``fn(*args, **kwargs)`` under stage ``name`` and return its result."""
        with self.stage(name):
            return fn(*args, **kwargs)

    @property
    def total_seconds(self) -> float:
        return round(sum(float(s.get("seconds", 0.0)) for s in self.stages), 4)

    def as_list(self) -> list[StageRecord]:
        return [dict(stage) for stage in self.stages]

    def by_stage(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for record in self.stages:
            name = str(record.get("stage", "?"))
            totals[name] = round(totals.get(name, 0.0) + float(record.get("seconds", 0.0)), 4)
        return totals

    def dump(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_list(), indent=2), encoding="utf-8")
        return path
