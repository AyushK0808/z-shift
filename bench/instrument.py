"""P1 - stage instrumentation, plus the environment capture every table needs.

``StageLog`` itself lives in ``spatial_ingestion.instrumentation`` so the
pipelines can emit timings without depending on this development-only tree;
it is re-exported here because the protocol refers to ``bench.instrument``.
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
from typing import Any

from spatial_ingestion.instrumentation import (
    StageLog,
    StageRecord,
    current_rss_bytes,
    peak_rss_bytes,
    peak_rss_mb,
)

__all__ = [
    "StageLog",
    "StageRecord",
    "current_rss_bytes",
    "cpu_model",
    "env_metadata",
    "peak_rss_bytes",
    "peak_rss_mb",
    "total_ram_gb",
]


class _MemoryStatusEx(ctypes.Structure):
    """Win32 ``MEMORYSTATUSEX`` (sysinfoapi.h)."""

    _fields_ = (
        ("dwLength", ctypes.c_uint32),
        ("dwMemoryLoad", ctypes.c_uint32),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    )


def total_ram_gb() -> float | None:
    """Installed physical memory in GiB, or None when it cannot be read."""
    win_dll = getattr(ctypes, "WinDLL", None)
    if win_dll is not None:
        try:
            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            kernel32 = win_dll("kernel32")
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return round(status.ullTotalPhys / 1024**3, 2)
        except (AttributeError, OSError):
            pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / 1024**3, 2)
    except (AttributeError, ValueError, OSError):
        return None


def cpu_model() -> str:
    """Best-effort marketing name of the CPU (not just the ISA family)."""
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            with key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(value).strip()
        except (ImportError, OSError):
            pass
    elif sys.platform == "darwin":
        try:
            out = subprocess.run(  # noqa: S603
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "unknown"


def _package_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except Exception:  # noqa: BLE001 - a broken optional dep must not kill a run
        return None
    return str(getattr(module, "__version__", "unknown"))


def git_commit() -> str | None:
    try:
        out = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def env_metadata() -> dict[str, Any]:
    """Everything the paper's "Setup" subsection has to state, in one dict.

    Emitted as columns on every results CSV so a row can never be separated
    from the machine that produced it.
    """
    meta: dict[str, Any] = {
        "cpu": cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "ram_gb": total_ram_gb(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": _package_version("numpy"),
        "scipy": _package_version("scipy"),
        "trimesh": _package_version("trimesh"),
        "pyvista": _package_version("pyvista"),
        "opencv": _package_version("cv2"),
        "torch": _package_version("torch"),
        "git_commit": git_commit(),
    }
    try:
        import torch

        meta["torch_threads"] = torch.get_num_threads()
        meta["cuda_available"] = bool(torch.cuda.is_available())
        meta["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001 - torch is optional for Tier A
        meta["torch_threads"] = None
        meta["cuda_available"] = False
        meta["gpu"] = None
    try:
        import pyvista as pv

        meta["vtk"] = str(pv.vtk_version_info)
    except Exception:  # noqa: BLE001
        meta["vtk"] = None
    return meta
