"""Fetch the DTU scans Tier B needs and write a scene manifest for them.

Local-machine counterpart of section 8 of notebooks/tier_b_gpu.ipynb. DTU ships
images (Rectified.zip, 129.6 GB) and ground truth (Points.zip, 6.97 GB) as two
archives; the server honours HTTP Range requests, so this reads each archive's
central directory and pulls only the members it needs -- one lighting
condition per scan (49 frames, 77-146 MB) plus that scan's structured-light
reference cloud (88-140 MB).

    uv run python -m bench.fetch_dtu --scans 24 37 40 --dest data/dtu

Writes <dest>/Rectified/scanNN/*.png, <dest>/Points/stl/stlNNN_total.ply,
DTU's camera matrices (<dest>/Calibration/cal18/pos_NNN.txt) and each scan's
observation mask and table plane (<dest>/ObsMask/) for bench/dtu_eval.py, and
<dest>/manifest.json. Everything lands under data/, which is gitignored.
Re-running is cheap: scans already on disk are not fetched again, and every
file is written to .part and renamed, so an interrupted download never leaves
a truncated frame or cloud that looks complete.
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import json
import logging
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

from bench.tier_b_common import DTU_TAU_MM, is_photo

DTU_IMAGES_URL = "https://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip"
DTU_POINTS_URL = "https://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip"
# DTU's evaluation kit (6.3 GB): only the camera matrices and each scan's
# observation mask and table plane are taken from it, a few MB in all.
DTU_SAMPLESET_URL = "https://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip"

# The head of DTU's standard 15-scan evaluation subset, so an F-score here is
# comparable with published DTU numbers.
DEFAULT_SCANS: tuple[int, ...] = (24, 37, 40, 55, 63)

# DTU photographs every viewpoint under 7 lighting conditions. Taking the
# directory whole hands MASt3R seven near-duplicate copies of each pose. "_3_"
# is the diffuse, evenly lit condition the notebook also uses.
DEFAULT_LIGHTING_GLOB = "*_3_r5000.png"
FRAMES_PER_SCAN = 49

DTU_STL_RE = re.compile(r"stl(\d+)_total\.ply$", re.IGNORECASE)
DTU_RECT_RE = re.compile(r"(?:^|/)scan(\d+)/([^/]+)$", re.IGNORECASE)
# cal18 is the calibration DTU's rectified images were produced with.
DTU_CAMERA_RE = re.compile(r"/Calibration/cal18/(pos_\d{3}\.txt)$", re.IGNORECASE)
DTU_EVAL_RE = re.compile(r"/ObsMask/((?:ObsMask(\d+)_10|Plane(\d+))\.mat)$", re.IGNORECASE)

# Every block is a fresh HTTPS round trip to Denmark, so blocks are large
# enough to serve several frames each; reads are sequential, so a small LRU
# of blocks is all the cache that is needed.
_RANGE_BLOCK = 8 << 20
_RANGE_CACHE_BLOCKS = 8

logger = logging.getLogger("bench.fetch_dtu")


def _ranged_get(url: str, start: int | None = None, stop: int | None = None):
    # The scheme is checked before the open so that a file:// or custom scheme
    # passed on the command line cannot turn a download into a local read.
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(f"refusing to fetch a non-HTTP url: {url}")
    span = "bytes=0-0" if start is None else f"bytes={start}-{stop}"
    request = urllib.request.Request(url, headers={"Range": span})  # noqa: S310
    return urllib.request.urlopen(request, timeout=120)  # noqa: S310


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file, backed by HTTP Range requests.

    zipfile needs only seek and read, so pointing it at one of these reads a
    remote archive's central directory and then the members asked for, leaving
    the rest of the archive on the server.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self.pos = 0
        self._blocks: dict[int, bytes] = {}
        self._order: list[int] = []
        # A ranged GET rather than HEAD: it reports the total in Content-Range
        # and proves the server honours ranges, in one request.
        with _ranged_get(url) as response:
            content_range = response.headers.get("Content-Range")
        if not content_range or "/" not in content_range:
            raise RuntimeError(f"{url} does not serve byte ranges")
        self.size = int(content_range.rsplit("/", 1)[1])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, min(self.size, base + offset))
        return self.pos

    def _block(self, index: int) -> bytes:
        cached = self._blocks.get(index)
        if cached is not None:
            return cached
        start = index * _RANGE_BLOCK
        stop = min(start + _RANGE_BLOCK, self.size) - 1
        with _ranged_get(self.url, start, stop) as response:
            data = response.read()
        self._blocks[index] = data
        self._order.append(index)
        if len(self._order) > _RANGE_CACHE_BLOCKS:
            self._blocks.pop(self._order.pop(0), None)
        return data

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self.pos
        size = min(size, self.size - self.pos)
        out = bytearray()
        while size > 0:
            index, offset = divmod(self.pos, _RANGE_BLOCK)
            chunk = self._block(index)[offset : offset + size]
            if not chunk:
                break
            out += chunk
            self.pos += len(chunk)
            size -= len(chunk)
        return bytes(out)

    def readinto(self, buffer) -> int:  # type: ignore[override]
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def _extract(archive: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path) -> None:
    part = target.with_suffix(target.suffix + ".part")
    with archive.open(info) as src, part.open("wb") as out:
        shutil.copyfileobj(src, out, 1 << 20)
    part.replace(target)


def scan_image_dir(dest: Path, scan: int) -> Path:
    return dest / "Rectified" / f"scan{scan}"


def scan_gt_path(dest: Path, scan: int) -> Path:
    return dest / "Points" / "stl" / f"stl{scan:03d}_total.ply"


def calibration_dir(dest: Path) -> Path:
    return dest / "Calibration" / "cal18"


def scan_obs_mask_path(dest: Path, scan: int) -> Path:
    return dest / "ObsMask" / f"ObsMask{scan}_10.mat"


def scan_plane_path(dest: Path, scan: int) -> Path:
    return dest / "ObsMask" / f"Plane{scan}.mat"


def _images_present(directory: Path, glob: str) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for path in directory.glob(glob) if is_photo(path))


def fetch_images(scans: list[int], dest: Path, glob: str, url: str = DTU_IMAGES_URL) -> None:
    missing = [
        scan
        for scan in scans
        if _images_present(scan_image_dir(dest, scan), glob) < FRAMES_PER_SCAN
    ]
    if not missing:
        logger.info("images for scans %s already on disk", scans)
        return

    logger.info("reading the Rectified.zip directory (~7 MB) for scans %s", missing)
    with zipfile.ZipFile(HttpRangeFile(url)) as archive:
        members: dict[int, list[zipfile.ZipInfo]] = {}
        for info in archive.infolist():
            match = DTU_RECT_RE.search(info.filename)
            if match is None:
                continue
            scan = int(match.group(1))
            if scan in missing and fnmatch.fnmatch(match.group(2), glob):
                members.setdefault(scan, []).append(info)
        absent = sorted(set(missing) - set(members))
        if absent:
            logger.warning("nothing matching %s for scan(s) %s; they are dropped", glob, absent)
        for scan, infos in sorted(members.items()):
            target_dir = scan_image_dir(dest, scan)
            target_dir.mkdir(parents=True, exist_ok=True)
            written = 0
            for info in sorted(infos, key=lambda item: item.filename):
                target = target_dir / info.filename.rsplit("/", 1)[-1]
                if target.exists():
                    continue
                _extract(archive, info, target)
                written += target.stat().st_size
            logger.info("  scan%-4d %3d frames  %7.1f MB", scan, len(infos), written / 1024**2)


def fetch_points(scans: list[int], dest: Path, url: str = DTU_POINTS_URL) -> None:
    missing = [scan for scan in scans if not scan_gt_path(dest, scan).exists()]
    if not missing:
        logger.info("reference clouds for scans %s already on disk", scans)
        return

    logger.info("reading the Points.zip directory for scans %s", missing)
    with zipfile.ZipFile(HttpRangeFile(url)) as archive:
        members = {}
        for info in archive.infolist():
            match = DTU_STL_RE.search(info.filename)
            if match:
                members[int(match.group(1))] = info
        absent = sorted(set(missing) - set(members))
        if absent:
            logger.warning("no reference cloud for scan(s) %s; they are dropped", absent)
        for scan in missing:
            info = members.get(scan)
            if info is None:
                continue
            target = scan_gt_path(dest, scan)
            target.parent.mkdir(parents=True, exist_ok=True)
            logger.info("  stl%03d_total.ply  %.1f MB ...", scan, info.file_size / 1024**2)
            _extract(archive, info, target)


def fetch_eval_files(scans: list[int], dest: Path, url: str = DTU_SAMPLESET_URL) -> None:
    """Camera matrices (shared by every scan) and each scan's mask and plane."""
    cameras_present = any(calibration_dir(dest).glob("pos_*.txt"))
    missing = [
        scan
        for scan in scans
        if not (scan_obs_mask_path(dest, scan).exists() and scan_plane_path(dest, scan).exists())
    ]
    if cameras_present and not missing:
        logger.info("evaluation files for scans %s already on disk", scans)
        return

    logger.info("reading the SampleSet.zip directory for DTU's cameras and masks")
    with zipfile.ZipFile(HttpRangeFile(url)) as archive:
        wanted: list[tuple[zipfile.ZipInfo, Path]] = []
        for info in archive.infolist():
            camera = DTU_CAMERA_RE.search(info.filename)
            if camera and not cameras_present:
                wanted.append((info, calibration_dir(dest) / camera.group(1)))
                continue
            evaluation = DTU_EVAL_RE.search(info.filename)
            if evaluation:
                scan = int(evaluation.group(2) or evaluation.group(3))
                if scan in missing:
                    wanted.append((info, dest / "ObsMask" / evaluation.group(1)))
        for info, target in wanted:
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            _extract(archive, info, target)
        logger.info("  %d evaluation files", len(wanted))
    absent = [scan for scan in missing if not scan_obs_mask_path(dest, scan).exists()]
    if absent:
        logger.warning("no observation mask for scan(s) %s; scored without DTU's protocol", absent)


def write_manifest(scans: list[int], dest: Path, glob: str) -> Path:
    """Write a SceneSet manifest; paths are relative to the manifest itself."""
    scenes = []
    for scan in scans:
        image_dir = scan_image_dir(dest, scan)
        gt_path = scan_gt_path(dest, scan)
        n_images = _images_present(image_dir, glob)
        if n_images == 0 or not gt_path.exists():
            logger.warning("scan%d incomplete on disk; left out of the manifest", scan)
            continue
        entry = {
            "name": f"scan{scan}",
            "image_dir": image_dir.relative_to(dest).as_posix(),
            "gt_path": gt_path.relative_to(dest).as_posix(),
            "image_glob": glob,
            "notes": f"DTU scan{scan}, {n_images} frames, structured-light ground truth",
        }
        eval_files = (
            calibration_dir(dest),
            scan_obs_mask_path(dest, scan),
            scan_plane_path(dest, scan),
        )
        if any(eval_files[0].glob("pos_*.txt")) and all(p.exists() for p in eval_files[1:]):
            entry["dtu_eval"] = dict(
                zip(
                    ("calibration_dir", "obs_mask", "plane"),
                    (p.relative_to(dest).as_posix() for p in eval_files),
                    strict=True,
                )
            )
        scenes.append(entry)
    manifest = {"tau": DTU_TAU_MM, "units": "mm", "scenes": scenes}
    path = dest / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("manifest with %d scene(s) -> %s", len(scenes), path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.fetch_dtu", description=__doc__)
    parser.add_argument("--scans", type=int, nargs="+", default=list(DEFAULT_SCANS))
    parser.add_argument("--dest", type=Path, default=Path("data") / "dtu")
    parser.add_argument("--lighting-glob", default=DEFAULT_LIGHTING_GLOB)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    fetch_images(args.scans, dest, args.lighting_glob)
    # Ground truth only for scans whose photographs actually arrived.
    have_images = [
        scan
        for scan in args.scans
        if _images_present(scan_image_dir(dest, scan), args.lighting_glob)
    ]
    fetch_points(have_images, dest)
    fetch_eval_files(have_images, dest)
    manifest = write_manifest(args.scans, dest, args.lighting_glob)
    print(manifest)
    return 0 if json.loads(manifest.read_text(encoding="utf-8"))["scenes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
