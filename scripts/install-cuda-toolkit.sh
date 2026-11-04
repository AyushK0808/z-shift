#!/usr/bin/env bash
# Install the parts of the CUDA toolkit needed to compile MASt3R's RoPE kernel
# (nvcc, its crt headers and nvvm backend, cudart, CCCL) into a user directory,
# without sudo, at the CUDA version the installed torch was built against.
#
#   PYTHON=~/zshift/venv/bin/python bash scripts/install-cuda-toolkit.sh
#
# Prints the toolkit directory (use it as CUDA_HOME) as the last line of stdout;
# progress goes to stderr. A no-op if that version is already installed.
#
# torch's pip wheels ship the CUDA runtime libraries but not the compiler, and
# a distro nvidia-cuda-toolkit package is rarely the version torch was built
# with; nvcc must match torch's CUDA major version for the extension to load.
set -euo pipefail

PY="${PYTHON:-python}"
ZSHIFT_HOME="${ZSHIFT_HOME:-$HOME/zshift}"
REDIST="https://developer.download.nvidia.com/compute/cuda/redist"
COMPONENTS="cuda_nvcc cuda_crt libnvvm cuda_cudart cuda_cccl"

log() { printf '%s\n' "$*" >&2; }

CUDA_VER="$("$PY" -c 'import torch; print(torch.version.cuda or "")')"
if [ -z "$CUDA_VER" ]; then
  log "torch at $PY is a CPU-only build; there is no CUDA version to match"
  exit 1
fi

DEST="${CUDA_HOME_TARGET:-$ZSHIFT_HOME/cuda-$CUDA_VER}"
if [ -x "$DEST/bin/nvcc" ] && [ -f "$DEST/.complete" ]; then
  log "CUDA $CUDA_VER toolkit already at $DEST"
  echo "$DEST"
  exit 0
fi

# Latest patch release of torch's major.minor, e.g. 13.0 -> redistrib_13.0.3.json.
MANIFEST="$(curl -fsSL "$REDIST/" \
  | grep -o "redistrib_${CUDA_VER//./\\.}\.[0-9]*\.json" \
  | sort -u -V | tail -1)"
if [ -z "$MANIFEST" ]; then
  log "no NVIDIA redistributable manifest for CUDA $CUDA_VER under $REDIST"
  exit 1
fi
log "installing CUDA toolkit from $MANIFEST into $DEST"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
curl -fsSL "$REDIST/$MANIFEST" -o "$WORK/manifest.json"

# One "relative_path sha256" line per component; components a release does not
# have (cuda_crt and libnvvm were split out of cuda_nvcc in 13.0) are skipped.
"$PY" - "$WORK/manifest.json" $COMPONENTS > "$WORK/archives" << 'EOF'
import json
import sys

manifest = json.load(open(sys.argv[1]))
for name in sys.argv[2:]:
    entry = manifest.get(name, {}).get("linux-x86_64")
    if entry:
        print(entry["relative_path"], entry["sha256"])
EOF

rm -rf "$DEST"
mkdir -p "$DEST"
while read -r path sha; do
  archive="$WORK/$(basename "$path")"
  log "  $(basename "$path")"
  curl -fsSL "$REDIST/$path" -o "$archive"
  echo "$sha  $archive" | sha256sum -c --quiet -
  tar -xJf "$archive" -C "$DEST" --strip-components=1
done < "$WORK/archives"

# glibc 2.42 declares rsqrt/rsqrtf noexcept, which CUDA < 13.2's crt headers
# contradict ("exception specification is incompatible"), so every .cu file
# fails to compile on Ubuntu 26.04. Backport 13.2's fix: the same declarations
# with a glibc-conditional _NV_RSQRT_SPECIFIER.
"$PY" - "$DEST/include/crt/math_functions.h" << 'EOF'
import re
import sys

path = sys.argv[1]
text = open(path).read()
if "_NV_RSQRT_SPECIFIER" not in text:
    text, count = re.subn(
        r"(rsqrtf?\((?:double|float) [ax]\))(\)?;)", r"\1 _NV_RSQRT_SPECIFIER\2", text
    )
    # Top level, right after the header's <features.h> block, as in 13.2: the
    # declarations sit in both the device and the host-only halves of the file.
    anchor = "#  define __NV_GLIBC_PROVIDES_IEC_60559_FUNCS 0\n#endif\n"
    first = text.index(anchor) + len(anchor)
    guard = (
        "#if defined(__cplusplus) && defined(__GLIBC__) && "
        "(__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 42))\n"
        "# define _NV_RSQRT_SPECIFIER noexcept(true)\n"
        "#else\n"
        "# define _NV_RSQRT_SPECIFIER\n"
        "#endif\n"
    )
    open(path, "w").write(text[:first] + guard + text[first:])
    print(f"patched {count} rsqrt declarations for glibc >= 2.42", file=sys.stderr)
EOF

"$DEST/bin/nvcc" --version >&2
echo "$MANIFEST" > "$DEST/.complete"
echo "$DEST"
