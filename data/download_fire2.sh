#!/usr/bin/env bash
# download_fire2.sh — fetch FIRE-2 DM-only cosmological box (L172) snapshot 010.
#
# Downloads the z=0 DM-only snapshot from the FIRE-2 public release
# (Wetzel et al. 2023, CC BY 4.0) and extracts PartType1/Coordinates into
# xx/yy/zz.f32 — the raw input format expected by
# experiments/exp_fire2_generalization.py.
#
# Dataset: boxes/L172 (172 cMpc comoving box, 1024^3 DM particles, Planck15),
# first of 4 file shards (snapshot_010.0.hdf5) → 268,397,821 particles.
#
# Usage:
#   bash data/download_fire2.sh [--dest <dir>]
#
#   --dest   extraction target; default: adae/data/fire2_raw

set -euo pipefail

DEST=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest) DEST="$2"; shift 2 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DEST:-${REPO_ROOT}/data/fire2_raw}"
mkdir -p "${DEST}"

URL="https://users.flatironinstitute.org/~mgrudic/fire2_public_release/boxes/L172/output/snapdir_010/snapshot_010.0.hdf5"
HDF5="${DEST}/snapshot_010.0.hdf5"
EXPECTED_BYTES=1073591284       # per-axis f32: 268,397,821 × 4
EXPECTED_PARTICLES=268397821

echo "[download_fire2] FIRE-2 L172 / snapdir_010 (DM-only, z=0)"
echo "  url:  ${URL}"
echo "  dest: ${DEST}"

# ── 1. Skip if f32 files already present at expected size ────────────────
all_present=1
for f in xx yy zz; do
    p="${DEST}/${f}.f32"
    if [[ ! -f "$p" ]] || [[ "$(stat -c%s "$p" 2>/dev/null || echo 0)" != "${EXPECTED_BYTES}" ]]; then
        all_present=0
        break
    fi
done
if [[ $all_present -eq 1 ]]; then
    echo "  [skip] xx/yy/zz.f32 already present at expected size."
    exit 0
fi

# ── 2. Download HDF5 (resumable) ─────────────────────────────────────────
echo "  fetching HDF5 → ${HDF5} (~4.6 GB)"
curl --fail --location --continue-at - -o "${HDF5}" "${URL}"

# ── 3. Extract PartType1/Coordinates → xx/yy/zz.f32 ──────────────────────
echo "  extracting Coordinates into ${DEST}/{xx,yy,zz}.f32"
python3 - "${HDF5}" "${DEST}" "${EXPECTED_PARTICLES}" << 'PY'
import sys
from pathlib import Path
import h5py
import numpy as np

hdf5_path, dest, expected_n = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
with h5py.File(hdf5_path, "r") as f:
    coords = f["PartType1"]["Coordinates"]
    n = coords.shape[0]
    if n != expected_n:
        raise SystemExit(f"particle count mismatch: got {n}, expected {expected_n}")
    arr = coords[:].astype(np.float32, copy=False)
for idx, name in enumerate(("xx", "yy", "zz")):
    out = dest / f"{name}.f32"
    arr[:, idx].tofile(str(out))
    print(f"    wrote {out.name}  ({out.stat().st_size} bytes)")
PY

# ── 4. Verify sizes ──────────────────────────────────────────────────────
for f in xx yy zz; do
    sz=$(stat -c%s "${DEST}/${f}.f32")
    if [[ "${sz}" != "${EXPECTED_BYTES}" ]]; then
        echo "  FAIL: ${DEST}/${f}.f32 size ${sz} != expected ${EXPECTED_BYTES}" >&2
        exit 3
    fi
done
echo "  OK: xx/yy/zz.f32 ($(( EXPECTED_BYTES / 1024 / 1024 )) MB each, ${EXPECTED_PARTICLES} particles)"

# ── 5. Remove HDF5 (keep f32 only) ───────────────────────────────────────
rm -f "${HDF5}"
