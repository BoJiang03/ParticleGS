#!/usr/bin/env bash
# download_data.sh — fetch HACC raw particle data from SDRbench (Globus).
#
# Usage:
#   bash data/download_data.sh [--medium|--big] [--dest <dir>]
#
#   --medium   280,953,867 particles, 5.5 GB tarball, ~3.4 GB extracted.
#              This is the canonical subset used by EXP-1..EXP-12. [default]
#
#   --big      1,073,726,487 particles, 20.8 GB tarball, ~13 GB extracted.
#              Only needed for HACC-region generalization experiments.
#
#   --dest     extraction target; default: adae/data/hacc_raw

set -euo pipefail

SIZE="medium"
DEST=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --medium) SIZE="medium"; shift ;;
        --big)    SIZE="big";    shift ;;
        --dest)   DEST="$2";     shift 2 ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DEST:-${REPO_ROOT}/data/hacc_raw}"
mkdir -p "${DEST}"

if [[ "${SIZE}" == "medium" ]]; then
    URL="https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/EXASKY/HACC/EXASKY-HACC-data-medium-size.tar.gz"
    INNER_DIR="280953867"
    EXPECTED_BYTES=1123815468   # per-axis file size
else
    URL="https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/EXASKY/HACC/EXASKY-HACC-data-big-size.tar.gz"
    INNER_DIR="1073726487"
    EXPECTED_BYTES=4294905948
fi

TAR="${DEST}.tar.gz"
echo "[download_data] size=${SIZE}"
echo "  url:  ${URL}"
echo "  dest: ${DEST}"

# ── 1. Skip if data already extracted ────────────────────────────────────
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

# ── 2. Download tarball (resumable) ──────────────────────────────────────
echo "  fetching tarball → ${TAR}"
curl --fail --location --continue-at - -o "${TAR}" "${URL}"

# ── 3. Extract only xx/yy/zz.f32 (skip velocities, etc.) ─────────────────
echo "  extracting ${INNER_DIR}/{xx,yy,zz}.f32 into ${DEST}/"
tar -xzf "${TAR}" -C "$(dirname "${DEST}")" \
    "${INNER_DIR}/xx.f32" "${INNER_DIR}/yy.f32" "${INNER_DIR}/zz.f32"

# Move into place (tar extracts into <parent>/<INNER_DIR>/).
for f in xx yy zz; do
    mv -f "$(dirname "${DEST}")/${INNER_DIR}/${f}.f32" "${DEST}/${f}.f32"
done
rmdir "$(dirname "${DEST}")/${INNER_DIR}" 2>/dev/null || true

# ── 4. Verify sizes ──────────────────────────────────────────────────────
for f in xx yy zz; do
    sz=$(stat -c%s "${DEST}/${f}.f32")
    if [[ "${sz}" != "${EXPECTED_BYTES}" ]]; then
        echo "  FAIL: ${DEST}/${f}.f32 size ${sz} != expected ${EXPECTED_BYTES}" >&2
        exit 3
    fi
done
echo "  OK: xx/yy/zz.f32 ($(( EXPECTED_BYTES / 1024 / 1024 )) MB each)"

# ── 5. Clean up tarball (reviewer can re-download if --big is needed later) ─
rm -f "${TAR}"
