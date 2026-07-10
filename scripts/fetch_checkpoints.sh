#!/usr/bin/env bash
# fetch_checkpoints.sh — download the pre-trained model bundle and unpack it into
# runs/, enabling the eval-only reproduction path:
#     bash scripts/fetch_checkpoints.sh
#     bash scripts/reproduce_ae.sh --eval-only
#
# The bundle (~200 MB) is published as an asset on the sc26-final GitHub Release
# and mirrored in the Zenodo archive (same DOI as the artifact). Override the
# source with CHECKPOINTS_URL if needed.
#
# Reviewers who retrain (the default reproduce_ae.sh path) do NOT need this.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Filled in at the sc26-final release/DOI freeze (adae.md Phase 3).
: "${CHECKPOINTS_URL:=https://github.com/BoJiang03/ParticleGS/releases/download/sc26-final/checkpoints.tar.gz}"

if [[ "${CHECKPOINTS_URL}" == *"sc26-final"* ]] \
   && ! curl -sfI "${CHECKPOINTS_URL}" >/dev/null 2>&1; then
    echo "NOTE: ${CHECKPOINTS_URL} not reachable yet." >&2
    echo "The checkpoint bundle is published at the sc26-final release / Zenodo DOI" >&2
    echo "freeze. Until then, use the retrain path: bash scripts/reproduce_ae.sh" >&2
    echo "Or set CHECKPOINTS_URL=<url> to point at a local/alternate copy." >&2
    exit 1
fi

echo "[fetch] ${CHECKPOINTS_URL}"
tmp="$(mktemp -d)"
curl -fL "${CHECKPOINTS_URL}" -o "${tmp}/checkpoints.tar.gz"
echo "[unpack] -> ${REPO_ROOT}/runs/"
mkdir -p "${REPO_ROOT}/runs"
tar -xzf "${tmp}/checkpoints.tar.gz" -C "${REPO_ROOT}"
rm -rf "${tmp}"
echo "done. Now: bash scripts/reproduce_ae.sh --eval-only"
