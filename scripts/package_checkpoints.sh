#!/usr/bin/env bash
# package_checkpoints.sh — bundle the trained models from a completed run into
# checkpoints.tar.gz, to attach to the sc26-final GitHub Release / Zenodo
# archive. Enables the eval-only reproduction path (scripts/fetch_checkpoints.sh
# + reproduce_ae.sh --eval-only). Maintainer tool — reviewers never run this.
#
# Ships only the final trained artifacts each experiment needs to skip its
# training stage (checkpoints, VizMapper weights, point clouds, results.json).
# It deliberately EXCLUDES the multi-GB intermediate GT/eval image trees.
#
# Usage: bash scripts/package_checkpoints.sh [RUNS_DIR] [OUT.tar.gz]

set -euo pipefail
RUNS="${1:-runs}"
OUT="${2:-checkpoints.tar.gz}"

if [[ ! -d "${RUNS}" ]]; then echo "no runs dir: ${RUNS}" >&2; exit 1; fi

# Model dirs + result manifests, no image/data trees.
manifest="$(mktemp)"
find "${RUNS}" \
     \( -name 'chkpnt*.pth' -o -name 'viz_mapper_*.pth' \
        -o -name 'point_cloud.ply' -o -name 'results.json' \
        -o -name 'normalization.json' \) \
     -not -path '*/data/*' -not -path '*/images/*' -not -path '*/eval_renders/*' \
     > "${manifest}"

echo "packaging $(wc -l < "${manifest}") files -> ${OUT}"
tar -czf "${OUT}" -T "${manifest}"
rm -f "${manifest}"
echo "done: ${OUT} ($(du -h "${OUT}" | cut -f1))"
echo "Attach to the sc26-final GitHub Release and include in the Zenodo archive."
