#!/usr/bin/env bash
# reproduce.sh — one-shot reproduction of every paper experiment.
#
# Downloads raw HACC data, retrains every model, runs every evaluation,
# and aggregates results into paper-table form. No pre-computed
# checkpoints are shipped — everything is reproduced from the raw
# particle files.
#
# Wall-clock on 2× RTX PRO 6000 (Blackwell): ~10 h for the default
# experiment set (EXP-1, 4, 6, 7, 8, 11, 13). On a single GPU, EXP-4
# block training runs serially and the total grows to ~14 h.
#
# EXP-12 (SSIM-augmented rate-distortion sweep) is NOT in the default
# set — it adds another ~5 h of SZ3/LCP re-evaluation and only
# supplies an optional SSIM column for Tab. 5. Pass it explicitly to
# reproduce that column:
#   bash scripts/reproduce.sh --exp 1,4,6,7,8,11,12,13
#
# Ablation and HACC-region generalization experiments (EXP-2, 3, 5,
# 9, 10, hacc) are not yet wired up.
#
# Usage:
#   bash scripts/reproduce.sh [--gpu N] [--num_gpus N] [--exp 1,4,6,7,8,11,13]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GPU=0
NUM_GPUS=2
EXP="1,4,6,7,8,11,13"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --exp) EXP="$2"; shift 2 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

echo "======================================================================"
echo "ParticleGS — full reproduction"
echo "  gpu:         ${GPU}"
echo "  num_gpus:    ${NUM_GPUS}"
echo "  experiments: ${EXP}"
echo "  repo root:   ${REPO_ROOT}"
echo "======================================================================"

# ── 1. Ensure raw HACC data is present ───────────────────────────────────
if [[ ! -f "${REPO_ROOT}/data/hacc_raw/xx.f32" \
   || ! -f "${REPO_ROOT}/data/hacc_raw/yy.f32" \
   || ! -f "${REPO_ROOT}/data/hacc_raw/zz.f32" ]]; then
    echo
    echo "[data] raw HACC files missing — fetching medium (280M) dataset..."
    bash "${REPO_ROOT}/data/download_data.sh" --medium
fi

# ── 1b. Ensure FIRE-2 raw data is present if EXP-13 is requested ─────────
if [[ ",${EXP}," == *",13,"* ]]; then
    if [[ ! -f "${REPO_ROOT}/data/fire2_raw/xx.f32" \
       || ! -f "${REPO_ROOT}/data/fire2_raw/yy.f32" \
       || ! -f "${REPO_ROOT}/data/fire2_raw/zz.f32" ]]; then
        echo
        echo "[data] raw FIRE-2 files missing — fetching L172 snapshot 010 (DM-only, z=0)..."
        bash "${REPO_ROOT}/data/download_fire2.sh"
    fi
fi

# ── 2. Run experiments ───────────────────────────────────────────────────
# run_all.py calls ensure_shared_data() once (generating particles.vtp,
# normalization.json, points3d.ply, and 3-orbit eval GT images), then
# runs the numbered experiments sequentially, each emitting
# runs/expN/results.json.
echo
echo "[run] experiments.run_all --exp ${EXP} --gpu ${GPU} --num_gpus ${NUM_GPUS}"
python -m experiments.run_all \
    --gpu "${GPU}" --num_gpus "${NUM_GPUS}" --exp "${EXP}"

# ── 3. Aggregate numeric results into paper-table form ───────────────────
echo
echo "[aggregate] collecting runs/exp*/results.json into tables..."
python "${REPO_ROOT}/scripts/aggregate_results.py" --out "${REPO_ROOT}/runs/summary"

echo
echo "======================================================================"
echo "DONE. Paper tables: runs/summary/"
echo "  tables.md         — Markdown view of Tab. 3 / 5 / 6 / 7 / 8"
echo "  summary.json      — machine-readable merged results"
echo "  fig_scale.json    — Fig. scale data (Gaussians vs particles/block)"
echo "======================================================================"
