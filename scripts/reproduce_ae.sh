#!/usr/bin/env bash
# reproduce_ae.sh — AE-focused reproduction sized for the SC reviewer budget.
#
# Runs the eight experiments that together produce all 26 enforced metrics
# (EXP-1, 4, 6, 7, 8, 11, 13, 14 → adae/reference_results.json) and then
# verifies them. Two levers keep it inside the ~8 h AE budget on a multi-GPU
# node:
#
#   1. EXP-1 quick mode (--ae): only the enforced rate-distortion points
#      (SZ3 #13, LCP #8, E25) are computed; the rest of the 15+11 SZ3/LCP
#      sweep — which exists only to draw the paper's R-D curve — is skipped.
#      EXP-1 drops from ~350 min to ~80 min. Verification still passes.
#   2. Parallel scheduling: EXP-4 (block training) and EXP-1 overlap on
#      disjoint GPU sets, then EXP-6/7/8/11/13/14 are pooled across all GPUs.
#
# Estimated wall-clock (cold, from raw data):
#   4× A100 / RTX PRO 6000 : ~5 h        (recommended AE node)
#   2× GPU                 : ~8 h
#   1× GPU                 : use scripts/reproduce.sh instead (this needs ≥2)
#
# Full-fidelity alternative (all 15+11 sweep points, ~11–15 h):
#   bash scripts/reproduce.sh
#
# Eval-only fallback (no training; verifies re-rendered metrics from shipped
# checkpoints, ~1–2 h) — if a reviewer lacks the time/GPUs to retrain:
#   bash scripts/fetch_checkpoints.sh      # downloads checkpoints.tar.gz -> runs/
#   bash scripts/reproduce_ae.sh --eval-only
#
# Usage:
#   bash scripts/reproduce_ae.sh [--num_gpus N] [--gpu BASE] [--sequential]
#                                [--exp LIST] [--eval-only] [--no-verify]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

NUM_GPUS=4
GPU=0
EXP="1,4,6,7,8,11,13,14"
SEQUENTIAL=""
EVAL_ONLY=0
VERIFY=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --exp) EXP="$2"; shift 2 ;;
        --sequential) SEQUENTIAL="--sequential"; shift ;;
        --eval-only) EVAL_ONLY=1; shift ;;
        --no-verify) VERIFY=0; shift ;;
        -h|--help) sed -n '2,38p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

echo "======================================================================"
echo "ParticleGS — AE reproduction (26 enforced metrics)"
echo "  num_gpus:    ${NUM_GPUS}"
echo "  gpu base:    ${GPU}"
echo "  experiments: ${EXP}"
echo "  mode:        $([ "${EVAL_ONLY}" = 1 ] && echo 'eval-only (shipped checkpoints)' || echo 'retrain (AE quick)')"
echo "  repo root:   ${REPO_ROOT}"
echo "======================================================================"

# ── 1. Raw data (skipped in eval-only, which reuses shipped checkpoints) ──
if [[ "${EVAL_ONLY}" = 0 ]]; then
    if [[ ! -f "${REPO_ROOT}/data/hacc_raw/xx.f32" \
       || ! -f "${REPO_ROOT}/data/hacc_raw/yy.f32" \
       || ! -f "${REPO_ROOT}/data/hacc_raw/zz.f32" ]]; then
        echo; echo "[data] raw HACC files missing — fetching medium (280M) dataset..."
        bash "${REPO_ROOT}/data/download_data.sh" --medium
    fi
    if [[ ",${EXP}," == *",13,"* ]]; then
        if [[ ! -f "${REPO_ROOT}/data/fire2_raw/xx.f32" \
           || ! -f "${REPO_ROOT}/data/fire2_raw/yy.f32" \
           || ! -f "${REPO_ROOT}/data/fire2_raw/zz.f32" ]]; then
            echo; echo "[data] raw FIRE-2 files missing — fetching L172 snapshot 010..."
            bash "${REPO_ROOT}/data/download_fire2.sh"
        fi
    fi
else
    if [[ ! -d "${REPO_ROOT}/runs" ]]; then
        echo "ERROR: --eval-only needs shipped checkpoints. Run first:" >&2
        echo "  bash scripts/fetch_checkpoints.sh" >&2
        exit 1
    fi
fi

# ── 2. Run the AE experiment set ─────────────────────────────────────────
# --ae puts EXP-1 in quick mode and (with num_gpus>1) enables parallel
# scheduling. Eval-only re-runs the same set; experiments that find an
# existing trained model on disk skip their training stage.
echo
echo "[run] experiments.run_all --ae --exp ${EXP} --gpu ${GPU} --num_gpus ${NUM_GPUS} ${SEQUENTIAL}"
python -m experiments.run_all \
    --ae --exp "${EXP}" --gpu "${GPU}" --num_gpus "${NUM_GPUS}" ${SEQUENTIAL}

# ── 3. Aggregate + verify ────────────────────────────────────────────────
echo
echo "[aggregate] collecting runs/exp*/results.json into tables..."
python "${REPO_ROOT}/scripts/aggregate_results.py" --out "${REPO_ROOT}/runs/summary" || true

if [[ "${VERIFY}" = 1 && -f "${REPO_ROOT}/verify_results.py" ]]; then
    echo
    echo "[verify] checking results against reference_results.json..."
    python "${REPO_ROOT}/verify_results.py"
fi

echo
echo "======================================================================"
echo "DONE. Tables: runs/summary/ ; per-experiment logs: runs/ae_logs/"
echo "======================================================================"
