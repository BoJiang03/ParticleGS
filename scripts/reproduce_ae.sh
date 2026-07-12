#!/usr/bin/env bash
# reproduce_ae.sh — AE-focused reproduction sized for the SC reviewer budget.
#
# One command on a bare node: it bootstraps conda + a driver-matched, built
# `particlegs` env (scripts/bootstrap_ae.sh), fetches data, runs the seven
# experiments that produce the reduced AE set of 18 enforced metrics (EXP-1, 4,
# 6, 7, 8, 11, 14 -> adae/reference_results.json), then verifies them.
#
# What the AE fast path drops vs the full reproduce.sh (26 metrics), to fit the
# ~8 h budget — none of it weakens an enforced number the full run also checks:
#   * The heavy trainings are replaced by shipped pre-trained models. The E25
#     single-block model (pretrained/e25/, ~41 MB) and the 4 sub-block models
#     (pretrained/blocks_4/, ~36 MB) are provided; AE trains only ONE model live:
#     the 4-block merge + 60k finetune (EXP-4, ~17 min). E25's 3-stage 39k-iter
#     training with 4K/6K GT rendering — the single biggest cost (~1.5 h on a
#     graphics GPU, hours on a render-bound A100) — is skipped in favor of the
#     shipped E25, which EXP-1/6/7/8/11/14 consume. ONLY models are shipped; all
#     ground truth (training and eval) is still rendered live on your node.
#     (To OBSERVE the skipped single-block E25 training cost live, run
#      scripts/reproduce_ae_single_block.sh — but see its banner: the paper's
#      training TIME is graphics-hardware-specific, so contact the authors to run
#      it on the authors' workstation for the accurate number.)
#   * EXP-13 (FIRE-2 full retrain) and EXP-4's 2-block config are skipped
#     (4-block only). The LCP baseline is skipped — it is strictly worse than
#     SZ3, so the iso-CR comparison only needs 3DGS vs SZ3.
# The full 26/26 set is still available via scripts/reproduce.sh (trains E25 and
# all blocks live, runs FIRE-2 + 2-block + LCP + the full SZ3/LCP sweep).
#
# Driver-adaptive build: the default env pins CUDA 13.0 / torch cu130 (needs an
# R580+ driver). On a node whose driver tops out at CUDA 12.x — e.g. Chameleon
# 4x A100 on driver 560.35.05 (CUDA 12.6) — bootstrap_ae.sh auto-selects the
# CUDA 12.6 / torch cu126 env instead. Pass --no-setup to skip bootstrap when
# you have already built and activated the env yourself.
#
# Two levers keep it inside the ~8 h AE budget on a multi-GPU node:
#
#   1. EXP-1 quick mode (--ae): E25 is not trained (the shipped model is
#      evaluated), and of the SZ3 sweep only the enforced iso-CR point (#13 at
#      ~292x, matched to E25's ~290x) is computed; the other 14 SZ3 points —
#      which exist only to draw the paper's R-D curve — are skipped. EXP-1
#      collapses from ~350 min (train + full sweep) to ~15-20 min (one SZ3 point
#      render + eval of the shipped E25). Verification still passes.
#   2. Three-segment scheduling (parallelize what can be, isolate what can't):
#        Seg 1 [isolated]: EXP-1 solo    -> shared data (VTP + eval GT), the SZ3
#                          iso-CR point, and eval of the shipped E25.
#        Seg 2 [mixed]:    EXP-4 loads the 4 shipped sub-blocks -> merge -> 60k
#                          finetune on the base GPU, while EXP-7/8/14 run in
#                          parallel on the other GPUs. No live block training,
#                          so the non-base GPUs are free from the start.
#        Seg 3 [isolated]: EXP-6 then EXP-11 solo -> clean FPS / time / memory.
#      Timing/FPS/memory metrics (EXP-6/11) are measured on an otherwise-idle
#      node; deterministic quality metrics (EXP-4/7/8/14) run in parallel.
#
# GPU CHOICE MATTERS MORE THAN GPU COUNT. Ground-truth generation renders 280M
# point-gaussians in ParaView — a graphics (fill-rate) workload, not compute. A
# graphics-class GPU (RTX PRO 6000 Blackwell, RTX 6000 Ada, L40/L40S) renders it
# several times faster than a compute-class card (A100/H100), which is render-
# bound on this pipeline and can overrun the budget. Prefer a graphics node.
#
# Wall-clock (cold, from raw data; no E25 train, so dominated by live GT
# rendering + the ~17-min finetune + EXP-7's CPU particle compare + EXP-11's 6K
# training-memory probe — hardware/driver-dependent):
# Two MEASURED end-to-end points from a fresh clone (env build + experiments +
# verify), both 18/18:
#   1x RTX 6000 (Turing), single GPU  : ~7.0 h  (Chameleon rtx_6000 node) —
#                               fits the ~8 h AE budget on its own.
#   2x RTX PRO 6000 (Blackwell)       : ~2.2 h  (env ~5 min + experiments 132 min).
#   compute node (A100/H100)          : render-bound; GT rendering slower still.
# Runs on a SINGLE GPU — there is no >=2-GPU requirement; more/faster GPUs just
# cut wall-clock by parallelizing rendering + training.
# Per-experiment on the 2x RTX PRO 6000: EXP-1 16 min (incl. one-time VTP
# conversion), EXP-4 15 (shipped blocks -> merge -> 60k finetune), EXP-7 47 (CPU
# 280M compare), EXP-14 33, EXP-11 32 (6K training-mem probe), EXP-6/8 ~11 each;
# eval GT (240 frames, 1080p) ~3 min.
#
# Full-fidelity alternative (all 15+11 sweep points, live block training, ~11-15 h):
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
#                                [--no-setup]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

NUM_GPUS=4
GPU=0
EXP="1,4,6,7,8,11,14"
SEQUENTIAL=""
EVAL_ONLY=0
VERIFY=1
SETUP=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --exp) EXP="$2"; shift 2 ;;
        --sequential) SEQUENTIAL="--sequential"; shift ;;
        --eval-only) EVAL_ONLY=1; shift ;;
        --no-verify) VERIFY=0; shift ;;
        --no-setup) SETUP=0; shift ;;
        -h|--help) sed -n '2,87p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── 0. Environment: conda + driver-matched, built particlegs env ─────────────
# bootstrap_ae.sh installs Miniforge if conda is absent and creates the env from
# the CUDA-13.0 or CUDA-12.6 spec depending on the driver's max CUDA version.
if [[ "${SETUP}" = 1 ]]; then
    echo; echo "[setup] ensuring conda + built 'particlegs' env..."
    bash "${REPO_ROOT}/scripts/bootstrap_ae.sh"
fi
# Make `python` below resolve to the particlegs env. Harmless if already active.
CONDA_ROOT="${PARTICLEGS_CONDA:-$HOME/miniforge3}"
_cbase=""
command -v conda >/dev/null 2>&1 && _cbase="$(conda info --base 2>/dev/null || true)"
[[ -z "${_cbase}" && -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]] && _cbase="${CONDA_ROOT}"
if [[ -n "${_cbase}" && -f "${_cbase}/etc/profile.d/conda.sh" ]]; then
    # conda's hooks aren't nounset-clean (base hook reads $PS1; cuda-nvcc 12.6's
    # activate.d expands $NVCC_PREPEND_FLAGS unset) — under `set -u` that aborts
    # mid-source before the command returns, so `|| true` can't catch it. Run
    # the source+activate with nounset temporarily off.
    set +u
    # shellcheck disable=SC1091
    source "${_cbase}/etc/profile.d/conda.sh"
    conda activate particlegs || true
    set -u
fi

echo "======================================================================"
echo "ParticleGS — AE reproduction (18 enforced metrics, reduced fast path)"
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
python -u -m experiments.run_all \
    --ae --exp "${EXP}" --gpu "${GPU}" --num_gpus "${NUM_GPUS}" ${SEQUENTIAL}

# ── 3. Aggregate + verify ────────────────────────────────────────────────
echo
echo "[aggregate] collecting runs/exp*/results.json into tables..."
python "${REPO_ROOT}/scripts/aggregate_results.py" --out "${REPO_ROOT}/runs/summary" || true

if [[ "${VERIFY}" = 1 && -f "${REPO_ROOT}/verify_results.py" ]]; then
    echo
    echo "[verify] checking results against reference_results.json (AE reduced set)..."
    python "${REPO_ROOT}/verify_results.py" --ae
fi

echo
echo "======================================================================"
echo "DONE. Tables: runs/summary/ ; per-experiment logs: runs/ae_logs/"
echo "======================================================================"
