#!/usr/bin/env bash
# reproduce_ae_single_block.sh — measure the LIVE single-block (E25) end-to-end
# training TIME that the main AE fast path deliberately skips.
#
# WHY THIS EXISTS: scripts/reproduce_ae.sh ships the pre-trained E25 model, so a
# reviewer never runs E25's 3-stage 39k-iter training (with 4K/6K ground-truth
# rendering) — the single biggest cost in the pipeline. Reviewers on Chameleon
# Cloud therefore cannot observe the single-block end-to-end training time. This
# script runs that training LIVE so the cost can be observed directly.
#
# IMPORTANT: the paper's single-block end-to-end training TIME was measured on
# the authors' specific graphics workstation (1x RTX PRO 6000 Blackwell). It is a
# GRAPHICS (fill-rate) workload and is highly hardware-dependent; other GPUs/CPUs
# deviate substantially and will NOT match the paper. See the banner below.
#
# It runs EXP-1's 3DGS branch only: E25 live training + eval, skipping the
# SZ3/LCP baselines and NOT using the shipped pre-trained model.
#
# Usage:
#   bash scripts/reproduce_ae_single_block.sh [--gpu N] [--no-setup]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GPU=0
SETUP=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --no-setup) SETUP=0; shift ;;
        -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── Prominent reviewer notice (shown FIRST, before any work) ─────────────────
cat <<'BANNER'

################################################################################
##                                                                            ##
##        SINGLE-BLOCK (E25) END-TO-END TRAINING-TIME REPRODUCTION            ##
##                                                                            ##
##  >>>>>>>>>>>>>>>>>>>>>>>>>>  PLEASE READ FIRST  <<<<<<<<<<<<<<<<<<<<<<<<<<  ##
##                                                                            ##
##  The single-block end-to-end training TIME reported in the paper was       ##
##  measured on the authors' specific graphics workstation                    ##
##  (1x RTX PRO 6000 Blackwell). Ground-truth generation is a GRAPHICS        ##
##  (fill-rate) workload, so the wall-clock is HIGHLY hardware-dependent:      ##
##                                                                            ##
##    * On a different GPU/CPU the time WILL deviate SUBSTANTIALLY and will    ##
##      NOT match the number reported in the paper.                           ##
##    * Compute-class GPUs (A100/H100) are render-bound on this pipeline and   ##
##      run much slower than the graphics card the paper used.                ##
##                                                                            ##
##  >>> TO OBTAIN THE ACCURATE PAPER END-TO-END TRAINING TIME, PLEASE      <<< ##
##  >>> CONTACT THE AUTHORS' GROUP TO SCHEDULE FREE TIME ON THE AUTHORS'   <<< ##
##  >>> WORKSTATION AND RUN THIS EXPERIMENT THERE.                         <<< ##
##                                                                            ##
##  (The main artifact ships this model pre-trained, so reproduce_ae.sh does   ##
##   NOT train it. This script exists only to let you observe the live        ##
##   single-block training cost that the fast path skips.)                    ##
##                                                                            ##
################################################################################

BANNER
# Give the reviewer a few seconds to read the notice before work scrolls it away.
sleep 6

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
    # conda's hooks aren't nounset-clean; source+activate with nounset off.
    set +u
    # shellcheck disable=SC1091
    source "${_cbase}/etc/profile.d/conda.sh"
    conda activate particlegs || true
    set -u
fi

echo "======================================================================"
echo "ParticleGS — single-block (E25) LIVE training-time reproduction"
echo "  gpu:       ${GPU}"
echo "  what:      EXP-1 3DGS branch only (E25 3-stage train + eval)"
echo "  baselines: SZ3 / LCP skipped; shipped model NOT used (trains live)"
echo "  repo root: ${REPO_ROOT}"
echo "======================================================================"

# ── 1. Raw data (the 280M-particle HACC medium dataset) ──────────────────────
if [[ ! -f "${REPO_ROOT}/data/hacc_raw/xx.f32" \
   || ! -f "${REPO_ROOT}/data/hacc_raw/yy.f32" \
   || ! -f "${REPO_ROOT}/data/hacc_raw/zz.f32" ]]; then
    echo; echo "[data] raw HACC files missing — fetching medium (280M) dataset..."
    bash "${REPO_ROOT}/data/download_data.sh" --medium
fi

# ── 2. Run EXP-1's 3DGS branch only: E25 live training + eval ────────────────
# --skip_sz3 --skip_lcp -> only run_exp1b (E25). No --use_pretrained_e25, so the
# 3-stage E25 model trains from scratch and run_exp1b times it end to end
# (training-data generation + 3-stage training) as exp1b_e25.end_to_end_train_min.
echo
echo "[run] experiments.exp1_rate_distortion --gpu ${GPU} --skip_sz3 --skip_lcp"
python -u -m experiments.exp1_rate_distortion \
    --gpu "${GPU}" --skip_sz3 --skip_lcp

# ── 3. Surface the measured time + repeat the hardware caveat ─────────────────
RES="${REPO_ROOT}/runs/exp1/results.json"
echo
echo "======================================================================"
if [[ -f "${RES}" ]]; then
    python - "${RES}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
e = d.get("exp1b_e25", {})
t = e.get("end_to_end_train_min")
mp = e.get("avg_masked_psnr")
print("SINGLE-BLOCK (E25) LIVE TRAINING RESULT")
print(f"  end-to-end train time : {t} min  (this machine)")
if mp is not None:
    print(f"  masked PSNR (eval)    : {mp:.2f} dB   (paper 26.26)")
    print(f"  compression ratio     : {e.get('cr')}x   (paper 290)")
    print(f"  Gaussians / size      : {e.get('num_gaussians')} / {e.get('size_mb')} MB")
PY
else
    echo "  (no runs/exp1/results.json produced — check the log above)"
fi
echo "----------------------------------------------------------------------"
echo "  NOTE: the time above is for THIS hardware only. It is a graphics"
echo "  (fill-rate) workload and will differ substantially from the paper's"
echo "  number on any other GPU/CPU. For the ACCURATE paper end-to-end"
echo "  training time, please CONTACT THE AUTHORS' GROUP to schedule time on"
echo "  the authors' workstation and run this experiment there."
echo "======================================================================"
