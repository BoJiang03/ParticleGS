#!/usr/bin/env bash
# bootstrap_ae.sh — bring a bare reviewer node up to a working ParticleGS build.
#
# Handles the two things a fresh Chameleon-style node lacks:
#   1. conda            — installs Miniforge into $HOME/miniforge3 if `conda` is
#                         not already on PATH (override dir with PARTICLEGS_CONDA).
#   2. a driver-matched CUDA toolchain — the default environment.yml pins CUDA
#                         13.0 / torch cu130, which needs an R580+ driver. Nodes
#                         whose driver tops out at CUDA 12.x (e.g. A100 on driver
#                         560.35.05 = CUDA 12.6) get environment_ae_cu126.yml
#                         (torch cu126) instead, selected automatically from the
#                         driver's reported max CUDA version.
#
# After this completes, `conda activate particlegs` yields a fully built env
# (three CUDA extensions + SZ3/LCP), self-checked by scripts/check_rasterizer.py.
#
# reproduce_ae.sh runs this automatically; you can also run it standalone:
#   bash scripts/bootstrap_ae.sh
#
# Env overrides:
#   PARTICLEGS_CONDA        conda base install dir      (default $HOME/miniforge3)
#   PARTICLEGS_ENV_FILE     force a specific env yml    (skip auto-detection)
#   TORCH_CUDA_ARCH_LIST    override the build arch list (skip auto-detection)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
ENV_NAME="particlegs"

# ── 1. conda ─────────────────────────────────────────────────────────────────
CONDA_ROOT="${PARTICLEGS_CONDA:-$HOME/miniforge3}"
if ! command -v conda >/dev/null 2>&1; then
    if [[ -x "${CONDA_ROOT}/bin/conda" ]]; then
        echo "[bootstrap] using existing conda at ${CONDA_ROOT}"
    else
        echo "[bootstrap] conda not found — installing Miniforge into ${CONDA_ROOT}"
        arch="$(uname -m)"   # x86_64 or aarch64
        url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${arch}.sh"
        tmp="$(mktemp -d)"
        curl -fL "${url}" -o "${tmp}/miniforge.sh"
        bash "${tmp}/miniforge.sh" -b -p "${CONDA_ROOT}"
        rm -rf "${tmp}"
    fi
    # shellcheck disable=SC1091
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
else
    echo "[bootstrap] using conda on PATH: $(command -v conda)"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# ── 2. pick a driver-matched env file + build arch list ──────────────────────
# nvidia-smi's header reports the highest CUDA runtime this driver supports.
maxcuda="$(nvidia-smi 2>/dev/null | grep -oiE 'CUDA Version: *[0-9]+\.[0-9]+' \
           | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)"
echo "[bootstrap] driver max CUDA: ${maxcuda:-unknown}"

ge() { awk "BEGIN{exit !(${1} >= ${2})}"; }   # ge A B  -> true if A >= B

if [[ -n "${PARTICLEGS_ENV_FILE:-}" ]]; then
    ENV_FILE="${PARTICLEGS_ENV_FILE}"
    echo "[bootstrap] env file forced via PARTICLEGS_ENV_FILE=${ENV_FILE}"
elif [[ -n "${maxcuda}" ]] && ge "${maxcuda}" 13.0; then
    ENV_FILE="environment.yml"                       # CUDA 13.0 / torch cu130
    : "${TORCH_CUDA_ARCH_LIST:=}"                     # let install.sh default apply
elif [[ -n "${maxcuda}" ]] && ge "${maxcuda}" 12.4; then
    ENV_FILE="environment_ae_cu126.yml"              # CUDA 12.6 / torch cu126
    # nvcc 12.6 knows sm_75..sm_90 only (no sm_100/sm_120) — cover Turing→Hopper,
    # +PTX on 9.0 so anything newer JITs forward. A100 = sm_80 (native SASS).
    : "${TORCH_CUDA_ARCH_LIST:=7.5;8.0;8.6;8.9;9.0+PTX}"
else
    echo "ERROR: driver reports max CUDA '${maxcuda:-none}', below the 12.4 floor" >&2
    echo "for the shipped torch cu126 wheels. Update the GPU driver (>= R550 for" >&2
    echo "CUDA 12.6), or set PARTICLEGS_ENV_FILE to a hand-built env." >&2
    exit 1
fi
[[ -n "${TORCH_CUDA_ARCH_LIST}" ]] && export TORCH_CUDA_ARCH_LIST
echo "[bootstrap] env file:  ${ENV_FILE}"
echo "[bootstrap] arch list: ${TORCH_CUDA_ARCH_LIST:-<install.sh default>}"

# ── 3. create the env (idempotent) ───────────────────────────────────────────
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[bootstrap] conda env '${ENV_NAME}' already exists — skipping create."
    echo "            (delete it with 'conda env remove -n ${ENV_NAME}' to rebuild)"
else
    echo "[bootstrap] creating env '${ENV_NAME}' from ${ENV_FILE}..."
    conda env create -n "${ENV_NAME}" -f "${REPO_ROOT}/${ENV_FILE}"
fi
conda activate "${ENV_NAME}"

# ── 4. build extensions + SZ3/LCP in the active env ──────────────────────────
echo "[bootstrap] building CUDA extensions + baselines (install.sh --no-env)..."
bash "${REPO_ROOT}/install.sh" --no-env

echo
echo "[bootstrap] done. Environment '${ENV_NAME}' is built and self-checked."
echo "            conda base: $(conda info --base)"
