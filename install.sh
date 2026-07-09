#!/bin/bash
# ParticleGS host installation.
# Creates the `particlegs` conda env from environment.yml, builds the three
# CUDA extensions and the SZ3 / LCP baseline compressors, and installs the
# particlegs package in editable mode.
#
# Usage:
#   bash install.sh              # full install (create env + build + install)
#   bash install.sh --no-env     # use the currently-active conda env

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="particlegs"
SKIP_ENV=false

for arg in "$@"; do
    case $arg in
        --no-env) SKIP_ENV=true ;;
    esac
done

echo "=== Installing ParticleGS ==="

# ── 1. Conda env ────────────────────────────────────────────────────────────
if [ "$SKIP_ENV" = false ]; then
    if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        echo "[1/4] Conda env '${ENV_NAME}' already exists — skipping create."
    else
        echo "[1/4] Creating conda env '${ENV_NAME}' from environment.yml..."
        conda env create -f environment.yml
    fi
    eval "$(conda shell.bash hook 2>/dev/null)"
    conda activate "$ENV_NAME"
else
    echo "[1/4] Skipping env creation (--no-env)"
fi

# ── 2. CUDA extensions ──────────────────────────────────────────────────────
# --no-build-isolation is required because each setup.py does `import torch`
# at build time, which pip's default isolated build env cannot see.
#
# Build against the conda-pinned CUDA 13.0 toolchain (environment.yml), NOT the
# host's /usr/local/cuda. On machines whose system CUDA is 13.1, nvcc 13.1
# miscompiles the Blackwell (sm_120) rasterizer kernel and every training run
# crashes at iteration 0 with a garbage multi-TiB allocation. Pointing CUDA_HOME
# at $CONDA_PREFIX makes the build reproducible and host-CUDA-independent.
#
# TORCH_CUDA_ARCH_LIST embeds native SASS for Turing→Blackwell plus Blackwell
# PTX, so the extensions run natively on reviewers' GPUs (any of Turing, Ampere,
# Ada, Hopper, Blackwell) and JIT forward to newer cards. (Note: torch drops
# this list if the build path contains the substring "arch"; correctness is
# unaffected because nvcc still emits sm_75 PTX that JITs forward, but a clean
# path additionally gives native SASS.)
if [ "$SKIP_ENV" = false ] || [ -n "${CONDA_PREFIX}" ]; then
    export CUDA_HOME="${CONDA_PREFIX}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
fi
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0;12.0+PTX}"
echo "[2/4] Building CUDA extensions (diff-gaussian-rasterization, simple-knn, fused-ssim)..."
echo "      CUDA_HOME=${CUDA_HOME}"
echo "      nvcc: $(command -v nvcc || echo 'NOT FOUND') ($(nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' || echo '?'))"
echo "      TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
pip install --no-build-isolation submodules/diff-gaussian-rasterization
pip install --no-build-isolation submodules/simple-knn
pip install --no-build-isolation submodules/fused-ssim

# Sanity-check the rasterizer build on this GPU before anyone invests hours in a
# reproduction. A bad nvcc/GPU-arch combination produces a binary that reads
# uninitialized memory and crashes at first use; catch it here, loudly.
echo "      Verifying the CUDA rasterizer loads and renders on this GPU..."
python "${SCRIPT_DIR}/scripts/check_rasterizer.py"

# ── 3. ParticleGS package ───────────────────────────────────────────────────
echo "[3/4] Installing particlegs package (editable)..."
pip install -e .

# ── 4. SZ3 + LCP (baseline compressors for EXP-1 / EXP-12) ──────────────────
# These are built into adae/SZ3/ and adae/LCP/ so the default path lookup in
# experiments/common.py and experiments/exp1_rate_distortion.py picks them up
# without needing PARTICLEGS_SZ3 / PARTICLEGS_LCP env vars.
echo "[4/4] Building SZ3 and LCP..."

if [ ! -x "${SCRIPT_DIR}/SZ3/build/tools/sz3/sz3" ]; then
    if [ ! -d "${SCRIPT_DIR}/SZ3" ]; then
        git clone --depth 1 https://github.com/szcompressor/SZ3.git "${SCRIPT_DIR}/SZ3"
    fi
    cmake -S "${SCRIPT_DIR}/SZ3" -B "${SCRIPT_DIR}/SZ3/build" \
          -DCMAKE_BUILD_TYPE=Release -GNinja
    cmake --build "${SCRIPT_DIR}/SZ3/build" --target sz3 -j
else
    echo "  SZ3 already built at ${SCRIPT_DIR}/SZ3/build/tools/sz3/sz3"
fi

if [ ! -x "${SCRIPT_DIR}/LCP/build/tools/sz3/lcp" ]; then
    if [ ! -d "${SCRIPT_DIR}/LCP" ]; then
        git clone --depth 1 https://github.com/hpdslab/LCP.git "${SCRIPT_DIR}/LCP"
    fi
    cmake -S "${SCRIPT_DIR}/LCP" -B "${SCRIPT_DIR}/LCP/build" \
          -DCMAKE_BUILD_TYPE=Release -GNinja
    cmake --build "${SCRIPT_DIR}/LCP/build" --target lcp -j
else
    echo "  LCP already built at ${SCRIPT_DIR}/LCP/build/tools/sz3/lcp"
fi

echo
echo "=== Installation complete ==="
echo
echo "Activate the environment:"
echo "  conda activate ${ENV_NAME}"
echo
echo "Run the full reproduction (HACC + FIRE-2 auto-download inside):"
echo "  bash scripts/reproduce.sh --gpu 0 --num_gpus 2"
