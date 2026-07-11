# ParticleGS — SC26 Artifact

Reviewer-facing quickstart for the **ParticleGS** paper artifact
(accepted at SC26, paper `pap525`).

> Full documentation is in the accompanying **Artifact Description (AD)** PDF.
> This README is a short path for reviewers who just want to verify the badge.

---

## 1. Badge target

- **Primary:** `Results Reproduced`
- **Also applied for:** `Artifacts Evaluated — Functional`, `Artifacts Available` (Zenodo DOI below)

Reproducibility claim is split into two tiers:

1. **Hardware-independent** (accuracy, model size, compression ratio) — expected to match our reference within small tolerance bands on any CUDA-capable GPU. Scored by `verify_results.py`.
2. **Hardware-dependent** (absolute FPS, wall-clock time, peak VRAM) — authoritative numbers apply only to our reference workstation; reviewers should observe the claimed *trends* (e.g. 3DGS FPS ≫ ParaView FPS) rather than the specific figures.

We do not require reviewer access to a specific system; reviewers run the artifact on their own hardware.

---

## 2. Artifact identifiers

| Item | Value |
|---|---|
| Paper title | *3D Gaussian Splatting for Scientific Particle Data Compression and Rendering* |
| Artifact version | `v1.0` |
| Repository | https://github.com/BoJiang03/ParticleGS |
| Archival DOI | `[Zenodo DOI — minted from GitHub release tag sc26-final at the SC26 artifact-freeze deadline]` |
| Contact | Bo Jiang \<bo.jiang@temple.edu\> |

---

## 3. Hardware requirements

Peak VRAM and timing are measured by `exp11_resource_profiling.py` on our reference workstation:

- Training peak GPU memory: **10.5 GB** (6K-resolution S3 stage)
- ParaView ground-truth rendering (280M particles): **6.9 GB** GPU
- 3DGS inference (600k Gaussians at 1920×1080): **0.4 GB** GPU

| Tier | GPU | CPU RAM | Disk | What it gets you |
|---|---|---|---|---|
| **Full reproduction** (recommended) | 1× CUDA GPU with **≥ 24 GB VRAM** (e.g. RTX 3090/4090/5090, A10, A5000, A100, H100, B200) | 32 GB | 40 GB | Retrain every model, reproduce all of Tab. 3/5/6/7/8 and Fig. scale/rd/qualitative/recovery |
| **Full reproduction, tight** | 1× CUDA GPU with **≥ 16 GB VRAM** (e.g. RTX 4080, A4000, T4 16 GB, Quadro RTX 6000) | 32 GB | 40 GB | Same as above, no headroom — reduce `resolution_scale` if OOM (§7) |
| **Authors' reference** | 1× RTX PRO 6000 Blackwell Max-Q, 96 GB | 256 GB | — | All absolute FPS / time numbers in the paper |

**GPU generation:** CUDA 13.0 (the PyTorch cu130 runtime this artifact ships) supports **Turing (compute 7.5) through Blackwell**; `install.sh` compiles native code for Turing, Ampere, Ada, Hopper, and both datacenter (sm_100) and consumer (sm_120) Blackwell, plus PTX so newer/future GPUs JIT forward. CUDA 13.0 **dropped Volta and Pascal**, so **sm_75 (Turing) is the minimum** — a V100 / P100 / GTX 10-series cannot run the cu130 build. Reviewers on those GPUs should use the Chameleon Cloud path (§ below) or an older CUDA. Reviewers on an unusual but ≥Turing GPU can force their arch: `TORCH_CUDA_ARCH_LIST="<major.minor>" bash install.sh`.

- **OS:** Linux (Ubuntu 22.04+ verified on 24.04).
- **Multi-GPU not required.** Training runs on a single GPU; the optional `--num_gpus 2` flag parallelises the per-block training loop across two GPUs to cut wall-clock by ~1.8×.
- **Disk** is dominated by cached ParaView ground-truth images (~11 GB) and per-block training state; ~40 GB free is enough for the default `--exp 1,4,6,7,8,11,13` run.

---

## 4. Software requirements

All software is installed into a conda environment named `particlegs`, declared in `environment.yml`.

- [conda](https://docs.conda.io) (miniforge / mambaforge recommended) ≥ 23.5 — **or nothing at all**: the AE wrapper `scripts/reproduce_ae.sh` installs Miniforge for you if `conda` is not on `PATH` (see §6.2a).
- NVIDIA driver supporting **CUDA ≥ 12.4**. The default `environment.yml` pins the CUDA 13.0 / PyTorch cu130 stack, which needs driver **≥ 580**; on a node whose driver tops out at CUDA 12.x (e.g. Chameleon 4× A100 on driver 560.35.05 = CUDA 12.6), the AE wrapper auto-selects a CUDA 12.6 / torch cu126 env (`environment_ae_cu126.yml`) instead — no manual step. Only the *driver* comes from the host; the CUDA *toolkit* used to build the extensions is pinned inside the conda env, so the build never depends on the host's `/usr/local/cuda`.
- C++ toolchain + CMake + Ninja — provided by the conda env (`build-essential` equivalents)

`bash install.sh` handles everything: creates the env from `environment.yml` (Python 3.12, ParaView 6.0.1, h5py, PyTorch cu130), compiles the three CUDA extensions (`diff-gaussian-rasterization`, `simple-knn`, `fused-ssim`), and builds the SZ3 / LCP baseline compressors from source. Total install time: ~15 min on a modern workstation, dominated by PyTorch wheel download and CUDA extension compilation.

---

## 5. Datasets

Raw particle data is **not included** in this artifact (tens of GB total). Download from the links below.

| Dataset | Used for | Size | Source |
|---|---|---|---|
| HACC 280M subset | Single-block + 2/4-block main results (EXP-1/4/6/7/8/11, optional EXP-12) | 3.2 GB (5.5 GB tarball) | SDRbench: https://sdrbench.github.io (auto-fetched by `data/download_data.sh`) |
| FIRE-2 L172 DM-only snapshot 010 | Cross-dataset generalization (EXP-13) | 3.2 GB extracted (4.6 GB HDF5 download) | FIRE-2 Public Release DR1, CC BY 4.0 (auto-fetched by `data/download_fire2.sh`) |
| HACC official (full) | Optional, HACC-region generalization (not in default run) | ~40 GB | SDRbench: https://sdrbench.github.io |

Both required datasets are downloaded automatically by `scripts/reproduce.sh` into `data/hacc_raw/` and `data/fire2_raw/` — reviewers do not need to fetch them manually.

---

## 6. Reproducing the paper (~11 h on 2 GPUs, ~15 h on 1 GPU)

Runs the full pipeline end-to-end: downloads raw HACC + FIRE-2 data,
retrains every model, and emits numeric results for Tab. 3 / 5 / 6 / 7 / 8
and Fig. scale / rd / qualitative / three_way / recovery, plus the FIRE-2
cross-dataset generalization row and the pose/viz-param generalization
subsection. No pre-trained checkpoints are shipped — everything is
reproduced from the raw particle files.

**Default experiment set:** EXP-1/4/6/7/8/11/13/14. EXP-14
(generalization to unseen camera poses + out-of-range viz params) is
inference-only (~40 min) and reuses the EXP-1 model. EXP-12 (optional
SSIM column for Tab. 5) is off by default — it adds ~5 h of SZ3/LCP
re-evaluation; request it explicitly with `--exp 1,4,6,7,8,11,12,13,14`.

### 6.1 Install the environment

```bash
# Creates the `particlegs` conda env + builds 3 CUDA extensions + builds SZ3/LCP.
bash install.sh

# Activate before running any experiment command.
conda activate particlegs
```

All commands below assume the `particlegs` env is active.

### 6.2 One-command reproduction

```bash
bash scripts/reproduce.sh --gpu 0 --num_gpus 2
```

The wrapper will:

1. Download `data/hacc_raw/{xx,yy,zz}.f32` from SDRbench (280 M particles, ~5.5 GB tarball) if not already present.
2. Download `data/fire2_raw/{xx,yy,zz}.f32` from the FIRE-2 Public Release (L172 DM-only, snapshot 010, ~4.6 GB HDF5 → extracted to 3.2 GB f32) if not already present.
3. Call `experiments.run_all --exp 1,4,6,7,8,11,13,14` (the default set), which in turn:
   - Runs `ensure_shared_data()` once (generates `particles.vtp`, `normalization.json`, `points3d.ply`, and 3-orbit eval GT images via ParaView; times the raw→VTP conversion into `runs/shared/timings.json`).
   - Trains the E25 single-block model (3 stages, 39 k iterations).
   - Trains the block-scan (2- and 4-block configurations with per-block training + merge + finetune; reviewer subset of the paper's 2/4/8/16 Tab. 3 scan, chosen to keep the default run within ~11 h).
   - Runs SZ3 / LCP sweeps, render benchmark, recovery methods, three-way comparison, and resource profiling. (Optional SSIM augmentation via EXP-12 is off by default.)
   - EXP-13: retrains E25 on FIRE-2 raw data with domain-scaled viz params (cross-dataset generalization).
   - EXP-14: reuses the EXP-1 model (inference-only) to measure graceful degradation on unseen camera poses and out-of-range radius/opacity factors.
4. Calls `scripts/aggregate_results.py`, writing `runs/summary/{tables.md, summary.json, fig_scale.json}`.

Partial runs are supported via `--exp <subset>` (e.g. `--exp 6,11` to re-run only the benchmarks).

### 6.2a AE fast path (~5 h on 4 GPUs, fits the 8 h reviewer budget)

If you are on a **multi-GPU node** (the recommended AE setup is a 4× A100 node,
e.g. a Chameleon `gpu_a100_pcie`/`gpu_a100_nvlink` lease) and want to stay inside
the SC AE ~8 h budget, use the AE wrapper instead of `reproduce.sh`:

```bash
# From a fresh clone on a bare node — no conda, no env needed first:
bash scripts/reproduce_ae.sh --num_gpus 4
```

This is a **single command on a bare node**. Before running the experiments it
calls `scripts/bootstrap_ae.sh`, which:

- installs **Miniforge** into `$HOME/miniforge3` if `conda` is not already on
  `PATH` (override the location with `PARTICLEGS_CONDA=/path`);
- reads the driver's max CUDA version from `nvidia-smi` and creates the
  `particlegs` env from the **matching** spec — `environment.yml` (cu130) on a
  driver ≥ CUDA 13.0, or `environment_ae_cu126.yml` (cu126) on a CUDA-12.x
  driver such as the 4× A100 node (driver 560.35.05). A100 is `sm_80`, natively
  supported by both toolchains; the build is self-checked by
  `scripts/check_rasterizer.py` before any training starts;
- builds the three CUDA extensions + SZ3/LCP.

If you have already built and activated the env yourself, skip that step with
`--no-setup`. To build the env without running experiments, run
`bash scripts/bootstrap_ae.sh` on its own.

It reproduces a **reduced set of 19 enforced metrics** (EXP-1/4/6/7/8/11/14) and
then runs `verify_results.py --ae`. Three levers keep it inside the budget:

- **Reduced set** — the two render-heaviest units are dropped from the fast path:
  EXP-13 (FIRE-2 full retrain) and EXP-4's 2-block config (EXP-4 runs 4-block
  only). That drops 7 of the 26 metrics; **all 26/26 remain reproducible via
  `scripts/reproduce.sh`.**
- **Rendering distributed across GPUs** — GT rendering (the dominant cost) is
  pinned per experiment / per exp-4 block to its own GPU, instead of the former
  behavior where every `pvbatch` render serialized onto CUDA 0. This is what
  actually lets 4 GPUs help; before, `--num_gpus` only parallelized training.
- **EXP-1 quick mode** — only the enforced rate-distortion points (SZ3 #13,
  LCP #8, E25) are computed; the remaining 15+11 SZ3/LCP sweep points, which
  exist only to draw the R-D *curve*, are skipped. EXP-1 drops ~350 → ~80 min.
- **Dependency-aware scheduling** — EXP-4 (the long pole) runs on the low GPUs
  while EXP-1 then EXP-6/7/8/11/14 fill the rest; the EXP-1 dependents unlock the
  moment EXP-1 finishes (no phase barrier). Per-experiment logs land in
  `runs/ae_logs/`.

Estimated wall-clock is **hardware-dependent**; on a 4× A100 node expect roughly
**~5–7 h** for the reduced retrain. Needs ≥ 2 GPUs; on a single GPU use
`scripts/reproduce.sh`. Add `--sequential` to disable parallel scheduling.

**Eval-only fallback (~1–2 h, no training).** If you cannot retrain within
budget, verify the reported numbers by re-rendering from the shipped model
bundle (published on the `sc26-final` release / Zenodo DOI, ~200 MB):

```bash
bash scripts/fetch_checkpoints.sh          # downloads checkpoints.tar.gz -> runs/
bash scripts/reproduce_ae.sh --eval-only
```

### 6.3 Manual low-level commands (optional)

If you want to invoke a single stage without the wrapper:

```bash
# Download data
bash data/download_data.sh --medium
bash data/download_fire2.sh    # only needed for EXP-13

# Run one experiment
python -m experiments.exp1_rate_distortion      --gpu 0
python -m experiments.exp4_block_training       --gpu 0 --num_gpus 2
python -m experiments.exp_fire2_generalization  --gpu 0

# Aggregate
python scripts/aggregate_results.py --out runs/summary
```

### 6.4 Expected results

After a full run, check numbers with:

```bash
python verify_results.py
```

This reads every `runs/<exp>/results.json` and compares against
`reference_results.json` (captured on the authors' RTX PRO 6000 Blackwell).
Each metric is scored under one of three tolerance classes:

#### Hardware-independent (scored, must pass)

Training is deterministic up to ±3 % Gaussian count / ±0.3 dB PSNR noise
from non-deterministic CUDA ops. These should match on any GPU.

| Quantity | Expected | Tolerance | Paper ref |
|---|---|---|---|
| ParticleGS (E25) masked PSNR | 26.28 dB | ± 0.3 dB | Tab. 5 / rd fig |
| ParticleGS (E25) compression ratio | 290× | ± 3 % | Tab. 5 / rd fig |
| SZ3 masked PSNR at matched CR (~233×) | 18.78 dB | ± 0.1 dB | rd fig |
| SZ3 compression ratio (nearest point) | 233.46× | ± 1 % | rd fig |
| LCP masked PSNR at matched CR (~240×) | 15.22 dB | ± 0.1 dB | rd fig |
| 2-block finetuned masked PSNR | 27.30 dB | ± 0.3 dB | Tab. 3 |
| 2-block finetuned # Gaussians | 564 062 | ± 3 % | Tab. 3 |
| 2-block finetuned size | 36.6 MB | ± 3 % | Tab. 3 |
| 4-block finetuned masked PSNR | 27.50 dB | ± 0.3 dB | Tab. 3 |
| 4-block finetuned # Gaussians | 605 702 | ± 3 % | Tab. 3 |
| 4-block finetuned size | 39.3 MB | ± 3 % | Tab. 3 |
| Recovery density correlation (V0 baseline) | 0.893 | ± 0.02 | recovery |
| Three-way recovered-render masked PSNR (far) | 19.41 dB | ± 0.3 dB | three-way |
| Generalization masked PSNR (1.3× extrap orbit) | 26.20 dB | ± 0.3 dB | gen. subsec |
| Generalization masked PSNR (in-range anchor) | 26.00 dB | ± 0.3 dB | gen. subsec |
| Generalization SSIM (in-range anchor) | 0.768 | ± 0.03 | gen. subsec |
| FIRE-2 masked PSNR | 25.05 dB | ± 0.3 dB | generalization row |
| FIRE-2 # Gaussians | 81 055 | ± 3 % | generalization row |
| FIRE-2 size | 5.3 MB | ± 3 % | generalization row |
| FIRE-2 compression ratio | 584× | ± 3 % | generalization row |
| 3DGS compression ratio (HACC) | 85.8× | ± 3 % | Tab. 6 |

The SZ3 and LCP rate-distortion points are the deterministic baselines'
values at the point nearest ParticleGS's compression ratio; the tight
tolerance reflects that these compressors are not stochastic. Together
with the ParticleGS row they enforce the paper's headline claim:
**at a matched ~230–290× compression ratio, ParticleGS leads SZ3 by
> 7 dB and LCP by > 11 dB in masked PSNR.**

#### Trend (scored, must pass)

| Claim | Expected | Threshold | Paper ref |
|---|---|---|---|
| 3DGS FPS / ParaView FPS | 2525× | > 100× | Tab. 6 speedup |
| Recovery density correlation (4-block) | 0.923 | > 0.9 | recovery |
| Generalization masked PSNR (2.5× out-of-range radius) | 20.67 dB | > 18 dB | gen. subsec |

The trend thresholds are set conservatively so any modern GPU / stochastic
training run passes while still enforcing the qualitative claim (3DGS is
orders of magnitude faster than ParaView; recovered density stays
well-correlated; quality degrades gracefully outside the trained factor
range rather than collapsing).

#### Hardware-dependent (reported only, not scored)

Absolute values are GPU-specific and not required to match.

| Quantity | Authors' reference (RTX PRO 6000) |
|---|---|
| 3DGS FPS @ 1920×1080 (4-block finetuned) | 803 FPS |
| ParaView FPS @ 1920×1080 (280M particles) | 0.32 FPS |
| Training peak GPU memory | 10.5 GB |
| Finetune wall time (60 k iter) | 14.4 min |
| Raw→VTP conversion (280M particles) | 6.85 min |

The raw→VTP conversion time is measured once, when `ensure_shared_data()`
first builds `particles.vtp`, and cached in `runs/shared/timings.json`
(it backs the Tab. 7 preprocessing row). On a rerun where the VTP already
exists it is read back from that file, not re-measured.

Full numerical tables appear in `runs/summary/tables.md`.

---

## 7. Troubleshooting

- **`pvbatch` renders on the wrong GPU.** `experiments/common.py` auto-probes the EGL→CUDA mapping via `eglQueryDeviceAttribEXT`. If you see `pvbatch` sitting on GPU 0 when you asked for GPU 1, check the log line `EGL device N → CUDA device M`.
- **CUDA OOM during training.** Reduce `resolution_scale` in the relevant stage of the JSON config (e.g. `"resolution_scale": 2` halves each side).
- **Training dies at iteration 0 with `Tried to allocate 131071.xx GiB` (a nonsense multi-TiB request on an empty GPU), and the number varies run-to-run.** This is *not* a real OOM — it is a mis-built `diff_gaussian_rasterization`: the rasterizer forward reads uninitialized memory for its buffer size. It happens when the extension was compiled by a system `nvcc` (e.g. `/usr/local/cuda` → 13.1) that does not match the PyTorch cu130 runtime, which miscompiles the Blackwell (sm_120) kernel. The shipped `environment.yml` pins a conda CUDA 13.0 toolchain and `install.sh` builds against it (`CUDA_HOME=$CONDA_PREFIX`) precisely to avoid this; `install.sh` also runs `scripts/check_rasterizer.py` right after the build to catch it before you start a reproduction. If you see this, re-create the env from `environment.yml` (do not build against a host `/usr/local/cuda`) and re-run `install.sh`. You can re-run the check any time with `python scripts/check_rasterizer.py`.
- **CUDA rasterizer illegal memory access.** Usually caused by `viz_opacity_factor < 0.4`. Fixed in `particlegs/renderer/__init__.py` by clamping to 0.4 — if you see this, make sure you're on the shipped version.
- **`diff_gaussian_rasterization` import error.** The extension is built for the `particlegs` env's Python 3.12 + CUDA 13 ABI. If you re-ran `install.sh` after changing the env's Python or torch versions, rebuild: `pip install --no-build-isolation --force-reinstall submodules/diff-gaussian-rasterization`.
- **`pvbatch: command not found`.** `conda activate particlegs` must be run before `reproduce.sh`. ParaView 6.0.1 is installed into the env by `environment.yml`.

---

## 8. Citation

Final BibTeX will be added once the camera-ready version and DOI are
available (SC26 proceedings, November 2026).

---

## 9. License

The authors' code in this artifact (pipelines, experiments, configs, and
documentation) is distributed under the **Gaussian-Splatting Research License**
(INRIA, non-commercial research only), inherited from the upstream
`diff-gaussian-rasterization` and `simple-knn` CUDA extensions this work builds on.

Third-party dependencies retain their original licenses; see the `LICENSE*`
files under each directory:

- `submodules/diff-gaussian-rasterization/LICENSE.md`, `submodules/simple-knn/LICENSE.md` — Gaussian-Splatting Research License (INRIA, non-commercial research only)
- `submodules/fused-ssim/LICENSE` — MIT
- `submodules/diff-gaussian-rasterization/third_party/glm/` — MIT (G-Truc Creation)
- `submodules/diff-gaussian-rasterization/third_party/stbi_image_write.h` — MIT / public domain (Sean Barrett)
- `SZ3/` and `LCP/` (cloned by `install.sh`) — see each project's own `LICENSE`
