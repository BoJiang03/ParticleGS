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
| **Full reproduction** (recommended) | 1× CUDA GPU with **≥ 24 GB VRAM** (e.g. RTX 3090/4090, A10, A5000, A100, H100) | 32 GB | 40 GB | Retrain every model, reproduce all of Tab. 3/5/6/7/8 and Fig. scale/rd/qualitative/recovery |
| **Full reproduction, tight** | 1× CUDA GPU with **≥ 16 GB VRAM** (e.g. RTX 4080, V100 16G, A4000) | 32 GB | 40 GB | Same as above, no headroom — reduce `resolution_scale` if OOM (§7) |
| **Authors' reference** | 1× RTX PRO 6000 Blackwell Max-Q, 96 GB | 256 GB | — | All absolute FPS / time numbers in the paper |

- **OS:** Linux (Ubuntu 22.04+ verified on 24.04).
- **Multi-GPU not required.** Training runs on a single GPU; the optional `--num_gpus 2` flag parallelises the per-block training loop across two GPUs to cut wall-clock by ~1.8×.
- **Disk** is dominated by cached ParaView ground-truth images (~11 GB) and per-block training state; ~40 GB free is enough for the default `--exp 1,4,6,7,8,11,12,13` run.

---

## 4. Software requirements

All software is installed into a conda environment named `particlegs`, declared in `environment.yml`.

- [conda](https://docs.conda.io) (miniforge / mambaforge recommended) ≥ 23.5
- Host NVIDIA driver ≥ 535 (CUDA 13 runtime is pulled as PyTorch cu130 wheels)
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

## 6. Reproducing the paper (~10 h on 2 GPUs, ~14 h on 1 GPU)

Runs the full pipeline end-to-end: downloads raw HACC + FIRE-2 data,
retrains every model, and emits numeric results for Tab. 3 / 5 / 6 / 7 / 8
and Fig. scale / rd / qualitative / three_way / recovery, plus the FIRE-2
cross-dataset generalization row. No pre-trained checkpoints are shipped —
everything is reproduced from the raw particle files.

**Default experiment set:** EXP-1/4/6/7/8/11/13. EXP-12 (optional
SSIM column for Tab. 5) is off by default — it adds ~5 h of SZ3/LCP
re-evaluation; request it explicitly with `--exp 1,4,6,7,8,11,12,13`.

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
3. Call `experiments.run_all --exp 1,4,6,7,8,11,13` (the default set), which in turn:
   - Runs `ensure_shared_data()` once (generates `particles.vtp`, `normalization.json`, `points3d.ply`, and 3-orbit eval GT images via ParaView).
   - Trains the E25 single-block model (3 stages, 39 k iterations).
   - Trains the block-scan (2- and 4-block configurations with per-block training + merge + finetune; reviewer subset of the paper's 2/4/8/16 Tab. 3 scan, chosen to keep the default run within ~10 h).
   - Runs SZ3 / LCP sweeps, render benchmark, recovery methods, three-way comparison, and resource profiling. (Optional SSIM augmentation via EXP-12 is off by default.)
   - EXP-13: retrains E25 on FIRE-2 raw data with domain-scaled viz params (cross-dataset generalization).
4. Calls `scripts/aggregate_results.py`, writing `runs/summary/{tables.md, summary.json, fig_scale.json}`.

Partial runs are supported via `--exp <subset>` (e.g. `--exp 6,11` to re-run only the benchmarks).

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
| 2-block finetuned masked PSNR | 27.30 dB | ± 0.3 dB | Tab. 3 |
| 2-block finetuned # Gaussians | 564 062 | ± 3 % | Tab. 3 |
| 2-block finetuned size | 36.6 MB | ± 3 % | Tab. 3 |
| 4-block finetuned masked PSNR | 27.50 dB | ± 0.3 dB | Tab. 3 |
| 4-block finetuned # Gaussians | 605 702 | ± 3 % | Tab. 3 |
| 4-block finetuned size | 39.3 MB | ± 3 % | Tab. 3 |
| FIRE-2 masked PSNR | 25.05 dB | ± 0.3 dB | generalization row |
| FIRE-2 # Gaussians | 81 055 | ± 3 % | generalization row |
| FIRE-2 size | 5.3 MB | ± 3 % | generalization row |
| FIRE-2 compression ratio | 584× | ± 3 % | generalization row |
| 3DGS compression ratio (HACC) | 85.8× | ± 3 % | Tab. 6 |

#### Trend (scored, must pass)

| Claim | Expected | Paper ref |
|---|---|---|
| 3DGS FPS / ParaView FPS | > 100× | Tab. 6 speedup |

Our reference ratio is 2525×; the threshold is set conservatively so any
modern GPU passes.

#### Hardware-dependent (reported only, not scored)

Absolute values are GPU-specific and not required to match.

| Quantity | Authors' reference (RTX PRO 6000) |
|---|---|
| 3DGS FPS @ 1920×1080 (4-block finetuned) | 803 FPS |
| ParaView FPS @ 1920×1080 (280M particles) | 0.32 FPS |
| Training peak GPU memory | 10.5 GB |
| Finetune wall time (60 k iter) | 14.4 min |

Full numerical tables appear in `runs/summary/tables.md`.

---

## 7. Troubleshooting

- **`pvbatch` renders on the wrong GPU.** `experiments/common.py` auto-probes the EGL→CUDA mapping via `eglQueryDeviceAttribEXT`. If you see `pvbatch` sitting on GPU 0 when you asked for GPU 1, check the log line `EGL device N → CUDA device M`.
- **CUDA OOM during training.** Reduce `resolution_scale` in the relevant stage of the JSON config (e.g. `"resolution_scale": 2` halves each side).
- **CUDA rasterizer illegal memory access.** Usually caused by `viz_opacity_factor < 0.4`. Fixed in `particlegs/renderer/__init__.py` by clamping to 0.4 — if you see this, make sure you're on the shipped version.
- **`diff_gaussian_rasterization` import error.** The extension is built for the `particlegs` env's Python 3.12 + CUDA 13 ABI. If you re-ran `install.sh` after changing the env's Python or torch versions, rebuild: `pip install --no-build-isolation --force-reinstall submodules/diff-gaussian-rasterization`.
- **`pvbatch: command not found`.** `conda activate particlegs` must be run before `reproduce.sh`. ParaView 6.0.1 is installed into the env by `environment.yml`.

---

## 8. Citation

```bibtex
[PLACEHOLDER — final BibTeX once paper is assigned a citation]
```

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
