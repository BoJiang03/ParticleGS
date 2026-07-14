# ParticleGS — SC26 Artifact

Reviewer quickstart for the **ParticleGS** paper (SC26, `pap525`):
*3D Gaussian Splatting for Scientific Particle Data Compression and Rendering*.

> Full details are in the separately-submitted **Artifact Description (AD)**.
> This README is the short path to verifying the badge.

## TL;DR

```bash
# Fast path for reviewers — verifies the 18 scored metrics. One command on a
# bare node: it installs the env and runs. ~7 h on a Chameleon Cloud
# gpu_rtx6000 node (validated, see §2 — fits the ~8 h AE budget), ~2.2 h on
# 2x RTX PRO 6000. Set --num_gpus to your GPU count.
bash scripts/reproduce_ae.sh --num_gpus 1
python verify_results.py --ae          # PASS/FAIL vs the paper

# Full reproduction — retrains everything, all 26 metrics (~11–15 h).
bash scripts/reproduce.sh --num_gpus 1
python verify_results.py
```

The fast path **ships the pre-trained E25 single-block model and the 4 sub-block
models** (the two slowest trainings), trains only the 4-block finetune live
(~17 min for the whole unit: merge + 60k-iter finetune + eval; the timed
finetune itself is 14.4 min), and re-renders all ground truth on your node.
The full path ships nothing and retrains from the raw particles.

---

## 1. Badges & how numbers are scored

Target badges: **Results Reproduced** (primary), *Artifacts Evaluated — Functional*,
*Artifacts Available* (Zenodo DOI).

`verify_results.py` scores each metric in one of two tiers:

- **Hardware-independent** (PSNR, Gaussian count, model size, compression ratio) —
  must match our reference within small tolerances on any GPU.
- **Hardware-dependent** (absolute FPS, wall-clock, peak VRAM) — reported only;
  check the *trend* (e.g. 3DGS ≫ ParaView FPS), not the exact figure.

## 2. Requirements

| | |
|---|---|
| **GPU** | 1× CUDA GPU, ≥ 16 GB VRAM, compute ≥ 7.5 (Turing or newer). A **graphics-class** card (RTX PRO 6000 / RTX 6000 Ada / L40) is strongly preferred: ground-truth generation renders 280 M point-gaussians in ParaView, a fill-rate workload, so compute cards (A100/H100) are much slower here. `--num_gpus N` spreads rendering/training across N GPUs. |
| **CPU / RAM / disk** | 32 GB RAM, ~40 GB free disk. |
| **OS / driver** | Linux; NVIDIA driver supporting CUDA ≥ 12.4. |
| **Validated (AE)** | **Chameleon Cloud, CHI@UC site, `gpu_rtx6000` node type** (1× Quadro RTX 6000, Turing, 24 GB; image `CC-Ubuntu24.04-CUDA`): fast path **18/18 in ~7.0 h** from a bare node. This is the exact reviewer recipe — see §4. |
| **Authors' reference** | 1× RTX PRO 6000 Blackwell (96 GB) — the machine all absolute FPS/time/VRAM numbers were measured on. Fast path from a fresh clone: 18/18 in ~2.2 h on 2× RTX PRO 6000. |

**Software.** Everything installs into a conda env named `particlegs`. You don't
need conda beforehand — `reproduce_ae.sh` installs Miniforge and builds the env
(matching your driver's CUDA version) if it's missing. To build manually:
`bash install.sh` (~15 min: PyTorch cu130 + 3 CUDA extensions + SZ3/LCP), then
`conda activate particlegs`.

## 3. Data

Raw particle data is **not** shipped and is fetched automatically on first run:

| Dataset | Used by | Size |
|---|---|---|
| HACC 280 M subset ([SDRbench](https://sdrbench.github.io)) | all main results | 3.4 GB |
| FIRE-2 L172 snapshot 010 (CC BY 4.0) | full run only (FIRE-2 generalization) | 3.2 GB |

## 4. Reproducing the paper

### Fast path (recommended for AE) — `reproduce_ae.sh`

```bash
bash scripts/reproduce_ae.sh --num_gpus 1      # set N to your GPU count; --no-setup if env built
```

Runs EXP-1/4/6/7/8/11/14 → **18 scored metrics**, then verifies them. It drops
the two render-heaviest units of the full run (FIRE-2 retrain, EXP-4's 2-block
config) and the LCP baseline, and — using the shipped models — trains only the
4-block finetune live. Measured end-to-end from a fresh clone (18/18 both):
**~7.0 h on 1× RTX 6000 (Turing, single GPU — fits the ~8 h AE budget)**, or
**~2.2 h on 2× RTX PRO 6000**. It is render-bound, so a compute-class GPU
(A100/H100) is slower. Runs on a single GPU; more/faster GPUs cut wall-clock.
Flags: `--no-setup` (skip env build), `--sequential` (disable parallel
scheduling), `--gpu B` (base GPU).

#### Validated recipe on Chameleon Cloud (recommended if you have no local GPU)

We validated the fast path end-to-end on Chameleon; reproducing our exact
setup takes three steps:

1. On **CHI@UC** (`chi.uc.chameleoncloud.org`), reserve a bare-metal lease for
   node type **`gpu_rtx6000`** (1× Quadro RTX 6000, Turing, 24 GB — check the
   availability calendar a few days ahead; GPU nodes are popular).
2. Launch it with the **`CC-Ubuntu24.04-CUDA`** image (ships an NVIDIA driver
   new enough for the cu130 build; no conda needed — the script installs
   Miniforge).
3. `git clone https://github.com/BoJiang03/ParticleGS && cd ParticleGS`, then
   run the TL;DR: `bash scripts/reproduce_ae.sh --num_gpus 1`.

On this node the fast path completed in **~7 h 01 min with 18/18 metrics
passing**. Avoid Chameleon's P100/V100 node types (Pascal/Volta — CUDA 13.0
dropped them; Turing cc 7.5 is the floor).

### Full path — `reproduce.sh`

```bash
bash scripts/reproduce.sh --num_gpus 2
```

Retrains from the raw particles (E25, the 2/4-block configs, FIRE-2) and runs
the full SZ3/LCP rate-distortion sweeps → **all 26 metrics**. ~11 h on 2 GPUs;
~15 h on a single GPU (EXP-4 block training runs serially). (The paper's
8/16-block rows and Fig. 7 scaling are outside AE scope.)

### Single-block training time — `reproduce_ae_single_block.sh`

The fast path ships E25 pre-trained, so reviewers never see the single-block
training cost. To observe it, run this **optional, supplementary** script — it
trains E25 live and reports the wall-clock (~1.5 h on 1× RTX 6000; run it
separately from the fast path, not back-to-back within the 8 h budget). The time
is graphics-hardware-specific; **for the exact paper number, contact the authors
to schedule time on the authors' workstation.**

## 5. Expected results

`verify_results.py [--ae]` compares `runs/<exp>/results.json` against
`reference_results.json` (captured on the authors' RTX PRO 6000). The full list
of expected values + tolerances for manual cross-checking is in
[`AE_EXPECTED.md`](AE_EXPECTED.md). Headline claims:

| Metric | Expected | Tol / rule | Paper |
|---|---|---|---|
| ParticleGS E25 — masked PSNR @ CR 290× | **26.28 dB** | ± 0.3 dB | Tab. VI / Fig. 8 ¹ |
| SZ3 at matched CR (~292×) — masked PSNR | **18.57 dB** | ± 0.1 dB | R-D fig ¹ |
| → ParticleGS lead at iso-CR | **+7.7 dB** | — | headline |
| 4-block finetuned — PSNR / #G / size | 27.5 dB / 606k / 39.3 MB | ± 0.3 dB, ± 3 % | Tab. III ¹ |
| Particle recovery, 4-block — density corr | 0.923 | > 0.9 | recovery |
| 3DGS vs ParaView render speedup | 2525× | > 100× | Tab. VII |
| Generalization, out-of-range radius | 20.67 dB | > 18 dB | gen. |

¹ The artifact enforces **masked** PSNR (foreground pixels, the stricter
metric); the paper's tables print **full-image** PSNR — Tab. VI single-block
28.80 dB, Tab. III 4-block 29.94 dB, Tab. IV FIRE-2 29.27 dB. Both metrics
come from the same renders; the masked references here are what
`verify_results.py` scores.

Training carries ±3 % Gaussian-count / ±0.3 dB PSNR stochastic noise; the SZ3
baseline is deterministic (hence the tight tolerance). The full run adds the
LCP baseline, the 2-block config, and the FIRE-2 row (26 metrics total).
Hardware-dependent references (reported, not scored): 3DGS 803 FPS, ParaView
0.32 FPS, training peak 10.5 GB (total nvidia-smi device usage; paper Tab. VII
prints 8.5 GB, the training process's allocator peak), finetune 14.4 min,
raw→VTP 6.85 min.
Full numerical tables land in `runs/summary/`.

## 6. Notes

- **Multi-GPU is optional** — everything runs on one GPU; `--num_gpus N` only
  cuts wall-clock by parallelizing rendering/training.
- **`pvbatch` on the wrong GPU?** `common.py` auto-probes the EGL→CUDA mapping;
  see the `EGL device N → CUDA device M` log line.
- **CUDA OOM?** Lower `resolution_scale` in the stage config (e.g. `2` halves each side).
- **Rasterizer built wrong** (training dies at iter 0 with a nonsense multi-TiB
  alloc): the env must build against its own pinned CUDA 13.0 toolchain, not a
  host `/usr/local/cuda`. `install.sh` self-checks this via
  `scripts/check_rasterizer.py`; re-run `bash install.sh` from a clean env.

---

**Repository:** https://github.com/BoJiang03/ParticleGS ·
**Archival DOI:** *Zenodo, minted from the `sc26-final` release at artifact freeze* ·
**Contact:** Bo Jiang \<bo.jiang@temple.edu\>

**License:** authors' code under the Gaussian-Splatting Research License (INRIA,
non-commercial research), inherited from `diff-gaussian-rasterization` / `simple-knn`.
Third-party components keep their own licenses (see `LICENSE*` under each submodule;
`fused-ssim` MIT, `glm` MIT). Citation BibTeX added with the camera-ready DOI.
