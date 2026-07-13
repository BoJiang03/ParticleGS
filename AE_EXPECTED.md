# ParticleGS SC26 AE — Expected Results

Expected values and tolerances for the **AE fast path** (`bash scripts/reproduce_ae.sh`).
Reference values were captured on the authors' **RTX PRO 6000 Blackwell**.

**Automated check:** `python verify_results.py --ae` reads each
`runs/<exp>/results.json` and scores every row below (PASS/FAIL). This file is
the human-readable version of the same reference (`reference_results.json`); if
they ever disagree, `reference_results.json` is authoritative.

**Stochastic noise:** 3DGS training carries **±3 % Gaussian count / ±0.3 dB
PSNR** run-to-run (non-deterministic CUDA ops). SZ3 is deterministic, hence its
tight ±0.1 dB band. Tolerances below already account for this.

---

## Enforced — must pass (18)

| # | Metric | `results.json` path | Expected | Tolerance / rule |
|---|---|---|---|---|
| 1 | E25 masked PSNR | `exp1.exp1b_e25.avg_masked_psnr` | 26.28 dB | ± 0.3 |
| 2 | E25 compression ratio | `exp1.exp1b_e25.cr` | 290.0 | ± 3 % |
| 3 | SZ3 masked PSNR @ iso-CR | `exp1.exp1a_sz3.13.avg_masked_psnr` | 18.57 dB | ± 0.1 |
| 4 | SZ3 compression ratio (iso-CR point) | `exp1.exp1a_sz3.13.cr` | 291.99 | ± 1 % |
| 5 | 4-block finetuned masked PSNR | `exp4.blocks_4.finetuned.masked_psnr` | 27.5 dB | ± 0.3 |
| 6 | 4-block finetuned #Gaussians | `exp4.blocks_4.finetuned.num_gaussians` | 605702 | ± 3 % |
| 7 | 4-block finetuned size | `exp4.blocks_4.finetuned.size_mb` | 39.3 MB | ± 3 % |
| 8 | 3DGS compression ratio (HACC) | `exp6.summary.compression_ratio` | 81.8 | ± 3 % |
| 9 | 3DGS model size | `exp6.summary.model_size_mb` | 39.3 MB | ± 3 % |
| 10 | 3DGS / ParaView render speedup | `exp6.summary.speedup` | 2525.3 | **> 100** |
| 11 | 4-block inference #Gaussians | `exp11.inference_memory.4_block_finetuned.num_gaussians` | 605702 | ± 3 % |
| 12 | Recovery density corr (V0, 1-block) | `exp7.7a.0.density_field.correlation` | 0.893 | ± 0.02 |
| 13 | Recovery density corr (4-block) | `exp7.7b.2.density_field.correlation` | 0.923 | **> 0.9** |
| 14 | Three-way recovered render (far orbit) | `exp8.recovered_eval.eval_far.masked_psnr` | 19.41 dB | ± 0.3 |
| 15 | Generalization: 1.3× extrapolated orbit | `exp14.axis_a_pose.0.masked_psnr` | 26.20 dB | ± 0.3 |
| 16 | Generalization: in-range radius | `exp14.axis_b_viz.radius.2.masked_psnr` | 26.00 dB | ± 0.3 |
| 17 | Generalization: in-range SSIM | `exp14.axis_b_viz.radius.2.ssim` | 0.768 | ± 0.03 |
| 18 | Generalization: 2.5× out-of-range radius | `exp14.axis_b_viz.radius.5.masked_psnr` | 20.67 dB | **> 18** |

**Headline claim (rows 1–4):** at a matched ~290× compression ratio, ParticleGS
(26.28 dB) leads SZ3 (18.57 dB) by **+7.7 dB masked PSNR**.

> Row 13 (`exp7.7b.2`) requires the current code: the 4-block recovery is at list
> index 2 via index-preserving placeholders. On stale code it reports MISSING.

> **Masked vs. full PSNR:** the artifact enforces *masked* PSNR (foreground
> pixels, the stricter metric); the paper's tables print *full-image* PSNR
> (Tab. VI single-block 28.80 dB, Tab. III 4-block 29.94 dB, Tab. IV FIRE-2
> 29.27 dB). Both come from the same renders.

> **Row 8:** the byte-ratio CR of the 4-block model (raw 3.37 GB / 39.3 MiB),
> same convention as rows 2/4. The paper has no 4-block CR table entry; the
> size column of Tab. III is the cross-check.

> **Row 12** expects the *single-block* (E25) model, which `reproduce_ae.sh` /
> `reproduce.sh` guarantee. If you additionally train an 8-block model
> (`runs/exp4/blocks_8`), EXP-7 will prefer it and V0 correlation moves to
> ~0.928 — outside this band. Re-run EXP-7 without `blocks_8` present.

> **Row 14** is measured on the single-block model; the paper's Fig. 12
> three-way figure uses the 8-block model (21.00 dB), hence the offset.

## Report-only — hardware-dependent, NOT scored (5)

Reference is the authors' RTX PRO 6000; your absolute values **will differ**
(especially on a compute-class GPU). Check the trend, not the number.

| Metric | `results.json` path | Reference (RTX PRO 6000) |
|---|---|---|
| 3DGS FPS @ 1080p | `exp6.summary.3dgs_fps` | 803.06 FPS |
| ParaView FPS @ 1080p (280M pts) | `exp6.summary.paraview_fps` | 0.318 FPS |
| Training peak GPU memory | `exp11.training_memory.peak_mb` | 10505 MB ² |
| Finetune wall time (60k iter) | `exp11.finetune_cost.finetune_time_min` | 14.4 min |
| Raw → VTP conversion (280M pts) | `exp11.vtp_conversion.vtp_conversion_min` | 6.85 min |

² Total nvidia-smi device usage during the S3 stage (includes CUDA context /
co-resident processes). Paper Tab. VII prints 8.5 GB — the training process's
allocator-level peak. Report-only either way.

---

## Only in the full run (`reproduce.sh`, +8 → 26 metrics)

These baselines/configs are dropped from the AE fast path (`verify_results.py`
*without* `--ae` enforces them):

| Metric | `results.json` path | Expected | Tolerance |
|---|---|---|---|
| LCP masked PSNR @ matched CR (~240×) | `exp1.exp1c_lcp.8.avg_masked_psnr` | 15.22 dB | ± 0.1 |
| 2-block finetuned masked PSNR | `exp4.blocks_2.finetuned.masked_psnr` | 27.30 dB | ± 0.3 |
| 2-block finetuned #Gaussians | `exp4.blocks_2.finetuned.num_gaussians` | 564062 | ± 3 % |
| 2-block finetuned size | `exp4.blocks_2.finetuned.size_mb` | 36.6 MB | ± 3 % |
| FIRE-2 masked PSNR | `exp_fire2.avg_masked_psnr` | 25.05 dB | ± 0.3 |
| FIRE-2 #Gaussians | `exp_fire2.num_gaussians` | 81055 | ± 3 % |
| FIRE-2 size | `exp_fire2.size_mb` | 5.3 MB | ± 3 % |
| FIRE-2 compression ratio | `exp_fire2.cr` | 584.0 | ± 3 % |
