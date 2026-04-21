#!/usr/bin/env python3
"""EXP-2: Training Strategy Ablation.

Each sub-experiment uses the full E25 3-stage progressive pipeline as baseline,
changing ONLY the ablated variable. This ensures absolute PSNR values are
comparable to the E25 reference (26.35 dB).

Sub-experiments:
  2a: Multi-orbit vs single-orbit camera training
  2c: Loss function (pure L1 vs L1+DSSIM)
  2d: Progressive resolution vs single-resolution
  2e: SH degree (SH0 vs SH1 vs SH2)
  2f: Antialiasing on/off
  2g: Stage 3 with vs without internal views

Usage:
    python -m experiments.exp2_training_ablation [--gpu 0] [--sub a,c,e,f,g]
"""

import copy
import json
import shutil
import time
from pathlib import Path

from experiments.common import *
from experiments.exp1_rate_distortion import E25_STAGES, run_e25_training


def _make_variant_stages(overrides_per_stage=None, overrides_all=None,
                         camera_override=None):
    """Create a variant of E25_STAGES by applying overrides.

    overrides_all: dict applied to ALL stages' train config.
    overrides_per_stage: list of dicts, one per stage.
    camera_override: dict to override camera config for non-mix stages.
    """
    stages = copy.deepcopy(E25_STAGES)
    for i, stage in enumerate(stages):
        if overrides_all:
            stage["train"].update(overrides_all)
        if overrides_per_stage and i < len(overrides_per_stage):
            stage["train"].update(overrides_per_stage[i])
        if camera_override and stage.get("camera") != "mix":
            stage["camera"].update(camera_override)
    return stages


def _run_variant(name, stages, output_dir, shared_data, gpu):
    """Run a full 3-stage variant and evaluate."""
    exp_dir = output_dir / name
    result_file = exp_dir / "result.json"
    if result_file.exists():
        print(f"  [Skip] {name}: result exists")
        return json.loads(result_file.read_text())

    # Run 3-stage training (reuse run_e25_training logic)
    logs = exp_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    vtp_path = shared_data["vtp"]
    norm_path = shared_data["normalization"]
    ply_path = shared_data["ply"]

    prev_checkpoint = None
    final_model_dir = None
    final_iteration = None

    for i, stage in enumerate(stages):
        stage_name = stage["name"]
        stage_dir = exp_dir / f"{i:02d}_{stage_name}"
        data_dir = stage_dir / "data"
        model_dir = stage_dir / "model"

        existing_chk = find_checkpoint(model_dir)
        if existing_chk:
            prev_checkpoint = existing_chk
            final_model_dir = model_dir
            final_iteration = int(existing_chk.stem.replace("chkpnt", ""))
            continue

        if stage.get("camera") == "mix":
            mc = stage["mix_cfg"]
            generate_mix_training_data(
                vtp_path, data_dir, norm_path,
                mc["ext_orbit_radii"], mc["ext_num_frames"], mc["ext_seed"],
                mc["int_num_frames"], mc["int_seed"], mc["int_bounds_scale"],
                mc["width"], mc["height"], mc["mix_ratio"],
                logs, stage_name)
        else:
            cam = stage["camera"]
            generate_training_data(
                vtp_path, data_dir, norm_path, cam["strategy"],
                cam["orbit_radii"], cam["num_frames"], cam["width"], cam["height"],
                logs, stage_name, viz_seed=cam.get("viz_seed", "142"))

        prepare_stage_data_dir(data_dir, ply_path, norm_path)
        final_iteration = run_stage_training(
            data_dir, model_dir, stage["train"], logs, f"train_{stage_name}",
            start_checkpoint=prev_checkpoint, init_iterations=10, gpu=gpu)
        prev_checkpoint = find_checkpoint(model_dir, final_iteration)
        final_model_dir = model_dir

        img_dir = data_dir / "images"
        if img_dir.exists():
            shutil.rmtree(img_dir)

    # Evaluate
    last_train = stages[-1]["train"]
    eval_results = evaluate_model(final_model_dir, final_iteration, shared_data,
                                  logs, gpu=gpu,
                                  factor_delta_opacity=last_train.get("factor_delta_opacity", 0.3),
                                  factor_delta_scale=last_train.get("factor_delta_scale", 0.1),
                                  min_opacity_clamp=last_train.get("min_opacity_clamp", 0.4),
                                  static_viz=last_train.get("static_viz", False))
    stats = get_model_stats(final_model_dir, final_iteration)

    result = {
        "name": name,
        "masked_psnr": eval_results["avg"]["masked_psnr"],
        "psnr": eval_results["avg"]["psnr"],
        "size_mb": round(stats["size_mb"], 1),
        "num_gaussians": stats["num_gaussians"],
        "eval": eval_results,
    }
    save_results(result, result_file)
    return result


# ── EXP-2a: Multi-orbit vs single-orbit ──────────────────────────────────

def run_exp2a(output_dir, shared_data, gpu):
    print("\n" + "="*70)
    print("EXP-2a: Multi-orbit vs Single-orbit (full 3-stage)")
    print("="*70)

    # Baseline: E25 (multi-orbit 1.0/0.7/0.5) — reuse from exp1
    from experiments.exp1_rate_distortion import run_e25_training
    e25_dir = RUNS_DIR / "exp1" / "e25"
    if (e25_dir / "02_S3_mix_6k" / "model").exists():
        chk = find_checkpoint(e25_dir / "02_S3_mix_6k" / "model")
        iteration = int(chk.stem.replace("chkpnt", ""))
        last_train = E25_STAGES[-1]["train"]
        eval_results = evaluate_model(
            str(e25_dir / "02_S3_mix_6k" / "model"), iteration, shared_data,
            output_dir / "logs", gpu=gpu,
            factor_delta_opacity=last_train["factor_delta_opacity"],
            factor_delta_scale=last_train["factor_delta_scale"],
            min_opacity_clamp=last_train["min_opacity_clamp"])
        stats = get_model_stats(str(e25_dir / "02_S3_mix_6k" / "model"), iteration)
        baseline = {"name": "E25_multi_orbit", "masked_psnr": eval_results["avg"]["masked_psnr"],
                    "size_mb": round(stats["size_mb"], 1)}
    else:
        baseline = _run_variant("2a_multi_orbit", E25_STAGES, output_dir, shared_data, gpu)

    # Variant: single orbit only (orbit_radii="1.0" for all stages)
    single_stages = _make_variant_stages(camera_override={"orbit_radii": "1.0"})
    # Also fix mix stage to use single orbit for external
    for s in single_stages:
        if s.get("camera") == "mix":
            s["mix_cfg"]["ext_orbit_radii"] = "1.0"
    variant = _run_variant("2a_single_orbit", single_stages, output_dir, shared_data, gpu)

    print_table(
        ["Config", "Masked PSNR", "Size (MB)"],
        [[baseline["name"], f"{baseline['masked_psnr']:.2f}", baseline.get("size_mb", "")],
         [variant["name"], f"{variant['masked_psnr']:.2f}", variant["size_mb"]]])
    delta = baseline["masked_psnr"] - variant["masked_psnr"]
    print(f"  Multi-orbit advantage: +{delta:.2f} dB")
    print(f"  Reference: +0.16 dB (multi > single)")
    return [baseline, variant]


# ── EXP-2c: L1 vs L1+DSSIM ──────────────────────────────────────────────

def run_exp2c(output_dir, shared_data, gpu):
    print("\n" + "="*70)
    print("EXP-2c: Pure L1 vs L1+DSSIM (full 3-stage)")
    print("="*70)

    # Baseline: E25 is already pure L1
    baseline = _run_variant("2c_pure_l1", E25_STAGES, output_dir, shared_data, gpu)

    # Variant: L1 + DSSIM(0.2)
    dssim_stages = _make_variant_stages(overrides_all={"lambda_dssim": 0.2})
    variant = _run_variant("2c_l1_dssim", dssim_stages, output_dir, shared_data, gpu)

    print_table(
        ["Config", "Loss", "Masked PSNR", "Size (MB)"],
        [["Pure L1", "L1", f"{baseline['masked_psnr']:.2f}", baseline["size_mb"]],
         ["L1+DSSIM", "L1+0.2*DSSIM", f"{variant['masked_psnr']:.2f}", variant["size_mb"]]])
    print(f"  Reference: L1 same quality, ~50% smaller model")
    return [baseline, variant]


# ── EXP-2d: Progressive resolution ───────────────────────────────────────

def run_exp2d(output_dir, shared_data, gpu):
    print("\n" + "="*70)
    print("EXP-2d: Progressive Resolution (full 3-stage vs single-res)")
    print("="*70)

    # Baseline: E25 progressive (2K→6K→mix)
    baseline = _run_variant("2d_progressive", E25_STAGES, output_dir, shared_data, gpu)

    # Variant: single resolution 2K, same total iterations (39k)
    single_res_stages = [{
        "name": "S1_single_2k",
        "camera": {"strategy": "multi_orbit", "orbit_radii": "1.0,0.7,0.5",
                    "num_frames": 400, "width": 3840, "height": 2160, "viz_seed": "142"},
        "train": {
            "iterations": 39000, "resolution_scale": 2, "sh_degree": 0,
            "densification_interval": 100, "opacity_reset_interval": 3000,
            "densify_until_iter": 30000, "densify_grad_threshold": 0.0002,
            "position_lr_init": 0.00016, "position_lr_max_steps": 50000,
            "scaling_lr": 0.005, "min_scale_pixels": 1.0, "min_opacity": 0.01,
            "antialiasing": True, "content_mask_loss": True,
            "lambda_dssim": 0.0, "factor_delta_opacity": 0.3,
            "factor_delta_scale": 0.1, "min_opacity_clamp": 0.4,
        },
    }]
    variant = _run_variant("2d_single_2k", single_res_stages, output_dir, shared_data, gpu)

    print_table(
        ["Config", "Masked PSNR", "Size (MB)"],
        [[baseline["name"], f"{baseline['masked_psnr']:.2f}", baseline["size_mb"]],
         [variant["name"], f"{variant['masked_psnr']:.2f}", variant["size_mb"]]])
    delta = baseline["masked_psnr"] - variant["masked_psnr"]
    print(f"  Progressive advantage: +{delta:.2f} dB")
    print(f"  Reference: +4.95 dB (progressive >> single-res)")
    return [baseline, variant]


# ── EXP-2e: SH degree ────────────────────────────────────────────────────

def run_exp2e(output_dir, shared_data, gpu):
    print("\n" + "="*70)
    print("EXP-2e: SH Degree (full 3-stage)")
    print("="*70)
    results = []

    for sh in [0, 1, 2]:
        stages = _make_variant_stages(overrides_all={"sh_degree": sh})
        r = _run_variant(f"2e_sh{sh}", stages, output_dir, shared_data, gpu)
        results.append(r)

    print_table(
        ["SH", "Masked PSNR", "Size (MB)", "Gaussians"],
        [[f"SH{r['name'][-1]}", f"{r['masked_psnr']:.2f}", r["size_mb"], r["num_gaussians"]]
         for r in results])
    print(f"  Reference: SH0=SH2, SH0 saves ~59% size")
    return results


# ── EXP-2f: Antialiasing ─────────────────────────────────────────────────

def run_exp2f(output_dir, shared_data, gpu):
    print("\n" + "="*70)
    print("EXP-2f: Antialiasing (full 3-stage)")
    print("="*70)

    # Baseline: E25 with AA
    baseline = _run_variant("2f_with_aa", E25_STAGES, output_dir, shared_data, gpu)

    # Variant: no AA
    no_aa_stages = _make_variant_stages(
        overrides_all={"antialiasing": False, "min_scale_pixels": None})
    variant = _run_variant("2f_no_aa", no_aa_stages, output_dir, shared_data, gpu)

    print_table(
        ["Config", "AA", "Masked PSNR", "Size (MB)"],
        [["With AA", "Y", f"{baseline['masked_psnr']:.2f}", baseline["size_mb"]],
         ["No AA", "N", f"{variant['masked_psnr']:.2f}", variant["size_mb"]]])
    size_reduction = (1 - baseline["size_mb"] / variant["size_mb"]) * 100 if variant["size_mb"] else 0
    print(f"  AA size reduction: {size_reduction:.0f}%")
    print(f"  Reference: +0.75 dB, -34% size with AA")
    return [baseline, variant]


# ── EXP-2g: Internal views in Stage 3 ──────────────────────────────────

def run_exp2g(output_dir, shared_data, gpu):
    print("\n" + "="*70)
    print("EXP-2g: Stage 3 with vs without internal views (full 3-stage)")
    print("="*70)

    # Baseline: E25 (Stage 3 = 80% external + 20% internal)
    baseline = _run_variant("2g_with_internal", E25_STAGES, output_dir, shared_data, gpu)

    # Variant: Stage 3 uses 100% external (no internal views)
    no_int_stages = copy.deepcopy(E25_STAGES)
    no_int_stages[2]["name"] = "S3_external_only_6k"
    no_int_stages[2]["camera"] = {
        "strategy": "multi_orbit", "orbit_radii": "1.0,0.7,0.5",
        "num_frames": 400, "width": 5760, "height": 3240, "viz_seed": "142",
    }
    del no_int_stages[2]["mix_cfg"]
    variant = _run_variant("2g_no_internal", no_int_stages, output_dir, shared_data, gpu)

    print_table(
        ["Config", "Stage 3", "Masked PSNR", "Size (MB)"],
        [["With internal", "80/20 mix", f"{baseline['masked_psnr']:.2f}", baseline["size_mb"]],
         ["No internal", "100% external", f"{variant['masked_psnr']:.2f}", variant["size_mb"]]])
    delta = baseline["masked_psnr"] - variant["masked_psnr"]
    print(f"  Internal views effect: {delta:+.2f} dB")
    return [baseline, variant]


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = base_parser("EXP-2: Training Strategy Ablation")
    parser.add_argument("--sub", type=str, default="all",
                        help="Run specific sub-experiment: a,c,d,e,f,g or all")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RUNS_DIR / "exp2"
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    shared_data = ensure_shared_data(gpu=args.gpu) if not args.skip_data_prep else get_shared_data_dict()

    results = {}
    subs = args.sub.split(",") if args.sub != "all" else list("acdefg")

    if "a" in subs: results["2a"] = run_exp2a(output_dir, shared_data, args.gpu)
    if "c" in subs: results["2c"] = run_exp2c(output_dir, shared_data, args.gpu)
    if "d" in subs: results["2d"] = run_exp2d(output_dir, shared_data, args.gpu)
    if "e" in subs: results["2e"] = run_exp2e(output_dir, shared_data, args.gpu)
    if "f" in subs: results["2f"] = run_exp2f(output_dir, shared_data, args.gpu)
    if "g" in subs: results["2g"] = run_exp2g(output_dir, shared_data, args.gpu)

    save_results(results, output_dir / "results.json")
    print(f"\nEXP-2 complete ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
