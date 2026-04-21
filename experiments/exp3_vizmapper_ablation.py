#!/usr/bin/env python3
"""EXP-3: VizMapper Ablation.

Validates VizMapper design choices (output range, identity regularization).

Sub-experiments:
  3a: VizMapper output range (factor_delta_opacity/scale)
  3b: Identity regularization strength (lambda_identity)
  3c: VizMapper on/off (full 3-stage E25 pipeline)

Usage:
    python -m experiments.exp3_vizmapper_ablation [--gpu 0] [--sub a,b,c]
"""

import json
import shutil
import time
from pathlib import Path

from experiments.common import *
from experiments.exp1_rate_distortion import E25_STAGES
from experiments.exp2_training_ablation import _make_variant_stages, _run_variant

BASE_TRAIN = {
    "iterations": 12000, "sh_degree": 0,
    "densification_interval": 100, "opacity_reset_interval": 3000,
    "densify_until_iter": 12000, "densify_grad_threshold": 0.0002,
    "position_lr_init": 0.00016, "position_lr_max_steps": 50000,
    "scaling_lr": 0.005, "min_scale_pixels": 1.0, "min_opacity": 0.01,
    "antialiasing": True, "content_mask_loss": True,
    "lambda_dssim": 0.0, "min_opacity_clamp": 0.4,
}

CAMERA_4K = {"strategy": "multi_orbit", "orbit_radii": "1.0,0.7,0.5",
             "num_frames": 400, "width": 3840, "height": 2160}


def _train_one(name, train_cfg, output_dir, shared_data, gpu):
    exp_dir = output_dir / name
    logs = exp_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    data_dir = exp_dir / "data"
    model_dir = exp_dir / "model"

    result_file = exp_dir / "result.json"
    if result_file.exists():
        print(f"  [Skip] {name}: result exists")
        return json.loads(result_file.read_text())

    existing = find_checkpoint(model_dir)
    if not existing:
        generate_training_data(
            shared_data["vtp"], data_dir, shared_data["normalization"],
            CAMERA_4K["strategy"], CAMERA_4K["orbit_radii"],
            CAMERA_4K["num_frames"], CAMERA_4K["width"], CAMERA_4K["height"],
            logs, name)
        prepare_stage_data_dir(data_dir, shared_data["ply"], shared_data["normalization"])
        iteration = run_stage_training(data_dir, model_dir, train_cfg, logs,
                                      f"train_{name}", init_iterations=10, gpu=gpu)
        img_dir = data_dir / "images"
        if img_dir.exists():
            shutil.rmtree(img_dir)
    else:
        iteration = int(existing.stem.replace("chkpnt", ""))

    eval_results = evaluate_model(model_dir, iteration, shared_data, logs, gpu=gpu,
                                   factor_delta_opacity=train_cfg.get("factor_delta_opacity", 0.3),
                                   factor_delta_scale=train_cfg.get("factor_delta_scale", 0.1),
                                   min_opacity_clamp=train_cfg.get("min_opacity_clamp", 0.4))
    stats = get_model_stats(model_dir, iteration)

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


def run_exp3a(output_dir, shared_data, gpu):
    """EXP-3a: VizMapper output range."""
    print("\n" + "="*70)
    print("EXP-3a: VizMapper Output Range")
    print("="*70)
    results = []
    configs = [
        ("3a_default_0.3_0.1", 0.3, 0.1),
        ("3a_wider_0.6_0.3", 0.6, 0.3),
        ("3a_widest_0.9_0.5", 0.9, 0.5),
    ]
    for name, fdo, fds in configs:
        cfg = dict(BASE_TRAIN, factor_delta_opacity=fdo, factor_delta_scale=fds,
                   lambda_identity=0.01)
        r = _train_one(name, cfg, output_dir, shared_data, gpu)
        results.append(r)

    print_table(
        ["Config", "delta_opa", "delta_scale", "Masked PSNR"],
        [[r["name"], r["name"].split("_")[2], r["name"].split("_")[3],
          f"{r['masked_psnr']:.2f}"] for r in results])
    return results


def run_exp3b(output_dir, shared_data, gpu):
    """EXP-3b: Identity regularization."""
    print("\n" + "="*70)
    print("EXP-3b: Identity Regularization")
    print("="*70)
    results = []
    configs = [
        ("3b_lambda_0.01", 0.01),
        ("3b_lambda_0.001", 0.001),
        ("3b_lambda_0.0", 0.0),
    ]
    for name, lam in configs:
        cfg = dict(BASE_TRAIN, lambda_identity=lam,
                   factor_delta_opacity=0.3, factor_delta_scale=0.1)
        r = _train_one(name, cfg, output_dir, shared_data, gpu)
        results.append(r)

    print_table(
        ["Config", "lambda_identity", "Masked PSNR"],
        [[r["name"], name.split("_")[-1], f"{r['masked_psnr']:.2f}"]
         for r, (name, _) in zip(results, configs)])
    return results


def run_exp3c(output_dir, shared_data, gpu):
    """EXP-3c: VizMapper on/off (full 3-stage E25 pipeline)."""
    print("\n" + "="*70)
    print("EXP-3c: VizMapper On/Off (full 3-stage E25)")
    print("="*70)

    # Baseline: reuse E25 from exp1 (same pattern as exp2a)
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
        baseline = {"name": "3c_with_vizmapper", "masked_psnr": eval_results["avg"]["masked_psnr"],
                    "psnr": eval_results["avg"]["psnr"],
                    "size_mb": round(stats["size_mb"], 1),
                    "num_gaussians": stats["num_gaussians"],
                    "eval": eval_results}
    else:
        baseline = _run_variant("3c_with_vizmapper", E25_STAGES, output_dir, shared_data, gpu)

    # Variant: static model — no VizMapper, no viz factor dynamics
    # static_viz=True forces viz_scale_factor=1.0, viz_opacity_factor=1.0 in both
    # training and eval, making the 3DGS model completely static.
    no_vm_stages = _make_variant_stages(
        overrides_all={"static_viz": True})
    variant = _run_variant("3c_no_vizmapper", no_vm_stages, output_dir, shared_data, gpu)

    print_table(
        ["Config", "VizMapper", "Masked PSNR", "Size (MB)", "Gaussians"],
        [["E25 + VizMapper", "ON", f"{baseline['masked_psnr']:.2f}",
          baseline["size_mb"], baseline["num_gaussians"]],
         ["E25 no VizMapper", "OFF", f"{variant['masked_psnr']:.2f}",
          variant["size_mb"], variant["num_gaussians"]]])
    delta = baseline["masked_psnr"] - variant["masked_psnr"]
    print(f"  VizMapper advantage: +{delta:.2f} dB")
    return [baseline, variant]


def main():
    parser = base_parser("EXP-3: VizMapper Ablation")
    parser.add_argument("--sub", type=str, default="all", help="a,b,c or all")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RUNS_DIR / "exp3"
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    shared_data = ensure_shared_data(gpu=args.gpu) if not args.skip_data_prep else get_shared_data_dict()

    results = {}
    subs = args.sub.split(",") if args.sub != "all" else ["a", "b", "c"]
    if "a" in subs: results["3a"] = run_exp3a(output_dir, shared_data, args.gpu)
    if "b" in subs: results["3b"] = run_exp3b(output_dir, shared_data, args.gpu)
    if "c" in subs: results["3c"] = run_exp3c(output_dir, shared_data, args.gpu)

    save_results(results, output_dir / "results.json")
    print(f"\nEXP-3 complete ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
