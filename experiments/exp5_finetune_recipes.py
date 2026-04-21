#!/usr/bin/env python3
"""EXP-5: Finetune Recipe Optimization (F series).

Tests finetune configurations on merged 8-block model.
Base model: 8-block merged (from EXP-4).

Sub-experiments:
  5a: Key finetune parameters (F01 baseline → F05 → F16 → F18 → F19)
  5b: Negative results (more iter, SH1, no AA, random bg, bigger VizMapper, DSSIM)

Usage:
    python -m experiments.exp5_finetune_recipes [--gpu 0] [--base_model <path>]
"""

import json
import shutil
import time
from pathlib import Path

from experiments.common import *

F_CONFIGS = {
    "F01_baseline": {
        "iterations": 10000, "sh_degree": 0,
        "densification_interval": 200, "opacity_reset_interval": 3000,
        "densify_until_iter": 8000, "densify_grad_threshold": 0.0003,
        "position_lr_init": 0.00008, "position_lr_max_steps": 10000,
        "scaling_lr": 0.003, "min_scale_pixels": 0.0, "min_opacity": 0.005,
        "antialiasing": True, "content_mask_loss": False,
        "lambda_dssim": 0.2, "lambda_identity": 0.01,
        "factor_delta_opacity": 0.3, "factor_delta_scale": 0.1,
        "min_opacity_clamp": 0.4,
    },
    "F05_mask_l1": {
        "iterations": 60000, "sh_degree": 0,
        "densification_interval": 200, "opacity_reset_interval": 3000,
        "densify_until_iter": 45000, "densify_grad_threshold": 0.0003,
        "position_lr_init": 0.00008, "position_lr_max_steps": 60000,
        "scaling_lr": 0.003, "min_scale_pixels": 0.0, "min_opacity": 0.005,
        "antialiasing": True, "content_mask_loss": True,
        "lambda_dssim": 0.0, "lambda_identity": 0.01,
        "factor_delta_opacity": 0.3, "factor_delta_scale": 0.1,
        "min_opacity_clamp": 0.4,
    },
    "F16_production": {
        "iterations": 60000, "sh_degree": 0,
        "densification_interval": 200, "opacity_reset_interval": 3000,
        "densify_until_iter": 45000, "densify_grad_threshold": 0.0003,
        "position_lr_init": 0.00008, "position_lr_max_steps": 60000,
        "scaling_lr": 0.003, "min_scale_pixels": 0.0, "min_opacity": 0.005,
        "antialiasing": True, "content_mask_loss": True,
        "lambda_dssim": 0.0, "lambda_identity": 0.0,
        "factor_delta_opacity": 0.8, "factor_delta_scale": 0.3,
        "min_opacity_clamp": 0.1,
    },
    "F19_widest": {
        "iterations": 60000, "sh_degree": 0,
        "densification_interval": 200, "opacity_reset_interval": 3000,
        "densify_until_iter": 45000, "densify_grad_threshold": 0.0003,
        "position_lr_init": 0.00008, "position_lr_max_steps": 60000,
        "scaling_lr": 0.003, "min_scale_pixels": 0.0, "min_opacity": 0.005,
        "antialiasing": True, "content_mask_loss": True,
        "lambda_dssim": 0.0, "lambda_identity": 0.0,
        "factor_delta_opacity": 0.95, "factor_delta_scale": 0.5,
        "min_opacity_clamp": 0.05,
    },
}

F_NEGATIVE = {
    "F10_no_densify": {
        # Disable densification entirely — shows it is needed
        **F_CONFIGS["F16_production"], "densify_grad_threshold": 1.0,
        "densify_until_iter": 0,
    },
    "F11_no_aa": {
        **F_CONFIGS["F16_production"], "antialiasing": False, "min_scale_pixels": None,
    },
    "F13_pure_dssim": {
        # Pure DSSIM loss (no L1) — shows L1 is better for particles
        **F_CONFIGS["F16_production"], "lambda_dssim": 1.0,
    },
    "F20_dssim_blend": {
        **F_CONFIGS["F16_production"], "lambda_dssim": 0.1,
    },
}


def _run_finetune(name, train_cfg, base_model_dir, shared_data, output_dir, gpu):
    """Run one finetune experiment."""
    ft_dir = output_dir / name
    logs = ft_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    model_dir = ft_dir / "model"
    data_dir = ft_dir / "data"

    result_file = ft_dir / "result.json"
    if result_file.exists():
        print(f"  [Skip] {name}: result exists")
        return json.loads(result_file.read_text())

    try:
        existing = find_checkpoint(model_dir)
        if not existing:
            # Generate finetune training data
            generate_training_data(
                shared_data["vtp"], data_dir, shared_data["normalization"],
                "multi_orbit", "1.0,0.7,0.5", 400, 3840, 2160,
                logs, name)
            prepare_stage_data_dir(data_dir, shared_data["ply"], shared_data["normalization"])

            # Create checkpoint from merged PLY as starting point
            import sys as _sys
            _sys.path.insert(0, str(PARTICLEGS_ROOT))
            from pipelines.finetune import create_checkpoint_from_ply
            model_dir.mkdir(parents=True, exist_ok=True)
            pc_dir = Path(base_model_dir) / "point_cloud"
            iter_dirs = sorted(pc_dir.glob("iteration_*"),
                               key=lambda p: int(p.name.split("_")[1]))
            src_iter = int(iter_dirs[-1].name.split("_")[1]) if iter_dirs else 0
            chkpnt_path, n_gs = create_checkpoint_from_ply(
                str(base_model_dir), str(model_dir), src_iter, train_cfg)
            print(f"    Created checkpoint from merged PLY: {n_gs} Gaussians")

            iteration = run_training(data_dir, model_dir, train_cfg, logs,
                                     f"train_{name}", start_checkpoint=chkpnt_path,
                                     gpu=gpu)
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
    except Exception as e:
        print(f"  FAILED {name}: {e}")
        result = {
            "name": name,
            "masked_psnr": None,
            "psnr": None,
            "size_mb": None,
            "num_gaussians": None,
            "error": str(e),
        }
    save_results(result, result_file)
    return result


def main():
    parser = base_parser("EXP-5: Finetune Recipe Optimization")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Path to merged 8-block model (from EXP-4)")
    parser.add_argument("--sub", type=str, default="all", help="a,b or all")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RUNS_DIR / "exp5"
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    shared_data = ensure_shared_data(gpu=args.gpu) if not args.skip_data_prep else get_shared_data_dict()

    base_model = args.base_model
    if not base_model:
        # Try to find EXP-4 8-block merged model
        exp4_merged = RUNS_DIR / "exp4" / "blocks_8" / "merged"
        if exp4_merged.exists():
            base_model = str(exp4_merged)
        else:
            print("ERROR: No base model found. Run EXP-4 first or provide --base_model")
            return

    results = {}
    subs = args.sub.split(",") if args.sub != "all" else ["a", "b"]

    if "a" in subs:
        print("\n" + "="*70)
        print("EXP-5a: Key Finetune Parameters")
        print("="*70)
        for name, cfg in F_CONFIGS.items():
            r = _run_finetune(name, cfg, base_model, shared_data, output_dir, args.gpu)
            results[name] = r

    if "b" in subs:
        print("\n" + "="*70)
        print("EXP-5b: Negative Results")
        print("="*70)
        for name, cfg in F_NEGATIVE.items():
            r = _run_finetune(name, cfg, base_model, shared_data, output_dir, args.gpu)
            results[name] = r

    # Summary
    headers = ["Config", "Masked PSNR", "Size (MB)"]
    rows = [[n, f"{r['masked_psnr']:.2f}" if r["masked_psnr"] else "N/A", r["size_mb"]]
            for n, r in results.items()]
    print_table(headers, rows)

    save_results(results, output_dir / "results.json")
    print(f"\nEXP-5 complete ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
