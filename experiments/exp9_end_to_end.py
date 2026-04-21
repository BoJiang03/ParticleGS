#!/usr/bin/env python3
"""EXP-9: End-to-End Pipeline Validation.

Proves the complete pipeline from raw data to evaluated model is reproducible.

Sub-experiments:
  9a: Single-block (E25 config) — target: ~26.35 dB masked PSNR
  9b: Multi-block (8-block + F16 finetune) — target: ~27.58 dB masked PSNR

Usage:
    python -m experiments.exp9_end_to_end [--gpu 0]
"""

import json
import time
from pathlib import Path

from experiments.common import *
from experiments.exp1_rate_distortion import E25_STAGES, run_e25_training
from experiments.exp4_block_training import (
    F16_FINETUNE, PER_BLOCK_STAGES, PER_BLOCK_TRAIN,
    finetune_merged, merge_blocks, partition_raw_data,
    prepare_block_data, train_block,
)


def run_exp9a(output_dir, shared_data, gpu):
    """EXP-9a: Single-block end-to-end."""
    print("\n" + "="*70)
    print("EXP-9a: Single-Block End-to-End (E25)")
    print("="*70)

    model_dir, iteration = run_e25_training(output_dir / "9a", shared_data, gpu=gpu)
    last_train = E25_STAGES[-1]["train"]
    eval_results = evaluate_model(model_dir, iteration, shared_data,
                                  output_dir / "9a" / "logs", gpu=gpu,
                                  factor_delta_opacity=last_train["factor_delta_opacity"],
                                  factor_delta_scale=last_train["factor_delta_scale"],
                                  min_opacity_clamp=last_train["min_opacity_clamp"])
    stats = get_model_stats(model_dir, iteration)

    result = {
        "masked_psnr": eval_results["avg"]["masked_psnr"],
        "psnr": eval_results["avg"]["psnr"],
        "size_mb": round(stats["size_mb"], 1),
        "num_gaussians": stats["num_gaussians"],
        "eval": eval_results,
    }
    print(f"\n  9a Single-block: {result['masked_psnr']:.2f} dB, {result['size_mb']} MB")
    print(f"  Reference: 26.35 dB, 11.2 MB")
    return result


def run_exp9b(output_dir, shared_data, gpu, num_gpus=2):
    """EXP-9b: 8-block + F16 finetune end-to-end."""
    print("\n" + "="*70)
    print("EXP-9b: 8-Block + F16 Finetune End-to-End")
    print("="*70)

    n_blocks = 8
    run_dir = output_dir / "9b"
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    # Partition
    partition_info, partition_dir = partition_raw_data(n_blocks, run_dir, logs)

    # Train blocks
    block_models = []
    block_run_dirs = []
    for bi in range(n_blocks):
        block_raw = partition_dir / f"block_{bi}"
        block_dir = run_dir / f"block_{bi:02d}"
        block_gpu = bi % num_gpus
        block_run_dirs.append(block_dir)
        block_data = prepare_block_data(bi, block_raw, block_dir, shared_data, logs)
        model_dir, iteration = train_block(bi, block_data, block_dir, logs, block_gpu)
        block_models.append((model_dir, iteration))

    # Merge
    merged_dir = merge_blocks(block_models, block_run_dirs,
                              shared_data["normalization"], run_dir, logs)

    # Eval merged — find the actual iteration from the merged PLY
    from pathlib import Path as _Path
    merged_pc = _Path(merged_dir) / "point_cloud"
    merged_iter_dirs = sorted(merged_pc.glob("iteration_*"),
                              key=lambda p: int(p.name.split("_")[1])) if merged_pc.exists() else []
    merged_iter = int(merged_iter_dirs[-1].name.split("_")[1]) if merged_iter_dirs else 0
    merged_eval = evaluate_model(merged_dir, merged_iter, shared_data, logs, gpu=gpu,
                                 factor_delta_opacity=0.3, factor_delta_scale=0.1)
    merged_stats = get_model_stats(merged_dir, merged_iter)

    # Finetune
    ft_model, ft_iter = finetune_merged(merged_dir, shared_data, run_dir, logs, gpu)

    # Eval finetuned
    ft_eval = evaluate_model(ft_model, ft_iter, shared_data, logs, gpu=gpu,
                             factor_delta_opacity=F16_FINETUNE["factor_delta_opacity"],
                             factor_delta_scale=F16_FINETUNE["factor_delta_scale"],
                             min_opacity_clamp=F16_FINETUNE["min_opacity_clamp"])
    ft_stats = get_model_stats(ft_model, ft_iter)

    result = {
        "merged": {
            "masked_psnr": merged_eval["avg"]["masked_psnr"],
            "size_mb": round(merged_stats["size_mb"], 1),
        },
        "finetuned": {
            "masked_psnr": ft_eval["avg"]["masked_psnr"],
            "psnr": ft_eval["avg"]["psnr"],
            "size_mb": round(ft_stats["size_mb"], 1),
            "num_gaussians": ft_stats["num_gaussians"],
            "eval": ft_eval,
        },
    }
    print(f"\n  9b Merged: {result['merged']['masked_psnr']:.2f} dB")
    print(f"  9b Finetuned: {result['finetuned']['masked_psnr']:.2f} dB, {result['finetuned']['size_mb']} MB")
    print(f"  Reference: merged ~22.43, finetuned 27.62 dB, 77.2 MB")
    return result


def main():
    parser = base_parser("EXP-9: End-to-End Pipeline Validation")
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--sub", type=str, default="all", help="a,b or all")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RUNS_DIR / "exp9"
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    shared_data = ensure_shared_data(gpu=args.gpu) if not args.skip_data_prep else get_shared_data_dict()

    results = {}
    subs = args.sub.split(",") if args.sub != "all" else ["a", "b"]
    if "a" in subs: results["9a"] = run_exp9a(output_dir, shared_data, args.gpu)
    if "b" in subs: results["9b"] = run_exp9b(output_dir, shared_data, args.gpu, args.num_gpus)

    save_results(results, output_dir / "results.json")
    print(f"\nEXP-9 complete ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
