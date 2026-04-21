#!/usr/bin/env python3
"""Master script: run all ParticleGS experiments sequentially.

This is the single entry point for reproducing all paper results.
Each experiment saves results to runs/expN/ and prints a summary table.

Usage:
    python -m experiments.run_all [--gpu 0] [--exp 1,2,3]
    python -m experiments.run_all --exp 1          # run only EXP-1
    python -m experiments.run_all --exp 1,4,6      # run selected experiments

Experiment dependency graph:
    EXP-1 (E25 single-block) ──┐
    EXP-2 (ablation)           │  independent
    EXP-3 (VizMapper)          │  independent
    EXP-4 (block training) ────┤
    EXP-5 (finetune) ──────────┤─ needs EXP-4 merged model
    EXP-6 (render benchmark) ──┤─ needs any trained model
    EXP-7 (recovery) ──────────┤─ needs trained models
    EXP-8 (3-way comparison) ──┤─ needs trained model
    EXP-9 (end-to-end)         │  independent (re-trains from scratch)
    EXP-10 (efficiency)        │  independent (re-trains from scratch)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from experiments.common import PARTICLEGS_ROOT, PYTHON_BIN, RUNS_DIR, ensure_shared_data

ALL_EXPERIMENTS = {
    1: ("exp1_rate_distortion", "3DGS vs SZ3 Rate-Distortion"),
    2: ("exp2_training_ablation", "Training Strategy Ablation"),
    3: ("exp3_vizmapper_ablation", "VizMapper Ablation"),
    4: ("exp4_block_training", "Block Training Scan"),
    5: ("exp5_finetune_recipes", "Finetune Recipe Optimization"),
    6: ("exp6_render_benchmark", "Rendering Benchmark"),
    7: ("exp7_particle_recovery", "Particle Recovery"),
    8: ("exp8_three_way_comparison", "Three-Way Comparison"),
    9: ("exp9_end_to_end", "End-to-End Validation"),
    10: ("exp10_pipeline_efficiency", "Pipeline Efficiency"),
    11: ("exp11_resource_profiling", "Resource Profiling (data loading, merge time)"),
    12: ("exp12_add_ssim", "SSIM Column for Rate-Distortion Table"),
    13: ("exp_fire2_generalization", "FIRE-2 Cross-Dataset Generalization"),
}

# Experiments that manage their own shared data (not covered by the HACC
# ensure_shared_data() called upfront in main()).
SELF_PREP_EXPERIMENTS = {13}


def run_experiment(exp_num, module_name, description, gpu, extra_args=None,
                   skip_data_prep=True):
    """Run one experiment as a subprocess."""
    print(f"\n{'#'*70}")
    print(f"# EXP-{exp_num}: {description}")
    print(f"{'#'*70}\n")

    cmd = [PYTHON_BIN, "-m", f"experiments.{module_name}", "--gpu", str(gpu)]
    if skip_data_prep:
        cmd.append("--skip_data_prep")
    if extra_args:
        cmd.extend(extra_args)

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PARTICLEGS_ROOT))
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"\n  EXP-{exp_num} {status} ({elapsed/60:.1f} min)")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all ParticleGS experiments")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device")
    parser.add_argument("--num_gpus", type=int, default=2, help="GPUs for block training")
    parser.add_argument("--exp", type=str, default=None,
                        help="Comma-separated experiment numbers (e.g. 1,4,6). Default: all")
    args = parser.parse_args()

    if args.exp:
        exp_nums = [int(x.strip()) for x in args.exp.split(",")]
    else:
        exp_nums = sorted(ALL_EXPERIMENTS.keys())

    print("="*70)
    print("ParticleGS — Full Experiment Reproduction")
    print("="*70)
    print(f"Experiments: {exp_nums}")
    print(f"GPU: {args.gpu}")
    print(f"Output: {RUNS_DIR}")
    print()

    # Step 1: Prepare shared data (VTP, normalization, PLY, eval GT images)
    print("="*70)
    print("Preparing shared data...")
    print("="*70)
    t0_total = time.time()
    ensure_shared_data(gpu=args.gpu)

    # Step 2: Run experiments
    results = {}
    for num in exp_nums:
        if num not in ALL_EXPERIMENTS:
            print(f"WARNING: Unknown experiment EXP-{num}, skipping")
            continue
        module, desc = ALL_EXPERIMENTS[num]
        extra = []
        if num == 4:
            extra = ["--num_gpus", str(args.num_gpus), "--blocks", "2,4"]
        ok = run_experiment(num, module, desc, args.gpu, extra_args=extra,
                            skip_data_prep=num not in SELF_PREP_EXPERIMENTS)
        results[num] = ok

    # Summary
    total_time = time.time() - t0_total
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    for num in exp_nums:
        if num in results:
            status = "PASS" if results[num] else "FAIL"
            _, desc = ALL_EXPERIMENTS[num]
            print(f"  EXP-{num:2d}: [{status}] {desc}")
    print(f"\nTotal time: {total_time/3600:.1f} hours")
    print(f"Results directory: {RUNS_DIR}")


if __name__ == "__main__":
    main()
