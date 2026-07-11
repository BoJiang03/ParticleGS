#!/usr/bin/env python3
"""Master script: run all ParticleGS experiments.

This is the single entry point for reproducing all paper results.
Each experiment saves results to runs/expN/ and prints a summary table.

Usage:
    python -m experiments.run_all [--gpu 0] [--exp 1,2,3]
    python -m experiments.run_all --exp 1          # run only EXP-1
    python -m experiments.run_all --exp 1,4,6      # run selected experiments

    # AE mode: the 8-experiment set covering all 26 enforced metrics, scheduled
    # in parallel across a dedicated multi-GPU node (fits the SC AE 8h budget).
    python -m experiments.run_all --ae --num_gpus 4

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
    EXP-14 (generalization) ───┴─ needs EXP-1 trained model (inference-only)
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
    14: ("exp14_generalization", "Pose + Viz-Param Generalization (needs EXP-1)"),
}

# Experiments that manage their own shared data (not covered by the HACC
# ensure_shared_data() called upfront in main()).
SELF_PREP_EXPERIMENTS = {13}

# AE mode: the reduced experiment set sized for the SC AE ~8 h budget on a
# multi-GPU A100 node. It produces 19 of the 26 enforced metrics — the two
# render-heaviest units are dropped from the fast path: EXP-13 (FIRE-2 full
# retrain) entirely, and EXP-4's 2-block config (EXP-4 runs 4-block only, see
# run_ae_parallel / the sequential branch). The full 26/26 set remains available
# via scripts/reproduce.sh. verify_results.py --ae enforces exactly these 19.
AE_EXPERIMENTS = [1, 4, 6, 7, 8, 11, 14]

# Single-GPU experiments that need EXP-1's trained E25 model on disk first.
NEEDS_EXP1 = {6, 7, 8, 14}

# Approximate cold single-GPU wall-clock (min), for longest-first pool ordering.
_COLD_COST_MIN = {13: 68, 7: 52, 11: 52, 6: 32, 8: 11, 14: 11, 1: 80, 4: 150}


def _exp_cmd(exp_num, gpu, extra_args, skip_data_prep):
    module, _ = ALL_EXPERIMENTS[exp_num]
    cmd = [PYTHON_BIN, "-m", f"experiments.{module}", "--gpu", str(gpu)]
    if skip_data_prep:
        cmd.append("--skip_data_prep")
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def run_experiment(exp_num, module_name, description, gpu, extra_args=None,
                   skip_data_prep=True):
    """Run one experiment as a subprocess (blocking, inherits stdout)."""
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


# ── Parallel (AE) scheduling ────────────────────────────────────────────────
# Each background experiment writes to its own log file (concurrent subprocesses
# cannot share the terminal legibly). Progress is printed on launch/finish.

def _spawn(exp_num, gpu, extra_args, skip_data_prep, log_path, gpus_held=None):
    cmd = _exp_cmd(exp_num, gpu, extra_args, skip_data_prep)
    logf = open(log_path, "w")
    logf.write("$ " + " ".join(cmd) + "\n\n")
    logf.flush()
    proc = subprocess.Popen(cmd, cwd=str(PARTICLEGS_ROOT),
                            stdout=logf, stderr=subprocess.STDOUT)
    print(f"  [launch] EXP-{exp_num:<2d} on GPU {gpu}  (log: {log_path.name})")
    return {"num": exp_num, "gpu": gpu, "proc": proc, "log": logf,
            "t0": time.time(), "path": log_path,
            "gpus_held": list(gpus_held) if gpus_held is not None else [gpu]}


def _reap(job, results):
    rc = job["proc"].wait()
    job["log"].close()
    ok = rc == 0
    results[job["num"]] = ok
    el = (time.time() - job["t0"]) / 60
    print(f"  [ done ] EXP-{job['num']:<2d} {'OK' if ok else 'FAILED'} "
          f"({el:.1f} min, GPU {job['gpu']})")
    return ok


def _run_pool(exp_nums, gpus, logdir, results):
    """Run single-GPU experiments across a GPU pool, longest-first."""
    queue = sorted(exp_nums, key=lambda n: -_COLD_COST_MIN.get(n, 30))
    free = list(gpus)
    running = []
    while queue or running:
        while queue and free:
            num = queue.pop(0)
            gpu = free.pop(0)
            skip_prep = num not in SELF_PREP_EXPERIMENTS
            running.append(_spawn(num, gpu, [], skip_prep, logdir / f"exp{num}.log"))
        time.sleep(5)
        still = []
        for job in running:
            if job["proc"].poll() is None:
                still.append(job)
            else:
                _reap(job, results)
                free.append(job["gpu"])
        running = still


def run_ae_parallel(exp_nums, num_gpus, ae_quick, logdir):
    """AE schedule on a node with physical GPUs 0..num_gpus-1 — one dependency-
    aware pool, NO phase barrier.

    EXP-4 (the long pole) reserves the low GPUs for its whole run — its blocks
    pin physical GPUs 0..k-1 (block_gpu = bi % k), and now render in parallel
    too. Every other experiment is a single-GPU job on the remaining GPU(s):
    EXP-1 runs first, and the moment it finishes EXP-6/7/8/14 (which need its
    model) unlock; EXP-11 is independent and runs whenever a GPU is free. When
    EXP-4 finishes, its GPUs rejoin the pool for the tail.

    This replaces the old two-phase design, whose barrier reaped EXP-4 before
    starting Phase 2 and left the EXP-1 GPU idle for hours behind EXP-4's tail.
    """
    logdir.mkdir(parents=True, exist_ok=True)
    exp_set = set(exp_nums)
    results = {}
    running = []
    free = list(range(num_gpus))

    # EXP-4 reserves the low GPUs for its entire duration.
    e4_gpus = []
    if 4 in exp_set:
        reserve_e1 = (1 in exp_set) and num_gpus > 1
        k4 = (num_gpus - 1) if reserve_e1 else num_gpus
        e4_gpus = free[:k4]
        free = free[k4:]
        blocks = "4" if ae_quick else "2,4"
        running.append(_spawn(4, e4_gpus[0], ["--num_gpus", str(k4), "--blocks", blocks],
                              True, logdir / "exp4.log", gpus_held=e4_gpus))

    # Everything else: single-GPU jobs. 6/7/8/14 wait for EXP-1's model when
    # EXP-1 is in this run; otherwise they assume it exists on disk already.
    have_exp1 = 1 in exp_set
    order = lambda xs: sorted(xs, key=lambda n: -_COLD_COST_MIN.get(n, 30))
    ready, waiting = [], []
    for n in exp_nums:
        if n == 4:
            continue
        (waiting if (n in NEEDS_EXP1 and have_exp1) else ready).append(n)
    ready = order(ready)

    print(f"\n{'='*70}\n=== AE parallel pool: EXP-4 on GPUs {e4_gpus or '-'}; "
          f"ready {ready}" + (f" then {waiting} (after EXP-1)" if waiting else "") +
          f" ===\n{'='*70}")

    while ready or waiting or running:
        while ready and free:
            num = ready.pop(0)
            gpu = free.pop(0)
            extra = ["--ae"] if (num == 1 and ae_quick) else []
            running.append(_spawn(num, gpu, extra, True,
                                  logdir / f"exp{num}.log", gpus_held=[gpu]))
        time.sleep(5)
        still = []
        for job in running:
            if job["proc"].poll() is None:
                still.append(job)
                continue
            _reap(job, results)
            free.extend(job["gpus_held"])         # release GPU(s) back to pool
            if job["num"] == 1:
                if not results.get(1, False):
                    print("  WARNING: EXP-1 failed; EXP-6/7/8/14 depend on its model.")
                ready = order(ready + waiting)     # unlock EXP-1 dependents
                waiting = []
        running = still

    return results


def main():
    parser = argparse.ArgumentParser(description="Run all ParticleGS experiments")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device (base)")
    parser.add_argument("--num_gpus", type=int, default=2,
                        help="GPUs for block training / AE parallel scheduling")
    parser.add_argument("--exp", type=str, default=None,
                        help="Comma-separated experiment numbers (e.g. 1,4,6). Default: all")
    parser.add_argument("--ae", action="store_true",
                        help="AE mode: run the 26-metric experiment set with EXP-1 in "
                             "quick mode and parallel multi-GPU scheduling (SC AE 8h budget)")
    parser.add_argument("--sequential", action="store_true",
                        help="Force sequential execution even in --ae mode")
    parser.add_argument("--logdir", type=str, default=None,
                        help="Directory for per-experiment logs in parallel mode "
                             "(default: <runs>/ae_logs)")
    args = parser.parse_args()

    if args.exp:
        exp_nums = [int(x.strip()) for x in args.exp.split(",")]
    elif args.ae:
        exp_nums = list(AE_EXPERIMENTS)
    else:
        exp_nums = sorted(ALL_EXPERIMENTS.keys())

    parallel = args.ae and not args.sequential and args.num_gpus > 1

    print("="*70)
    print("ParticleGS — Experiment Reproduction" + ("  [AE mode]" if args.ae else ""))
    print("="*70)
    print(f"Experiments: {exp_nums}")
    print(f"GPU base: {args.gpu}   num_gpus: {args.num_gpus}   "
          f"mode: {'parallel' if parallel else 'sequential'}")
    print(f"Output: {RUNS_DIR}")
    print()

    # Step 1: Prepare shared data ONCE (VTP, normalization, PLY, eval GT images)
    # so concurrent experiments never race on generating it.
    print("="*70)
    print("Preparing shared data...")
    print("="*70)
    t0_total = time.time()
    ensure_shared_data(gpu=args.gpu)

    # Step 2: Run experiments
    if parallel:
        for n in exp_nums:
            if n not in ALL_EXPERIMENTS:
                print(f"WARNING: Unknown experiment EXP-{n}, skipping")
        exp_nums = [n for n in exp_nums if n in ALL_EXPERIMENTS]
        logdir = Path(args.logdir) if args.logdir else RUNS_DIR / "ae_logs"
        results = run_ae_parallel(exp_nums, args.num_gpus, args.ae, logdir)
    else:
        results = {}
        for num in exp_nums:
            if num not in ALL_EXPERIMENTS:
                print(f"WARNING: Unknown experiment EXP-{num}, skipping")
                continue
            module, desc = ALL_EXPERIMENTS[num]
            extra = []
            if num == 1 and args.ae:
                extra = ["--ae"]
            if num == 4:
                blocks = "4" if args.ae else "2,4"
                extra = ["--num_gpus", str(args.num_gpus), "--blocks", blocks]
            ok = run_experiment(num, module, desc, args.gpu, extra_args=extra,
                                skip_data_prep=num not in SELF_PREP_EXPERIMENTS)
            results[num] = ok

    # Summary
    total_time = time.time() - t0_total
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    n_ok = sum(1 for v in results.values() if v)
    for num in exp_nums:
        if num in results:
            status = "PASS" if results[num] else "FAIL"
            _, desc = ALL_EXPERIMENTS[num]
            print(f"  EXP-{num:2d}: [{status}] {desc}")
    print(f"\n{n_ok}/{len(results)} experiments OK")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Results directory: {RUNS_DIR}")
    # Non-zero exit if any experiment failed, so wrappers can detect it.
    if any(not v for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
