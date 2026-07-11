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
import math
import os
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

def _cpu_env(threads):
    """Subprocess env with CPU-thread caps. Rendering (ParaView/VTK) is CPU-heavy
    while training is GPU-heavy; in the parallel segment several ParaView renders
    run at once, so cap each process's SMP/OpenMP/BLAS threads to cores//concurrency
    to keep concurrent renderers from all grabbing every core and thrashing the
    CPU. In isolated segments each experiment gets the full core count."""
    env = os.environ.copy()
    t = str(max(1, int(threads)))
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VTK_SMP_MAX_THREADS", "NUMBA_NUM_THREADS"):
        env[k] = t
    return env


def _spawn(exp_num, gpu, extra_args, skip_data_prep, log_path, gpus_held=None, env=None):
    cmd = _exp_cmd(exp_num, gpu, extra_args, skip_data_prep)
    logf = open(log_path, "w")
    logf.write("$ " + " ".join(cmd) + "\n\n")
    logf.flush()
    proc = subprocess.Popen(cmd, cwd=str(PARTICLEGS_ROOT),
                            stdout=logf, stderr=subprocess.STDOUT, env=env)
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


# Timing/performance-sensitive experiments: their FPS / wall-clock / peak-memory
# numbers are corrupted by concurrent CPU rendering or GPU sharing, so they must
# run ISOLATED (exclusive node). exp1 = clean single-block end-to-end train time;
# exp6 = 3dgs/paraview FPS + the ENFORCED speedup trend; exp11 = finetune time +
# training peak memory. (exp6.speedup is enforced, so contamination could even
# FAIL verification, not just misreport.)
ISOLATED_TIMING = [1, 6, 11]
# Contention-independent quality metrics (PSNR / Gaussians / size / CR / SSIM /
# correlation) — deterministic given the model, safe to parallelize.
PARALLEL_METRIC = [4, 7, 8, 14]


def _run_ae_pool(exp_nums, gpus, ae_quick, cpu_env, logdir, results):
    """Parallel pool for the quality-metric segment. EXP-4 reserves the fewest
    GPUs for its block round count (freeing the rest); 7/8/14 are single-GPU.
    All get the shared CPU-thread cap so concurrent ParaView renders don't
    oversubscribe the CPU."""
    num_gpus = len(gpus)
    free = list(gpus)
    running = []

    if 4 in exp_nums:
        blocks = "4" if ae_quick else "2,4"
        override = os.environ.get("PARTICLEGS_AE_EXP4_GPUS")
        if override:
            k4 = max(1, min(num_gpus, int(override)))
        elif num_gpus > 1:
            n_blk = max(int(x) for x in blocks.split(","))
            rounds = math.ceil(n_blk / (num_gpus - 1))
            k4 = max(1, math.ceil(n_blk / rounds))
        else:
            k4 = num_gpus
        e4 = free[:k4]
        free = free[k4:]
        running.append(_spawn(4, e4[0], ["--num_gpus", str(k4), "--blocks", blocks],
                              True, logdir / "exp4.log", gpus_held=e4, env=cpu_env))

    queue = sorted([n for n in exp_nums if n != 4],
                   key=lambda n: -_COLD_COST_MIN.get(n, 30))
    while queue or running:
        while queue and free:
            num = queue.pop(0)
            gpu = free.pop(0)
            running.append(_spawn(num, gpu, [], True, logdir / f"exp{num}.log",
                                  gpus_held=[gpu], env=cpu_env))
        time.sleep(5)
        still = []
        for job in running:
            if job["proc"].poll() is None:
                still.append(job)
                continue
            _reap(job, results)
            free.extend(job["gpus_held"])
        running = still


def run_ae_parallel(exp_nums, num_gpus, ae_quick, logdir):
    """Three-segment AE schedule — parallelize what can be, isolate what can't.

      Segment 1 (isolated): EXP-1 solo, full CPU cores — clean single-block
                            end-to-end train time + the E25 model 6/7/8/14 need.
      Segment 2 (parallel): EXP-4/7/8/14 quality metrics across all GPUs, each
                            capped to cores//num_gpus threads so concurrent
                            ParaView renders (CPU-heavy) don't thrash the CPU.
      Segment 3 (isolated): EXP-6 then EXP-11, one at a time, node otherwise
                            idle, full cores — valid FPS / finetune-time / memory.

    Isolation is required because rendering is CPU-heavy and training is GPU-heavy
    with very different utilization; overlapping them corrupts any timing/FPS/
    memory measurement (and EXP-6's enforced speedup). Quality metrics are
    deterministic, so Segment 2 parallelizes safely.
    """
    logdir.mkdir(parents=True, exist_ok=True)
    gpus = list(range(num_gpus))
    exp_set = set(exp_nums)
    results = {}
    cores = os.cpu_count() or (num_gpus * 8)
    full_env = _cpu_env(cores)                          # isolated: all cores
    par_env = _cpu_env(max(1, cores // max(1, num_gpus)))  # parallel: share cores

    # ---- Segment 1: EXP-1 isolated ----
    if 1 in exp_set:
        print(f"\n{'='*70}\n=== AE Segment 1 (isolated): EXP-1 solo — clean "
              f"end-to-end train time + E25 model ===\n{'='*70}")
        job = _spawn(1, gpus[0], (["--ae"] if ae_quick else []), True,
                     logdir / "exp1.log", env=full_env)
        _reap(job, results)
        if not results.get(1, False):
            print("  WARNING: EXP-1 failed; EXP-6/7/8/14 depend on its model.")

    # ---- Segment 2: parallel quality-metric experiments ----
    seg2 = [n for n in exp_nums if n in PARALLEL_METRIC]
    if seg2:
        print(f"\n{'='*70}\n=== AE Segment 2 (parallel): {seg2} across {num_gpus} "
              f"GPUs (CPU cap {max(1, cores // max(1, num_gpus))} threads/proc) "
              f"===\n{'='*70}")
        _run_ae_pool(seg2, gpus, ae_quick, par_env, logdir, results)

    # ---- Segment 3: isolated timing/perf experiments, one at a time ----
    for n in ISOLATED_TIMING:
        if n == 1 or n not in exp_set:
            continue
        print(f"\n{'='*70}\n=== AE Segment 3 (isolated): EXP-{n} solo — valid "
              f"timing/perf ===\n{'='*70}")
        job = _spawn(n, gpus[0], [], True, logdir / f"exp{n}.log", env=full_env)
        _reap(job, results)

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
