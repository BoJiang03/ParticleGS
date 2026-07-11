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
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from experiments.common import PARTICLEGS_ROOT, PYTHON_BIN, RUNS_DIR, ensure_shared_data

# Reviewer-facing progress (schedule banner, heartbeat, per-experiment PASS/FAIL
# digest) must appear live even when stdout is redirected to a log / tee / nohup
# / tmux pipe — the common way an AE reviewer captures a multi-hour run. Piped
# stdout is block-buffered by default, so every update sits unseen until ~8 KB
# accumulates; the heartbeat, whose whole point is to prove liveness during long
# silent stretches, is exactly what gets swallowed. Force line buffering so each
# print flushes at its newline regardless of how the run was launched.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

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

# Metric-path prefixes not enforced in the AE fast path (mirrors verify --ae).
AE_SKIP_PREFIXES = ("exp_fire2.", "exp4.blocks_2.", "exp1.exp1c_lcp.")


# ── Reviewer-facing progress + per-experiment result digest ─────────────────
# The AE run launches experiments as background subprocesses that log to files,
# so the top-level terminal would otherwise be quiet for hours. These helpers
# keep the reviewer oriented: overall progress, a heartbeat for long-running
# experiments, and — the moment each experiment finishes — a compact PASS/FAIL
# digest of its enforced metrics against the paper's reference values, plus
# pointers to the results.json and log to inspect.

_PROGRESS = {"t0": None, "total": 0, "done": 0}


def _fmt_dur(seconds):
    m = int(seconds // 60)
    if m < 60:
        return f"{m}m"
    return f"{m // 60}h{m % 60:02d}m"


def _print_progress(tail=""):
    if not _PROGRESS["t0"]:
        return
    el = time.time() - _PROGRESS["t0"]
    bar_n = _PROGRESS["total"] or 1
    filled = int(round(20 * _PROGRESS["done"] / bar_n))
    bar = "#" * filled + "." * (20 - filled)
    print(f"  [progress] [{bar}] {_PROGRESS['done']}/{_PROGRESS['total']} "
          f"experiments done | elapsed {_fmt_dur(el)}{tail}")


_REF_CACHE = {}


def _load_reference():
    if "ref" not in _REF_CACHE:
        p = PARTICLEGS_ROOT / "reference_results.json"
        _REF_CACHE["ref"] = json.loads(p.read_text()) if p.exists() else None
    return _REF_CACHE["ref"]


def _digest_experiment(exp_num):
    """Print a compact PASS/FAIL digest of one experiment's enforced metrics
    the moment it finishes, so reviewers see the headline numbers (and whether
    they match the paper) without waiting for the final verify pass."""
    ref = _load_reference()
    if ref is None:
        return
    # Reuse verify_results as the single source of truth for tolerance logic.
    if str(PARTICLEGS_ROOT) not in sys.path:
        sys.path.insert(0, str(PARTICLEGS_ROOT))
    try:
        import verify_results as vr
    except Exception:
        return

    exp_name = f"exp{exp_num}"
    metrics = [m for m in ref["metrics"]
               if m["path"].split(".", 1)[0] == exp_name
               and not m["path"].startswith(AE_SKIP_PREFIXES)]
    if not metrics:
        return

    res_path = RUNS_DIR / exp_name / "results.json"
    log_hint = RUNS_DIR / "ae_logs" / f"{exp_name}.log"
    if not res_path.exists():
        print(f"  └─ EXP-{exp_num} produced no results.json "
              f"(see {log_hint}) — metrics will show MISSING in verify")
        return
    data = json.loads(res_path.read_text())

    print(f"  ┌─ EXP-{exp_num} results vs paper "
          f"({len(metrics)} enforced metric{'s' if len(metrics) != 1 else ''}):")
    for m in metrics:
        inner = m["path"].split(".", 1)[1]
        actual = vr.get_nested(data, inner)
        unit = m.get("unit", "")
        cls = m["class"]
        short = m["path"].split(".", 1)[1]
        if cls == "hw_independent":
            ok, _ = vr.check_hw_independent(actual, m.get("expected"),
                                            m.get("tolerance_abs"),
                                            m.get("tolerance_rel"))
            tag = "PASS" if ok else "FAIL"
        elif cls == "trend":
            ok, _ = vr.check_trend(actual, m.get("trend_rule", "gt"),
                                   m.get("trend_threshold", 0))
            tag = "PASS" if ok else "FAIL"
        else:  # hw_dependent — reported only, never scored
            tag = "INFO"
        av = vr.format_val(actual, unit)
        ev = vr.format_val(m.get("expected"), unit)
        print(f"  │   [{tag}] {short:<38} {av:>12}  (paper {ev})")
    print(f"  └─ details: {res_path}   log: {log_hint}")


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
    _PROGRESS["done"] += 1
    print(f"\n  [ done ] EXP-{job['num']:<2d} {'OK' if ok else 'FAILED'} "
          f"({el:.1f} min, GPU {job['gpu']})")
    _digest_experiment(job["num"])
    _print_progress()
    return ok


_HEARTBEAT = {"last": 0.0}


def _heartbeat(running, interval=120):
    """Every `interval` seconds, print which experiments are still running and
    for how long — so the terminal isn't silent during multi-hour experiments."""
    now = time.time()
    if not running or now - _HEARTBEAT["last"] < interval:
        return
    _HEARTBEAT["last"] = now
    parts = [f"EXP-{j['num']} ({_fmt_dur(now - j['t0'])}, GPU {j['gpu']})"
             for j in sorted(running, key=lambda j: j["num"])]
    _print_progress(tail=f" | running: {', '.join(parts)}")


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
        _heartbeat(running)
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


def _run_ae_seg2(exp_nums, gpus, ae_quick, cpu_env, logdir, results):
    """Segment 2 (quality metrics): give EXP-4 every GPU for its parallel block-
    training round (4 blocks -> 4 GPUs = one round instead of two), then, the
    moment EXP-4 signals `blocktrain_done` and drops into its single-GPU
    merge/eval/finetune tail, release the non-base GPUs to the EXP-7/8/14 pool so
    they run concurrently with that tail instead of the GPUs sitting idle.

    EXP-4 pins its tail to the base GPU (gpus[0]); 7/8/14 take gpus[1:]. All
    share the CPU-thread cap so concurrent ParaView renders (EXP-4's finetune
    render + the pool's GT renders) don't oversubscribe the CPU. If EXP-4 isn't
    in this segment, 7/8/14 simply pool across all GPUs."""
    base = gpus[0]
    others = gpus[1:]
    running = []
    exp4_job = None

    if 4 in exp_nums:
        blocks = "4" if ae_quick else "2,4"
        n_blk = max(int(x) for x in blocks.split(","))
        marker = RUNS_DIR / "exp4" / f"blocks_{n_blk}" / "blocktrain_done"
        marker.unlink(missing_ok=True)  # clear any stale signal from a prior run
        exp4_args = ["--num_gpus", str(len(gpus)), "--blocks", blocks]
        if ae_quick:
            # AE fast path (1+4+1 → 1+1): consume the shipped 4 sub-block models
            # instead of training them live. EXP-4 signals blocktrain_done almost
            # immediately and drops into its merge→finetune tail, freeing the
            # non-base GPUs for the 7/8/14 pool right away.
            exp4_args.append("--use_pretrained_blocks")
        mode = "shipped sub-blocks → merge/finetune" if ae_quick else "live block training"
        print(f"  EXP-4 {n_blk}-block ({mode}); GPUs {others} free up for "
              f"EXP-7/8/14 once EXP-4 signals blocktrain_done.")
        exp4_job = _spawn(4, base, exp4_args,
                          True, logdir / "exp4.log", gpus_held=list(gpus), env=cpu_env)
        # Hold all GPUs until EXP-4 finishes block-training (marker) or exits.
        while exp4_job["proc"].poll() is None and not marker.exists():
            time.sleep(5)
            _heartbeat([exp4_job])
        if exp4_job["proc"].poll() is not None:
            _reap(exp4_job, results)          # cached/failed before the marker
            exp4_job = None
            free = list(gpus)
        else:
            print(f"\n  [pool] EXP-4 block-training done — releasing GPUs {others} "
                  f"to the EXP-7/8/14 pool while EXP-4 finetunes on GPU {base}")
            free = list(others)               # base stays with EXP-4's tail
    else:
        free = list(gpus)

    queue = sorted([n for n in exp_nums if n != 4],
                   key=lambda n: -_COLD_COST_MIN.get(n, 30))
    while queue or running or exp4_job is not None:
        while queue and free:
            num = queue.pop(0)
            gpu = free.pop(0)
            running.append(_spawn(num, gpu, [], True, logdir / f"exp{num}.log",
                                  gpus_held=[gpu], env=cpu_env))
        time.sleep(5)
        _heartbeat(running + ([exp4_job] if exp4_job else []))
        still = []
        for job in running:
            if job["proc"].poll() is None:
                still.append(job)
                continue
            _reap(job, results)
            free.extend(job["gpus_held"])
        running = still
        # EXP-4's single-GPU tail: when it finishes, its base GPU rejoins the pool.
        if exp4_job is not None and exp4_job["proc"].poll() is not None:
            _reap(exp4_job, results)
            free.append(base)
            exp4_job = None


def run_ae_parallel(exp_nums, num_gpus, ae_quick, logdir):
    """Three-segment AE schedule — parallelize what can be, isolate what can't.

      Segment 1 (isolated): EXP-1 solo, full CPU cores — clean single-block
                            end-to-end train time + the E25 model 6/7/8/14 need.
      Segment 2 (mixed):    EXP-4 takes ALL GPUs for its block-training round,
                            then releases the non-base GPUs to the EXP-7/8/14
                            quality-metric pool while its single-GPU finetune
                            tail runs — every GPU stays busy. Each proc is capped
                            to cores//num_gpus threads so concurrent ParaView
                            renders (CPU-heavy) don't thrash the CPU.
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

    _PROGRESS["t0"] = time.time()
    _PROGRESS["total"] = len([n for n in exp_nums if n in ALL_EXPERIMENTS])
    _PROGRESS["done"] = 0

    seg2 = [n for n in exp_nums if n in PARALLEL_METRIC]
    seg3 = [n for n in ISOLATED_TIMING if n != 1 and n in exp_set]
    print(f"\nSchedule ({num_gpus} GPUs, {cores} CPU cores):")
    print(f"  Segment 1 [isolated]: EXP-1"
          f"{' (quick)' if ae_quick else ''} — end-to-end train time + E25 model")
    print(f"  Segment 2 [mixed]:    {seg2 or '(none)'} — EXP-4 all-GPU block train, "
          f"then 7/8/14 overlap its finetune tail")
    print(f"  Segment 3 [isolated]: {seg3 or '(none)'} — clean FPS / time / memory")

    # ---- Segment 1: EXP-1 isolated ----
    if 1 in exp_set:
        print(f"\n{'='*70}\n=== AE Segment 1/3 (isolated): EXP-1 solo — clean "
              f"end-to-end train time + E25 model ===\n{'='*70}")
        job = _spawn(1, gpus[0], (["--ae"] if ae_quick else []), True,
                     logdir / "exp1.log", env=full_env)
        _reap(job, results)
        if not results.get(1, False):
            print("  WARNING: EXP-1 failed; EXP-6/7/8/14 depend on its model.")

    # ---- Segment 2: EXP-4 all-GPU block train, then 7/8/14 overlap the tail ----
    if seg2:
        print(f"\n{'='*70}\n=== AE Segment 2/3 (mixed): {seg2} across {num_gpus} "
              f"GPUs (CPU cap {max(1, cores // max(1, num_gpus))} threads/proc) "
              f"===\n{'='*70}")
        _run_ae_seg2(seg2, gpus, ae_quick, par_env, logdir, results)

    # ---- Segment 3: isolated timing/perf experiments, one at a time ----
    for n in ISOLATED_TIMING:
        if n == 1 or n not in exp_set:
            continue
        print(f"\n{'='*70}\n=== AE Segment 3/3 (isolated): EXP-{n} solo — valid "
              f"timing/perf ===\n{'='*70}")
        job = _spawn(n, gpus[0], (["--ae"] if ae_quick else []), True,
                     logdir / f"exp{n}.log", env=full_env)
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
        _PROGRESS["t0"] = time.time()
        _PROGRESS["total"] = len([n for n in exp_nums if n in ALL_EXPERIMENTS])
        _PROGRESS["done"] = 0
        for num in exp_nums:
            if num not in ALL_EXPERIMENTS:
                print(f"WARNING: Unknown experiment EXP-{num}, skipping")
                continue
            module, desc = ALL_EXPERIMENTS[num]
            # AE mode passes --ae to every experiment (report-only reductions;
            # no-op where unused). Full reproduce.sh leaves args.ae False -> extra
            # stays empty here and EXP-4 keeps blocks "2,4" with no pretrained.
            extra = ["--ae"] if args.ae else []
            if num == 4:
                blocks = "4" if args.ae else "2,4"
                extra += ["--num_gpus", str(args.num_gpus), "--blocks", blocks]
                if args.ae:
                    extra.append("--use_pretrained_blocks")  # 1+4+1 → 1+1
            ok = run_experiment(num, module, desc, args.gpu, extra_args=extra,
                                skip_data_prep=num not in SELF_PREP_EXPERIMENTS)
            results[num] = ok
            _PROGRESS["done"] += 1
            if args.ae:
                _digest_experiment(num)
            _print_progress()

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

    # Guide the reviewer to the next step: the aggregated tables + the metric
    # verification that decides AE pass/fail.
    if args.ae:
        print(f"\nNext:")
        print(f"  • Per-experiment numbers vs paper were printed above as each "
              f"experiment finished.")
        print(f"  • Per-experiment logs:  {RUNS_DIR / 'ae_logs'}/exp*.log")
        print(f"  • Aggregated tables:    {RUNS_DIR / 'summary'}/  "
              f"(scripts/aggregate_results.py)")
        print(f"  • Final verification:   python verify_results.py --ae  "
              f"(the 19 enforced AE metrics)")
        if not results.get(1, True):
            print(f"  ! EXP-1 FAILED — EXP-6/7/8/14 depend on its E25 model and "
                  f"will report MISSING. Re-run after fixing EXP-1.")

    # Non-zero exit if any experiment failed, so wrappers can detect it.
    if any(not v for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
