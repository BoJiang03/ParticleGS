#!/usr/bin/env python3
"""Master script: run all ParticleGS experiments.

This is the single entry point for reproducing all paper results.
Each experiment saves results to runs/expN/ and prints a summary table.

Usage:
    python -m experiments.run_all [--gpu 0] [--exp 1,2,3]
    python -m experiments.run_all --exp 1          # run only EXP-1
    python -m experiments.run_all --exp 1,4,6      # run selected experiments

    # AE mode: the 7-experiment set covering the 18 enforced AE metrics,
    # scheduled in parallel across the available GPUs (fits the SC AE 8h budget).
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
import threading
import time
from contextlib import contextmanager
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
# single graphics-class GPU (more GPUs cut wall-clock). It produces 18 of the
# 26 enforced metrics — three render-heavy units are dropped from the fast path:
# EXP-13 (FIRE-2 full retrain) and the LCP baseline entirely, and EXP-4's
# 2-block config (EXP-4 runs 4-block only, see run_ae_parallel / the sequential
# branch). The full 26/26 set remains available via scripts/reproduce.sh.
# verify_results.py --ae enforces exactly these 18.
AE_EXPERIMENTS = [1, 4, 6, 7, 8, 11, 14]

# Single-GPU experiments that need EXP-1's trained E25 model on disk first.
NEEDS_EXP1 = {6, 7, 8, 14}

# Approximate cold single-GPU wall-clock (min), for longest-first pool ordering.
_COLD_COST_MIN = {13: 68, 7: 52, 11: 52, 6: 32, 8: 11, 14: 11, 1: 80, 4: 150}

# Expected AE-mode wall-clock per experiment (minutes) on the VALIDATED reviewer
# recipe — 1x Quadro RTX 6000, single GPU (Chameleon gpu_rtx_6000 node,
# CC-Ubuntu24.04-CUDA image). Printed at launch, in the heartbeat, and at
# finish, so a reviewer watching a multi-hour run can tell "slow but alive"
# from "stuck". These are per-experiment expectations only; shared data prep
# (VTP conversion + eval GT render, before EXP-1) adds ~20 min on top, and
# the conda env build (~10 min) + 3.9 GB data download (~3 min) happen even
# earlier, in reproduce_ae.sh, outside these timers.
# Measured on the 2026-07 Chameleon validation run (fast path, sequential
# single GPU, all 18 metrics PASS): EXP-1 33.9, EXP-4 44.5, EXP-6 36.2,
# EXP-7 87.3, EXP-8 23.6, EXP-11 72.6, EXP-14 75.9, prep ~20 min.
_AE_EXPECTED_MIN = {1: 34, 4: 45, 6: 36, 7: 87, 8: 24, 11: 73, 14: 76}
_AE_EXPECTED_GPU = "1x RTX 6000"

# Metric-path prefixes not enforced in the AE fast path (mirrors verify --ae).
AE_SKIP_PREFIXES = ("exp_fire2.", "exp4.blocks_2.", "exp1.exp1c_lcp.")


# ── Color (interactive terminals only) ──────────────────────────────────────
# Plain text when stdout is piped/tee'd to a log, when NO_COLOR is set, or on
# a dumb terminal — ANSI escapes in a captured AE log would be noise.
_USE_COLOR = (hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
              and "NO_COLOR" not in os.environ
              and os.environ.get("TERM") != "dumb")

_GREEN, _RED, _YELLOW, _CYAN, _DIM = "32", "1;31", "33", "36", "2"
# Structural text: bold for top-level section banners, bold-cyan for the
# per-experiment / per-segment separators — so a reviewer scrolling the run
# can spot experiment boundaries at a glance.
_BOLD, _BCYAN = "1", "1;36"


def _c(code, s):
    return f"\033[{code}m{s}\033[0m" if _USE_COLOR else s


def _ctag(tag):
    """Color a PASS/FAIL/INFO digest tag."""
    return _c({"PASS": _GREEN, "FAIL": _RED, "INFO": _CYAN}.get(tag, _YELLOW),
              tag)


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


def _tree_activity(root_pid):
    """(cpu_seconds, io_chars) consumed by `root_pid` and all live descendants,
    read from /proc. Two complementary progress signals, compared between two
    ticker samples: compute-bound phases (training, ParaView render, SZ3, the
    CPU particle compare) burn CPU continuously, while IO-bound phases (the
    3.4 GB data download, VTP writes) may use little CPU but keep rchar/wchar
    (all read/write syscalls, sockets included) climbing. A genuinely hung
    tree — deadlock, futex wait, dead EGL context — goes flat on BOTH. CPU
    counts reaped children via cutime/cstime; a spinning-but-stuck process is
    the one case this cannot distinguish from work (the per-experiment
    expected times and the final verify pass are the backstop for that).
    Returns None if /proc is unavailable (non-Linux) — callers then fall back
    to elapsed-only lines."""
    try:
        hz = os.sysconf("SC_CLK_TCK")
        procs = {}
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                with open(f"/proc/{d}/stat") as f:
                    st = f.read()
            except OSError:
                continue  # process exited mid-scan
            # comm (field 2) may contain spaces/parens — split after last ')'
            rest = st[st.rindex(")") + 2:].split()
            # rest[0]=state, [1]=ppid, [11]/[12]=utime/stime, [13]/[14]=cutime/cstime
            cpu = (int(rest[11]) + int(rest[12])
                   + int(rest[13]) + int(rest[14]))
            io = 0
            try:
                with open(f"/proc/{d}/io") as f:
                    for line in f:
                        if line.startswith(("rchar:", "wchar:")):
                            io += int(line.split()[1])
            except OSError:
                pass  # exited mid-scan or not ours; CPU signal still counts
            procs[int(d)] = (int(rest[1]), cpu, io)
        kids = {}
        for pid, (ppid, _, _) in procs.items():
            kids.setdefault(ppid, []).append(pid)
        cpu_total, io_total, stack = 0, 0, [root_pid]
        while stack:
            p = stack.pop()
            if p in procs:
                cpu_total += procs[p][1]
                io_total += procs[p][2]
                stack.extend(kids.get(p, []))
        return cpu_total / hz, io_total
    except Exception:
        return None


@contextmanager
def _alive_ticker(tag, expected_min=None, interval=300, pid=None):
    """Print a periodic liveness line while a long blocking call runs in the
    foreground. Several phases are silent for many minutes at a stretch
    (ParaView GT rendering, SZ3 compression, EXP-7's CPU particle compare),
    and a reviewer watching the terminal cannot tell 'quiet but working' from
    'hung'. The line is NOT a bare we-are-still-looping print: each tick
    samples the CPU time and IO volume of the process tree rooted at `pid`
    and compares them to the previous tick — real work moves at least one of
    the two, so [alive] reports what was consumed in the window, and a window
    flat on BOTH prints [warn ] ... possibly stuck instead. stdout is NOT
    piped or intercepted: the child keeps the terminal, so tqdm bars and
    pvbatch output are untouched."""
    t0 = time.time()
    stop = threading.Event()
    state = {"act": _tree_activity(pid) if pid else None}

    def _tick():
        while not stop.wait(interval):
            el = time.time() - t0
            # "42m/~75m (56%)" — elapsed vs expected on the validated GPU
            # (named once in the banner), and the fraction of it used.
            prog = (f"{_fmt_dur(el)}/~{expected_min}m "
                    f"({100 * el / (expected_min * 60):.0f}%)"
                    if expected_min else _fmt_dur(el))
            act = _tree_activity(pid) if pid else None
            if act is not None and state["act"] is not None:
                dcpu = act[0] - state["act"][0]
                dio = act[1] - state["act"][1]
                used = (f"cpu {dcpu:.0f}s io {dio / 1048576:.0f}MiB "
                        f"/{_fmt_dur(interval)}")
                # ≥1% of one core, or ≥1 MiB of read/write syscall traffic,
                # counts as progress; flat on both means nothing is moving.
                if dcpu < 0.01 * interval and dio < 1048576:
                    print(_c(_YELLOW,
                             f"  [warn ] {tag}  {prog}  {used} — possibly "
                             f"stuck; check output above / nvidia-smi"))
                else:
                    print(_c(_DIM, f"  [alive] {tag}  {prog}  {used}"))
            else:
                print(_c(_DIM, f"  [alive] {tag}  {prog}"))
            state["act"] = act

    th = threading.Thread(target=_tick, daemon=True)
    th.start()
    try:
        yield
    finally:
        stop.set()
        th.join(timeout=1)


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


# Set by main() when parallel mode redirects each experiment's output to
# per-experiment log files under runs/ae_logs/. Stays None in sequential mode,
# where experiments inherit the terminal and no log files exist — never point
# reviewers at a directory that was not created.
_AE_LOG_DIR = None


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
    where = (f"see {_AE_LOG_DIR / (exp_name + '.log')}" if _AE_LOG_DIR
             else "see the output above")
    if not res_path.exists():
        print(_c(_YELLOW, f"  └─ EXP-{exp_num} produced no results.json "
                          f"({where}) — metrics will show MISSING "
                          f"in verify"))
        return
    data = json.loads(res_path.read_text())

    print(_c(_CYAN, f"  ┌─ EXP-{exp_num} results vs paper "
                    f"({len(metrics)} enforced "
                    f"metric{'s' if len(metrics) != 1 else ''}):"))
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
        print(f"  │   [{_ctag(tag)}] {short:<38} {av:>12}  (paper {ev})")
    log_part = f"   log: {_AE_LOG_DIR / (exp_name + '.log')}" if _AE_LOG_DIR else ""
    print(f"  └─ details: {res_path}{log_part}")


def _exp_cmd(exp_num, gpu, extra_args, skip_data_prep):
    module, _ = ALL_EXPERIMENTS[exp_num]
    cmd = [PYTHON_BIN, "-m", f"experiments.{module}", "--gpu", str(gpu)]
    if skip_data_prep:
        cmd.append("--skip_data_prep")
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def run_experiment(exp_num, module_name, description, gpu, extra_args=None,
                   skip_data_prep=True, expected_min=None):
    """Run one experiment as a subprocess (blocking, inherits stdout)."""
    print(_c(_BCYAN, f"\n{'#'*70}"))
    print(_c(_BCYAN, f"# EXP-{exp_num}: {description}"))
    if expected_min:
        print(_c(_BCYAN, f"# expected ~{expected_min} min on {_AE_EXPECTED_GPU} "
                         f"(validated recipe; other GPUs differ)"))
    print(_c(_BCYAN, f"{'#'*70}") + "\n")

    cmd = [PYTHON_BIN, "-m", f"experiments.{module_name}", "--gpu", str(gpu)]
    if skip_data_prep:
        cmd.append("--skip_data_prep")
    if extra_args:
        cmd.extend(extra_args)

    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=str(PARTICLEGS_ROOT))
    with _alive_ticker(f"EXP-{exp_num}", expected_min, pid=proc.pid):
        rc = proc.wait()
    elapsed = time.time() - t0

    status = _c(_GREEN, "OK") if rc == 0 else _c(_RED, "FAILED")
    t = (f"{elapsed / 60:.1f}m/~{expected_min}m "
         f"({100 * elapsed / (expected_min * 60):.0f}%)"
         if expected_min else f"{elapsed / 60:.1f}m")
    print(f"\n  EXP-{exp_num} {status}  {t}")
    return rc == 0


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
    exp_min = _AE_EXPECTED_MIN.get(exp_num)
    eta = f", expected ~{exp_min}m" if exp_min else ""
    print(f"  [{_c(_CYAN, 'launch')}] EXP-{exp_num:<2d} on GPU {gpu}  "
          f"(log: {log_path.name}{eta})")
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
    exp_min = _AE_EXPECTED_MIN.get(job["num"])
    t = (f"{el:.1f}m/~{exp_min}m ({100 * el / exp_min:.0f}%)"
         if exp_min else f"{el:.1f}m")
    print(f"\n  [ done ] EXP-{job['num']:<2d} "
          f"{_c(_GREEN, 'OK') if ok else _c(_RED, 'FAILED')}  "
          f"{t}  GPU {job['gpu']}")
    _digest_experiment(job["num"])
    _print_progress()
    return ok


def _wait_and_reap(job, results):
    """Block on one background job, but keep the 120 s heartbeat going — used
    by the isolated segments (EXP-1/6/11), whose output goes to a log file and
    would otherwise leave the terminal silent for the whole experiment."""
    while job["proc"].poll() is None:
        time.sleep(5)
        _heartbeat([job])
    return _reap(job, results)


_HEARTBEAT = {"last": 0.0}


def _heartbeat(running, interval=120):
    """Every `interval` seconds, print which experiments are still running and
    for how long — so the terminal isn't silent during multi-hour experiments."""
    now = time.time()
    if not running or now - _HEARTBEAT["last"] < interval:
        return
    _HEARTBEAT["last"] = now
    parts = []
    for j in sorted(running, key=lambda j: j["num"]):
        exp_min = _AE_EXPECTED_MIN.get(j["num"])
        el = now - j["t0"]
        eta = (f"/~{exp_min}m {100 * el / (exp_min * 60):.0f}%"
               if exp_min else "")
        # Same stuck-detection as _alive_ticker: compare the job's process-tree
        # CPU time and IO volume against the previous heartbeat — flat on both
        # means possibly hung.
        act = _tree_activity(j["proc"].pid)
        mark = ""
        if act is not None and j.get("hb_act") is not None:
            dt = now - j["hb_t"]
            if (dt > 0 and (act[0] - j["hb_act"][0]) < 0.01 * dt
                    and (act[1] - j["hb_act"][1]) < 1048576):
                mark = " " + _c(_YELLOW,
                                "NO CPU/IO ACTIVITY — possibly stuck, check log")
        j["hb_act"], j["hb_t"] = act, now
        parts.append(f"EXP-{j['num']} ({_fmt_dur(now - j['t0'])}{eta}, "
                     f"GPU {j['gpu']}{mark})")
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
        print(_c(_BCYAN, f"\n{'='*70}\n=== AE Segment 1/3 (isolated): EXP-1 solo — "
              f"{'shipped E25 -> eval + SZ3 iso-CR' if ae_quick else 'end-to-end train time + E25 model'} "
              f"===\n{'='*70}"))
        job = _spawn(1, gpus[0],
                     (["--ae", "--use_pretrained_e25"] if ae_quick else []), True,
                     logdir / "exp1.log", env=full_env)
        _wait_and_reap(job, results)
        if not results.get(1, False):
            print("  WARNING: EXP-1 failed; EXP-6/7/8/14 depend on its model.")

    # ---- Segment 2: EXP-4 all-GPU block train, then 7/8/14 overlap the tail ----
    if seg2:
        print(_c(_BCYAN, f"\n{'='*70}\n=== AE Segment 2/3 (mixed): {seg2} across {num_gpus} "
              f"GPUs (CPU cap {max(1, cores // max(1, num_gpus))} threads/proc) "
              f"===\n{'='*70}"))
        _run_ae_seg2(seg2, gpus, ae_quick, par_env, logdir, results)

    # ---- Segment 3: isolated timing/perf experiments, one at a time ----
    for n in ISOLATED_TIMING:
        if n == 1 or n not in exp_set:
            continue
        print(_c(_BCYAN, f"\n{'='*70}\n=== AE Segment 3/3 (isolated): EXP-{n} solo — valid "
              f"timing/perf ===\n{'='*70}"))
        job = _spawn(n, gpus[0], (["--ae"] if ae_quick else []), True,
                     logdir / f"exp{n}.log", env=full_env)
        _wait_and_reap(job, results)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run all ParticleGS experiments")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device (base)")
    parser.add_argument("--num_gpus", type=int, default=2,
                        help="GPUs for block training / AE parallel scheduling")
    parser.add_argument("--exp", type=str, default=None,
                        help="Comma-separated experiment numbers (e.g. 1,4,6). Default: all")
    parser.add_argument("--ae", action="store_true",
                        help="AE mode: run the 18-metric AE experiment set with EXP-1 in "
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

    print(_c(_BOLD, "="*70))
    print(_c(_BOLD, "ParticleGS — Experiment Reproduction"
                    + ("  [AE mode]" if args.ae else "")))
    print(_c(_BOLD, "="*70))
    print(f"Experiments: {exp_nums}")
    print(f"GPU base: {args.gpu}   num_gpus: {args.num_gpus}   "
          f"mode: {'parallel' if parallel else 'sequential'}")
    print(f"Output: {RUNS_DIR}")
    if args.ae:
        known = [n for n in exp_nums if n in _AE_EXPECTED_MIN]
        if known:
            total = sum(_AE_EXPECTED_MIN[n] for n in known)
            print(f"Expected per-experiment wall-clock on the validated recipe "
                  f"({_AE_EXPECTED_GPU}, single GPU):")
            print("  " + "  ".join(f"EXP-{n} ~{_AE_EXPECTED_MIN[n]}m"
                                   for n in known))
            print(f"  sequential sum ~{_fmt_dur(total * 60)} + ~20m shared data "
                  f"prep (multi-GPU overlaps Segment 2; other GPUs differ)")
    print()

    # Step 1: Prepare shared data ONCE (VTP, normalization, PLY, eval GT images)
    # so concurrent experiments never race on generating it.
    print(_c(_BOLD, "="*70))
    print(_c(_BOLD, "Preparing shared data..."))
    print(_c(_BOLD, "="*70))
    if args.ae:
        print(f"(expected ~20 min on {_AE_EXPECTED_GPU} when cold: one-time VTP "
              f"conversion + eval GT render; seconds if already cached)")
    t0_total = time.time()
    t0_prep = time.time()
    # Monitor our own process tree: ensure_shared_data runs in-process but does
    # its heavy lifting (VTP conversion, pvbatch GT renders) in subprocesses.
    with _alive_ticker("shared data prep", 20 if args.ae else None,
                       pid=os.getpid()):
        ensure_shared_data(gpu=args.gpu)
    prep_min = (time.time() - t0_prep) / 60
    if args.ae:
        print(f"  shared data prep done  {prep_min:.1f}m/~20m "
              f"({100 * prep_min / 20:.0f}%)")

    # Step 2: Run experiments
    if parallel:
        for n in exp_nums:
            if n not in ALL_EXPERIMENTS:
                print(f"WARNING: Unknown experiment EXP-{n}, skipping")
        exp_nums = [n for n in exp_nums if n in ALL_EXPERIMENTS]
        logdir = Path(args.logdir) if args.logdir else RUNS_DIR / "ae_logs"
        global _AE_LOG_DIR
        _AE_LOG_DIR = logdir
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
            if num == 1 and args.ae:
                extra.append("--use_pretrained_e25")  # ship E25 -> eval-only
            if num == 4:
                blocks = "4" if args.ae else "2,4"
                extra += ["--num_gpus", str(args.num_gpus), "--blocks", blocks]
                if args.ae:
                    extra.append("--use_pretrained_blocks")  # 1+4+1 → 1+1
            ok = run_experiment(num, module, desc, args.gpu, extra_args=extra,
                                skip_data_prep=num not in SELF_PREP_EXPERIMENTS,
                                expected_min=(_AE_EXPECTED_MIN.get(num)
                                              if args.ae else None))
            results[num] = ok
            _PROGRESS["done"] += 1
            if args.ae:
                _digest_experiment(num)
            _print_progress()

    # Summary
    total_time = time.time() - t0_total
    print(_c(_BOLD, f"\n{'='*70}"))
    print(_c(_BOLD, "FINAL SUMMARY"))
    print(_c(_BOLD, f"{'='*70}"))
    n_ok = sum(1 for v in results.values() if v)
    for num in exp_nums:
        if num in results:
            status = _ctag("PASS" if results[num] else "FAIL")
            _, desc = ALL_EXPERIMENTS[num]
            print(f"  EXP-{num:2d}: [{status}] {desc}")
    print(f"\n{_c(_GREEN if n_ok == len(results) else _RED, f'{n_ok}/{len(results)} experiments OK')}")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Results directory: {RUNS_DIR}")

    # Guide the reviewer to the next step: the aggregated tables + the metric
    # verification that decides AE pass/fail.
    if args.ae:
        print(f"\nNext:")
        print(f"  • Per-experiment numbers vs paper were printed above as each "
              f"experiment finished.")
        if _AE_LOG_DIR:
            print(f"  • Per-experiment logs:  {_AE_LOG_DIR}/exp*.log")
        print(f"  • Aggregated tables:    {RUNS_DIR / 'summary'}/  "
              f"(scripts/aggregate_results.py)")
        print(f"  • Final verification:   python verify_results.py --ae  "
              f"(the 18 enforced AE metrics)")
        if not results.get(1, True):
            print(f"  ! EXP-1 FAILED — EXP-6/7/8/14 depend on its E25 model and "
                  f"will report MISSING. Re-run after fixing EXP-1.")

    # Non-zero exit if any experiment failed, so wrappers can detect it.
    if any(not v for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
