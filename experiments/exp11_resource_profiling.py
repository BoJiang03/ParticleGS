#!/usr/bin/env python3
"""EXP-11: Resource Profiling — Memory, Loading Time, Merge & Finetune Cost.

Measures:
  1. GPU memory (inference): 3DGS single-block and 8-block models
  2. GPU memory (ParaView): 280M-particle VTP via pvbatch
  3. GPU memory (training peak): short training run to capture peak allocation
  4. Model loading time: PLY + VizMapper
  5. ParaView VTP loading time
  6. Merge cost: time to merge 8 block PLYs
  7. Finetune cost: time for 60k-iteration finetuning

Usage:
    python -m experiments.exp11_resource_profiling [--gpu 0]
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from experiments.common import *

sys.path.insert(0, str(PARTICLEGS_ROOT))


# ── Helpers ──────────────────────────────────────────────────────────────

def find_best_blocks_dir():
    """Return (n_blocks, path) for the largest trained blocks_N dir, or None.

    Prefers 8 → 4 → 2. Requires evidence of a completed S3 stage in block_00
    so stale/partial dirs (e.g. from a prior crash) are skipped. The default
    reviewer run trains only 2- and 4-block configurations, so this falls
    back to whichever block count is available.
    """
    for n in (8, 4, 2):
        d = RUNS_DIR / "exp4" / f"blocks_{n}"
        s3_model = d / "block_00" / "02_S3_mix" / "model"
        if d.exists() and s3_model.exists() and find_checkpoint(s3_model):
            return n, d
    return None


# ── 1. GPU memory (inference) ────────────────────────────────────────────

def measure_inference_memory(model_dir, iteration, label="model"):
    """Load a 3DGS model (+ VizMapper if present) and measure GPU memory."""
    import torch
    from particlegs.model.gaussian_model import GaussianModel

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    # Load Gaussians
    ply_path = Path(model_dir) / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    gaussians = GaussianModel(sh_degree=0)
    gaussians.load_ply(str(ply_path))
    n_gaussians = gaussians.get_xyz.shape[0]

    mem_after_ply = torch.cuda.memory_allocated()

    # Load VizMapper if present
    has_viz_mapper = False
    mapper_path = Path(model_dir) / f"viz_mapper_{iteration}.pth"
    if mapper_path.exists():
        from particlegs.model.viz_mapper import VizMapper
        ckpt = torch.load(str(mapper_path), map_location="cuda", weights_only=False)
        cfg = ckpt.get("config", {})
        mapper = VizMapper(
            hidden_dim=cfg.get("hidden_dim", 64),
            num_layers=cfg.get("num_layers", 2),
            use_xyz=cfg.get("use_xyz", False),
            factor_delta_scale=ckpt.get("factor_delta_scale", 0.3),
            factor_delta_opacity=ckpt.get("factor_delta_opacity", 0.8),
            min_opacity_clamp=ckpt.get("min_opacity_clamp", 0.1),
        ).cuda()
        mapper.load_state_dict(ckpt["model"])
        mapper.eval()
        has_viz_mapper = True

    mem_after_all = torch.cuda.memory_allocated()

    # Run one render to capture render-time peak
    from benchmarks.fps_benchmark import make_orbit_cameras, _Pipe
    from particlegs.renderer import render as gs_render
    pipe = _Pipe(antialiasing=True)
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    cameras = make_orbit_cameras(1, 1920, 1080)

    if has_viz_mapper:
        from benchmarks.fps_benchmark import OptimizedVizMapperRunner
        runner = OptimizedVizMapperRunner(mapper, gaussians)
        viz_s, viz_o = runner(1.0, 1.0)
    else:
        viz_s, viz_o = 1.0, 1.0
    gs_render(cameras[0], gaussians, pipe, bg,
              viz_scale_factor=viz_s, viz_opacity_factor=viz_o)
    torch.cuda.synchronize()

    mem_peak = torch.cuda.max_memory_allocated()

    result = {
        "label": label,
        "num_gaussians": n_gaussians,
        "has_viz_mapper": has_viz_mapper,
        "model_memory_mb": round((mem_after_all - mem_before) / 1e6, 1),
        "peak_render_memory_mb": round(mem_peak / 1e6, 1),
    }
    print(f"  {label}: {n_gaussians:,} Gaussians, "
          f"model={result['model_memory_mb']} MB, "
          f"peak_render={result['peak_render_memory_mb']} MB")

    # Cleanup
    del gaussians
    if has_viz_mapper:
        del mapper, runner
    torch.cuda.empty_cache()

    return result


# ── 2. GPU memory (ParaView) ────────────────────────────────────────────

def measure_paraview_memory(vtp_path, logs_dir):
    """Measure ParaView GPU memory for loading and rendering 280M particles.

    Uses nvidia-smi polling from the parent process (avoids GPU index
    mismatch inside pvbatch's EGL context).
    """
    import threading

    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    # Get baseline memory (before pvbatch starts)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "-i", gpu_id],
            text=True)
        mem_before = int(out.strip())
    except Exception:
        mem_before = 0

    # pvbatch script: load VTP, render, then sleep so we can measure
    script_content = '''
import sys, time
from paraview.simple import *
data_source = OpenDataFile(sys.argv[1])
rv = CreateRenderView()
rv.ViewSize = [1920, 1080]
d = Show(data_source, rv)
d.Representation = "Point Gaussian"
d.ShaderPreset = "Gaussian Blur"
d.GaussianRadius = 0.01
d.Opacity = 0.05
rv.ResetCamera()
Render()
print("PV_READY", flush=True)
time.sleep(5)  # Hold GPU memory for measurement
print("PV_DONE", flush=True)
'''
    script_path = logs_dir / "_pv_mem_probe.py"
    script_path.write_text(script_content)

    cmd = [PVBATCH_BIN, "--force-offscreen-rendering",
           "--", str(script_path), str(vtp_path)]

    # Start pvbatch and poll memory
    peak_mem = [mem_before]
    stop_event = threading.Event()

    def poller():
        while not stop_event.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits", "-i", gpu_id],
                    text=True, timeout=5)
                mem = int(out.strip())
                peak_mem[0] = max(peak_mem[0], mem)
            except Exception:
                pass
            stop_event.wait(0.3)

    poll_thread = threading.Thread(target=poller, daemon=True)
    poll_thread.start()

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except Exception as e:
        print(f"  WARNING: ParaView memory probe failed: {e}")

    stop_event.set()
    poll_thread.join(timeout=2)

    delta = peak_mem[0] - mem_before
    result = {
        "mem_before_mb": mem_before,
        "peak_mb": peak_mem[0],
        "delta_mb": delta,
    }
    print(f"  ParaView: baseline={mem_before} MB, peak={peak_mem[0]} MB, "
          f"delta=+{delta} MB")
    return result


# ── 3. GPU memory (training peak) ───────────────────────────────────────

def _poll_gpu_memory(gpu_id, interval=0.5, stop_event=None):
    """Poll nvidia-smi for GPU memory usage. Returns peak value in MB."""
    import threading
    peak = 0
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits", "-i", str(gpu_id)],
                text=True, timeout=5)
            mem = int(out.strip())
            peak = max(peak, mem)
        except Exception:
            pass
        stop_event.wait(interval)
    return peak


def measure_training_peak_memory(shared_data, output_dir, logs_dir, gpu=0):
    """Run a short training session and report peak GPU memory.

    Generates 6K training data (matching real S3 stage) then runs 2000
    iterations with densification to capture realistic peak memory.
    Uses nvidia-smi polling since training runs in a subprocess.
    """
    import threading

    from experiments.exp1_rate_distortion import E25_STAGES
    s3_stage = E25_STAGES[-1]
    s3_cfg = s3_stage["train"]
    config = dict(s3_cfg)
    config["iterations"] = 2000  # enough for densification to kick in
    config["densify_until_iter"] = 1500

    # Generate 6K training data (same resolution as real S3)
    mc = s3_stage["mix_cfg"]
    data_dir = output_dir / "train_mem_probe" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    imgs_dir = data_dir / "images"
    if not imgs_dir.exists() or len(list(imgs_dir.glob("*.png"))) == 0:
        print(f"  Generating 6K training data ({mc['width']}x{mc['height']})...")
        generate_mix_training_data(
            shared_data["vtp"], data_dir, shared_data["normalization"],
            mc["ext_orbit_radii"], mc["ext_num_frames"], mc["ext_seed"],
            mc["int_num_frames"], mc["int_seed"], mc["int_bounds_scale"],
            mc["width"], mc["height"], mc["mix_ratio"],
            logs_dir, "train_mem_data")
        prepare_stage_data_dir(data_dir, shared_data["ply"],
                               shared_data["normalization"])
    else:
        print(f"  [Skip] 6K training data exists ({len(list(imgs_dir.glob('*.png')))} images)")
    work_dir = output_dir / "train_mem_probe"
    model_dir = work_dir / "model"
    # Clean model dir from previous probe (keep data)
    if model_dir.exists():
        import shutil as _sh
        _sh.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Get baseline GPU memory
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "-i", str(gpu)],
            text=True)
        mem_before = int(out.strip())
    except Exception:
        mem_before = 0

    # Start polling in background
    stop_event = threading.Event()
    peak_mem = [0]

    def poller():
        peak_mem[0] = _poll_gpu_memory(gpu, interval=0.3, stop_event=stop_event)

    poll_thread = threading.Thread(target=poller, daemon=True)
    poll_thread.start()

    run_training(data_dir, model_dir, config, logs_dir,
                 "train_mem_probe", gpu=gpu)

    stop_event.set()
    poll_thread.join(timeout=2)

    # Cleanup entire probe directory (6K data + model)
    import shutil
    if work_dir.exists():
        shutil.rmtree(work_dir)

    delta = peak_mem[0] - mem_before
    result = {
        "mem_before_mb": mem_before,
        "peak_mb": peak_mem[0],
        "delta_mb": delta,
    }
    print(f"  Training peak GPU memory: {peak_mem[0]} MB "
          f"(+{delta} MB over baseline {mem_before} MB)")
    return result


# ── 4. Model loading time ───────────────────────────────────────────────

def measure_loading_time(model_dir, iteration, label="model", n_runs=5):
    """Time PLY + VizMapper loading."""
    import torch
    from particlegs.model.gaussian_model import GaussianModel

    ply_path = Path(model_dir) / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    mapper_path = Path(model_dir) / f"viz_mapper_{iteration}.pth"

    times = []
    for _ in range(n_runs):
        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        g = GaussianModel(sh_degree=0)
        g.load_ply(str(ply_path))
        if mapper_path.exists():
            torch.load(str(mapper_path), map_location="cuda", weights_only=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        del g

    avg_ms = sum(times) / len(times) * 1000
    result = {
        "label": label,
        "has_viz_mapper": mapper_path.exists(),
        "avg_load_ms": round(avg_ms, 1),
        "n_runs": n_runs,
    }
    print(f"  {label}: {avg_ms:.1f} ms (avg of {n_runs} runs)")
    return result


# ── 5. ParaView VTP loading time ────────────────────────────────────────

def measure_paraview_load_time(vtp_path, logs_dir):
    """Time ParaView VTP loading via pvbatch."""
    script_content = '''
import json, sys, time
from paraview.simple import *

vtp_path = sys.argv[1]
t0 = time.perf_counter()
data_source = OpenDataFile(vtp_path)
rv = CreateRenderView()
rv.ViewSize = [1920, 1080]
d = Show(data_source, rv)
d.Representation = "Point Gaussian"
d.ShaderPreset = "Gaussian Blur"
d.GaussianRadius = 0.01
d.Opacity = 0.05
rv.ResetCamera()
Render()
elapsed = time.perf_counter() - t0

result = {"load_and_first_render_s": round(elapsed, 2)}
print("PV_LOAD_JSON:" + json.dumps(result))
'''
    script_path = logs_dir / "_pv_load_probe.py"
    script_path.write_text(script_content)

    cmd = [PVBATCH_BIN, "--force-offscreen-rendering",
           "--", str(script_path), str(vtp_path)]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT,
                                      text=True, timeout=300)
        for line in out.splitlines():
            if line.startswith("PV_LOAD_JSON:"):
                result = json.loads(line.split(":", 1)[1])
                print(f"  ParaView load + first render: "
                      f"{result['load_and_first_render_s']:.1f} s")
                return result
    except Exception as e:
        print(f"  WARNING: ParaView load time probe failed: {e}")
    return None


# ── 6. Merge cost ────────────────────────────────────────────────────────

def measure_merge_cost(output_dir, logs_dir):
    """Time the merge of N block PLYs into a single model.

    Uses the largest trained blocks_N dir (prefers 8, falls back to 4, 2).
    """
    from experiments.exp4_block_training import merge_blocks

    best = find_best_blocks_dir()
    if best is None:
        print("  WARNING: no blocks_N dir found, skipping merge cost")
        return None
    n_blocks, blocks_dir = best

    # Collect block models
    block_models = []
    block_run_dirs = []
    for bi in range(n_blocks):
        block_dir = blocks_dir / f"block_{bi:02d}"
        model_dir = block_dir / "02_S3_mix" / "model"
        chk = find_checkpoint(model_dir)
        if not chk:
            print(f"  WARNING: block {bi} checkpoint not found")
            return None
        iteration = int(chk.stem.replace("chkpnt", ""))
        block_models.append((str(model_dir), iteration))
        block_run_dirs.append(block_dir)

    target_norm = SHARED_DIR / "normalization.json"
    merge_out = output_dir / "merge_test"

    t0 = time.perf_counter()
    merge_blocks(block_models, block_run_dirs, target_norm, merge_out, logs_dir)
    elapsed = time.perf_counter() - t0

    # Cleanup
    import shutil
    if merge_out.exists():
        shutil.rmtree(merge_out)

    result = {
        "n_blocks": n_blocks,
        f"merge_{n_blocks}block_s": round(elapsed, 2),
    }
    print(f"  Merge {n_blocks} blocks: {elapsed:.1f} s")
    return result


# ── 7. Finetune cost ────────────────────────────────────────────────────

def measure_finetune_cost(shared_data, output_dir, logs_dir, gpu=0):
    """Measure finetune wall-clock time for the merged model.

    On a reviewer's first pass the finetune runs for real (60k
    iterations) and is timed. `finetune_merged` short-circuits if a
    finetuned checkpoint already exists, so subsequent EXP-11 reruns
    would otherwise report 0 s — detect that case up front and skip
    the metric rather than writing a misleading zero.
    """
    from experiments.exp4_block_training import F16_FINETUNE, finetune_merged

    best = find_best_blocks_dir()
    if best is None:
        print("  WARNING: no blocks_N dir found, skipping finetune cost")
        return None
    n_blocks, blocks_dir = best
    merged_dir = blocks_dir / "merged"
    if not merged_dir.exists():
        print(f"  WARNING: merged model not found in blocks_{n_blocks}, skipping finetune cost")
        return None

    ft_out = output_dir / "finetune_timing"
    ft_model_dir = ft_out / "finetuned" / "model"
    if find_checkpoint(ft_model_dir) is not None:
        print(f"  Finetune timing: cached checkpoint exists in {ft_model_dir} — "
              f"skipping (rerun would short-circuit to 0 s)")
        return {
            "n_blocks": n_blocks,
            "finetune_time_s": None,
            "finetune_time_min": None,
            "iterations": F16_FINETUNE["iterations"],
            "source": "skipped_cache_hit",
        }

    ft_out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ft_model, ft_iter = finetune_merged(
        merged_dir, shared_data, ft_out, logs_dir, gpu)
    elapsed = time.perf_counter() - t0

    result = {
        "n_blocks": n_blocks,
        "finetune_time_s": round(elapsed, 1),
        "finetune_time_min": round(elapsed / 60, 1),
        "iterations": F16_FINETUNE["iterations"],
        "source": "measured",
    }
    print(f"  Finetune {n_blocks}-block merged (60k iter): {elapsed/60:.1f} min")
    return result


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = base_parser("EXP-11: Resource Profiling")
    parser.add_argument("--skip_finetune", action="store_true",
                        help="Skip finetune timing (slow)")
    parser.add_argument("--skip_train_mem", action="store_true",
                        help="Skip training peak memory measurement")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RUNS_DIR / "exp11"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs = output_dir / "logs"
    logs.mkdir(exist_ok=True)
    t0 = time.time()

    shared_data = (ensure_shared_data(gpu=args.gpu)
                   if not args.skip_data_prep else get_shared_data_dict())

    results = {}

    # ── 1. GPU memory (inference) ────────────────────────────────────────
    print("\n" + "="*70)
    print("[1] GPU Memory — Inference")
    print("="*70)

    import torch
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    results["inference_memory"] = {}

    # Single-block model (exp10 or exp2)
    for candidate in [
        RUNS_DIR / "exp10" / "e25" / "02_S3_mix_6k" / "model",
        RUNS_DIR / "exp2" / "2g_with_internal" / "02_S3_mix_6k" / "model",
    ]:
        if candidate.exists():
            chk = find_checkpoint(candidate)
            if chk:
                it = int(chk.stem.replace("chkpnt", ""))
                results["inference_memory"]["single_block"] = \
                    measure_inference_memory(candidate, it, "single-block (170k)")
                break

    # N-block finetuned model (prefer 8 → 4 → 2)
    ft_dir = None
    ft_n_blocks = None
    for n in (8, 4, 2):
        candidate = RUNS_DIR / "exp4" / f"blocks_{n}" / "finetuned" / "model"
        if candidate.exists() and find_checkpoint(candidate):
            ft_dir = candidate
            ft_n_blocks = n
            break
    if ft_dir is not None:
        chk = find_checkpoint(ft_dir)
        it = int(chk.stem.replace("chkpnt", ""))
        results["inference_memory"][f"{ft_n_blocks}_block_finetuned"] = \
            measure_inference_memory(ft_dir, it, f"{ft_n_blocks}-block finetuned")

    # ── 2. GPU memory (ParaView) ─────────────────────────────────────────
    print("\n" + "="*70)
    print("[2] GPU Memory — ParaView (280M particles)")
    print("="*70)
    results["paraview_memory"] = measure_paraview_memory(
        shared_data["vtp"], logs)

    # ── 3. GPU memory (training peak) ────────────────────────────────────
    if not args.skip_train_mem:
        print("\n" + "="*70)
        print("[3] GPU Memory — Training Peak")
        print("="*70)
        results["training_memory"] = measure_training_peak_memory(
            shared_data, output_dir, logs, gpu=args.gpu)
    else:
        print("\n[3] Skipped training peak memory")

    # ── 4. Model loading time ────────────────────────────────────────────
    print("\n" + "="*70)
    print("[4] Model Loading Time")
    print("="*70)
    results["loading_time"] = {}

    for candidate in [
        RUNS_DIR / "exp10" / "e25" / "02_S3_mix_6k" / "model",
        RUNS_DIR / "exp2" / "2g_with_internal" / "02_S3_mix_6k" / "model",
    ]:
        if candidate.exists():
            chk = find_checkpoint(candidate)
            if chk:
                it = int(chk.stem.replace("chkpnt", ""))
                results["loading_time"]["single_block"] = \
                    measure_loading_time(candidate, it, "single-block (170k)")
                break

    if ft_dir is not None and ft_dir.exists():
        chk = find_checkpoint(ft_dir)
        if chk:
            it = int(chk.stem.replace("chkpnt", ""))
            results["loading_time"][f"{ft_n_blocks}_block_finetuned"] = \
                measure_loading_time(ft_dir, it, f"{ft_n_blocks}-block finetuned")

    # ── 5. ParaView VTP loading time ─────────────────────────────────────
    print("\n" + "="*70)
    print("[5] ParaView VTP Loading Time")
    print("="*70)
    results["paraview_load_time"] = measure_paraview_load_time(
        shared_data["vtp"], logs)

    # ── 6. Merge cost ────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("[6] Merge Cost (8 blocks)")
    print("="*70)
    results["merge_cost"] = measure_merge_cost(output_dir, logs)

    # ── 7. Finetune cost ─────────────────────────────────────────────────
    if not args.skip_finetune:
        print("\n" + "="*70)
        print("[7] Finetune Cost (60k iterations)")
        print("="*70)
        results["finetune_cost"] = measure_finetune_cost(
            shared_data, output_dir, logs, gpu=args.gpu)
    else:
        print("\n[7] Skipped finetune timing")

    # ── Summary ──────────────────────────────────────────────────────────
    total = time.time() - t0
    print(f"\n{'='*70}")
    print("EXP-11: Resource Profiling — Summary")
    print(f"{'='*70}")

    if "inference_memory" in results:
        for k, v in results["inference_memory"].items():
            print(f"  Inference mem ({v['label']}): "
                  f"model={v['model_memory_mb']} MB, "
                  f"peak={v['peak_render_memory_mb']} MB")
    if results.get("paraview_memory"):
        pv = results["paraview_memory"]
        print(f"  ParaView GPU mem: +{pv['delta_mb']} MB "
              f"(peak {pv['peak_mb']} MB)")
    if results.get("training_memory"):
        tm = results["training_memory"]
        print(f"  Training peak mem: {tm['peak_mb']} MB "
              f"(+{tm['delta_mb']} MB over baseline)")
    if "loading_time" in results:
        for k, v in results["loading_time"].items():
            print(f"  Load time ({v['label']}): {v['avg_load_ms']:.1f} ms")
    if results.get("paraview_load_time"):
        print(f"  ParaView load: "
              f"{results['paraview_load_time']['load_and_first_render_s']:.1f} s")
    if results.get("merge_cost"):
        mc = results["merge_cost"]
        merge_key = next((k for k in mc if k.startswith("merge_") and k.endswith("block_s")), None)
        if merge_key:
            n = merge_key.replace("merge_", "").replace("block_s", "")
            print(f"  Merge {n} blocks: {mc[merge_key]:.1f} s")
    if results.get("finetune_cost"):
        print(f"  Finetune: {results['finetune_cost']['finetune_time_min']:.1f} min")

    save_results(results, output_dir / "results.json")
    print(f"\nEXP-11 complete ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
