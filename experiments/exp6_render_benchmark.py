#!/usr/bin/env python3
"""EXP-6: Rendering Performance Benchmark — ParaView vs 3DGS.

Compares pure rendering FPS (no disk I/O) between:
  - ParaView (280M particles, Point Gaussian representation)
  - 3DGS CUDA rasterizer (~170K-1M Gaussians)

Usage:
    python -m experiments.exp6_render_benchmark [--gpu 0] [--model_dir <path>]
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

from experiments.common import *


def benchmark_paraview_fps(vtp_path, norm_path, output_dir, logs_dir):
    """Run pure ParaView render FPS benchmark via pvbatch."""
    result_file = output_dir / "pv_fps.json"
    if result_file.exists():
        print("  [Skip] ParaView FPS results exist")
        return json.loads(result_file.read_text())

    pv_script = PARTICLEGS_ROOT / "benchmarks" / "pv_fps_benchmark.py"
    pv_args = [
        "--vtp_path", str(vtp_path),
        "--normalization", str(norm_path),
        "--width", "1920", "--height", "1080",
        "--n_frames", "80", "--warmup", "10",
        "--orbit_radii", "1.0,0.7,0.5",
        "--output", str(result_file),
    ]
    run_cmd(pvbatch_cmd(pv_script, *pv_args),
            log_path=logs_dir / "pv_fps_benchmark.log")

    if result_file.exists():
        return json.loads(result_file.read_text())
    return None


def benchmark_3dgs_fps(model_dir, iteration, output_dir, logs_dir):
    """Run pure CUDA rasterizer FPS benchmark."""
    result_file = output_dir / "gs_fps.json"
    if result_file.exists():
        print("  [Skip] 3DGS FPS results exist")
        return json.loads(result_file.read_text())

    fps_script = PARTICLEGS_ROOT / "benchmarks" / "fps_benchmark.py"
    cmd = [
        PYTHON_BIN, str(fps_script),
        "--model_dir", str(model_dir),
        "--iteration", str(iteration),
        "--resolutions", "1920x1080",
        "--n_iters", "300", "--warmup", "20",
        "--output", str(result_file),
    ]
    run_cmd(cmd, log_path=logs_dir / "gs_fps_benchmark.log")

    if result_file.exists():
        return json.loads(result_file.read_text())
    return None


def main():
    parser = base_parser("EXP-6: Rendering Performance Benchmark")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to trained 3DGS model")
    parser.add_argument("--iteration", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RUNS_DIR / "exp6"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs = output_dir / "logs"
    logs.mkdir(exist_ok=True)
    t0 = time.time()

    shared_data = ensure_shared_data(gpu=args.gpu) if not args.skip_data_prep else get_shared_data_dict()

    # Find model
    model_dir = args.model_dir
    iteration = args.iteration
    if not model_dir:
        for candidate in [
            RUNS_DIR / "exp4" / "blocks_8" / "finetuned" / "model",
            RUNS_DIR / "exp4" / "blocks_4" / "finetuned" / "model",
            RUNS_DIR / "exp4" / "blocks_2" / "finetuned" / "model",
            RUNS_DIR / "exp1" / "e25" / "02_S3_mix_6k" / "model",
        ]:
            if candidate.exists():
                model_dir = str(candidate)
                break
        if not model_dir:
            print("ERROR: No model found. Run EXP-1 or EXP-4 first, or provide --model_dir")
            return
    if not iteration:
        chk = find_checkpoint(model_dir)
        iteration = int(chk.stem.replace("chkpnt", "")) if chk else None
        if not iteration:
            print("ERROR: No checkpoint found")
            return

    results = {}

    # ── ParaView pure render FPS ──────────────────────────────────────────
    print("\n" + "="*70)
    print("ParaView Pure Render FPS (280M particles, 1080p)")
    print("="*70)
    pv_results = benchmark_paraview_fps(
        shared_data["vtp"], shared_data["normalization"], output_dir, logs)
    results["paraview"] = pv_results

    # ── 3DGS CUDA rasterizer pure render FPS ──────────────────────────────
    print("\n" + "="*70)
    print(f"3DGS CUDA Rasterizer Pure Render FPS (1080p)")
    print("="*70)
    gs_results = benchmark_3dgs_fps(model_dir, iteration, output_dir, logs)
    results["3dgs"] = gs_results

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("EXP-6: Pure Rendering FPS Comparison (1080p)")
    print(f"{'='*70}")

    pv_fps = 0
    pv_ms = 0
    if pv_results:
        if "benchmarks" in pv_results:
            for b in pv_results["benchmarks"]:
                if b.get("width") == 1920 and b.get("mode") == "combined":
                    pv_fps = b.get("avg_fps", 0)
                    pv_ms = b.get("avg_ms_per_frame", 0)
                    break
        else:
            pv_fps = pv_results.get("avg_fps", 0)
            pv_ms = pv_results.get("avg_ms_per_frame", 0)
    gs_fps = 0
    gs_ms = 0
    gs_n = 0
    if gs_results and "benchmarks" in gs_results:
        for b in gs_results["benchmarks"]:
            if b.get("width") == 1920:
                gs_fps = b["fps"]
                gs_ms = b["ms_per_frame"]
        gs_n = gs_results.get("num_gaussians", 0)

    speedup = gs_fps / pv_fps if pv_fps > 0 else 0

    print(f"  ParaView (280M particles):  {pv_fps:.2f} FPS  ({pv_ms:.1f} ms/frame)")
    print(f"  3DGS ({gs_n:,} Gaussians):  {gs_fps:.1f} FPS  ({gs_ms:.2f} ms/frame)")
    print(f"  Speedup: {speedup:.0f}x")

    # Data size comparison
    raw_size_gb = sum(f.stat().st_size for f in RAW_DIR.iterdir() if f.suffix == ".f32") / 1e9
    model_stats = get_model_stats(model_dir, iteration)
    cr = raw_size_gb * 1000 / model_stats["size_mb"]
    print(f"\n  Data: {raw_size_gb:.2f} GB raw vs {model_stats['size_mb']:.0f} MB PLY "
          f"({cr:.0f}x compression)")

    results["summary"] = {
        "paraview_fps": pv_fps,
        "3dgs_fps": gs_fps,
        "speedup": round(speedup, 1),
        "raw_size_gb": round(raw_size_gb, 2),
        "model_size_mb": round(model_stats["size_mb"], 1),
        "compression_ratio": round(cr, 1),
    }

    save_results(results, output_dir / "results.json")
    print(f"\nEXP-6 complete ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
