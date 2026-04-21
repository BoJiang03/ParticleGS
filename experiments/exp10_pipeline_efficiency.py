#!/usr/bin/env python3
"""EXP-10: Pipeline Efficiency.

Measures training time, rendering FPS (pure CUDA rasterizer), and resource usage.

Usage:
    python -m experiments.exp10_pipeline_efficiency [--gpu 0]
"""

import json
import sys
import time
from pathlib import Path

from experiments.common import *
from experiments.exp1_rate_distortion import E25_STAGES, run_e25_training

sys.path.insert(0, str(PARTICLEGS_ROOT))


def benchmark_render_fps(model_dir, iteration, resolutions="1920x1080,3840x2160",
                         n_iters=300, warmup=20):
    """Run pure CUDA rasterizer FPS benchmark. Returns dict of results."""
    from benchmarks.fps_benchmark import benchmark_fps, make_orbit_cameras, _Pipe
    from particlegs.model.gaussian_model import GaussianModel
    import torch

    ply_path = Path(model_dir) / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply_path.exists():
        print(f"  WARNING: PLY not found: {ply_path}")
        return None

    gaussians = GaussianModel(sh_degree=0)
    gaussians.load_ply(str(ply_path))
    n_gaussians = gaussians.get_xyz.shape[0]

    pipe = _Pipe(antialiasing=True)
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")

    results = {
        "num_gaussians": n_gaussians,
        "gpu": torch.cuda.get_device_name(0),
    }

    for res_str in resolutions.split(","):
        w, h = [int(x) for x in res_str.strip().split("x")]
        cameras = make_orbit_cameras(60, w, h)
        fps, ms = benchmark_fps(gaussians, cameras, pipe, bg,
                                warmup=warmup, n_iters=n_iters)
        results[f"{w}x{h}"] = {
            "fps": round(fps, 2),
            "ms_per_frame": round(ms, 2),
        }
        print(f"    {w}x{h}: {fps:.1f} FPS ({ms:.2f} ms/frame)")

    return results


def main():
    parser = base_parser("EXP-10: Pipeline Efficiency")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RUNS_DIR / "exp10"
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_data = ensure_shared_data(gpu=args.gpu) if not args.skip_data_prep else get_shared_data_dict()

    print("\n" + "="*70)
    print("EXP-10: Pipeline Efficiency")
    print("="*70)

    t0 = time.time()

    # ── Part 1: Training time ─────────────────────────────────────────────
    print("\n[1] Single-block E25 training time...")
    t_train_start = time.time()
    model_dir, iteration = run_e25_training(output_dir, shared_data, gpu=args.gpu)
    t_train_end = time.time()
    total_train = t_train_end - t_train_start

    # ── Part 2: Eval time ─────────────────────────────────────────────────
    print("\n[2] Evaluation time...")
    t_eval_start = time.time()
    last_train = E25_STAGES[-1]["train"]
    eval_results = evaluate_model(model_dir, iteration, shared_data,
                                  output_dir / "e25" / "logs", gpu=args.gpu,
                                  factor_delta_opacity=last_train["factor_delta_opacity"],
                                  factor_delta_scale=last_train["factor_delta_scale"],
                                  min_opacity_clamp=last_train["min_opacity_clamp"])
    t_eval_end = time.time()
    total_eval = t_eval_end - t_eval_start

    # ── Part 3: Render FPS benchmark ──────────────────────────────────────
    print("\n[3] CUDA rasterizer FPS benchmark...")

    # Benchmark current model
    fps_single = benchmark_render_fps(model_dir, iteration)

    # Also benchmark 8-block finetuned model if available
    fps_8block = None
    exp4_ft = RUNS_DIR / "exp4" / "blocks_8" / "finetuned" / "model"
    if exp4_ft.exists():
        chk = find_checkpoint(exp4_ft)
        if chk:
            ft_iter = int(chk.stem.replace("chkpnt", ""))
            print(f"\n  Also benchmarking 8-block finetuned model ({exp4_ft.name})...")
            fps_8block = benchmark_render_fps(exp4_ft, ft_iter)

    total = time.time() - t0
    stats = get_model_stats(model_dir, iteration)

    # ── Results ───────────────────────────────────────────────────────────
    result = {
        "timing": {
            "total_time_s": round(total, 1),
            "train_time_s": round(total_train, 1),
            "eval_time_s": round(total_eval, 1),
        },
        "quality": {
            "masked_psnr": eval_results["avg"]["masked_psnr"],
            "size_mb": round(stats["size_mb"], 1),
            "num_gaussians": stats["num_gaussians"],
        },
        "render_fps": {
            "single_block": fps_single,
        },
    }
    if fps_8block:
        result["render_fps"]["8_block_finetuned"] = fps_8block

    print(f"\n{'='*70}")
    print("EXP-10: Results Summary")
    print(f"{'='*70}")
    print(f"  Training (3 stages):  {total_train/60:.1f} min")
    print(f"  Evaluation:           {total_eval/60:.1f} min")
    print(f"  Total:                {total/60:.1f} min")
    print(f"  Quality:              {result['quality']['masked_psnr']:.2f} dB masked PSNR")
    print(f"  Model:                {result['quality']['size_mb']} MB, "
          f"{result['quality']['num_gaussians']} Gaussians")
    if fps_single:
        for res, data in fps_single.items():
            if isinstance(data, dict) and "fps" in data:
                print(f"  Render FPS ({res}):  {data['fps']:.1f} FPS ({data['ms_per_frame']:.2f} ms)")

    save_results(result, output_dir / "results.json")
    print(f"\nEXP-10 complete ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
