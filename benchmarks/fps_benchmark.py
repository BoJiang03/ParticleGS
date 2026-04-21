#!/usr/bin/env python3
"""3DGS rendering FPS benchmark with VizMapper.

Full pipeline: VizMapper MLP → CUDA rasterizer, per-frame viz parameter
sampling (matching eval dataset generation). Measures what a deployed
3DGS viewer would actually achieve.

Usage:
    CUDA_VISIBLE_DEVICES=0 python fps_benchmark.py \
        --model_dir runs/exp4/blocks_8/finetuned/model \
        --iteration 60000
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PARTICLEGS_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PARTICLEGS_ROOT))

from particlegs.model.gaussian_model import GaussianModel
from particlegs.model.viz_mapper import VizMapper
from particlegs.renderer import render
from particlegs.scene.cameras import MiniCam
from particlegs.utils.graphics_utils import getWorld2View2, getProjectionMatrix


class _Pipe:
    """Minimal pipeline params for render()."""
    def __init__(self, antialiasing=True):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = antialiasing


def make_orbit_cameras(n_frames, width, height, radius=5.0, elevation=0.3):
    """Generate orbit cameras looking at the origin."""
    fovx = math.radians(60)
    fovy = 2 * math.atan(math.tan(fovx / 2) * height / width)

    proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy
                               ).transpose(0, 1).cuda()
    cameras = []
    for i in range(n_frames):
        theta = 2 * math.pi * i / n_frames
        eye = np.array([radius * math.cos(theta),
                        radius * math.sin(elevation),
                        radius * math.sin(theta)])
        target = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])

        fwd = target - eye
        fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        right = right / np.linalg.norm(right)
        up2 = np.cross(right, fwd)

        R = np.array([right, up2, -fwd])
        T = -R @ eye

        w2v = torch.tensor(getWorld2View2(R, T), dtype=torch.float32).transpose(0, 1).cuda()
        full_proj = w2v.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)

        cam = MiniCam(width, height, fovy, fovx, 0.01, 100.0, w2v, full_proj)
        cameras.append(cam)
    return cameras


def load_viz_mapper(model_dir, iteration):
    """Load VizMapper from checkpoint. Returns (mapper, config) or (None, None)."""
    mapper_path = Path(model_dir) / f"viz_mapper_{iteration}.pth"
    if not mapper_path.exists():
        # Try any mapper
        candidates = sorted(Path(model_dir).glob("viz_mapper_*.pth"),
                            key=lambda p: int(p.stem.split("_")[-1]))
        mapper_path = candidates[-1] if candidates else None

    if mapper_path is None:
        return None, {}

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

    return mapper, {
        "factor_delta_scale": ckpt.get("factor_delta_scale", 0.3),
        "factor_delta_opacity": ckpt.get("factor_delta_opacity", 0.8),
        "min_opacity_clamp": ckpt.get("min_opacity_clamp", 0.1),
    }


class OptimizedVizMapperRunner:
    """Pre-cached, compiled VizMapper runner for maximum throughput.

    Caches per-Gaussian features that don't change across frames.
    Pre-allocates input tensor and only overwrites per-frame columns.
    Uses torch.compile for kernel fusion.
    """

    def __init__(self, mapper, gaussians):
        self.mapper = mapper
        N = gaussians.get_xyz.shape[0]
        self.N = N

        # Cache invariants (same every frame)
        with torch.inference_mode():
            g_log_scale = gaussians._scaling.mean(dim=1).detach()
            g_opacity_logit = gaussians._opacity.squeeze(1).detach()

            # Pre-build input tensor [N, 4] with cached columns 2,3
            self.inp = torch.zeros(N, 4, device="cuda", dtype=torch.float32)
            self.inp[:, 2] = (g_log_scale + 8.0) / 2.0
            self.inp[:, 3] = (g_opacity_logit + 3.0) / 4.0

        # Compile the hot path
        try:
            self._compiled_forward = torch.compile(self._forward_core, mode="reduce-overhead")
        except Exception:
            self._compiled_forward = self._forward_core

    def _forward_core(self, inp):
        """Core MLP forward + post-processing. Compiled target."""
        out = self.mapper.net(inp)
        scale_corr = 1.0 + self.mapper.factor_delta_scale * torch.tanh(out[..., 0])
        opacity_corr = 1.0 + self.mapper.factor_delta_opacity * torch.tanh(out[..., 1])
        return scale_corr, opacity_corr

    @torch.inference_mode()
    def __call__(self, viz_scale_factor, viz_opacity_factor):
        """Run VizMapper for one frame. Returns (scale_factor[N], opacity_factor[N])."""
        # Only update the 2 per-frame columns (columns 0,1)
        self.inp[:, 0] = viz_scale_factor - 1.0
        self.inp[:, 1] = viz_opacity_factor - 1.0

        scale_corr, opacity_corr = self._compiled_forward(self.inp)

        final_scale = viz_scale_factor * scale_corr
        final_opacity = torch.clamp(viz_opacity_factor * opacity_corr,
                                    min=self.mapper.min_opacity_clamp)
        return final_scale, final_opacity


def benchmark_fps(gaussians, cameras, pipe, bg, mapper_runner=None,
                  viz_params=None, warmup=10, n_iters=200):
    """Time full pipeline with per-component breakdown.

    Returns dict with:
      combined_fps, combined_ms,
      viz_mapper_ms, rasterizer_ms  (per-component averages)
    """
    beta_a = 3.0
    r_min, r_max = 0.0025, 0.0175
    o_min, o_max = 0.0125, 0.0875
    base_radius, base_opacity = 0.01, 0.05

    def sample_viz_with(rng):
        r = r_min + (r_max - r_min) * rng.beta(beta_a, beta_a)
        o = o_min + (o_max - o_min) * rng.beta(beta_a, beta_a)
        return r / base_radius, min(o, 1.0) / base_opacity

    # Warmup (also warms up torch.compile)
    rng_warmup = np.random.RandomState(142)
    for i in range(warmup):
        cam = cameras[i % len(cameras)]
        sf, of = sample_viz_with(rng_warmup)
        if mapper_runner:
            viz_s, viz_o = mapper_runner(sf, of)
        else:
            viz_s, viz_o = sf, of
        render(cam, gaussians, pipe, bg,
               viz_scale_factor=viz_s, viz_opacity_factor=viz_o)
    torch.cuda.synchronize()

    # Pass 1: combined throughput (no inter-component sync — real throughput)
    rng2 = np.random.RandomState(142)
    t0 = time.perf_counter()
    for i in range(n_iters):
        cam = cameras[i % len(cameras)]
        sf, of = sample_viz_with(rng2)
        if mapper_runner:
            viz_s, viz_o = mapper_runner(sf, of)
        else:
            viz_s, viz_o = sf, of
        render(cam, gaussians, pipe, bg,
               viz_scale_factor=viz_s, viz_opacity_factor=viz_o)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    combined_fps = n_iters / elapsed
    combined_ms = elapsed * 1000 / n_iters

    # Pass 2: per-component timing (with sync between components)
    rng3 = np.random.RandomState(142)
    vm_times = []
    rast_times = []
    for i in range(n_iters):
        cam = cameras[i % len(cameras)]
        sf, of = sample_viz_with(rng3)

        torch.cuda.synchronize()
        t_vm = time.perf_counter()
        if mapper_runner:
            viz_s, viz_o = mapper_runner(sf, of)
        else:
            viz_s, viz_o = sf, of
        torch.cuda.synchronize()
        t_rast = time.perf_counter()

        render(cam, gaussians, pipe, bg,
               viz_scale_factor=viz_s, viz_opacity_factor=viz_o)
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        vm_times.append(t_rast - t_vm)
        rast_times.append(t_end - t_rast)

    vm_ms = sum(vm_times) / len(vm_times) * 1000
    rast_ms = sum(rast_times) / len(rast_times) * 1000

    return {
        "combined_fps": combined_fps,
        "combined_ms": combined_ms,
        "viz_mapper_ms": vm_ms,
        "rasterizer_ms": rast_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="3DGS + VizMapper FPS benchmark")
    parser.add_argument("--model_dir", required=True, help="Path to model directory")
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--resolutions", type=str, default="1920x1080,3840x2160")
    parser.add_argument("--n_frames", type=int, default=60)
    parser.add_argument("--n_iters", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # Find iteration
    iteration = args.iteration
    if iteration is None:
        pc_dir = model_dir / "point_cloud"
        iter_dirs = sorted(pc_dir.glob("iteration_*"),
                           key=lambda p: int(p.name.split("_")[1]))
        iteration = int(iter_dirs[-1].name.split("_")[1]) if iter_dirs else 0

    ply_path = model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply_path.exists():
        print(f"ERROR: PLY not found: {ply_path}")
        sys.exit(1)

    # Load model
    print(f"Loading model: {ply_path}")
    gaussians = GaussianModel(sh_degree=0)
    gaussians.load_ply(str(ply_path))
    n_gaussians = gaussians.get_xyz.shape[0]
    print(f"  {n_gaussians:,} Gaussians loaded")

    # Load VizMapper
    mapper, mapper_cfg = load_viz_mapper(model_dir, iteration)
    mapper_runner = None
    if mapper:
        print(f"  VizMapper loaded (delta_s={mapper_cfg['factor_delta_scale']}, "
              f"delta_o={mapper_cfg['factor_delta_opacity']})")
        mapper_runner = OptimizedVizMapperRunner(mapper, gaussians)
        print(f"  VizMapper compiled & cached ({n_gaussians} Gaussians)")
    else:
        print(f"  No VizMapper found — using raw viz factors")

    pipe = _Pipe(antialiasing=True)
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")

    # Parse resolutions
    resolutions = []
    for res_str in args.resolutions.split(","):
        w, h = res_str.strip().split("x")
        resolutions.append((int(w), int(h)))

    results = {
        "model": str(model_dir),
        "iteration": iteration,
        "num_gaussians": n_gaussians,
        "has_viz_mapper": mapper is not None,
        "n_iters": args.n_iters,
        "warmup": args.warmup,
        "gpu": torch.cuda.get_device_name(0),
        "benchmarks": [],
    }

    print(f"\nGPU: {results['gpu']}")
    mode = "VizMapper + Rasterizer" if mapper else "Rasterizer only"
    print(f"Mode: {mode}")
    print(f"{'='*60}")
    print(f"{'Resolution':<16} {'FPS':>8} {'ms/frame':>10} {'VizMapper':>10} {'Rasterizer':>11}")
    print(f"{'-'*68}")

    for w, h in resolutions:
        cameras = make_orbit_cameras(args.n_frames, w, h)
        bm = benchmark_fps(gaussians, cameras, pipe, bg,
                           mapper_runner=mapper_runner,
                           warmup=args.warmup, n_iters=args.n_iters)

        entry = {
            "width": w, "height": h,
            "fps": round(bm["combined_fps"], 2),
            "ms_per_frame": round(bm["combined_ms"], 2),
            "viz_mapper_ms": round(bm["viz_mapper_ms"], 2),
            "rasterizer_ms": round(bm["rasterizer_ms"], 2),
        }
        results["benchmarks"].append(entry)
        print(f"{w}x{h:<11} {bm['combined_fps']:>8.1f} {bm['combined_ms']:>9.2f}ms "
              f"{bm['viz_mapper_ms']:>9.2f}ms {bm['rasterizer_ms']:>10.2f}ms")

    print(f"{'='*60}")

    out_path = Path(args.output) if args.output else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
