#!/usr/bin/env python3
"""ParaView rendering FPS benchmark — realistic eval dataset workload.

Each frame: change camera + change GaussianRadius/Opacity via standard API + Render().
No ctypes bypass — measures what a normal ParaView user would see.

Usage:
    pvbatch --force-offscreen-rendering -- pv_fps_benchmark.py \
        --vtp_path /path/to/particles.vtp \
        --normalization /path/to/normalization.json
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

from paraview.simple import *


# ── Vector math ────────────────────────────────────────────────────────

def vec_sub(a, b): return [x - y for x, y in zip(a, b)]
def vec_norm(a): return math.sqrt(sum(x * x for x in a))
def vec_normalize(a):
    m = vec_norm(a)
    return [x / m for x in a] if m > 0 else [0.0, 0.0, 1.0]
def vec_cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
def vec_dot(a, b): return sum(x*y for x, y in zip(a, b))

def choose_view_up(dir_vec):
    up = [0.0, 1.0, 0.0]
    if abs(vec_dot(vec_normalize(dir_vec), up)) > 0.95:
        return [0.0, 0.0, 1.0]
    return up


def generate_orbit_positions(center, base_radius, orbit_radius_frac,
                             n_frames, elev_range=(-0.3, 0.3)):
    """Generate camera positions on an orbit around center."""
    radius = base_radius * orbit_radius_frac
    positions = []
    for i in range(n_frames):
        t = i / n_frames
        theta = 2 * math.pi * t
        elev = elev_range[0] + (elev_range[1] - elev_range[0]) * (
            0.5 + 0.5 * math.sin(2 * math.pi * t))
        x = center[0] + radius * math.cos(theta) * math.cos(elev)
        y = center[1] + radius * math.sin(elev)
        z = center[2] + radius * math.sin(theta) * math.cos(elev)
        positions.append([x, y, z])
    return positions


def _benchmark_orbit(render_view, display, center, base_radius, orbit_r,
                     n_frames, warmup, change_viz, viz_rng_seed, a,
                     r_min, r_max, o_min, o_max):
    """Benchmark a single orbit. Returns (fps, ms_per_frame, total_s)."""
    positions = generate_orbit_positions(
        center, base_radius, orbit_r, n_frames + warmup)
    viz_rng = np.random.RandomState(viz_rng_seed)

    for i in range(warmup):
        pos = positions[i]
        render_view.CameraPosition = pos
        render_view.CameraFocalPoint = center
        render_view.CameraViewUp = choose_view_up(vec_sub(center, pos))
        try:
            render_view.ResetCameraClippingRange()
        except AttributeError:
            pass
        if change_viz:
            r = r_min + (r_max - r_min) * viz_rng.beta(a, a)
            o = o_min + (o_max - o_min) * viz_rng.beta(a, a)
            display.GaussianRadius = r
            display.Opacity = min(o, 1.0)
        Render()

    t0 = time.perf_counter()
    for i in range(warmup, warmup + n_frames):
        pos = positions[i]
        render_view.CameraPosition = pos
        render_view.CameraFocalPoint = center
        render_view.CameraViewUp = choose_view_up(vec_sub(center, pos))
        try:
            render_view.ResetCameraClippingRange()
        except AttributeError:
            pass
        if change_viz:
            r = r_min + (r_max - r_min) * viz_rng.beta(a, a)
            o = o_min + (o_max - o_min) * viz_rng.beta(a, a)
            display.GaussianRadius = r
            display.Opacity = min(o, 1.0)
        Render()
    elapsed = time.perf_counter() - t0

    fps = n_frames / elapsed
    ms = elapsed * 1000 / n_frames
    return round(fps, 3), round(ms, 1), round(elapsed, 2)


def main():
    parser = argparse.ArgumentParser(description="ParaView render FPS benchmark")
    parser.add_argument("--vtp_path", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--resolutions", type=str, default="1920x1080,3840x2160",
                        help="Comma-separated WxH resolutions")
    parser.add_argument("--n_frames", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--orbit_radii", type=str, default="1.0,0.7,0.5")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--radius_min", type=float, default=0.0025)
    parser.add_argument("--radius_max", type=float, default=0.0175)
    parser.add_argument("--opacity_min", type=float, default=0.0125)
    parser.add_argument("--opacity_max", type=float, default=0.0875)
    parser.add_argument("--viz_seed", type=int, default=142)
    parser.add_argument("--viz_beta_concentration", type=float, default=3.0)
    # Legacy flag (ignored — both modes always run)
    parser.add_argument("--camera_only", action="store_true", help=argparse.SUPPRESS)
    args, _ = parser.parse_known_args()

    orbit_radii = [float(x) for x in args.orbit_radii.split(",")]
    resolutions = []
    for s in args.resolutions.split(","):
        w, h = s.strip().split("x")
        resolutions.append((int(w), int(h)))

    with open(args.normalization) as f:
        norm = json.load(f)
    center = norm["center"]

    print(f"Loading VTP: {args.vtp_path}")
    data_source = OpenDataFile(args.vtp_path)
    if not data_source:
        print("ERROR: Could not open VTP")
        sys.exit(1)

    render_view = CreateRenderView()
    render_view.OrientationAxesVisibility = 0
    render_view.UseColorPaletteForBackground = 0
    render_view.Background = [0.0, 0.0, 0.0]
    render_view.Background2 = [0.0, 0.0, 0.0]
    try:
        render_view.BackgroundColorMode = 0
    except AttributeError:
        render_view.UseGradientBackground = 0

    display = Show(data_source, render_view)
    display.Representation = "Point Gaussian"
    display.ShaderPreset = "Gaussian Blur"
    display.GaussianRadius = 0.01
    display.Opacity = 0.05
    display.ColorArrayName = [None, ""]
    display.DiffuseColor = [1.0, 1.0, 1.0]

    render_view.ViewSize = [resolutions[0][0], resolutions[0][1]]
    layout = GetLayout(render_view)
    if layout:
        layout.SetSize(resolutions[0][0], resolutions[0][1])
    render_view.ResetCamera()
    Render()

    original_pos = list(render_view.CameraPosition)
    base_radius = norm.get("base_radius", vec_norm(vec_sub(original_pos, center)))

    try:
        render_view.UseLODForStillRender = 0
    except AttributeError:
        pass
    try:
        render_view.UseLODForInteractiveRender = 0
    except AttributeError:
        pass

    a = args.viz_beta_concentration
    results = {"benchmarks": []}

    for w, h in resolutions:
        # Resize
        render_view.ViewSize = [w, h]
        if layout:
            layout.SetSize(w, h)
        Render()

        for mode_name, change_viz in [("combined", True), ("camera_only", False)]:
            orbit_results = {}
            for orbit_r in orbit_radii:
                seed = args.viz_seed + int(orbit_r * 1000)
                fps, ms, total = _benchmark_orbit(
                    render_view, display, center, base_radius, orbit_r,
                    args.n_frames, args.warmup, change_viz, seed, a,
                    args.radius_min, args.radius_max,
                    args.opacity_min, args.opacity_max)
                orbit_results[f"orbit_{orbit_r}"] = {
                    "fps": fps, "ms_per_frame": ms, "total_s": total}
                print(f"  {w}x{h} {mode_name} orbit_{orbit_r}: "
                      f"{fps:.3f} FPS ({ms:.1f} ms/frame)")

            all_fps = [v["fps"] for v in orbit_results.values()]
            all_ms = [v["ms_per_frame"] for v in orbit_results.values()]
            avg_fps = round(sum(all_fps) / len(all_fps), 3)
            avg_ms = round(sum(all_ms) / len(all_ms), 1)

            results["benchmarks"].append({
                "width": w, "height": h, "mode": mode_name,
                "n_frames": args.n_frames, "warmup": args.warmup,
                "orbits": orbit_results,
                "avg_fps": avg_fps, "avg_ms_per_frame": avg_ms,
            })
            print(f"  => {w}x{h} {mode_name} avg: {avg_fps:.3f} FPS "
                  f"({avg_ms:.1f} ms/frame)")
        print()

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

    return results


if __name__ == "__main__":
    main()
