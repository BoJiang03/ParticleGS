#!/usr/bin/env python3
"""SZ3 CR vs PSNR Benchmark for Particle Visualization.

Sweeps SZ3 ABS error bounds, compresses/decompresses particle data,
renders with ParaView, and computes visualization PSNR vs GT images.
Produces a CR-PSNR curve.

Prerequisites:
    - SZ3 binary built and accessible
    - pvbatch for ParaView rendering

Usage:
    python sz3_benchmark.py \
        --sz3_bin /path/to/sz3 \
        --gt_raw_dir data/hacc_raw \
        --eval_base runs/shared/evaluation \
        --normalization runs/shared/normalization.json \
        --output_dir benchmarks/sz3_results
"""

import argparse
import ctypes
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PARTICLEGS_ROOT = SCRIPT_DIR.parent

PYTHON_BIN = sys.executable
PVBATCH_BIN = shutil.which("pvbatch") or "pvbatch"

PREPARE_SCRIPT = PARTICLEGS_ROOT / "data" / "prepare_data.py"
GENERATE_SCRIPT = PARTICLEGS_ROOT / "data" / "generate_images.py"

NUM_POINTS = 280_953_867

EVAL_DATASETS = {
    "eval_far":  {"orbit_radii": "1.0"},
    "eval_mid":  {"orbit_radii": "0.7"},
    "eval_near": {"orbit_radii": "0.5"},
}

RENDER_PARAMS = {
    "gaussian_radius": "0.01",
    "opacity": "0.05",
    "viz_mode": "sampled",
    "viz_distribution": "beta",
    "viz_beta_concentration": "3.0",
    "radius_min": "0.0025",
    "radius_max": "0.0175",
    "opacity_min": "0.0125",
    "opacity_max": "0.0875",
    "viz_seed": "142",
    "num_frames": "80",
    "width": "1920",
    "height": "1080",
    "train_ratio": "1.0",
    "split_seed": "42",
}

# Default EB sweep (log-spaced)
DEFAULT_EB_SWEEP = [0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0, 1.3]

_BG_THRESH = 2.5


# ── EGL GPU pinning ──────────────────────────────────────────────────────

def _probe_egl_cuda_mapping():
    try:
        libEGL = ctypes.CDLL("libEGL.so.1")
    except OSError:
        return None
    eglGetProcAddress = libEGL.eglGetProcAddress
    eglGetProcAddress.restype = ctypes.c_void_p
    eglGetProcAddress.argtypes = [ctypes.c_char_p]
    addr_query = eglGetProcAddress(b"eglQueryDevicesEXT")
    addr_attrib = eglGetProcAddress(b"eglQueryDeviceAttribEXT")
    if not addr_query or not addr_attrib:
        return None
    FUNCTYPE_Q = ctypes.CFUNCTYPE(
        ctypes.c_uint, ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int))
    FUNCTYPE_A = ctypes.CFUNCTYPE(
        ctypes.c_uint, ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_longlong))
    eglQueryDevicesEXT = FUNCTYPE_Q(addr_query)
    eglQueryDeviceAttribEXT = FUNCTYPE_A(addr_attrib)
    max_devices = 16
    devices = (ctypes.c_void_p * max_devices)()
    num_devices = ctypes.c_int(0)
    if eglQueryDevicesEXT(max_devices, devices, ctypes.byref(num_devices)) != 1:
        return None
    EGL_DEVICE_CUDA_NV = 0x323A
    mapping = {}
    for i in range(num_devices.value):
        cuda_idx = ctypes.c_longlong(-1)
        if eglQueryDeviceAttribEXT(devices[i], EGL_DEVICE_CUDA_NV, ctypes.byref(cuda_idx)) == 1:
            if cuda_idx.value >= 0:
                mapping[i] = int(cuda_idx.value)
    return mapping if mapping else None


def _get_pvbatch_egl_args():
    egl_to_cuda = _probe_egl_cuda_mapping()
    if egl_to_cuda is None:
        return ["--force-offscreen-rendering"]
    for egl_idx, cuda_idx in egl_to_cuda.items():
        if cuda_idx == 0:
            return ["--force-offscreen-rendering",
                    "--opengl-window-backend", "EGL",
                    "--displays", str(egl_idx), "--"]
    return ["--force-offscreen-rendering"]


# ── PSNR ──────────────────────────────────────────────────────────────────

def compute_psnr_from_dirs(gt_dir, render_dir, max_workers=8):
    gt_dir, render_dir = Path(gt_dir), Path(render_dir)
    if not gt_dir.exists() or not render_dir.exists():
        return None, None, 0
    gt_files = sorted(p for p in gt_dir.iterdir() if p.suffix.lower() == ".png")

    def _one(gp):
        rp = render_dir / gp.name
        if not rp.exists():
            return None, None
        try:
            g = np.array(Image.open(gp)).astype(np.float64)
            r = np.array(Image.open(rp)).astype(np.float64)
        except OSError:
            return None, None
        if g.shape != r.shape:
            return None, None
        mse = np.mean((g - r) ** 2)
        psnr = 100.0 if mse == 0 else 20 * math.log10(255.0 / math.sqrt(mse))
        mask = (g.mean(2) > _BG_THRESH) | (r.mean(2) > _BG_THRESH)
        if mask.sum() == 0:
            return psnr, None
        m3 = np.stack([mask] * 3, axis=2)
        mse_m = np.mean((g[m3] - r[m3]) ** 2)
        mpsnr = 100.0 if mse_m == 0 else 20 * math.log10(255.0 / math.sqrt(mse_m))
        return psnr, mpsnr

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_one, gt_files))
    ps = [r[0] for r in results if r[0] is not None]
    ms = [r[1] for r in results if r[1] is not None]
    return (sum(ps)/len(ps) if ps else None,
            sum(ms)/len(ms) if ms else None,
            len(ps))


# ── Helpers ───────────────────────────────────────────────────────────────

def run_cmd(cmd, log_path, cwd=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        subprocess.check_call([str(c) for c in cmd],
                              cwd=str(cwd or PARTICLEGS_ROOT),
                              stdout=f, stderr=subprocess.STDOUT)


def compress_decompress_axis(sz3_bin, raw_path, out_dir, eb, num_points, axis_name):
    """Compress and decompress one axis with SZ3."""
    compressed = out_dir / f"{axis_name}.sz3"
    decompressed = out_dir / f"{axis_name}.f32"

    # Compress
    run_cmd([sz3_bin, "-f", "-i", str(raw_path), "-z", str(compressed),
             "-1", str(num_points), "-M", "ABS", str(eb)],
            out_dir / f"compress_{axis_name}.log")
    compressed_size = compressed.stat().st_size

    # Decompress
    run_cmd([sz3_bin, "-f", "-z", str(compressed), "-o", str(decompressed),
             "-1", str(num_points), "-M", "ABS", str(eb)],
            out_dir / f"decompress_{axis_name}.log")

    return compressed_size


def run_eb_point(sz3_bin, gt_raw_dir, norm_json, eb, output_dir, pvbatch_egl, gt_eval_images):
    """Run one error bound point: compress, decompress, render, compute PSNR."""
    eb_dir = output_dir / f"eb_{eb:.4e}"
    if (eb_dir / "results.json").exists():
        with open(eb_dir / "results.json") as f:
            return json.load(f)

    eb_dir.mkdir(parents=True, exist_ok=True)
    logs = eb_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    raw_size = 0
    compressed_size = 0
    for axis, fname in [("xx", "xx.f32"), ("yy", "yy.f32"), ("zz", "zz.f32")]:
        raw_path = gt_raw_dir / fname
        raw_size += raw_path.stat().st_size
        cs = compress_decompress_axis(sz3_bin, raw_path, eb_dir, eb, NUM_POINTS, axis)
        compressed_size += cs

    cr = raw_size / compressed_size if compressed_size > 0 else 0
    print(f"\n  EB={eb:.4e}: CR={cr:.2f}x ({compressed_size/1e6:.1f} MB)")

    # Create VTP from decompressed
    vtp_dir = eb_dir / "vtp"
    run_cmd([PVBATCH_BIN] + pvbatch_egl + [
        str(PREPARE_SCRIPT),
        "--raw_x", str(eb_dir / "xx.f32"),
        "--raw_y", str(eb_dir / "yy.f32"),
        "--raw_z", str(eb_dir / "zz.f32"),
        "--output_dir", str(vtp_dir),
        "--num_points_raw", "0",
        "--skip_images",
    ], logs / "create_vtp.log")
    vtp_path = vtp_dir / "particles.vtp"

    # Render and compute PSNR for each eval dataset
    eval_results = {}
    for eval_name, cfg in EVAL_DATASETS.items():
        render_out = eb_dir / f"render_{eval_name}"
        pv_args = [
            "--vtp_path", str(vtp_path),
            "--output_dir", str(render_out),
            "--camera_strategy", "multi_orbit",
            "--orbit_radii", cfg["orbit_radii"],
            "--normalization_path", str(norm_json),
        ]
        for k, v in RENDER_PARAMS.items():
            pv_args += [f"--{k}", v]

        run_cmd([PVBATCH_BIN] + pvbatch_egl + [str(GENERATE_SCRIPT)] + pv_args,
                logs / f"render_{eval_name}.log")

        gt_images = gt_eval_images.get(eval_name)
        if gt_images and gt_images.exists():
            psnr, mpsnr, n = compute_psnr_from_dirs(
                gt_images, render_out / "images")
            eval_results[eval_name] = {"psnr": psnr, "masked_psnr": mpsnr, "n": n}
            if psnr:
                print(f"    {eval_name}: PSNR={psnr:.2f}  masked={mpsnr:.2f}")

        # Cleanup rendered images
        if render_out.exists():
            shutil.rmtree(render_out)

    # Cleanup VTP
    if vtp_path.exists():
        vtp_path.unlink()

    result = {
        "eb": eb,
        "cr": cr,
        "raw_bytes": raw_size,
        "compressed_bytes": compressed_size,
        "eval": eval_results,
    }
    with open(eb_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description="SZ3 CR vs PSNR benchmark")
    parser.add_argument("--sz3_bin", required=True, help="Path to sz3 binary")
    parser.add_argument("--gt_raw_dir", required=True)
    parser.add_argument("--eval_base", required=True,
                        help="Path to eval datasets (with 01_eval_far/data/images, etc.)")
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eb_values", type=str, default=None,
                        help="Comma-separated error bound values")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_raw_dir = Path(args.gt_raw_dir)
    eval_base = Path(args.eval_base)
    norm_json = Path(args.normalization)

    pvbatch_egl = _get_pvbatch_egl_args()

    if args.eb_values:
        eb_values = [float(x.strip()) for x in args.eb_values.split(",")]
    else:
        eb_values = DEFAULT_EB_SWEEP

    gt_eval_images = {}
    for name, cfg in EVAL_DATASETS.items():
        sub = {"eval_far": "01_eval_far", "eval_mid": "02_eval_mid", "eval_near": "03_eval_near"}[name]
        gt_eval_images[name] = eval_base / sub / "data" / "images"

    print(f"SZ3 Benchmark: {len(eb_values)} error bounds")
    print(f"  EB values: {eb_values}")

    all_results = []
    for eb in sorted(eb_values):
        result = run_eb_point(
            args.sz3_bin, gt_raw_dir, norm_json, eb, output_dir,
            pvbatch_egl, gt_eval_images)
        all_results.append(result)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"SZ3 CR vs PSNR Summary")
    print(f"{'='*80}")
    print(f"{'EB':>12} {'CR':>8} {'Far PSNR':>10} {'Mid PSNR':>10} {'Near PSNR':>10} {'Avg mPSNR':>10}")
    print("-" * 64)
    for r in all_results:
        far = r["eval"].get("eval_far", {}).get("masked_psnr", 0) or 0
        mid = r["eval"].get("eval_mid", {}).get("masked_psnr", 0) or 0
        near = r["eval"].get("eval_near", {}).get("masked_psnr", 0) or 0
        vals = [v for v in [far, mid, near] if v > 0]
        avg = sum(vals) / len(vals) if vals else 0
        print(f"{r['eb']:>12.4e} {r['cr']:>8.2f} {far:>10.2f} {mid:>10.2f} {near:>10.2f} {avg:>10.2f}")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
