#!/usr/bin/env python3
"""Benchmark: ParaView (280M particles) vs 3DGS rendering speed.

Renders the standard eval datasets (eval_far/mid/near, 80 frames x 1080p)
with both pipelines and compares wall-clock time.

Usage:
    CUDA_VISIBLE_DEVICES=0 python render_benchmark.py \
        --model_dir runs/my_exp/model \
        --gt_raw_dir data/hacc_raw \
        --eval_base runs/my_exp/shared/evaluation \
        --normalization runs/my_exp/model/normalization.json
"""

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PARTICLEGS_ROOT = SCRIPT_DIR.parent

PYTHON_BIN = sys.executable
PVBATCH_BIN = shutil.which("pvbatch") or "pvbatch"

PREPARE_SCRIPT = PARTICLEGS_ROOT / "data" / "prepare_data.py"
GENERATE_SCRIPT = PARTICLEGS_ROOT / "data" / "generate_images.py"
RENDER_CMD = [PYTHON_BIN, "-m", "particlegs.evaluation.render"]

EVAL_DATASETS = {
    "eval_far":  {"subdir": "01_eval_far",  "orbit_radii": "1.0"},
    "eval_mid":  {"subdir": "02_eval_mid",  "orbit_radii": "0.7"},
    "eval_near": {"subdir": "03_eval_near", "orbit_radii": "0.5"},
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
            print(f"  pvbatch GPU pin: EGL device {egl_idx} -> CUDA 0")
            return ["--force-offscreen-rendering",
                    "--opengl-window-backend", "EGL",
                    "--displays", str(egl_idx), "--"]
    return ["--force-offscreen-rendering"]


def run_timed(cmd, log_path, desc=""):
    print(f"  [{desc}] Running...")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "w") as f:
        subprocess.check_call([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    print(f"  [{desc}] {elapsed:.2f}s")
    return elapsed


def count_pngs(d):
    return len(list(d.glob("*.png"))) if d.exists() else 0


def main():
    parser = argparse.ArgumentParser(description="ParaView vs 3DGS render benchmark")
    parser.add_argument("--model_dir", required=True, help="Path to trained 3DGS model")
    parser.add_argument("--gt_raw_dir", required=True, help="Path to GT raw f32 files")
    parser.add_argument("--eval_base", required=True, help="Path to evaluation datasets")
    parser.add_argument("--normalization", required=True, help="Path to normalization.json")
    parser.add_argument("--iteration", type=int, default=60000)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    gt_raw_dir = Path(args.gt_raw_dir)
    eval_base = Path(args.eval_base)
    norm_json = Path(args.normalization)

    run_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "runs" / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    results = {"paraview": {}, "3dgs": {}, "meta": {}}

    pvbatch_egl = _get_pvbatch_egl_args()

    # Meta
    gt_bytes = sum((gt_raw_dir / f"{a}.f32").stat().st_size for a in ["xx", "yy", "zz"])
    model_ply = list((model_dir / "point_cloud").glob("iteration_*/point_cloud.ply"))
    model_bytes = model_ply[0].stat().st_size if model_ply else 0
    results["meta"]["gt_raw_mb"] = gt_bytes / 1e6
    results["meta"]["model_ply_mb"] = model_bytes / 1e6
    print(f"GT: {gt_bytes/1e9:.2f} GB raw, Model: {model_bytes/1e6:.1f} MB PLY")

    # Step 1: Create VTP
    print("\n=== Step 1: Create GT VTP ===")
    gt_vtp_dir = run_dir / "gt_vtp"
    t_vtp = run_timed([PVBATCH_BIN] + pvbatch_egl + [
        str(PREPARE_SCRIPT),
        "--raw_x", str(gt_raw_dir / "xx.f32"),
        "--raw_y", str(gt_raw_dir / "yy.f32"),
        "--raw_z", str(gt_raw_dir / "zz.f32"),
        "--output_dir", str(gt_vtp_dir),
        "--num_points_raw", "0",
        "--skip_images",
    ], logs / "create_vtp.log", desc="create_vtp")
    gt_vtp = gt_vtp_dir / "particles.vtp"
    results["paraview"]["vtp_creation_s"] = t_vtp

    # Step 2: ParaView rendering
    print("\n=== Step 2: ParaView rendering ===")
    for name, cfg in EVAL_DATASETS.items():
        out = run_dir / f"pv_{name}"
        pv_args = [
            "--vtp_path", str(gt_vtp),
            "--output_dir", str(out),
            "--camera_strategy", "multi_orbit",
            "--orbit_radii", cfg["orbit_radii"],
            "--normalization_path", str(norm_json),
        ]
        for k, v in RENDER_PARAMS.items():
            pv_args += [f"--{k}", v]
        elapsed = run_timed(
            [PVBATCH_BIN] + pvbatch_egl + [str(GENERATE_SCRIPT)] + pv_args,
            logs / f"pv_{name}.log", desc=f"ParaView {name}")
        n = count_pngs(out / "images")
        results["paraview"][name] = {
            "total_s": elapsed, "n_frames": n,
            "fps": n / elapsed if elapsed > 0 else 0,
            "ms_per_frame": elapsed * 1000 / n if n > 0 else 0,
        }

    # Cleanup VTP
    if gt_vtp.exists():
        gt_vtp.unlink()

    # Step 3: 3DGS rendering
    print("\n=== Step 3: 3DGS rendering ===")
    for name, cfg in EVAL_DATASETS.items():
        eval_data = eval_base / cfg["subdir"] / "data"
        out = run_dir / f"dgs_{name}"
        elapsed = run_timed(RENDER_CMD + [
            "-s", str(eval_data),
            "-m", str(model_dir),
            "--iteration", str(args.iteration),
            "--skip_test",
            "--output_dir", str(out),
            "--factor_delta_opacity", "0.8",
            "--factor_delta_scale", "0.3",
            "--min_opacity_clamp", "0.1",
            "--antialiasing",
        ], logs / f"dgs_{name}.log", desc=f"3DGS {name}")
        renders = out / "train" / f"ours_{args.iteration}" / "renders"
        n = count_pngs(renders)
        results["3dgs"][name] = {
            "total_s": elapsed, "n_frames": n,
            "fps": n / elapsed if elapsed > 0 else 0,
            "ms_per_frame": elapsed * 1000 / n if n > 0 else 0,
        }

    # Cleanup renders
    for d in run_dir.iterdir():
        if d.is_dir() and d.name.startswith(("pv_", "dgs_", "gt_")):
            shutil.rmtree(d)

    # Report
    print(f"\n{'='*80}")
    print("RENDERING BENCHMARK: ParaView vs 3DGS")
    print(f"{'='*80}")
    pv_t, dgs_t = 0, 0
    for name in EVAL_DATASETS:
        pv = results["paraview"].get(name, {})
        dg = results["3dgs"].get(name, {})
        sp = pv.get("total_s", 1) / dg["total_s"] if dg.get("total_s", 0) > 0 else 0
        label = name.replace("eval_", "").upper()
        print(f"  {label}: ParaView {pv.get('total_s',0):.1f}s vs 3DGS {dg.get('total_s',0):.1f}s ({sp:.1f}x)")
        pv_t += pv.get("total_s", 0)
        dgs_t += dg.get("total_s", 0)

    sp = pv_t / dgs_t if dgs_t > 0 else 0
    print(f"  TOTAL: {sp:.1f}x speedup")

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()
