#!/usr/bin/env python3
"""Shared infrastructure for all ParticleGS experiments.

Provides: path constants, data preparation, training/rendering helpers,
PSNR computation, EGL GPU pinning, result logging.
"""

import argparse
import ctypes
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

# ── Path constants ────────────────────────────────────────────────────────

EXPERIMENTS_DIR = Path(__file__).resolve().parent
PARTICLEGS_ROOT = EXPERIMENTS_DIR.parent
REPO_ROOT = PARTICLEGS_ROOT.parent

RAW_DIR = PARTICLEGS_ROOT / "data" / "hacc_raw"
RAW_X = RAW_DIR / "xx.f32"
RAW_Y = RAW_DIR / "yy.f32"
RAW_Z = RAW_DIR / "zz.f32"

RUNS_DIR = PARTICLEGS_ROOT / "runs"
SHARED_DIR = RUNS_DIR / "shared"

PREPARE_SCRIPT = PARTICLEGS_ROOT / "data" / "prepare_data.py"
GENERATE_SCRIPT = PARTICLEGS_ROOT / "data" / "generate_images.py"
PARTITION_SCRIPT = PARTICLEGS_ROOT / "data" / "partition.py"
FROZEN_NORM = PARTICLEGS_ROOT / "data" / "shared_normalization.json"

# Dynamic path detection — no hardcoded user paths.
# sys.executable = the python running this script (correct conda env).
PYTHON_BIN = os.environ.get("PARTICLEGS_PYTHON", sys.executable)

# pvbatch lives alongside python in the conda env bin/ directory
def _find_pvbatch():
    # 1. Env var override
    env_val = os.environ.get("PARTICLEGS_PVBATCH")
    if env_val:
        return env_val
    # 2. Same directory as python (conda env)
    py_dir = Path(sys.executable).parent
    candidate = py_dir / "pvbatch"
    if candidate.exists():
        return str(candidate)
    # 3. PATH
    found = shutil.which("pvbatch")
    if found:
        return found
    # 4. Fallback
    return "pvbatch"

PVBATCH_BIN = _find_pvbatch()
SZ3_BIN = os.environ.get("PARTICLEGS_SZ3",
    str(PARTICLEGS_ROOT / "SZ3" / "build" / "tools" / "sz3" / "sz3")
    if (PARTICLEGS_ROOT / "SZ3" / "build" / "tools" / "sz3" / "sz3").exists()
    else str(REPO_ROOT / "SZ3" / "build" / "tools" / "sz3" / "sz3")
    if (REPO_ROOT / "SZ3" / "build" / "tools" / "sz3" / "sz3").exists()
    else shutil.which("sz3") or "sz3")

NUM_PARTICLES = 280_953_867
NUM_INIT_PLY = 200_000
BG_THRESH = 2.5

# Eval datasets used across all experiments
EVAL_DATASETS = [
    {"id": "eval_far",  "orbit_radii": "1.0", "subdir": "01_eval_far"},
    {"id": "eval_mid",  "orbit_radii": "0.7", "subdir": "02_eval_mid"},
    {"id": "eval_near", "orbit_radii": "0.5", "subdir": "03_eval_near"},
]

# Default viz rendering parameters (beta distribution, concentration=3.0)
DEFAULT_VIZ_PARAMS = {
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
}


# ── Shell helpers ─────────────────────────────────────────────────────────

def run_cmd(cmd, log_path=None, cwd=None, env=None, timeout=None):
    """Run command, optionally logging stdout/stderr to file."""
    cmd_str = [str(c) for c in cmd]
    label = " ".join(cmd_str[-6:])
    print(f"  $ {label}")
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            r = subprocess.run(cmd_str, stdout=f, stderr=subprocess.STDOUT,
                               cwd=str(cwd or PARTICLEGS_ROOT), env=env,
                               timeout=timeout)
    else:
        r = subprocess.run(cmd_str, cwd=str(cwd or PARTICLEGS_ROOT), env=env,
                           timeout=timeout)
    if r.returncode != 0:
        print(f"  FAILED (exit {r.returncode})")
        if log_path and Path(log_path).exists():
            for line in Path(log_path).read_text().splitlines()[-20:]:
                print(f"    {line}")
        raise RuntimeError(f"Command failed: {label}")
    return r.returncode


# ── EGL GPU pinning ───────────────────────────────────────────────────────

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


def get_pvbatch_egl_args(cuda_device=0):
    """Get pvbatch arguments for EGL GPU pinning to a given CUDA device.

    pvbatch honors neither CUDA_VISIBLE_DEVICES nor EGL_VISIBLE_DEVICES, so to
    render on CUDA device ``cuda_device`` we pass ``--displays <egl_idx>`` where
    egl_idx is the EGL display whose EGL_DEVICE_CUDA_NV attribute maps to that
    CUDA index (the EGL order differs from nvidia-smi order on this system).

    On hosts where the EGL→CUDA probe succeeds (multi-GPU workstation), pin to
    the EGL display matching ``cuda_device``. Otherwise fall back to EGL display
    ``cuda_device`` — required in Docker where no X server is available and plain
    --force-offscreen-rendering still defaults to GLX and segfaults.

    Passing the target device is what lets concurrent renders spread across GPUs
    instead of all piling onto CUDA 0 (the previous hardcoded behavior).
    """
    egl_to_cuda = _probe_egl_cuda_mapping()
    if egl_to_cuda is not None:
        for egl_idx, cuda_idx in egl_to_cuda.items():
            if cuda_idx == cuda_device:
                return ["--force-offscreen-rendering",
                        "--opengl-window-backend", "EGL",
                        "--displays", str(egl_idx), "--"]
        # Requested CUDA device not in the probed mapping — fall through to the
        # best-effort display index below rather than silently pinning CUDA 0.
    return ["--force-offscreen-rendering",
            "--opengl-window-backend", "EGL",
            "--displays", str(cuda_device), "--"]


# Process-wide default CUDA device for pvbatch rendering. Each experiment sets
# this once from its --gpu (set_pvbatch_cuda_device) so every GT render in that
# process pins to its assigned GPU instead of all piling onto CUDA 0. Because
# exp4 trains/prepares blocks in separate ProcessPoolExecutor processes, each
# worker gets its own copy of this global and can pin to its own block_gpu.
_PVBATCH_CUDA_DEVICE = 0


def set_pvbatch_cuda_device(cuda_device):
    """Set the process-wide default GPU for subsequent pvbatch renders."""
    global _PVBATCH_CUDA_DEVICE
    _PVBATCH_CUDA_DEVICE = int(cuda_device)


def pvbatch_cmd(script, *args, cuda_device=None):
    """Build a pvbatch command with EGL pinning.

    ``cuda_device=None`` uses the process default (``set_pvbatch_cuda_device``,
    itself defaulting to 0). Concurrent callers that need a specific GPU — e.g.
    exp4 per-block prepare — pass ``cuda_device`` explicitly to override it.
    """
    dev = _PVBATCH_CUDA_DEVICE if cuda_device is None else cuda_device
    return ([PVBATCH_BIN] + get_pvbatch_egl_args(dev)
            + [str(script)] + [str(a) for a in args])


# ── PSNR computation ─────────────────────────────────────────────────────

def compute_psnr_pair(gt_path, render_path):
    """Compute full and masked PSNR for a single image pair."""
    g = np.array(Image.open(gt_path)).astype(np.float64)
    r = np.array(Image.open(render_path)).astype(np.float64)
    if g.shape != r.shape:
        return None, None
    mse = np.mean((g - r) ** 2)
    psnr = 100.0 if mse == 0 else 20 * math.log10(255.0 / math.sqrt(mse))
    mask = (g.mean(2) > BG_THRESH) | (r.mean(2) > BG_THRESH)
    if mask.sum() == 0:
        return psnr, None
    m3 = np.stack([mask] * 3, axis=2)
    mse_m = np.mean((g[m3] - r[m3]) ** 2)
    mpsnr = 100.0 if mse_m == 0 else 20 * math.log10(255.0 / math.sqrt(mse_m))
    return psnr, mpsnr


def compute_psnr_dirs(gt_dir, render_dir, max_workers=8):
    """Compute average full and masked PSNR between two directories of PNGs.

    Matches files by sorted index (not filename), handling different naming
    conventions (e.g., 0000.png vs 00000.png).
    """
    gt_dir, render_dir = Path(gt_dir), Path(render_dir)
    if not gt_dir.exists() or not render_dir.exists():
        return None, None, 0
    gt_files = sorted(p for p in gt_dir.iterdir() if p.suffix.lower() == ".png")
    render_files = sorted(p for p in render_dir.iterdir() if p.suffix.lower() == ".png")
    if not gt_files or not render_files:
        return None, None, 0

    # Match by sorted index (handles 0000.png vs 00000.png)
    n = min(len(gt_files), len(render_files))
    pairs = list(zip(gt_files[:n], render_files[:n]))

    def _one(pair):
        gp, rp = pair
        try:
            return compute_psnr_pair(gp, rp)
        except Exception:
            return None, None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_one, pairs))
    ps = [r[0] for r in results if r[0] is not None]
    ms = [r[1] for r in results if r[1] is not None]
    return (sum(ps)/len(ps) if ps else None,
            sum(ms)/len(ms) if ms else None,
            len(ps))


def compute_ssim_dirs(gt_dir, render_dir, max_workers=8):
    """Compute average SSIM between two directories of PNGs."""
    from skimage.metrics import structural_similarity
    gt_dir, render_dir = Path(gt_dir), Path(render_dir)
    if not gt_dir.exists() or not render_dir.exists():
        return None, 0
    gt_files = sorted(p for p in gt_dir.iterdir() if p.suffix.lower() == ".png")
    render_files = sorted(p for p in render_dir.iterdir() if p.suffix.lower() == ".png")
    if not gt_files or not render_files:
        return None, 0
    n = min(len(gt_files), len(render_files))
    pairs = list(zip(gt_files[:n], render_files[:n]))

    def _one(pair):
        gp, rp = pair
        try:
            g = np.array(Image.open(gp)).astype(np.uint8)
            r = np.array(Image.open(rp)).astype(np.uint8)
            if g.shape != r.shape:
                return None
            return structural_similarity(g, r, channel_axis=2, data_range=255)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_one, pairs))
    ssims = [v for v in results if v is not None]
    if not ssims:
        return None, 0
    return sum(ssims) / len(ssims), len(ssims)


# ── Shared data preparation ──────────────────────────────────────────────

def _record_shared_timing(key, seconds):
    """Persist a shared-prep timing so EXP-11 can report it later.

    Shared data is generated once and cached, so timings must survive
    across runs in a file rather than being re-measured.
    """
    path = SHARED_DIR / "timings.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = seconds
    path.write_text(json.dumps(data, indent=2))


def ensure_shared_data(gpu=0):
    """Prepare VTP, normalization.json, points3d.ply, and eval GT images.

    Returns dict with paths to all shared data.
    """
    set_pvbatch_cuda_device(gpu)
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    logs = SHARED_DIR / "logs"
    logs.mkdir(exist_ok=True)

    vtp_path = SHARED_DIR / "particles.vtp"
    norm_path = SHARED_DIR / "normalization.json"
    ply_path = SHARED_DIR / "points3d.ply"

    # Phase 1: Create VTP (pvbatch required for vtk import)
    if not vtp_path.exists():
        print("\n[Shared] Creating VTP from raw data...")
        t0 = time.perf_counter()
        run_cmd(
            pvbatch_cmd(PREPARE_SCRIPT,
                        "--raw_x", RAW_X, "--raw_y", RAW_Y, "--raw_z", RAW_Z,
                        "--output_dir", SHARED_DIR,
                        "--num_points_raw", "0",
                        "--skip_images"),
            log_path=logs / "create_vtp.log")
        _record_shared_timing("vtp_conversion_s", round(time.perf_counter() - t0, 1))
    else:
        print(f"[Shared] VTP exists: {vtp_path}")

    # Phase 2: Generate normalization.json via first eval orbit render
    eval_dirs = {}
    first_eval = EVAL_DATASETS[0]
    first_eval_dir = SHARED_DIR / first_eval["subdir"] / "data"

    if not norm_path.exists() and FROZEN_NORM.exists():
        print(f"\n[Shared] Using frozen normalization: {FROZEN_NORM}")
        shutil.copy2(FROZEN_NORM, norm_path)

    if not norm_path.exists():
        print("\n[Shared] Generating normalization.json + first eval images...")
        _generate_eval_images(
            vtp_path, first_eval_dir, first_eval["orbit_radii"],
            norm_path=None, logs_dir=logs, eval_id=first_eval["id"])
        first_norm = first_eval_dir / "normalization.json"
        if first_norm.exists():
            shutil.copy2(first_norm, norm_path)
    else:
        print(f"[Shared] Normalization exists: {norm_path}")

    # Generate remaining eval GT images
    for evd in EVAL_DATASETS:
        ed = SHARED_DIR / evd["subdir"] / "data"
        eval_dirs[evd["id"]] = ed
        img_dir = ed / "images"
        if img_dir.exists() and len(list(img_dir.glob("*.png"))) >= 80:
            print(f"[Shared] Eval images exist: {evd['id']}")
            continue
        print(f"\n[Shared] Generating eval images: {evd['id']}...")
        _generate_eval_images(
            vtp_path, ed, evd["orbit_radii"],
            norm_path=norm_path, logs_dir=logs, eval_id=evd["id"])

    # Phase 3: Create initial PLY
    if not ply_path.exists():
        print("\n[Shared] Creating initial PLY...")
        run_cmd(
            [PYTHON_BIN, str(PREPARE_SCRIPT),
             "--raw_x", str(RAW_X), "--raw_y", str(RAW_Y), "--raw_z", str(RAW_Z),
             "--output_dir", str(SHARED_DIR),
             "--only_ply", "--num_points_ply", str(NUM_INIT_PLY)],
            log_path=logs / "create_ply.log")
    else:
        print(f"[Shared] PLY exists: {ply_path}")

    return {
        "vtp": vtp_path,
        "normalization": norm_path,
        "ply": ply_path,
        "eval_dirs": eval_dirs,
    }


def _generate_eval_images(vtp_path, output_dir, orbit_radii, norm_path, logs_dir, eval_id):
    """Generate eval GT images for one orbit."""
    args = [
        "--vtp_path", str(vtp_path),
        "--output_dir", str(output_dir),
        "--camera_strategy", "multi_orbit",
        "--orbit_radii", orbit_radii,
        "--num_frames", "80",
        "--width", "1920", "--height", "1080",
        "--train_ratio", "1.0",
        "--split_seed", "42",
    ]
    for k, v in DEFAULT_VIZ_PARAMS.items():
        args += [f"--{k}", v]
    if norm_path and Path(norm_path).exists():
        args += ["--normalization_path", str(norm_path)]

    run_cmd(pvbatch_cmd(GENERATE_SCRIPT, *args),
            log_path=logs_dir / f"generate_{eval_id}.log")


# ── Training data generation ──────────────────────────────────────────────

def generate_training_data(vtp_path, output_dir, norm_path, camera_strategy,
                           orbit_radii, num_frames, width, height,
                           logs_dir, stage_name, viz_seed="142",
                           internal_bounds_scale=None, camera_seed=42,
                           train_ratio=1.0, split_seed=42, viz_params=None):
    """Generate training images with ParaView."""
    output_dir = Path(output_dir)
    img_dir = output_dir / "images"
    if img_dir.exists() and len(list(img_dir.glob("*.png"))) >= num_frames:
        print(f"  [Skip] Training images exist: {stage_name} ({len(list(img_dir.glob('*.png')))} frames)")
        return

    args = [
        "--vtp_path", str(vtp_path),
        "--output_dir", str(output_dir),
        "--camera_strategy", camera_strategy,
        "--num_frames", str(num_frames),
        "--width", str(width), "--height", str(height),
        "--train_ratio", str(train_ratio),
        "--split_seed", str(split_seed),
        "--camera_seed", str(camera_seed),
    ]
    if orbit_radii:
        args += ["--orbit_radii", orbit_radii]
    if internal_bounds_scale is not None:
        args += ["--internal_bounds_scale", str(internal_bounds_scale)]

    viz = dict(viz_params or DEFAULT_VIZ_PARAMS)
    viz["viz_seed"] = str(viz_seed)
    for k, v in viz.items():
        args += [f"--{k}", v]
    if norm_path:
        args += ["--normalization_path", str(norm_path)]

    run_cmd(pvbatch_cmd(GENERATE_SCRIPT, *args),
            log_path=logs_dir / f"generate_{stage_name}.log")


def prepare_stage_data_dir(data_dir, ply_path, norm_path):
    """Symlink points3d.ply and normalization.json into a training data directory."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dst_ply = data_dir / "points3d.ply"
    if not dst_ply.exists():
        os.symlink(str(Path(ply_path).resolve()), str(dst_ply))
    dst_norm = data_dir / "normalization.json"
    if not dst_norm.exists():
        shutil.copy2(str(norm_path), str(dst_norm))
    # Training code requires transforms_test.json; create empty if missing
    dst_test = data_dir / "transforms_test.json"
    if not dst_test.exists():
        train_tf = data_dir / "transforms_train.json"
        if train_tf.exists():
            tf = json.loads(train_tf.read_text())
            test_tf = {"camera_angle_x": tf.get("camera_angle_x", 0), "frames": []}
            dst_test.write_text(json.dumps(test_tf, indent=2))


def generate_mix_training_data(vtp_path, output_dir, norm_path,
                               ext_orbit_radii, ext_num_frames, ext_seed,
                               int_num_frames, int_seed, int_bounds_scale,
                               width, height, mix_ratio,
                               logs_dir, stage_name, split_seed=42,
                               viz_params=None):
    """Generate mixed external + internal training data."""
    output_dir = Path(output_dir)
    img_dir = output_dir / "images"
    tf = output_dir / "transforms_train.json"
    if tf.exists() and img_dir.exists() and len(list(img_dir.glob("*.png"))) > 0:
        print(f"  [Skip] Mix data exists: {stage_name}")
        return

    ext_dir = output_dir / "_ext"
    int_dir = output_dir / "_int"

    generate_training_data(
        vtp_path, ext_dir, norm_path, "multi_orbit", ext_orbit_radii,
        ext_num_frames, width, height, logs_dir, f"{stage_name}_ext",
        viz_seed=ext_seed, train_ratio=1.0, split_seed=split_seed,
        viz_params=viz_params)

    generate_training_data(
        vtp_path, int_dir, norm_path, "internal_uniform", None,
        int_num_frames, width, height, logs_dir, f"{stage_name}_int",
        viz_seed=int_seed, internal_bounds_scale=int_bounds_scale,
        train_ratio=1.0, split_seed=split_seed,
        viz_params=viz_params)

    # Merge
    img_dir.mkdir(parents=True, exist_ok=True)
    ext_tf = json.loads((ext_dir / "transforms_train.json").read_text())
    int_tf = json.loads((int_dir / "transforms_train.json").read_text())

    ext_frames = ext_tf["frames"]
    int_frames = int_tf["frames"]

    n_ext = int(len(ext_frames) * mix_ratio)
    rng = np.random.RandomState(split_seed)
    ext_sel = rng.choice(len(ext_frames), n_ext, replace=False)
    int_sel = rng.choice(len(int_frames), len(ext_frames) - n_ext, replace=False)

    merged_frames = []
    idx = 0
    for i in ext_sel:
        src = ext_dir / (ext_frames[i]["file_path"] + ".png")
        dst_name = f"{idx:04d}.png"
        shutil.copy2(str(src), str(img_dir / dst_name))
        frame = dict(ext_frames[i])
        frame["file_path"] = f"images/{idx:04d}"
        merged_frames.append(frame)
        idx += 1
    for i in int_sel:
        src = int_dir / (int_frames[i]["file_path"] + ".png")
        dst_name = f"{idx:04d}.png"
        shutil.copy2(str(src), str(img_dir / dst_name))
        frame = dict(int_frames[i])
        frame["file_path"] = f"images/{idx:04d}"
        merged_frames.append(frame)
        idx += 1

    merged_tf = {"camera_angle_x": ext_tf["camera_angle_x"], "frames": merged_frames}
    (output_dir / "transforms_train.json").write_text(json.dumps(merged_tf, indent=2))
    # Training code requires transforms_test.json even if empty
    test_tf = {"camera_angle_x": ext_tf["camera_angle_x"], "frames": []}
    (output_dir / "transforms_test.json").write_text(json.dumps(test_tf, indent=2))

    shutil.rmtree(ext_dir)
    shutil.rmtree(int_dir)


# ── Training ──────────────────────────────────────────────────────────────

def _get_checkpoint_iteration(checkpoint_path):
    """Extract iteration number from checkpoint filename (chkpntN.pth)."""
    if not checkpoint_path:
        return 0
    import re
    m = re.search(r'chkpnt(\d+)\.pth', str(checkpoint_path))
    return int(m.group(1)) if m else 0


def run_training(data_dir, model_dir, train_cfg, logs_dir, log_name,
                 start_checkpoint=None, gpu=0):
    """Run particlegs training.

    train_cfg: dict with training hyperparameters.
    train_cfg["iterations"] is the number of iterations FOR THIS STAGE (relative).
    Internally computes absolute iteration = base_iter + stage_iters for train.py.

    Returns the absolute iteration number of the final checkpoint.
    """
    stage_iters = train_cfg["iterations"]
    base_iter = _get_checkpoint_iteration(start_checkpoint)
    # train.py uses absolute iteration numbering:
    #   for iteration in range(first_iter+1, opt.iterations+1)
    # So we need: opt.iterations = base_iter + stage_iters
    abs_iterations = base_iter + stage_iters

    cmd = [
        PYTHON_BIN, "-m", "particlegs.training.train",
        "--source_path", str(data_dir),
        "--model_path", str(model_dir),
        "--iterations", str(abs_iterations),
        "--sh_degree", str(train_cfg.get("sh_degree", 0)),
        "--position_lr_init", str(train_cfg.get("position_lr_init", 0.00016)),
        "--position_lr_final", str(train_cfg.get("position_lr_final", 0.0000016)),
        "--position_lr_max_steps", str(train_cfg.get("position_lr_max_steps", 50000)),
        "--scaling_lr", str(train_cfg.get("scaling_lr", 0.005)),
        "--feature_lr", str(train_cfg.get("feature_lr", 0.0025)),
        "--opacity_lr", str(train_cfg.get("opacity_lr", 0.025)),
        "--rotation_lr", str(train_cfg.get("rotation_lr", 0.001)),
        "--lambda_dssim", str(train_cfg.get("lambda_dssim", 0.0)),
        "--lambda_identity", str(train_cfg.get("lambda_identity", 0.0)),
        "--densification_interval", str(train_cfg.get("densification_interval", 100)),
        "--densify_grad_threshold", str(train_cfg.get("densify_grad_threshold", 0.0004)),
        "--densify_from_iter", str(train_cfg.get("densify_from_iter", 500)),
        "--densify_until_iter", str(train_cfg.get("densify_until_iter", abs_iterations)),
        "--percent_dense", str(train_cfg.get("percent_dense", 0.01)),
        "--opacity_reset_interval", str(train_cfg.get("opacity_reset_interval", 3000)),
        "--min_opacity", str(train_cfg.get("min_opacity", 0.005)),
        "--factor_delta_opacity", str(train_cfg.get("factor_delta_opacity", 0.3)),
        "--factor_delta_scale", str(train_cfg.get("factor_delta_scale", 0.1)),
        "--min_opacity_clamp", str(train_cfg.get("min_opacity_clamp", 0.4)),
        "--save_iterations", str(abs_iterations),
        "--checkpoint_iterations", str(abs_iterations),
        "--test_iterations", str(abs_iterations),
        "--disable_viewer",
        "--quiet",
        "--data_device", "cpu",
    ]

    if train_cfg.get("antialiasing", True):
        cmd.append("--antialiasing")
    if train_cfg.get("content_mask_loss", False):
        cmd.append("--content_mask_loss")
    if train_cfg.get("static_viz", False):
        cmd.append("--static_viz")

    res = train_cfg.get("resolution_scale", 1)
    cmd.extend(["--resolution", str(res)])

    min_scale_pixels = train_cfg.get("min_scale_pixels")
    if min_scale_pixels is not None:
        cmd.extend(["--min_scale_pixels", str(min_scale_pixels)])

    if start_checkpoint:
        cmd.extend(["--start_checkpoint", str(start_checkpoint)])

    extra = train_cfg.get("extra_args", "")
    if extra:
        cmd.extend(extra.split())

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    run_cmd(cmd, log_path=logs_dir / f"{log_name}.log", env=env)
    return abs_iterations


def run_init_training(data_dir, model_dir, train_cfg, logs_dir, log_name,
                      start_checkpoint=None, init_iterations=10, gpu=0):
    """Run a short initialization phase before main training.

    The old pipeline did this: 10 iterations with default params to initialize
    the Gaussian model before the main training loop.
    Returns the checkpoint path for the init phase.
    """
    base_iter = _get_checkpoint_iteration(start_checkpoint)
    init_abs = base_iter + init_iterations

    cmd = [
        PYTHON_BIN, "-m", "particlegs.training.train",
        "--source_path", str(data_dir),
        "--model_path", str(model_dir),
        "--iterations", str(init_abs),
        "--sh_degree", str(train_cfg.get("sh_degree", 0)),
        "--position_lr_init", str(train_cfg.get("position_lr_init", 0.00016)),
        "--position_lr_final", str(train_cfg.get("position_lr_final", 0.0000016)),
        "--position_lr_max_steps", str(train_cfg.get("position_lr_max_steps", 50000)),
        "--scaling_lr", str(train_cfg.get("scaling_lr", 0.005)),
        "--feature_lr", str(train_cfg.get("feature_lr", 0.0025)),
        "--opacity_lr", str(train_cfg.get("opacity_lr", 0.025)),
        "--rotation_lr", str(train_cfg.get("rotation_lr", 0.001)),
        "--lambda_dssim", "0.0",
        "--lambda_identity", "0.0",
        "--densification_interval", "99999",
        "--densify_from_iter", "99999",
        "--densify_until_iter", "0",
        "--opacity_reset_interval", "99999",
        "--factor_delta_opacity", str(train_cfg.get("factor_delta_opacity", 0.3)),
        "--factor_delta_scale", str(train_cfg.get("factor_delta_scale", 0.1)),
        "--min_opacity_clamp", str(train_cfg.get("min_opacity_clamp", 0.4)),
        "--save_iterations", str(init_abs),
        "--checkpoint_iterations", str(init_abs),
        "--test_iterations", "-1",
        "--disable_viewer",
        "--quiet",
        "--data_device", "cpu",
    ]

    if train_cfg.get("antialiasing", True):
        cmd.append("--antialiasing")

    res = train_cfg.get("resolution_scale", 1)
    cmd.extend(["--resolution", str(res)])

    if start_checkpoint:
        cmd.extend(["--start_checkpoint", str(start_checkpoint)])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    run_cmd(cmd, log_path=logs_dir / f"{log_name}_init.log", env=env)
    return find_checkpoint(model_dir, init_abs)


def run_stage_training(data_dir, model_dir, train_cfg, logs_dir, log_name,
                       start_checkpoint=None, init_iterations=10, gpu=0):
    """Run a complete training stage: init phase + main training.

    Matches the old run_experiment.py 2-phase approach:
      Phase 1: init_iterations with no densification
      Phase 2: main training with full config

    Returns absolute iteration of final checkpoint.
    """
    # Phase 1: Init
    init_chk = run_init_training(
        data_dir, model_dir, train_cfg, logs_dir, log_name,
        start_checkpoint=start_checkpoint, init_iterations=init_iterations, gpu=gpu)

    # Phase 2: Main training
    abs_iter = run_training(
        data_dir, model_dir, train_cfg, logs_dir, log_name,
        start_checkpoint=init_chk, gpu=gpu)

    # Cleanup init checkpoint and viz_mapper (keep only final)
    if init_chk and Path(init_chk).exists():
        init_iter = _get_checkpoint_iteration(init_chk)
        Path(init_chk).unlink(missing_ok=True)
        vm_init = Path(model_dir) / f"viz_mapper_{init_iter}.pth"
        vm_init.unlink(missing_ok=True)
        # Remove init point cloud dir
        init_pc = Path(model_dir) / "point_cloud" / f"iteration_{init_iter}"
        if init_pc.exists():
            shutil.rmtree(init_pc)

    return abs_iter


# ── Rendering ─────────────────────────────────────────────────────────────

def run_render(model_dir, data_dir, iteration, logs_dir, log_name, gpu=0,
               factor_delta_opacity=0.3, factor_delta_scale=0.1,
               min_opacity_clamp=0.4, min_scale_pixels=0.0,
               extra_args=None, static_viz=False):
    """Run particlegs rendering on a dataset.

    Note: min_scale_pixels defaults to 0.0 for eval (not the training value).
    factor_delta_* should match training config for correct VizMapper behavior.
    """
    cmd = [
        PYTHON_BIN, "-m", "particlegs.evaluation.render",
        "--model_path", str(model_dir),
        "--source_path", str(data_dir),
        "--iteration", str(iteration),
        "--skip_test",  # eval data uses train_ratio=1.0, all frames are "train"
        "--antialiasing",
        "--quiet",
        "--factor_delta_opacity", str(factor_delta_opacity),
        "--factor_delta_scale", str(factor_delta_scale),
        "--min_opacity_clamp", str(min_opacity_clamp),
        "--min_scale_pixels", str(min_scale_pixels),
    ]
    if static_viz:
        cmd.append("--static_viz")
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    run_cmd(cmd, log_path=logs_dir / f"{log_name}.log", env=env)


# ── Evaluation ────────────────────────────────────────────────────────────

def evaluate_model(model_dir, iteration, shared_data, logs_dir, gpu=0,
                   factor_delta_opacity=0.3, factor_delta_scale=0.1,
                   min_opacity_clamp=0.4, static_viz=False,
                   compute_ssim=False):
    """Render eval datasets and compute PSNR (and optionally SSIM).

    VizMapper params should match the training config.
    static_viz: if True, render with viz factors=1.0 (static model).
    compute_ssim: if True, also compute SSIM for each eval dataset.
    Returns dict: {eval_id: {psnr, masked_psnr, ssim, n}, "avg": {...}}.
    """
    results = {}
    for evd in EVAL_DATASETS:
        eval_id = evd["id"]
        data_dir = shared_data["eval_dirs"][eval_id]
        gt_dir = data_dir / "images"

        # Use separate output dir per eval dataset to avoid overwriting
        eval_render_dir = Path(model_dir).parent / "eval_renders" / eval_id
        eval_render_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Rendering {eval_id}...")
        run_render(model_dir, data_dir, iteration, logs_dir,
                   f"render_{eval_id}", gpu=gpu,
                   factor_delta_opacity=factor_delta_opacity,
                   factor_delta_scale=factor_delta_scale,
                   min_opacity_clamp=min_opacity_clamp,
                   min_scale_pixels=0.0,  # eval always uses 0.0
                   static_viz=static_viz,
                   extra_args=["--output_dir", str(eval_render_dir),
                               "--resolution", "1"])

        render_dir = eval_render_dir / "train" / f"ours_{iteration}" / "renders"
        psnr, mpsnr, n = compute_psnr_dirs(gt_dir, render_dir)
        entry = {"psnr": psnr, "masked_psnr": mpsnr, "n": n}
        if compute_ssim:
            ssim_val, _ = compute_ssim_dirs(gt_dir, render_dir)
            entry["ssim"] = ssim_val
        results[eval_id] = entry
        # psnr/mpsnr can independently be None (mpsnr is None when a pair has no
        # foreground above BG_THRESH, e.g. an all-black render from a bad/stale
        # VTP). Format each guarded so logging never crashes on None.
        if psnr is None:
            msg = f"    {eval_id}: N/A"
        else:
            mp_str = f"{mpsnr:.2f}" if mpsnr is not None else "N/A"
            msg = f"    {eval_id}: PSNR={psnr:.2f}  masked={mp_str}"
        if compute_ssim and entry.get("ssim"):
            msg += f"  SSIM={entry['ssim']:.4f}"
        if psnr is not None:
            msg += f"  (n={n})"
        print(msg)

    # Cleanup eval renders (saves ~600 MB per experiment)
    eval_renders_root = Path(model_dir).parent / "eval_renders"
    if eval_renders_root.exists():
        shutil.rmtree(eval_renders_root)

    # Compute averages
    all_p = [v["psnr"] for v in results.values() if v.get("psnr")]
    all_m = [v["masked_psnr"] for v in results.values() if v.get("masked_psnr")]
    avg = {
        "psnr": sum(all_p)/len(all_p) if all_p else None,
        "masked_psnr": sum(all_m)/len(all_m) if all_m else None,
    }
    if compute_ssim:
        all_s = [v["ssim"] for v in results.values() if v.get("ssim")]
        avg["ssim"] = sum(all_s)/len(all_s) if all_s else None
    results["avg"] = avg
    return results


def get_model_stats(model_dir, iteration):
    """Get model size and gaussian count from saved PLY."""
    ply_path = Path(model_dir) / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    stats = {"size_mb": 0, "num_gaussians": 0}
    if ply_path.exists():
        stats["size_mb"] = ply_path.stat().st_size / (1024 * 1024)
        with open(ply_path, "rb") as f:
            for line in f:
                line = line.decode("ascii", errors="ignore").strip()
                if line.startswith("element vertex"):
                    stats["num_gaussians"] = int(line.split()[-1])
                    break
                if line == "end_header":
                    break
    return stats


def find_checkpoint(model_dir, iteration=None):
    """Find checkpoint path in model directory."""
    model_dir = Path(model_dir)
    if iteration:
        p = model_dir / f"chkpnt{iteration}.pth"
        if p.exists():
            return p
    chkpts = sorted(model_dir.glob("chkpnt*.pth"))
    return chkpts[-1] if chkpts else None


# ── Result helpers ────────────────────────────────────────────────────────

def save_results(results, path):
    """Save results dict to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {path}")


def print_table(headers, rows, col_widths=None):
    """Print a formatted table."""
    if not rows:
        return
    if col_widths is None:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2
                      for i, h in enumerate(headers)]
    header = "| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, col_widths)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"
    print(header)
    print(sep)
    for row in rows:
        print("| " + " | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)) + " |")


def get_shared_data_dict():
    """Get shared data paths dict without running preparation."""
    return {
        "vtp": SHARED_DIR / "particles.vtp",
        "normalization": SHARED_DIR / "normalization.json",
        "ply": SHARED_DIR / "points3d.ply",
        "eval_dirs": {evd["id"]: SHARED_DIR / evd["subdir"] / "data"
                      for evd in EVAL_DATASETS},
    }


# ── Common argparse ───────────────────────────────────────────────────────

def base_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--skip_data_prep", action="store_true",
                        help="Skip shared data preparation (assume exists)")
    parser.add_argument("--ae", action="store_true",
                        help="AE mode: reduce report-only sampling (eval/FPS "
                             "frames, redundant profiling reruns) to fit the SC "
                             "AE budget. Never changes enforced metrics; left off "
                             "by full reproduce.sh.")
    return parser
