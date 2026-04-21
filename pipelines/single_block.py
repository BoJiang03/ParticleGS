#!/usr/bin/env python3
"""Single-block training pipeline (E25 config).

Runs a 3-stage progressive training: 4K -> 6K -> mix 6K.
This is the simplest pipeline for quick experiments.

Usage:
    python single_block.py --name E25_test --config configs/training/E25_single_block.json
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PARTICLEGS_ROOT = SCRIPT_DIR.parent

PYTHON_BIN = sys.executable
PVBATCH_BIN = shutil.which("pvbatch") or "pvbatch"

PREPARE_SCRIPT = PARTICLEGS_ROOT / "data" / "prepare_data.py"
GENERATE_SCRIPT = PARTICLEGS_ROOT / "data" / "generate_images.py"

RAW_DIR = PARTICLEGS_ROOT / "data" / "hacc_raw"

DEFAULT_CONFIG = PARTICLEGS_ROOT / "configs" / "training" / "E25_single_block.json"

_BG_THRESH = 2.5


def run(cmd, log_path=None, env=None):
    cmd_str = [str(c) for c in cmd]
    short = " ".join(cmd_str[-6:])
    print(f"  $ {short}")
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            r = subprocess.run(cmd_str, stdout=f, stderr=subprocess.STDOUT,
                               cwd=str(PARTICLEGS_ROOT), env=env)
    else:
        r = subprocess.run(cmd_str, cwd=str(PARTICLEGS_ROOT), env=env)
    if r.returncode != 0:
        print(f"  FAILED (exit {r.returncode})")
        if log_path and Path(log_path).exists():
            for line in Path(log_path).read_text().splitlines()[-20:]:
                print(f"    {line}")
    return r.returncode


def pvbatch_prefix():
    return [PVBATCH_BIN, "--force-offscreen-rendering",
            "--opengl-window-backend", "EGL", "--"]


def compute_psnr_from_dirs(gt_dir, render_dir):
    gt_dir, render_dir = Path(gt_dir), Path(render_dir)
    if not gt_dir.exists() or not render_dir.exists():
        return None, None, 0
    gt_files = sorted(p for p in gt_dir.iterdir() if p.suffix.lower() == ".png")
    if not gt_files:
        return None, None, 0

    from concurrent.futures import ThreadPoolExecutor

    def _one(gp):
        rp = render_dir / gp.name
        if not rp.exists():
            return None, None
        try:
            g = np.array(Image.open(gp)).astype(np.float64)
            r = np.array(Image.open(rp)).astype(np.float64)
        except Exception:
            return None, None
        if g.shape != r.shape:
            return None, None
        mse = np.mean((g - r) ** 2)
        psnr = 100.0 if mse == 0 else 20 * math.log10(255 / math.sqrt(mse))
        mask = (g.mean(2) > _BG_THRESH) | (r.mean(2) > _BG_THRESH)
        if mask.sum() == 0:
            return psnr, None
        m3 = np.stack([mask] * 3, axis=2)
        mse_m = np.mean((g[m3] - r[m3]) ** 2)
        mp = 100.0 if mse_m == 0 else 20 * math.log10(255 / math.sqrt(mse_m))
        return psnr, mp

    with ThreadPoolExecutor(8) as pool:
        res = list(pool.map(_one, gt_files))
    ps = [r[0] for r in res if r[0] is not None]
    ms = [r[1] for r in res if r[1] is not None]
    return (sum(ps)/len(ps) if ps else None,
            sum(ms)/len(ms) if ms else None,
            len(ps))


def run_training_stage(stage_cfg, model_path, prev_model_path, data_dir, logs_dir,
                       stage_idx, gpu=0):
    """Run a single training stage."""
    train_cfg = stage_cfg["training"]
    iterations = train_cfg["iterations"]

    cmd = [
        PYTHON_BIN, "-m", "particlegs.training.train",
        "--source_path", str(data_dir),
        "--model_path", str(model_path),
        "--iterations", str(iterations),
        "--sh_degree", "0",
        "--resolution", str(stage_cfg["data"].get("resolution", 1)),
        "--position_lr_init", str(train_cfg.get("position_lr_init", 0.00016)),
        "--position_lr_final", str(train_cfg.get("position_lr_final", 0.0000016)),
        "--position_lr_max_steps", str(iterations),
        "--scaling_lr", str(train_cfg.get("scaling_lr", 0.001)),
        "--feature_lr", str(train_cfg.get("feature_lr", 0.0025)),
        "--opacity_lr", str(train_cfg.get("opacity_lr", 0.025)),
        "--rotation_lr", str(train_cfg.get("rotation_lr", 0.001)),
        "--lambda_dssim", str(train_cfg.get("lambda_dssim", 0.0)),
        "--lambda_identity", str(train_cfg.get("lambda_identity", 0.0)),
        "--densification_interval", str(train_cfg.get("densification_interval", 100)),
        "--densify_grad_threshold", str(train_cfg.get("densify_grad_threshold", 0.0004)),
        "--densify_from_iter", str(train_cfg.get("densify_from_iter", 500)),
        "--densify_until_iter", str(train_cfg.get("densify_until_iter", iterations)),
        "--percent_dense", str(train_cfg.get("percent_dense", 0.01)),
        "--opacity_reset_interval", str(train_cfg.get("opacity_reset_interval", 3000)),
        "--min_opacity", str(train_cfg.get("min_opacity", 0.005)),
        "--factor_delta_opacity", str(train_cfg.get("factor_delta_opacity", 0.3)),
        "--factor_delta_scale", str(train_cfg.get("factor_delta_scale", 0.1)),
        "--min_opacity_clamp", str(train_cfg.get("min_opacity_clamp", 0.4)),
        "--save_iterations", str(iterations),
        "--checkpoint_iterations", str(iterations),
        "--test_iterations", str(iterations),
        "--disable_viewer",
        "--quiet",
        "--data_device", "cpu",
        "--antialiasing",
    ]

    if train_cfg.get("content_mask_loss", False):
        cmd.append("--content_mask_loss")

    if prev_model_path:
        # Find checkpoint from previous stage
        chkpnt = list(Path(prev_model_path).glob("chkpnt*.pth"))
        if chkpnt:
            cmd.extend(["--start_checkpoint", str(sorted(chkpnt)[-1])])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    rc = run(cmd, log_path=logs_dir / f"stage_{stage_idx:02d}.log", env=env)
    if rc != 0:
        raise RuntimeError(f"Stage {stage_idx} training failed")


def main():
    parser = argparse.ArgumentParser(description="Single-block training pipeline")
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output_root", default=str(PARTICLEGS_ROOT / "runs"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    output_dir = Path(args.output_root) / args.name
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    print(f"Single-block pipeline: {args.name}")
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    t_start = time.time()

    # Data preparation would go here (VTP + images + PLY)
    # For now, we assume data is pre-prepared and specified in config

    stages = cfg.get("stages", [])
    prev_model = None

    for i, stage in enumerate(stages):
        print(f"\n{'='*60}")
        print(f"STAGE {i+1}/{len(stages)}: {stage.get('name', f'stage_{i}')}")
        print(f"{'='*60}")

        stage_dir = output_dir / "stages" / f"{i:02d}_{stage.get('name', f'stage_{i}')}"
        model_dir = stage_dir / "model"
        data_dir = stage_dir / "data"

        # TODO: Generate stage-specific data if needed
        # For now assume data_dir exists or is specified in stage config

        run_training_stage(
            stage, model_dir, prev_model, data_dir, logs,
            stage_idx=i, gpu=args.gpu)
        prev_model = model_dir

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
