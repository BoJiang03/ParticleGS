#!/usr/bin/env python3
"""Fine-tune a merged block model.

Creates a checkpoint from a merged PLY, then runs particlegs training.
Settings loaded from config JSON (default: configs/training/F16_finetune.json).

Usage:
    python finetune.py \
        --merged_dir runs/my_exp/merged \
        --train_data runs/my_exp/shared/finetune/data \
        --output_dir runs/my_exp/finetuned \
        --gpu 0
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.Image import UnidentifiedImageError
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PARTICLEGS_ROOT = SCRIPT_DIR.parent

# Use the installed particlegs package's training module
PYTHON_BIN = sys.executable
TRAIN_CMD = [PYTHON_BIN, "-m", "particlegs.training.train"]
RENDER_CMD = [PYTHON_BIN, "-m", "particlegs.evaluation.render"]

DEFAULT_CONFIG = PARTICLEGS_ROOT / "configs" / "training" / "F16_finetune.json"

_BG_THRESH = 2.5


def load_config(config_path, overrides=None):
    """Load config JSON and apply key=value overrides."""
    with open(config_path) as f:
        cfg = json.load(f)
    cfg.pop("_comment", None)
    if overrides:
        for item in overrides:
            key, val = item.split("=", 1)
            if key in cfg:
                orig = cfg[key]
                if isinstance(orig, bool):
                    val = val.lower() in ("true", "1", "yes")
                elif isinstance(orig, int):
                    val = int(val)
                elif isinstance(orig, float):
                    val = float(val)
            else:
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
            cfg[key] = val
    return cfg


def compute_psnr_from_dirs(gt_dir, render_dir):
    """Compute full and masked PSNR between two directories of PNGs."""
    gt_dir, render_dir = Path(gt_dir), Path(render_dir)
    if not gt_dir.exists() or not render_dir.exists():
        return None, None, 0
    gt_files = sorted(p for p in gt_dir.iterdir() if p.suffix.lower() == ".png")
    if not gt_files:
        return None, None, 0

    def _one(gt_path):
        rp = render_dir / gt_path.name
        if not rp.exists():
            return None, None
        try:
            g = np.array(Image.open(gt_path)).astype(np.float64)
            r = np.array(Image.open(rp)).astype(np.float64)
        except (UnidentifiedImageError, OSError):
            return None, None
        if g.shape != r.shape:
            return None, None
        mse = np.mean((g - r) ** 2)
        psnr = 100.0 if mse == 0 else 20 * math.log10(255.0 / math.sqrt(mse))
        gb = g.mean(axis=2)
        rb = r.mean(axis=2)
        mask = (gb > _BG_THRESH) | (rb > _BG_THRESH)
        if mask.sum() == 0:
            return psnr, None
        m3 = np.stack([mask] * 3, axis=2)
        mse_m = np.mean((g[m3] - r[m3]) ** 2)
        mpsnr = 100.0 if mse_m == 0 else 20 * math.log10(255.0 / math.sqrt(mse_m))
        return psnr, mpsnr

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_one, gt_files))
    psnrs = [r[0] for r in results if r[0] is not None]
    mpsnrs = [r[1] for r in results if r[1] is not None]
    return (sum(psnrs) / len(psnrs) if psnrs else None,
            sum(mpsnrs) / len(mpsnrs) if mpsnrs else None,
            len(psnrs))


def create_checkpoint_from_ply(merged_dir, model_dir, src_iteration, cfg):
    """Create a training checkpoint from a merged PLY file."""
    sys.path.insert(0, str(PARTICLEGS_ROOT))
    from particlegs.model.gaussian_model import GaussianModel

    ply_path = Path(merged_dir) / "point_cloud" / f"iteration_{src_iteration}" / "point_cloud.ply"
    assert ply_path.exists(), f"PLY not found: {ply_path}"

    print(f"Loading merged PLY: {ply_path}")
    g = GaussianModel(0)
    g.load_ply(str(ply_path))

    n = g._xyz.shape[0]
    print(f"  {n} Gaussians loaded")

    _adam_defaults = dict(
        betas=(0.9, 0.999), eps=1e-15, weight_decay=0, amsgrad=False,
        maximize=False, foreach=None, capturable=False, differentiable=False,
        fused=None, decoupled_weight_decay=False,
    )
    # NOTE: optimizer.load_state_dict() OVERWRITES param_groups from the
    # checkpoint, so command-line LR values are ignored. The LR here is
    # the effective LR used during fine-tuning. Reference uses scaling=0.005
    # (same as per-block training) — higher scaling LR is critical for fine-tune.
    optimizer_sd = {
        'state': {},
        'param_groups': [
            {'params': [i], 'name': name, 'lr': lr, **_adam_defaults}
            for i, (name, lr) in enumerate([
                ('xyz', 0.0),
                ('f_dc', cfg.get('feature_lr', 0.0025)),
                ('f_rest', cfg.get('feature_lr', 0.0025) / 20.0),
                ('opacity', cfg.get('opacity_lr', 0.025)),
                ('scaling', 0.005),   # higher works better for fine-tune (ref)
                ('rotation', cfg.get('rotation_lr', 0.001)),
            ])
        ],
    }

    state = (
        0,
        g._xyz.cuda(),
        g._features_dc.cuda(),
        g._features_rest.cuda(),
        g._scaling.cuda(),
        g._rotation.cuda(),
        g._opacity.cuda(),
        torch.zeros(n, device='cuda'),
        torch.zeros(n, 1, device='cuda'),
        torch.zeros(n, 1, device='cuda'),
        optimizer_sd,
        1.0,  # spatial_lr_scale
    )

    os.makedirs(model_dir, exist_ok=True)
    chkpnt_path = Path(model_dir) / "chkpnt0.pth"
    torch.save((tuple(state), 0), str(chkpnt_path))
    print(f"  Saved checkpoint: {chkpnt_path} (iteration 0)")

    # Copy VizMapper (rename to iteration 0)
    vm_src = Path(merged_dir) / f"viz_mapper_{src_iteration}.pth"
    vm_dst = Path(model_dir) / "viz_mapper_0.pth"
    if vm_src.exists():
        shutil.copy2(vm_src, vm_dst)
        print(f"  Copied VizMapper: {vm_dst}")

    norm_src = Path(merged_dir) / "normalization.json"
    if norm_src.exists():
        shutil.copy2(norm_src, Path(model_dir) / "normalization.json")

    return str(chkpnt_path), n


def run_finetune(model_dir, train_data, chkpnt_path, n_gaussians, cfg, gpu):
    """Run training with config settings."""
    iterations = cfg['iterations']
    cmd = TRAIN_CMD + [
        "--source_path", str(train_data),
        "--model_path", str(model_dir),
        "--start_checkpoint", str(chkpnt_path),
        "--iterations", str(iterations),
        "--sh_degree", str(cfg['sh_degree']),
        "--resolution", "1",
        "--position_lr_init", str(cfg['position_lr_init']),
        "--position_lr_final", str(cfg['position_lr_final']),
        "--position_lr_max_steps", str(iterations),
        "--scaling_lr", str(cfg['scaling_lr']),
        "--feature_lr", str(cfg['feature_lr']),
        "--opacity_lr", str(cfg['opacity_lr']),
        "--rotation_lr", str(cfg['rotation_lr']),
        "--lambda_dssim", str(cfg['lambda_dssim']),
        "--lambda_identity", str(cfg['lambda_identity']),
        "--densification_interval", str(cfg['densification_interval']),
        "--densify_grad_threshold", str(cfg['densify_grad_threshold']),
        "--densify_from_iter", str(cfg['densify_from_iter']),
        "--densify_until_iter", str(cfg['densify_until_iter']),
        "--percent_dense", str(cfg['percent_dense']),
        "--opacity_reset_interval", str(cfg['opacity_reset_interval']),
        "--min_opacity", str(cfg['min_opacity']),
        "--factor_delta_opacity", str(cfg['factor_delta_opacity']),
        "--factor_delta_scale", str(cfg['factor_delta_scale']),
        "--min_opacity_clamp", str(cfg['min_opacity_clamp']),
        "--save_iterations", str(iterations),
        "--checkpoint_iterations", str(iterations),
        "--test_iterations", str(iterations),
        "--disable_viewer",
        "--quiet",
        "--data_device", "cpu",
        "--antialiasing",
    ]
    if cfg.get('content_mask_loss', True):
        cmd.append("--content_mask_loss")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"\nRunning fine-tune: {iterations} iter, GPU {gpu}")
    print(f"  Start: {n_gaussians} Gaussians")

    log_path = Path(model_dir).parent / "train.log"
    with open(log_path, "w") as logf:
        result = subprocess.run(cmd, cwd=str(PARTICLEGS_ROOT), env=env,
                                stdout=logf, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        print(f"ERROR: Training failed (exit {result.returncode})")
        with open(log_path) as f:
            for line in f.readlines()[-20:]:
                print(f"  {line.rstrip()}")
        sys.exit(1)

    final_ply = Path(model_dir) / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    n_final = None
    ply_size_mb = 0
    if final_ply.exists():
        ply_size_mb = final_ply.stat().st_size / (1024 * 1024)
        with open(final_ply, 'rb') as f:
            for line in f:
                line = line.decode('ascii', errors='ignore').strip()
                if line.startswith('element vertex'):
                    n_final = int(line.split()[-1])
                    reduction = (1 - n_final / n_gaussians) * 100
                    print(f"\nFine-tune complete: {n_gaussians} -> {n_final} "
                          f"({reduction:+.1f}%), {ply_size_mb:.1f} MB")
                    break
                if line == 'end_header':
                    break

    return iterations, n_final, ply_size_mb


def run_eval(model_dir, eval_ref, iteration, cfg, gpu):
    """Run evaluation on eval_far/mid/near."""
    model_dir = Path(model_dir).resolve()
    eval_dir = Path(eval_ref).resolve() / "evaluation"
    if not eval_dir.exists():
        print(f"Eval directory not found: {eval_dir}")
        return {}

    datasets = [
        ("eval_far", "01_eval_far"),
        ("eval_mid", "02_eval_mid"),
        ("eval_near", "03_eval_near"),
    ]

    results = {}
    for eval_name, eval_subdir in datasets:
        source_dir = eval_dir / eval_subdir / "data"
        if not source_dir.exists():
            continue

        render_out = model_dir / "eval" / eval_name
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        cmd = RENDER_CMD + [
            "-m", str(model_dir),
            "-s", str(source_dir),
            "--iteration", str(iteration),
            "--skip_test",
            "--min_scale_pixels", "0.0",
            "--factor_delta_opacity", str(cfg['factor_delta_opacity']),
            "--factor_delta_scale", str(cfg['factor_delta_scale']),
            "--min_opacity_clamp", str(cfg['min_opacity_clamp']),
            "--output_dir", str(render_out),
            "--antialiasing",
        ]

        log_path = Path(model_dir).parent / f"eval_{eval_name}.log"
        with open(log_path, "w") as logf:
            rc = subprocess.run(cmd, cwd=str(PARTICLEGS_ROOT), env=env,
                                stdout=logf, stderr=subprocess.STDOUT)
        if rc.returncode != 0:
            print(f"  {eval_name}: RENDER FAILED")
            results[eval_name] = {"psnr": None, "masked_psnr": None}
            continue

        gt_dir = render_out / "train" / f"ours_{iteration}" / "gt"
        render_dir_path = render_out / "train" / f"ours_{iteration}" / "renders"
        psnr, mpsnr, n = compute_psnr_from_dirs(gt_dir, render_dir_path)
        results[eval_name] = {"psnr": psnr, "masked_psnr": mpsnr, "n_images": n}
        if psnr is not None:
            print(f"  {eval_name}: PSNR={psnr:.2f}  masked={mpsnr:.2f}  ({n} imgs)")

    psnr_vals = [r["psnr"] for r in results.values() if r.get("psnr")]
    mpsnr_vals = [r["masked_psnr"] for r in results.values() if r.get("masked_psnr")]
    if psnr_vals:
        results["avg_psnr"] = sum(psnr_vals) / len(psnr_vals)
    if mpsnr_vals:
        results["avg_masked_psnr"] = sum(mpsnr_vals) / len(mpsnr_vals)

    return results


def main():
    parser = argparse.ArgumentParser(description="Fine-tune merged block model")
    parser.add_argument("--merged_dir", required=True)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_ref", default=None,
                        help="Path to shared_reference for eval")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--override", nargs="*", metavar="KEY=VAL")
    parser.add_argument("--src_iteration", type=int, default=32000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    print(f"Config: {args.config}")

    merged_dir = Path(args.merged_dir).resolve()
    train_data = Path(args.train_data).resolve()
    output_dir = Path(args.output_dir).resolve()
    model_dir = output_dir / "model"

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "finetune_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Step 1: Create checkpoint
    print("=" * 60)
    print("STEP 1: Creating checkpoint from merged PLY")
    print("=" * 60)
    chkpnt_path, n_gaussians = create_checkpoint_from_ply(
        merged_dir, model_dir, args.src_iteration, cfg)

    # Step 2: Fine-tune
    print("\n" + "=" * 60)
    print("STEP 2: Fine-tuning")
    print("=" * 60)
    final_iter, n_final, ply_size_mb = run_finetune(
        model_dir, train_data, chkpnt_path, n_gaussians, cfg, args.gpu)

    # Step 3: Eval
    results = {}
    if not args.skip_eval and args.eval_ref:
        print("\n" + "=" * 60)
        print("STEP 3: Evaluating fine-tuned model")
        print("=" * 60)
        results = run_eval(model_dir, args.eval_ref, final_iter, cfg, args.gpu)
        results["n_start"] = n_gaussians
        results["n_final"] = n_final
        results["ply_size_mb"] = round(ply_size_mb, 1)
        with open(output_dir / "eval_results.json", "w") as f:
            json.dump(results, f, indent=2)
        avg_m = results.get("avg_masked_psnr", 0)
        print(f"\n  Avg masked PSNR={avg_m:.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
