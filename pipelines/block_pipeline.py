#!/usr/bin/env python3
"""Block Training Pipeline — Partition, train blocks, merge, finetune, evaluate.

Pipeline steps:
  1. Partition raw data into N blocks (KD-tree)
  2. Generate per-block configs (3-stage progressive)
  3. Train N blocks in parallel (multi-GPU)
  4. Merge block models into unified coordinate system
  5. Evaluate merged model
  6. Fine-tune merged model (F16 recipe)
  7. Evaluate fine-tuned model

Usage:
    python block_pipeline.py --num_blocks 8 --name my_exp_8block
    python block_pipeline.py --num_blocks 8 --name my_exp_8block --resume
"""

import argparse
import concurrent.futures
import copy
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
from PIL.Image import UnidentifiedImageError

SCRIPT_DIR = Path(__file__).resolve().parent
PARTICLEGS_ROOT = SCRIPT_DIR.parent

PYTHON_BIN = sys.executable
PVBATCH_BIN = shutil.which("pvbatch") or "pvbatch"

# Internal scripts
PARTITION_SCRIPT = PARTICLEGS_ROOT / "data" / "partition.py"
MERGE_SCRIPT = SCRIPT_DIR / "merge_blocks.py"
FINETUNE_SCRIPT = SCRIPT_DIR / "finetune.py"
PREPARE_SCRIPT = PARTICLEGS_ROOT / "data" / "prepare_data.py"
GENERATE_SCRIPT = PARTICLEGS_ROOT / "data" / "generate_images.py"

# Training/rendering via installed package
TRAIN_CMD = [PYTHON_BIN, "-m", "particlegs.training.train"]
RENDER_CMD = [PYTHON_BIN, "-m", "particlegs.evaluation.render"]

RAW_DIR = PARTICLEGS_ROOT / "data" / "hacc_raw"

NUM_GPUS = 2
FINAL_ITERATION = 32000

# Viz render params for eval/finetune data generation
VIZ_RENDER = dict(
    radius="0.01", opacity="0.05",
    radius_min="0.0025", radius_max="0.0175",
    opacity_min="0.0125", opacity_max="0.0875",
    seed="142", distribution="beta", concentration="3.0",
)

_BG_THRESH = 2.5


# ── PSNR computation ─────────────────────────────────────────────────

def compute_psnr_from_dirs(gt_dir, render_dir):
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

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_one, gt_files))
    psnrs = [r[0] for r in results if r[0] is not None]
    mpsnrs = [r[1] for r in results if r[1] is not None]
    return (sum(psnrs) / len(psnrs) if psnrs else None,
            sum(mpsnrs) / len(mpsnrs) if mpsnrs else None,
            len(psnrs))


# ── Helpers ───────────────────────────────────────────────────────────

def run(cmd, log_path=None, env=None):
    cmd_str = [str(c) for c in cmd]
    short = " ".join(cmd_str[-6:])
    print(f"  $ {short}")
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            result = subprocess.run(cmd_str, stdout=f, stderr=subprocess.STDOUT,
                                    cwd=str(PARTICLEGS_ROOT), env=env)
    else:
        result = subprocess.run(cmd_str, cwd=str(PARTICLEGS_ROOT), env=env)
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        if log_path and Path(log_path).exists():
            for line in Path(log_path).read_text().splitlines()[-20:]:
                print(f"    {line}")
    return result.returncode


def run_parallel(cmd_pairs, max_parallel=2):
    results = {}
    def _run_one(item):
        cmd, log_path, gpu_id = item
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        return run(cmd, log_path, env=env)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {pool.submit(_run_one, item): i for i, item in enumerate(cmd_pairs)}
        for future in concurrent.futures.as_completed(futures):
            results[futures[future]] = future.result()
    return results


def pvbatch_prefix(gpu=0):
    """pvbatch with EGL GPU pinning."""
    return [PVBATCH_BIN, "--force-offscreen-rendering",
            "--opengl-window-backend", "EGL", "--"]


def header(n, title):
    print(f"\n{'='*60}\nSTEP {n}: {title}\n{'='*60}")


def generate_images_cmd(vtp, out_dir, norm, orbit_radii, num_frames,
                        width=1920, height=1080, train_ratio="1.0"):
    v = VIZ_RENDER
    return pvbatch_prefix() + [
        GENERATE_SCRIPT,
        "--vtp_path", vtp, "--output_dir", out_dir,
        "--gaussian_radius", v["radius"], "--opacity", v["opacity"],
        "--width", str(width), "--height", str(height),
        "--num_frames", str(num_frames),
        "--train_ratio", train_ratio,
        "--split_seed", "42",
        "--camera_strategy", "multi_orbit",
        "--orbit_radii", orbit_radii,
        "--viz_mode", "sampled",
        "--radius_min", v["radius_min"], "--radius_max", v["radius_max"],
        "--opacity_min", v["opacity_min"], "--opacity_max", v["opacity_max"],
        "--viz_seed", v["seed"],
        "--viz_distribution", v["distribution"],
        "--viz_beta_concentration", v["concentration"],
        "--normalization_path", norm,
    ]


# ── Pipeline steps ────────────────────────────────────────────────────

def step_ensure_shared_data(shared, logs):
    """Create VTP + normalization + PLY from full raw data."""
    shared.mkdir(parents=True, exist_ok=True)
    needed = ["particles.vtp", "normalization.json", "points3d.ply"]
    if all((shared / f).exists() for f in needed):
        print("[SKIP] Shared data exists")
        return
    header("0a", "Preparing shared data (VTP + normalization + PLY)")
    rc = run(pvbatch_prefix() + [
        PREPARE_SCRIPT,
        "--raw_x", RAW_DIR / "xx.f32",
        "--raw_y", RAW_DIR / "yy.f32",
        "--raw_z", RAW_DIR / "zz.f32",
        "--output_dir", shared,
        "--num_points_raw", "0",
        "--num_points_ply", "200000",
        "--skip_images",
    ], log_path=logs / "00a_prepare.log")
    if rc != 0:
        raise RuntimeError("Shared data preparation failed")


def step_ensure_eval_datasets(shared, logs):
    """Generate eval GT images (80 frames x 3 orbits)."""
    eval_dir = shared / "evaluation"
    vtp = shared / "particles.vtp"
    norm = shared / "normalization.json"
    eval_sets = [
        ("eval_far", "01_eval_far", "1.0"),
        ("eval_mid", "02_eval_mid", "0.7"),
        ("eval_near", "03_eval_near", "0.5"),
    ]
    if all((eval_dir / sub / "data" / "transforms_train.json").exists()
           for _, sub, _ in eval_sets):
        print("[SKIP] Eval datasets exist")
        return eval_dir
    header("0b", "Generating eval datasets")
    for name, sub, radius in eval_sets:
        d = eval_dir / sub / "data"
        if (d / "transforms_train.json").exists():
            continue
        print(f"  {name} (orbit_radius={radius})...")
        rc = run(
            generate_images_cmd(vtp, d, norm, radius, num_frames=80),
            log_path=logs / f"00b_{name}.log")
        if rc != 0:
            raise RuntimeError(f"Eval dataset {name} generation failed")
    return eval_dir


def step_ensure_finetune_data(shared, logs):
    """Generate finetune training data (600 frames, 3 orbits)."""
    ft = shared / "finetune" / "data"
    vtp = shared / "particles.vtp"
    norm = shared / "normalization.json"
    if (ft / "transforms_train.json").exists():
        print("[SKIP] Finetune data exists")
        return ft
    header("6a", "Generating finetune training data")
    rc = run(
        generate_images_cmd(vtp, ft, norm, "1.0,0.7,0.5", num_frames=600),
        log_path=logs / "06a_finetune_data.log")
    if rc != 0:
        raise RuntimeError("Finetune data generation failed")
    return ft


def step_partition(output_dir, num_blocks, resume=False):
    partition_dir = output_dir / "partitions"
    info_path = partition_dir / "partition_info.json"
    if resume and info_path.exists():
        with open(info_path) as f:
            if json.load(f)["num_blocks"] == num_blocks:
                print(f"[SKIP] Partitions exist ({num_blocks} blocks)")
                return partition_dir
    header(1, f"Partitioning into {num_blocks} blocks")
    partition_dir.mkdir(parents=True, exist_ok=True)
    rc = run([
        PYTHON_BIN, PARTITION_SCRIPT,
        "--raw_x", RAW_DIR / "xx.f32",
        "--raw_y", RAW_DIR / "yy.f32",
        "--raw_z", RAW_DIR / "zz.f32",
        "--num_blocks", str(num_blocks),
        "--output_dir", partition_dir,
    ], log_path=output_dir / "logs" / "01_partition.log")
    if rc != 0:
        raise RuntimeError("Partition failed")
    return partition_dir


def step_create_configs(output_dir, partition_dir, num_blocks, name, base_config_path):
    header(2, f"Creating {num_blocks} block configs")
    with open(base_config_path) as f:
        base = json.load(f)
    configs_dir = output_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(num_blocks):
        cfg = copy.deepcopy(base)
        block_raw = partition_dir / f"block_{i}"
        cfg["experiment"]["name"] = f"{name}_block_{i}"
        cfg["experiment"]["output_root"] = str(output_dir / "blocks")
        cfg["paths"]["raw_x"] = str(block_raw / "xx.f32")
        cfg["paths"]["raw_y"] = str(block_raw / "yy.f32")
        cfg["paths"]["raw_z"] = str(block_raw / "zz.f32")
        cfg["options"] = {
            "skip_intermediate_eval": True,
            "skip_final_render": True,
        }
        cfg.pop("evaluation", None)
        p = configs_dir / f"block_{i}.json"
        with open(p, "w") as f:
            json.dump(cfg, f, indent=2)
        paths.append(p)
        print(f"  block_{i}: {block_raw}")
    return paths


def step_train_blocks(output_dir, config_paths, num_blocks, name, resume=False):
    header(3, f"Training {num_blocks} blocks ({NUM_GPUS} GPUs)")
    # Note: block training uses run_experiment.py from the old pipeline.
    # For the self-contained package, we call the training directly.
    # This requires the experiment runner or a simplified training loop.
    # For now, we call training via subprocess with the particlegs package.
    for batch_start in range(0, num_blocks, NUM_GPUS):
        batch = list(range(batch_start, min(batch_start + NUM_GPUS, num_blocks)))
        print(f"\n--- Batch: blocks {batch} ---")
        cmd_pairs = []
        for j, block_id in enumerate(batch):
            gpu_id = j % NUM_GPUS
            block_dir = output_dir / "blocks" / f"{name}_block_{block_id}"
            # Check if already complete
            if resume and (block_dir / "stages").exists():
                stages = sorted((block_dir / "stages").iterdir())
                if len(stages) >= 3:
                    print(f"  block_{block_id}: SKIP (complete)")
                    continue

            # Each block config is a multi-stage JSON — the experiment runner handles it.
            # For the simplified self-contained version, we train directly.
            cmd = [
                PYTHON_BIN, "-m", "particlegs.training.train",
                "--source_path", str(block_dir / "data"),
                "--model_path", str(block_dir / "model"),
                "--sh_degree", "0",
                "--antialiasing",
                "--disable_viewer",
                "--quiet",
            ]
            log_path = output_dir / "logs" / f"03_train_block_{block_id}.log"
            cmd_pairs.append((cmd, log_path, gpu_id))
            print(f"  block_{block_id} -> GPU {gpu_id}")

        if not cmd_pairs:
            continue
        t0 = time.time()
        results = run_parallel(cmd_pairs, max_parallel=NUM_GPUS)
        print(f"  Batch time: {time.time()-t0:.0f}s")
        for j, rc in sorted(results.items()):
            if rc != 0:
                raise RuntimeError(f"Block training failed in batch {batch}")


def step_merge(output_dir, target_norm, resume=False):
    merged_dir = output_dir / "merged"
    if resume and list((merged_dir / "point_cloud").glob("iteration_*") if (merged_dir / "point_cloud").exists() else []):
        print("[SKIP] Merged model exists")
        return merged_dir
    header(4, "Merging block models")
    rc = run([
        PYTHON_BIN, MERGE_SCRIPT,
        "--block_dir", output_dir / "blocks",
        "--target_norm", target_norm,
        "--output_dir", merged_dir,
        "--iteration", str(FINAL_ITERATION),
    ], log_path=output_dir / "logs" / "04_merge.log")
    if rc != 0:
        raise RuntimeError("Merge failed")
    return merged_dir


def step_eval(output_dir, model_dir, iteration, shared, label="merged"):
    eval_dir = shared / "evaluation"
    eval_sets = [
        ("eval_far", "01_eval_far"),
        ("eval_mid", "02_eval_mid"),
        ("eval_near", "03_eval_near"),
    ]
    results = {}
    for eval_name, eval_subdir in eval_sets:
        src = eval_dir / eval_subdir / "data"
        if not src.exists():
            continue
        render_out = Path(model_dir) / "eval" / eval_name
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        rc = run(RENDER_CMD + [
            "-m", str(model_dir), "-s", str(src),
            "--iteration", str(iteration),
            "--skip_test",
            "--min_scale_pixels", "0.0",
            "--output_dir", str(render_out),
            "--antialiasing",
        ], log_path=output_dir / "logs" / f"eval_{label}_{eval_name}.log", env=env)
        if rc != 0:
            results[eval_name] = {"psnr": None, "masked_psnr": None}
            continue
        gt = render_out / "train" / f"ours_{iteration}" / "gt"
        rd = render_out / "train" / f"ours_{iteration}" / "renders"
        p, m, n = compute_psnr_from_dirs(gt, rd)
        results[eval_name] = {"psnr": p, "masked_psnr": m, "n": n}
        if p:
            print(f"  {eval_name}: PSNR={p:.2f}  masked={m:.2f}")

    ps = [r["psnr"] for r in results.values() if r.get("psnr")]
    ms = [r["masked_psnr"] for r in results.values() if r.get("masked_psnr")]
    if ps:
        results["avg_psnr"] = sum(ps) / len(ps)
        results["avg_masked_psnr"] = sum(ms) / len(ms) if ms else 0
    return results


def main():
    parser = argparse.ArgumentParser(description="Block Training Pipeline")
    parser.add_argument("--num_blocks", type=int, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--output_root", type=str,
                        default=str(PARTICLEGS_ROOT / "runs"))
    parser.add_argument("--base_config", type=str,
                        default=str(PARTICLEGS_ROOT / "configs" / "training" / "C55_per_block.json"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_finetune", action="store_true")
    args = parser.parse_args()

    nb = args.num_blocks
    if nb & (nb - 1) != 0 or nb < 2:
        parser.error("--num_blocks must be a power of 2 (2, 4, 8, ...)")

    output_dir = Path(args.output_root) / args.name
    shared = output_dir / "shared"
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    print(f"Block Pipeline: {args.name} ({nb} blocks)")
    t_start = time.time()

    # Step 0: Shared data
    step_ensure_shared_data(shared, logs)
    step_ensure_eval_datasets(shared, logs)

    # Step 1-3: Partition, config, train
    part = step_partition(output_dir, nb, args.resume)
    cfgs = step_create_configs(output_dir, part, nb, args.name, args.base_config)
    step_train_blocks(output_dir, cfgs, nb, args.name, args.resume)

    # Step 4: Merge
    target_norm = shared / "normalization.json"
    merged = step_merge(output_dir, target_norm, args.resume)

    # Step 5: Eval merged
    header(5, "Evaluating merged model")
    merged_res = step_eval(output_dir, merged, FINAL_ITERATION, shared, "merged")
    with open(merged / "eval_results.json", "w") as f:
        json.dump(merged_res, f, indent=2)

    if not args.skip_finetune:
        # Step 6: Finetune
        ft_data = step_ensure_finetune_data(shared, logs)
        header(6, "Fine-tuning merged model (F16)")
        ft_dir = output_dir / "finetuned"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        rc = run([
            PYTHON_BIN, FINETUNE_SCRIPT,
            "--merged_dir", merged,
            "--train_data", ft_data,
            "--output_dir", ft_dir,
            "--src_iteration", str(FINAL_ITERATION),
            "--eval_ref", shared,
            "--gpu", "0",
        ], log_path=logs / "06_finetune.log", env=env)
        if rc != 0:
            raise RuntimeError("Finetune failed")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")


if __name__ == "__main__":
    main()
