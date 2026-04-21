#!/usr/bin/env python3
"""EXP-4: Block Training Scan.

Demonstrates spatial block partitioning improves quality by reducing per-block
particle density.

Pipeline: KD-tree partition -> per-block 3-stage training -> merge -> finetune (v5b: 20k iter)

Sub-experiments:
  4a: Block count vs quality (1, 2, 4, 8, 16 blocks)
  4b: Per-distance breakdown (far/mid/near)

Usage:
    python -m experiments.exp4_block_training [--gpu 0] [--num_gpus 2]
"""

import json
import os
import shutil
import time
from pathlib import Path

from experiments.common import *

# Per-block training — uses E25 config (best single-block, validated at 26.24 dB).
# Same config for single-block and per-block ensures fair comparison.
PER_BLOCK_TRAIN = {
    "iterations": 12000, "sh_degree": 0, "resolution_scale": 2,
    "densification_interval": 100, "opacity_reset_interval": 3000,
    "densify_until_iter": 12000, "densify_grad_threshold": 0.0002,
    "position_lr_init": 0.00016, "position_lr_max_steps": 50000,
    "scaling_lr": 0.005, "min_scale_pixels": 1.0, "min_opacity": 0.01,
    "antialiasing": True, "content_mask_loss": True,
    "lambda_dssim": 0.0, "factor_delta_opacity": 0.3,
    "factor_delta_scale": 0.1, "min_opacity_clamp": 0.4,
}

PER_BLOCK_STAGES = [
    {"name": "S1_4k", "res_scale": 2, "width": 3840, "height": 2160,
     "iterations": 12000, "densify_until": 12000, "densify_grad": 0.0002,
     "densify_interval": 100},
    {"name": "S2_6k", "res_scale": 1, "width": 5760, "height": 3240,
     "iterations": 15000, "densify_until": 24000, "densify_grad": 0.0003,
     "densify_interval": 200},
    {"name": "S3_mix", "res_scale": 1, "width": 5760, "height": 3240,
     "iterations": 12000, "densify_until": 36000, "densify_grad": 0.0003,
     "densify_interval": 200, "mix": True},
]

# Finetune config — F16 recipe (validated: +2.37 dB on C55 2-block merged).
# Key: wider VizMapper (0.8/0.3), lower clamp (0.1), full LR, 60k iterations.
# NOTE: optimizer.load_state_dict overwrites LR from checkpoint, so the
# effective LR is set in create_checkpoint_from_ply, not here.
F16_FINETUNE = {
    "iterations": 60000, "sh_degree": 0,
    "densification_interval": 100, "opacity_reset_interval": 99999,
    "densify_from_iter": 1, "densify_until_iter": 30000,
    "densify_grad_threshold": 0.0002, "percent_dense": 0.01,
    "position_lr_init": 0.00016, "position_lr_max_steps": 60000,
    "position_lr_final": 0.0000016,
    "scaling_lr": 0.005, "feature_lr": 0.0025, "opacity_lr": 0.025,
    "rotation_lr": 0.001,
    "min_scale_pixels": 0.0, "min_opacity": 0.0,
    "antialiasing": True, "content_mask_loss": True,
    "lambda_dssim": 0.0, "lambda_identity": 0.0,
    "factor_delta_opacity": 0.8, "factor_delta_scale": 0.3,
    "min_opacity_clamp": 0.1,
}

CAMERA_4K = {"strategy": "multi_orbit", "orbit_radii": "1.0,0.7,0.5",
             "num_frames": 400, "width": 3840, "height": 2160}


def partition_raw_data(n_blocks, output_dir, logs_dir):
    """Partition raw data into n_blocks using KD-tree."""
    partition_dir = output_dir / f"partition_{n_blocks}"
    meta_file = partition_dir / "partition_info.json"
    if meta_file.exists():
        return json.loads(meta_file.read_text()), partition_dir

    run_cmd([PYTHON_BIN, str(PARTITION_SCRIPT),
             "--raw_x", str(RAW_X), "--raw_y", str(RAW_Y), "--raw_z", str(RAW_Z),
             "--num_blocks", str(n_blocks),
             "--output_dir", str(partition_dir)],
            log_path=logs_dir / f"partition_{n_blocks}.log")

    return json.loads(meta_file.read_text()), partition_dir


def prepare_block_data(block_id, block_raw_dir, block_run_dir, shared_data, logs_dir):
    """Create VTP + normalization + PLY for one block."""
    block_data = block_run_dir / "shared"
    vtp_path = block_data / "particles.vtp"
    norm_path = block_data / "normalization.json"
    ply_path = block_data / "points3d.ply"

    if vtp_path.exists() and norm_path.exists() and ply_path.exists():
        return {"vtp": vtp_path, "normalization": norm_path, "ply": ply_path}

    block_data.mkdir(parents=True, exist_ok=True)

    # Create VTP
    if not vtp_path.exists():
        run_cmd(
            pvbatch_cmd(PREPARE_SCRIPT,
                        "--raw_x", block_raw_dir / "xx.f32",
                        "--raw_y", block_raw_dir / "yy.f32",
                        "--raw_z", block_raw_dir / "zz.f32",
                        "--output_dir", block_data, "--num_points_raw", "0",
                        "--skip_images"),
            log_path=logs_dir / f"block{block_id}_vtp.log")

    # Generate normalization via quick render
    if not norm_path.exists():
        tmp_data = block_data / "_norm_render"
        gen_args = [
            "--vtp_path", str(vtp_path),
            "--output_dir", str(tmp_data),
            "--camera_strategy", "multi_orbit", "--orbit_radii", "1.0",
            "--num_frames", "1", "--width", "256", "--height", "256",
            "--train_ratio", "1.0",
        ]
        for k, v in DEFAULT_VIZ_PARAMS.items():
            gen_args += [f"--{k}", v]
        run_cmd(pvbatch_cmd(GENERATE_SCRIPT, *gen_args),
                log_path=logs_dir / f"block{block_id}_norm.log")
        shutil.copy2(tmp_data / "normalization.json", norm_path)
        shutil.rmtree(tmp_data)

    # Create PLY
    if not ply_path.exists():
        run_cmd(
            [PYTHON_BIN, str(PREPARE_SCRIPT),
             "--raw_x", str(block_raw_dir / "xx.f32"),
             "--raw_y", str(block_raw_dir / "yy.f32"),
             "--raw_z", str(block_raw_dir / "zz.f32"),
             "--output_dir", str(block_data),
             "--only_ply", "--num_points_ply", str(NUM_INIT_PLY)],
            log_path=logs_dir / f"block{block_id}_ply.log")

    return {"vtp": vtp_path, "normalization": norm_path, "ply": ply_path}


def train_block(block_id, block_data, block_run_dir, logs_dir, gpu):
    """Train one block through 3 progressive stages."""
    prev_checkpoint = None
    final_model = None
    final_iter = None

    for si, stage in enumerate(PER_BLOCK_STAGES):
        stage_name = f"block{block_id}_{stage['name']}"
        stage_dir = block_run_dir / f"{si:02d}_{stage['name']}"
        data_dir = stage_dir / "data"
        model_dir = stage_dir / "model"

        existing = find_checkpoint(model_dir)
        if existing:
            prev_checkpoint = existing
            final_model = model_dir
            final_iter = int(existing.stem.replace("chkpnt", ""))
            continue

        # Generate data
        n_frames = stage.get("num_frames", 400)
        if stage.get("mix"):
            generate_mix_training_data(
                block_data["vtp"], data_dir, block_data["normalization"],
                "1.0,0.7,0.5", n_frames, "142", n_frames, "242", 0.98,
                stage["width"], stage["height"], 0.8,
                logs_dir, stage_name)
        else:
            generate_training_data(
                block_data["vtp"], data_dir, block_data["normalization"],
                "multi_orbit", "1.0,0.7,0.5", n_frames,
                stage["width"], stage["height"], logs_dir, stage_name)

        prepare_stage_data_dir(data_dir, block_data["ply"], block_data["normalization"])

        train_cfg = dict(PER_BLOCK_TRAIN,
                         iterations=stage["iterations"],
                         resolution_scale=stage["res_scale"],
                         densify_until_iter=stage["densify_until"],
                         densify_grad_threshold=stage["densify_grad"],
                         densification_interval=stage["densify_interval"])

        final_iter = run_stage_training(data_dir, model_dir, train_cfg, logs_dir,
                                       f"train_{stage_name}",
                                       start_checkpoint=prev_checkpoint,
                                       init_iterations=10, gpu=gpu)
        prev_checkpoint = find_checkpoint(model_dir, final_iter)
        final_model = model_dir

        # Cleanup
        img_dir = data_dir / "images"
        if img_dir.exists():
            shutil.rmtree(img_dir)

    return final_model, final_iter


def merge_blocks(block_models, block_run_dirs, target_norm_path, output_dir, logs_dir):
    """Merge block PLYs into single model.

    Calls merge logic directly (not CLI) to avoid directory naming issues.
    block_models: list of (model_dir, iteration) tuples.
    block_run_dirs: list of block run directories (each containing shared/).
    """
    import sys as _sys
    _sys.path.insert(0, str(PARTICLEGS_ROOT))
    from pipelines.merge_blocks import load_normalization, load_ply_raw
    import math as _math

    merged_dir = output_dir / "merged"
    merged_ply_check = None
    # Find if any merged PLY already exists
    pc_dir = merged_dir / "point_cloud"
    if pc_dir.exists():
        for d in sorted(pc_dir.glob("iteration_*")):
            p = d / "point_cloud.ply"
            if p.exists():
                merged_ply_check = p
                break
    if merged_ply_check:
        print(f"  [Skip] Merged PLY exists: {merged_ply_check}")
        return merged_dir

    import numpy as _np
    from plyfile import PlyData as _PD, PlyElement as _PE

    target_center, target_sf = load_normalization(str(target_norm_path))
    print(f"  Target norm: center={target_center.tolist()}, scale={target_sf:.8f}")

    all_vertices = []
    total = 0
    actual_iter = None

    for i, ((model_dir, iteration), block_dir) in enumerate(zip(block_models, block_run_dirs)):
        # Load block normalization
        block_norm = block_dir / "shared" / "normalization.json"
        block_center, block_sf = load_normalization(str(block_norm))

        # Find PLY
        ply_path = Path(model_dir) / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        if not ply_path.exists():
            raise FileNotFoundError(f"Block {i} PLY not found: {ply_path}")
        if actual_iter is None:
            actual_iter = iteration

        vertex = load_ply_raw(str(ply_path))
        n = len(vertex.data)
        total += n
        print(f"  Block {i}: {n} Gaussians, norm={block_center.tolist()}")

        xyz = _np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(_np.float64)
        scale_ratio = target_sf / block_sf
        offset = (block_center - target_center) * target_sf
        xyz_target = xyz * scale_ratio + offset
        log_corr = _math.log(scale_ratio)

        new_data = vertex.data.copy()
        new_data["x"] = xyz_target[:, 0].astype(_np.float32)
        new_data["y"] = xyz_target[:, 1].astype(_np.float32)
        new_data["z"] = xyz_target[:, 2].astype(_np.float32)
        for sn in [p.name for p in vertex.properties if p.name.startswith("scale_")]:
            new_data[sn] = (new_data[sn].astype(_np.float64) + log_corr).astype(_np.float32)
        all_vertices.append(new_data)

    merged = _np.concatenate(all_vertices)
    print(f"  Total merged: {total} Gaussians")

    out_pc = merged_dir / "point_cloud" / f"iteration_{actual_iter}"
    out_pc.mkdir(parents=True, exist_ok=True)
    _PE.describe(merged, "vertex")
    _PD([_PE.describe(merged, "vertex")]).write(str(out_pc / "point_cloud.ply"))

    # Copy VizMapper from block 0
    block0_model = block_models[0][0]
    block0_iter = block_models[0][1]
    vm_src = Path(block0_model) / f"viz_mapper_{block0_iter}.pth"
    if vm_src.exists():
        shutil.copy2(vm_src, merged_dir / f"viz_mapper_{actual_iter}.pth")

    # Write cfg_args
    (merged_dir / "cfg_args").write_text(
        f"Namespace(sh_degree=0, source_path='', model_path='{merged_dir}', "
        f"images='images', depths='', resolution=1, white_background=False, "
        f"train_test_exp=False, data_device='cuda', eval=False)\n")
    shutil.copy2(str(target_norm_path), str(merged_dir / "normalization.json"))

    return merged_dir


def finetune_merged(merged_dir, shared_data, output_dir, logs_dir, gpu):
    """Finetune merged model with v5b recipe (validated in C60-C63 reference).

    Creates checkpoint from merged PLY, then trains with V5B config.
    VizMapper factor_delta stays at 0.3/0.1 (same as per-block training).
    """
    import sys as _sys
    _sys.path.insert(0, str(PARTICLEGS_ROOT))

    ft_dir = output_dir / "finetuned"
    model_dir = ft_dir / "model"

    # Check if finetune already completed (checkpoint at iteration > 0)
    existing = find_checkpoint(model_dir)
    if existing:
        existing_iter = int(existing.stem.replace("chkpnt", ""))
        if existing_iter > 0:  # chkpnt0 is just the starting point, not a result
            return model_dir, existing_iter

    # Find merged PLY iteration
    pc_dir = Path(merged_dir) / "point_cloud"
    iter_dirs = sorted(pc_dir.glob("iteration_*"),
                       key=lambda p: int(p.name.split("_")[1]))
    src_iter = int(iter_dirs[-1].name.split("_")[1]) if iter_dirs else 0

    # Create checkpoint from merged PLY
    from pipelines.finetune import create_checkpoint_from_ply
    model_dir.mkdir(parents=True, exist_ok=True)
    chkpnt_path, n_gaussians = create_checkpoint_from_ply(
        str(merged_dir), str(model_dir), src_iter, F16_FINETUNE)
    print(f"  Created checkpoint from merged PLY: {n_gaussians} Gaussians")

    # Generate finetune training data (2K, 600 frames — matches reference F16)
    data_dir = ft_dir / "data"
    generate_training_data(
        shared_data["vtp"], data_dir, shared_data["normalization"],
        "multi_orbit", "1.0,0.7,0.5", 600, 1920, 1080,
        logs_dir, "finetune_data")
    prepare_stage_data_dir(data_dir, shared_data["ply"], shared_data["normalization"])

    # Train with v5b config, starting from merged checkpoint
    iteration = run_training(data_dir, model_dir, F16_FINETUNE, logs_dir,
                             "train_finetune",
                             start_checkpoint=chkpnt_path, gpu=gpu)

    img_dir = data_dir / "images"
    if img_dir.exists():
        shutil.rmtree(img_dir)

    return model_dir, iteration


def run_n_blocks(n_blocks, output_dir, shared_data, gpu=0, num_gpus=1,
                 compute_ssim=False):
    """Run full block pipeline for n_blocks."""
    print(f"\n{'='*70}")
    print(f"Block Pipeline: {n_blocks} blocks")
    print(f"{'='*70}")

    run_dir = output_dir / f"blocks_{n_blocks}"
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    result_file = run_dir / "result.json"
    if result_file.exists():
        existing = json.loads(result_file.read_text())
        # Re-eval for SSIM if requested but missing
        if compute_ssim and "ssim" not in existing.get("finetuned", {}):
            ft_model = run_dir / "finetuned" / "model"
            ft_pc = ft_model / "point_cloud"
            if ft_pc.exists():
                ft_iters = sorted(ft_pc.glob("iteration_*"),
                                  key=lambda p: int(p.name.split("_")[1]))
                ft_iter = int(ft_iters[-1].name.split("_")[1]) if ft_iters else 0
                print(f"  Re-evaluating finetuned model for SSIM (iter {ft_iter})...")
                ft_eval = evaluate_model(ft_model, ft_iter, shared_data, logs, gpu=gpu,
                                         factor_delta_opacity=F16_FINETUNE["factor_delta_opacity"],
                                         factor_delta_scale=F16_FINETUNE["factor_delta_scale"],
                                         min_opacity_clamp=F16_FINETUNE["min_opacity_clamp"],
                                         compute_ssim=True)
                if ft_eval["avg"].get("ssim"):
                    existing["finetuned"]["ssim"] = ft_eval["avg"]["ssim"]
                    result_file.write_text(json.dumps(existing, indent=2))
                    print(f"  SSIM={ft_eval['avg']['ssim']:.4f} saved")
        return existing

    # Partition
    print(f"\n[1] Partitioning into {n_blocks} blocks...")
    partition_info, partition_dir = partition_raw_data(n_blocks, run_dir, logs)

    # Prepare all blocks first (sequential — needs pvbatch)
    block_run_dirs = []
    block_data_list = []
    for bi in range(n_blocks):
        block_raw = partition_dir / f"block_{bi}"
        block_dir = run_dir / f"block_{bi:02d}"
        block_run_dirs.append(block_dir)

        print(f"\n[2.{bi}] Preparing block {bi}...")
        block_data = prepare_block_data(bi, block_raw, block_dir, shared_data, logs)
        block_data_list.append(block_data)

    # Train blocks in parallel (batch of num_gpus at a time)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    block_models = [None] * n_blocks

    for batch_start in range(0, n_blocks, num_gpus):
        batch_end = min(batch_start + num_gpus, n_blocks)
        batch = list(range(batch_start, batch_end))
        print(f"\n[3] Training blocks {batch} in parallel ({len(batch)} GPUs)...")

        futures = {}
        with ProcessPoolExecutor(max_workers=len(batch)) as executor:
            for bi in batch:
                block_gpu = bi % num_gpus
                fut = executor.submit(
                    train_block, bi, block_data_list[bi],
                    block_run_dirs[bi], logs, block_gpu)
                futures[fut] = bi

            for fut in as_completed(futures):
                bi = futures[fut]
                model_dir, iteration = fut.result()
                block_models[bi] = (model_dir, iteration)
                print(f"  Block {bi} done: {iteration} iterations")

    # Merge
    print(f"\n[4] Merging {n_blocks} blocks...")
    merged_dir = merge_blocks(block_models, block_run_dirs,
                              shared_data["normalization"], run_dir, logs)

    # Evaluate merged — find actual iteration from PLY
    merged_pc = Path(merged_dir) / "point_cloud"
    _iters = sorted(merged_pc.glob("iteration_*"),
                    key=lambda p: int(p.name.split("_")[1])) if merged_pc.exists() else []
    merged_iter = int(_iters[-1].name.split("_")[1]) if _iters else 0
    print(f"\n[5] Evaluating merged model (iteration {merged_iter})...")
    merged_eval = evaluate_model(merged_dir, merged_iter, shared_data, logs, gpu=gpu)
    merged_stats = get_model_stats(merged_dir, merged_iter)

    # Finetune
    print(f"\n[6] Finetuning merged model...")
    ft_model, ft_iter = finetune_merged(merged_dir, shared_data, run_dir, logs, gpu)

    # Evaluate finetuned (use F16 VizMapper params)
    print(f"\n[7] Evaluating finetuned model...")
    ft_eval = evaluate_model(ft_model, ft_iter, shared_data, logs, gpu=gpu,
                             factor_delta_opacity=F16_FINETUNE["factor_delta_opacity"],
                             factor_delta_scale=F16_FINETUNE["factor_delta_scale"],
                             min_opacity_clamp=F16_FINETUNE["min_opacity_clamp"],
                             compute_ssim=compute_ssim)
    ft_stats = get_model_stats(ft_model, ft_iter)

    result = {
        "n_blocks": n_blocks,
        "merged": {
            "masked_psnr": merged_eval["avg"]["masked_psnr"],
            "psnr": merged_eval["avg"]["psnr"],
            "num_gaussians": merged_stats["num_gaussians"],
            "size_mb": round(merged_stats["size_mb"], 1),
            "eval": merged_eval,
        },
        "finetuned": {
            "masked_psnr": ft_eval["avg"]["masked_psnr"],
            "psnr": ft_eval["avg"]["psnr"],
            "num_gaussians": ft_stats["num_gaussians"],
            "size_mb": round(ft_stats["size_mb"], 1),
            "eval": ft_eval,
        },
    }
    if compute_ssim and ft_eval["avg"].get("ssim"):
        result["finetuned"]["ssim"] = ft_eval["avg"]["ssim"]
    save_results(result, result_file)
    return result


def main():
    parser = base_parser("EXP-4: Block Training Scan")
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--blocks", type=str, default="2,4,8,16",
                        help="Comma-separated block counts to test")
    parser.add_argument("--compute_ssim", action="store_true",
                        help="Also compute SSIM (requires skimage)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RUNS_DIR / "exp4"
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    shared_data = ensure_shared_data(gpu=args.gpu) if not args.skip_data_prep else {
        "vtp": SHARED_DIR / "particles.vtp",
        "normalization": SHARED_DIR / "normalization.json",
        "ply": SHARED_DIR / "points3d.ply",
        "eval_dirs": {evd["id"]: SHARED_DIR / evd["subdir"] / "data"
                      for evd in EVAL_DATASETS},
    }

    block_counts = [int(x) for x in args.blocks.split(",")]
    results = {}

    for n in block_counts:
        results[f"blocks_{n}"] = run_n_blocks(
            n, output_dir, shared_data, gpu=args.gpu, num_gpus=args.num_gpus,
            compute_ssim=args.compute_ssim)

    # Summary table
    print(f"\n{'='*70}")
    print("EXP-4: Block Training Summary")
    print(f"{'='*70}")
    headers = ["Blocks", "Merged mPSNR", "FT mPSNR", "Gaussians", "Size (MB)"]
    if args.compute_ssim:
        headers.append("SSIM")
    rows = []
    for n in block_counts:
        r = results[f"blocks_{n}"]
        row = [
            n,
            f"{r['merged']['masked_psnr']:.2f}" if r["merged"]["masked_psnr"] else "N/A",
            f"{r['finetuned']['masked_psnr']:.2f}" if r["finetuned"]["masked_psnr"] else "N/A",
            f"{r['finetuned']['num_gaussians']/1000:.0f}k",
            r["finetuned"]["size_mb"],
        ]
        if args.compute_ssim:
            ssim = r["finetuned"].get("ssim")
            row.append(f"{ssim:.4f}" if ssim else "N/A")
        rows.append(row)
    print_table(headers, rows)

    save_results(results, output_dir / "results.json")
    print(f"\nEXP-4 complete ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
