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

# ── AE ship-4-blocks fast path ──────────────────────────────────────────────
# The AE artifact ships the N trained sub-block models so the reviewer does NOT
# re-run the N per-block end-to-end trainings (each = a full 4K/6K GT render +
# 39k-iter train). Combined with the shipped E25 model (EXP-1, pretrained/e25/),
# the AE trains only ONE model live: this merged-model 60k finetune (~17 min).
# The sub-blocks and E25 are provided pre-trained; all GT is still rendered live.
#
# Upload layout (self-contained, only what merge_blocks() needs), per block i:
#     pretrained/blocks_<N>/block_<ii>/
#         model/point_cloud/iteration_<it>/point_cloud.ply
#         model/viz_mapper_<it>.pth
#         shared/normalization.json
#
# Full scripts/reproduce.sh ships no pretrained/ dir → _find_pretrained_blocks
# returns None → blocks train live exactly as before. Old behavior untouched.
PRETRAINED_DIR = PARTICLEGS_ROOT / "pretrained"


def _find_pretrained_blocks(n_blocks):
    """Return (block_models, block_run_dirs) from shipped pretrained/blocks_<N>/,
    matching the train_block()/merge_blocks() contract, or None if any block is
    absent/incomplete (→ caller trains the blocks live)."""
    base = PRETRAINED_DIR / f"blocks_{n_blocks}"
    if not base.is_dir():
        return None
    block_models, block_run_dirs = [], []
    for bi in range(n_blocks):
        bdir = base / f"block_{bi:02d}"
        model_dir = bdir / "model"
        pc = model_dir / "point_cloud"
        norm = bdir / "shared" / "normalization.json"
        iters = sorted(pc.glob("iteration_*"),
                       key=lambda p: int(p.name.split("_")[1])) if pc.is_dir() else []
        if not iters or not norm.exists():
            print(f"  [pretrained] {bdir} incomplete — training all blocks live")
            return None
        it = int(iters[-1].name.split("_")[1])
        if not (model_dir / f"viz_mapper_{it}.pth").exists() and bi == 0:
            print(f"  [pretrained] block 0 missing viz_mapper_{it}.pth — live")
            return None
        block_models.append((model_dir, it))
        block_run_dirs.append(bdir)
    return block_models, block_run_dirs


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


def render_stage_data(block_id, block_data, block_run_dir, si, stage, logs_dir):
    """Generate GT training data for one block stage and return its data_dir.

    A stage's GT render depends only on the VTP + camera/viz params, NOT on any
    training checkpoint, so stages can be rendered ahead of (and concurrently
    with) training. Idempotent: skips if the stage is already trained, and the
    underlying generate_* helpers skip if the images already exist.
    """
    stage_name = f"block{block_id}_{stage['name']}"
    stage_dir = block_run_dir / f"{si:02d}_{stage['name']}"
    data_dir = stage_dir / "data"

    # If this stage is already fully trained (restart), skip rendering entirely.
    if find_checkpoint(stage_dir / "model"):
        return data_dir

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
    return data_dir


def train_block(block_id, block_data, block_run_dir, logs_dir, gpu):
    """Train one block through 3 progressive stages, pipelining render vs train.

    Runs in a ProcessPoolExecutor worker, so set_pvbatch_cuda_device pins this
    block's GT rendering to its own GPU (parallel blocks render concurrently
    instead of all serializing pvbatch onto CUDA 0).

    Within the block, rendering (ParaView, CPU-heavy) and training (3DGS,
    GPU-heavy) use complementary resources, so a single-slot prefetch thread
    renders stage k+1's GT while stage k trains on the GPU. The render for a
    stage is checkpoint-independent, so this is safe; it overlaps the render
    bottleneck with training on the same block_gpu instead of running them
    strictly back-to-back.
    """
    from concurrent.futures import ThreadPoolExecutor

    set_pvbatch_cuda_device(gpu)
    prev_checkpoint = None
    final_model = None
    final_iter = None

    with ThreadPoolExecutor(max_workers=1) as prefetch:
        # Kick off stage 0's render; each iteration prefetches the next stage's
        # render before training the current one.
        render_fut = prefetch.submit(
            render_stage_data, block_id, block_data, block_run_dir,
            0, PER_BLOCK_STAGES[0], logs_dir)

        for si, stage in enumerate(PER_BLOCK_STAGES):
            stage_name = f"block{block_id}_{stage['name']}"
            stage_dir = block_run_dir / f"{si:02d}_{stage['name']}"
            model_dir = stage_dir / "model"

            data_dir = render_fut.result()  # this stage's GT (prefetched)

            # Start prefetching the NEXT stage's render so it overlaps whatever
            # we do for this stage (train or skip) below.
            if si + 1 < len(PER_BLOCK_STAGES):
                render_fut = prefetch.submit(
                    render_stage_data, block_id, block_data, block_run_dir,
                    si + 1, PER_BLOCK_STAGES[si + 1], logs_dir)

            existing = find_checkpoint(model_dir)
            if existing:
                prev_checkpoint = existing
                final_model = model_dir
                final_iter = int(existing.stem.replace("chkpnt", ""))
                continue

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

            # Cleanup this stage's images once its training has consumed them.
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
                 compute_ssim=False, use_pretrained=False):
    """Run full block pipeline for n_blocks.

    use_pretrained: AE fast path only. When True AND pretrained/blocks_<N>/ is
    shipped, skip partition + per-block end-to-end training and go straight to
    merge → finetune (1+4+1 → 1+1). Full reproduce.sh leaves this False, so it
    trains all blocks live even though pretrained/ is present on disk — the
    'full reproduction' semantics are preserved.
    """
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

    # AE fast path (1+4+1 → 1+1): only when explicitly enabled (AE mode) AND the
    # N sub-block models are shipped, skip partition + per-block end-to-end
    # training and go straight to merge. Full reproduce.sh keeps use_pretrained
    # False → trains live even if pretrained/ exists on disk.
    pretrained = _find_pretrained_blocks(n_blocks) if use_pretrained else None
    if pretrained is not None:
        block_models, block_run_dirs = pretrained
        print(f"\n[AE] Using {n_blocks} shipped sub-block models — skipping "
              f"partition + per-block training (1+4+1 → 1+1)")
        for bi, (md, it) in enumerate(block_models):
            print(f"  block {bi}: {md} @ iter {it}")
        (run_dir / "blocktrain_done").touch()
    else:
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

        # Signal to the AE scheduler that the multi-GPU block-training phase is
        # complete. Everything after this (merge / eval / finetune) runs on the
        # single base GPU, so the scheduler can release the other GPUs to the
        # 7/8/14 quality-metric pool while this finetune tail runs.
        (run_dir / "blocktrain_done").touch()

    # Merge
    print(f"\n[4] Merging {n_blocks} blocks...")
    _merge_t0 = time.time()
    merged_dir = merge_blocks(block_models, block_run_dirs,
                              shared_data["normalization"], run_dir, logs)
    merge_time_min = round((time.time() - _merge_t0) / 60, 2)

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
    _ft_t0 = time.time()
    ft_model, ft_iter = finetune_merged(merged_dir, shared_data, run_dir, logs, gpu)
    finetune_time_min = round((time.time() - _ft_t0) / 60, 2)

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
            "merge_time_min": merge_time_min,
            "eval": merged_eval,
        },
        "finetuned": {
            "masked_psnr": ft_eval["avg"]["masked_psnr"],
            "psnr": ft_eval["avg"]["psnr"],
            "num_gaussians": ft_stats["num_gaussians"],
            "size_mb": round(ft_stats["size_mb"], 1),
            "finetune_time_min": finetune_time_min,
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
    parser.add_argument("--use_pretrained_blocks", action="store_true",
                        help="AE fast path: consume shipped pretrained/blocks_N/ "
                             "sub-block models instead of training them live "
                             "(1+4+1 → 1+1). Full reproduce.sh must NOT set this.")
    args = parser.parse_args()
    set_pvbatch_cuda_device(args.gpu)  # prepare/merge/finetune renders on base GPU

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
            compute_ssim=args.compute_ssim,
            use_pretrained=args.use_pretrained_blocks)

    # Summary table
    print(f"\n{'='*70}")
    print("EXP-4: Block Training Summary")
    print(f"{'='*70}")
    headers = ["Blocks", "Merged PSNR", "Merged mPSNR", "FT PSNR", "FT mPSNR",
               "Gaussians", "Size (MB)"]
    if args.compute_ssim:
        headers.append("SSIM")

    def _f(v):
        return f"{v:.2f}" if v else "N/A"

    rows = []
    for n in block_counts:
        r = results[f"blocks_{n}"]
        row = [
            n,
            _f(r["merged"]["psnr"]),
            _f(r["merged"]["masked_psnr"]),
            _f(r["finetuned"]["psnr"]),
            _f(r["finetuned"]["masked_psnr"]),
            f"{r['finetuned']['num_gaussians']/1000:.0f}k",
            r["finetuned"]["size_mb"],
        ]
        if args.compute_ssim:
            ssim = r["finetuned"].get("ssim")
            row.append(f"{ssim:.4f}" if ssim else "N/A")
        rows.append(row)
    print_table(headers, rows)
    print("  PSNR = full-image (what paper Tab. III prints); "
          "mPSNR = foreground-masked (what verify_results.py enforces)")

    save_results(results, output_dir / "results.json")
    print(f"\nEXP-4 complete ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
