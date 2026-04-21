#!/usr/bin/env python3
"""Merge multiple block-trained 3DGS PLY files into a single PLY.

Each block has its own normalization (center, scale). This script:
1. Loads each block's PLY and normalization
2. Transforms xyz from block-local to target normalized space
3. Adjusts Gaussian log-scales for the coordinate system change
4. Concatenates all Gaussians and writes a merged PLY
5. Copies VizMapper weights from block 0

Usage:
    python merge_blocks.py \
        --block_dir runs/my_exp/blocks \
        --target_norm runs/my_exp/shared/normalization.json \
        --output_dir runs/my_exp/merged
"""

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def load_normalization(path):
    with open(path) as f:
        data = json.load(f)
    return np.array(data["center"], dtype=np.float64), float(data["scale"])


def load_ply_raw(path):
    """Load PLY and return vertex element data as structured numpy array."""
    plydata = PlyData.read(path)
    return plydata.elements[0]


def find_block_dirs(block_dir):
    """Find block subdirectories sorted by index."""
    block_dir = Path(block_dir)
    dirs = sorted(d for d in block_dir.glob("*_block_*") if d.is_dir())
    if not dirs:
        raise FileNotFoundError(f"No block directories found in {block_dir}")
    return dirs


def find_model_path(block_dir):
    """Find the final stage model directory within a block."""
    stages = sorted((block_dir / "stages").iterdir())
    if not stages:
        raise FileNotFoundError(f"No stages found in {block_dir}")
    return stages[-1] / "model"


def find_ply(model_dir, iteration=None):
    """Find the PLY file at the given iteration (or highest available)."""
    pc_dir = model_dir / "point_cloud"
    if iteration is not None:
        ply_path = pc_dir / f"iteration_{iteration}" / "point_cloud.ply"
        if ply_path.exists():
            return ply_path, iteration
    iter_dirs = sorted(pc_dir.glob("iteration_*"),
                       key=lambda p: int(p.name.split("_")[1]))
    if not iter_dirs:
        raise FileNotFoundError(f"No iteration dirs in {pc_dir}")
    best = iter_dirs[-1]
    it = int(best.name.split("_")[1])
    return best / "point_cloud.ply", it


def merge_blocks(block_dir, target_norm_path, output_dir, iteration=None):
    block_dir = Path(block_dir)
    output_dir = Path(output_dir)

    target_center, target_sf = load_normalization(target_norm_path)
    print(f"Target normalization: center={target_center.tolist()}, scale={target_sf:.8f}")

    block_dirs = find_block_dirs(block_dir)
    print(f"Found {len(block_dirs)} blocks: {[d.name for d in block_dirs]}")

    all_vertices = []
    total_gaussians = 0
    actual_iteration = None

    for i, bd in enumerate(block_dirs):
        norm_path = bd / "normalization.json"
        block_center, block_sf = load_normalization(norm_path)

        model_dir = find_model_path(bd)
        ply_path, it = find_ply(model_dir, iteration)
        if actual_iteration is None:
            actual_iteration = it
        print(f"\nBlock {i} ({bd.name}):")
        print(f"  normalization: center={block_center.tolist()}, scale={block_sf:.8f}")
        print(f"  PLY: {ply_path} (iteration {it})")

        vertex = load_ply_raw(ply_path)
        n = len(vertex.data)
        total_gaussians += n
        print(f"  Gaussians: {n}")

        xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)

        # Transform: block normalized -> world -> target normalized
        scale_ratio = target_sf / block_sf
        offset = (block_center - target_center) * target_sf
        xyz_target = xyz * scale_ratio + offset

        log_correction = math.log(scale_ratio)
        print(f"  scale_ratio: {scale_ratio:.6f}, log-scale correction: {log_correction:.6f}")

        new_data = vertex.data.copy()
        new_data["x"] = xyz_target[:, 0].astype(np.float32)
        new_data["y"] = xyz_target[:, 1].astype(np.float32)
        new_data["z"] = xyz_target[:, 2].astype(np.float32)

        scale_names = [p.name for p in vertex.properties if p.name.startswith("scale_")]
        for sn in scale_names:
            new_data[sn] = (new_data[sn].astype(np.float64) + log_correction).astype(np.float32)

        all_vertices.append(new_data)

    print(f"\nTotal Gaussians: {total_gaussians}")

    merged = np.concatenate(all_vertices)
    assert len(merged) == total_gaussians

    xyz_merged = np.stack([merged["x"], merged["y"], merged["z"]], axis=1)
    print(f"Merged xyz range: [{xyz_merged.min(axis=0).tolist()}, {xyz_merged.max(axis=0).tolist()}]")

    out_pc_dir = output_dir / "point_cloud" / f"iteration_{actual_iteration}"
    out_pc_dir.mkdir(parents=True, exist_ok=True)
    out_ply = out_pc_dir / "point_cloud.ply"

    el = PlyElement.describe(merged, "vertex")
    PlyData([el]).write(str(out_ply))
    print(f"\nWrote merged PLY: {out_ply} ({total_gaussians} Gaussians)")

    # Copy VizMapper from block 0
    block0_model = find_model_path(block_dirs[0])
    vm_src = block0_model / f"viz_mapper_{actual_iteration}.pth"
    vm_dst = output_dir / f"viz_mapper_{actual_iteration}.pth"
    if vm_src.exists():
        shutil.copy2(vm_src, vm_dst)
        print(f"Copied VizMapper: {vm_src.name}")
    else:
        print(f"WARNING: VizMapper not found at {vm_src}")

    # Write cfg_args for render compatibility
    cfg_args = output_dir / "cfg_args"
    cfg_args.write_text(
        f"Namespace(sh_degree=0, source_path='', "
        f"model_path='{output_dir}', images='images', depths='', "
        f"resolution=1, white_background=False, train_test_exp=False, "
        f"data_device='cuda', eval=False)\n"
    )

    shutil.copy2(target_norm_path, output_dir / "normalization.json")

    return output_dir, actual_iteration, total_gaussians


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge block-trained 3DGS models")
    parser.add_argument("--block_dir", required=True)
    parser.add_argument("--target_norm", required=True, help="Path to target normalization.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--iteration", type=int, default=None)
    args = parser.parse_args()

    merge_blocks(args.block_dir, args.target_norm, args.output_dir, args.iteration)
