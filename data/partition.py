#!/usr/bin/env python3
"""KD-tree partition of raw .f32 particle files into spatial blocks.

Usage:
    python partition.py \
        --raw_x data/hacc_raw/xx.f32 \
        --raw_y data/hacc_raw/yy.f32 \
        --raw_z data/hacc_raw/zz.f32 \
        --num_blocks 8 \
        --output_dir /tmp/partitions
"""

import argparse
import json
import numpy as np
from pathlib import Path


def kdtree_partition(coords, num_blocks):
    """Recursively split particles on axis with max spread at median.

    Uses np.argpartition for exact median split, guaranteeing balanced
    particle counts across blocks.

    Args:
        coords: (N, 3) array of particle coordinates
        num_blocks: number of blocks (must be power of 2)

    Returns:
        List of index arrays, one per block
    """
    if num_blocks & (num_blocks - 1) != 0 or num_blocks < 1:
        raise ValueError(f"num_blocks must be a power of 2, got {num_blocks}")

    leaves = [np.arange(coords.shape[0])]

    while len(leaves) < num_blocks:
        new_leaves = []
        for idx_array in leaves:
            block_coords = coords[idx_array]
            ranges = block_coords.max(axis=0) - block_coords.min(axis=0)
            axis = int(np.argmax(ranges))
            axis_vals = block_coords[:, axis]
            mid = len(axis_vals) // 2
            partition_order = np.argpartition(axis_vals, mid)
            new_leaves.append(idx_array[partition_order[:mid]])
            new_leaves.append(idx_array[partition_order[mid:]])
        leaves = new_leaves

    return leaves


def main():
    parser = argparse.ArgumentParser(description="Partition raw .f32 particle files into spatial blocks")
    parser.add_argument("--raw_x", type=str, required=True, help="Path to xx.f32")
    parser.add_argument("--raw_y", type=str, required=True, help="Path to yy.f32")
    parser.add_argument("--raw_z", type=str, required=True, help="Path to zz.f32")
    parser.add_argument("--num_blocks", type=int, required=True,
                        help="Number of blocks (must be power of 2)")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print(f"Loading raw particle data...")
    x = np.fromfile(args.raw_x, dtype=np.float32)
    y = np.fromfile(args.raw_y, dtype=np.float32)
    z = np.fromfile(args.raw_z, dtype=np.float32)
    assert len(x) == len(y) == len(z), \
        f"Particle count mismatch: x={len(x)}, y={len(y)}, z={len(z)}"

    total = len(x)
    print(f"Total particles: {total:,}")
    print(f"Partitioning into {args.num_blocks} blocks...")

    coords = np.stack([x, y, z], axis=1)
    block_indices = kdtree_partition(coords, args.num_blocks)

    block_metadata = []
    for i, indices in enumerate(block_indices):
        block_dir = output_dir / f"block_{i}"
        block_dir.mkdir(parents=True, exist_ok=True)

        bx, by, bz = x[indices], y[indices], z[indices]
        bx.tofile(block_dir / "xx.f32")
        by.tofile(block_dir / "yy.f32")
        bz.tofile(block_dir / "zz.f32")

        bounds = {
            "x": [float(bx.min()), float(bx.max())],
            "y": [float(by.min()), float(by.max())],
            "z": [float(bz.min()), float(bz.max())],
        }
        block_metadata.append({
            "id": i,
            "num_particles": len(indices),
            "bounds": bounds,
        })
        print(f"  Block {i}: {len(indices):,} particles, "
              f"x=[{bounds['x'][0]:.2f}, {bounds['x'][1]:.2f}], "
              f"y=[{bounds['y'][0]:.2f}, {bounds['y'][1]:.2f}], "
              f"z=[{bounds['z'][0]:.2f}, {bounds['z'][1]:.2f}]")

    partition_info = {
        "num_blocks": args.num_blocks,
        "method": "kdtree",
        "total_particles": total,
        "blocks": block_metadata,
    }
    info_path = output_dir / "partition_info.json"
    with open(info_path, "w") as f:
        json.dump(partition_info, f, indent=2)

    print(f"Partition info saved to {info_path}")


if __name__ == "__main__":
    main()
