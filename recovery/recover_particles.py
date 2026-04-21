#!/usr/bin/env python3
"""Recover particle positions from a trained 3DGS model.

Implements multiple sampling strategies (V0-V11) for converting Gaussian
distributions back into particle positions. The primary use case is
data decompression: 3DGS model → recovered particle coordinates.

Usage:
    python recover_particles.py \
        --ply_path runs/my_exp/model/point_cloud/iteration_60000/point_cloud.ply \
        --normalization runs/my_exp/model/normalization.json \
        --output_dir runs/my_exp/recovered \
        --num_points 280953867 \
        --method V0_baseline
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData


NUM_POINTS_FULL = 280_953_867
SEED = 42


# ── Gaussian loading ─────────────────────────────────────────────────────

def load_gaussians(ply_path):
    """Load Gaussian parameters from a PLY file."""
    plydata = PlyData.read(str(ply_path))
    v = plydata["vertex"]
    means = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float64)
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1).astype(np.float64))
    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1).astype(np.float64)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-12
    opacity_logit = v["opacity"].astype(np.float64)
    opacities = 1.0 / (1.0 + np.exp(-opacity_logit))
    print(f"Loaded {len(means)} Gaussians")
    print(f"  scale range: [{scales.min():.6f}, {scales.max():.6f}]")
    print(f"  opacity range: [{opacities.min():.4f}, {opacities.max():.4f}]")
    volumes = scales[:, 0] * scales[:, 1] * scales[:, 2]
    print(f"  volume range: [{volumes.min():.2e}, {volumes.max():.2e}], median={np.median(volumes):.2e}")
    return means, scales, quats, opacities


def quat_to_rotation(quats):
    """Convert quaternions (w,x,y,z) to rotation matrices."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    R = np.zeros((len(quats), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2*(y*y + z*z); R[:, 0, 1] = 2*(x*y - w*z); R[:, 0, 2] = 2*(x*z + w*y)
    R[:, 1, 0] = 2*(x*y + w*z); R[:, 1, 1] = 1 - 2*(x*x + z*z); R[:, 1, 2] = 2*(y*z - w*x)
    R[:, 2, 0] = 2*(x*z - w*y); R[:, 2, 1] = 2*(y*z + w*x); R[:, 2, 2] = 1 - 2*(x*x + y*y)
    return R


def save_raw_f32(points, output_dir):
    """Save particle coordinates as separate xx/yy/zz.f32 files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for axis, name in enumerate(["xx", "yy", "zz"]):
        points[:, axis].astype(np.float32).tofile(str(output_dir / f"{name}.f32"))


# ── Sampling strategies ──────────────────────────────────────────────────

def sample_baseline(means, scales, quats, opacities, num_points, rng):
    """V0: weight = opacity, Gaussian sampling."""
    weights = opacities / opacities.sum()
    idx = rng.choice(len(means), size=num_points, p=weights)
    rotations = quat_to_rotation(quats)
    z = rng.standard_normal((num_points, 3))
    points = means[idx] + np.einsum("nij,nj->ni", rotations[idx], z * scales[idx])
    return points


def sample_volume_weighted(means, scales, quats, opacities, num_points, rng):
    """V1: weight = opacity x volume."""
    volumes = scales[:, 0] * scales[:, 1] * scales[:, 2]
    raw_weights = opacities * volumes
    weights = raw_weights / raw_weights.sum()
    idx = rng.choice(len(means), size=num_points, p=weights)
    rotations = quat_to_rotation(quats)
    z = rng.standard_normal((num_points, 3))
    points = means[idx] + np.einsum("nij,nj->ni", rotations[idx], z * scales[idx])
    return points


def sample_scale_factor(means, scales, quats, opacities, num_points, rng, factor=3.0):
    """V2+: weight = opacity, scales multiplied by factor."""
    weights = opacities / opacities.sum()
    idx = rng.choice(len(means), size=num_points, p=weights)
    rotations = quat_to_rotation(quats)
    z = rng.standard_normal((num_points, 3))
    scaled = scales * factor
    points = means[idx] + np.einsum("nij,nj->ni", rotations[idx], z * scaled[idx])
    return points


def sample_volume_weighted_scale_factor(means, scales, quats, opacities, num_points, rng, factor=3.0):
    """V3: Volume-weighted + scale factor."""
    volumes = scales[:, 0] * scales[:, 1] * scales[:, 2]
    raw_weights = opacities * volumes
    weights = raw_weights / raw_weights.sum()
    idx = rng.choice(len(means), size=num_points, p=weights)
    rotations = quat_to_rotation(quats)
    z = rng.standard_normal((num_points, 3))
    scaled = scales * factor
    points = means[idx] + np.einsum("nij,nj->ni", rotations[idx], z * scaled[idx])
    return points


def sample_uniform_ellipsoid(means, scales, quats, opacities, num_points, rng, radius_sigma=3.0):
    """V4: Uniform sampling within each Gaussian's ellipsoid."""
    weights = opacities / opacities.sum()
    idx = rng.choice(len(means), size=num_points, p=weights)
    rotations = quat_to_rotation(quats)
    z = rng.standard_normal((num_points, 3))
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
    r = rng.uniform(0, 1, size=(num_points, 1)) ** (1.0 / 3.0)
    z = z * r * radius_sigma
    points = means[idx] + np.einsum("nij,nj->ni", rotations[idx], z * scales[idx])
    return points


def sample_volume_uniform_ellipsoid(means, scales, quats, opacities, num_points, rng, radius_sigma=3.0):
    """V5: Volume-weighted + uniform ellipsoid."""
    volumes = scales[:, 0] * scales[:, 1] * scales[:, 2]
    raw_weights = opacities * volumes
    weights = raw_weights / raw_weights.sum()
    idx = rng.choice(len(means), size=num_points, p=weights)
    rotations = quat_to_rotation(quats)
    z = rng.standard_normal((num_points, 3))
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
    r = rng.uniform(0, 1, size=(num_points, 1)) ** (1.0 / 3.0)
    z = z * r * radius_sigma
    points = means[idx] + np.einsum("nij,nj->ni", rotations[idx], z * scales[idx])
    return points


def sample_density_field(means, scales, quats, opacities, num_points, rng, grid_size=512):
    """V6: Build 3D density field from Gaussians, then sample from it."""
    margin = 0.1
    box_min = means.min(axis=0) - margin
    box_max = means.max(axis=0) + margin
    box_size = box_max - box_min
    voxel_size = box_size / grid_size

    print(f"  Building density field ({grid_size}^3)...")
    density = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
    idx_3d = ((means - box_min) / box_size * grid_size).astype(np.int64)
    np.clip(idx_3d, 0, grid_size - 1, out=idx_3d)
    volumes = scales[:, 0] * scales[:, 1] * scales[:, 2]
    weights = opacities * volumes
    np.add.at(density, (idx_3d[:, 0], idx_3d[:, 1], idx_3d[:, 2]), weights)

    density_flat = density.ravel()
    density_flat /= density_flat.sum()

    print(f"  Sampling {num_points:,} particles from density field...")
    voxel_indices = rng.choice(len(density_flat), size=num_points, p=density_flat)

    iz = voxel_indices % grid_size
    iy = (voxel_indices // grid_size) % grid_size
    ix = voxel_indices // (grid_size * grid_size)

    jitter = rng.uniform(0, 1, size=(num_points, 3))
    points = np.stack([
        box_min[0] + (ix + jitter[:, 0]) * voxel_size[0],
        box_min[1] + (iy + jitter[:, 1]) * voxel_size[1],
        box_min[2] + (iz + jitter[:, 2]) * voxel_size[2],
    ], axis=-1)
    return points


# ── Method registry ──────────────────────────────────────────────────────

METHODS = {
    "V0_baseline": lambda m, s, q, o, n, rng: sample_baseline(m, s, q, o, n, rng),
    "V1_volume_weight": lambda m, s, q, o, n, rng: sample_volume_weighted(m, s, q, o, n, rng),
    "V2_scale_3x": lambda m, s, q, o, n, rng: sample_scale_factor(m, s, q, o, n, rng, factor=3.0),
    "V3_volume_scale_3x": lambda m, s, q, o, n, rng: sample_volume_weighted_scale_factor(m, s, q, o, n, rng, factor=3.0),
    "V4_uniform_ellipsoid": lambda m, s, q, o, n, rng: sample_uniform_ellipsoid(m, s, q, o, n, rng, radius_sigma=3.0),
    "V5_volume_uniform": lambda m, s, q, o, n, rng: sample_volume_uniform_ellipsoid(m, s, q, o, n, rng, radius_sigma=3.0),
    "V6_density_field": lambda m, s, q, o, n, rng: sample_density_field(m, s, q, o, n, rng, grid_size=512),
    "V7_scale_1.5x": lambda m, s, q, o, n, rng: sample_scale_factor(m, s, q, o, n, rng, factor=1.5),
    "V8_scale_2.0x": lambda m, s, q, o, n, rng: sample_scale_factor(m, s, q, o, n, rng, factor=2.0),
    "V9_scale_2.5x": lambda m, s, q, o, n, rng: sample_scale_factor(m, s, q, o, n, rng, factor=2.5),
    "V10_scale_1.2x": lambda m, s, q, o, n, rng: sample_scale_factor(m, s, q, o, n, rng, factor=1.2),
    "V11_scale_1.8x": lambda m, s, q, o, n, rng: sample_scale_factor(m, s, q, o, n, rng, factor=1.8),
}


def recover(ply_path, normalization_path, output_dir, method="V0_baseline",
            num_points=NUM_POINTS_FULL, seed=SEED):
    """Run particle recovery with the given method.

    Returns:
        (N, 3) float32 array of recovered particle positions in world coordinates.
    """
    means, scales, quats, opacities = load_gaussians(ply_path)

    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}. Available: {list(METHODS.keys())}")

    print(f"\nRecovering {num_points:,} particles using method: {method}")
    t0 = time.time()
    rng = np.random.default_rng(seed)
    points = METHODS[method](means, scales, quats, opacities, num_points, rng)
    t_sample = time.time() - t0
    print(f"  Sampled in {t_sample:.1f}s")
    print(f"  Range (normalized): x=[{points[:,0].min():.4f}, {points[:,0].max():.4f}]")

    # Denormalize
    with open(normalization_path) as f:
        norm = json.load(f)
    center = np.array(norm["center"], dtype=np.float64)
    scale = float(norm["scale"])
    points = points / scale + center
    print(f"  Range (world): x=[{points[:,0].min():.2f}, {points[:,0].max():.2f}]")

    # Save
    output_dir = Path(output_dir)
    save_raw_f32(points, output_dir)
    print(f"  Saved to {output_dir}")

    # Save metadata
    meta = {
        "method": method,
        "num_points": num_points,
        "seed": seed,
        "ply_path": str(ply_path),
        "sample_time_s": t_sample,
    }
    with open(output_dir / "recovery_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return points.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Recover particles from trained 3DGS model")
    parser.add_argument("--ply_path", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", default="V0_baseline",
                        choices=list(METHODS.keys()))
    parser.add_argument("--num_points", type=int, default=NUM_POINTS_FULL)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--all_methods", action="store_true",
                        help="Run all methods sequentially")
    args = parser.parse_args()

    if args.all_methods:
        for method_name in METHODS:
            out = Path(args.output_dir) / method_name
            recover(args.ply_path, args.normalization, out,
                    method=method_name, num_points=args.num_points, seed=args.seed)
    else:
        recover(args.ply_path, args.normalization, args.output_dir,
                method=args.method, num_points=args.num_points, seed=args.seed)


if __name__ == "__main__":
    main()
