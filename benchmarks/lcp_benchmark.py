#!/usr/bin/env python3
"""LCP CR vs PSNR Benchmark for Particle Visualization.

Sweeps LCP ABS error bounds, compresses/decompresses particle data,
renders with ParaView, and computes visualization PSNR vs GT images.
Produces a CR-PSNR curve comparable to the SZ3 benchmark.

LCP (Lossy Compression for Particles) compresses all three coordinate
files jointly, exploiting spatial locality via particle reordering.

Prerequisites:
    - LCP binary built (https://github.com/hpdslab/LCP.git)
    - pvbatch for ParaView rendering

Usage:
    python lcp_benchmark.py \
        --lcp_bin /path/to/lcp \
        --gt_raw_dir data/hacc_raw \
        --eval_base runs/shared/evaluation \
        --normalization runs/shared/normalization.json \
        --output_dir benchmarks/lcp_results
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from sz3_benchmark import (
    EVAL_DATASETS,
    NUM_POINTS,
    PREPARE_SCRIPT,
    GENERATE_SCRIPT,
    RENDER_PARAMS,
    PARTICLEGS_ROOT,
    PVBATCH_BIN,
    _get_pvbatch_egl_args,
    compute_psnr_from_dirs,
    run_cmd,
)

# Default EB sweep (log-spaced, starting from 0.005 to avoid LCP decompress bug at very small EB)
DEFAULT_EB_SWEEP = [0.005, 0.01, 0.02, 0.035, 0.055, 0.08, 0.12, 0.18, 0.27, 0.4, 0.6]


# ── LCP compress/decompress ──────────────────────────────────────────────

def lcp_compress(lcp_bin, raw_x, raw_y, raw_z, output_lcp, error_bound, num_points, log_path):
    """Compress three f32 files with LCP (single timestep, ABS mode)."""
    run_cmd([lcp_bin,
             "-i", str(raw_x), str(raw_y), str(raw_z),
             "-1", str(num_points),
             "-eb", str(error_bound),
             "-z", str(output_lcp)],
            log_path)
    return Path(output_lcp).stat().st_size


def lcp_decompress(lcp_bin, input_lcp, out_x, out_y, out_z, num_points, log_path):
    """Decompress an LCP file back to three f32 files."""
    run_cmd([lcp_bin,
             "-z", str(input_lcp),
             "-o", str(out_x), str(out_y), str(out_z),
             "-1", str(num_points)],
            log_path)


# ── Per-error-bound processing ────────────────────────────────────────────

def run_eb_point(lcp_bin, gt_raw_dir, norm_json, eb, output_dir, pvbatch_egl, gt_eval_images):
    """Run one error bound point: compress, decompress, render, compute PSNR."""
    eb_dir = output_dir / f"eb_{eb:.4e}"
    if (eb_dir / "results.json").exists():
        with open(eb_dir / "results.json") as f:
            return json.load(f)

    eb_dir.mkdir(parents=True, exist_ok=True)
    logs = eb_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    # Compress (LCP takes all 3 axes at once → single compressed file)
    lcp_path = eb_dir / "particles.lcp"
    raw_x = gt_raw_dir / "xx.f32"
    raw_y = gt_raw_dir / "yy.f32"
    raw_z = gt_raw_dir / "zz.f32"
    raw_size = sum(f.stat().st_size for f in [raw_x, raw_y, raw_z])

    compressed_size = lcp_compress(
        lcp_bin, raw_x, raw_y, raw_z, lcp_path, eb, NUM_POINTS,
        logs / "compress.log")

    cr = raw_size / compressed_size if compressed_size > 0 else 0
    print(f"\n  EB={eb:.4e}: CR={cr:.2f}x ({compressed_size/1e6:.1f} MB)")

    # Decompress
    dec_x = eb_dir / "xx.f32"
    dec_y = eb_dir / "yy.f32"
    dec_z = eb_dir / "zz.f32"
    lcp_decompress(lcp_bin, lcp_path, dec_x, dec_y, dec_z, NUM_POINTS,
                   logs / "decompress.log")

    # Create VTP from decompressed data
    vtp_dir = eb_dir / "vtp"
    run_cmd([PVBATCH_BIN] + pvbatch_egl + [
        str(PREPARE_SCRIPT),
        "--raw_x", str(dec_x),
        "--raw_y", str(dec_y),
        "--raw_z", str(dec_z),
        "--output_dir", str(vtp_dir),
        "--num_points_raw", "0",
        "--skip_images",
    ], logs / "create_vtp.log")
    vtp_path = vtp_dir / "particles.vtp"

    # Render and compute PSNR for each eval dataset
    eval_results = {}
    for eval_name, cfg in EVAL_DATASETS.items():
        render_out = eb_dir / f"render_{eval_name}"
        pv_args = [
            "--vtp_path", str(vtp_path),
            "--output_dir", str(render_out),
            "--camera_strategy", "multi_orbit",
            "--orbit_radii", cfg["orbit_radii"],
            "--normalization_path", str(norm_json),
        ]
        for k, v in RENDER_PARAMS.items():
            pv_args += [f"--{k}", v]

        run_cmd([PVBATCH_BIN] + pvbatch_egl + [str(GENERATE_SCRIPT)] + pv_args,
                logs / f"render_{eval_name}.log")

        gt_images = gt_eval_images.get(eval_name)
        if gt_images and gt_images.exists():
            psnr, mpsnr, n = compute_psnr_from_dirs(
                gt_images, render_out / "images")
            eval_results[eval_name] = {"psnr": psnr, "masked_psnr": mpsnr, "n": n}
            if psnr:
                print(f"    {eval_name}: PSNR={psnr:.2f}  masked={mpsnr:.2f}")

        # Cleanup rendered images
        if render_out.exists():
            shutil.rmtree(render_out)

    # Cleanup VTP + decompressed data + compressed file
    if vtp_path.exists():
        vtp_path.unlink()
    for f in [dec_x, dec_y, dec_z, lcp_path]:
        if f.exists():
            f.unlink()
    # Remove other generated data files
    for name in ["normalization.json", "points3d.ply",
                 "transforms_train.json", "transforms_test.json"]:
        p = eb_dir / name
        if p.exists():
            p.unlink()
    for d in [eb_dir / "images", vtp_dir]:
        if d.exists():
            shutil.rmtree(d)

    result = {
        "eb": eb,
        "cr": cr,
        "raw_bytes": raw_size,
        "compressed_bytes": compressed_size,
        "eval": eval_results,
    }
    with open(eb_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LCP CR vs PSNR benchmark")
    parser.add_argument("--lcp_bin", required=True, help="Path to lcp binary")
    parser.add_argument("--gt_raw_dir", required=True)
    parser.add_argument("--eval_base", required=True,
                        help="Path to eval datasets (with 01_eval_far/data/images, etc.)")
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eb_values", type=str, default=None,
                        help="Comma-separated error bound values")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_raw_dir = Path(args.gt_raw_dir)
    eval_base = Path(args.eval_base)
    norm_json = Path(args.normalization)

    pvbatch_egl = _get_pvbatch_egl_args()

    if args.eb_values:
        eb_values = [float(x.strip()) for x in args.eb_values.split(",")]
    else:
        eb_values = DEFAULT_EB_SWEEP

    gt_eval_images = {}
    for name, cfg in EVAL_DATASETS.items():
        sub = {"eval_far": "01_eval_far", "eval_mid": "02_eval_mid",
               "eval_near": "03_eval_near"}[name]
        gt_eval_images[name] = eval_base / sub / "data" / "images"

    print(f"LCP Benchmark: {len(eb_values)} error bounds")
    print(f"  EB values: {eb_values}")

    all_results = []
    for eb in sorted(eb_values):
        result = run_eb_point(
            args.lcp_bin, gt_raw_dir, norm_json, eb, output_dir,
            pvbatch_egl, gt_eval_images)
        all_results.append(result)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"LCP CR vs PSNR Summary")
    print(f"{'='*80}")
    print(f"{'EB':>12} {'CR':>8} {'Far PSNR':>10} {'Mid PSNR':>10} "
          f"{'Near PSNR':>10} {'Avg mPSNR':>10}")
    print("-" * 64)
    for r in all_results:
        far = r["eval"].get("eval_far", {}).get("masked_psnr", 0) or 0
        mid = r["eval"].get("eval_mid", {}).get("masked_psnr", 0) or 0
        near = r["eval"].get("eval_near", {}).get("masked_psnr", 0) or 0
        vals = [v for v in [far, mid, near] if v > 0]
        avg = sum(vals) / len(vals) if vals else 0
        print(f"{r['eb']:>12.4e} {r['cr']:>8.2f} {far:>10.2f} "
              f"{mid:>10.2f} {near:>10.2f} {avg:>10.2f}")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
