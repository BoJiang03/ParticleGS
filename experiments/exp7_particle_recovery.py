#!/usr/bin/env python3
"""EXP-7: 3DGS Particle Recovery (GMM Sampling).

Demonstrates 3DGS-as-GMM can recover large-scale particle statistical properties.

Sub-experiments:
  7a: Recovery algorithm comparison (V0-V6 strategies, 10M particles)
  7b: Block count vs recovery quality (1-16 blocks, 280M particles)
  7c: Scale factor scan (1.0x-3.0x)

Usage:
    python -m experiments.exp7_particle_recovery [--gpu 0]
"""

import json
import time
from pathlib import Path

from experiments.common import *

RECOVER_SCRIPT = PARTICLEGS_ROOT / "recovery" / "recover_particles.py"
COMPARE_SCRIPT = PARTICLEGS_ROOT / "recovery" / "compare_particles.py"


def run_recovery(model_ply, output_dir, num_particles, method="V0_baseline",
                 logs_dir=None):
    """Run particle recovery with a given method."""
    output_dir = Path(output_dir)
    compare_output = output_dir / "comparison"
    result_file = compare_output / "summary.json"
    if result_file.exists():
        return json.loads(result_file.read_text())

    output_dir.mkdir(parents=True, exist_ok=True)

    # recover_particles.py expects: --ply_path, --normalization, --output_dir, --method, --num_points
    norm_path = SHARED_DIR / "normalization.json"
    recover_cmd = [
        PYTHON_BIN, str(RECOVER_SCRIPT),
        "--ply_path", str(model_ply),
        "--normalization", str(norm_path),
        "--output_dir", str(output_dir),
        "--num_points", str(num_particles),
        "--method", method,
    ]
    run_cmd(recover_cmd, log_path=logs_dir / f"recover_{method}.log" if logs_dir else None)

    # compare_particles.py expects: --gt_dir, --rec_dir, --output
    # gt_dir should contain xx.f32, yy.f32, zz.f32
    # Don't pass --normalization: recovery already denormalized the particles
    compare_cmd = [
        PYTHON_BIN, str(COMPARE_SCRIPT),
        "--gt_dir", str(RAW_DIR),
        "--rec_dir", str(output_dir),
        "--output", str(compare_output),
    ]
    run_cmd(compare_cmd, log_path=logs_dir / f"compare_{method}.log" if logs_dir else None)

    if result_file.exists():
        return json.loads(result_file.read_text())
    return None


def run_exp7a(output_dir, shared_data, gpu):
    """EXP-7a: Recovery algorithm comparison (10M particles)."""
    print("\n" + "="*70)
    print("EXP-7a: Recovery Algorithm Comparison (10M particles)")
    print("="*70)

    # Need a trained model — use EXP-4 F16 or EXP-1 E25
    model_ply = None
    for candidate in [
        RUNS_DIR / "exp4" / "blocks_8" / "finetuned" / "model",
        RUNS_DIR / "exp1" / "e25" / "02_S3_mix_6k" / "model",
    ]:
        if candidate.exists():
            chk = find_checkpoint(candidate)
            if chk:
                iteration = int(chk.stem.replace("chkpnt", ""))
                ply = candidate / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
                if ply.exists():
                    model_ply = ply
                    break
    if not model_ply:
        print("ERROR: No trained model found. Run EXP-1 or EXP-4 first.")
        return None

    logs = output_dir / "7a" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    methods = ["V0_baseline", "V1_volume_weight", "V4_uniform_ellipsoid", "V6_density_field"]
    results = []

    for method in methods:
        print(f"\n  Method: {method}")
        r = run_recovery(model_ply, output_dir / "7a" / method,
                         num_particles=NUM_PARTICLES, method=method,
                         logs_dir=logs)
        if r:
            results.append({"method": method, **r})

    if results:
        headers = ["Method", "Density Corr", "NN Mean", "NN Ratio"]
        rows = [[r["method"],
                 f"{r.get('density_field', {}).get('correlation', 0):.3f}",
                 f"{r.get('nn_distance', {}).get('rec_mean', 0):.3f}",
                 f"{r.get('nn_distance', {}).get('rec_mean', 0)/0.665:.2f}x"]
                for r in results]
        print_table(headers, rows)
    return results


def run_exp7b(output_dir, shared_data, gpu):
    """EXP-7b: Block count vs recovery quality (280M particles)."""
    print("\n" + "="*70)
    print("EXP-7b: Block Count vs Recovery Quality (280M particles)")
    print("="*70)

    logs = output_dir / "7b" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    results = []

    for n_blocks in [1, 2, 4, 8, 16]:
        # Find model from EXP-4
        if n_blocks == 1:
            model_base = RUNS_DIR / "exp1" / "e25" / "02_S3_mix_6k" / "model"
        else:
            model_base = RUNS_DIR / "exp4" / f"blocks_{n_blocks}" / "finetuned" / "model"

        if not model_base.exists():
            print(f"  [Skip] No model for {n_blocks} blocks")
            continue

        chk = find_checkpoint(model_base)
        if not chk:
            continue
        iteration = int(chk.stem.replace("chkpnt", ""))
        ply = model_base / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        if not ply.exists():
            continue

        print(f"\n  {n_blocks} blocks...")
        r = run_recovery(ply, output_dir / "7b" / f"blocks_{n_blocks}",
                         num_particles=NUM_PARTICLES, method="V0_baseline",
                         logs_dir=logs)
        if r:
            results.append({"n_blocks": n_blocks, **r})

    if results:
        headers = ["Blocks", "Density Corr", "NN Mean", "NN Ratio"]
        rows = [[r["n_blocks"],
                 f"{r.get('density_field', {}).get('correlation', 0):.3f}",
                 f"{r.get('nn_distance', {}).get('rec_mean', 0):.3f}",
                 f"{r.get('nn_distance', {}).get('rec_mean', 0)/0.665:.2f}x"]
                for r in results]
        print_table(headers, rows)
    return results


def run_exp7c(output_dir, shared_data, gpu):
    """EXP-7c: Scale factor scan."""
    print("\n" + "="*70)
    print("EXP-7c: Scale Factor Scan (10M particles)")
    print("="*70)

    model_ply = None
    for candidate in [
        RUNS_DIR / "exp4" / "blocks_8" / "finetuned" / "model",
        RUNS_DIR / "exp1" / "e25" / "02_S3_mix_6k" / "model",
    ]:
        if candidate.exists():
            chk = find_checkpoint(candidate)
            if chk:
                iteration = int(chk.stem.replace("chkpnt", ""))
                ply = candidate / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
                if ply.exists():
                    model_ply = ply
                    break
    if not model_ply:
        print("ERROR: No trained model found.")
        return None

    logs = output_dir / "7c" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    results = []

    # Scale factor is encoded in method name (recover_particles.py convention)
    scale_methods = [
        (1.0, "V0_baseline"),
        (1.5, "V7_scale_1.5x"),
        (2.0, "V8_scale_2.0x"),
        (2.5, "V9_scale_2.5x"),
        (3.0, "V2_scale_3x"),
    ]
    for sf, method in scale_methods:
        print(f"\n  Scale factor: {sf}x (method: {method})")
        r = run_recovery(model_ply, output_dir / "7c" / f"sf_{sf}",
                         num_particles=NUM_PARTICLES, method=method,
                         logs_dir=logs)
        if r:
            results.append({"scale_factor": sf, **r})

    if results:
        headers = ["Scale", "Density Corr", "NN Mean", "NN Ratio"]
        rows = [[f"{r['scale_factor']}x",
                 f"{r.get('density_field', {}).get('correlation', 0):.3f}",
                 f"{r.get('nn_distance', {}).get('rec_mean', 0):.3f}",
                 f"{r.get('nn_distance', {}).get('rec_mean', 0)/0.665:.2f}x"]
                for r in results]
        print_table(headers, rows)
    return results


def main():
    parser = base_parser("EXP-7: Particle Recovery")
    parser.add_argument("--sub", type=str, default="all", help="a,b,c or all")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RUNS_DIR / "exp7"
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    shared_data = ensure_shared_data(gpu=args.gpu) if not args.skip_data_prep else get_shared_data_dict()

    results = {}
    subs = args.sub.split(",") if args.sub != "all" else ["a", "b", "c"]
    if "a" in subs: results["7a"] = run_exp7a(output_dir, shared_data, args.gpu)
    if "b" in subs: results["7b"] = run_exp7b(output_dir, shared_data, args.gpu)
    if "c" in subs: results["7c"] = run_exp7c(output_dir, shared_data, args.gpu)

    save_results(results, output_dir / "results.json")
    print(f"\nEXP-7 complete ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
