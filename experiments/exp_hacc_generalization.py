#!/usr/bin/env python3
"""EXP-HACC: Generalization across spatial regions.

Downloads and partitions the 1-billion-particle HACC snapshot into spatial
blocks (~250M particles each, comparable to our 280M training set), then runs
the standard E25 single-block training pipeline on each block.

Demonstrates that our method generalizes to different spatial regions of the
same simulation, not just the single sub-volume used in other experiments.

Usage:
    python -m experiments.exp_hacc_generalization [--gpu 0] [--num_blocks 4]
"""

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from experiments.common import (
    PARTICLEGS_ROOT, REPO_ROOT, RUNS_DIR,
    PYTHON_BIN, PVBATCH_BIN, PREPARE_SCRIPT, GENERATE_SCRIPT,
    NUM_INIT_PLY, EVAL_DATASETS, DEFAULT_VIZ_PARAMS,
    run_cmd, pvbatch_cmd, get_pvbatch_egl_args,
    generate_training_data, generate_mix_training_data,
    prepare_stage_data_dir, run_stage_training, find_checkpoint,
    evaluate_model, get_model_stats, compute_psnr_dirs,
    save_results, print_table, base_parser,
)
from experiments.exp1_rate_distortion import E25_STAGES

# ── Data paths ───────────────────────────────────────────────────────────

HACC_BIG_TAR = PARTICLEGS_ROOT / "data" / "hacc_big.tar.gz"
HACC_BIG_DIR = PARTICLEGS_ROOT / "data" / "hacc_big"
HACC_BIG_URL = ("https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/"
                "Data-Reduction-Repo/raw-data/EXASKY/HACC/"
                "EXASKY-HACC-data-big-size.tar.gz")


# ── Data preparation ─────────────────────────────────────────────────────

def ensure_big_hacc_data():
    """Download and extract the 1B-particle HACC dataset if needed.

    Returns paths to xx.f32, yy.f32, zz.f32.
    """
    raw_x = HACC_BIG_DIR / "xx.f32"
    raw_y = HACC_BIG_DIR / "yy.f32"
    raw_z = HACC_BIG_DIR / "zz.f32"

    if raw_x.exists() and raw_y.exists() and raw_z.exists():
        print(f"[HACC-big] Raw data exists: {HACC_BIG_DIR}")
        return raw_x, raw_y, raw_z

    # Download
    if not HACC_BIG_TAR.exists():
        print(f"[HACC-big] Downloading (~19 GB)...")
        run_cmd(["wget", "-q", "--show-progress", "-O", str(HACC_BIG_TAR),
                 HACC_BIG_URL], timeout=7200)
    else:
        print(f"[HACC-big] Tarball exists: {HACC_BIG_TAR}")

    # Extract
    print(f"[HACC-big] Extracting...")
    HACC_BIG_DIR.mkdir(parents=True, exist_ok=True)
    run_cmd(["tar", "xzf", str(HACC_BIG_TAR),
             "-C", str(HACC_BIG_DIR), "--strip-components=1"])

    assert raw_x.exists(), f"Expected {raw_x} after extraction"
    print(f"[HACC-big] Extracted: {raw_x.stat().st_size / 1e9:.1f} GB per axis")
    return raw_x, raw_y, raw_z


def partition_hacc(raw_x, raw_y, raw_z, num_blocks, output_dir):
    """KD-tree partition into spatial blocks.

    Returns list of dicts: [{id, num_particles, raw_dir, bounds}, ...].
    """
    info_path = output_dir / "partition_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        if info["num_blocks"] == num_blocks:
            print(f"[Partition] Existing partition: {num_blocks} blocks")
            blocks = []
            for b in info["blocks"]:
                bdir = output_dir / f"block_{b['id']}"
                blocks.append({
                    "id": b["id"],
                    "num_particles": b["num_particles"],
                    "raw_dir": bdir,
                    "bounds": b["bounds"],
                })
            return blocks

    print(f"[Partition] Loading 1B-particle data...")
    x = np.fromfile(str(raw_x), dtype=np.float32)
    y = np.fromfile(str(raw_y), dtype=np.float32)
    z = np.fromfile(str(raw_z), dtype=np.float32)
    total = len(x)
    print(f"[Partition] Total: {total:,} particles")
    print(f"[Partition] Splitting into {num_blocks} blocks via KD-tree...")

    coords = np.stack([x, y, z], axis=1)

    # KD-tree split (inline to avoid loading partition.py as module)
    leaves = [np.arange(total)]
    while len(leaves) < num_blocks:
        new_leaves = []
        for idx_arr in leaves:
            bc = coords[idx_arr]
            ranges = bc.max(axis=0) - bc.min(axis=0)
            axis = int(np.argmax(ranges))
            mid = len(idx_arr) // 2
            order = np.argpartition(bc[:, axis], mid)
            new_leaves.append(idx_arr[order[:mid]])
            new_leaves.append(idx_arr[order[mid:]])
        leaves = new_leaves

    output_dir.mkdir(parents=True, exist_ok=True)
    blocks = []
    block_meta = []
    for i, indices in enumerate(leaves):
        bdir = output_dir / f"block_{i}"
        bdir.mkdir(parents=True, exist_ok=True)
        bx, by, bz = x[indices], y[indices], z[indices]
        bx.tofile(str(bdir / "xx.f32"))
        by.tofile(str(bdir / "yy.f32"))
        bz.tofile(str(bdir / "zz.f32"))
        bounds = {
            "x": [float(bx.min()), float(bx.max())],
            "y": [float(by.min()), float(by.max())],
            "z": [float(bz.min()), float(bz.max())],
        }
        blocks.append({
            "id": i,
            "num_particles": len(indices),
            "raw_dir": bdir,
            "bounds": bounds,
        })
        block_meta.append({
            "id": i,
            "num_particles": len(indices),
            "bounds": bounds,
        })
        print(f"  Block {i}: {len(indices):,} particles  "
              f"x=[{bounds['x'][0]:.1f},{bounds['x'][1]:.1f}]  "
              f"y=[{bounds['y'][0]:.1f},{bounds['y'][1]:.1f}]  "
              f"z=[{bounds['z'][0]:.1f},{bounds['z'][1]:.1f}]")

    info = {"num_blocks": num_blocks, "method": "kdtree",
            "total_particles": total, "blocks": block_meta}
    info_path.write_text(json.dumps(info, indent=2))
    # Free memory
    del x, y, z, coords, leaves
    return blocks


# ── Per-block shared data ────────────────────────────────────────────────

def ensure_block_shared_data(block, shared_dir, gpu=0):
    """Prepare VTP, normalization, eval GT images, PLY for one block.

    Mirrors ensure_shared_data() from common.py but for a custom raw data dir.
    """
    shared_dir = Path(shared_dir)
    shared_dir.mkdir(parents=True, exist_ok=True)
    logs = shared_dir / "logs"
    logs.mkdir(exist_ok=True)

    raw_dir = block["raw_dir"]
    n_particles = block["num_particles"]

    vtp_path = shared_dir / "particles.vtp"
    norm_path = shared_dir / "normalization.json"
    ply_path = shared_dir / "points3d.ply"

    # Phase 1: VTP
    if not vtp_path.exists():
        print(f"\n  [Block {block['id']}] Creating VTP ({n_particles:,} particles)...")
        run_cmd(
            pvbatch_cmd(PREPARE_SCRIPT,
                        "--raw_x", raw_dir / "xx.f32",
                        "--raw_y", raw_dir / "yy.f32",
                        "--raw_z", raw_dir / "zz.f32",
                        "--output_dir", shared_dir,
                        "--num_points_raw", "0",
                        "--skip_images"),
            log_path=logs / "create_vtp.log")
    else:
        print(f"  [Block {block['id']}] VTP exists")

    # Phase 2: Normalization + eval GT images
    eval_dirs = {}
    first_eval = EVAL_DATASETS[0]
    first_eval_dir = shared_dir / first_eval["subdir"] / "data"

    if not norm_path.exists():
        print(f"  [Block {block['id']}] Generating normalization + first eval images...")
        _gen_eval(vtp_path, first_eval_dir, first_eval["orbit_radii"],
                  None, logs, first_eval["id"])
        first_norm = first_eval_dir / "normalization.json"
        if first_norm.exists():
            shutil.copy2(first_norm, norm_path)
    else:
        print(f"  [Block {block['id']}] Normalization exists")

    for evd in EVAL_DATASETS:
        ed = shared_dir / evd["subdir"] / "data"
        eval_dirs[evd["id"]] = ed
        img_dir = ed / "images"
        if img_dir.exists() and len(list(img_dir.glob("*.png"))) >= 80:
            print(f"  [Block {block['id']}] Eval images exist: {evd['id']}")
            continue
        print(f"  [Block {block['id']}] Generating eval images: {evd['id']}...")
        _gen_eval(vtp_path, ed, evd["orbit_radii"], norm_path, logs, evd["id"])

    # Phase 3: PLY
    if not ply_path.exists():
        print(f"  [Block {block['id']}] Creating initial PLY...")
        run_cmd(
            [PYTHON_BIN, str(PREPARE_SCRIPT),
             "--raw_x", str(raw_dir / "xx.f32"),
             "--raw_y", str(raw_dir / "yy.f32"),
             "--raw_z", str(raw_dir / "zz.f32"),
             "--output_dir", str(shared_dir),
             "--only_ply", "--num_points_ply", str(NUM_INIT_PLY)],
            log_path=logs / "create_ply.log")
    else:
        print(f"  [Block {block['id']}] PLY exists")

    return {
        "vtp": vtp_path,
        "normalization": norm_path,
        "ply": ply_path,
        "eval_dirs": eval_dirs,
    }


def _gen_eval(vtp_path, output_dir, orbit_radii, norm_path, logs_dir, eval_id):
    """Generate eval GT images for one orbit (same as common._generate_eval_images)."""
    args = [
        "--vtp_path", str(vtp_path),
        "--output_dir", str(output_dir),
        "--camera_strategy", "multi_orbit",
        "--orbit_radii", orbit_radii,
        "--num_frames", "80",
        "--width", "1920", "--height", "1080",
        "--train_ratio", "1.0",
        "--split_seed", "42",
    ]
    for k, v in DEFAULT_VIZ_PARAMS.items():
        args += [f"--{k}", v]
    if norm_path and Path(norm_path).exists():
        args += ["--normalization_path", str(norm_path)]
    run_cmd(pvbatch_cmd(GENERATE_SCRIPT, *args),
            log_path=logs_dir / f"generate_{eval_id}.log")


# ── Per-block training ───────────────────────────────────────────────────

def train_block_e25(block, shared_data, block_run_dir, gpu=0):
    """Run E25 3-stage progressive training for one block.

    Returns (model_dir, iteration).
    """
    e25_dir = block_run_dir / "e25"
    logs = e25_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    vtp_path = shared_data["vtp"]
    norm_path = shared_data["normalization"]
    ply_path = shared_data["ply"]

    prev_checkpoint = None
    final_model_dir = None
    final_iteration = None

    for i, stage in enumerate(E25_STAGES):
        stage_name = stage["name"]
        stage_dir = e25_dir / f"{i:02d}_{stage_name}"
        data_dir = stage_dir / "data"
        model_dir = stage_dir / "model"

        print(f"\n  --- Block {block['id']} Stage {i+1}/{len(E25_STAGES)}: {stage_name} ---")

        existing_chk = find_checkpoint(model_dir)
        if existing_chk:
            print(f"    [Skip] Checkpoint exists: {existing_chk}")
            prev_checkpoint = existing_chk
            final_model_dir = model_dir
            final_iteration = int(existing_chk.stem.replace("chkpnt", ""))
            continue

        if stage.get("camera") == "mix":
            mc = stage["mix_cfg"]
            generate_mix_training_data(
                vtp_path, data_dir, norm_path,
                mc["ext_orbit_radii"], mc["ext_num_frames"], mc["ext_seed"],
                mc["int_num_frames"], mc["int_seed"], mc["int_bounds_scale"],
                mc["width"], mc["height"], mc["mix_ratio"],
                logs, stage_name)
        else:
            cam = stage["camera"]
            generate_training_data(
                vtp_path, data_dir, norm_path, cam["strategy"],
                cam["orbit_radii"], cam["num_frames"], cam["width"], cam["height"],
                logs, stage_name, viz_seed=cam.get("viz_seed", "142"))

        prepare_stage_data_dir(data_dir, ply_path, norm_path)

        final_iteration = run_stage_training(
            data_dir, model_dir, stage["train"], logs, f"train_{stage_name}",
            start_checkpoint=prev_checkpoint, init_iterations=10, gpu=gpu)

        prev_checkpoint = find_checkpoint(model_dir, final_iteration)
        final_model_dir = model_dir

        # Cleanup training images to save disk
        img_dir = data_dir / "images"
        if img_dir.exists():
            shutil.rmtree(img_dir)
            print(f"    Cleaned up training images")

    return final_model_dir, final_iteration


# ── Main experiment ──────────────────────────────────────────────────────

def run_hacc_generalization(num_blocks=4, gpu=0, output_dir=None,
                            skip_data_prep=False, only_block=None):
    """Run the full HACC generalization experiment."""
    output_dir = Path(output_dir) if output_dir else RUNS_DIR / "exp_hacc"
    output_dir.mkdir(parents=True, exist_ok=True)
    partition_dir = HACC_BIG_DIR / f"partitions_{num_blocks}"

    t0 = time.time()

    # Step 1: Download + extract
    if not skip_data_prep:
        raw_x, raw_y, raw_z = ensure_big_hacc_data()
    else:
        raw_x = HACC_BIG_DIR / "xx.f32"
        raw_y = HACC_BIG_DIR / "yy.f32"
        raw_z = HACC_BIG_DIR / "zz.f32"

    # Step 2: Partition
    blocks = partition_hacc(raw_x, raw_y, raw_z, num_blocks, partition_dir)

    # Step 3: Train and evaluate each block
    # Load existing results to avoid overwriting other blocks' results
    results_path = output_dir / "results.json"
    if results_path.exists():
        all_results = json.loads(results_path.read_text())
    else:
        all_results = {}
    for block in blocks:
        bid = block["id"]
        if only_block is not None and bid != only_block:
            print(f"\n[Skip] Block {bid} (--only_block {only_block})")
            continue

        print(f"\n{'='*70}")
        print(f"Block {bid}: {block['num_particles']:,} particles")
        print(f"{'='*70}")

        block_shared = output_dir / f"block_{bid}" / "shared"
        block_run = output_dir / f"block_{bid}"

        # Prepare shared data for this block
        shared_data = ensure_block_shared_data(block, block_shared, gpu=gpu)

        # Train E25
        model_dir, iteration = train_block_e25(block, shared_data, block_run, gpu=gpu)

        # Evaluate
        print(f"\n  Evaluating block {bid} model (iteration {iteration})...")
        last_train = E25_STAGES[-1]["train"]
        eval_results = evaluate_model(
            model_dir, iteration, shared_data,
            block_run / "e25" / "logs", gpu=gpu,
            factor_delta_opacity=last_train["factor_delta_opacity"],
            factor_delta_scale=last_train["factor_delta_scale"],
            min_opacity_clamp=last_train["min_opacity_clamp"])

        stats = get_model_stats(model_dir, iteration)
        raw_size = block["num_particles"] * 3 * 4  # 3 axes × float32
        cr = raw_size / (stats["size_mb"] * 1024 * 1024) if stats["size_mb"] > 0 else 0

        result = {
            "block_id": bid,
            "num_particles": block["num_particles"],
            "bounds": block["bounds"],
            "num_gaussians": stats["num_gaussians"],
            "size_mb": round(stats["size_mb"], 1),
            "cr": round(cr, 0),
            "avg_psnr": eval_results["avg"]["psnr"],
            "avg_masked_psnr": eval_results["avg"]["masked_psnr"],
            "eval": eval_results,
            "model_dir": str(model_dir),
            "iteration": iteration,
        }
        all_results[f"block_{bid}"] = result

        mpsnr = result.get("avg_masked_psnr")
        print(f"\n  Block {bid}: "
              + (f"{mpsnr:.2f} dB masked PSNR, " if mpsnr else "N/A, ")
              + f"{result['size_mb']} MB, {result['num_gaussians']} Gaussians, "
              + f"~{result['cr']:.0f}x CR")

        # Save intermediate results
        save_results(all_results, output_dir / "results.json")

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"HACC Generalization Experiment Complete ({elapsed/60:.1f} min)")
    print(f"{'='*70}")

    headers = ["Block", "Particles", "Masked PSNR", "Size (MB)", "Gaussians", "CR"]
    rows = []
    mpsnrs = []
    for key in sorted(all_results.keys()):
        r = all_results[key]
        mp = r.get("avg_masked_psnr")
        if mp:
            mpsnrs.append(mp)
        rows.append([
            f"Block {r['block_id']}",
            f"{r['num_particles']:,}",
            f"{mp:.2f}" if mp else "N/A",
            f"{r['size_mb']:.1f}",
            f"{r['num_gaussians']:,}",
            f"{r['cr']:.0f}x",
        ])
    print_table(headers, rows)

    if mpsnrs:
        print(f"\n  Mean masked PSNR: {sum(mpsnrs)/len(mpsnrs):.2f} dB")
        print(f"  Std:              {np.std(mpsnrs):.2f} dB")
        print(f"  Min:              {min(mpsnrs):.2f} dB")
        print(f"  Max:              {max(mpsnrs):.2f} dB")

    save_results(all_results, output_dir / "results.json")
    print(f"\nTotal time: {elapsed/60:.1f} min")
    return all_results


def main():
    parser = base_parser("EXP-HACC: Generalization across spatial regions")
    parser.add_argument("--num_blocks", type=int, default=4,
                        help="Number of spatial blocks (power of 2, default=4)")
    parser.add_argument("--only_block", type=int, default=None,
                        help="Only run a specific block (0-indexed)")
    args = parser.parse_args()

    run_hacc_generalization(
        num_blocks=args.num_blocks,
        gpu=args.gpu,
        output_dir=args.output_dir,
        skip_data_prep=args.skip_data_prep,
        only_block=args.only_block,
    )


if __name__ == "__main__":
    main()
