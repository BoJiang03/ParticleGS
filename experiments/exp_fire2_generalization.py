#!/usr/bin/env python3
"""EXP-FIRE2: Cross-dataset generalization on FIRE-2 simulation.

Runs the standard E25 single-block training pipeline on the FIRE-2 galaxy
formation simulation data (~268M particles), using the exact same
hyperparameters as the HACC experiments. Demonstrates that our method
generalizes across different simulation codes and physical phenomena.

Usage:
    python -m experiments.exp_fire2_generalization [--gpu 0]
"""

import json
import shutil
import time
from pathlib import Path

import numpy as np

from experiments.common import (
    PARTICLEGS_ROOT, RUNS_DIR,
    PYTHON_BIN, PVBATCH_BIN, PREPARE_SCRIPT, GENERATE_SCRIPT,
    NUM_INIT_PLY, EVAL_DATASETS, DEFAULT_VIZ_PARAMS,
    run_cmd, pvbatch_cmd,
    generate_training_data, generate_mix_training_data,
    prepare_stage_data_dir, run_stage_training, find_checkpoint,
    evaluate_model, get_model_stats,
    save_results, print_table, base_parser,
)
from experiments.exp1_rate_distortion import E25_STAGES

# ── Data paths ───────────────────────────────────────────────────────────

FIRE2_RAW_DIR = PARTICLEGS_ROOT / "data" / "fire2_raw"
FROZEN_FIRE2_NORM = PARTICLEGS_ROOT / "data" / "fire2_normalization.json"

# The OLD reference run generated the initial PLY with a ParaView-build-
# dependent dynamic scale (1.572e-05), and shared/normalization.json was
# later overwritten with the frozen scale (1.022e-05 = current ParaView
# build). That scale mismatch is baked into the reference model (82k
# Gaussians vs 116k when initialized at the frozen scale). Override the
# PLY scale to reproduce the legacy initialization; 199,877/200,000
# sample points match bit-exact, the remaining 123 differ by 1 float32
# ULP due to rounding order.
LEGACY_FIRE2_PLY_SCALE = 1.5724429691e-05

# FIRE-2 domain is [0, 116960]^3 vs HACC's [0, 256]^3.
# Scale viz params so particles are rendered at the same relative size.
_DOMAIN_SCALE = 116960.0 / 256.0  # ~457x

FIRE2_VIZ_PARAMS = {
    "gaussian_radius": f"{0.01 * _DOMAIN_SCALE:.4f}",
    "opacity": "0.05",
    "viz_mode": "sampled",
    "viz_distribution": "beta",
    "viz_beta_concentration": "3.0",
    "radius_min": f"{0.0025 * _DOMAIN_SCALE:.4f}",
    "radius_max": f"{0.0175 * _DOMAIN_SCALE:.4f}",
    "opacity_min": "0.0125",
    "opacity_max": "0.0875",
}


# ── Shared data preparation ─────────────────────────────────────────────

def ensure_fire2_shared_data(shared_dir, gpu=0):
    """Prepare VTP, normalization, eval GT images, PLY for FIRE-2 data.

    Same pipeline as HACC blocks — mirrors ensure_block_shared_data().
    """
    shared_dir = Path(shared_dir)
    shared_dir.mkdir(parents=True, exist_ok=True)
    logs = shared_dir / "logs"
    logs.mkdir(exist_ok=True)

    raw_dir = FIRE2_RAW_DIR
    vtp_path = shared_dir / "particles.vtp"
    norm_path = shared_dir / "normalization.json"
    ply_path = shared_dir / "points3d.ply"

    # Phase 0: Copy frozen normalization before anything else so that
    # prepare_data.py --skip_images sees norm.json and skips regenerating it.
    if not norm_path.exists() and FROZEN_FIRE2_NORM.exists():
        print(f"\n  [FIRE-2] Using frozen normalization: {FROZEN_FIRE2_NORM}")
        shutil.copy2(FROZEN_FIRE2_NORM, norm_path)

    # Phase 1: VTP
    if not vtp_path.exists():
        print(f"\n  [FIRE-2] Creating VTP...")
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
        print(f"  [FIRE-2] VTP exists")

    # Phase 2: Normalization + eval GT images
    first_eval = EVAL_DATASETS[0]
    first_eval_dir = shared_dir / first_eval["subdir"] / "data"

    if not norm_path.exists():
        print(f"  [FIRE-2] Generating normalization + first eval images...")
        _gen_eval(vtp_path, first_eval_dir, first_eval["orbit_radii"],
                  None, logs, first_eval["id"])
        first_norm = first_eval_dir / "normalization.json"
        if first_norm.exists():
            shutil.copy2(first_norm, norm_path)
    else:
        print(f"  [FIRE-2] Normalization exists")

    eval_dirs = {}
    for evd in EVAL_DATASETS:
        ed = shared_dir / evd["subdir"] / "data"
        eval_dirs[evd["id"]] = ed
        img_dir = ed / "images"
        if img_dir.exists() and len(list(img_dir.glob("*.png"))) >= 80:
            print(f"  [FIRE-2] Eval images exist: {evd['id']}")
            continue
        print(f"  [FIRE-2] Generating eval images: {evd['id']}...")
        _gen_eval(vtp_path, ed, evd["orbit_radii"], norm_path, logs, evd["id"])

    # Phase 3: PLY. Pass the legacy scale override so the initial PLY is
    # reproduced at the same scale the OLD run used (see
    # LEGACY_FIRE2_PLY_SCALE docstring above).
    if not ply_path.exists():
        print(f"  [FIRE-2] Creating initial PLY (legacy scale {LEGACY_FIRE2_PLY_SCALE:.6e})...")
        run_cmd(
            [PYTHON_BIN, str(PREPARE_SCRIPT),
             "--raw_x", str(raw_dir / "xx.f32"),
             "--raw_y", str(raw_dir / "yy.f32"),
             "--raw_z", str(raw_dir / "zz.f32"),
             "--output_dir", str(shared_dir),
             "--only_ply", "--num_points_ply", str(NUM_INIT_PLY),
             "--ply_scale_override", f"{LEGACY_FIRE2_PLY_SCALE:.10e}"],
            log_path=logs / "create_ply.log")
    else:
        print(f"  [FIRE-2] PLY exists")

    return {
        "vtp": vtp_path,
        "normalization": norm_path,
        "ply": ply_path,
        "eval_dirs": eval_dirs,
    }


def _gen_eval(vtp_path, output_dir, orbit_radii, norm_path, logs_dir, eval_id):
    """Generate eval GT images for one orbit."""
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
    for k, v in FIRE2_VIZ_PARAMS.items():
        args += [f"--{k}", v]
    if norm_path and Path(norm_path).exists():
        args += ["--normalization_path", str(norm_path)]
    run_cmd(pvbatch_cmd(GENERATE_SCRIPT, *args),
            log_path=logs_dir / f"generate_{eval_id}.log")


# ── Training ─────────────────────────────────────────────────────────────

def train_fire2_e25(shared_data, run_dir, gpu=0):
    """Run E25 3-stage progressive training on FIRE-2 data.

    Returns (model_dir, iteration).
    """
    e25_dir = run_dir / "e25"
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

        print(f"\n  --- Stage {i+1}/{len(E25_STAGES)}: {stage_name} ---")

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
                logs, stage_name, viz_params=FIRE2_VIZ_PARAMS)
        else:
            cam = stage["camera"]
            generate_training_data(
                vtp_path, data_dir, norm_path, cam["strategy"],
                cam["orbit_radii"], cam["num_frames"], cam["width"], cam["height"],
                logs, stage_name, viz_seed=cam.get("viz_seed", "142"),
                viz_params=FIRE2_VIZ_PARAMS)

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

def run_fire2_experiment(gpu=0, output_dir=None, skip_data_prep=False):
    """Run the FIRE-2 generalization experiment."""
    output_dir = Path(output_dir) if output_dir else RUNS_DIR / "exp_fire2"
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Check raw data exists
    for axis in ["xx", "yy", "zz"]:
        f = FIRE2_RAW_DIR / f"{axis}.f32"
        assert f.exists(), f"Missing raw data: {f}"

    n_particles = FIRE2_RAW_DIR.stat  # just use file size
    raw_x = FIRE2_RAW_DIR / "xx.f32"
    n_particles = raw_x.stat().st_size // 4
    raw_size_bytes = n_particles * 3 * 4

    print(f"FIRE-2 dataset: {n_particles:,} particles ({raw_size_bytes/1e9:.2f} GB)")

    # Step 1: Prepare shared data (VTP, normalization, eval GT, PLY)
    shared_dir = output_dir / "shared"
    if not skip_data_prep:
        shared_data = ensure_fire2_shared_data(shared_dir, gpu=gpu)
    else:
        shared_data = {
            "vtp": shared_dir / "particles.vtp",
            "normalization": shared_dir / "normalization.json",
            "ply": shared_dir / "points3d.ply",
            "eval_dirs": {evd["id"]: shared_dir / evd["subdir"] / "data"
                          for evd in EVAL_DATASETS},
        }

    # Step 2: Train E25
    model_dir, iteration = train_fire2_e25(shared_data, output_dir, gpu=gpu)

    # Step 3: Evaluate
    print(f"\n  Evaluating model (iteration {iteration})...")
    last_train = E25_STAGES[-1]["train"]
    eval_results = evaluate_model(
        model_dir, iteration, shared_data,
        output_dir / "e25" / "logs", gpu=gpu,
        factor_delta_opacity=last_train["factor_delta_opacity"],
        factor_delta_scale=last_train["factor_delta_scale"],
        min_opacity_clamp=last_train["min_opacity_clamp"])

    stats = get_model_stats(model_dir, iteration)
    cr = raw_size_bytes / (stats["size_mb"] * 1024 * 1024) if stats["size_mb"] > 0 else 0

    result = {
        "dataset": "fire2",
        "num_particles": n_particles,
        "num_gaussians": stats["num_gaussians"],
        "size_mb": round(stats["size_mb"], 1),
        "cr": round(cr, 0),
        "avg_psnr": eval_results["avg"]["psnr"],
        "avg_masked_psnr": eval_results["avg"]["masked_psnr"],
        "eval": eval_results,
        "model_dir": str(model_dir),
        "iteration": iteration,
    }

    save_results(result, output_dir / "results.json")

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"FIRE-2 Experiment Complete ({elapsed/60:.1f} min)")
    print(f"{'='*70}")

    mpsnr = result.get("avg_masked_psnr")
    print(f"  Particles:    {n_particles:,}")
    print(f"  Masked PSNR:  {mpsnr:.2f} dB" if mpsnr else "  Masked PSNR:  N/A")
    print(f"  Model size:   {result['size_mb']:.1f} MB")
    print(f"  Gaussians:    {result['num_gaussians']:,}")
    print(f"  CR:           ~{result['cr']:.0f}x")
    print(f"  Time:         {elapsed/60:.1f} min")

    return result


def main():
    parser = base_parser("EXP-FIRE2: Cross-dataset generalization")
    args = parser.parse_args()

    run_fire2_experiment(
        gpu=args.gpu,
        output_dir=args.output_dir,
        skip_data_prep=args.skip_data_prep,
    )


if __name__ == "__main__":
    main()
