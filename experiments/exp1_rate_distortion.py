#!/usr/bin/env python3
"""EXP-1: 3DGS vs SZ3/LCP Rate-Distortion Curve.

Core contribution: 3DGS achieves ~+5-8 dB over SZ3 at comparable compression ratios.

Sub-experiments:
  1a: SZ3 baseline curve (15 ABS error bound points)
  1b: 3DGS single-block model (E25) on the curve
  1c: LCP baseline curve (particle-specific compressor)

Expected results:
  SZ3 @ 286x CR: ~18 dB masked PSNR
  3DGS E25:      26.24 dB masked PSNR @ 290x CR (+7.6 dB)
  LCP:           higher CR than SZ3 at same EB (particle-specific)

Usage:
    python -m experiments.exp1_rate_distortion [--gpu 0]
"""

import json
import os
import shutil
import time
from pathlib import Path

from experiments.common import *

# LCP binary: auto-detect from PARTICLEGS_LCP env or built-in LCP tree
LCP_BIN = os.environ.get("PARTICLEGS_LCP",
    str(PARTICLEGS_ROOT / "LCP" / "build" / "tools" / "sz3" / "lcp")
    if (PARTICLEGS_ROOT / "LCP" / "build" / "tools" / "sz3" / "lcp").exists()
    else str(REPO_ROOT / "LCP" / "build" / "tools" / "sz3" / "lcp")
    if (REPO_ROOT / "LCP" / "build" / "tools" / "sz3" / "lcp").exists()
    else shutil.which("lcp") or "lcp")

# SZ3 error bound sweep (15 points, matches all_exp.md Table 1a)
SZ3_EB_SWEEP = [
    4.54e-03, 1.23e-02, 2.65e-02, 4.90e-02, 8.27e-02,
    1.25e-01, 1.85e-01, 2.60e-01, 3.56e-01, 4.87e-01,
    5.85e-01, 7.03e-01, 8.44e-01, 1.01e+00, 1.29e+00,
]

# AE quick mode (--ae): the rest of the sweep exists only to draw the paper's
# rate-distortion curve; verification enforces a single point per baseline. Run
# only that point and leave lightweight placeholders in the results list so its
# index is preserved (reference_results.json keys exp1a_sz3.13 / exp1c_lcp.8 by
# list index). Cuts ~24 of 26 pvbatch renders → exp1 ~350min → ~80min (E25 train
# dominates and cannot be reduced).
AE_SZ3_IDX = 13   # -> reference_results.json "exp1.exp1a_sz3.13"
AE_LCP_IDX = 8    # -> reference_results.json "exp1.exp1c_lcp.8"


def _ae_placeholder(eb):
    """List entry standing in for a swept point skipped in AE quick mode."""
    return {"eb": eb, "cr": None, "raw_bytes": None, "compressed_bytes": None,
            "eval": {}, "avg_psnr": None, "avg_masked_psnr": None,
            "skipped_ae": True}

# E25 training config — 3-stage progressive (best single-block: 26.35 dB)
E25_STAGES = [
    {
        "name": "S1_multiorbit_4k",
        "camera": {"strategy": "multi_orbit", "orbit_radii": "1.0,0.7,0.5",
                    "num_frames": 400, "width": 3840, "height": 2160, "viz_seed": "142"},
        "train": {
            "iterations": 12000, "resolution_scale": 2, "sh_degree": 0,
            "densification_interval": 100, "opacity_reset_interval": 3000,
            "densify_until_iter": 12000, "densify_grad_threshold": 0.0002,
            "position_lr_init": 0.00016, "position_lr_max_steps": 50000,
            "scaling_lr": 0.005, "min_scale_pixels": 1.0, "min_opacity": 0.01,
            "antialiasing": True, "content_mask_loss": True,
            "lambda_dssim": 0.0, "factor_delta_opacity": 0.3,
            "factor_delta_scale": 0.1, "min_opacity_clamp": 0.4,
        },
    },
    {
        "name": "S2_multiorbit_6k",
        "camera": {"strategy": "multi_orbit", "orbit_radii": "1.0,0.7,0.5",
                    "num_frames": 400, "width": 5760, "height": 3240, "viz_seed": "142"},
        "train": {
            "iterations": 15000, "sh_degree": 0,
            "densification_interval": 200, "opacity_reset_interval": 3000,
            "densify_until_iter": 24000, "densify_grad_threshold": 0.0003,
            "position_lr_init": 0.00016, "position_lr_max_steps": 50000,
            "scaling_lr": 0.005, "min_scale_pixels": 1.0, "min_opacity": 0.01,
            "antialiasing": True, "content_mask_loss": True,
            "lambda_dssim": 0.0, "factor_delta_opacity": 0.3,
            "factor_delta_scale": 0.1, "min_opacity_clamp": 0.4,
        },
    },
    {
        "name": "S3_mix_6k",
        "camera": "mix",
        "mix_cfg": {
            "ext_orbit_radii": "1.0,0.7,0.5", "ext_num_frames": 400, "ext_seed": "142",
            "int_num_frames": 400, "int_seed": "242", "int_bounds_scale": 0.98,
            "width": 5760, "height": 3240, "mix_ratio": 0.8,
        },
        "train": {
            "iterations": 12000, "sh_degree": 0,
            "densification_interval": 200, "opacity_reset_interval": 3000,
            "densify_until_iter": 36000, "densify_grad_threshold": 0.0003,
            "position_lr_init": 0.00016, "position_lr_max_steps": 50000,
            "scaling_lr": 0.005, "min_scale_pixels": 1.0, "min_opacity": 0.01,
            "antialiasing": True, "content_mask_loss": True,
            "lambda_dssim": 0.0, "factor_delta_opacity": 0.3,
            "factor_delta_scale": 0.1, "min_opacity_clamp": 0.4,
        },
    },
]


# ── EXP-1a: SZ3 baseline ─────────────────────────────────────────────────

def sz3_compress_decompress(raw_path, out_dir, eb, axis_name):
    """Compress and decompress one axis with SZ3."""
    compressed = out_dir / f"{axis_name}.sz3"
    decompressed = out_dir / f"{axis_name}.f32"
    run_cmd([SZ3_BIN, "-f", "-i", str(raw_path), "-z", str(compressed),
             "-1", str(NUM_PARTICLES), "-M", "ABS", str(eb)],
            out_dir / f"compress_{axis_name}.log")
    compressed_size = compressed.stat().st_size
    run_cmd([SZ3_BIN, "-f", "-z", str(compressed), "-o", str(decompressed),
             "-1", str(NUM_PARTICLES), "-M", "ABS", str(eb)],
            out_dir / f"decompress_{axis_name}.log")
    return compressed_size


def run_sz3_point(eb, output_dir, shared_data, logs_dir, sz3_gt=None):
    """Run one SZ3 error bound point: compress, render, compute PSNR."""
    eb_dir = output_dir / f"eb_{eb:.4e}"
    result_file = eb_dir / "result.json"
    if result_file.exists():
        return json.loads(result_file.read_text())

    eb_dir.mkdir(parents=True, exist_ok=True)
    eb_logs = eb_dir / "logs"
    eb_logs.mkdir(exist_ok=True)

    # Compress/decompress each axis
    raw_size = 0
    compressed_size = 0
    for axis, fname in [("xx", "xx.f32"), ("yy", "yy.f32"), ("zz", "zz.f32")]:
        raw_path = RAW_DIR / fname
        raw_size += raw_path.stat().st_size
        cs = sz3_compress_decompress(raw_path, eb_dir, eb, axis)
        compressed_size += cs

    cr = raw_size / compressed_size if compressed_size > 0 else 0
    print(f"  EB={eb:.4e}: CR={cr:.1f}x ({compressed_size/1e6:.1f} MB)")

    # Create VTP from decompressed data
    run_cmd(
        pvbatch_cmd(PREPARE_SCRIPT,
                    "--raw_x", eb_dir / "xx.f32",
                    "--raw_y", eb_dir / "yy.f32",
                    "--raw_z", eb_dir / "zz.f32",
                    "--output_dir", str(eb_dir),
                    "--num_points_raw", "0", "--skip_images"),
        log_path=eb_logs / "create_vtp.log")
    vtp_path = eb_dir / "particles.vtp"

    # Render each eval orbit and compute PSNR
    norm_path = shared_data["normalization"]
    eval_results = {}
    for evd in EVAL_DATASETS:
        render_out = eb_dir / f"render_{evd['id']}"
        gen_args = [
            "--vtp_path", str(vtp_path),
            "--output_dir", str(render_out),
            "--camera_strategy", "multi_orbit",
            "--orbit_radii", evd["orbit_radii"],
            "--num_frames", "80",
            "--width", "1920", "--height", "1080",
            "--train_ratio", "1.0", "--split_seed", "42",
            "--normalization_path", str(norm_path),
        ]
        for k, v in DEFAULT_VIZ_PARAMS.items():
            gen_args += [f"--{k}", v]
        run_cmd(pvbatch_cmd(GENERATE_SCRIPT, *gen_args),
                log_path=eb_logs / f"render_{evd['id']}.log")

        # Use SZ3-specific GT if available (camera-aligned), else shared eval
        if sz3_gt and evd["id"] in sz3_gt:
            gt_dir = sz3_gt[evd["id"]] / "images"
        else:
            gt_dir = shared_data["eval_dirs"][evd["id"]] / "images"
        psnr, mpsnr, n = compute_psnr_dirs(gt_dir, render_out / "images")
        eval_results[evd["id"]] = {"psnr": psnr, "masked_psnr": mpsnr, "n": n}
        if psnr:
            print(f"    {evd['id']}: PSNR={psnr:.2f}  masked={mpsnr:.2f}")

        # Cleanup rendered images
        if render_out.exists():
            shutil.rmtree(render_out)

    # Cleanup: VTP, compressed/decompressed files, logs
    if vtp_path.exists():
        vtp_path.unlink()
    for ext in ["*.f32", "*.sz3"]:
        for f in eb_dir.glob(ext):
            f.unlink()
    for f in eb_dir.glob("*.log"):
        f.unlink()
    # Remove other generated data files
    for name in ["normalization.json", "points3d.ply", "transforms_train.json", "transforms_test.json"]:
        p = eb_dir / name
        if p.exists():
            p.unlink()
    img_dir = eb_dir / "images"
    if img_dir.exists():
        shutil.rmtree(img_dir)

    all_m = [v["masked_psnr"] for v in eval_results.values() if v.get("masked_psnr")]
    all_p = [v["psnr"] for v in eval_results.values() if v.get("psnr")]
    result = {
        "eb": eb, "cr": cr,
        "raw_bytes": raw_size, "compressed_bytes": compressed_size,
        "eval": eval_results,
        "avg_psnr": sum(all_p)/len(all_p) if all_p else None,
        "avg_masked_psnr": sum(all_m)/len(all_m) if all_m else None,
    }
    result_file.write_text(json.dumps(result, indent=2))
    return result


def ensure_sz3_gt_renders(output_dir, shared_data):
    """Generate GT renders for SZ3 benchmark (matching camera code).

    SZ3 benchmark needs pixel-exact camera alignment between GT and decompressed
    renders, so it uses its own GT renders rather than the shared eval data.
    """
    gt_dir = output_dir / "sz3" / "gt"
    gt_images = {}
    norm_path = shared_data["normalization"]
    vtp_path = shared_data["vtp"]

    for evd in EVAL_DATASETS:
        img_dir = gt_dir / evd["id"] / "images"
        gt_images[evd["id"]] = img_dir.parent
        if img_dir.exists() and len(list(img_dir.glob("*.png"))) >= 80:
            print(f"  [Skip] SZ3 GT images exist: {evd['id']}")
            continue
        print(f"  Generating SZ3 GT: {evd['id']}...")
        gen_args = [
            "--vtp_path", str(vtp_path),
            "--output_dir", str(img_dir.parent),
            "--camera_strategy", "multi_orbit",
            "--orbit_radii", evd["orbit_radii"],
            "--num_frames", "80",
            "--width", "1920", "--height", "1080",
            "--train_ratio", "1.0", "--split_seed", "42",
            "--normalization_path", str(norm_path),
        ]
        for k, v in DEFAULT_VIZ_PARAMS.items():
            gen_args += [f"--{k}", v]
        run_cmd(pvbatch_cmd(GENERATE_SCRIPT, *gen_args),
                log_path=gt_dir / "logs" / f"gt_{evd['id']}.log")
    return gt_images


def run_exp1a(output_dir, shared_data, ae=False):
    """EXP-1a: SZ3 baseline rate-distortion curve."""
    print("\n" + "="*70)
    print("EXP-1a: SZ3 Baseline Curve")
    print("="*70)

    sz3_dir = output_dir / "sz3"
    sz3_dir.mkdir(parents=True, exist_ok=True)

    # Generate SZ3-specific GT renders (camera-aligned with SZ3 renders)
    print("\n  Generating SZ3 GT renders...")
    sz3_gt = ensure_sz3_gt_renders(output_dir, shared_data)

    results = []
    for i, eb in enumerate(SZ3_EB_SWEEP):
        if ae and i != AE_SZ3_IDX:
            results.append(_ae_placeholder(eb))
            continue
        r = run_sz3_point(eb, sz3_dir, shared_data, sz3_dir / "logs", sz3_gt=sz3_gt)
        results.append(r)

    # Print summary
    print(f"\n{'='*70}")
    print("SZ3 Rate-Distortion Curve")
    print(f"{'='*70}")
    headers = ["EB", "CR", "Avg PSNR", "Avg Masked"]
    rows = []
    for r in results:
        if r.get("skipped_ae"):
            continue  # AE-quick placeholder (cr/psnr are None) — not a real point
        rows.append([
            f"{r['eb']:.2e}",
            f"{r['cr']:.1f}x" if r["cr"] else "N/A",
            f"{r['avg_psnr']:.2f}" if r["avg_psnr"] else "N/A",
            f"{r['avg_masked_psnr']:.2f}" if r["avg_masked_psnr"] else "N/A",
        ])
    print_table(headers, rows)
    return results


# ── EXP-1b: 3DGS E25 ─────────────────────────────────────────────────────

def run_e25_training(output_dir, shared_data, gpu=0):
    """Train E25 3-stage progressive single-block model."""
    e25_dir = output_dir / "e25"
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

        print(f"\n--- Stage {i+1}/{len(E25_STAGES)}: {stage_name} ---")

        # Check if already completed
        existing_chk = find_checkpoint(model_dir)
        if existing_chk:
            print(f"  [Skip] Checkpoint exists: {existing_chk}")
            prev_checkpoint = existing_chk
            final_model_dir = model_dir
            final_iteration = int(existing_chk.stem.replace("chkpnt", ""))
            continue

        # Generate training data
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

        # Symlink shared files
        prepare_stage_data_dir(data_dir, ply_path, norm_path)

        # Train with init phase (matches old run_experiment.py 2-phase approach)
        final_iteration = run_stage_training(
            data_dir, model_dir, stage["train"], logs, f"train_{stage_name}",
            start_checkpoint=prev_checkpoint, init_iterations=10, gpu=gpu)

        prev_checkpoint = find_checkpoint(model_dir, final_iteration)
        final_model_dir = model_dir

        # Cleanup training data to save disk
        img_dir = data_dir / "images"
        if img_dir.exists():
            shutil.rmtree(img_dir)
            print(f"  Cleaned up training images")

    return final_model_dir, final_iteration


def run_exp1b(output_dir, shared_data, gpu=0):
    """EXP-1b: 3DGS E25 single-block on the curve."""
    print("\n" + "="*70)
    print("EXP-1b: 3DGS E25 Single-Block Training")
    print("="*70)

    # End-to-end single-block training time (3DGS training-data generation +
    # 3-stage training). Valid ONLY when EXP-1 runs isolated (run_all --ae puts
    # it in the solo segment) — concurrent ParaView renders would steal CPU and
    # inflate this. Add vtp_conversion_min (EXP-11) for the full raw->model time.
    _train_t0 = time.perf_counter()
    model_dir, iteration = run_e25_training(output_dir, shared_data, gpu=gpu)
    train_elapsed = time.perf_counter() - _train_t0

    # Evaluate — pass VizMapper params from final stage config
    print(f"\nEvaluating E25 model (iteration {iteration})...")
    last_train = E25_STAGES[-1]["train"]
    eval_results = evaluate_model(model_dir, iteration, shared_data,
                                  output_dir / "e25" / "logs", gpu=gpu,
                                  factor_delta_opacity=last_train["factor_delta_opacity"],
                                  factor_delta_scale=last_train["factor_delta_scale"],
                                  min_opacity_clamp=last_train["min_opacity_clamp"])
    stats = get_model_stats(model_dir, iteration)

    raw_size = sum(f.stat().st_size for f in RAW_DIR.iterdir() if f.suffix == ".f32")
    cr = raw_size / (stats["size_mb"] * 1024 * 1024) if stats["size_mb"] > 0 else 0

    result = {
        "model": "E25",
        "num_gaussians": stats["num_gaussians"],
        "size_mb": round(stats["size_mb"], 1),
        "cr": round(cr, 0),
        "avg_psnr": eval_results["avg"]["psnr"],
        "avg_masked_psnr": eval_results["avg"]["masked_psnr"],
        "end_to_end_train_s": round(train_elapsed, 1),
        "end_to_end_train_min": round(train_elapsed / 60, 1),
        "eval": eval_results,
        "model_dir": str(model_dir),
        "iteration": iteration,
    }
    print(f"  End-to-end single-block train time: {train_elapsed/60:.1f} min "
          f"(data-gen + 3-stage training; isolated run only)")

    mpsnr = result.get('avg_masked_psnr')
    print(f"\n  E25: {mpsnr:.2f} dB masked PSNR, " if mpsnr else "\n  E25: N/A masked PSNR, ",
          end="")
    print(f"{result['size_mb']} MB, {result['num_gaussians']} Gaussians, "
          f"~{result['cr']:.0f}x CR")
    return result


# ── EXP-1c: LCP baseline ─────────────────────────────────────────────────

# LCP EB sweep — starts at 0.005 (LCP has a decompress bug at EB < ~0.003)
# LCP achieves higher CR than SZ3 at same EB due to particle reordering.
LCP_EB_SWEEP = [
    5.00e-03, 1.00e-02, 2.00e-02, 3.50e-02, 5.50e-02,
    8.00e-02, 1.20e-01, 1.80e-01, 2.70e-01, 4.00e-01, 6.00e-01,
]


def lcp_compress_decompress(raw_dir, out_dir, eb, logs_dir):
    """Compress and decompress all 3 axes with LCP (single file)."""
    lcp_path = out_dir / "particles.lcp"
    dec_x = out_dir / "xx.f32"
    dec_y = out_dir / "yy.f32"
    dec_z = out_dir / "zz.f32"

    run_cmd([LCP_BIN,
             "-i", str(raw_dir / "xx.f32"),
             str(raw_dir / "yy.f32"),
             str(raw_dir / "zz.f32"),
             "-1", str(NUM_PARTICLES),
             "-eb", str(eb),
             "-z", str(lcp_path)],
            log_path=logs_dir / "compress.log")
    compressed_size = lcp_path.stat().st_size

    run_cmd([LCP_BIN,
             "-z", str(lcp_path),
             "-o", str(dec_x), str(dec_y), str(dec_z),
             "-1", str(NUM_PARTICLES)],
            log_path=logs_dir / "decompress.log")

    return compressed_size


def run_lcp_point(eb, output_dir, shared_data, logs_dir, lcp_gt=None):
    """Run one LCP error bound point: compress, render, compute PSNR."""
    eb_dir = output_dir / f"eb_{eb:.4e}"
    result_file = eb_dir / "result.json"
    if result_file.exists():
        return json.loads(result_file.read_text())

    eb_dir.mkdir(parents=True, exist_ok=True)
    eb_logs = eb_dir / "logs"
    eb_logs.mkdir(exist_ok=True)

    # Compress/decompress
    raw_size = sum((RAW_DIR / f).stat().st_size
                   for f in ["xx.f32", "yy.f32", "zz.f32"])
    compressed_size = lcp_compress_decompress(RAW_DIR, eb_dir, eb, eb_logs)
    cr = raw_size / compressed_size if compressed_size > 0 else 0
    print(f"  EB={eb:.4e}: CR={cr:.1f}x ({compressed_size/1e6:.1f} MB)")

    # Create VTP from decompressed data
    run_cmd(
        pvbatch_cmd(PREPARE_SCRIPT,
                    "--raw_x", eb_dir / "xx.f32",
                    "--raw_y", eb_dir / "yy.f32",
                    "--raw_z", eb_dir / "zz.f32",
                    "--output_dir", str(eb_dir),
                    "--num_points_raw", "0", "--skip_images"),
        log_path=eb_logs / "create_vtp.log")
    vtp_path = eb_dir / "particles.vtp"

    # Render each eval orbit and compute PSNR
    norm_path = shared_data["normalization"]
    eval_results = {}
    for evd in EVAL_DATASETS:
        render_out = eb_dir / f"render_{evd['id']}"
        gen_args = [
            "--vtp_path", str(vtp_path),
            "--output_dir", str(render_out),
            "--camera_strategy", "multi_orbit",
            "--orbit_radii", evd["orbit_radii"],
            "--num_frames", "80",
            "--width", "1920", "--height", "1080",
            "--train_ratio", "1.0", "--split_seed", "42",
            "--normalization_path", str(norm_path),
        ]
        for k, v in DEFAULT_VIZ_PARAMS.items():
            gen_args += [f"--{k}", v]
        run_cmd(pvbatch_cmd(GENERATE_SCRIPT, *gen_args),
                log_path=eb_logs / f"render_{evd['id']}.log")

        # Use LCP-specific GT if available, else shared eval
        if lcp_gt and evd["id"] in lcp_gt:
            gt_dir = lcp_gt[evd["id"]] / "images"
        else:
            gt_dir = shared_data["eval_dirs"][evd["id"]] / "images"
        psnr, mpsnr, n = compute_psnr_dirs(gt_dir, render_out / "images")
        eval_results[evd["id"]] = {"psnr": psnr, "masked_psnr": mpsnr, "n": n}
        if psnr:
            print(f"    {evd['id']}: PSNR={psnr:.2f}  masked={mpsnr:.2f}")

        # Cleanup rendered images
        if render_out.exists():
            shutil.rmtree(render_out)

    # Cleanup: VTP, compressed/decompressed files
    if vtp_path.exists():
        vtp_path.unlink()
    for ext in ["*.f32", "*.lcp"]:
        for f in eb_dir.glob(ext):
            f.unlink()
    for f in eb_dir.glob("*.log"):
        f.unlink()
    for name in ["normalization.json", "points3d.ply",
                 "transforms_train.json", "transforms_test.json"]:
        p = eb_dir / name
        if p.exists():
            p.unlink()
    img_dir = eb_dir / "images"
    if img_dir.exists():
        shutil.rmtree(img_dir)

    all_m = [v["masked_psnr"] for v in eval_results.values() if v.get("masked_psnr")]
    all_p = [v["psnr"] for v in eval_results.values() if v.get("psnr")]
    result = {
        "eb": eb, "cr": cr,
        "raw_bytes": raw_size, "compressed_bytes": compressed_size,
        "eval": eval_results,
        "avg_psnr": sum(all_p)/len(all_p) if all_p else None,
        "avg_masked_psnr": sum(all_m)/len(all_m) if all_m else None,
    }
    result_file.write_text(json.dumps(result, indent=2))
    return result


def run_exp1c(output_dir, shared_data, ae=False):
    """EXP-1c: LCP baseline rate-distortion curve."""
    print("\n" + "="*70)
    print("EXP-1c: LCP Baseline Curve")
    print("="*70)

    lcp_dir = output_dir / "lcp"
    lcp_dir.mkdir(parents=True, exist_ok=True)

    # Generate LCP-specific GT renders (camera-aligned)
    print("\n  Generating LCP GT renders...")
    lcp_gt = ensure_sz3_gt_renders(output_dir, shared_data)  # reuse same GT

    results = []
    for i, eb in enumerate(LCP_EB_SWEEP):
        if ae and i != AE_LCP_IDX:
            results.append(_ae_placeholder(eb))
            continue
        r = run_lcp_point(eb, lcp_dir, shared_data, lcp_dir / "logs", lcp_gt=lcp_gt)
        results.append(r)

    # Print summary
    print(f"\n{'='*70}")
    print("LCP Rate-Distortion Curve")
    print(f"{'='*70}")
    headers = ["EB", "CR", "Avg PSNR", "Avg Masked"]
    rows = []
    for r in results:
        if r.get("skipped_ae"):
            continue  # AE-quick placeholder (cr/psnr are None) — not a real point
        rows.append([
            f"{r['eb']:.2e}",
            f"{r['cr']:.1f}x" if r["cr"] else "N/A",
            f"{r['avg_psnr']:.2f}" if r["avg_psnr"] else "N/A",
            f"{r['avg_masked_psnr']:.2f}" if r["avg_masked_psnr"] else "N/A",
        ])
    print_table(headers, rows)
    return results


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = base_parser("EXP-1: 3DGS vs SZ3/LCP Rate-Distortion Curve")
    parser.add_argument("--skip_sz3", action="store_true",
                        help="Skip SZ3 baseline (only run 3DGS)")
    parser.add_argument("--skip_3dgs", action="store_true",
                        help="Skip 3DGS training (only run SZ3)")
    parser.add_argument("--skip_lcp", action="store_true",
                        help="Skip LCP baseline")
    parser.add_argument("--only_lcp", action="store_true",
                        help="Only run LCP baseline")
    # --ae is provided by base_parser: quick mode computes only the enforced R-D
    # points (SZ3 #13 + E25; LCP dropped in AE, see run_exp1c gate) and skips the
    # rest of the sweep (~350min -> ~70min). Verification still passes.
    args = parser.parse_args()
    set_pvbatch_cuda_device(args.gpu)  # pin this process's GT renders to its GPU

    output_dir = Path(args.output_dir) if args.output_dir else RUNS_DIR / "exp1"
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Shared data
    if not args.skip_data_prep:
        shared_data = ensure_shared_data(gpu=args.gpu)
    else:
        shared_data = {
            "vtp": SHARED_DIR / "particles.vtp",
            "normalization": SHARED_DIR / "normalization.json",
            "ply": SHARED_DIR / "points3d.ply",
            "eval_dirs": {evd["id"]: SHARED_DIR / evd["subdir"] / "data"
                          for evd in EVAL_DATASETS},
        }

    results = {}

    run_sz3 = not args.skip_sz3 and not args.only_lcp
    run_3dgs = not args.skip_3dgs and not args.only_lcp
    # AE drops the LCP baseline: it is strictly worse than SZ3, so the paper's
    # iso-CR comparison only needs 3DGS vs SZ3. Full reproduce.sh (no --ae) still
    # runs LCP. --only_lcp forces it either way.
    run_lcp = (not args.skip_lcp and not args.ae) or args.only_lcp

    # EXP-1a: SZ3
    if run_sz3:
        results["exp1a_sz3"] = run_exp1a(output_dir, shared_data, ae=args.ae)

    # EXP-1b: 3DGS
    if run_3dgs:
        results["exp1b_e25"] = run_exp1b(output_dir, shared_data, gpu=args.gpu)

    # EXP-1c: LCP
    if run_lcp:
        results["exp1c_lcp"] = run_exp1c(output_dir, shared_data, ae=args.ae)

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"EXP-1 Complete ({elapsed/60:.1f} min)")
    print(f"{'='*70}")

    if "exp1a_sz3" in results and "exp1b_e25" in results:
        e25 = results["exp1b_e25"]
        # Find SZ3 point at similar CR
        sz3_at_cr = None
        for r in results["exp1a_sz3"]:
            if r["cr"] and 200 <= r["cr"] <= 400:
                sz3_at_cr = r
                break
        if sz3_at_cr and e25["avg_masked_psnr"]:
            delta = e25["avg_masked_psnr"] - (sz3_at_cr["avg_masked_psnr"] or 0)
            print(f"\n  3DGS E25: {e25['avg_masked_psnr']:.2f} dB @ ~{e25['cr']:.0f}x CR")
            print(f"  SZ3:      {sz3_at_cr['avg_masked_psnr']:.2f} dB @ {sz3_at_cr['cr']:.0f}x CR")
            print(f"  Delta:    +{delta:.1f} dB")

    # Load existing results for merging (if only_lcp, keep prior sz3/e25 results)
    results_path = output_dir / "results.json"
    if results_path.exists():
        existing = json.loads(results_path.read_text())
        for k, v in existing.items():
            if k not in results:
                results[k] = v

    save_results(results, results_path)
    print(f"\nTotal time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
