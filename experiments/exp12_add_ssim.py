#!/usr/bin/env python3
"""EXP-12: Add SSIM to rate-distortion results.

Re-runs SZ3/LCP sweeps and 3DGS model eval to compute SSIM.
Rendered images are discarded after SSIM computation (same as exp1).

Saves results to runs/exp1/ssim_results.json.

Usage:
    python -m experiments.exp12_add_ssim [--gpu 0] [--only_3dgs] [--only_sz3] [--only_lcp]
"""

import json
import os
import shutil
import time
from pathlib import Path

from experiments.common import *
from experiments.exp1_rate_distortion import (
    SZ3_EB_SWEEP, LCP_EB_SWEEP,
    sz3_compress_decompress, lcp_compress_decompress,
    ensure_sz3_gt_renders,
)

OUTPUT_FILE = "ssim_results.json"


def load_existing(output_dir):
    p = output_dir / OUTPUT_FILE
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_results(output_dir, results):
    p = output_dir / OUTPUT_FILE
    p.write_text(json.dumps(results, indent=2))
    print(f"  Saved → {p}")


# ── SZ3 SSIM sweep ──────────────────────────────────────────────────────

def run_sz3_ssim_point(eb, sz3_dir, shared_data, sz3_gt):
    """Run one SZ3 point: compress → decompress → render → compute SSIM."""
    eb_dir = sz3_dir / f"eb_{eb:.4e}"
    eb_dir.mkdir(parents=True, exist_ok=True)
    eb_logs = eb_dir / "logs"
    eb_logs.mkdir(exist_ok=True)

    # Compress/decompress each axis
    raw_size = 0
    compressed_size = 0
    for axis in ["xx", "yy", "zz"]:
        raw_path = RAW_DIR / f"{axis}.f32"
        raw_size += raw_path.stat().st_size
        cs = sz3_compress_decompress(raw_path, eb_dir, eb, axis)
        compressed_size += cs

    cr = raw_size / compressed_size if compressed_size > 0 else 0
    print(f"  EB={eb:.4e}: CR={cr:.1f}x")

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

    # Render each eval orbit and compute SSIM
    norm_path = shared_data["normalization"]
    ssim_per_orbit = {}
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

        gt_dir = sz3_gt[evd["id"]] / "images"
        ssim, n = compute_ssim_dirs(gt_dir, render_out / "images")
        ssim_per_orbit[evd["id"]] = {"ssim": ssim, "n": n}
        if ssim:
            print(f"    {evd['id']}: SSIM={ssim:.4f}")

        # Cleanup rendered images
        if render_out.exists():
            shutil.rmtree(render_out)

    # Cleanup
    for p in [vtp_path] + list(eb_dir.glob("*.f32")) + list(eb_dir.glob("*.sz3")) + \
             list(eb_dir.glob("*.log")):
        if p.exists():
            p.unlink()
    for name in ["normalization.json", "points3d.ply",
                 "transforms_train.json", "transforms_test.json"]:
        p = eb_dir / name
        if p.exists():
            p.unlink()
    img_dir = eb_dir / "images"
    if img_dir.exists():
        shutil.rmtree(img_dir)

    all_s = [v["ssim"] for v in ssim_per_orbit.values() if v.get("ssim")]
    return {
        "eb": eb, "cr": cr,
        "eval": ssim_per_orbit,
        "avg_ssim": sum(all_s) / len(all_s) if all_s else None,
    }


# ── LCP SSIM sweep ──────────────────────────────────────────────────────

def run_lcp_ssim_point(eb, lcp_dir, shared_data, lcp_gt):
    """Run one LCP point: compress → decompress → render → compute SSIM."""
    eb_dir = lcp_dir / f"eb_{eb:.4e}"
    eb_dir.mkdir(parents=True, exist_ok=True)
    eb_logs = eb_dir / "logs"
    eb_logs.mkdir(exist_ok=True)

    raw_size = sum((RAW_DIR / f).stat().st_size for f in ["xx.f32", "yy.f32", "zz.f32"])
    compressed_size = lcp_compress_decompress(RAW_DIR, eb_dir, eb, eb_logs)
    cr = raw_size / compressed_size if compressed_size > 0 else 0
    print(f"  EB={eb:.4e}: CR={cr:.1f}x")

    run_cmd(
        pvbatch_cmd(PREPARE_SCRIPT,
                    "--raw_x", eb_dir / "xx.f32",
                    "--raw_y", eb_dir / "yy.f32",
                    "--raw_z", eb_dir / "zz.f32",
                    "--output_dir", str(eb_dir),
                    "--num_points_raw", "0", "--skip_images"),
        log_path=eb_logs / "create_vtp.log")
    vtp_path = eb_dir / "particles.vtp"

    norm_path = shared_data["normalization"]
    ssim_per_orbit = {}
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

        gt_dir = lcp_gt[evd["id"]] / "images"
        ssim, n = compute_ssim_dirs(gt_dir, render_out / "images")
        ssim_per_orbit[evd["id"]] = {"ssim": ssim, "n": n}
        if ssim:
            print(f"    {evd['id']}: SSIM={ssim:.4f}")

        if render_out.exists():
            shutil.rmtree(render_out)

    for p in [vtp_path] + list(eb_dir.glob("*.f32")) + list(eb_dir.glob("*.lcp")) + \
             list(eb_dir.glob("*.log")):
        if p.exists():
            p.unlink()
    for name in ["normalization.json", "points3d.ply",
                 "transforms_train.json", "transforms_test.json"]:
        p = eb_dir / name
        if p.exists():
            p.unlink()
    img_dir = eb_dir / "images"
    if img_dir.exists():
        shutil.rmtree(img_dir)

    all_s = [v["ssim"] for v in ssim_per_orbit.values() if v.get("ssim")]
    return {
        "eb": eb, "cr": cr,
        "eval": ssim_per_orbit,
        "avg_ssim": sum(all_s) / len(all_s) if all_s else None,
    }


# ── 3DGS model SSIM ─────────────────────────────────────────────────────

def compute_3dgs_ssim(model_dir, iteration, shared_data, logs_dir, gpu=0,
                      factor_delta_opacity=0.3, factor_delta_scale=0.1,
                      min_opacity_clamp=0.4):
    """Render 3DGS model and compute SSIM against GT."""
    ssim_per_orbit = {}
    for evd in EVAL_DATASETS:
        eval_id = evd["id"]
        data_dir = shared_data["eval_dirs"][eval_id]
        gt_dir = data_dir / "images"

        eval_render_dir = Path(model_dir).parent / "ssim_renders" / eval_id
        eval_render_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Rendering {eval_id}...")
        run_render(model_dir, data_dir, iteration, logs_dir,
                   f"ssim_render_{eval_id}", gpu=gpu,
                   factor_delta_opacity=factor_delta_opacity,
                   factor_delta_scale=factor_delta_scale,
                   min_opacity_clamp=min_opacity_clamp,
                   min_scale_pixels=0.0,
                   extra_args=["--output_dir", str(eval_render_dir),
                               "--resolution", "1"])

        render_dir = eval_render_dir / "train" / f"ours_{iteration}" / "renders"
        ssim, n = compute_ssim_dirs(gt_dir, render_dir)
        ssim_per_orbit[eval_id] = {"ssim": ssim, "n": n}
        if ssim:
            print(f"    {eval_id}: SSIM={ssim:.4f}")

    # Cleanup
    ssim_root = Path(model_dir).parent / "ssim_renders"
    if ssim_root.exists():
        shutil.rmtree(ssim_root)

    all_s = [v["ssim"] for v in ssim_per_orbit.values() if v.get("ssim")]
    return {
        "eval": ssim_per_orbit,
        "avg_ssim": sum(all_s) / len(all_s) if all_s else None,
    }


# ── LCP GT renders (same camera alignment as SZ3 GT) ────────────────────

def ensure_lcp_gt_renders(output_dir, shared_data):
    """Generate LCP-specific GT renders (reuse SZ3 GT if available)."""
    # LCP uses same camera code as SZ3, so reuse SZ3 GT
    sz3_gt_dir = output_dir / "sz3" / "gt"
    if sz3_gt_dir.exists():
        gt_images = {}
        for evd in EVAL_DATASETS:
            gt_images[evd["id"]] = sz3_gt_dir / evd["id"]
        if all((d / "images").exists() for d in gt_images.values()):
            print("  [Reuse] LCP GT = SZ3 GT")
            return gt_images
    # Fallback: generate fresh
    return ensure_sz3_gt_renders(output_dir, shared_data)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = base_parser("EXP-12: Add SSIM to rate-distortion results")
    parser.add_argument("--only_3dgs", action="store_true")
    parser.add_argument("--only_sz3", action="store_true")
    parser.add_argument("--only_lcp", action="store_true")
    args = parser.parse_args()

    output_dir = RUNS_DIR / "exp1"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Shared data (skip prep — already exists from exp1)
    shared_data = {
        "vtp": SHARED_DIR / "particles.vtp",
        "normalization": SHARED_DIR / "normalization.json",
        "ply": SHARED_DIR / "points3d.ply",
        "eval_dirs": {evd["id"]: SHARED_DIR / evd["subdir"] / "data"
                      for evd in EVAL_DATASETS},
    }

    results = load_existing(output_dir)
    t0 = time.time()

    run_sz3 = not args.only_3dgs and not args.only_lcp
    run_lcp = not args.only_3dgs and not args.only_sz3
    run_3dgs = not args.only_sz3 and not args.only_lcp

    # ── SZ3 ──
    if run_sz3:
        print("\n" + "=" * 70)
        print("SZ3 SSIM Sweep")
        print("=" * 70)
        sz3_gt = ensure_sz3_gt_renders(output_dir, shared_data)
        sz3_results = results.get("sz3_ssim", [])
        done_ebs = {r["eb"] for r in sz3_results}
        for eb in SZ3_EB_SWEEP:
            if eb in done_ebs:
                print(f"  [Skip] EB={eb:.4e}")
                continue
            r = run_sz3_ssim_point(eb, output_dir / "sz3", shared_data, sz3_gt)
            sz3_results.append(r)
            results["sz3_ssim"] = sz3_results
            save_results(output_dir, results)
        print(f"\nSZ3 SSIM: {len(sz3_results)} points done")

    # ── LCP ──
    if run_lcp:
        print("\n" + "=" * 70)
        print("LCP SSIM Sweep")
        print("=" * 70)
        lcp_gt = ensure_lcp_gt_renders(output_dir, shared_data)
        lcp_results = results.get("lcp_ssim", [])
        done_ebs = {r["eb"] for r in lcp_results}
        for eb in LCP_EB_SWEEP:
            if eb in done_ebs:
                print(f"  [Skip] EB={eb:.4e}")
                continue
            r = run_lcp_ssim_point(eb, output_dir / "lcp", shared_data, lcp_gt)
            lcp_results.append(r)
            results["lcp_ssim"] = lcp_results
            save_results(output_dir, results)
        print(f"\nLCP SSIM: {len(lcp_results)} points done")

    # ── 3DGS models ──
    if run_3dgs:
        print("\n" + "=" * 70)
        print("3DGS Model SSIM")
        print("=" * 70)
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        # Single-block E25
        if "e25_ssim" not in results:
            print("\n  E25 single-block:")
            e25_model = output_dir / "e25" / "02_S3_mix_6k" / "model"
            r = compute_3dgs_ssim(e25_model, 39030, shared_data, logs_dir,
                                  gpu=args.gpu,
                                  factor_delta_opacity=0.3,
                                  factor_delta_scale=0.1,
                                  min_opacity_clamp=0.4)
            results["e25_ssim"] = r
            save_results(output_dir, results)
            print(f"  E25 avg SSIM: {r['avg_ssim']:.4f}")
        else:
            print(f"  [Skip] E25 SSIM: {results['e25_ssim']['avg_ssim']:.4f}")

        # N-block + FT (F16) — prefer 8, fall back to 4, 2
        ft_n = None
        ft_model = None
        for n in (8, 4, 2):
            candidate = RUNS_DIR / "exp4" / f"blocks_{n}" / "finetuned" / "model"
            if (candidate / "cfg_args").exists():
                ft_n = n
                ft_model = candidate
                break
        ssim_key = f"{ft_n}blk_ft_ssim" if ft_n else None
        if ft_n is None:
            print("\n  [Skip] No blocks_N FT model found for SSIM")
        elif ssim_key not in results:
            print(f"\n  {ft_n}-block + FT (F16):")
            r = compute_3dgs_ssim(ft_model, 60000, shared_data, logs_dir,
                                  gpu=args.gpu,
                                  factor_delta_opacity=0.8,
                                  factor_delta_scale=0.3,
                                  min_opacity_clamp=0.1)
            results[ssim_key] = r
            save_results(output_dir, results)
            print(f"  {ft_n}-blk+FT avg SSIM: {r['avg_ssim']:.4f}")
        else:
            print(f"  [Skip] {ft_n}-blk+FT SSIM: {results[ssim_key]['avg_ssim']:.4f}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"EXP-12 Complete ({elapsed / 60:.1f} min)")
    print(f"{'=' * 70}")

    # Summary table
    print("\nSSIM Summary:")
    if "e25_ssim" in results:
        print(f"  E25 single-block:  {results['e25_ssim']['avg_ssim']:.4f}")
    for n in (8, 4, 2):
        k = f"{n}blk_ft_ssim"
        if k in results:
            print(f"  {n}-blk + FT (F16):  {results[k]['avg_ssim']:.4f}")
    for label, key in [("SZ3", "sz3_ssim"), ("LCP", "lcp_ssim")]:
        if key in results:
            for r in sorted(results[key], key=lambda x: x["cr"]):
                print(f"  {label} EB={r['eb']:.2e}: CR={r['cr']:.1f}x  SSIM={r['avg_ssim']:.4f}")


if __name__ == "__main__":
    main()
