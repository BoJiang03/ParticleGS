#!/usr/bin/env python3
"""EXP-14: Generalization to unseen camera poses and out-of-range viz params.

The standard evaluation (orbits 1.0/0.7/0.5x, per-frame radius/opacity sampled
from the *trained* Beta range) is entirely in-distribution: zero unseen camera
poses, zero out-of-range (r, a). This experiment measures graceful degradation
along two independent axes, reusing the single E25 model trained by EXP-1
(inference-only, ~40 min):

  Axis A — camera pose: orbit radius interpolated between trained orbits
           (0.85x, 0.6x) and extrapolated beyond them (1.3x far, 0.35x near),
           with (r, a) held in-distribution.
  Axis B — viz parameters: at a fixed trained orbit (0.7x), the radius and
           opacity factors are swept out to 2.0-2.5x, beyond the trained
           factor support of [0.25, 1.75].

For each setting we render the model and score full PSNR, masked PSNR, and SSIM
against ParaView ground truth generated with the SAME normalization as training.

Requires: EXP-1 must have run first (trained model under runs/exp1/e25).

Usage:
    python -m experiments.exp14_generalization --gpu 0 [--model_dir <dir>] [--frames 80]
"""

import argparse
import json
import shutil
import time
from pathlib import Path

from experiments.common import *

BASE_RADIUS = float(DEFAULT_VIZ_PARAMS["gaussian_radius"])   # 0.01
BASE_OPACITY = float(DEFAULT_VIZ_PARAMS["opacity"])          # 0.05

TRAINED_ORBITS = [1.0, 0.7, 0.5]
# Trained factor support: r in [0.0025,0.0175]/0.01, a in [0.0125,0.0875]/0.05
TRAINED_FACTOR_LO, TRAINED_FACTOR_HI = 0.25, 1.75

# Axis A: (orbit_radius, regime)
ORBIT_GRID = [
    (1.30, "extrap-far"),
    (1.00, "trained"),
    (0.85, "interp"),
    (0.70, "trained"),
    (0.60, "interp"),
    (0.50, "trained"),
    (0.35, "extrap-near"),
]

# Axis B: factor sweep at fixed mid orbit (0.7x). Factors >1.75 (or <0.25) are
# outside the trained range.
EVAL_ORBIT_B = 0.7
RADIUS_FACTOR_GRID = [0.20, 0.50, 1.00, 1.50, 2.00, 2.50]
OPACITY_FACTOR_GRID = [0.20, 0.50, 1.00, 1.50, 2.00, 2.40]


def regime_for_factor(f):
    if f < TRAINED_FACTOR_LO or f > TRAINED_FACTOR_HI:
        return "out-of-range"
    return "in-range"


def find_trained_model(output_root):
    """Find the final-stage E25 model dir + iteration under runs/exp1/e25."""
    e25 = Path(output_root) / "e25"
    candidates = []
    for model_dir in e25.glob("*/model"):
        chk = find_checkpoint(model_dir)
        if chk:
            it = int(chk.stem.replace("chkpnt", ""))
            candidates.append((it, model_dir))
    if not candidates:
        return None, None
    it, model_dir = max(candidates, key=lambda x: x[0])
    return model_dir, it


def gen_gt(vtp, norm, out_dir, orbit_radii, viz_params, eval_id, logs, frames):
    """Generate ParaView ground-truth for one eval set."""
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    if img_dir.exists() and len(list(img_dir.glob("*.png"))) >= frames:
        print(f"  [Skip GT] {eval_id} ({len(list(img_dir.glob('*.png')))} frames)")
        return
    args = [
        "--vtp_path", str(vtp),
        "--output_dir", str(out_dir),
        "--camera_strategy", "multi_orbit",
        "--orbit_radii", str(orbit_radii),
        "--num_frames", str(frames),
        "--width", "1920", "--height", "1080",
        "--train_ratio", "1.0", "--split_seed", "42",
        "--normalization_path", str(norm),
    ]
    for k, v in viz_params.items():
        args += [f"--{k}", str(v)]
    run_cmd(pvbatch_cmd(GENERATE_SCRIPT, *args), log_path=logs / f"gt_{eval_id}.log")


def score(model_dir, iteration, data_dir, logs, gpu, tag):
    """Render model on data_dir cameras and score vs GT (PSNR full/masked + SSIM)."""
    out = Path(model_dir).parent / "gen_renders" / tag
    out.mkdir(parents=True, exist_ok=True)
    run_render(model_dir, data_dir, iteration, logs, f"render_{tag}", gpu=gpu,
               factor_delta_opacity=0.3, factor_delta_scale=0.1,
               min_opacity_clamp=0.4, min_scale_pixels=0.0,
               extra_args=["--output_dir", str(out), "--resolution", "1"])
    render_dir = out / "train" / f"ours_{iteration}" / "renders"
    gt_dir = Path(data_dir) / "images"
    psnr, mpsnr, n = compute_psnr_dirs(gt_dir, render_dir)
    ssim, _ = compute_ssim_dirs(gt_dir, render_dir)
    shutil.rmtree(out, ignore_errors=True)
    return {"psnr": psnr, "masked_psnr": mpsnr, "ssim": ssim, "n": n}


def viz_sampled():
    """In-distribution viz params (matches paper eval): Beta-sampled in trained range."""
    return dict(DEFAULT_VIZ_PARAMS)


def viz_fixed(radius_abs, opacity_abs):
    """Fixed (r, a) for all frames — used to probe out-of-range factors."""
    return {
        "gaussian_radius": f"{radius_abs:.6f}",
        "opacity": f"{opacity_abs:.6f}",
        "viz_mode": "fixed",
    }


def run_axis_a(vtp, norm, work, model_dir, iteration, logs, gpu, frames):
    print("\n" + "=" * 70)
    print("Axis A — camera pose generalization (orbit radius), (r,a) in-distribution")
    print("=" * 70)
    rows = []
    for orbit, regime in ORBIT_GRID:
        tag = f"orbitA_{orbit:.2f}"
        data_dir = work / tag
        gen_gt(vtp, norm, data_dir, orbit, viz_sampled(), tag, logs, frames)
        s = score(model_dir, iteration, data_dir, logs, gpu, tag)
        s.update({"orbit": orbit, "regime": regime})
        rows.append(s)
        print(f"  orbit {orbit:.2f}x [{regime:11s}] "
              f"PSNR={s['psnr']:.2f} masked={s['masked_psnr']:.2f} SSIM={s['ssim']:.4f}")
        shutil.rmtree(data_dir / "images", ignore_errors=True)
    return rows


def run_axis_b(vtp, norm, work, model_dir, iteration, logs, gpu, frames):
    print("\n" + "=" * 70)
    print(f"Axis B — viz-param extrapolation at fixed {EVAL_ORBIT_B}x orbit")
    print("=" * 70)
    radius_rows, opacity_rows = [], []

    print("  -- radius factor sweep (opacity fixed at base) --")
    for f in RADIUS_FACTOR_GRID:
        tag = f"radB_{f:.2f}"
        data_dir = work / tag
        gen_gt(vtp, norm, data_dir, EVAL_ORBIT_B,
               viz_fixed(BASE_RADIUS * f, BASE_OPACITY), tag, logs, frames)
        s = score(model_dir, iteration, data_dir, logs, gpu, tag)
        s.update({"radius_factor": f, "regime": regime_for_factor(f)})
        radius_rows.append(s)
        print(f"  r-factor {f:.2f} [{s['regime']:12s}] "
              f"PSNR={s['psnr']:.2f} masked={s['masked_psnr']:.2f} SSIM={s['ssim']:.4f}")
        shutil.rmtree(data_dir / "images", ignore_errors=True)

    print("  -- opacity factor sweep (radius fixed at base) --")
    for f in OPACITY_FACTOR_GRID:
        tag = f"opaB_{f:.2f}"
        data_dir = work / tag
        gen_gt(vtp, norm, data_dir, EVAL_ORBIT_B,
               viz_fixed(BASE_RADIUS, BASE_OPACITY * f), tag, logs, frames)
        s = score(model_dir, iteration, data_dir, logs, gpu, tag)
        s.update({"opacity_factor": f, "regime": regime_for_factor(f)})
        opacity_rows.append(s)
        print(f"  a-factor {f:.2f} [{s['regime']:12s}] "
              f"PSNR={s['psnr']:.2f} masked={s['masked_psnr']:.2f} SSIM={s['ssim']:.4f}")
        shutil.rmtree(data_dir / "images", ignore_errors=True)

    return {"radius": radius_rows, "opacity": opacity_rows}


def make_plot(axis_a, axis_b, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [plot skipped] {e}")
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # Axis A
    ax = axes[0]
    ox = [r["orbit"] for r in axis_a]
    oy = [r["masked_psnr"] for r in axis_a]
    ax.plot(ox, oy, "-o", color="#2c7fb8")
    for r in axis_a:
        if r["regime"] != "trained":
            ax.annotate(r["regime"].split("-")[0], (r["orbit"], r["masked_psnr"]),
                        fontsize=7, textcoords="offset points", xytext=(0, 6))
    for t in TRAINED_ORBITS:
        ax.axvline(t, color="#aaaaaa", ls=":", lw=0.8)
    ax.set_xlabel("orbit radius (x base)")
    ax.set_ylabel("masked PSNR (dB)")
    ax.set_title("A: camera pose (dotted = trained orbits)")

    # Axis B radius
    ax = axes[1]
    fx = [r["radius_factor"] for r in axis_b["radius"]]
    fy = [r["masked_psnr"] for r in axis_b["radius"]]
    ax.plot(fx, fy, "-o", color="#d95f0e")
    ax.axvspan(TRAINED_FACTOR_LO, TRAINED_FACTOR_HI, color="#d9f0a3", alpha=0.5,
               label="trained range")
    ax.set_xlabel("radius factor")
    ax.set_ylabel("masked PSNR (dB)")
    ax.set_title("B: radius extrapolation (0.7x orbit)")
    ax.legend(fontsize=8)

    # Axis B opacity
    ax = axes[2]
    fx = [r["opacity_factor"] for r in axis_b["opacity"]]
    fy = [r["masked_psnr"] for r in axis_b["opacity"]]
    ax.plot(fx, fy, "-o", color="#756bb1")
    ax.axvspan(TRAINED_FACTOR_LO, TRAINED_FACTOR_HI, color="#d9f0a3", alpha=0.5,
               label="trained range")
    ax.set_xlabel("opacity factor")
    ax.set_ylabel("masked PSNR (dB)")
    ax.set_title("B: opacity extrapolation (0.7x orbit)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"  Plot saved: {out_png}")


def main():
    parser = base_parser("EXP-14: pose + viz-param generalization")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Trained E25 model dir (auto-detect from runs/exp1 if omitted)")
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--frames", type=int, default=80)
    args = parser.parse_args()
    set_pvbatch_cuda_device(args.gpu)  # pin this process's GT renders to its GPU

    src_root = Path(args.output_dir) if args.output_dir else RUNS_DIR / "exp1"
    out_dir = RUNS_DIR / "exp14"
    work = out_dir / "work"
    logs = out_dir / "logs"
    work.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    shared = get_shared_data_dict()
    vtp, norm = shared["vtp"], shared["normalization"]
    if not Path(vtp).exists() or not Path(norm).exists():
        raise SystemExit(f"Shared data missing ({vtp}). Run training/exp1 first.")

    if args.model_dir:
        model_dir = Path(args.model_dir)
        iteration = args.iteration or int(find_checkpoint(model_dir).stem.replace("chkpnt", ""))
    else:
        model_dir, iteration = find_trained_model(src_root)
    if not model_dir:
        raise SystemExit("No trained model found under runs/exp1/e25. Run EXP-1 first.")
    print(f"Model: {model_dir}  iteration={iteration}")
    stats = get_model_stats(model_dir, iteration)
    print(f"  {stats['num_gaussians']} Gaussians, {stats['size_mb']:.1f} MB")

    t0 = time.time()
    axis_a = run_axis_a(vtp, norm, work, model_dir, iteration, logs, args.gpu, args.frames)
    axis_b = run_axis_b(vtp, norm, work, model_dir, iteration, logs, args.gpu, args.frames)
    elapsed = time.time() - t0

    results = {
        "model_dir": str(model_dir), "iteration": iteration,
        "num_gaussians": stats["num_gaussians"], "size_mb": round(stats["size_mb"], 1),
        "base_radius": BASE_RADIUS, "base_opacity": BASE_OPACITY,
        "trained_orbits": TRAINED_ORBITS,
        "trained_factor_range": [TRAINED_FACTOR_LO, TRAINED_FACTOR_HI],
        "frames": args.frames,
        "axis_a_pose": axis_a,
        "axis_b_viz": axis_b,
        "elapsed_min": round(elapsed / 60, 1),
    }
    save_results(results, out_dir / "results.json")
    make_plot(axis_a, axis_b, out_dir / "generalization.png")
    print(f"\nEXP-14 complete ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
