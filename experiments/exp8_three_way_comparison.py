#!/usr/bin/env python3
"""EXP-8: 3DGS vs Recovered Particles vs GT (Three-Way Visual Comparison).

Qualitative and quantitative comparison of rendering quality.

Sub-experiments:
  8a: Quantitative PSNR (3DGS renders vs recovered particle renders vs GT)
  8b: Visual side-by-side examples

Usage:
    python -m experiments.exp8_three_way_comparison [--gpu 0]
"""

import json
import shutil
import time
from pathlib import Path

from experiments.common import *

RECOVER_SCRIPT = PARTICLEGS_ROOT / "recovery" / "recover_particles.py"


def main():
    parser = base_parser("EXP-8: Three-Way Visual Comparison")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to trained 3DGS model (e.g. F16)")
    parser.add_argument("--iteration", type=int, default=None)
    args = parser.parse_args()
    set_pvbatch_cuda_device(args.gpu)  # pin this process's GT renders to its GPU

    output_dir = Path(args.output_dir) if args.output_dir else RUNS_DIR / "exp8"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs = output_dir / "logs"
    logs.mkdir(exist_ok=True)
    t0 = time.time()

    shared_data = ensure_shared_data(gpu=args.gpu) if not args.skip_data_prep else {
        "vtp": SHARED_DIR / "particles.vtp",
        "normalization": SHARED_DIR / "normalization.json",
        "ply": SHARED_DIR / "points3d.ply",
        "eval_dirs": {evd["id"]: SHARED_DIR / evd["subdir"] / "data"
                      for evd in EVAL_DATASETS},
    }

    # Find model
    model_dir = args.model_dir
    iteration = args.iteration
    if not model_dir:
        for candidate in [
            RUNS_DIR / "exp4" / "blocks_8" / "finetuned" / "model",
            RUNS_DIR / "exp1" / "e25" / "02_S3_mix_6k" / "model",
        ]:
            if candidate.exists():
                model_dir = str(candidate)
                break
    if not model_dir:
        print("ERROR: No model found. Run EXP-1 or EXP-4 first.")
        return

    if not iteration:
        chk = find_checkpoint(model_dir)
        iteration = int(chk.stem.replace("chkpnt", "")) if chk else None

    # ── EXP-8a: Quantitative comparison ───────────────────────────────────
    print("\n" + "="*70)
    print("EXP-8a: Quantitative Three-Way Comparison")
    print("="*70)

    # 1. 3DGS rendering PSNR (same as eval)
    print("\n  [1] 3DGS rendering...")
    gs_eval = evaluate_model(model_dir, iteration, shared_data, logs, gpu=args.gpu)

    # 2. Recover particles and render with ParaView
    print("\n  [2] Recovering particles (V0, 280M)...")
    ply_path = Path(model_dir) / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    recov_dir = output_dir / "recovered"
    recov_vtp = recov_dir / "particles.vtp"

    # recover_particles.py outputs xx.f32, yy.f32, zz.f32 to output_dir
    if not (recov_dir / "xx.f32").exists():
        run_cmd([PYTHON_BIN, str(RECOVER_SCRIPT),
                 "--ply_path", str(ply_path),
                 "--normalization", str(shared_data["normalization"]),
                 "--output_dir", str(recov_dir),
                 "--num_points", str(NUM_PARTICLES),
                 "--method", "V0_baseline"],
                log_path=logs / "recover.log")

    # 3. Create VTP from recovered particles and render
    print("\n  [3] Rendering recovered particles with ParaView...")

    if not recov_vtp.exists():
        run_cmd(
            pvbatch_cmd(PREPARE_SCRIPT,
                        "--raw_x", recov_dir / "xx.f32",
                        "--raw_y", recov_dir / "yy.f32",
                        "--raw_z", recov_dir / "zz.f32",
                        "--output_dir", recov_dir,
                        "--num_points_raw", "0", "--skip_images"),
            log_path=logs / "recovered_vtp.log")

    # Render recovered particles at each eval orbit
    recov_eval = {}
    norm_path = shared_data["normalization"]
    for evd in EVAL_DATASETS:
        render_out = recov_dir / f"render_{evd['id']}"
        img_dir = render_out / "images"
        if not img_dir.exists() or len(list(img_dir.glob("*.png"))) < 80:
            gen_args = [
                "--vtp_path", str(recov_vtp),
                "--output_dir", str(render_out),
                "--camera_strategy", "multi_orbit",
                "--orbit_radii", evd["orbit_radii"],
                "--num_frames", "80", "--width", "1920", "--height", "1080",
                "--train_ratio", "1.0", "--split_seed", "42",
                "--normalization_path", str(norm_path),
            ]
            for k, v in DEFAULT_VIZ_PARAMS.items():
                gen_args += [f"--{k}", v]
            run_cmd(pvbatch_cmd(GENERATE_SCRIPT, *gen_args),
                    log_path=logs / f"recovered_render_{evd['id']}.log")

        gt_dir = shared_data["eval_dirs"][evd["id"]] / "images"
        psnr, mpsnr, n = compute_psnr_dirs(gt_dir, img_dir)
        recov_eval[evd["id"]] = {"psnr": psnr, "masked_psnr": mpsnr, "n": n}
        if psnr:
            print(f"    {evd['id']}: recovered PSNR={psnr:.2f}  masked={mpsnr:.2f}")

    # Summary
    print(f"\n{'='*70}")
    print("EXP-8a: Three-Way Quantitative Comparison")
    print(f"{'='*70}")
    headers = ["Dataset", "3DGS Full", "3DGS Masked", "Recov Full", "Recov Masked"]
    rows = []
    for evd in EVAL_DATASETS:
        eid = evd["id"]
        gs = gs_eval.get(eid, {})
        rc = recov_eval.get(eid, {})
        rows.append([
            eid,
            f"{gs.get('psnr', 0):.2f}", f"{gs.get('masked_psnr', 0):.2f}",
            f"{rc.get('psnr', 0):.2f}", f"{rc.get('masked_psnr', 0):.2f}",
        ])
    print_table(headers, rows)

    gs_avg = gs_eval.get("avg", {})
    rc_all_p = [v["psnr"] for v in recov_eval.values() if v.get("psnr")]
    rc_all_m = [v["masked_psnr"] for v in recov_eval.values() if v.get("masked_psnr")]
    rc_avg_p = sum(rc_all_p)/len(rc_all_p) if rc_all_p else 0
    rc_avg_m = sum(rc_all_m)/len(rc_all_m) if rc_all_m else 0

    print(f"\n  3DGS avg masked: {gs_avg.get('masked_psnr', 0):.2f} dB")
    print(f"  Recovered avg masked: {rc_avg_m:.2f} dB")
    print(f"  Delta: +{(gs_avg.get('masked_psnr', 0) - rc_avg_m):.1f} dB")

    results = {
        "3dgs_eval": gs_eval,
        "recovered_eval": recov_eval,
    }
    save_results(results, output_dir / "results.json")
    print(f"\nEXP-8 complete ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
