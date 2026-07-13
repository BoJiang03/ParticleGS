#!/usr/bin/env python3
"""Aggregate runs/exp*/results.json into paper-table form.

Invoked at the end of reproduce_ae.sh / reproduce.sh. Emits three files in --out:
  summary.json     — merged raw results from every runs/expN/results.json
  tables.md        — Markdown view of Tab. III / VI / VII / VIII / IX
  fig_scale.json   — per-block Gaussians vs particles (Fig. scale)

Usage:
  python scripts/aggregate_results.py [--runs runs] [--out runs/summary]
"""

import argparse
import json
from pathlib import Path


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def fmt(v, n=2, unit=""):
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        return f"{v:.{n}f}{unit}"
    return f"{v}{unit}"


def tab3_block_scan(exp4):
    """Tab. III — merged and finetuned columns across block counts."""
    if not exp4:
        return "*(no runs/exp4/results.json — EXP-4 did not run)*"
    lines = ["### Tab. III — Block Training Scan",
             "",
             "| Blocks | Merged mPSNR | Merged Gaussians | Merged MB | FT mPSNR | FT Gaussians | FT MB | FT SSIM |",
             "|--------|--------------|------------------|-----------|----------|--------------|-------|---------|"]
    for k in sorted(exp4.keys(), key=lambda s: int(s.split("_")[1])):
        r = exp4[k]
        m, ft = r.get("merged", {}), r.get("finetuned", {})
        lines.append(
            f"| {r.get('n_blocks', '?')} "
            f"| {fmt(m.get('masked_psnr'))} "
            f"| {fmt(m.get('num_gaussians'), 0)} "
            f"| {fmt(m.get('size_mb'), 1)} "
            f"| {fmt(ft.get('masked_psnr'))} "
            f"| {fmt(ft.get('num_gaussians'), 0)} "
            f"| {fmt(ft.get('size_mb'), 1)} "
            f"| {fmt(ft.get('ssim'), 4)} |")
    return "\n".join(lines)


def tab5_rate_distortion(exp1, exp12):
    """Tab. VI — 3DGS vs SZ3/LCP rate-distortion."""
    if not exp1:
        return "*(no runs/exp1/results.json — EXP-1 did not run)*"
    lines = ["### Tab. VI — Rate-Distortion (3DGS vs SZ3/LCP)",
             "",
             "**E25 single-block (3DGS):**",
             ""]
    e25 = exp1.get("exp1b_e25") or {}
    if e25:
        lines.append(f"- PSNR: {fmt(e25.get('avg_psnr'))} dB  /  masked: {fmt(e25.get('avg_masked_psnr'))} dB")
        lines.append(f"- Gaussians: {fmt(e25.get('num_gaussians'), 0)}  size: {fmt(e25.get('size_mb'), 1)} MB  CR: {fmt(e25.get('cr'), 0)}×")
        if exp12 and "e25_ssim" in exp12:
            lines.append(f"- SSIM: {fmt(exp12['e25_ssim'], 4)}")
    else:
        lines.append("*(exp1b_e25 empty — E25 training did not complete)*")
    lines += ["", "**SZ3 sweep (15 points):**", "",
              "| EB | CR | Avg PSNR | Avg mPSNR |",
              "|----|------|----------|-----------|"]
    for pt in exp1.get("exp1a_sz3") or []:
        lines.append(f"| {pt.get('eb'):.4g} | {fmt(pt.get('cr'), 1)}× "
                     f"| {fmt(pt.get('avg_psnr'))} | {fmt(pt.get('avg_masked_psnr'))} |")
    lines += ["", "**LCP sweep (11 points):**", "",
              "| EB | CR | Avg PSNR | Avg mPSNR |",
              "|----|------|----------|-----------|"]
    for pt in exp1.get("exp1c_lcp") or []:
        lines.append(f"| {pt.get('eb'):.4g} | {fmt(pt.get('cr'), 1)}× "
                     f"| {fmt(pt.get('avg_psnr'))} | {fmt(pt.get('avg_masked_psnr'))} |")
    return "\n".join(lines)


def tab6_perf(exp6, exp11):
    """Tab. VII — performance (FPS / memory / loading / merge)."""
    if not exp6 and not exp11:
        return "*(no EXP-6 / EXP-11 results)*"
    lines = ["### Tab. VII — Rendering and Inference Performance", ""]
    if exp6 and "summary" in exp6:
        s = exp6["summary"]
        lines += [
            f"- 3DGS FPS @ 1920×1080: **{fmt(s.get('3dgs_fps'), 1)}**",
            f"- ParaView FPS @ 1920×1080: {fmt(s.get('paraview_fps'), 3)}",
            f"- Speedup: {fmt(s.get('speedup'), 1)}×",
            f"- Model size: {fmt(s.get('model_size_mb'), 1)} MB  /  CR: {fmt(s.get('compression_ratio'), 1)}×",
            "",
        ]
    if exp6 and "3dgs" in exp6:
        benches = exp6["3dgs"].get("benchmarks", [])
        if benches:
            lines += ["| Resolution | FPS | ms/frame | MPixels/s |",
                      "|------------|-----|----------|-----------|"]
            for b in benches:
                lines.append(f"| {b.get('width')}×{b.get('height')} "
                             f"| {fmt(b.get('fps'), 1)} "
                             f"| {fmt(b.get('ms_per_frame'), 2)} "
                             f"| {fmt(b.get('mpixels_per_s'), 1)} |")
            lines.append("")
    if exp11:
        mem = exp11.get("inference_memory", {})
        load = exp11.get("loading_time", {})
        lines += ["**Memory and loading:**", ""]
        for key in ("single_block", "8_block_finetuned"):
            m = mem.get(key, {})
            lt = load.get(key, {})
            if m:
                lines.append(f"- {m.get('label')}: "
                             f"model {fmt(m.get('model_memory_mb'), 1)} MB, "
                             f"peak render {fmt(m.get('peak_render_memory_mb'), 1)} MB, "
                             f"load {fmt(lt.get('avg_load_ms'), 1)} ms")
        pvm = exp11.get("paraview_load_time", {})
        mc = exp11.get("merge_cost", {})
        if pvm:
            lines.append(f"- ParaView load+first-render: {fmt(pvm.get('load_and_first_render_s'), 1)} s")
        if mc:
            n = mc.get("n_blocks", 8)
            lines.append(f"- {n}-block merge cost: {fmt(mc.get(f'merge_{n}block_s'), 2)} s")
    return "\n".join(lines)


def tab7_recovery_methods(exp7):
    """Tab. VIII — recovery methods (V0/V1/V4/V6)."""
    arr = (exp7 or {}).get("7a") or []
    if not arr:
        return "*(no EXP-7 7a entries)*"
    lines = ["### Tab. VIII — Particle Recovery Methods",
             "",
             "| Method | Density RMSE | Density Corr | NN GT mean | NN rec mean | NN rec std |",
             "|--------|--------------|--------------|------------|-------------|-----------|"]
    for r in arr:
        df = r.get("density_field", {}) or {}
        nn = r.get("nn_distance", {}) or {}
        lines.append(f"| {r.get('method')} "
                     f"| {fmt(df.get('rmse'), 3)} "
                     f"| {fmt(df.get('correlation'), 4)} "
                     f"| {fmt(nn.get('gt_mean'), 4)} "
                     f"| {fmt(nn.get('rec_mean'), 4)} "
                     f"| {fmt(nn.get('rec_std'), 4)} |")
    return "\n".join(lines)


def tab8_recovery_blocks(exp7):
    """Tab. IX — recovery across 1/2/4/8/16 blocks."""
    arr = (exp7 or {}).get("7c") or (exp7 or {}).get("7b") or []
    if not arr:
        return "*(no EXP-7 7b/7c entries)*"
    lines = ["### Tab. IX — Particle Recovery across Block Counts (or Scales)",
             "",
             "| Key | Density RMSE | Density Corr | NN GT mean | NN rec mean |",
             "|-----|--------------|--------------|------------|-------------|"]
    for r in arr:
        df = r.get("density_field", {}) or {}
        nn = r.get("nn_distance", {}) or {}
        key = r.get("n_blocks") or r.get("scale_factor") or "?"
        lines.append(f"| {key} "
                     f"| {fmt(df.get('rmse'), 3)} "
                     f"| {fmt(df.get('correlation'), 4)} "
                     f"| {fmt(nn.get('gt_mean'), 4)} "
                     f"| {fmt(nn.get('rec_mean'), 4)} |")
    return "\n".join(lines)


def fire2_generalization(fire2):
    """FIRE-2 cross-dataset generalization (EXP-13)."""
    if not fire2:
        return "*(no runs/exp_fire2/results.json — EXP-13 did not run)*"
    lines = ["### FIRE-2 Cross-Dataset Generalization (EXP-13)",
             "",
             "E25 single-block pipeline applied to the FIRE-2 L172 DM-only "
             "cosmological box (Wetzel et al. 2023, CC BY 4.0).",
             "",
             "| Metric | Value |",
             "|--------|-------|",
             f"| Particles | {fmt(fire2.get('num_particles'), 0)} |",
             f"| PSNR | {fmt(fire2.get('avg_psnr'))} dB |",
             f"| Masked PSNR | {fmt(fire2.get('avg_masked_psnr'))} dB |",
             f"| Gaussians | {fmt(fire2.get('num_gaussians'), 0)} |",
             f"| Model size | {fmt(fire2.get('size_mb'), 1)} MB |",
             f"| Compression ratio | {fmt(fire2.get('cr'), 0)}× |"]
    return "\n".join(lines)


def fig_scale_data(runs_dir: Path):
    """Fig. scale — per-block Gaussians vs particles from exp4 per-block PLYs."""
    try:
        from plyfile import PlyData
    except ImportError:
        return [{"error": "plyfile not installed — Fig. scale skipped"}]
    rows = []
    for N in (2, 4, 8, 16):
        bdir = runs_dir / f"exp4/blocks_{N}"
        pinfo = bdir / f"partition_{N}" / "partition_info.json"
        counts = {}
        if pinfo.exists():
            info = json.loads(pinfo.read_text())
            for b in info.get("blocks", []):
                counts[f"block_{b['id']:02d}"] = b.get("num_particles")
        for sub in sorted(bdir.glob("block_*")):
            plys = list((sub / "02_S3_mix" / "model" / "point_cloud").glob("iteration_*/point_cloud.ply"))
            if not plys:
                continue
            ply = max(plys, key=lambda p: int(p.parent.name.split("_")[1]))
            try:
                n_g = len(PlyData.read(str(ply))["vertex"])
            except Exception as e:
                rows.append({"n_blocks": N, "block_id": sub.name, "error": str(e)})
                continue
            rows.append({
                "n_blocks": N,
                "block_id": sub.name,
                "n_particles": counts.get(sub.name),
                "n_gaussians": n_g,
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs", help="Runs directory")
    ap.add_argument("--out", default="runs/summary", help="Output directory")
    args = ap.parse_args()

    runs = Path(args.runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    exp1  = load(runs / "exp1"  / "results.json")
    exp4  = load(runs / "exp4"  / "results.json")
    exp6  = load(runs / "exp6"  / "results.json")
    exp7  = load(runs / "exp7"  / "results.json")
    exp8  = load(runs / "exp8"  / "results.json")
    exp11 = load(runs / "exp11" / "results.json")
    exp12 = load(runs / "exp12" / "results.json")
    fire2 = load(runs / "exp_fire2" / "results.json")

    summary = {
        "exp1":  exp1,
        "exp4":  exp4,
        "exp6":  exp6,
        "exp7":  exp7,
        "exp8":  exp8,
        "exp11": exp11,
        "exp12": exp12,
        "exp_fire2": fire2,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    fig = fig_scale_data(runs)
    (out / "fig_scale.json").write_text(json.dumps(fig, indent=2))

    sections = [
        "# ParticleGS — Paper Results",
        "",
        "Aggregated from `runs/exp*/results.json`.",
        "",
        tab3_block_scan(exp4), "",
        tab5_rate_distortion(exp1, exp12), "",
        tab6_perf(exp6, exp11), "",
        tab7_recovery_methods(exp7), "",
        tab8_recovery_blocks(exp7), "",
        fire2_generalization(fire2), "",
        "### Fig. scale",
        "",
        f"Per-block Gaussians vs particles: `{out}/fig_scale.json` ({len(fig)} rows)",
    ]
    (out / "tables.md").write_text("\n".join(sections))

    print(f"[aggregate] wrote {out}/summary.json, tables.md, fig_scale.json")


if __name__ == "__main__":
    main()
