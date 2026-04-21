#!/usr/bin/env python3
"""Compare recovered particles vs ground truth — basic stats + HACC science metrics.

Metrics:
  1. Per-axis statistics (mean, std, min, max, skewness, kurtosis)
  2. Marginal distributions (1D histograms per axis)
  3. 3D density field + density PDF
  4. Power spectrum P(k) — the primary HACC cosmological statistic
  5. Two-point correlation function xi(r) (from density field via FFT)
  6. Nearest-neighbor distance distribution (subsampled)

Usage:
    python compare_particles.py --gt_dir <gt/> --rec_dir <recover/> --output <analysis/>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Font sizes tuned for IEEE two-column paper:
# stats plots displayed at ~0.32\textwidth (~2.4in), slices at ~0.70\textwidth (~5.1in)
STATS_FONT = 16   # for PDF, P(k), xi(r) — figsize ~(5,6), displayed at ~2.4in
SLICES_FONT = 18  # for density slices — figsize ~(14,9), displayed at ~5.1in


# ── I/O ──────────────────────────────────────────────────────────────────

def load_f32(directory):
    """Load xx.f32, yy.f32, zz.f32 -> (N, 3) float32."""
    d = Path(directory)
    x = np.fromfile(str(d / "xx.f32"), dtype=np.float32)
    y = np.fromfile(str(d / "yy.f32"), dtype=np.float32)
    z = np.fromfile(str(d / "zz.f32"), dtype=np.float32)
    assert len(x) == len(y) == len(z), f"Length mismatch: {len(x)}, {len(y)}, {len(z)}"
    return np.stack([x, y, z], axis=-1)


# ── 1. Per-axis statistics ────────────────────────────────────────────────

def compute_stats(pts, label):
    from scipy.stats import skew, kurtosis
    stats = {}
    for i, ax in enumerate(["x", "y", "z"]):
        col = pts[:, i]
        stats[ax] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "median": float(np.median(col)),
            "skewness": float(skew(col)),
            "kurtosis": float(kurtosis(col)),
        }
    return stats


def print_stats_table(gt_stats, rec_stats):
    print(f"\n{'Axis':<5} {'Metric':<12} {'GT':>14} {'Recover':>14} {'Diff':>14}")
    print("-" * 63)
    for ax in ["x", "y", "z"]:
        for m in ["mean", "std", "min", "max", "median", "skewness", "kurtosis"]:
            g = gt_stats[ax][m]
            r = rec_stats[ax][m]
            diff = r - g
            print(f"{ax:<5} {m:<12} {g:>14.6f} {r:>14.6f} {diff:>+14.6f}")
        print()


# ── 2. Marginal distributions ────────────────────────────────────────────

def plot_marginals(gt, rec, output_dir, nbins=500):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (ax_name, ax) in enumerate(zip(["x", "y", "z"], axes)):
        lo = min(gt[:, i].min(), rec[:, i].min())
        hi = max(gt[:, i].max(), rec[:, i].max())
        bins = np.linspace(lo, hi, nbins + 1)
        ax.hist(gt[:, i], bins=bins, density=True, alpha=0.6, label="GT", color="steelblue")
        ax.hist(rec[:, i], bins=bins, density=True, alpha=0.6, label="Recover", color="coral")
        ax.set_xlabel(ax_name)
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_title(f"Marginal distribution — {ax_name}")
    plt.tight_layout()
    path = Path(output_dir) / "marginal_distributions.png"
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ── 3. Density field + PDF ────────────────────────────────────────────────

def particles_to_density_ngp(pts, grid_size, box_min, box_max):
    normed = (pts - box_min) / (box_max - box_min) * grid_size
    indices = np.floor(normed).astype(np.int64)
    np.clip(indices, 0, grid_size - 1, out=indices)
    density = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
    np.add.at(density, (indices[:, 0], indices[:, 1], indices[:, 2]), 1.0)
    mean_density = density.mean()
    if mean_density > 0:
        delta = density / mean_density - 1.0
    else:
        delta = density
    return delta, mean_density


def plot_density_pdf(delta_gt, delta_rec, output_dir, nbins=200):
    fig, ax = plt.subplots(figsize=(5, 6))
    plt.rcParams.update({"font.size": STATS_FONT})
    for delta, label, color in [(delta_gt, "GT", "steelblue"),
                                 (delta_rec, "Recover", "coral")]:
        vals = delta.ravel()
        lo, hi = vals.min(), vals.max()
        bins = np.linspace(lo, min(hi, 20), nbins)
        ax.hist(vals, bins=bins, density=True, alpha=0.6, label=label, color=color)
    ax.set_xlabel(r"$\delta = \rho/\bar{\rho} - 1$", fontsize=STATS_FONT)
    ax.set_ylabel(r"$P(\delta)$", fontsize=STATS_FONT)
    ax.set_yscale("log")
    ax.set_title("Density PDF", fontsize=STATS_FONT + 2)
    ax.legend(fontsize=STATS_FONT - 2)
    ax.tick_params(labelsize=STATS_FONT - 2)
    plt.tight_layout()
    path = Path(output_dir) / "density_pdf.png"
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_density_slices(delta_gt, delta_rec, output_dir):
    """Plot central slices using log10(1+delta) = log10(rho/rho_bar) for visibility."""
    mid = delta_gt.shape[0] // 2
    # log10(1+delta) maps voids (delta=-1) to -inf, mean (delta=0) to 0,
    # overdense (delta=100) to ~2.  Clamp 1+delta >= 0.01 to avoid -inf.
    log_gt = np.log10(np.maximum(1.0 + delta_gt, 0.01))
    log_rec = np.log10(np.maximum(1.0 + delta_rec, 0.01))

    # x-slice only, single-column figure for paper
    sl = (mid, slice(None), slice(None))
    vmin = min(log_gt[sl].min(), log_rec[sl].min())
    vmax = max(log_gt[sl].max(), log_rec[sl].max())

    fig, axes = plt.subplots(2, 1, figsize=(4.5, 7.5))
    im0 = axes[0].imshow(log_gt[sl], vmin=vmin, vmax=vmax, cmap="inferno", origin="lower")
    axes[0].set_title("Ground Truth", fontsize=SLICES_FONT)
    axes[0].tick_params(labelsize=SLICES_FONT - 4)
    cb0 = plt.colorbar(im0, ax=axes[0], shrink=0.85)
    cb0.ax.tick_params(labelsize=SLICES_FONT - 4)
    im1 = axes[1].imshow(log_rec[sl], vmin=vmin, vmax=vmax, cmap="inferno", origin="lower")
    axes[1].set_title("Recovered", fontsize=SLICES_FONT)
    axes[1].tick_params(labelsize=SLICES_FONT - 4)
    cb1 = plt.colorbar(im1, ax=axes[1], shrink=0.85)
    cb1.ax.tick_params(labelsize=SLICES_FONT - 4)
    plt.suptitle(r"Density field central slice — $\log_{10}(\rho\,/\,\bar\rho)$",
                 fontsize=SLICES_FONT + 1, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = Path(output_dir) / "density_slices.png"
    plt.savefig(str(path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── 4. Power spectrum P(k) ────────────────────────────────────────────────

def compute_power_spectrum(delta, box_length, nbins=80):
    grid_size = delta.shape[0]
    V = box_length ** 3
    delta_k = np.fft.rfftn(delta)
    pk_3d = (np.abs(delta_k) ** 2) * (V / grid_size ** 6)
    kf = 2 * np.pi / box_length
    kx = np.fft.fftfreq(grid_size, d=1.0 / grid_size) * kf
    ky = np.fft.fftfreq(grid_size, d=1.0 / grid_size) * kf
    kz = np.fft.rfftfreq(grid_size, d=1.0 / grid_size) * kf
    kgrid = np.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2)
    k_max = kf * grid_size / 2
    k_edges = np.linspace(kf, k_max, nbins + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])
    pk = np.zeros(nbins)
    counts = np.zeros(nbins, dtype=np.int64)
    kgrid_flat = kgrid.ravel()
    pk3d_flat = pk_3d.ravel()
    bin_idx = np.digitize(kgrid_flat, k_edges) - 1
    for i in range(nbins):
        mask = bin_idx == i
        if mask.any():
            pk[i] = pk3d_flat[mask].mean()
            counts[i] = mask.sum()
    valid = counts > 0
    return k_centers[valid], pk[valid]


def plot_power_spectrum(k_gt, pk_gt, k_rec, pk_rec, output_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 7), gridspec_kw={"height_ratios": [3, 1]})
    ax1.loglog(k_gt, pk_gt, label="GT", color="steelblue", lw=2)
    ax1.loglog(k_rec, pk_rec, label="Recover", color="coral", lw=2, ls="--")
    ax1.set_xlabel("$k$", fontsize=STATS_FONT)
    ax1.set_ylabel("$P(k)$", fontsize=STATS_FONT)
    ax1.set_title("Power Spectrum", fontsize=STATS_FONT + 2)
    ax1.legend(fontsize=STATS_FONT - 2)
    ax1.tick_params(labelsize=STATS_FONT - 2)
    ax1.grid(True, alpha=0.3)
    pk_rec_interp = np.interp(k_gt, k_rec, pk_rec)
    ratio = pk_rec_interp / pk_gt
    ax2.semilogx(k_gt, ratio, color="black", lw=1.5)
    ax2.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax2.fill_between(k_gt, 0.95, 1.05, alpha=0.15, color="green", label=r"$\pm$5%")
    ax2.fill_between(k_gt, 0.90, 1.10, alpha=0.10, color="orange", label=r"$\pm$10%")
    ax2.set_xlabel("$k$", fontsize=STATS_FONT)
    ax2.set_ylabel("$P_\\mathrm{rec}/P_\\mathrm{GT}$", fontsize=STATS_FONT)
    ax2.set_ylim(0.0, 2.0)
    ax2.legend(fontsize=STATS_FONT - 4)
    ax2.tick_params(labelsize=STATS_FONT - 2)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    path = Path(output_dir) / "power_spectrum.png"
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"  Saved {path}")
    return k_gt, ratio


# ── 5. Two-point correlation function ξ(r) ────────────────────────────────

def compute_tpcf(delta, box_length, nbins=60):
    """Compute two-point correlation function ξ(r) via FFT (Wiener-Khinchin).

    ξ(r) = <δ(x)δ(x+r)> = IFFT(|FFT(δ)|²) / N³,
    then averaged in spherical shells.
    """
    grid_size = delta.shape[0]
    cell_size = box_length / grid_size

    delta_k = np.fft.fftn(delta)
    xi_3d = np.fft.ifftn(np.abs(delta_k) ** 2).real / grid_size ** 3

    # Physical distance for each grid offset
    offsets = np.fft.fftfreq(grid_size, d=1.0 / grid_size) * cell_size
    rgrid = np.sqrt(offsets[:, None, None] ** 2 +
                    offsets[None, :, None] ** 2 +
                    offsets[None, None, :] ** 2)

    # Bin into spherical shells (skip r=0 which is just the variance)
    r_max = box_length / 2
    r_edges = np.linspace(cell_size, r_max, nbins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    xi = np.zeros(nbins)
    counts = np.zeros(nbins, dtype=np.int64)
    rgrid_flat = rgrid.ravel()
    xi3d_flat = xi_3d.ravel()
    bin_idx = np.digitize(rgrid_flat, r_edges) - 1

    for i in range(nbins):
        mask = bin_idx == i
        if mask.any():
            xi[i] = xi3d_flat[mask].mean()
            counts[i] = mask.sum()

    valid = counts > 0
    return r_centers[valid], xi[valid]


def plot_tpcf(r_gt, xi_gt, r_rec, xi_rec, output_dir):
    """Plot ξ(r) with ratio panel (same layout as power spectrum)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 7),
                                    gridspec_kw={"height_ratios": [3, 1]})
    pos_gt = xi_gt > 0
    pos_rec = xi_rec > 0
    ax1.loglog(r_gt[pos_gt], xi_gt[pos_gt], label="GT", color="steelblue", lw=2)
    ax1.loglog(r_rec[pos_rec], xi_rec[pos_rec], label="Recover", color="coral",
               lw=2, ls="--")
    ax1.set_xlabel("$r$", fontsize=STATS_FONT)
    ax1.set_ylabel(r"$\xi(r)$", fontsize=STATS_FONT)
    ax1.set_title("Two-Point Correlation", fontsize=STATS_FONT + 2)
    ax1.legend(fontsize=STATS_FONT - 2)
    ax1.tick_params(labelsize=STATS_FONT - 2)
    ax1.grid(True, alpha=0.3)

    xi_rec_interp = np.interp(r_gt, r_rec, xi_rec)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(xi_gt != 0, xi_rec_interp / xi_gt, np.nan)
    ax2.semilogx(r_gt, ratio, color="black", lw=1.5)
    ax2.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax2.fill_between(r_gt, 0.95, 1.05, alpha=0.15, color="green", label=r"$\pm$5%")
    ax2.fill_between(r_gt, 0.90, 1.10, alpha=0.10, color="orange", label=r"$\pm$10%")
    ax2.set_xlabel("$r$", fontsize=STATS_FONT)
    ax2.set_ylabel(r"$\xi_\mathrm{rec}/\xi_\mathrm{GT}$", fontsize=STATS_FONT)
    ax2.set_ylim(0.0, 2.0)
    ax2.legend(fontsize=STATS_FONT - 4)
    ax2.tick_params(labelsize=STATS_FONT - 2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = Path(output_dir) / "tpcf.png"
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"  Saved {path}")
    return r_gt, ratio


# ── 6. Nearest-neighbor distances ─────────────────────────────────────────

def nn_distance_distribution(gt, rec, n_sample=500000, seed=42):
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    idx_gt = rng.choice(len(gt), size=min(n_sample, len(gt)), replace=False)
    idx_rec = rng.choice(len(rec), size=min(n_sample, len(rec)), replace=False)
    sub_gt = gt[idx_gt]
    sub_rec = rec[idx_rec]
    print("  Building KD-trees (subsampled)...")
    tree_gt = cKDTree(sub_gt)
    tree_rec = cKDTree(sub_rec)
    print("  Querying nearest neighbors...")
    dd_gt, _ = tree_gt.query(sub_gt, k=2)
    dd_rec, _ = tree_rec.query(sub_rec, k=2)
    nn_gt = dd_gt[:, 1]
    nn_rec = dd_rec[:, 1]
    return nn_gt, nn_rec


def plot_nn_distribution(nn_gt, nn_rec, output_dir, nbins=300):
    fig, ax = plt.subplots(figsize=(10, 6))
    hi = max(np.percentile(nn_gt, 99.5), np.percentile(nn_rec, 99.5))
    bins = np.linspace(0, hi, nbins)
    ax.hist(nn_gt, bins=bins, density=True, alpha=0.6, label="GT", color="steelblue")
    ax.hist(nn_rec, bins=bins, density=True, alpha=0.6, label="Recover", color="coral")
    ax.set_xlabel("Nearest-neighbor distance")
    ax.set_ylabel("Density")
    ax.set_title("Nearest-Neighbor Distance Distribution")
    ax.legend()
    ax.text(0.98, 0.95, f"GT mean: {nn_gt.mean():.6f}\nRec mean: {nn_rec.mean():.6f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    path = Path(output_dir) / "nn_distance.png"
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare recovered vs GT particles")
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--rec_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--grid_size", type=int, default=256)
    parser.add_argument("--nn_samples", type=int, default=500000)
    parser.add_argument("--normalization", default=None,
                        help="Path to normalization.json — denormalize recover data before comparing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading GT...")
    gt = load_f32(args.gt_dir)
    print(f"  GT: {len(gt):,} particles")

    print("Loading Recover...")
    rec = load_f32(args.rec_dir)
    print(f"  Recover: {len(rec):,} particles")

    if args.normalization:
        with open(args.normalization) as f:
            norm = json.load(f)
        center = np.array(norm["center"], dtype=np.float64)
        scale = float(norm["scale"])
        rec = rec.astype(np.float64) / scale + center
        rec = rec.astype(np.float32)
        print(f"  Denormalized: x=[{rec[:,0].min():.2f}, {rec[:,0].max():.2f}]")

    # 1. Per-axis statistics
    print("\n== 1. Per-axis Statistics ==")
    gt_stats = compute_stats(gt, "GT")
    rec_stats = compute_stats(rec, "Recover")
    print_stats_table(gt_stats, rec_stats)
    with open(output_dir / "stats.json", "w") as f:
        json.dump({"gt": gt_stats, "recover": rec_stats}, f, indent=2)

    # 2. Marginal distributions
    print("\n== 2. Marginal Distributions ==")
    plot_marginals(gt, rec, output_dir)

    # 3. Density field
    print(f"\n== 3. Density Field (grid={args.grid_size}^3) ==")
    box_min = np.minimum(gt.min(axis=0), rec.min(axis=0))
    box_max = np.maximum(gt.max(axis=0), rec.max(axis=0))
    box_length = (box_max - box_min).max()
    center = (box_min + box_max) / 2
    box_min_cube = center - box_length / 2
    box_max_cube = center + box_length / 2

    print("  Computing GT density field...")
    delta_gt, rho_gt = particles_to_density_ngp(gt, args.grid_size, box_min_cube, box_max_cube)
    print("  Computing Recover density field...")
    delta_rec, rho_rec = particles_to_density_ngp(rec, args.grid_size, box_min_cube, box_max_cube)

    delta_diff = delta_rec - delta_gt
    rmse = np.sqrt(np.mean(delta_diff ** 2))
    mae = np.mean(np.abs(delta_diff))
    cc = np.corrcoef(delta_gt.ravel(), delta_rec.ravel())[0, 1]
    print(f"  Density field RMSE: {rmse:.4f}")
    print(f"  Density field MAE:  {mae:.4f}")
    print(f"  Density field correlation: {cc:.6f}")

    plot_density_pdf(delta_gt, delta_rec, output_dir)
    plot_density_slices(delta_gt, delta_rec, output_dir)

    # 4. Power spectrum
    print(f"\n== 4. Power Spectrum ==")
    k_gt, pk_gt = compute_power_spectrum(delta_gt, box_length)
    k_rec, pk_rec = compute_power_spectrum(delta_rec, box_length)
    k_common, pk_ratio = plot_power_spectrum(k_gt, pk_gt, k_rec, pk_rec, output_dir)
    within_10 = np.abs(pk_ratio - 1.0) < 0.10
    if within_10.any():
        print(f"  P(k) within +/-10% up to k = {k_common[within_10][-1]:.2f}")

    # 5. Two-point correlation function
    print(f"\n== 5. Two-Point Correlation Function ξ(r) ==")
    print("  Computing GT ξ(r)...")
    r_gt_xi, xi_gt = compute_tpcf(delta_gt, box_length)
    print("  Computing Recover ξ(r)...")
    r_rec_xi, xi_rec = compute_tpcf(delta_rec, box_length)
    r_common_xi, xi_ratio = plot_tpcf(r_gt_xi, xi_gt, r_rec_xi, xi_rec, output_dir)
    valid_xi = ~np.isnan(xi_ratio)
    within_10_xi = valid_xi & (np.abs(xi_ratio - 1.0) < 0.10)
    outside_10_xi = valid_xi & (np.abs(xi_ratio - 1.0) >= 0.10)
    if outside_10_xi.any():
        # ξ(r) deviates at small r; find where it settles within ±10%
        last_outside = np.where(outside_10_xi)[0][-1]
        if last_outside + 1 < len(r_common_xi):
            print(f"  ξ(r) within ±10% for r > {r_common_xi[last_outside + 1]:.2f}")
        # Report peak deviation
        peak_idx = np.nanargmax(np.abs(xi_ratio - 1.0))
        peak_dev = (xi_ratio[peak_idx] - 1.0) * 100
        print(f"  Peak deviation: {peak_dev:+.1f}% at r = {r_common_xi[peak_idx]:.2f}")
    elif within_10_xi.any():
        print(f"  ξ(r) within ±10% across all r")

    # 6. Nearest-neighbor distances
    print(f"\n== 6. Nearest-Neighbor Distances ({args.nn_samples:,} subsamples) ==")
    nn_gt, nn_rec = nn_distance_distribution(gt, rec, n_sample=args.nn_samples, seed=args.seed)
    print(f"  GT  NN mean: {nn_gt.mean():.6f}, std: {nn_gt.std():.6f}")
    print(f"  Rec NN mean: {nn_rec.mean():.6f}, std: {nn_rec.std():.6f}")
    plot_nn_distribution(nn_gt, nn_rec, output_dir)

    # Save summary
    summary = {
        "num_particles_gt": int(len(gt)),
        "num_particles_rec": int(len(rec)),
        "grid_size": args.grid_size,
        "density_field": {
            "rmse": float(rmse),
            "mae": float(mae),
            "correlation": float(cc),
            "mean_density_gt": float(rho_gt),
            "mean_density_rec": float(rho_rec),
        },
        "tpcf": {
            "r": [float(v) for v in r_gt_xi],
            "xi_gt": [float(v) for v in xi_gt],
            "xi_rec": [float(v) for v in np.interp(r_gt_xi, r_rec_xi, xi_rec)],
        },
        "nn_distance": {
            "gt_mean": float(nn_gt.mean()),
            "gt_std": float(nn_gt.std()),
            "rec_mean": float(nn_rec.mean()),
            "rec_std": float(nn_rec.std()),
        },
        "per_axis_stats": {"gt": gt_stats, "recover": rec_stats},
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
