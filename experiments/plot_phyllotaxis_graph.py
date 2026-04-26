"""
Plot phyllotaxis Casimir graph results.

Reads the lattice points from data/experiments/phyllotaxis_graph.h5,
recomputes per-site energy and net Casimir force, and produces a
2-row 3-column figure:

    Row 1: scatter of each lattice (Vogel, hex, square) with each site
           colored by log10 |F_i|. A dashed circle marks the inner disk
           (sites whose r_cut neighborhood is entirely inside the lattice);
           sites outside the circle are boundary-undersampled and shown
           as hollow markers.
    Row 2: histogram of log10 |F_i| for inner sites of each lattice,
           shared x-axis across the three panels.

Usage:
    python -m nothing_engine.experiments.plot_phyllotaxis_graph [--n 800]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


def per_site_forces(points: np.ndarray, r_cut: float) -> np.ndarray:
    """Return |F_i| for every site (not just inner ones)."""
    tree = cKDTree(points)
    neighbor_lists = tree.query_ball_point(points, r=r_cut)
    fmag = np.zeros(points.shape[0])
    for i, nbrs in enumerate(neighbor_lists):
        nbrs_arr = np.asarray([j for j in nbrs if j != i], dtype=np.intp)
        if nbrs_arr.size == 0:
            continue
        diffs = points[nbrs_arr] - points[i]
        r = np.maximum(np.linalg.norm(diffs, axis=1), 1e-12)
        r_hat = diffs / r[:, None]
        f_vec = np.sum(r_hat * (np.pi / (24.0 * r * r))[:, None], axis=0)
        fmag[i] = float(np.linalg.norm(f_vec))
    return fmag


def plot(h5_path: Path, out_path: Path, n_target: int):
    with h5py.File(h5_path, "r") as f:
        lattices = {}
        for name in ("vogel", "hex", "square"):
            key = f"{name}_N{n_target}"
            if key not in f:
                raise KeyError(f"Missing {key} in {h5_path}")
            lattices[name] = {
                "points": f[key]["points"][:],
                "r_cut": float(f[key].attrs["r_cut"]),
                "r_bound": float(f[key].attrs["r_bound"]),
                "e_per_inner_site_mean": float(f[key].attrs["e_per_inner_site_mean"]),
                "f_mag_mean": float(f[key].attrs["f_mag_mean"]),
                "f_mag_max": float(f[key].attrs["f_mag_max"]),
            }

    for name, d in lattices.items():
        d["f_mag"] = per_site_forces(d["points"], d["r_cut"])
        d["inner_mask"] = (
            np.linalg.norm(d["points"], axis=1) < (d["r_bound"] - d["r_cut"])
        )

    # Color scale is keyed to Vogel's dynamic range: hex and square are
    # machine-zero and would otherwise squash the scale across 16+ decades
    # and hide Vogel's internal structure. With this choice, hex/square
    # sites saturate at the low end of the colormap (accurate: "near zero")
    # and Vogel's hot/cold contrast is preserved.
    vogel_f = lattices["vogel"]["f_mag"][lattices["vogel"]["inner_mask"]]
    vogel_f = np.maximum(vogel_f, 1e-18)
    vmin = max(float(np.min(vogel_f)), 1e-18)
    vmax = float(np.max(vogel_f))
    # Row-2 histogram range still covers all three lattices
    f_all = np.concatenate(
        [np.maximum(d["f_mag"][d["inner_mask"]], 1e-18) for d in lattices.values()]
    )

    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[1, 1, 1, 0.04],
        height_ratios=[1.2, 1],
        hspace=0.32,
        wspace=0.30,
        left=0.06,
        right=0.94,
        top=0.91,
        bottom=0.08,
    )
    order = ["vogel", "hex", "square"]
    top_axes = [fig.add_subplot(gs[0, c]) for c in range(3)]
    cbar_ax = fig.add_subplot(gs[0, 3])
    bot_axes = [fig.add_subplot(gs[1, c]) for c in range(3)]

    for col, name in enumerate(order):
        ax = top_axes[col]
        d = lattices[name]
        pts = d["points"]
        fmag = np.maximum(d["f_mag"], 1e-18)
        inner = d["inner_mask"]
        ax.scatter(
            pts[~inner, 0],
            pts[~inner, 1],
            s=8,
            facecolors="none",
            edgecolors="lightgray",
            linewidths=0.4,
        )
        sc = ax.scatter(
            pts[inner, 0],
            pts[inner, 1],
            c=fmag[inner],
            s=16,
            cmap="magma",
            norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
            edgecolors="black",
            linewidths=0.15,
        )
        r_inner = d["r_bound"] - d["r_cut"]
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(
            r_inner * np.cos(theta),
            r_inner * np.sin(theta),
            "--",
            color="gray",
            linewidth=0.7,
            alpha=0.8,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(name, fontsize=13, pad=8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.text(
            0.02,
            0.98,
            f"mean |F| = {d['f_mag_mean']:.2e}\nmax  |F| = {d['f_mag_max']:.2e}",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
        )

    fig.colorbar(sc, cax=cbar_ax, label=r"$|\mathbf{F}_i|$  (log scale)")

    log_f_all = np.log10(f_all)
    lo, hi = float(np.min(log_f_all)), float(np.max(log_f_all))
    bins = np.linspace(lo - 0.1, hi + 0.1, 40)
    for col, name in enumerate(order):
        ax = bot_axes[col]
        d = lattices[name]
        vals = np.log10(np.maximum(d["f_mag"][d["inner_mask"]], 1e-18))
        ax.hist(vals, bins=bins, color="steelblue", alpha=0.85, edgecolor="black", linewidth=0.3)
        ax.set_xlabel(r"$\log_{10} |\mathbf{F}_i|$  (inner sites)")
        ax.set_ylabel("count")
        ax.set_title(f"{name} per-site |F| distribution", fontsize=11)
        ax.text(
            0.02,
            0.95,
            f"E/inner = {d['e_per_inner_site_mean']:.3e}",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
        )

    fig.suptitle(
        f"Phyllotaxis Casimir graph  |  N = {n_target}  |  "
        r"per-site residual force under $E_{ij} = -\pi / (24\,r_{ij})$",
        fontsize=13,
        y=0.975,
    )
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800, help="Lattice size to plot (must exist in HDF5)")
    ap.add_argument(
        "--input",
        type=str,
        default="data/experiments/phyllotaxis_graph.h5",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path (default: data/experiments/phyllotaxis_graph_N{N}.png)",
    )
    args = ap.parse_args()

    h5_path = Path(args.input)
    out_path = Path(args.output) if args.output else Path(
        f"data/experiments/phyllotaxis_graph_N{args.n}.png"
    )
    plot(h5_path, out_path, args.n)


if __name__ == "__main__":
    main()
