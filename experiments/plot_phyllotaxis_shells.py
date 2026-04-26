"""
Shell-resolved |F|(r) analysis for the phyllotaxis Casimir-graph experiment.

Produces |F_i|(r) for each lattice: bin sites by radial distance from
the lattice center, plot mean +/- std of |F_i| in each bin. For hex and
square this is flat at machine zero (symmetry-driven cancellation); for
Vogel, the curve reveals the radial structure of the residual force.

A secondary panel overlays radii r_{F_k} = c * sqrt(F_k) for the Fibonacci
sequence. At these radii, the golden-angle multiples n*phi land nearest
to previous Fibonacci-indexed sites, making local packing briefly more
regular. The hypothesis is that |F|(r) has troughs near these radii and
peaks between them.

NOTE: This is a GRAPH-GEOMETRY observable using the 1+1D Casimir pair
formula E_ij = -pi/(24 r_ij). It is not a prediction about real mirror
arrays — see validation gates P.1 through P.5 in the research notes.

Usage:
    python -m nothing_engine.experiments.plot_phyllotaxis_shells [--n 3200]
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


def fibonacci_up_to(n_max: int) -> list[int]:
    """Fibonacci numbers (excluding 0, 1, 1) up to n_max."""
    fibs: list[int] = [2, 3]
    while fibs[-1] + fibs[-2] <= n_max:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs


def shell_bin(
    radii: np.ndarray, values: np.ndarray, r_max: float, n_bins: int = 40
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin values by radius. Returns (bin_centers, mean, std, count)."""
    edges = np.linspace(0.0, r_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean = np.full(n_bins, np.nan)
    std = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)
    for k in range(n_bins):
        mask = (radii >= edges[k]) & (radii < edges[k + 1])
        n = int(np.sum(mask))
        count[k] = n
        if n > 0:
            mean[k] = float(np.mean(values[mask]))
            std[k] = float(np.std(values[mask]))
    return centers, mean, std, count


def plot(h5_path: Path, out_path: Path, n_target: int, n_bins: int = 40):
    with h5py.File(h5_path, "r") as f:
        data = {}
        for name in ("vogel", "hex", "square"):
            key = f"{name}_N{n_target}"
            if key not in f:
                raise KeyError(f"Missing {key} in {h5_path}")
            data[name] = {
                "points": f[key]["points"][:],
                "r_cut": float(f[key].attrs["r_cut"]),
                "r_bound": float(f[key].attrs["r_bound"]),
            }

    for name, d in data.items():
        d["radii"] = np.linalg.norm(d["points"], axis=1)
        d["fmag"] = per_site_forces(d["points"], d["r_cut"])
        d["inner_mask"] = d["radii"] < (d["r_bound"] - d["r_cut"])

    r_max = min(d["r_bound"] - d["r_cut"] for d in data.values())

    fig, (ax_linear, ax_log) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    colors = {"vogel": "C3", "hex": "C0", "square": "C2"}
    for name in ("vogel", "hex", "square"):
        d = data[name]
        inside = d["radii"] < r_max
        centers, mean, std, count = shell_bin(
            d["radii"][inside], d["fmag"][inside], r_max, n_bins=n_bins
        )
        valid = count > 0
        for ax in (ax_linear, ax_log):
            ax.plot(centers[valid], mean[valid], "o-", color=colors[name],
                    label=f"{name} (mean over shell)", markersize=4, linewidth=1.2)
            ax.fill_between(
                centers[valid],
                np.maximum(mean[valid] - std[valid], 1e-20),
                mean[valid] + std[valid],
                color=colors[name],
                alpha=0.15,
            )

    # Fibonacci-resonance radii for Vogel: r_{F_k} = c * sqrt(F_k)
    # Vogel was constructed with density=1 → c = 1/sqrt(pi).
    c_vogel = 1.0 / np.sqrt(np.pi)
    fibs = fibonacci_up_to(n_target)
    fib_radii = [c_vogel * np.sqrt(F) for F in fibs]
    for ax in (ax_linear, ax_log):
        for r_f, F in zip(fib_radii, fibs):
            if r_f > r_max:
                continue
            ax.axvline(r_f, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
            y_top = ax.get_ylim()[1] if ax is ax_linear else None
    # Label a few Fibonacci radii at the top of ax_linear
    y0, y1 = ax_linear.get_ylim()
    label_y = y1 * 0.92 if y1 > 0 else 0.0
    for r_f, F in zip(fib_radii, fibs):
        if r_f > r_max:
            continue
        ax_linear.text(
            r_f,
            label_y,
            f"F={F}",
            fontsize=8,
            rotation=90,
            va="top",
            ha="center",
            color="gray",
            alpha=0.9,
        )

    ax_linear.set_ylabel(r"$\langle |\mathbf{F}_i| \rangle_{\mathrm{shell}}$ (linear)")
    ax_linear.set_title(
        f"Shell-resolved per-site Casimir-graph force  |  N = {n_target}  |  "
        r"$E_{ij} = -\pi/(24\,r_{ij})$"
    )
    ax_linear.legend(loc="upper right", fontsize=9)
    ax_linear.grid(alpha=0.3)

    ax_log.set_xlabel("radial distance from lattice center")
    ax_log.set_ylabel(r"$\langle |\mathbf{F}_i| \rangle_{\mathrm{shell}}$ (log)")
    ax_log.set_yscale("log")
    ax_log.grid(alpha=0.3, which="both")
    ax_log.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")

    # Also print the shell profile for Vogel numerically so the Fibonacci
    # residue hypothesis can be judged on numbers, not just eyeballing.
    d = data["vogel"]
    inside = d["radii"] < r_max
    centers, mean, std, count = shell_bin(
        d["radii"][inside], d["fmag"][inside], r_max, n_bins=n_bins
    )
    print()
    print("Vogel |F| shell profile (radius bin, <|F|>, std, count):")
    print(f"{'r_center':>10} | {'<|F|>':>12} | {'std':>12} | {'count':>6}")
    for c, m, s, n in zip(centers, mean, std, count):
        if n == 0:
            continue
        print(f"{c:>10.3f} | {m:>12.4e} | {s:>12.4e} | {n:>6d}")
    print()
    print("Fibonacci radii (r_F = sqrt(F/pi)):")
    for F in fibs:
        r_f = c_vogel * np.sqrt(F)
        if r_f <= r_max:
            print(f"  F = {F:>5d}   r_F = {r_f:>8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3200)
    ap.add_argument("--bins", type=int, default=50)
    ap.add_argument("--input", type=str, default="data/experiments/phyllotaxis_graph.h5")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    h5_path = Path(args.input)
    out_path = Path(args.output) if args.output else Path(
        f"data/experiments/phyllotaxis_shells_N{args.n}.png"
    )
    plot(h5_path, out_path, args.n, n_bins=args.bins)


if __name__ == "__main__":
    main()
