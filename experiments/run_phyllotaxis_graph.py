"""
Phyllotaxis static Casimir graph (Path A).

For each of three 2D point arrangements — Vogel's sunflower spiral,
hexagonal, square — place N points at matched number density, compute a
pairwise "Casimir graph" energy using the 1+1D pair formula
E_ij = -pi/(24 r_ij), and report nearest-neighbor structural statistics.

The question: does phyllotactic packing give a qualitatively different
static vacuum-energy landscape than the regular lattices at matched density?

Edge formula: 1+1D Casimir pair energy E = -pi/(24 a), summed over all
unordered pairs with r_ij <= r_cut. The cutoff r_cut = r_cut_scale *
median(nearest-neighbor distance). This keeps the graph sparse/local and
avoids the 1/r sum being dominated by boundary geometry.

Per-inner-site energy E_i = sum_{j != i, r_ij <= r_cut} -pi/(24 r_ij)
is the density-normalized observable; averaging over inner sites (those
far enough from the bounding disk edge that their r_cut neighborhood is
entirely inside the lattice) removes boundary-undersum artifacts.

Usage:
    python -m nothing_engine.experiments.run_phyllotaxis_graph [--quick]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import h5py
from scipy.spatial import cKDTree

GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))  # ~137.507764 degrees


def vogel(n_points: int, density: float = 1.0) -> np.ndarray:
    """Vogel's sunflower: r_n = c*sqrt(n), theta_n = n*golden_angle.

    sigma = 1 / (pi c^2), so c = 1 / sqrt(pi * sigma).
    """
    c = 1.0 / np.sqrt(np.pi * density)
    n = np.arange(1, n_points + 1)
    r = c * np.sqrt(n)
    theta = n * GOLDEN_ANGLE
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def hex_lattice(n_points: int, density: float = 1.0) -> np.ndarray:
    """Triangular (hex-packed) lattice, N points closest to origin.

    Primitive cell area = (sqrt(3)/2) a^2, sigma = 2/(sqrt(3) a^2),
    so a = sqrt(2 / (sqrt(3) * sigma)).
    """
    a = np.sqrt(2.0 / (np.sqrt(3.0) * density))
    m = int(np.ceil(np.sqrt(n_points))) + 8
    i, j = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1), indexing="ij")
    x = a * (i + 0.5 * j)
    y = a * (np.sqrt(3.0) / 2.0) * j
    pts = np.column_stack([x.ravel(), y.ravel()])
    d2 = np.sum(pts * pts, axis=1)
    order = np.argsort(d2)
    return pts[order[:n_points]]


def square_lattice(n_points: int, density: float = 1.0) -> np.ndarray:
    """Square lattice, N points closest to origin. sigma = 1/a^2."""
    a = 1.0 / np.sqrt(density)
    m = int(np.ceil(np.sqrt(n_points))) + 8
    i, j = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1), indexing="ij")
    pts = np.column_stack([(a * i).ravel(), (a * j).ravel()])
    d2 = np.sum(pts * pts, axis=1)
    order = np.argsort(d2)
    return pts[order[:n_points]]


def nn_stats(points: np.ndarray, k: int = 6) -> dict:
    """Nearest-neighbor distance and angular regularity (k = 6 by default).

    angular_std_mean is the mean-over-sites of the stddev of the gaps
    between successive sorted bearings to the k nearest neighbors. For a
    perfect hex lattice this is 0; for disordered arrangements it grows.
    """
    tree = cKDTree(points)
    d, idx = tree.query(points, k=k + 1)  # +1 because self included
    nn = d[:, 1]
    ang_std_list = []
    for i, site in enumerate(points):
        vecs = points[idx[i, 1 : k + 1]] - site
        bearings = np.sort(np.arctan2(vecs[:, 1], vecs[:, 0]))
        gaps = np.diff(np.concatenate([bearings, [bearings[0] + 2 * np.pi]]))
        ang_std_list.append(float(np.std(gaps)))
    return {
        "mean_nn": float(np.mean(nn)),
        "median_nn": float(np.median(nn)),
        "std_nn": float(np.std(nn)),
        "cv_nn": float(np.std(nn) / np.mean(nn)),
        "angular_std_mean": float(np.mean(ang_std_list)),
    }


def casimir_graph_energy(points: np.ndarray, r_cut: float) -> tuple[float, int]:
    """Sum -pi/(24 r_ij) over unordered pairs with r_ij <= r_cut."""
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=r_cut, output_type="ndarray")
    if pairs.shape[0] == 0:
        return 0.0, 0
    r = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
    r = np.maximum(r, 1e-12)
    return float(np.sum(-np.pi / (24.0 * r))), int(pairs.shape[0])


def per_inner_site_energy_and_force(
    points: np.ndarray, inner_mask: np.ndarray, r_cut: float
) -> dict:
    """Mean energy and force magnitude at an inner site, plus spread.

    E_i = sum_{j != i, r_ij <= r_cut} -pi/(24 r_ij)
    F_i = sum_{j != i, r_ij <= r_cut} (-dE_ij/dr_ij) r_hat_ij
        = sum ... (-pi/(24 r_ij^2)) r_hat_ij   (attractive pull on i from j)

    For perfect lattices (hex/square) F_i vanishes by symmetry; any nonzero
    |F_i| for Vogel is the interesting observable.

    Returned: mean/std of E_i and of |F_i|, plus n_inner.
    """
    inner_idx = np.where(inner_mask)[0]
    if inner_idx.size == 0:
        return {
            "e_mean": float("nan"),
            "e_std": float("nan"),
            "f_mag_mean": float("nan"),
            "f_mag_std": float("nan"),
            "f_mag_max": float("nan"),
            "n_inner": 0,
        }
    tree = cKDTree(points)
    neighbor_lists = tree.query_ball_point(points[inner_idx], r=r_cut)
    site_energies = np.empty(inner_idx.size)
    site_force_mags = np.empty(inner_idx.size)
    for k, (i, nbrs) in enumerate(zip(inner_idx, neighbor_lists)):
        nbrs_arr = np.asarray([j for j in nbrs if j != i], dtype=np.intp)
        if nbrs_arr.size == 0:
            site_energies[k] = 0.0
            site_force_mags[k] = 0.0
            continue
        diffs = points[nbrs_arr] - points[i]  # vectors from i to j
        r = np.maximum(np.linalg.norm(diffs, axis=1), 1e-12)
        site_energies[k] = float(np.sum(-np.pi / (24.0 * r)))
        # Attractive pair force on i pulls toward j: +r_hat * pi/(24 r^2)
        r_hat = diffs / r[:, None]
        f_vec = np.sum(r_hat * (np.pi / (24.0 * r * r))[:, None], axis=0)
        site_force_mags[k] = float(np.linalg.norm(f_vec))
    return {
        "e_mean": float(np.mean(site_energies)),
        "e_std": float(np.std(site_energies)),
        "f_mag_mean": float(np.mean(site_force_mags)),
        "f_mag_std": float(np.std(site_force_mags)),
        "f_mag_max": float(np.max(site_force_mags)),
        "n_inner": int(inner_idx.size),
    }


def analyze(name: str, points: np.ndarray, r_cut_scale: float = 4.0) -> dict:
    stats = nn_stats(points)
    r_cut = r_cut_scale * stats["median_nn"]
    r_bound = float(np.max(np.linalg.norm(points, axis=1)))
    inner_mask = np.linalg.norm(points, axis=1) < (r_bound - r_cut)
    e_total, n_pairs = casimir_graph_energy(points, r_cut)
    inner = per_inner_site_energy_and_force(points, inner_mask, r_cut)
    return {
        "name": name,
        "n_points": int(points.shape[0]),
        "r_bound": r_bound,
        "r_cut": r_cut,
        "r_cut_scale": r_cut_scale,
        "mean_nn": stats["mean_nn"],
        "median_nn": stats["median_nn"],
        "std_nn": stats["std_nn"],
        "cv_nn": stats["cv_nn"],
        "angular_std_mean": stats["angular_std_mean"],
        "n_pairs_in_cut": n_pairs,
        "e_total": e_total,
        "e_per_inner_site_mean": inner["e_mean"],
        "e_per_inner_site_std": inner["e_std"],
        "f_mag_mean": inner["f_mag_mean"],
        "f_mag_std": inner["f_mag_std"],
        "f_mag_max": inner["f_mag_max"],
        "n_inner_sites": inner["n_inner"],
    }


def save_hdf5(out_path: Path, density: float, r_cut_scale: float, rows: list[dict], lattices: dict):
    with h5py.File(out_path, "w") as f:
        f.attrs["density"] = density
        f.attrs["r_cut_scale"] = r_cut_scale
        for row in rows:
            key = f"{row['name']}_N{row['n_points']}"
            grp = f.create_group(key)
            grp.create_dataset("points", data=lattices[key])
            for k, v in row.items():
                if isinstance(v, str):
                    grp.attrs[k] = v
                elif isinstance(v, (int, np.integer)):
                    grp.attrs[k] = int(v)
                elif isinstance(v, (float, np.floating)):
                    grp.attrs[k] = float(v)


def print_table(rows: list[dict]):
    print()
    print("=" * 120)
    print("  PHYLLOTAXIS CASIMIR GRAPH  (E_ij = -pi/(24 r_ij), r_cut = r_cut_scale * median NN)")
    print("=" * 120)
    print(
        f"{'lattice':>8} | {'N':>5} | {'mean_nn':>8} | {'cv_nn':>6} | "
        f"{'ang_std':>8} | {'n_pairs':>7} | {'E/inner':>11} | "
        f"{'|F|_mean':>10} | {'|F|_std':>10} | {'|F|_max':>10} | {'n_in':>5}"
    )
    print("-" * 120)
    for r in rows:
        print(
            f"{r['name']:>8} | {r['n_points']:>5} | {r['mean_nn']:>8.4f} | "
            f"{r['cv_nn']:>6.4f} | {r['angular_std_mean']:>8.4f} | "
            f"{r['n_pairs_in_cut']:>7d} | {r['e_per_inner_site_mean']:>11.4e} | "
            f"{r['f_mag_mean']:>10.3e} | {r['f_mag_std']:>10.3e} | "
            f"{r['f_mag_max']:>10.3e} | {r['n_inner_sites']:>5d}"
        )


def print_ratios(rows: list[dict]):
    print()
    print("=" * 80)
    print("  DENSITY-NORMALIZED: per-inner-site energy comparison at each N")
    print("=" * 80)
    n_values = sorted({r["n_points"] for r in rows})
    for n in n_values:
        try:
            v = next(r for r in rows if r["name"] == "vogel" and r["n_points"] == n)
            h = next(r for r in rows if r["name"] == "hex" and r["n_points"] == n)
            s = next(r for r in rows if r["name"] == "square" and r["n_points"] == n)
        except StopIteration:
            continue
        ev, eh, es = (
            v["e_per_inner_site_mean"],
            h["e_per_inner_site_mean"],
            s["e_per_inner_site_mean"],
        )
        print(f"  N={n}: vogel={ev:.6e}   hex={eh:.6e}   square={es:.6e}")
        print(
            f"         vogel/hex = {ev / eh:.4f}    "
            f"vogel/square = {ev / es:.4f}    "
            f"hex/square = {eh / es:.4f}"
        )


def main():
    quick = "--quick" in sys.argv
    n_values = [200, 800] if quick else [200, 800, 3200]
    density = 1.0
    r_cut_scale = 4.0

    out_dir = Path("data/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phyllotaxis_graph.h5"

    rows: list[dict] = []
    lattices: dict[str, np.ndarray] = {}

    generators = [
        ("vogel", vogel),
        ("hex", hex_lattice),
        ("square", square_lattice),
    ]

    for n in n_values:
        for name, gen in generators:
            print(f"[{name} N={n}] generating and analyzing...", flush=True)
            t0 = time.perf_counter()
            pts = gen(n, density=density)
            row = analyze(name, pts, r_cut_scale=r_cut_scale)
            row["elapsed"] = time.perf_counter() - t0
            key = f"{name}_N{n}"
            lattices[key] = pts
            rows.append(row)
            print(
                f"  done in {row['elapsed']:.2f}s  "
                f"E_total={row['e_total']:+.4e}  "
                f"E/inner={row['e_per_inner_site_mean']:+.4e}",
                flush=True,
            )

    save_hdf5(out_path, density, r_cut_scale, rows, lattices)
    print_table(rows)
    print_ratios(rows)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
