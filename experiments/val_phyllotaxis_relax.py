"""
P.3 — Overdamped gradient-descent relaxation of the phyllotaxis Casimir graph.

Physics background: Because F_i = -grad_i E is conservative (verified in P.4),
a closed cyclic path in configuration space encloses zero work — no cyclic
extraction is possible. The only energy available is the once-off drop from
the initial Vogel configuration to a local minimum reachable by free descent.
This script quantifies that drop.

Setup: load a lattice from HDF5, pin all sites with r >= r_pin (outer annulus),
relax the remaining free sites under the overdamped dynamics x_dot = -grad E
via L-BFGS-B on the full O(N^2) pair sum. At convergence, report:

  - E_initial, E_final, dE = E_initial - E_final
  - dE / |E_initial|          (fractional energy drop)
  - dE / N_free               (per-free-site energy drop)
  - max |x_final - x_initial| (largest displacement)
  - final max |F| on free sites (should be near zero)
  - average nearest-neighbor angle distribution for free sites
    (hex has 60 deg symmetry; deviation measures how hex-like the relaxed
    interior became)

Run on Vogel (expected: non-zero dE), hex (expected: near-zero dE, already
at a local min), square (expected: small dE, relaxes toward hex-adjacent).

Usage:
    python -m nothing_engine.experiments.val_phyllotaxis_relax [--n 800]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import h5py
from scipy.optimize import minimize
from scipy.spatial import cKDTree


def total_energy_full(points: np.ndarray) -> float:
    n = points.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    diffs = points[iu] - points[ju]
    r = np.linalg.norm(diffs, axis=1)
    r = np.maximum(r, 1e-12)
    return float(np.sum(-np.pi / (24.0 * r)))


def analytic_site_forces_full(points: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    F = np.zeros_like(points)
    for i in range(n):
        diffs = np.delete(points, i, axis=0) - points[i]
        r = np.maximum(np.linalg.norm(diffs, axis=1), 1e-12)
        F[i] = np.sum(diffs * (np.pi / (24.0 * r**3))[:, None], axis=0)
    return F


def nn_angle_stats(points: np.ndarray, free_mask: np.ndarray, k: int = 6) -> dict:
    """For each free site, find its k nearest neighbors (in the full lattice),
    compute angles between successive neighbor vectors around the site, and
    return summary stats. A hex lattice gives 6 neighbors at 60 deg spacing;
    the std of successive-angle differences is a proxy for "how hex-like".
    """
    tree = cKDTree(points)
    _, idxs = tree.query(points[free_mask], k=k + 1)  # +1 to drop self
    idxs = idxs[:, 1:]  # drop self
    all_angle_stds = []
    for site_i, nbrs in zip(np.where(free_mask)[0], idxs):
        vecs = points[nbrs] - points[site_i]
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        angles = np.sort(angles)
        # Successive differences (wrap around)
        diffs = np.diff(np.concatenate([angles, angles[:1] + 2 * np.pi]))
        # For hex, all diffs = 60 deg = pi/3; std = 0
        all_angle_stds.append(float(np.std(diffs)))
    return {
        "mean_std": float(np.mean(all_angle_stds)),
        "median_std": float(np.median(all_angle_stds)),
        "max_std": float(np.max(all_angle_stds)),
    }


def relax_lattice(
    points: np.ndarray, r_bound: float, pin_fraction: float, gtol: float, maxiter: int
) -> dict:
    radii = np.linalg.norm(points, axis=1)
    r_pin = pin_fraction * r_bound
    pinned_mask = radii >= r_pin
    free_mask = ~pinned_mask
    n_free = int(np.sum(free_mask))
    n_pinned = int(np.sum(pinned_mask))

    free_idx = np.where(free_mask)[0]
    x0 = points[free_mask].copy().flatten()  # (2 * n_free,)

    def assemble(x_free_flat: np.ndarray) -> np.ndarray:
        x_free = x_free_flat.reshape(-1, 2)
        full = points.copy()
        full[free_mask] = x_free
        return full

    def E_and_grad(x_free_flat: np.ndarray) -> tuple[float, np.ndarray]:
        full = assemble(x_free_flat)
        E = total_energy_full(full)
        F_full = analytic_site_forces_full(full)
        grad = -F_full[free_mask].flatten()
        return E, grad

    E_init = total_energy_full(points)
    F_init = analytic_site_forces_full(points)
    max_F_init_free = float(np.max(np.linalg.norm(F_init[free_mask], axis=1)))

    result = minimize(
        E_and_grad,
        x0,
        jac=True,
        method="L-BFGS-B",
        options={"gtol": gtol, "maxiter": maxiter, "ftol": 1e-14},
    )

    final_points = assemble(result.x)
    E_final = float(result.fun)
    F_final = analytic_site_forces_full(final_points)
    max_F_final_free = float(np.max(np.linalg.norm(F_final[free_mask], axis=1)))

    displacements = final_points - points
    disp_mag = np.linalg.norm(displacements, axis=1)
    max_disp = float(np.max(disp_mag[free_mask]))
    mean_disp = float(np.mean(disp_mag[free_mask]))

    # Check no free site migrated into or past the pinned annulus
    final_radii = np.linalg.norm(final_points, axis=1)
    n_escapees = int(np.sum(final_radii[free_mask] >= r_pin))

    ang_init = nn_angle_stats(points, free_mask, k=6)
    ang_final = nn_angle_stats(final_points, free_mask, k=6)

    return {
        "E_init": float(E_init),
        "E_final": E_final,
        "dE": float(E_init) - E_final,
        "frac_dE": (float(E_init) - E_final) / abs(float(E_init)),
        "dE_per_free": (float(E_init) - E_final) / max(n_free, 1),
        "max_F_init_free": max_F_init_free,
        "max_F_final_free": max_F_final_free,
        "max_disp": max_disp,
        "mean_disp": mean_disp,
        "n_free": n_free,
        "n_pinned": n_pinned,
        "n_escapees": n_escapees,
        "r_pin": r_pin,
        "converged": bool(result.success),
        "n_iter": int(result.nit),
        "n_fev": int(result.nfev),
        "ang_init_mean_std": ang_init["mean_std"],
        "ang_final_mean_std": ang_final["mean_std"],
        "ang_final_max_std": ang_final["max_std"],
        "final_points": final_points,
        "free_mask": free_mask,
        "pinned_mask": pinned_mask,
    }


def run(
    h5_path: Path,
    n_target: int,
    pin_fraction: float,
    gtol: float,
    maxiter: int,
    out_h5: Path,
) -> None:
    print(f"P.3 relaxation gate  (N = {n_target}, pin_fraction = {pin_fraction})")
    print("=" * 100)

    results = {}
    for name in ("vogel", "hex", "square"):
        with h5py.File(h5_path, "r") as f:
            key = f"{name}_N{n_target}"
            if key not in f:
                raise KeyError(f"Missing {key} in {h5_path}")
            grp = f[key]
            points = grp["points"][:]
            r_bound = float(grp.attrs["r_bound"])

        print(f"\n--- {name} ---")
        r = relax_lattice(points, r_bound, pin_fraction, gtol, maxiter)
        results[name] = r
        print(f"  N_free = {r['n_free']}, N_pinned = {r['n_pinned']}, r_pin = {r['r_pin']:.3f}")
        print(f"  E_init = {r['E_init']:.6e}   E_final = {r['E_final']:.6e}")
        print(
            f"  dE = E_init - E_final = {r['dE']:.6e}   "
            f"dE / |E_init| = {r['frac_dE']:.3e}   "
            f"dE / N_free = {r['dE_per_free']:.3e}"
        )
        print(
            f"  max |F| on free sites: initial {r['max_F_init_free']:.3e}  "
            f"final {r['max_F_final_free']:.3e}"
        )
        print(f"  max displacement (free) = {r['max_disp']:.3e}  mean = {r['mean_disp']:.3e}")
        print(
            f"  NN-angle std (free): initial {r['ang_init_mean_std']:.3e}  "
            f"final {r['ang_final_mean_std']:.3e}  (hex = 0)"
        )
        print(
            f"  converged = {r['converged']}   n_iter = {r['n_iter']}   "
            f"n_fev = {r['n_fev']}   escapees into pinned annulus = {r['n_escapees']}"
        )

    # Write relaxed configurations for downstream plotting
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r") as fin, h5py.File(out_h5, "w") as fout:
        for name, r in results.items():
            grp = fout.create_group(f"{name}_relaxed_N{n_target}")
            grp.create_dataset("points_initial", data=fin[f"{name}_N{n_target}"]["points"][:])
            grp.create_dataset("points_final", data=r["final_points"])
            grp.create_dataset("free_mask", data=r["free_mask"])
            grp.create_dataset("pinned_mask", data=r["pinned_mask"])
            grp.attrs["E_init"] = r["E_init"]
            grp.attrs["E_final"] = r["E_final"]
            grp.attrs["dE"] = r["dE"]
            grp.attrs["frac_dE"] = r["frac_dE"]
            grp.attrs["pin_fraction"] = pin_fraction
            grp.attrs["r_pin"] = r["r_pin"]
            grp.attrs["n_free"] = r["n_free"]
            grp.attrs["n_pinned"] = r["n_pinned"]
            grp.attrs["converged"] = r["converged"]
            grp.attrs["max_F_final_free"] = r["max_F_final_free"]
            grp.attrs["max_disp"] = r["max_disp"]
            grp.attrs["ang_init_mean_std"] = r["ang_init_mean_std"]
            grp.attrs["ang_final_mean_std"] = r["ang_final_mean_std"]
    print(f"\nWrote relaxed configurations to {out_h5}")

    # Summary verdict
    print()
    print("SUMMARY")
    print("-" * 70)
    print(f"{'lattice':>8} | {'dE':>12} | {'dE/|E|':>10} | {'dE/N_free':>12} | {'conv':>5}")
    for name in ("vogel", "hex", "square"):
        r = results[name]
        print(
            f"{name:>8} | {r['dE']:>12.4e} | {r['frac_dE']:>10.3e} | "
            f"{r['dE_per_free']:>12.3e} | {str(r['converged']):>5}"
        )
    print()
    print(
        "Interpretation: dE is the ONCE-OFF energy released by letting the "
        "interior relax under the pair kernel E_ij = -pi/(24 r_ij) with the "
        "outer annulus pinned. This is the only energy accessible under any "
        "(hypothetical) extraction scheme on this synthetic graph; F is "
        "conservative (P.4), so cyclic paths yield zero."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument(
        "--input", type=str, default="data/experiments/phyllotaxis_graph.h5"
    )
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HDF5 (default: data/experiments/phyllotaxis_relaxed_N{N}.h5)",
    )
    ap.add_argument(
        "--pin-fraction",
        type=float,
        default=0.6,
        help="Pin sites with r >= pin_fraction * r_bound",
    )
    ap.add_argument("--gtol", type=float, default=1e-7)
    ap.add_argument("--maxiter", type=int, default=5000)
    args = ap.parse_args()

    out_h5 = Path(args.output) if args.output else Path(
        f"data/experiments/phyllotaxis_relaxed_N{args.n}.h5"
    )
    run(Path(args.input), args.n, args.pin_fraction, args.gtol, args.maxiter, out_h5)


if __name__ == "__main__":
    main()
