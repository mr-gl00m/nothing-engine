"""
P.4 — Force/energy implementation-consistency gate for the phyllotaxis
Casimir-graph experiment.

Physics background: F_i is defined as -grad_i E({r_j}) from the pairwise
potential E_ij = -pi/(24 r_ij). Conservativity of F is true analytically
by construction (F is the gradient of a scalar). The point of this gate
is NOT to "discover" that F is conservative but to verify that the
analytic force implementation matches the finite-difference gradient of
the analytic energy implementation.

Two checks:

1. Site-gradient consistency. For each lattice site i, compare
   F_analytic_i against -(E(r_i + eps*e_x) - E(r_i - eps*e_x))/(2 eps)
   and similarly for e_y. Centered-difference error is O(eps^2) on top
   of machine precision, so with eps=1e-5 and F ~ 1e-2 we expect
   |F_analytic - F_FD|/|F_analytic| well under 1e-6.

   Both the analytic and the FD paths here use the FULL O(N^2) pair sum
   (no r_cut). This avoids a subtle cutoff-boundary inconsistency: if
   FD uses r_cut + pad and analytic uses strict r_cut, a pair right at
   r_cut (hex shell #7 lives at exactly 4*a) gets counted asymmetrically.

2. Closed-loop line-integral check. For a small square loop in the
   lattice interior, well clear of any lattice site, compute the oriented
   line integral of F_field . dl. For a conservative field this is zero
   up to FD truncation. We avoid sampling F near its 1/r^2 cusps at
   lattice sites by choosing loops that do not enclose any site; the
   earlier grid-curl check sampled near cusps and was dominated by FD
   error, not physics.

Usage:
    python -m nothing_engine.experiments.val_phyllotaxis_consistency [--n 200]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import h5py
from scipy.spatial import cKDTree


def total_energy_full(points: np.ndarray) -> float:
    """Full O(N^2) pair sum. No cutoff, no boundary ambiguity."""
    n = points.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    diffs = points[iu] - points[ju]
    r = np.linalg.norm(diffs, axis=1)
    r = np.maximum(r, 1e-12)
    return float(np.sum(-np.pi / (24.0 * r)))


def analytic_site_forces_full(points: np.ndarray) -> np.ndarray:
    """Analytic F_i using all pairs.

    F_i = -grad_i E = -grad_i sum_{j != i} -pi/(24 |r_i - r_j|)
        = sum_{j != i} (pi/24) * (r_j - r_i) / |r_i - r_j|^3
    """
    n = points.shape[0]
    F = np.zeros_like(points)
    for i in range(n):
        diffs = np.delete(points, i, axis=0) - points[i]
        r = np.maximum(np.linalg.norm(diffs, axis=1), 1e-12)
        F[i] = np.sum(diffs * (np.pi / (24.0 * r**3))[:, None], axis=0)
    return F


def fd_site_gradient_full(points: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Centered-difference -grad_i E using full pair sum."""
    n = points.shape[0]
    grad = np.zeros_like(points)
    for i in range(n):
        for axis in (0, 1):
            plus = points.copy()
            plus[i, axis] += eps
            e_plus = total_energy_full(plus)
            minus = points.copy()
            minus[i, axis] -= eps
            e_minus = total_energy_full(minus)
            grad[i, axis] = -(e_plus - e_minus) / (2.0 * eps)
    return grad


def field_force_full(points: np.ndarray, probe: np.ndarray) -> np.ndarray:
    """Force at probe position due to all lattice sites (no r_cut)."""
    diffs = points - probe[None, :]
    r = np.linalg.norm(diffs, axis=1)
    r_safe = np.maximum(r, 1e-12)
    return np.sum(diffs * (np.pi / (24.0 * r_safe**3))[:, None], axis=0)


def find_safe_loop_center(
    points: np.ndarray, extent: float, n_candidates: int = 2000, rng_seed: int = 0
) -> tuple[np.ndarray, float]:
    """Return the candidate point inside [-extent, extent]^2 that is farthest
    from any lattice site. Also returns that minimum distance.
    """
    rng = np.random.default_rng(rng_seed)
    tree = cKDTree(points)
    cands = rng.uniform(-extent, extent, size=(n_candidates, 2))
    d, _ = tree.query(cands, k=1)
    idx = int(np.argmax(d))
    return cands[idx], float(d[idx])


def line_integral_square(
    points: np.ndarray, center: np.ndarray, side: float, n_per_side: int = 40
) -> tuple[float, float]:
    """Oriented line integral of F around a square loop. Returns (integral, max_|F|_on_loop).
    For a conservative field (F = -grad U) this is zero up to discretization error.
    """
    half = side / 2.0
    corners = np.array(
        [
            [center[0] - half, center[1] - half],
            [center[0] + half, center[1] - half],
            [center[0] + half, center[1] + half],
            [center[0] - half, center[1] + half],
            [center[0] - half, center[1] - half],
        ]
    )
    total = 0.0
    max_F = 0.0
    for k in range(4):
        p0, p1 = corners[k], corners[k + 1]
        ts = np.linspace(0.0, 1.0, n_per_side + 1)
        pts = p0 + (p1 - p0)[None, :] * ts[:, None]
        Fs = np.array([field_force_full(points, p) for p in pts])
        max_F = max(max_F, float(np.max(np.linalg.norm(Fs, axis=1))))
        dl = (p1 - p0) / n_per_side
        Fdotdir = Fs @ dl
        total += 0.5 * (Fdotdir[0] + Fdotdir[-1]) + float(np.sum(Fdotdir[1:-1]))
    return float(total), max_F


def run_for_lattice(h5_path: Path, name: str, n_target: int, eps: float) -> dict:
    with h5py.File(h5_path, "r") as f:
        key = f"{name}_N{n_target}"
        if key not in f:
            raise KeyError(f"Missing {key} in {h5_path}")
        grp = f[key]
        points = grp["points"][:]
        r_bound = float(grp.attrs["r_bound"])

    # Full O(N^2) site-grad consistency — no cutoff-boundary games.
    F_analytic = analytic_site_forces_full(points)
    F_fd = fd_site_gradient_full(points, eps=eps)

    # Use all sites (no r_cut, so no boundary to exclude; the lattice is
    # still finite, but the analytic and FD paths see the same finite patch).
    diff = F_analytic - F_fd
    diff_mag = np.linalg.norm(diff, axis=1)
    F_mag = np.linalg.norm(F_analytic, axis=1)
    mask = F_mag > 1e-15
    rel_err = np.full_like(diff_mag, np.nan)
    rel_err[mask] = diff_mag[mask] / F_mag[mask]

    max_abs_err = float(np.max(diff_mag))
    median_abs_err = float(np.median(diff_mag))

    # Closed-loop line-integral check in a site-free patch of the interior.
    # Pick a loop center by maximin over a random candidate set, then size
    # the loop to half the gap to the nearest site so we stay clear of cusps.
    extent = max(0.3 * r_bound, 1.5)
    center, gap = find_safe_loop_center(points, extent, n_candidates=2000, rng_seed=0)
    side = 0.5 * gap
    loop_int, max_F_loop = line_integral_square(points, center, side, n_per_side=80)
    loop_ratio = abs(loop_int) / (max_F_loop * 4 * side) if max_F_loop > 0 else float("nan")

    return {
        "name": name,
        "n_points": int(points.shape[0]),
        "n_sites": int(points.shape[0]),
        "max_abs_err": max_abs_err,
        "median_abs_err": median_abs_err,
        "max_rel_err": float(np.nanmax(rel_err)) if np.any(mask) else float("nan"),
        "median_rel_err": float(np.nanmedian(rel_err)) if np.any(mask) else float("nan"),
        "loop_center": center.tolist(),
        "loop_side": side,
        "loop_gap": gap,
        "loop_integral": loop_int,
        "loop_max_F": max_F_loop,
        "loop_ratio": loop_ratio,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument(
        "--input", type=str, default="data/experiments/phyllotaxis_graph.h5"
    )
    args = ap.parse_args()

    print(f"P.4 consistency gate  (N = {args.n}, FD eps = {args.eps:.1e})")
    print("=" * 100)
    print(
        f"{'lattice':>8} | {'N':>4} | {'max_abs_err':>12} | {'median_abs':>12} | "
        f"{'max_rel_err':>12} | {'loop_side':>10} | {'loop_integral':>14} | {'loop_ratio':>12}"
    )
    print("-" * 100)
    rows = []
    for name in ("vogel", "hex", "square"):
        r = run_for_lattice(Path(args.input), name, args.n, args.eps)
        rows.append(r)
        print(
            f"{r['name']:>8} | {r['n_sites']:>4d} | "
            f"{r['max_abs_err']:>12.3e} | {r['median_abs_err']:>12.3e} | "
            f"{r['max_rel_err']:>12.3e} | "
            f"{r['loop_side']:>10.3e} | {r['loop_integral']:>14.3e} | {r['loop_ratio']:>12.3e}"
        )

    print()
    print("GATE VERDICT")
    print("-" * 70)
    any_fail = False
    for r in rows:
        if r["name"] == "vogel":
            thr = 1e-5
            val = r["max_rel_err"]
            ok = np.isfinite(val) and val < thr
            verdict = "PASS" if ok else "FAIL"
            print(
                f"  {r['name']}: max relative grad error {val:.3e}  (threshold {thr:.1e})  {verdict}"
            )
        else:
            thr = 1e-8
            val = r["max_abs_err"]
            ok = val < thr
            verdict = "PASS" if ok else "FAIL"
            print(
                f"  {r['name']}: max absolute grad error {val:.3e}  (threshold {thr:.1e})  {verdict}"
            )
        any_fail = any_fail or (not ok)

        thr_loop = 1e-4
        val_loop = r["loop_ratio"]
        ok_loop = np.isfinite(val_loop) and val_loop < thr_loop
        verdict_loop = "PASS" if ok_loop else "FAIL"
        print(
            f"      loop |oint F.dl|/(|F|_max * perim) = {val_loop:.3e}  "
            f"(threshold {thr_loop:.1e}, loop side={r['loop_side']:.3e}, gap={r['loop_gap']:.3e})  "
            f"{verdict_loop}"
        )
        any_fail = any_fail or (not ok_loop)
    print()
    print("OVERALL:", "FAIL" if any_fail else "PASS")


if __name__ == "__main__":
    main()
