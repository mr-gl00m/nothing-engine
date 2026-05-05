"""
Post-run analysis for run_convergence.py output.

Beyond the fitted (alpha, tau, E0) that run_convergence prints, this
script answers the two questions the prior halo attempt could not:

1. Did the cavity stay open? Report min(q(t)) per run. Prior N=256 attempt
   collapsed to q -> 0 under UV-divergent static Casimir well. With the
   form factor + per-mode vacuum subtraction, q should stay comfortably
   above zero.

2. How cleanly is energy conserved? Max |E_total(t) - E_total(0)| normalized
   by max(E_plate_0, |E_Casimir(a0)|). The Casimir-scale denominator is
   the load-bearing one in the coupled ODE (see Gate 4.3 rationale).

3. Is there UV mode pileup? Report max particle number across modes and
   max tail ratio (sum of top-10%-by-index |beta|^2 vs total). A healthy
   converged run has the bulk of particle production in low-to-mid modes;
   UV pileup shows up as the tail growing faster than the sum.

Usage:
    python -m nothing_engine.experiments.analyze_convergence

Expects data/experiments/convergence_N{32,64,128,256}.h5 to exist.
"""

from __future__ import annotations

import sys
import glob
import re
from pathlib import Path

import h5py
import numpy as np

from nothing_engine.analysis.ringdown_fit import fit_ringdown, load_ringdown_data
from nothing_engine.core.constants import casimir_energy_1d


def diagnose_run(path: str) -> dict:
    with h5py.File(path, "r") as f:
        t = f["timeseries/t"][:]
        q = f["timeseries/plate_q"][:]
        e_plate = f["timeseries/E_plate"][:]
        e_total = f["timeseries/E_total"][:]
        particles = f["timeseries/particle_number"][:]  # (T, N)
        cfg = f["config"]
        n_modes = int(cfg.attrs["n_modes"])
        q0 = float(cfg.attrs["q0"])
        v0 = float(cfg.attrs["v0"])
        plate_mass = float(cfg.attrs["plate_mass"])

    e_plate_0 = 0.5 * plate_mass * v0 * v0
    e_casimir = abs(casimir_energy_1d(q0))
    scale = max(e_plate_0, e_casimir)

    a_min = float(np.min(q))
    a_min_frac = a_min / q0
    e_drift_abs = float(np.max(np.abs(e_total - e_total[0])))
    e_drift_rel = e_drift_abs / scale

    # UV pileup: split modes into bottom 90% and top 10% by index.
    n_tail = max(1, n_modes // 10)
    particles_final = particles[-1]
    total_part_final = float(np.sum(particles_final))
    tail_part_final = float(np.sum(particles_final[-n_tail:]))
    tail_frac = tail_part_final / total_part_final if total_part_final > 0 else 0.0
    peak_mode = int(np.argmax(particles_final))
    max_particle_any_mode = float(np.max(particles_final))

    return {
        "path": path,
        "n_modes": n_modes,
        "a_min": a_min,
        "a_min_frac": a_min_frac,
        "e_drift_abs": e_drift_abs,
        "e_drift_rel_casimir_scale": e_drift_rel,
        "total_particles_final": total_part_final,
        "tail_10pct_fraction": tail_frac,
        "peak_mode": peak_mode,
        "peak_mode_particles": max_particle_any_mode,
        "t": t,
        "e_plate": e_plate,
    }


def print_diagnostics(rows: list[dict]) -> None:
    print()
    print("=" * 100)
    print("  CAVITY & ENERGY DIAGNOSTICS")
    print("=" * 100)
    print(
        f"{'N':>4} | {'a_min':>10} | {'a_min/a0':>10} | "
        f"{'|dE|_max':>12} | {'|dE|/scale':>12} | "
        f"{'N_particles':>12} | {'tail_10%':>10} | {'peak_mode':>10}"
    )
    print("-" * 100)
    for r in rows:
        print(
            f"{r['n_modes']:>4} | {r['a_min']:>10.4e} | {r['a_min_frac']:>10.4f} | "
            f"{r['e_drift_abs']:>12.4e} | {r['e_drift_rel_casimir_scale']:>12.4e} | "
            f"{r['total_particles_final']:>12.4e} | {r['tail_10pct_fraction']:>10.4f} | "
            f"{r['peak_mode']:>10d}"
        )
    print()
    # Judgement: the prior failure was cavity collapse at N=256.
    worst_collapse = min(r["a_min_frac"] for r in rows)
    worst_drift = max(r["e_drift_rel_casimir_scale"] for r in rows)
    worst_tail = max(r["tail_10pct_fraction"] for r in rows)
    print(f"  worst a_min/a0 across runs: {worst_collapse:.4f}  "
          f"({'COLLAPSE' if worst_collapse < 0.1 else 'OPEN'})")
    print(f"  worst energy drift vs Casimir scale: {worst_drift:.4e}  "
          f"({'FAIL' if worst_drift > 1e-5 else 'OK'})")
    print(f"  worst UV-tail particle fraction: {worst_tail:.4f}  "
          f"({'UV PILEUP' if worst_tail > 0.3 else 'OK'})")


def print_fits(rows: list[dict]) -> None:
    print("=" * 100)
    print("  RINGDOWN FITS (existing analysis, reproduced for convenience)")
    print("=" * 100)
    print(f"{'N':>4} | {'alpha':>10} | {'tau':>12} | {'E0':>12} | "
          f"{'E_final':>12} | {'E_final/E_0':>12}")
    print("-" * 100)
    for r in rows:
        t, e_plate = r["t"], r["e_plate"]
        res = fit_ringdown(t, e_plate)
        e_final = e_plate[-1]
        e_0 = e_plate[0] if e_plate[0] > 0 else 1.0
        if res.power_law and res.power_law.converged:
            p = res.power_law.params
            print(
                f"{r['n_modes']:>4} | {p['alpha']:>10.4f} | {p['tau']:>12.2f} | "
                f"{p['E0']:>12.4e} | {e_final:>12.4e} | {e_final/e_0:>12.4f}"
            )
        else:
            print(
                f"{r['n_modes']:>4} | {'FAIL':>10} | {'-':>12} | {'-':>12} | "
                f"{e_final:>12.4e} | {e_final/e_0:>12.4f}"
            )


def main():
    paths = sorted(
        glob.glob("data/experiments/convergence_N*.h5"),
        key=lambda p: int(re.search(r"N(\d+)", p).group(1)),
    )
    if not paths:
        print("No convergence files found at data/experiments/convergence_N*.h5",
              file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(paths)} convergence runs: {[Path(p).name for p in paths]}")

    rows = [diagnose_run(p) for p in paths]
    print_fits(rows)
    print_diagnostics(rows)

    # Alpha convergence judgement.
    alphas = []
    for r in rows:
        res = fit_ringdown(r["t"], r["e_plate"])
        if res.power_law and res.power_law.converged:
            alphas.append((r["n_modes"], res.power_law.params["alpha"]))

    if len(alphas) >= 2:
        print()
        print("=" * 100)
        print("  ALPHA CONVERGENCE")
        print("=" * 100)
        for (n1, a1), (n2, a2) in zip(alphas[:-1], alphas[1:]):
            d_rel = abs(a2 - a1) / max(abs(a1), 1e-12)
            print(f"  alpha(N={n2}) vs alpha(N={n1}): "
                  f"delta = {a2 - a1:+.4e}  rel = {d_rel:.2%}")
        last_delta = abs(alphas[-1][1] - alphas[-2][1]) / max(abs(alphas[-2][1]), 1e-12)
        print(f"\n  final-step relative change: {last_delta:.2%}  "
              f"({'CONVERGED' if last_delta < 0.1 else 'NOT CONVERGED'} at 10% threshold)")


if __name__ == "__main__":
    main()
