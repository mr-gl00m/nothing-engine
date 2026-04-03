"""
Topology comparison experiment: closed vs periodic ringdown.

Runs identical physics (M=100, v0=0.01, N=128, form factor) with
two boundary conditions to test whether ring topology produces
qualitatively different vacuum friction.

Closed:   Dirichlet walls, standing waves, omega_n = n*pi/a
Periodic: Ring topology, traveling waves, omega_n = 2*n*pi/L

This is the paper's central result: does boundary topology change
the character of vacuum friction (e.g. power-law vs exponential)?

Usage:
    python -m nothing_engine.experiments.run_topology_comparison [--quick]
"""

import sys
import logging
from pathlib import Path

import numpy as np
import h5py

from nothing_engine.core.bogoliubov import SimulationConfig
from nothing_engine.experiments.runner import ExperimentRunner, RunConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def make_configs(mode: str):
    """Return list of (label, sim_cfg, run_cfg, output_path) tuples."""
    if mode == "quick":
        n_modes, mass, v0, total_time = 16, 100.0, 0.01, 500.0
        seg_time, ckpt_interval = 100.0, 250.0
    else:
        n_modes, mass, v0, total_time = 128, 100.0, 0.01, 5000.0
        seg_time, ckpt_interval = 500.0, 1000.0

    run_cfg = RunConfig(
        total_time=total_time,
        segment_time=seg_time,
        samples_per_unit_time=8,
        checkpoint_interval=ckpt_interval,
        log_interval_segments=2,
    )

    configs = []
    for boundary in ["closed", "periodic"]:
        sim_cfg = SimulationConfig(
            n_modes=n_modes,
            plate_mass=mass,
            spring_k=0.0,
            q0=1.0,
            v0=v0,
            x_left=0.0,
            boundary=boundary,
            rtol=1.0e-10,
            atol=1.0e-12,
            max_step=0.01,
        )
        suffix = "quick" if mode == "quick" else "light"
        output_path = f"data/experiments/topology_{boundary}_{suffix}.h5"
        configs.append((boundary, sim_cfg, run_cfg, output_path))

    return configs


def summarize_results(configs):
    """Print comparison table from completed HDF5 files."""
    print()
    print("=" * 70)
    print("  TOPOLOGY COMPARISON RESULTS")
    print("=" * 70)

    # Try to import ringdown fitting
    try:
        from nothing_engine.analysis.ringdown_fit import fit_ringdown
        has_fitter = True
    except ImportError:
        has_fitter = False

    results = []
    for label, sim_cfg, run_cfg, output_path in configs:
        path = Path(output_path)
        if not path.exists():
            print(f"  {label}: NO DATA")
            continue

        with h5py.File(path, "r") as f:
            t = f["timeseries"]["t"][:]
            e_plate = f["timeseries"]["E_plate"][:]
            n_particles = f["timeseries"]["total_particles"][:]
            e_total = f["timeseries"]["E_total"][:]

        E0 = e_plate[0]
        E_final = e_plate[-1]
        depletion = 1.0 - E_final / E0

        result = {
            "label": label,
            "E0": E0,
            "E_final": E_final,
            "depletion": depletion,
            "N_particles": n_particles[-1],
            "E_conserve": abs(e_total[-1] - e_total[0]) / abs(e_total[0]),
        }

        if has_fitter:
            try:
                fits = fit_ringdown(t, e_plate)
                best = min(fits, key=lambda x: x.get("aic", np.inf))
                result["best_model"] = best["model"]
                result["alpha"] = best.get("alpha", np.nan)
                result["tau"] = best.get("tau", np.nan)
                result["aic"] = best["aic"]
            except Exception:
                result["best_model"] = "fit_failed"

        results.append(result)

    if not results:
        print("  No results found.")
        return

    # Print table
    print(f" {'Boundary':>10} | {'E0':>12} | {'E_final':>12} | {'Depletion':>9} | {'Particles':>10} | {'E_conserve':>10}")
    print("-" * 80)
    for r in results:
        print(f" {r['label']:>10} | {r['E0']:>12.6e} | {r['E_final']:>12.6e} | {r['depletion']:>8.1%} | {r['N_particles']:>10.4e} | {r['E_conserve']:>10.2e}")

    if has_fitter and any("best_model" in r for r in results):
        print()
        print(f" {'Boundary':>10} | {'Best Model':>15} | {'alpha':>10} | {'tau':>12} | {'AIC':>14}")
        print("-" * 75)
        for r in results:
            if "best_model" in r:
                print(f" {r['label']:>10} | {r['best_model']:>15} | {r.get('alpha', np.nan):>10.4f} | {r.get('tau', np.nan):>12.2f} | {r.get('aic', np.nan):>14.2f}")


def main():
    mode = "quick" if "--quick" in sys.argv else "standard"

    configs = make_configs(mode)

    for label, sim_cfg, run_cfg, output_path in configs:
        e0 = 0.5 * sim_cfg.plate_mass * sim_cfg.v0**2
        print("=" * 60)
        print(f"  Topology: {label.upper()}")
        print(f"  N_modes={sim_cfg.n_modes}, M={sim_cfg.plate_mass}, v0={sim_cfg.v0}")
        print(f"  E_plate(0) = {e0:.6e}")
        print(f"  t_max = {run_cfg.total_time:.0f}")
        print(f"  boundary = {sim_cfg.boundary}")
        print(f"  Output: {output_path}")
        print("=" * 60)

        runner = ExperimentRunner(sim_cfg, run_cfg, output_path=output_path)
        path = runner.run()
        print(f"  Done -> {path}")
        print()

    summarize_results(configs)


if __name__ == "__main__":
    main()
