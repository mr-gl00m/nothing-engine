"""
Mode convergence test for closed-system light plate.

Runs the same light-plate experiment (M=100, v0=0.01, t=5000) at
multiple mode counts to verify that the power-law exponent alpha
and decay timescale tau are converged.

Usage:
    python -m nothing_engine.experiments.run_convergence
"""

import sys
import logging
import time
import numpy as np

from nothing_engine.core.bogoliubov import SimulationConfig
from nothing_engine.experiments.runner import ExperimentRunner, RunConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def run_single(n_modes, total_time=5000.0):
    """Run one convergence test and return the output path."""
    sim_cfg = SimulationConfig(
        n_modes=n_modes,
        plate_mass=100.0,
        spring_k=0.0,
        q0=1.0,
        v0=0.01,
        x_left=0.0,
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=0.01,
    )
    run_cfg = RunConfig(
        total_time=total_time,
        segment_time=500.0,
        samples_per_unit_time=8,
        checkpoint_interval=1000.0,  # intermediate checkpoints for crash safety
        log_interval_segments=5,
    )
    output_path = f"data/experiments/convergence_N{n_modes}.h5"

    print(f"\n{'='*60}")
    print(f"  Convergence run: N_modes = {n_modes}")
    print(f"  M=100, v0=0.01, t={total_time:.0f}")
    print(f"{'='*60}")

    runner = ExperimentRunner(sim_cfg, run_cfg, output_path=output_path)
    t0 = time.perf_counter()
    path = runner.run()
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s -> {path}")
    return str(path)


def analyze_results(paths):
    """Fit power law to each run and compare alpha, tau."""
    try:
        from nothing_engine.analysis.ringdown_fit import (
            fit_ringdown, load_ringdown_data, power_law_model,
        )
    except ImportError:
        print("\nCannot import analysis module; skipping fit comparison.")
        return

    print(f"\n{'='*60}")
    print("  CONVERGENCE RESULTS")
    print(f"{'='*60}")
    print(f"{'N_modes':>8} | {'alpha':>10} | {'tau':>12} | {'E0':>12} | {'AIC':>12} | {'E_final':>12}")
    print("-" * 80)

    for path in paths:
        t, e_plate = load_ringdown_data(path)
        results = fit_ringdown(t, e_plate)

        # Extract mode count from filename
        import re
        m = re.search(r'N(\d+)', path)
        n_modes = m.group(1) if m else "?"

        if results.power_law and results.power_law.converged:
            pl = results.power_law
            print(f"{n_modes:>8} | {pl.params['alpha']:>10.4f} | "
                  f"{pl.params['tau']:>12.2f} | {pl.params['E0']:>12.6e} | "
                  f"{pl.aic:>12.2f} | {e_plate[-1]:>12.6e}")
        else:
            print(f"{n_modes:>8} | {'FAILED':>10} |")


def main():
    mode_counts = [32, 64, 128, 256]
    if "--extended" in sys.argv:
        mode_counts = [32, 64, 128, 256, 512]
    elif "--quick" in sys.argv:
        mode_counts = [32, 64, 128]

    paths = []
    for n in mode_counts:
        p = run_single(n)
        paths.append(p)

    analyze_results(paths)


if __name__ == "__main__":
    main()
