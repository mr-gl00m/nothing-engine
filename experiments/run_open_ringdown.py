"""Large finite cavity proxy retained for historical reproducibility.

This entry point does not implement an open boundary, an external continuum, or
outgoing flux. It places the moving coordinate at the end of a larger closed
Dirichlet interval. Reflections remain in the model. Results can probe box size
sensitivity in the diagonal approximation and cannot establish radiation loss.

Usage:
    python -m experiments.run_open_ringdown [--quick]
"""

import sys
import logging
from pathlib import Path

from nothing_engine.core.bogoliubov import SimulationConfig
from nothing_engine.experiments.runner import ExperimentRunner, RunConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def main():
    mode = "full"
    if "--quick" in sys.argv:
        mode = "quick"
    elif "--light" in sys.argv:
        mode = "light"

    # Large finite box sensitivity study.
    # Mode frequencies are omega_n = n*pi/L_domain, so we need
    # N_modes_open = N_modes_closed * (L_domain / a0) to match UV cutoff.

    if mode == "light":
        # Light plate in a larger finite interval.
        L_domain = 20.0
        n_modes_open = 128 * 20  # 2560 modes to match closed UV cutoff
        sim_cfg = SimulationConfig(
            n_modes=n_modes_open,
            plate_mass=100.0,
            spring_k=0.0,
            q0=L_domain,
            v0=0.01,
            x_left=0.0,
            rtol=1.0e-10,
            atol=1.0e-12,
            max_step=0.005,
        )
        run_cfg = RunConfig(
            total_time=5000.0,
            segment_time=250.0,
            samples_per_unit_time=8,
            checkpoint_interval=1000.0,
            log_interval_segments=4,
        )
        output_path = "data/experiments/open_ringdown_light.h5"
    elif mode == "quick":
        L_domain = 5.0
        n_modes_open = 16 * 5  # 80 modes for quick test
        sim_cfg = SimulationConfig(
            n_modes=n_modes_open,
            plate_mass=1.0e4,
            spring_k=0.0,
            q0=L_domain,
            v0=1.0e-3,
            x_left=0.0,
            rtol=1.0e-10,
            atol=1.0e-12,
            max_step=0.01,
        )
        run_cfg = RunConfig(
            total_time=1000.0,
            segment_time=200.0,
            checkpoint_interval=500.0,
        )
        output_path = "data/experiments/open_ringdown_quick.h5"
    else:
        L_domain = 10.0
        n_modes_open = 256 * 10  # 2560 modes to match UV cutoff
        sim_cfg = SimulationConfig(
            n_modes=n_modes_open,
            plate_mass=1.0e4,
            spring_k=0.0,
            q0=L_domain,
            v0=1.0e-3,
            x_left=0.0,
            rtol=1.0e-10,
            atol=1.0e-12,
            max_step=0.005,  # Tighter step for higher mode count
        )
        run_cfg = RunConfig(
            total_time=1.0e5,
            segment_time=500.0,        # Shorter segments (bigger state)
            samples_per_unit_time=8,
            checkpoint_interval=1.0e4,
            log_interval_segments=10,
        )
        output_path = "data/experiments/open_ringdown.h5"

    print(f"=== Large Finite Cavity Proxy ({mode}) ===")
    print("  WARNING: this is a closed box and has no outgoing flux boundary.")
    print(f"  Modes: {sim_cfg.n_modes}")
    print(f"  Domain size: {sim_cfg.q0}")
    print(f"  Mass: {sim_cfg.plate_mass}")
    print(f"  v0: {sim_cfg.v0}")
    print(f"  E_plate(0): {0.5 * sim_cfg.plate_mass * sim_cfg.v0**2:.6e}")
    print(f"  Total time: {run_cfg.total_time:.0f}")
    print(f"  Output: {output_path}")
    print()

    runner = ExperimentRunner(sim_cfg, run_cfg, output_path=output_path)
    path = runner.run()
    print(f"\nDone. Output saved to: {path}")


if __name__ == "__main__":
    main()
