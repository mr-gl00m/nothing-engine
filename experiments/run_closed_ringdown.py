"""
Phase 2, Task 2.1 — Closed-system ringdown (Track B, 10^5 cycles).

Closed cavity: N_modes=256, M=1e4, v0=1e-3, k=0.
No spring, no open boundary — pure vacuum friction in a closed box.

Expected outcome: plate kinetic energy depletes via dynamical Casimir
photon creation. Ringdown curve saved to HDF5 for fitting.

Usage:
    python -m experiments.run_closed_ringdown [--quick]
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
    elif "--medium" in sys.argv:
        mode = "medium"

    if mode == "quick":
        # Short test run: 1000 time units, 16 modes
        sim_cfg = SimulationConfig(
            n_modes=16,
            plate_mass=1.0e4,
            spring_k=0.0,
            q0=1.0,
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
        output_path = "data/experiments/closed_ringdown_quick.h5"
    elif mode == "light":
        # Light plate: strong coupling, clean exponential decay
        # M=100 (100x lighter), v0=0.01 (10x faster), 128 modes, t=5000
        sim_cfg = SimulationConfig(
            n_modes=128,
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
            total_time=5000.0,
            segment_time=500.0,
            samples_per_unit_time=8,
            checkpoint_interval=1000.0,
            log_interval_segments=2,
        )
        output_path = "data/experiments/closed_ringdown_light.h5"
    elif mode == "medium":
        # Medium run: 128 modes, heavy plate, long duration
        sim_cfg = SimulationConfig(
            n_modes=128,
            plate_mass=1.0e4,
            spring_k=0.0,
            q0=1.0,
            v0=1.0e-3,
            x_left=0.0,
            rtol=1.0e-10,
            atol=1.0e-12,
            max_step=0.01,
        )
        run_cfg = RunConfig(
            total_time=1.0e4,
            segment_time=1000.0,
            samples_per_unit_time=8,
            checkpoint_interval=5000.0,
            log_interval_segments=5,
        )
        output_path = "data/experiments/closed_ringdown_medium.h5"
    else:
        # Full science run per PRE_REGISTRATION.md §5.1
        sim_cfg = SimulationConfig(
            n_modes=256,
            plate_mass=1.0e4,
            spring_k=0.0,
            q0=1.0,
            v0=1.0e-3,
            x_left=0.0,
            rtol=1.0e-10,
            atol=1.0e-12,
            max_step=0.01,
        )
        run_cfg = RunConfig(
            total_time=1.0e5,
            segment_time=1000.0,
            samples_per_unit_time=8,
            checkpoint_interval=1.0e4,
            log_interval_segments=10,
        )
        output_path = "data/experiments/closed_ringdown.h5"

    print(f"=== Closed-System Ringdown ({mode}) ===")
    print(f"  Modes: {sim_cfg.n_modes}")
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
