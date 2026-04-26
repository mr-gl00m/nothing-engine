"""
Phase 2, Task 2.2 — "Open"-system ringdown (Track B, 10^5 cycles).

LARGE-BOX APPROXIMATION, not a true open boundary. The current physics
layer is a closed Dirichlet cavity between x_left and the plate at q(t).
This script simulates a cavity that is simply much wider (q0 = L_domain,
x_left = 0), with a proportionally larger mode count to hold the UV
cutoff fixed. Long-wavelength modes (wavelength >> a0_physical) see a
nearly translation-invariant vacuum and approximate an "open" environment
over finite times, but there is no absorbing/radiative boundary condition
at x = L_domain — modes still reflect off the far wall once a photon has
had time to traverse the box. For T_run << L_domain / c the reflections
are cosmetic; for T_run comparable to or greater than L_domain, the
"open" regime breaks down.

Comparing this directly against run_closed_ringdown.py conflates:
    - boundary topology (closed Dirichlet vs approximate open)
    - cavity width (1.0 vs L_domain)
    - mode density per unit length (same by construction)
The width mismatch is the big one. Casimir force scales as 1/a^2 so the
wider cavity has a much weaker static force; any observed difference in
"friction" has to be divided out from this before attributing to topology.

Per PRE_REGISTRATION.md §5.1:
    Same parameters as closed, extended mode space for large-box approximation.

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

    # Open-system strategy: large-box approximation.
    # Physical cavity width a0 = 1.0, but domain is L_domain.
    # Mode frequencies are omega_n = n*pi/L_domain, so we need
    # N_modes_open = N_modes_closed * (L_domain / a0) to match UV cutoff.

    if mode == "light":
        # Light plate, large domain — matches closed light-plate for comparison
        L_domain = 20.0  # 20x cavity width for clean radiation escape
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

    print(f"=== Open-System Ringdown ({mode}) ===")
    print("  WARNING: This is a LARGE-BOX APPROXIMATION, not a true")
    print("  open/absorbing boundary. Cavity width differs from the closed")
    print("  ringdown run — any direct comparison must account for the 1/a^2")
    print("  Casimir scaling difference. See module docstring.")
    print(f"  Modes: {sim_cfg.n_modes}")
    print(f"  Domain size (= cavity width here): {sim_cfg.q0}")
    print(f"  Light-crossing time L/c: {sim_cfg.q0:.2f} "
          f"(open-regime approximation breaks beyond T ~ this)")
    print(f"  Mass: {sim_cfg.plate_mass}")
    print(f"  v0: {sim_cfg.v0}")
    print(f"  E_plate(0): {0.5 * sim_cfg.plate_mass * sim_cfg.v0**2:.6e}")
    print(f"  Total time: {run_cfg.total_time:.0f}")
    print(f"  Output: {output_path}")
    if run_cfg.total_time > sim_cfg.q0 * 10.0:
        print(f"  NOTE: total_time ({run_cfg.total_time:.0f}) >> L "
              f"({sim_cfg.q0:.1f}); far-wall reflections will contaminate late times.")
    print()

    runner = ExperimentRunner(sim_cfg, run_cfg, output_path=output_path)
    path = runner.run()
    print(f"\nDone. Output saved to: {path}")


if __name__ == "__main__":
    main()
