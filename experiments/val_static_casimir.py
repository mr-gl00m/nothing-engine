"""Validation Gate 4.1: static Casimir energy and force.

This gate exercises the production energy and force paths. It checks that an
initialized vacuum state has zero excitation energy, the correct regularized
static interaction energy, and the matching mechanical force. Both supported
boundary conditions are tested across several finite mode counts.
"""

import sys

from nothing_engine.core import constants
from nothing_engine.core.bogoliubov import (
    SimulationConfig,
    build_initial_state,
    make_rhs,
)
from nothing_engine.core import energy


def run_validation() -> bool:
    cavity_size = 1.25
    mode_counts = [8, 32, 128, 512]
    tolerance = 2.0e-11
    passed = True

    print("=" * 72)
    print("Gate 4.1: static Casimir energy and force")
    print("=" * 72)
    print("boundary  modes       E_error       F_error  excitation")

    for boundary in ("closed", "periodic"):
        expected_energy = constants.casimir_energy_1d(cavity_size, boundary)
        expected_force = constants.casimir_force_1d(cavity_size, boundary)

        for n_modes in mode_counts:
            cfg = SimulationConfig(
                n_modes=n_modes,
                plate_mass=7.0,
                spring_k=0.0,
                q0=cavity_size,
                v0=0.0,
                boundary=boundary,
                plate_thickness=0.0,
                t_span=(0.0, 0.1),
                audit_halt=False,
            )
            state = build_initial_state(cfg)
            mode_state = state[:4 * n_modes]
            components = energy.energy_components(
                mode_state, n_modes, cavity_size,
                cfg.plate_mass, 0.0,
                cfg.spring_k, cfg.q0, cfg.q_eq,
                cfg.g_n, cfg.ns_pi,
                cfg.boundary, cfg.mode_degeneracy,
            )
            derivative = make_rhs(cfg)(0.0, state)
            measured_force = cfg.plate_mass * derivative[-1]

            energy_error = abs(components["E_field"] - expected_energy)
            force_error = abs(measured_force - expected_force)
            excitation = abs(components["E_excitation"])
            row_passed = max(energy_error, force_error, excitation) < tolerance
            passed = passed and row_passed

            print(
                f"{boundary:>8} {n_modes:>6d}  "
                f"{energy_error:>12.3e}  {force_error:>12.3e}  "
                f"{excitation:>10.3e}"
            )

    print()
    print(f"GATE 4.1: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    sys.exit(0 if run_validation() else 1)
