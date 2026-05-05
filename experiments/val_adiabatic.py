"""
Validation Gate 4.6: Adiabatic Limit

Verify that a very heavy, slow plate creates negligible particles.
In the adiabatic limit (omega_plate << omega_1), the field adjusts
instantaneously and no real particles are created.

Method:
    Heavy plate (M = 1e8), tiny velocity (v0 = 1e-6).
    Run for T = 50. Check total particle number.

PASS: Total N(T) < 1e-6 (100x below resonant case).
"""

import sys
import numpy as np

from nothing_engine.core import mode_space
from nothing_engine.core.bogoliubov import SimulationConfig, run_simulation
from nothing_engine.config import get_gate_criterion

_GATE = "gate_4_6_adiabatic_limit"


def run_validation():
    print("=" * 60)
    print("Gate 4.6: Adiabatic Limit — Negligible Particle Creation")
    print("=" * 60)

    # YAML is the source of truth. Criteria live in validation_criteria.yaml.
    N = int(get_gate_criterion(_GATE, "N_modes", default=32))
    M_heavy = float(get_gate_criterion(_GATE, "plate_mass", default=1.0e6))
    v0_slow = float(get_gate_criterion(_GATE, "initial_velocity", default=1.0e-6))
    duration = float(get_gate_criterion(_GATE, "duration", default=50.0))
    particle_threshold = float(
        get_gate_criterion(_GATE, "pass_criterion_total_particles", default=1.0e-6)
    )
    a0 = 1.0

    cfg = SimulationConfig(
        n_modes=N,
        plate_mass=M_heavy,
        spring_k=0.0,
        q0=a0,
        v0=v0_slow,
        t_span=(0.0, duration),
        rtol=1e-12,
        atol=1e-14,
        max_step=0.0,  # auto Nyquist-safe
        audit_halt=False,
    )

    omega_1 = np.pi / a0
    omega_plate = v0_slow / a0  # characteristic plate frequency
    print(f"Parameters: N={N}, M={M_heavy:.0e}, v0={v0_slow:.0e}")
    print(f"omega_1 = {omega_1:.4f}")
    print(f"omega_plate ~ v0/a0 = {omega_plate:.4e}")
    print(f"Adiabaticity: omega_plate / omega_1 = {omega_plate/omega_1:.4e}")
    print()

    print("Running simulation...")
    result = run_simulation(cfg)
    print(f"Completed: {len(result.t)} time points")
    print()

    # Extract particle number
    N_total_final = result.total_particle_number_at(-1)
    pn_final = result.particle_number_at(-1)

    print(f"Total particles at T={result.t[-1]:.1f}: N = {N_total_final:.4e}")
    print(f"Mode 1: |beta_1|^2 = {pn_final[0]:.4e}")
    print(f"Max per-mode: {np.max(np.abs(pn_final)):.4e}")

    # Plate displacement check
    q_final = result.plate_q[-1]
    displacement = abs(q_final - a0)
    print(f"\nPlate displacement: |q(T) - q0| = {displacement:.4e}")
    print(f"Relative: {displacement/a0:.4e}")

    passed = abs(N_total_final) < particle_threshold
    print(f"\nThreshold (from {_GATE}.pass_criterion_total_particles): {particle_threshold:.2e}")
    print(f"GATE 4.6: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
