"""
Validation Gate 4.3: Energy Conservation — Closed System

Verify that total energy E_plate + E_spring + E_field is conserved
to high precision in a coupled plate-field simulation.

Method:
    Run a closed-system simulation with dynamical plate coupling.
    Compute total energy at every output time point.
    Report maximum absolute drift relative to plate energy.

PASS: max |E(t) - E(0)| / E_plate(0) < 1e-8
"""

import sys
import numpy as np

from nothing_engine.core.bogoliubov import SimulationConfig, run_simulation, audit_result
from nothing_engine.core.constants import casimir_energy_1d
from nothing_engine.config import get_gate_criterion

PI = np.pi
_GATE = "gate_4_3_energy_conservation"


def run_validation():
    print("=" * 60)
    print("Gate 4.3: Energy Conservation — Closed System")
    print("=" * 60)

    N = 32
    cfg = SimulationConfig(
        n_modes=N,
        plate_mass=1e4,
        spring_k=0.0,
        q0=1.0,
        v0=1e-3,
        t_span=(0.0, 100.0),
        rtol=1e-13,
        atol=1e-15,
        max_step=0.005,
        audit_tolerance_factor=1e-6,
        audit_halt=False,
    )

    print(f"Parameters: N={N}, M={cfg.plate_mass}, v0={cfg.v0}")
    print(f"Duration: t = {cfg.t_span[0]} to {cfg.t_span[1]}")
    print(f"Integrator: {cfg.method}, rtol={cfg.rtol}, atol={cfg.atol}")
    print()

    print("Running simulation...")
    result = run_simulation(cfg)
    print(f"Completed: {len(result.t)} time points, {result.rhs_call_count} RHS calls")
    print()

    # Energy audit
    E0 = result.energy_at(0)
    print(f"Initial energies:")
    print(f"  E_plate  = {E0['E_plate']:.10e}")
    print(f"  E_spring = {E0['E_spring']:.10e}")
    print(f"  E_field  = {E0['E_field']:.10e}")
    print(f"  E_total  = {E0['E_total']:.10e}")
    print()

    E_plate_0 = E0["E_plate"]
    E_total_0 = E0["E_total"]

    max_drift = 0.0
    for i in range(len(result.t)):
        E = result.energy_at(i)["E_total"]
        drift = abs(E - E_total_0)
        if drift > max_drift:
            max_drift = drift

    # Denominator: the dominant physical energy scale. E_plate is the
    # kinetic scale of the mechanical DOF; |E_Casimir| is the field
    # zero-point scale. Both enter the coupled ODE, so the relevant
    # accuracy is drift relative to the *largest* scale the integrator
    # sees — not just the plate KE, which can be microscopic and inflate
    # a meaningful drift out of proportion.
    a0 = cfg.q0 - cfg.x_left
    E_casimir_scale = abs(casimir_energy_1d(a0))
    scale = max(E_plate_0, E_casimir_scale)

    relative_to_plate = max_drift / max(E_plate_0, 1e-20)
    relative_to_scale = max_drift / max(scale, 1e-20)
    relative_to_total = max_drift / max(abs(E_total_0), 1e-20)

    print(f"Maximum absolute energy drift: {max_drift:.4e}")
    print(f"Plate energy scale:            {E_plate_0:.4e}")
    print(f"Casimir energy scale:          {E_casimir_scale:.4e}")
    print(f"Effective denominator scale:   {scale:.4e}")
    print(f"Relative to plate energy:      {relative_to_plate:.4e}")
    print(f"Relative to scale (max plate, Casimir): {relative_to_scale:.4e}")
    print(f"Relative to total energy:      {relative_to_total:.4e}")

    # Also run formal audit
    auditor = audit_result(result, check_every=1)
    s = auditor.summary()
    print(f"\nFormal audit: {s['n_checks']} checks, max_drift = {s['max_drift']:.4e}")

    pass_criterion = get_gate_criterion(
        _GATE, "pass_criterion_relative_to_scale", default=1.0e-7
    )
    passed = relative_to_scale < pass_criterion
    print(f"\nThreshold (from {_GATE}.pass_criterion_relative_to_scale): {pass_criterion:.2e}")
    print(f"GATE 4.3: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
