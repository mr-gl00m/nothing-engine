"""
Validation Gate 4.7: Residual Motion Baseline

Verify that a plate held fixed in vacuum shows zero particle creation.
This establishes the zero-point fluctuation baseline.

Method:
    Prescribed static plate motion: q(t) = a0, v(t) = 0.
    Evolve for full experimental duration.
    Report max |beta_n|^2 across all modes and time.

PASS: max |beta_n|^2 < threshold (declared in validation_criteria.yaml).
"""

import sys
import numpy as np

from nothing_engine.core import mode_space
from nothing_engine.core.bogoliubov import SimulationConfig, run_simulation, PrecomputedArrays
from nothing_engine.config import get_gate_criterion

_GATE = "gate_4_7_residual_baseline"


def run_validation():
    print("=" * 60)
    print("Gate 4.7: Residual Motion Baseline — Static Plate")
    print("=" * 60)

    N = int(get_gate_criterion(_GATE, "N_modes", default=64))
    v0 = float(get_gate_criterion(_GATE, "initial_velocity", default=0.0))
    duration = float(get_gate_criterion(_GATE, "duration", default=100.0))
    beta_threshold = float(
        get_gate_criterion(_GATE, "pass_criterion_max_beta_sq", default=1.0e-12)
    )
    a0 = 1.0

    cfg = SimulationConfig(
        n_modes=N,
        plate_mass=1e4,
        q0=a0,
        v0=v0,
        t_span=(0.0, duration),
        rtol=1e-13,
        atol=1e-15,
        max_step=0.0,  # auto Nyquist-safe
        audit_halt=False,
    )
    pre = PrecomputedArrays.from_config(cfg)

    def q_const(t):
        return a0

    def v_zero(t):
        return 0.0

    print(f"Parameters: N={N}, a0={a0}")
    print(f"Duration: T = {cfg.t_span[1]}")
    print(f"Prescribed motion: q(t) = {a0}, v(t) = 0")
    print()

    print("Running simulation...")
    result = run_simulation(cfg, prescribed_motion=(q_const, v_zero), precomputed=pre)
    print(f"Completed: {len(result.t)} time points, {result.rhs_call_count} RHS calls")
    print()

    # Check particle number at multiple time points
    max_beta = 0.0
    worst_mode = -1
    worst_time = -1.0

    check_indices = np.linspace(0, len(result.t) - 1, min(50, len(result.t)), dtype=int)

    for i in check_indices:
        ms = result.mode_state_at(i)
        beta_sq = mode_space.particle_number(ms, N, a0, pre.g_n, pre.ns_pi)
        max_this = np.max(np.abs(beta_sq))
        if max_this > max_beta:
            max_beta = max_this
            worst_mode = int(np.argmax(np.abs(beta_sq))) + 1
            worst_time = result.t[i]

    print(f"Maximum |beta_n|^2 across all modes and times: {max_beta:.4e}")
    print(f"  Worst mode: n = {worst_mode}")
    print(f"  Worst time: t = {worst_time:.2f}")

    # Wronskian check
    ms_final = result.mode_state_at(-1)
    w = mode_space.wronskian(ms_final, N)
    w_drift = np.max(np.abs(w + 0.5))
    print(f"\nWronskian max drift from -0.5: {w_drift:.4e}")

    passed = max_beta < beta_threshold
    print(f"\nThreshold (from {_GATE}.pass_criterion_max_beta_sq): {beta_threshold:.2e}")
    print(f"GATE 4.7: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
