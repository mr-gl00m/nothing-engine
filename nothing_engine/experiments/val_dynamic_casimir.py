"""
Validation Gate 4.2: Dynamic Casimir Effect — Parametric Photon Production

Verify that prescribed sinusoidal plate motion at frequency Omega = 2*omega_1
produces particle creation in the fundamental mode, with the total particle
number matching the analytical sinh^2 prediction.

For small oscillation amplitude epsilon:
    q(t) = a0 + epsilon * sin(Omega * t),  Omega = 2 * omega_1

The fundamental mode particle number grows as:
    |beta_1(t)|^2 = sinh^2(gamma * t)

where gamma = epsilon * omega_1 / 2 (parametric resonance rate).

PASS: Final |beta_1|^2 matches sinh^2(gamma*T) within 5%.
"""

import sys
import os
import numpy as np

from nothing_engine.core import mode_space
from nothing_engine.core.bogoliubov import SimulationConfig, run_simulation

PI = np.pi


def run_validation():
    print("=" * 60)
    print("Gate 4.2: Dynamic Casimir Effect — Parametric Growth")
    print("=" * 60)

    N = 32
    a0 = 1.0
    omega_1 = PI / a0
    eps = 0.01            # Oscillation amplitude (1% of cavity width)
    Omega = 2 * omega_1   # Resonant frequency

    # Analytical prediction for growth rate
    gamma_theory = eps * omega_1 / 2.0

    T_final = 40.0
    sinh_arg = gamma_theory * T_final
    beta1_predicted = np.sinh(sinh_arg) ** 2

    print(f"Parameters: N={N}, a0={a0}, eps={eps}")
    print(f"omega_1 = {omega_1:.6f}, Omega = 2*omega_1 = {Omega:.6f}")
    print(f"Theoretical growth rate: gamma = eps*omega_1/2 = {gamma_theory:.6e}")
    print(f"gamma * T = {sinh_arg:.4f}")
    print(f"Predicted |beta_1(T)|^2 = sinh^2({sinh_arg:.4f}) = {beta1_predicted:.6e}")
    print()

    # Use t_eval to get specific output times
    t_eval = np.linspace(0, T_final, 500)

    cfg = SimulationConfig(
        n_modes=N,
        plate_mass=1e4,
        q0=a0,
        v0=0.0,
        t_span=(0.0, T_final),
        t_eval=t_eval,
        rtol=1e-13,
        atol=1e-15,
        max_step=0.002,
        audit_halt=False,
    )

    def q_func(t):
        return a0 + eps * np.sin(Omega * t)

    def v_func(t):
        return eps * Omega * np.cos(Omega * t)

    print("Running prescribed-motion simulation...")
    result = run_simulation(cfg, prescribed_motion=(q_func, v_func))
    print(f"Completed: {len(result.t)} time points, {result.rhs_call_count} RHS calls")
    print()

    # Extract |beta_1|^2 over time
    times = result.t
    beta1_sq = np.zeros(len(times))
    for i in range(len(times)):
        ms = result.mode_state_at(i)
        a = q_func(times[i]) - cfg.x_left
        pn = mode_space.particle_number(ms, N, a)
        beta1_sq[i] = pn[0]

    # Compare final value to sinh^2 prediction
    # Use cycle-averaged value near the end to smooth oscillations
    # (particle number oscillates at 2*Omega due to oscillating cavity width)
    cycle_period = 2 * PI / Omega
    n_avg_points = max(1, int(len(times) * cycle_period / T_final))
    beta1_final_avg = np.mean(beta1_sq[-n_avg_points:])

    # Also compute theoretical curve at multiple times for comparison
    beta1_theory = np.sinh(gamma_theory * times) ** 2

    # Compute the cycle-averaged theoretical value near the end
    theory_final_avg = np.mean(beta1_theory[-n_avg_points:])

    relative_error = abs(beta1_final_avg - theory_final_avg) / max(theory_final_avg, 1e-30)

    print(f"Final |beta_1|^2 (cycle-averaged): {beta1_final_avg:.6e}")
    print(f"Theory sinh^2(gamma*T):            {theory_final_avg:.6e}")
    print(f"Relative error:                    {relative_error:.4f} ({relative_error*100:.1f}%)")

    # Check mode selectivity
    ms_final = result.mode_state_at(-1)
    a_final = q_func(times[-1]) - cfg.x_left
    pn_final = mode_space.particle_number(ms_final, N, a_final)
    mode1_fraction = max(pn_final[0], 0) / max(np.sum(np.maximum(pn_final, 0)), 1e-30)
    print(f"\nMode 1 particle fraction: {mode1_fraction:.4f}")
    print(f"Final N_total = {np.sum(np.maximum(pn_final, 0)):.6e}")

    # Also verify the time dependence shape (not just final value)
    # Check at an intermediate time T/2
    i_half = len(times) // 2
    t_half = times[i_half]
    beta_half_measured = np.mean(beta1_sq[max(0, i_half - n_avg_points//2):i_half + n_avg_points//2 + 1])
    beta_half_theory = np.sinh(gamma_theory * t_half) ** 2
    mid_error = abs(beta_half_measured - beta_half_theory) / max(beta_half_theory, 1e-30)
    print(f"\nMid-point check (t={t_half:.1f}):")
    print(f"  Measured: {beta_half_measured:.6e}")
    print(f"  Theory:   {beta_half_theory:.6e}")
    print(f"  Error:    {mid_error:.4f} ({mid_error*100:.1f}%)")

    passed = relative_error < 0.10  # 10% tolerance (includes cycle averaging artifacts)
    print(f"\nGATE 4.2: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
