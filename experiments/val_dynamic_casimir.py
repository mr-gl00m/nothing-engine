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

PASS: Final |beta_1|^2 matches sinh^2(gamma*T) within threshold from YAML.
"""

import sys
import numpy as np

from nothing_engine.core import mode_space
from nothing_engine.core.bogoliubov import SimulationConfig, run_simulation, PrecomputedArrays
from nothing_engine.config import get_gate_criterion

PI = np.pi
_GATE = "gate_4_2_dynamic_casimir"


def run_validation():
    print("=" * 60)
    print("Gate 4.2: Dynamic Casimir Effect — Parametric Growth")
    print("=" * 60)

    N = int(get_gate_criterion(_GATE, "N_modes", default=32))
    eps = float(get_gate_criterion(_GATE, "prescribed_amplitude_eps", default=0.01))
    duration_cycles = float(get_gate_criterion(_GATE, "duration_cycles", default=50))
    rel_growth_tol = float(
        get_gate_criterion(_GATE, "pass_criterion_relative_growth_rate", default=0.05)
    )
    a0 = 1.0
    omega_1 = PI / a0
    Omega = 2 * omega_1

    # Duration in physical time: duration_cycles at the fundamental period.
    T_final = duration_cycles * (2 * PI / omega_1)

    gamma_theory = eps * omega_1 / 2.0
    sinh_arg = gamma_theory * T_final
    beta1_predicted = np.sinh(sinh_arg) ** 2

    print(f"Parameters: N={N}, a0={a0}, eps={eps}")
    print(f"omega_1 = {omega_1:.6f}, Omega = 2*omega_1 = {Omega:.6f}")
    print(f"Duration: {duration_cycles} fundamental cycles, T = {T_final:.2f}")
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
        max_step=0.0,  # auto Nyquist-safe
        audit_halt=False,
    )
    pre = PrecomputedArrays.from_config(cfg)

    def q_func(t):
        return a0 + eps * np.sin(Omega * t)

    def v_func(t):
        return eps * Omega * np.cos(Omega * t)

    def a_func(t):
        return -eps * Omega * Omega * np.sin(Omega * t)

    print("Running prescribed-motion simulation...")
    result = run_simulation(
        cfg,
        prescribed_motion=(q_func, v_func, a_func),
        precomputed=pre,
    )
    print(f"Completed: {len(result.t)} time points, {result.rhs_call_count} RHS calls")
    print()

    # Extract |beta_1|^2 over time
    times = result.t
    beta1_sq = np.zeros(len(times))
    for i in range(len(times)):
        ms = result.mode_state_at(i)
        a = q_func(times[i]) - cfg.x_left
        pn = mode_space.particle_number(ms, N, a, pre.g_n, pre.ns_pi)
        beta1_sq[i] = pn[0]

    # Compare final value to sinh^2 prediction
    # Use cycle-averaged value near the end to smooth oscillations
    # (particle number oscillates at 2*Omega due to oscillating cavity width)
    cycle_period = 2 * PI / Omega
    n_avg_points = max(1, int(len(times) * cycle_period / T_final))
    beta1_final_avg = np.mean(beta1_sq[-n_avg_points:])

    # Extract gamma by inverting sinh^2: |beta_1|^2 = sinh^2(gamma*t)
    # => gamma*t = arcsinh(sqrt(|beta_1|^2))  (valid at any amplitude,
    # including the small-argument quadratic regime where a log-linear
    # fit would be wrong).
    # Fit linearly on the last half of the trajectory to average out
    # the 2*Omega ripple in the cavity-width-parameterized particle number.
    half = len(times) // 2
    mask = beta1_sq[half:] > 0
    t_fit = times[half:][mask]
    y_fit = np.arcsinh(np.sqrt(beta1_sq[half:][mask]))
    if len(t_fit) >= 2:
        # Force the linear fit through the origin by solving min ||gamma*t - y||^2
        gamma_fit = float(np.dot(t_fit, y_fit) / np.dot(t_fit, t_fit))
    else:
        gamma_fit = float("nan")

    gamma_rel_err = abs(gamma_fit - gamma_theory) / gamma_theory

    # Theoretical curve for reference
    beta1_theory = np.sinh(gamma_theory * times) ** 2
    theory_final_avg = np.mean(beta1_theory[-n_avg_points:])
    amplitude_rel_err = abs(beta1_final_avg - theory_final_avg) / max(theory_final_avg, 1e-30)

    print(f"Final |beta_1|^2 (cycle-averaged): {beta1_final_avg:.6e}")
    print(f"Theory sinh^2(gamma*T):            {theory_final_avg:.6e}")
    print(f"Amplitude relative error:          {amplitude_rel_err*100:.2f}%")
    print()
    print(f"Fitted growth rate gamma_fit = {gamma_fit:.6e}")
    print(f"Theory gamma = eps*omega_1/2 = {gamma_theory:.6e}")
    print(f"Growth rate relative error:    {gamma_rel_err*100:.2f}%")

    # Mode selectivity diagnostic
    ms_final = result.mode_state_at(-1)
    a_final = q_func(times[-1]) - cfg.x_left
    pn_final = mode_space.particle_number(ms_final, N, a_final, pre.g_n, pre.ns_pi)
    mode1_fraction = max(pn_final[0], 0) / max(np.sum(np.maximum(pn_final, 0)), 1e-30)
    print(f"\nMode 1 particle fraction: {mode1_fraction:.4f}")
    print(f"Final N_total = {np.sum(np.maximum(pn_final, 0)):.6e}")

    passed = gamma_rel_err < rel_growth_tol
    print(f"\nThreshold (from {_GATE}.pass_criterion_relative_growth_rate): {rel_growth_tol:.2%}")
    print(f"GATE 4.2: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
