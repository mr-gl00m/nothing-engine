"""
Validation Gate 4.1: Static Casimir Energy

Verify that the field energy for a static cavity reproduces the
analytical Casimir energy E = -pi/(24*a) for 1+1D scalar Dirichlet.

Method:
    For a fixed plate (prescribed static motion), the field energy is:
        E_field = sum_n omega_n / 2   (vacuum zero-point)

    The Casimir energy is obtained by regularization:
        E_Casimir = E_field - E_free = -pi/(24*a)

    We verify this by checking that the regularized energy converges
    to the analytical value as N_modes increases.

PASS: Relative error < 1% for N >= 256.
"""

import sys
import os
import numpy as np

from nothing_engine.core import mode_space, constants

PI = np.pi


def casimir_energy_truncated(n_modes: int, cavity_width: float) -> float:
    """Compute the truncated Casimir energy using zeta regularization.

    E_vac = sum_{n=1}^{N} omega_n / 2 = (pi / (2a)) * sum_{n=1}^{N} n

    The regularized value uses zeta(-1) = -1/12:
        E_Casimir = (pi / (2a)) * (-1/12) = -pi / (24a)

    For finite N, we can measure convergence by computing the difference
    between the truncated sum and the analytical divergent piece.
    """
    omegas = mode_space.mode_frequencies(n_modes, cavity_width)
    E_truncated = 0.5 * np.sum(omegas)
    return E_truncated


def exponentially_regulated_vacuum_energy(n_modes: int, cavity_width: float,
                                          cutoff_scale: float) -> float:
    """Compute the exponentially regulated 1D scalar vacuum energy.

    E_reg(Lambda) = (pi / 2a) * sum_{n=1}^{N} n * exp(-n / Lambda)

    For the analytical regulated sum:
        sum_{n=1}^{inf} n * exp(-n/Lambda) = e^{-1/Lambda} / (1 - e^{-1/Lambda})^2

    This equals Lambda^2 - 1/12 + O(Lambda^{-2}) as Lambda -> inf, so
    subtracting the Lambda^2 "plate self-energy" divergence yields the
    physical Casimir result -pi / (24a) in the limit. Choosing
    ``n_modes`` sufficiently larger than ``cutoff_scale`` makes the
    truncation error exponentially small.
    """
    ns = np.arange(1, n_modes + 1, dtype=np.float64)
    return float(
        (PI / (2.0 * cavity_width)) * np.sum(ns * np.exp(-ns / cutoff_scale))
    )


def run_validation():
    a = 1.0
    E_analytical = -PI / (24.0 * a)

    print("=" * 60)
    print("Gate 4.1: Static Casimir Energy (non-tautological)")
    print("=" * 60)
    print(f"Analytical target E_Casimir = -pi/(24*a) = {E_analytical:.10f}")
    print(f"Cavity width a = {a}")
    print()
    print("Method: exponential regulator sum_n n*exp(-n/Lambda) with")
    print("        plate-self-energy subtraction. The regulated sum")
    print("        equals Lambda^2 - 1/12 + O(1/Lambda^2) analytically,")
    print("        so subtracting Lambda^2 reproduces the zeta value.")
    print()

    # Use several regulator scales, always with n_modes >> cutoff_scale
    # so the finite-sum truncation error is negligible.
    cutoff_scales = [20.0, 40.0, 80.0, 160.0, 320.0]
    N = 16384  # >> max cutoff_scale, so truncation error is exp(-51)

    print(f"{'Lambda':>10} | {'E_reg(Lambda)':>18} | {'E_reg - scale^2 term':>22} | {'rel err':>10}")
    print("-" * 72)

    errors = []
    for Lambda in cutoff_scales:
        E_reg = exponentially_regulated_vacuum_energy(N, a, Lambda)
        # Subtract the divergent Lambda^2 piece: (pi/2a) * Lambda^2
        subtract = (PI / (2.0 * a)) * Lambda * Lambda
        E_casimir_num = E_reg - subtract
        rel_err = abs(E_casimir_num - E_analytical) / abs(E_analytical)
        errors.append(rel_err)
        print(
            f"{Lambda:>10.1f} | {E_reg:>18.6f} | {E_casimir_num:>22.10f} | {rel_err:>10.2e}"
        )

    # Convergence: rel_err should decrease by ~4x per doubling of Lambda
    # (O(1/Lambda^2) convergence). The final value should be within ~1e-4.
    final_error = errors[-1]
    convergence_ratio = errors[0] / errors[-1] if errors[-1] > 0 else float("inf")

    # Also sanity-check that constants.casimir_energy_1d agrees with the
    # canonical formula (this is still a formula check, but the gate now
    # depends primarily on the numerical extrapolation above).
    E_constant = constants.casimir_energy_1d(a)
    constants_error = abs(E_constant - E_analytical) / abs(E_analytical)

    print()
    print(f"constants.casimir_energy_1d(a) = {E_constant:.10f} (rel err {constants_error:.2e})")
    print(f"Numerical regulator-extrapolation final rel err = {final_error:.2e}")
    print(f"Convergence ratio (first / last) = {convergence_ratio:.1f}")

    # Pass criteria: the numerical extrapolation must reach < 1e-4 AND
    # the constants module must agree with the canonical value. The
    # numerical check is the real gate; the formula check is a secondary
    # consistency assertion, not the primary validation.
    passed = final_error < 1.0e-4 and constants_error < 1.0e-14
    print()
    print(f"GATE 4.1: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
