"""
Field -> plate force computation (Track B).

Energy-conserving force in the mode-function representation:

    F_field = sum_n (n^2 * pi^2 / a^3) * |f_n|^2

where a = q - x_L is the cavity width and |f_n|^2 = u_n^2 + v_n^2.

Energy conservation proof:
    E_field = sum_n 0.5 * (|f_dot_n|^2 + omega_n^2 * |f_n|^2)
    dE_field/dt = sum_n omega_n * omega_dot_n * |f_n|^2
                = -v * sum_n (n^2 * pi^2 / a^3) * |f_n|^2
                = -v * F_field

    Combined with dE_plate/dt = v * F_total, total energy is conserved.

At t=0 (vacuum): reproduces Casimir force -pi/(24a^2) via zeta
regularization of sum_n n.

NOTE: The architecture's formula F = sum (n*pi/a^2)(|beta_n|^2 + 1/2)
is correct in Bogoliubov canonical variables but not energy-conserving
in the mode-function (f, f_dot) representation. We use the mode-function
formula which is provably consistent.
"""

import numpy as np
from numpy.typing import NDArray

PI = np.pi


def field_force_track_b(mode_state: NDArray, n_modes: int,
                        cavity_width: float,
                        ns_pi_sq: NDArray) -> float:
    """Compute the field force on the plate from mode functions.

    F = sum_n (n^2 * pi^2 / a^3) * |f_n|^2

    Parameters
    ----------
    mode_state : NDArray, shape (4*N,)
        Mode state sub-vector.
    n_modes : int
        Number of modes.
    cavity_width : float
        Current cavity width a = q - x_L. Must be positive.
    ns_pi_sq : NDArray, shape (N,)
        Pre-computed n^2 * pi^2 array (passed from RHS closure).

    Returns
    -------
    float
        Force on the right plate (positive = rightward).
    """
    idx = 4 * n_modes
    u = mode_state[0:idx:4]
    v = mode_state[2:idx:4]
    f_sq = u * u + v * v
    a3 = cavity_width ** 3
    return float(np.dot(ns_pi_sq, f_sq) / a3)


def field_force_track_b_from_fsq(f_sq: NDArray, cavity_width: float,
                                  ns_pi_sq: NDArray) -> float:
    """Force from pre-computed |f_n|^2 array. For use inside RHS hot path."""
    return float(np.dot(ns_pi_sq, f_sq) / cavity_width**3)


def static_casimir_force_truncated(n_modes: int, cavity_width: float) -> float:
    """Truncated (unregularized) static Casimir force from N modes.

    At t=0: |f_n|^2 = 1/(2*omega_n) = a/(2*n*pi).
    F = sum_{n=1}^{N} (n^2*pi^2/a^3) * a/(2*n*pi) = sum n*pi/(2*a^2)
      = pi * N*(N+1) / (4*a^2)

    This diverges as N -> infinity. For finite N it is a large positive
    force (pushing the plate outward), which is the UV-divergent vacuum
    contribution. The physical (renormalized) force is attractive.

    Parameters
    ----------
    n_modes : int
    cavity_width : float

    Returns
    -------
    float
        Truncated vacuum force (positive, unphysical for large N).
    """
    return PI * n_modes * (n_modes + 1) / (4.0 * cavity_width**2)


def static_casimir_force_regularized(cavity_width: float) -> float:
    """Zeta-regularized Casimir force: -pi/(24*a^2).

    Uses zeta(-1) = -1/12: sum_{n=1}^{inf} n = -1/12 (Ramanujan/zeta).

    Returns
    -------
    float
        Physical Casimir force (negative = attractive).
    """
    return -PI / (24.0 * cavity_width**2)
