"""Field force calculations for the diagonal mode approximation.

The raw finite oscillator force is the negative derivative of the raw mode
Hamiltonian. The reported physical force subtracts the matching finite vacuum
term and restores the analytic static Casimir force. The same decomposition is
used in core.energy, so the continuous reduced equations conserve total energy.
"""

import numpy as np
from numpy.typing import NDArray

from . import constants

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


def renormalized_field_force(mode_state: NDArray, n_modes: int,
                             cavity_width: float,
                             ns_pi_sq_g: NDArray,
                             ns_pi: NDArray,
                             g_n: NDArray,
                             boundary: str = "closed",
                             degeneracy: int | None = None) -> float:
    """Return excitation back reaction plus the static Casimir force."""
    if cavity_width <= 0.0:
        raise ValueError("cavity_width must be positive")
    if degeneracy is None:
        degeneracy = constants.mode_degeneracy_1d(boundary)
    idx = 4 * n_modes
    f_sq = mode_state[0:idx:4] ** 2 + mode_state[2:idx:4] ** 2
    excitation = float(np.sum(
        ns_pi_sq_g * f_sq / cavity_width**3
        - 0.5 * ns_pi * np.sqrt(g_n) / cavity_width**2
    ))
    return (degeneracy * excitation
            + constants.casimir_force_1d(cavity_width, boundary))


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


def static_casimir_force_regularized(cavity_width: float,
                                     boundary: str = "closed") -> float:
    """Regularized static force for a supported ideal scalar field.

    For the closed interval this is -pi/(24*a^2). For the periodic circle it
    is -pi/(6*L^2).

    Returns
    -------
    float
        Physical Casimir force (negative = attractive).
    """
    return constants.casimir_force_1d(cavity_width, boundary)
