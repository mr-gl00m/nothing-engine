"""
Energy component computation for Track B.

All energy quantities in natural units (hbar = c = 1).

Components:
    E_plate   = 0.5 * M * v^2
    E_spring  = 0.5 * k * (q - q_eq)^2
    E_field   = sum_n 0.5 * (|f_dot_n|^2 + g_n * omega_ideal_n^2 * |f_n|^2)

With form factor g_n, the effective frequency is omega_n = sqrt(g_n) * n*pi/a.
The vacuum energy (subtracted for renormalization) is sum_n omega_n / 2.

Total energy E_plate + E_spring + E_field_ren is exactly conserved.
"""

import numpy as np
from numpy.typing import NDArray

from . import mode_space


def plate_kinetic_energy(mass: float, velocity: float) -> float:
    """E_plate = 0.5 * M * v^2."""
    return 0.5 * mass * velocity**2


def spring_potential_energy(spring_k: float, q: float, q_eq: float) -> float:
    """E_spring = 0.5 * k * (q - q_eq)^2."""
    return 0.5 * spring_k * (q - q_eq)**2


def vacuum_energy(n_modes: int, cavity_width: float,
                  g_n: NDArray = None,
                  ns_pi: NDArray = None) -> float:
    """Static vacuum zero-point energy: E_vac = sum_n omega_n / 2.

    General: E_vac = (1/(2*a)) * sum_n ns_pi[n] * sqrt(g_n).
    Closed (ns_pi = n*pi): E_vac = (pi/(2*a)) * sum_n n * sqrt(g_n).
    Periodic (ns_pi = 2*n*pi): E_vac = (pi/a) * sum_n n * sqrt(g_n).
    """
    if ns_pi is None:
        ns_pi = mode_space.mode_indices(n_modes) * np.pi
    if g_n is not None:
        return float(1.0 / (2.0 * cavity_width) * np.dot(ns_pi, np.sqrt(g_n)))
    return float(np.sum(ns_pi) / (2.0 * cavity_width))


def field_energy(mode_state: NDArray, n_modes: int,
                 cavity_width: float, g_n: NDArray = None,
                 ns_pi: NDArray = None) -> float:
    """Renormalized field energy (vacuum zero-point subtracted).

    E_field_ren = sum_n 0.5 * (|f_dot_n|^2 + g_n * (ns_pi[n]/a)^2 * |f_n|^2)
                  - E_vac(a, g_n, ns_pi)

    Parameters
    ----------
    mode_state : NDArray, shape (4*N,)
    n_modes : int
    cavity_width : float
    g_n : NDArray, optional
        Form factor. If None, ideal Dirichlet.
    ns_pi : NDArray, optional
        Boundary-aware frequency coefficients. If None, defaults to n*pi.

    Returns
    -------
    float
    """
    omegas_sq = mode_space.mode_frequencies_squared(n_modes, cavity_width, g_n, ns_pi)
    f_sq = mode_space.extract_mode_amplitudes_squared(mode_state, n_modes)
    fdot_sq = mode_space.extract_mode_velocities_squared(mode_state, n_modes)
    E_raw = float(0.5 * np.sum(fdot_sq + omegas_sq * f_sq))
    return E_raw - vacuum_energy(n_modes, cavity_width, g_n, ns_pi)


def field_energy_per_mode(mode_state: NDArray, n_modes: int,
                          cavity_width: float,
                          g_n: NDArray = None,
                          ns_pi: NDArray = None) -> NDArray[np.float64]:
    """Renormalized per-mode energy: E_n_raw - omega_n/2.

    Returns
    -------
    NDArray, shape (N,)
    """
    omegas = mode_space.mode_frequencies(n_modes, cavity_width, g_n, ns_pi)
    omegas_sq = omegas ** 2
    f_sq = mode_space.extract_mode_amplitudes_squared(mode_state, n_modes)
    fdot_sq = mode_space.extract_mode_velocities_squared(mode_state, n_modes)
    return 0.5 * (fdot_sq + omegas_sq * f_sq) - 0.5 * omegas


def particle_energy(mode_state: NDArray, n_modes: int,
                    cavity_width: float, g_n: NDArray = None,
                    ns_pi: NDArray = None) -> float:
    """Energy stored in created particles (above vacuum).

    E_particles = sum_n omega_n * |beta_n|^2
    """
    omegas = mode_space.mode_frequencies(n_modes, cavity_width, g_n, ns_pi)
    betas_sq = mode_space.particle_number(mode_state, n_modes, cavity_width, g_n, ns_pi)
    return float(np.dot(omegas, betas_sq))


def total_energy(mode_state: NDArray, n_modes: int, cavity_width: float,
                 mass: float, velocity: float,
                 spring_k: float, q: float, q_eq: float,
                 g_n: NDArray = None,
                 ns_pi: NDArray = None) -> float:
    """Total system energy: E_plate + E_spring + E_field_ren.

    Exactly conserved by the coupled ODE.
    """
    return (plate_kinetic_energy(mass, velocity)
            + spring_potential_energy(spring_k, q, q_eq)
            + field_energy(mode_state, n_modes, cavity_width, g_n, ns_pi))


def energy_components(mode_state: NDArray, n_modes: int, cavity_width: float,
                      mass: float, velocity: float,
                      spring_k: float, q: float, q_eq: float,
                      g_n: NDArray = None,
                      ns_pi: NDArray = None) -> dict:
    """Return all energy components as a dictionary."""
    e_plate = plate_kinetic_energy(mass, velocity)
    e_spring = spring_potential_energy(spring_k, q, q_eq)
    e_field = field_energy(mode_state, n_modes, cavity_width, g_n, ns_pi)
    return {
        "E_plate": e_plate,
        "E_spring": e_spring,
        "E_field": e_field,
        "E_total": e_plate + e_spring + e_field,
    }
