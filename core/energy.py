"""Energy bookkeeping for the diagonal mode approximation.

All quantities use natural units with hbar = c = 1. The finite oscillator
zero point sum is subtracted mode by mode. The analytic, regularized static
Casimir energy is then restored as a separate term. This separates dynamical
excitations from the static interaction without leaving a cutoff dependent
vacuum baseline in the reported field energy.
"""

import numpy as np
from numpy.typing import NDArray

from . import constants
from . import mode_space


def plate_kinetic_energy(mass: float, velocity: float) -> float:
    """E_plate = 0.5 * M * v^2."""
    return 0.5 * mass * velocity**2


def spring_potential_energy(spring_k: float, q: float, q_eq: float) -> float:
    """E_spring = 0.5 * k * (q - q_eq)^2."""
    return 0.5 * spring_k * (q - q_eq)**2


def vacuum_energy(n_modes: int, cavity_width: float,
                  g_n: NDArray | None = None,
                  ns_pi: NDArray | None = None,
                  degeneracy: int = 1) -> float:
    """Finite zero point sum used for mode by mode subtraction.

    General: E_vac,N = d/(2a) * sum_n ns_pi[n] * sqrt(g_n).
    Closed (ns_pi = n*pi): E_vac = (pi/(2*a)) * sum_n n * sqrt(g_n).
    Periodic positive modes use d = 2.

    This truncated positive quantity is a counterterm. It is distinct from the
    finite Casimir interaction energy returned by constants.casimir_energy_1d.
    """
    if cavity_width <= 0.0:
        raise ValueError("cavity_width must be positive")
    if degeneracy < 1:
        raise ValueError("degeneracy must be at least one")
    if ns_pi is None:
        ns_pi = mode_space.mode_indices(n_modes) * np.pi
    if g_n is not None:
        return float(degeneracy / (2.0 * cavity_width)
                     * np.dot(ns_pi, np.sqrt(g_n)))
    return float(degeneracy * np.sum(ns_pi) / (2.0 * cavity_width))


def field_excitation_energy(mode_state: NDArray, n_modes: int,
                            cavity_width: float,
                            g_n: NDArray | None = None,
                            ns_pi: NDArray | None = None,
                            degeneracy: int = 1) -> float:
    """Field energy above the instantaneous oscillator vacuum."""
    omegas_sq = mode_space.mode_frequencies_squared(
        n_modes, cavity_width, g_n, ns_pi,
    )
    omegas = np.sqrt(omegas_sq)
    f_sq = mode_space.extract_mode_amplitudes_squared(mode_state, n_modes)
    fdot_sq = mode_space.extract_mode_velocities_squared(mode_state, n_modes)
    return float(0.5 * degeneracy * np.sum(
        fdot_sq + omegas_sq * f_sq - omegas
    ))


def field_energy(mode_state: NDArray, n_modes: int,
                 cavity_width: float, g_n: NDArray | None = None,
                 ns_pi: NDArray | None = None,
                 boundary: str = "closed",
                 degeneracy: int | None = None) -> float:
    """Renormalized field energy including the static Casimir term.

    E_field = E_excitation + E_Casimir.

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
    if degeneracy is None:
        degeneracy = constants.mode_degeneracy_1d(boundary)
    return (field_excitation_energy(
        mode_state, n_modes, cavity_width, g_n, ns_pi, degeneracy,
    ) + constants.casimir_energy_1d(cavity_width, boundary))


def field_energy_per_mode(mode_state: NDArray, n_modes: int,
                          cavity_width: float,
                          g_n: NDArray | None = None,
                          ns_pi: NDArray | None = None,
                          degeneracy: int = 1) -> NDArray[np.float64]:
    """Excitation energy per stored positive mode index.

    The static Casimir term is global and is therefore absent from this array.

    Returns
    -------
    NDArray, shape (N,)
    """
    omegas = mode_space.mode_frequencies(n_modes, cavity_width, g_n, ns_pi)
    omegas_sq = omegas ** 2
    f_sq = mode_space.extract_mode_amplitudes_squared(mode_state, n_modes)
    fdot_sq = mode_space.extract_mode_velocities_squared(mode_state, n_modes)
    return degeneracy * (
        0.5 * (fdot_sq + omegas_sq * f_sq) - 0.5 * omegas
    )


def particle_energy(mode_state: NDArray, n_modes: int,
                    cavity_width: float, g_n: NDArray | None = None,
                    ns_pi: NDArray | None = None,
                    degeneracy: int = 1) -> float:
    """Energy stored in created particles (above vacuum).

    E_particles = sum_n omega_n * |beta_n|^2
    """
    omegas = mode_space.mode_frequencies(n_modes, cavity_width, g_n, ns_pi)
    betas_sq = mode_space.particle_number(mode_state, n_modes, cavity_width, g_n, ns_pi)
    return float(degeneracy * np.dot(omegas, betas_sq))


def total_energy(mode_state: NDArray, n_modes: int, cavity_width: float,
                 mass: float, velocity: float,
                 spring_k: float, q: float, q_eq: float,
                 g_n: NDArray | None = None,
                 ns_pi: NDArray | None = None,
                 boundary: str = "closed",
                 degeneracy: int | None = None) -> float:
    """Total system energy: E_plate + E_spring + E_field_ren.

    The continuous reduced equations conserve this Hamiltonian.
    """
    return (plate_kinetic_energy(mass, velocity)
            + spring_potential_energy(spring_k, q, q_eq)
            + field_energy(
                mode_state, n_modes, cavity_width, g_n, ns_pi,
                boundary, degeneracy,
            ))


def energy_components(mode_state: NDArray, n_modes: int, cavity_width: float,
                      mass: float, velocity: float,
                      spring_k: float, q: float, q_eq: float,
                      g_n: NDArray | None = None,
                      ns_pi: NDArray | None = None,
                      boundary: str = "closed",
                      degeneracy: int | None = None) -> dict[str, float]:
    """Return all energy components as a dictionary."""
    if degeneracy is None:
        degeneracy = constants.mode_degeneracy_1d(boundary)
    e_plate = plate_kinetic_energy(mass, velocity)
    e_spring = spring_potential_energy(spring_k, q, q_eq)
    e_excitation = field_excitation_energy(
        mode_state, n_modes, cavity_width, g_n, ns_pi, degeneracy,
    )
    e_casimir = constants.casimir_energy_1d(cavity_width, boundary)
    e_field = e_excitation + e_casimir
    return {
        "E_plate": e_plate,
        "E_spring": e_spring,
        "E_excitation": e_excitation,
        "E_casimir": e_casimir,
        "E_field": e_field,
        "E_total": e_plate + e_spring + e_field,
    }
