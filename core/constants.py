"""
Physical constants and unit conversion helpers.

Natural units: ℏ = c = 1.
Physical units restored for output and interpretation.
"""

import numpy as np

# Fundamental constants (SI)
HBAR_SI = 1.054571817e-34      # J·s
C_SI = 299792458.0             # m/s
KB_SI = 1.380649e-23           # J/K

# Natural units (used internally)
HBAR = 1.0
C = 1.0

# Derived
PI = np.pi


def si_to_natural(mass_kg: float, length_m: float, spring_k_si: float):
    """Convert SI parameters to natural units.

    In natural units with ℏ = c = 1:
    - [length] = 1/[energy]
    - [time] = 1/[energy]
    - [mass] = [energy]

    We choose the cavity width a₀ as the reference length scale.

    Parameters
    ----------
    mass_kg : float
        Plate mass in kg.
    length_m : float
        Reference length (cavity width a₀) in meters.
    spring_k_si : float
        Spring constant in N/m.

    Returns
    -------
    dict
        Natural-unit parameters with conversion factors.
    """
    # Energy scale: ℏc / a₀
    E_scale = HBAR_SI * C_SI / length_m
    # Time scale: a₀ / c
    T_scale = length_m / C_SI
    # Mass in natural units: M_si * c² / E_scale
    M_natural = mass_kg * C_SI**2 / E_scale

    return {
        "M": M_natural,
        "k": spring_k_si * length_m**2 / E_scale,
        "E_scale": E_scale,
        "T_scale": T_scale,
        "L_scale": length_m,
    }


def cavity_mode_frequency(n: int, cavity_width: float) -> float:
    """Instantaneous cavity mode frequency ωₙ = nπ/a.

    Parameters
    ----------
    n : int
        Mode number (≥ 1).
    cavity_width : float
        Current cavity width q - x_L.

    Returns
    -------
    float
        Mode frequency in natural units.
    """
    return n * PI / cavity_width


def casimir_energy_1d(cavity_width: float) -> float:
    """Analytical Casimir energy for 1+1D scalar Dirichlet-Dirichlet.

    E_Casimir = -π / (24a)

    Parameters
    ----------
    cavity_width : float
        Cavity width.

    Returns
    -------
    float
        Casimir energy (negative).
    """
    return -PI / (24.0 * cavity_width)


def casimir_force_1d(cavity_width: float) -> float:
    """Analytical Casimir force for 1+1D scalar Dirichlet-Dirichlet.

    F_Casimir = -dE/da = -π / (24a²)

    Parameters
    ----------
    cavity_width : float
        Cavity width.

    Returns
    -------
    float
        Casimir force (attractive, negative).
    """
    return -PI / (24.0 * cavity_width**2)
