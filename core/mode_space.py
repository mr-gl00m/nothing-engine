"""
Cavity mode decomposition for Track B.

Provides instantaneous cavity mode frequencies, their derivatives,
vacuum initial conditions, and particle number extraction from
the complex mode function representation.

Mode frequencies for a 1+1D scalar field with Dirichlet BCs:
    omega_n(q) = n * pi / a,    a = q - x_L

Mode function approach: each mode n has a complex function f_n(t) = u_n + i*v_n
satisfying f_n'' + omega_n^2(q(t)) * f_n = 0.

Vacuum initial conditions (positive-frequency mode):
    u_n(0) = 1 / sqrt(2 * omega_n0)
    u_dot_n(0) = 0
    v_n(0) = 0
    v_dot_n(0) = -sqrt(omega_n0 / 2)

Particle number:
    |beta_n|^2 = 0.5 * (omega_n * |f_n|^2 + |f_dot_n|^2 / omega_n) - 0.5
"""

import numpy as np
from numpy.typing import NDArray

PI = np.pi


def mode_indices(n_modes: int) -> NDArray[np.float64]:
    """Return array [1, 2, ..., N] as float64."""
    return np.arange(1, n_modes + 1, dtype=np.float64)


def form_factor(n_modes: int, n_cutoff: float,
                shape: str = "sigmoid") -> NDArray[np.float64]:
    """UV form factor modeling finite plate thickness.

    Models the finite thickness of the plate: modes with n >> n_cutoff
    (wavelength << plate thickness) do not see the plate as a hard
    boundary and are suppressed.

    Parameters
    ----------
    n_modes : int
        Number of modes.
    n_cutoff : float
        Cutoff mode number. Set to n_cutoff = a0/delta where delta
        is the plate thickness. Use np.inf to disable (ideal Dirichlet).
    shape : str
        "gaussian" — exp(-(n/n_cutoff)^2). Gentle roll-off.
        "sigmoid"  — 1/(1+exp((n - n_cutoff)/Delta_n)) with Delta_n = n_cutoff/10.
                     Sharp cutoff: ~1 for n < 0.8*n_cutoff, ~0 for n > 1.2*n_cutoff.

    Returns
    -------
    NDArray, shape (N,)
        Form factor values in [0, 1].
    """
    if n_cutoff == np.inf:
        return np.ones(n_modes, dtype=np.float64)
    ns = mode_indices(n_modes)
    if shape == "gaussian":
        return np.exp(-(ns / n_cutoff) ** 2)
    else:  # sigmoid
        delta_n = max(n_cutoff / 10.0, 1.0)
        return 1.0 / (1.0 + np.exp((ns - n_cutoff) / delta_n))


def mode_frequencies(n_modes: int, cavity_width: float,
                     g_n: NDArray = None,
                     ns_pi: NDArray = None) -> NDArray[np.float64]:
    """Instantaneous cavity mode frequencies omega_n = ns_pi[n]/a * sqrt(g_n).

    The form factor enters as sqrt(g_n) on the frequency so that g_n
    multiplies omega_n^2 in the mode ODE (f_n'' + g_n * omega_ideal^2 * f_n = 0).

    Parameters
    ----------
    n_modes : int
        Number of modes (1 to N).
    cavity_width : float
        Current cavity width a = q - x_L. Must be positive.
    g_n : NDArray, optional
        Form factor array. If None, ideal Dirichlet (g_n=1).
    ns_pi : NDArray, optional
        Boundary-aware frequency coefficients: n*pi (closed) or 2*n*pi (periodic).
        If None, defaults to n*pi (Dirichlet).

    Returns
    -------
    NDArray, shape (N,)
        Mode frequencies in natural units.
    """
    if ns_pi is None:
        ns_pi = mode_indices(n_modes) * PI
    omega_ideal = ns_pi / cavity_width
    if g_n is not None:
        return omega_ideal * np.sqrt(g_n)
    return omega_ideal


def mode_frequencies_squared(n_modes: int, cavity_width: float,
                             g_n: NDArray = None,
                             ns_pi: NDArray = None) -> NDArray[np.float64]:
    """omega_n^2 = g_n * (ns_pi[n]/a)^2. Pre-computed for the ODE RHS hot path."""
    if ns_pi is None:
        ns_pi = mode_indices(n_modes) * PI
    omega_sq_ideal = (ns_pi / cavity_width) ** 2
    if g_n is not None:
        return g_n * omega_sq_ideal
    return omega_sq_ideal


def mode_frequency_derivatives(n_modes: int, cavity_width: float,
                               g_n: NDArray = None) -> NDArray[np.float64]:
    """d(omega_n)/dq = -n*pi/a^2 * sqrt(g_n).

    Returns
    -------
    NDArray, shape (N,)
        Frequency derivatives (negative: frequency decreases as cavity widens).
    """
    ns = mode_indices(n_modes)
    deriv_ideal = -ns * PI / cavity_width**2
    if g_n is not None:
        return deriv_ideal * np.sqrt(g_n)
    return deriv_ideal


def vacuum_initial_conditions(n_modes: int, cavity_width_0: float,
                              g_n: NDArray = None,
                              ns_pi: NDArray = None) -> NDArray[np.float64]:
    """Build the 4N-element mode initial condition vector for exact vacuum.

    The state is packed as [u_1, u_dot_1, v_1, v_dot_1, u_2, ...].

    For each mode n:
        u_n(0) = 1 / sqrt(2 * omega_n0)     [real part of f_n]
        u_dot_n(0) = 0
        v_n(0) = 0                           [imaginary part of f_n]
        v_dot_n(0) = -sqrt(omega_n0 / 2)

    These satisfy the Wronskian: u_n * v_dot_n - v_n * u_dot_n = -1/2
    and give |beta_n|^2 = 0 (exact vacuum, zero particles).

    Parameters
    ----------
    n_modes : int
        Number of modes.
    cavity_width_0 : float
        Initial cavity width a0.
    g_n : NDArray, optional
        Form factor array. If None, ideal Dirichlet.
    ns_pi : NDArray, optional
        Boundary-aware frequency coefficients. If None, defaults to n*pi.

    Returns
    -------
    NDArray, shape (4*N,)
        Mode state vector.
    """
    omegas = mode_frequencies(n_modes, cavity_width_0, g_n, ns_pi)
    state = np.zeros(4 * n_modes)
    idx = 4 * n_modes  # explicit stop for stride slicing

    # u_n(0) = 1 / sqrt(2 * omega_n)
    state[0:idx:4] = 1.0 / np.sqrt(2.0 * omegas)

    # u_dot_n(0) = 0  (already zero)

    # v_n(0) = 0  (already zero)

    # v_dot_n(0) = -sqrt(omega_n / 2)
    state[3:idx:4] = -np.sqrt(omegas / 2.0)

    return state


def extract_mode_amplitudes_squared(mode_state: NDArray, n_modes: int) -> NDArray[np.float64]:
    """|f_n|^2 = u_n^2 + v_n^2.

    Parameters
    ----------
    mode_state : NDArray, shape (4*N,)
        Mode state sub-vector (excluding plate DOFs).
    n_modes : int
        Number of modes.

    Returns
    -------
    NDArray, shape (N,)
    """
    idx = 4 * n_modes
    u = mode_state[0:idx:4]
    v = mode_state[2:idx:4]
    return u**2 + v**2


def extract_mode_velocities_squared(mode_state: NDArray, n_modes: int) -> NDArray[np.float64]:
    """|f_dot_n|^2 = u_dot_n^2 + v_dot_n^2.

    Parameters
    ----------
    mode_state : NDArray, shape (4*N,)
    n_modes : int

    Returns
    -------
    NDArray, shape (N,)
    """
    idx = 4 * n_modes
    u_dot = mode_state[1:idx:4]
    v_dot = mode_state[3:idx:4]
    return u_dot**2 + v_dot**2


def particle_number(mode_state: NDArray, n_modes: int,
                    cavity_width: float,
                    g_n: NDArray = None,
                    ns_pi: NDArray = None) -> NDArray[np.float64]:
    """|beta_n|^2 = 0.5 * (omega_n * |f_n|^2 + |f_dot_n|^2 / omega_n) - 0.5

    This is the number of real particles created from vacuum in each mode.

    Parameters
    ----------
    mode_state : NDArray, shape (4*N,)
    n_modes : int
    cavity_width : float
        Current cavity width (for instantaneous omega_n).
    g_n : NDArray, optional
        Form factor array. If None, ideal Dirichlet.
    ns_pi : NDArray, optional
        Boundary-aware frequency coefficients. If None, defaults to n*pi.

    Returns
    -------
    NDArray, shape (N,)
        Particle number per mode. Zero for vacuum state.
    """
    omegas = mode_frequencies(n_modes, cavity_width, g_n, ns_pi)
    f_sq = extract_mode_amplitudes_squared(mode_state, n_modes)
    fdot_sq = extract_mode_velocities_squared(mode_state, n_modes)
    return 0.5 * (omegas * f_sq + fdot_sq / omegas) - 0.5


def total_particle_number(mode_state: NDArray, n_modes: int,
                          cavity_width: float,
                          g_n: NDArray = None,
                          ns_pi: NDArray = None) -> float:
    """N(t) = sum_n |beta_n|^2. Total particles created from vacuum."""
    return float(np.sum(particle_number(mode_state, n_modes, cavity_width, g_n, ns_pi)))


def wronskian(mode_state: NDArray, n_modes: int) -> NDArray[np.float64]:
    """Compute per-mode Wronskian: w_n = u_n * v_dot_n - v_n * u_dot_n.

    For exact vacuum IC, w_n = -1/2 for all n. This is conserved by the
    linear ODE and serves as an independent integrator accuracy check.

    Returns
    -------
    NDArray, shape (N,)
    """
    idx = 4 * n_modes
    u = mode_state[0:idx:4]
    u_dot = mode_state[1:idx:4]
    v = mode_state[2:idx:4]
    v_dot = mode_state[3:idx:4]
    return u * v_dot - v * u_dot
