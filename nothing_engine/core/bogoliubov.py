"""
Time-dependent Bogoliubov ODE evolution (Track B).

Evolves complex mode functions f_n(t) = u_n(t) + i*v_n(t) coupled
to the dynamical plate motion. Each mode satisfies:

    f_n'' + omega_n^2(q(t)) * f_n = 0

where omega_n(q) = n*pi / (q - x_L) is the instantaneous cavity mode
frequency. The plate obeys:

    M * q'' = -k*(q - q_eq) + F_field

with F_field = sum_n (n^2 * pi^2 / a^3) * |f_n|^2 (energy-conserving).

State vector layout (4*N + 2 real variables):
    y[0:4N:4]   = u_n       (mode real parts)
    y[1:4N:4]   = u_dot_n   (mode real part velocities)
    y[2:4N:4]   = v_n       (mode imaginary parts)
    y[3:4N:4]   = v_dot_n   (mode imaginary part velocities)
    y[4N]       = q          (plate position)
    y[4N+1]     = v_plate    (plate velocity)

Integrated via scipy.integrate.solve_ivp (adaptive RK45/DOP853).
"""

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp, OdeSolution
from dataclasses import dataclass, field as dataclass_field
from typing import Optional, Callable

from . import mode_space
from . import energy as energy_mod
from . import energy_audit as audit_mod
from . import radiation_pressure as rp

PI = np.pi


@dataclass
class SimulationConfig:
    """Configuration for a Track B simulation run.

    Boundary types:
        "closed"   — Dirichlet walls, standing waves: omega_n = n*pi/a
        "periodic" — Ring topology, traveling waves: omega_n = 2*n*pi/L
                     The plate is inside a ring of circumference L = q0.
                     Photons emitted from one side circulate and return
                     from the other side, creating collective memory.
    """

    # Physics
    n_modes: int = 256
    plate_mass: float = 1.0e4
    spring_k: float = 0.0
    q0: float = 1.0
    v0: float = 1.0e-3
    x_left: float = 0.0
    boundary: str = "closed"  # "closed" or "periodic"

    # UV regularization: plate thickness form factor
    # n_cutoff = a0 / plate_thickness. Use np.inf to disable.
    plate_thickness: float = 0.0  # 0 = auto (a0/100)
    cutoff_shape: str = "sigmoid"  # "sigmoid" or "gaussian"

    # Integrator
    method: str = "RK45"
    rtol: float = 1.0e-10
    atol: float = 1.0e-12
    max_step: float = 0.01
    t_span: tuple = (0.0, 1000.0)
    t_eval: Optional[NDArray] = None
    dense_output: bool = True

    # Energy audit
    audit_tolerance_factor: float = 1.0e-6
    audit_halt: bool = True

    def __post_init__(self):
        """Pre-compute constant arrays for the RHS hot path."""
        ns = np.arange(1, self.n_modes + 1, dtype=np.float64)
        self.ns = ns

        # Mode frequency coefficients depend on boundary type
        if self.boundary == "periodic":
            # Traveling waves on a ring: omega_n = 2*n*pi / L
            # The factor of 2 vs Dirichlet comes from periodic BCs
            # requiring integer wavelengths (lambda = L/n) vs half-integer
            # (lambda = 2a/n for standing waves).
            self.ns_pi = 2.0 * ns * PI
            self.ns_pi_sq = (2.0 * ns * PI) ** 2
        else:
            # Dirichlet standing waves: omega_n = n*pi / a
            self.ns_pi = ns * PI
            self.ns_pi_sq = (ns * PI) ** 2

        self.q_eq = self.q0

        # Form factor for UV regularization
        a0 = self.q0 - self.x_left
        if self.plate_thickness <= 0:
            # Auto: delta = a0/100, so n_cutoff = 100
            self.n_cutoff = 100.0
        else:
            self.n_cutoff = a0 / self.plate_thickness
        self.g_n = mode_space.form_factor(self.n_modes, self.n_cutoff, self.cutoff_shape)

        # Form-factor-weighted pre-computed arrays for the RHS hot path
        # omega_n^2 = g_n * (n*pi)^2 / a^2  [or g_n * (2*n*pi)^2 / L^2]
        self.ns_pi_sq_g = self.ns_pi_sq * self.g_n


@dataclass
class SimulationResult:
    """Results from a Track B simulation run."""

    t: NDArray                            # time points, shape (n_t,)
    y: NDArray                            # state history, shape (4N+2, n_t)
    sol: Optional[OdeSolution] = None     # dense output (if requested)
    config: Optional[SimulationConfig] = None
    rhs_call_count: int = 0
    energy_violations: list = dataclass_field(default_factory=list)
    termination_message: str = ""

    @property
    def n_modes(self) -> int:
        return self.config.n_modes if self.config else (self.y.shape[0] - 2) // 4

    @property
    def plate_q(self) -> NDArray:
        """Plate position time series."""
        idx = 4 * self.n_modes
        return self.y[idx, :]

    @property
    def plate_v(self) -> NDArray:
        """Plate velocity time series."""
        idx = 4 * self.n_modes
        return self.y[idx + 1, :]

    def mode_state_at(self, i: int) -> NDArray:
        """Mode sub-vector at time index i. Shape (4N,)."""
        idx = 4 * self.n_modes
        return self.y[:idx, i]

    def particle_number_at(self, i: int) -> NDArray:
        """Per-mode particle number at time index i. Shape (N,)."""
        ms = self.mode_state_at(i)
        a = self.y[4 * self.n_modes, i] - self.config.x_left
        return mode_space.particle_number(ms, self.n_modes, a, self.config.g_n, self.config.ns_pi)

    def total_particle_number_at(self, i: int) -> float:
        return float(np.sum(self.particle_number_at(i)))

    def energy_at(self, i: int) -> dict:
        """Energy components at time index i."""
        idx = 4 * self.n_modes
        ms = self.y[:idx, i]
        q = self.y[idx, i]
        v = self.y[idx + 1, i]
        a = q - self.config.x_left
        return energy_mod.energy_components(
            ms, self.n_modes, a,
            self.config.plate_mass, v,
            self.config.spring_k, q, self.config.q_eq,
            self.config.g_n, self.config.ns_pi,
        )


def pack_state(mode_state: NDArray, q: float, v: float) -> NDArray:
    """Concatenate mode state and plate state into full ODE vector.

    Parameters
    ----------
    mode_state : NDArray, shape (4*N,)
    q : float
        Plate position.
    v : float
        Plate velocity.

    Returns
    -------
    NDArray, shape (4*N + 2,)
    """
    return np.concatenate([mode_state, [q, v]])


def unpack_state(y: NDArray, n_modes: int) -> tuple[NDArray, float, float]:
    """Split full state into (mode_state, q, v).

    Parameters
    ----------
    y : NDArray, shape (4*N + 2,)
    n_modes : int

    Returns
    -------
    tuple of (NDArray[4N], float, float)
    """
    idx = 4 * n_modes
    return y[:idx], float(y[idx]), float(y[idx + 1])


def make_rhs(cfg: SimulationConfig) -> Callable:
    """Build the ODE right-hand-side function as a closure.

    Captures pre-computed arrays (ns_pi_sq, etc.) to minimize
    per-call overhead. The returned function has the signature
    required by scipy.integrate.solve_ivp: rhs(t, y) -> dydt.

    The closure also tracks call count via rhs.call_count.
    """
    N = cfg.n_modes
    idx = 4 * N
    M = cfg.plate_mass
    k = cfg.spring_k
    q_eq = cfg.q_eq
    x_L = cfg.x_left
    ns_pi_sq_g = cfg.ns_pi_sq_g  # shape (N,): g_n * n^2 * pi^2

    # Static vacuum force with form factor (boundary-aware):
    # F_vac = sum_n ns_pi_sq_g[n] / a^3 * |f_n_vac|^2
    #       = sum_n ns_pi_sq_g[n] / a^3 * a/(2*ns_pi[n]*sqrt(g_n))
    #       = (1/(2*a^2)) * sum_n ns_pi[n] * sqrt(g_n)
    _vac_force_coeff = 0.5 * float(np.dot(cfg.ns_pi, np.sqrt(cfg.g_n)))

    call_count = [0]

    def rhs(t: float, y: NDArray) -> NDArray:
        call_count[0] += 1
        dydt = np.empty_like(y)

        # -- Extract plate state --
        q = y[idx]
        v_plate = y[idx + 1]
        a = q - x_L  # cavity width

        # -- Mode frequencies squared: omega_n^2 = g_n * (n*pi)^2 / a^2 --
        inv_a_sq = 1.0 / (a * a)
        omega_sq = ns_pi_sq_g * inv_a_sq  # shape (N,)

        # -- Extract mode variables (stride-4 with explicit stop) --
        u = y[0:idx:4]          # u_n
        u_dot = y[1:idx:4]      # u_dot_n
        v = y[2:idx:4]          # v_n (imaginary part, not plate velocity)
        v_dot = y[3:idx:4]      # v_dot_n

        # -- Mode ODEs: f_n'' + omega_n^2 * f_n = 0 --
        dydt[0:idx:4] = u_dot
        dydt[1:idx:4] = -omega_sq * u
        dydt[2:idx:4] = v_dot
        dydt[3:idx:4] = -omega_sq * v

        # -- Renormalized field force on plate --
        # F_raw = sum g_n * (n^2 * pi^2 / a^3) * |f_n|^2
        # F_vac = sum g_n * n * pi / (2 * a^2)   [static vacuum with form factor]
        # F_ren = F_raw - F_vac
        f_sq = u * u + v * v  # |f_n|^2, shape (N,)
        F_raw = np.dot(ns_pi_sq_g, f_sq) / (a * a * a)
        F_vac = _vac_force_coeff * inv_a_sq
        F_field = F_raw - F_vac

        # -- Plate ODE --
        dydt[idx] = v_plate
        dydt[idx + 1] = (-k * (q - q_eq) + F_field) / M

        return dydt

    rhs.call_count = call_count
    return rhs


def make_prescribed_rhs(cfg: SimulationConfig,
                        q_func: Callable[[float], float],
                        v_func: Callable[[float], float]) -> Callable:
    """Build RHS with externally prescribed plate motion.

    Used for validation Gate 4.2 (dynamic Casimir effect with known
    analytical solution). The plate DOFs are overridden; only mode
    ODEs evolve dynamically.

    Parameters
    ----------
    cfg : SimulationConfig
    q_func : callable
        q(t) -> plate position at time t.
    v_func : callable
        v(t) -> plate velocity at time t. Must be consistent derivative of q_func.
    """
    N = cfg.n_modes
    idx = 4 * N
    x_L = cfg.x_left
    ns_pi_sq_g = cfg.ns_pi_sq_g

    call_count = [0]

    def rhs(t: float, y: NDArray) -> NDArray:
        call_count[0] += 1
        dydt = np.empty_like(y)

        # Prescribed plate motion
        q = q_func(t)
        a = q - x_L
        inv_a_sq = 1.0 / (a * a)
        omega_sq = ns_pi_sq_g * inv_a_sq

        # Mode ODEs
        dydt[0:idx:4] = y[1:idx:4]           # u_dot
        dydt[1:idx:4] = -omega_sq * y[0:idx:4]  # u_ddot
        dydt[2:idx:4] = y[3:idx:4]           # v_dot
        dydt[3:idx:4] = -omega_sq * y[2:idx:4]  # v_ddot

        # Plate DOFs track prescribed values (small integration error is cosmetic)
        dydt[idx] = v_func(t)
        dydt[idx + 1] = 0.0  # placeholder

        return dydt

    rhs.call_count = call_count
    return rhs


def build_initial_state(cfg: SimulationConfig) -> NDArray:
    """Construct the initial state vector from config.

    Mode functions initialized to exact vacuum. Plate at q0 with velocity v0.

    Returns
    -------
    NDArray, shape (4*N + 2,)
    """
    a0 = cfg.q0 - cfg.x_left
    mode_ic = mode_space.vacuum_initial_conditions(cfg.n_modes, a0, cfg.g_n, cfg.ns_pi)
    return pack_state(mode_ic, cfg.q0, cfg.v0)


def _make_cavity_collapse_event(cfg: SimulationConfig):
    """Terminal event: stop if cavity width drops below 1% of initial."""
    a0 = cfg.q0 - cfg.x_left
    min_a = 0.01 * a0
    idx = 4 * cfg.n_modes

    def cavity_collapse(t, y):
        return y[idx] - cfg.x_left - min_a

    cavity_collapse.terminal = True
    cavity_collapse.direction = -1
    return cavity_collapse


def run_simulation(cfg: SimulationConfig,
                   prescribed_motion: Optional[tuple[Callable, Callable]] = None,
                   extra_events: Optional[list] = None) -> SimulationResult:
    """Run a full Track B simulation.

    Parameters
    ----------
    cfg : SimulationConfig
        Full simulation configuration.
    prescribed_motion : optional tuple of (q_func, v_func)
        If provided, plate follows prescribed motion (for validation).
    extra_events : optional list
        Additional event functions for solve_ivp.

    Returns
    -------
    SimulationResult
    """
    # Build initial state
    y0 = build_initial_state(cfg)

    # Build RHS
    if prescribed_motion is not None:
        q_func, v_func = prescribed_motion
        rhs = make_prescribed_rhs(cfg, q_func, v_func)
    else:
        rhs = make_rhs(cfg)

    # Events
    events = [_make_cavity_collapse_event(cfg)]
    if extra_events:
        events.extend(extra_events)

    # Integrate
    sol = solve_ivp(
        rhs,
        t_span=cfg.t_span,
        y0=y0,
        method=cfg.method,
        rtol=cfg.rtol,
        atol=cfg.atol,
        max_step=cfg.max_step,
        t_eval=cfg.t_eval,
        dense_output=cfg.dense_output,
        events=events,
    )

    # Build result
    result = SimulationResult(
        t=sol.t,
        y=sol.y,
        sol=sol.sol if cfg.dense_output else None,
        config=cfg,
        rhs_call_count=rhs.call_count[0],
        termination_message=sol.message,
    )

    return result


def audit_result(result: SimulationResult,
                 check_every: int = 1) -> audit_mod.EnergyAuditor:
    """Run energy audit on a completed simulation result.

    Parameters
    ----------
    result : SimulationResult
    check_every : int
        Audit every N-th time point (1 = every point).

    Returns
    -------
    EnergyAuditor
        Auditor with all records populated. Access .summary() for results.

    Raises
    ------
    PhysicalIntegrityError
        If conservation is violated (and audit_halt is True in config).
    """
    cfg = result.config
    auditor = audit_mod.EnergyAuditor(
        tolerance_factor=cfg.audit_tolerance_factor,
        halt_on_violation=cfg.audit_halt,
    )

    # Reference energy at t=0
    e0 = result.energy_at(0)
    auditor.set_reference(e0["E_total"], e0["E_plate"])

    # Check at selected time points
    n_t = len(result.t)
    for i in range(0, n_t, check_every):
        e = result.energy_at(i)
        auditor.check(
            t=result.t[i],
            E_plate=e["E_plate"],
            E_spring=e["E_spring"],
            E_field=e["E_field"],
            step=i,
        )

    return auditor
