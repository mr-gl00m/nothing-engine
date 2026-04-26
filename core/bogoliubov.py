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

    Pure value type: every field is declared, nothing is derived here.
    Call :meth:`precompute` (or :func:`PrecomputedArrays.from_config`)
    to obtain the derived hot-path arrays needed by RHS builders,
    initial-state builders, energy/particle analysis, and checkpoint
    restore. Keeping derivation out of ``__post_init__`` makes the
    config fully round-trippable through HDF5 attrs or YAML.

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
    # Opt-in: recompute g_n(a(t)) each step so the UV cutoff n_cutoff =
    # a(t) / plate_thickness tracks the plate position. Default False
    # preserves reproducibility of runs where g_n was fixed at a0.
    form_factor_tracks_plate: bool = False

    # Integrator
    method: str = "DOP853"
    rtol: float = 1.0e-10
    atol: float = 1.0e-12
    max_step: float = 0.0  # 0 = auto (Nyquist-safe from n_modes and q0)
    t_span: tuple = (0.0, 1000.0)
    t_eval: Optional[NDArray] = None
    dense_output: bool = True

    # Energy audit
    audit_tolerance_factor: float = 1.0e-6
    audit_halt: bool = True

    def precompute(self) -> "PrecomputedArrays":
        return PrecomputedArrays.from_config(self)

    def effective_max_step(self) -> float:
        """Nyquist-safe max_step derived from n_modes and q0.

        The highest cavity mode has period T_N = 2*a/N (closed) or L/N
        (periodic). Returning T_N / 4 keeps RK family integrators well
        below the Nyquist limit. Honored only when max_step == 0.
        """
        if self.max_step > 0.0:
            return self.max_step
        a0 = self.q0 - self.x_left
        if self.boundary == "periodic":
            period_hi = a0 / self.n_modes
        else:
            period_hi = 2.0 * a0 / self.n_modes
        return period_hi / 4.0


@dataclass
class PrecomputedArrays:
    """Derived hot-path arrays computed once from a :class:`SimulationConfig`.

    Split out from the config so that the config is a pure, serializable
    value type. Callers that hit the RHS, build initial conditions,
    compute energies, or analyze results pass this alongside the config.
    """

    ns: NDArray
    ns_pi: NDArray
    ns_pi_sq: NDArray
    ns_pi_sq_g: NDArray
    g_n: NDArray
    n_cutoff: float
    q_eq: float
    vac_force_coeff: float
    # Per-mode vacuum |f_n|^2 scale: |f_n|^2_vac(a) = a * inv_2_ns_pi_sqrt_g.
    # Used by the RHS to subtract the vacuum force per-mode, avoiding the
    # catastrophic cancellation of F_raw - F_vac when both are O(N^3).
    inv_2_ns_pi_sqrt_g: NDArray
    # Plate-thickness-tracking opt-in: the physical delta that sets the
    # cutoff, kept so the RHS can recompute n_cutoff(a(t)) = a(t)/delta.
    plate_thickness_effective: float = 0.0
    cutoff_shape: str = "sigmoid"

    @classmethod
    def from_config(cls, cfg: SimulationConfig) -> "PrecomputedArrays":
        ns = np.arange(1, cfg.n_modes + 1, dtype=np.float64)

        if cfg.boundary == "periodic":
            ns_pi = 2.0 * ns * PI
        elif cfg.boundary == "closed":
            ns_pi = ns * PI
        else:
            raise ValueError(
                f"Unknown boundary={cfg.boundary!r}; expected 'closed' or 'periodic'"
            )
        ns_pi_sq = ns_pi ** 2

        a0 = cfg.q0 - cfg.x_left
        if a0 <= 0.0:
            raise ValueError(f"Non-positive initial cavity width a0={a0}")
        if cfg.plate_thickness <= 0.0:
            # Auto: delta = a0/100 -> n_cutoff = 100 at a=a0.
            plate_thickness_eff = a0 / 100.0
        else:
            plate_thickness_eff = cfg.plate_thickness
        n_cutoff = a0 / plate_thickness_eff
        g_n = mode_space.form_factor(cfg.n_modes, n_cutoff, cfg.cutoff_shape)

        ns_pi_sq_g = ns_pi_sq * g_n
        sqrt_g = np.sqrt(g_n)
        # Static vacuum force with form factor: F_vac * a^2 = 0.5 * sum ns_pi * sqrt(g_n)
        vac_force_coeff = 0.5 * float(np.dot(ns_pi, sqrt_g))
        # 1 / (2 * omega_n * a) = 1 / (2 * ns_pi * sqrt(g_n))
        inv_2_ns_pi_sqrt_g = 1.0 / (2.0 * ns_pi * sqrt_g)

        q_eq = cfg.q0  # spring anchor; physics decision kept stable for now

        return cls(
            ns=ns,
            ns_pi=ns_pi,
            ns_pi_sq=ns_pi_sq,
            ns_pi_sq_g=ns_pi_sq_g,
            g_n=g_n,
            n_cutoff=n_cutoff,
            q_eq=q_eq,
            vac_force_coeff=vac_force_coeff,
            inv_2_ns_pi_sqrt_g=inv_2_ns_pi_sqrt_g,
            plate_thickness_effective=plate_thickness_eff,
            cutoff_shape=cfg.cutoff_shape,
        )


@dataclass
class SimulationResult:
    """Results from a Track B simulation run."""

    t: NDArray                                       # time points, shape (n_t,)
    y: NDArray                                       # state history, shape (4N+2, n_t)
    sol: Optional[OdeSolution] = None                # dense output (if requested)
    config: Optional[SimulationConfig] = None
    precomputed: Optional[PrecomputedArrays] = None
    rhs_call_count: int = 0
    energy_violations: list = dataclass_field(default_factory=list)
    termination_message: str = ""

    def _pre(self) -> PrecomputedArrays:
        if self.precomputed is None:
            if self.config is None:
                raise ValueError("SimulationResult missing both config and precomputed")
            self.precomputed = PrecomputedArrays.from_config(self.config)
        return self.precomputed

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
        pre = self._pre()
        ms = self.mode_state_at(i)
        a = self.y[4 * self.n_modes, i] - self.config.x_left
        return mode_space.particle_number(ms, self.n_modes, a, pre.g_n, pre.ns_pi)

    def total_particle_number_at(self, i: int) -> float:
        return float(np.sum(self.particle_number_at(i)))

    def energy_at(self, i: int) -> dict:
        """Energy components at time index i."""
        pre = self._pre()
        idx = 4 * self.n_modes
        ms = self.y[:idx, i]
        q = self.y[idx, i]
        v = self.y[idx + 1, i]
        a = q - self.config.x_left
        return energy_mod.energy_components(
            ms, self.n_modes, a,
            self.config.plate_mass, v,
            self.config.spring_k, q, pre.q_eq,
            pre.g_n, pre.ns_pi,
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


def make_rhs(cfg: SimulationConfig, pre: PrecomputedArrays) -> Callable:
    """Build the ODE right-hand-side function as a closure.

    Captures the precomputed hot-path arrays to minimize per-call
    overhead. The returned function has the signature required by
    scipy.integrate.solve_ivp: rhs(t, y) -> dydt.

    The closure also tracks call count via rhs.call_count.
    """
    N = cfg.n_modes
    idx = 4 * N
    M = cfg.plate_mass
    k = cfg.spring_k
    q_eq = pre.q_eq
    x_L = cfg.x_left
    ns_pi = pre.ns_pi
    ns_pi_sq = pre.ns_pi_sq
    ns_pi_sq_g = pre.ns_pi_sq_g  # shape (N,): g_n * n^2 * pi^2 (static g_n)
    inv_2_ns_pi_sqrt_g = pre.inv_2_ns_pi_sqrt_g  # shape (N,): 1/(2*ns_pi*sqrt(g_n))
    track_g = cfg.form_factor_tracks_plate
    delta_plate = pre.plate_thickness_effective
    cutoff_shape = pre.cutoff_shape
    two_inv_ns_pi = 1.0 / (2.0 * ns_pi)  # used only in tracking branch

    call_count = [0]

    def rhs(t: float, y: NDArray) -> NDArray:
        call_count[0] += 1
        dydt = np.empty_like(y)

        # -- Extract plate state --
        q = y[idx]
        v_plate = y[idx + 1]
        a = q - x_L  # cavity width

        # -- Form factor: static (default) or tracking the plate --
        if track_g:
            n_cut_t = a / delta_plate
            g_n_t = mode_space.form_factor(N, n_cut_t, cutoff_shape)
            ns_pi_sq_g_t = ns_pi_sq * g_n_t
            inv_2_ns_pi_sqrt_g_t = two_inv_ns_pi / np.sqrt(g_n_t)
        else:
            g_n_t = None  # unused
            ns_pi_sq_g_t = ns_pi_sq_g
            inv_2_ns_pi_sqrt_g_t = inv_2_ns_pi_sqrt_g

        # -- Mode frequencies squared: omega_n^2 = g_n * (n*pi)^2 / a^2 --
        inv_a_sq = 1.0 / (a * a)
        omega_sq = ns_pi_sq_g_t * inv_a_sq  # shape (N,)

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

        # -- Renormalized field force on plate (per-mode subtraction) --
        # Algebraically F_ren = sum g_n * (n*pi)^2 / a^3 * (|f_n|^2 - 1/(2*omega_n))
        # where 1/(2*omega_n) = a / (2*ns_pi*sqrt(g_n)) is the vacuum |f_n|^2.
        # Subtracting per-mode avoids the O(N^3) catastrophic cancellation
        # of F_raw - F_vac at the sum level.
        f_sq = u * u + v * v  # |f_n|^2, shape (N,)
        f_sq_excess = f_sq - a * inv_2_ns_pi_sqrt_g_t
        F_field = np.dot(ns_pi_sq_g_t, f_sq_excess) / (a * a * a)

        # -- Plate ODE --
        dydt[idx] = v_plate
        dydt[idx + 1] = (-k * (q - q_eq) + F_field) / M

        return dydt

    rhs.call_count = call_count
    return rhs


def make_prescribed_rhs(cfg: SimulationConfig,
                        pre: PrecomputedArrays,
                        q_func: Callable[[float], float],
                        v_func: Callable[[float], float],
                        a_func: Optional[Callable[[float], float]] = None) -> Callable:
    """Build RHS with externally prescribed plate motion.

    Used for validation Gate 4.2 (dynamic Casimir effect with known
    analytical solution). The plate DOFs are overridden; only mode
    ODEs evolve dynamically.

    Parameters
    ----------
    cfg : SimulationConfig
    pre : PrecomputedArrays
    q_func : callable
        q(t) -> plate position at time t.
    v_func : callable
        v(t) -> plate velocity at time t. Must be consistent derivative of q_func.
    a_func : callable, optional
        a(t) -> plate acceleration. When provided, the integrated plate
        velocity variable is steered back to ``v_func(t)`` exactly; when
        omitted, a proportional corrector keeps drift bounded.
    """
    N = cfg.n_modes
    idx = 4 * N
    x_L = cfg.x_left
    ns_pi_sq_g = pre.ns_pi_sq_g

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

        # Plate DOFs track prescribed values. Use acceleration when given;
        # otherwise pull v_plate toward v_func(t) so it does not wander.
        dydt[idx] = v_func(t)
        if a_func is not None:
            dydt[idx + 1] = a_func(t)
        else:
            dydt[idx + 1] = v_func(t) - y[idx + 1]

        return dydt

    rhs.call_count = call_count
    return rhs


def build_initial_state(cfg: SimulationConfig, pre: PrecomputedArrays) -> NDArray:
    """Construct the initial state vector from config.

    Mode functions initialized to exact vacuum. Plate at q0 with velocity v0.

    Returns
    -------
    NDArray, shape (4*N + 2,)
    """
    a0 = cfg.q0 - cfg.x_left
    mode_ic = mode_space.vacuum_initial_conditions(cfg.n_modes, a0, pre.g_n, pre.ns_pi)
    return pack_state(mode_ic, cfg.q0, cfg.v0)


def _make_cavity_collapse_event(cfg: SimulationConfig, pre: PrecomputedArrays):
    """Terminal event: stop if cavity width drops below max(1% of a0, 2*plate_thickness).

    The lower bound protects against the form factor cutoff range:
    ``a_min >= 2*plate_thickness`` keeps the RHS form factor meaningful.
    """
    a0 = cfg.q0 - cfg.x_left
    delta = a0 / pre.n_cutoff if pre.n_cutoff > 0 else 0.0
    min_a = max(0.01 * a0, 2.0 * delta)
    idx = 4 * cfg.n_modes

    def cavity_collapse(t, y):
        return y[idx] - cfg.x_left - min_a

    cavity_collapse.terminal = True
    cavity_collapse.direction = -1
    return cavity_collapse


def _make_nonfinite_event(cfg: SimulationConfig):
    """Terminal event: stop if the state develops a NaN or Inf.

    ``solve_ivp`` polls the event function after each accepted step,
    so this catches pathological growth within one step of occurring
    rather than letting NaNs propagate until the integrator itself
    gives up with an opaque failure.
    """
    def finite_guard(t, y):
        return 1.0 if np.isfinite(y).all() else -1.0

    finite_guard.terminal = True
    finite_guard.direction = -1
    return finite_guard


def run_simulation(cfg: SimulationConfig,
                   prescribed_motion: Optional[tuple] = None,
                   extra_events: Optional[list] = None,
                   precomputed: Optional[PrecomputedArrays] = None) -> SimulationResult:
    """Run a full Track B simulation.

    Parameters
    ----------
    cfg : SimulationConfig
        Full simulation configuration.
    prescribed_motion : optional tuple
        ``(q_func, v_func)`` or ``(q_func, v_func, a_func)``. If provided,
        plate follows prescribed motion (for validation).
    extra_events : optional list
        Additional event functions for solve_ivp.
    precomputed : optional PrecomputedArrays
        Reuse a cached derivation (useful in segmented runs).

    Returns
    -------
    SimulationResult
    """
    pre = precomputed if precomputed is not None else PrecomputedArrays.from_config(cfg)

    y0 = build_initial_state(cfg, pre)

    if prescribed_motion is not None:
        if len(prescribed_motion) == 3:
            q_func, v_func, a_func = prescribed_motion
        else:
            q_func, v_func = prescribed_motion
            a_func = None
        rhs = make_prescribed_rhs(cfg, pre, q_func, v_func, a_func)
    else:
        rhs = make_rhs(cfg, pre)

    events = [_make_cavity_collapse_event(cfg, pre), _make_nonfinite_event(cfg)]
    if extra_events:
        events.extend(extra_events)

    sol = solve_ivp(
        rhs,
        t_span=cfg.t_span,
        y0=y0,
        method=cfg.method,
        rtol=cfg.rtol,
        atol=cfg.atol,
        max_step=cfg.effective_max_step(),
        t_eval=cfg.t_eval,
        dense_output=cfg.dense_output,
        events=events,
    )

    result = SimulationResult(
        t=sol.t,
        y=sol.y,
        sol=sol.sol if cfg.dense_output else None,
        config=cfg,
        precomputed=pre,
        rhs_call_count=rhs.call_count[0],
        termination_message=sol.message,
    )

    return result


def make_rhs_auto(cfg: SimulationConfig) -> Callable:
    """Convenience for callers that do not want to manage a PrecomputedArrays.

    Equivalent to ``make_rhs(cfg, cfg.precompute())``.
    """
    return make_rhs(cfg, cfg.precompute())


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
    auditor.set_reference(e0["E_total"], e0["E_plate"], e0.get("E_field"))

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
