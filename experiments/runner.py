"""
Segmented experiment runner for long Track B simulations.

Runs solve_ivp in time segments to avoid holding the full trajectory
in memory. Streams scalar observables to HDF5 and checkpoints full
state periodically for restart.

Typical usage:
    from experiments.runner import ExperimentRunner, RunConfig
    rc = RunConfig(total_time=1e5, segment_time=1000.0)
    runner = ExperimentRunner(sim_cfg, rc, output_path="data/experiments/closed_ringdown.h5")
    runner.run()

Restart from checkpoint:
    runner = ExperimentRunner.from_checkpoint("data/experiments/closed_ringdown.h5")
    runner.run()
"""

# h5py ships loose type stubs (Group.__getitem__ -> Group|Dataset|Datatype), so
# correct dataset/attr usage in this I/O glue trips the type checker. Suppress
# those stub-driven rules here; the physics core keeps full type checking.
# pyright: reportIndexIssue=false, reportArgumentType=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportFunctionMemberAccess=false

import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import h5py
from numpy.typing import NDArray

from nothing_engine.core.bogoliubov import (
    SimulationConfig, SimulationResult,
    build_initial_state, pack_state, unpack_state,
    make_rhs, _make_cavity_collapse_event,
)
from nothing_engine.core import energy as energy_mod
from nothing_engine.core import mode_space
from nothing_engine.core import radiation_pressure
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)

MODEL_REVISION = "0.4.0"


@dataclass
class RunConfig:
    """Configuration for a long experiment run."""

    total_time: float = 1.0e5
    segment_time: float = 1000.0
    samples_per_unit_time: int = 8
    checkpoint_interval: float = 1.0e4
    log_interval_segments: int = 10

    @property
    def n_segments(self) -> int:
        return int(np.ceil(self.total_time / self.segment_time))


def _create_output_file(path: Path, sim_cfg: SimulationConfig,
                        run_cfg: RunConfig) -> h5py.File:
    """Create the HDF5 output file with datasets and metadata."""
    f = h5py.File(path, "w")
    f.attrs["model_revision"] = MODEL_REVISION

    # Store configs as attributes
    g = f.create_group("config")
    for attr in ["n_modes", "plate_mass", "spring_k", "q0", "v0",
                 "x_left", "method", "rtol", "atol", "max_step",
                 "boundary", "plate_thickness", "cutoff_shape"]:
        g.attrs[attr] = getattr(sim_cfg, attr)
    g.attrs["q_eq"] = sim_cfg.q_eq
    g.attrs["total_time"] = run_cfg.total_time
    g.attrs["segment_time"] = run_cfg.segment_time
    g.attrs["samples_per_unit_time"] = run_cfg.samples_per_unit_time
    g.attrs["checkpoint_interval"] = run_cfg.checkpoint_interval

    # Scalar time-series datasets (extensible along axis 0)
    ts = f.create_group("timeseries")
    maxshape = (None,)
    ts.create_dataset("t", shape=(0,), maxshape=maxshape, dtype="f8")
    ts.create_dataset("plate_q", shape=(0,), maxshape=maxshape, dtype="f8")
    ts.create_dataset("plate_v", shape=(0,), maxshape=maxshape, dtype="f8")
    ts.create_dataset("E_plate", shape=(0,), maxshape=maxshape, dtype="f8")
    ts.create_dataset("E_spring", shape=(0,), maxshape=maxshape, dtype="f8")
    ts.create_dataset("E_excitation", shape=(0,), maxshape=maxshape, dtype="f8")
    ts.create_dataset("E_casimir", shape=(0,), maxshape=maxshape, dtype="f8")
    ts.create_dataset("E_field", shape=(0,), maxshape=maxshape, dtype="f8")
    ts.create_dataset("E_total", shape=(0,), maxshape=maxshape, dtype="f8")
    ts.create_dataset("total_particles", shape=(0,), maxshape=maxshape, dtype="f8")
    ts.create_dataset("force_field", shape=(0,), maxshape=maxshape, dtype="f8")

    # Per-mode particle number (extensible: time × modes)
    n_modes = sim_cfg.n_modes
    ts.create_dataset("particle_number",
                      shape=(0, n_modes), maxshape=(None, n_modes), dtype="f8")

    # Checkpoint group
    f.create_group("checkpoints")

    # Run state
    f.attrs["status"] = "running"
    f.attrs["t_current"] = 0.0
    f.attrs["segment_index"] = 0
    f.attrs["wall_time_seconds"] = 0.0

    return f


def _append_observables(f: h5py.File, t_pts: NDArray, y_pts: NDArray,
                        sim_cfg: SimulationConfig):
    """Extract observables from state array and append to HDF5 datasets."""
    ts = f["timeseries"]
    n_pts = len(t_pts)
    n_modes = sim_cfg.n_modes
    idx = 4 * n_modes

    # Plate state
    q_arr = y_pts[idx, :]
    v_arr = y_pts[idx + 1, :]

    # Compute scalar observables
    e_plate = 0.5 * sim_cfg.plate_mass * v_arr**2
    e_spring = 0.5 * sim_cfg.spring_k * (q_arr - sim_cfg.q_eq)**2

    e_excitation = np.empty(n_pts)
    e_casimir = np.empty(n_pts)
    e_field = np.empty(n_pts)
    n_total = np.empty(n_pts)
    f_field = np.empty(n_pts)
    pn_arr = np.empty((n_pts, n_modes))

    ns_pi_sq_g = sim_cfg.ns_pi_sq_g
    g_n = sim_cfg.g_n

    for i in range(n_pts):
        ms = y_pts[:idx, i]
        a = q_arr[i] - sim_cfg.x_left
        components = energy_mod.energy_components(
            ms, n_modes, a,
            sim_cfg.plate_mass, float(v_arr[i]),
            sim_cfg.spring_k, float(q_arr[i]), sim_cfg.q_eq,
            g_n, sim_cfg.ns_pi,
            sim_cfg.boundary, sim_cfg.mode_degeneracy,
        )
        e_excitation[i] = components["E_excitation"]
        e_casimir[i] = components["E_casimir"]
        e_field[i] = components["E_field"]
        pn = sim_cfg.mode_degeneracy * mode_space.particle_number(
            ms, n_modes, a, g_n, sim_cfg.ns_pi,
        )
        pn_arr[i] = pn
        n_total[i] = float(np.sum(pn))
        f_field[i] = radiation_pressure.renormalized_field_force(
            ms, n_modes, a, ns_pi_sq_g, sim_cfg.ns_pi, g_n,
            sim_cfg.boundary, sim_cfg.mode_degeneracy,
        )

    e_total = e_plate + e_spring + e_field

    # Append to datasets
    old_len = ts["t"].shape[0]
    new_len = old_len + n_pts

    for name in ("E_excitation", "E_casimir"):
        if name not in ts:
            ds = ts.create_dataset(
                name, shape=(old_len,), maxshape=(None,), dtype="f8",
            )
            ds[:] = np.nan

    for name, data in [("t", t_pts), ("plate_q", q_arr), ("plate_v", v_arr),
                       ("E_plate", e_plate), ("E_spring", e_spring),
                       ("E_excitation", e_excitation),
                       ("E_casimir", e_casimir),
                       ("E_field", e_field), ("E_total", e_total),
                       ("total_particles", n_total), ("force_field", f_field)]:
        ds = ts[name]
        ds.resize(new_len, axis=0)
        ds[old_len:new_len] = data

    ds = ts["particle_number"]
    ds.resize(new_len, axis=0)
    ds[old_len:new_len, :] = pn_arr


def _save_checkpoint(f: h5py.File, t: float, state: NDArray, seg_idx: int):
    """Save full state vector as a checkpoint."""
    ckpt = f["checkpoints"]
    name = f"seg_{seg_idx:06d}"
    if name in ckpt:
        del ckpt[name]
    g = ckpt.create_group(name)
    g.attrs["t"] = t
    g.attrs["segment_index"] = seg_idx
    g.create_dataset("state", data=state)


class ExperimentRunner:
    """Runs a long reduced-mode simulation in segments, streaming to HDF5."""

    def __init__(self, sim_cfg: SimulationConfig, run_cfg: RunConfig,
                 output_path: str,
                 initial_state: Optional[NDArray] = None,
                 start_time: float = 0.0,
                 start_segment: int = 0,
                 is_resume: bool = False,
                 progress_callback: Optional[Callable[[dict], None]] = None):
        self.sim_cfg = sim_cfg
        self.run_cfg = run_cfg
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Optional per-segment progress hook. Default None = unchanged behavior;
        # the scenario CLIs and library callers are unaffected.
        self._progress_callback = progress_callback

        if initial_state is not None:
            self._state = initial_state
        else:
            self._state = build_initial_state(sim_cfg)

        self._t_current = start_time
        self._seg_idx = start_segment
        self._last_checkpoint_t = start_time
        # True only when constructed via from_checkpoint. Decides append-vs-recreate in
        # run(): never key that off seg_idx, which is 0 for a crash-before-first-checkpoint.
        self._is_resume = is_resume

    @classmethod
    def from_checkpoint(cls, hdf5_path: str) -> "ExperimentRunner":
        """Resume a run from the last checkpoint in an existing HDF5 file."""
        with h5py.File(hdf5_path, "r") as f:
            revision = str(f.attrs.get("model_revision", "legacy"))
            if revision != MODEL_REVISION:
                raise ValueError(
                    "Checkpoint model revision "
                    f"{revision!r} cannot resume under {MODEL_REVISION!r}. "
                    "The static force, periodic degeneracy, and cutoff defaults "
                    "changed. Analyze the old file as-is or rerun it."
                )
            cfg_grp = f["config"]

            # Physics-affecting fields must round-trip or the resumed run silently
            # changes physics (e.g. a periodic run reverting to the closed default).
            # boundary/plate_thickness are always written; cutoff_shape is guarded for
            # files produced before it was persisted.
            sim_cfg = SimulationConfig(
                n_modes=int(cfg_grp.attrs["n_modes"]),
                plate_mass=float(cfg_grp.attrs["plate_mass"]),
                spring_k=float(cfg_grp.attrs["spring_k"]),
                q0=float(cfg_grp.attrs["q0"]),
                v0=float(cfg_grp.attrs["v0"]),
                x_left=float(cfg_grp.attrs["x_left"]),
                boundary=str(cfg_grp.attrs["boundary"]),
                plate_thickness=float(cfg_grp.attrs["plate_thickness"]),
                cutoff_shape=(str(cfg_grp.attrs["cutoff_shape"])
                              if "cutoff_shape" in cfg_grp.attrs else "sigmoid"),
                method=str(cfg_grp.attrs["method"]),
                rtol=float(cfg_grp.attrs["rtol"]),
                atol=float(cfg_grp.attrs["atol"]),
                max_step=float(cfg_grp.attrs["max_step"]),
            )

            run_cfg = RunConfig(
                total_time=float(cfg_grp.attrs["total_time"]),
                segment_time=float(cfg_grp.attrs["segment_time"]),
                samples_per_unit_time=int(cfg_grp.attrs["samples_per_unit_time"]),
                checkpoint_interval=float(cfg_grp.attrs["checkpoint_interval"]),
            )

            # Find latest checkpoint
            ckpts = f["checkpoints"]
            if len(ckpts) == 0:
                raise ValueError("No checkpoints found in file")

            latest = sorted(ckpts.keys())[-1]
            g = ckpts[latest]
            state = g["state"][:]
            t_current = float(g.attrs["t"])
            seg_idx = int(g.attrs["segment_index"])

        runner = cls(
            sim_cfg=sim_cfg,
            run_cfg=run_cfg,
            output_path=hdf5_path,
            initial_state=state,
            start_time=t_current,
            start_segment=seg_idx,
            is_resume=True,
        )
        return runner

    def _emit_progress(self, t_end: float, wall_start: float, status: str):
        """Push a progress snapshot to the callback (no-op if none registered).

        Cheap relative to the ODE solve: one field-energy + particle-number pass
        over the current state. Field quantities are skipped (NaN) if the cavity
        has collapsed (a <= 0), so the terminal callback can't divide by zero.
        """
        cb = self._progress_callback
        if cb is None:
            return
        cfg = self.sim_cfg
        n = cfg.n_modes
        idx = 4 * n
        q = float(self._state[idx])
        v = float(self._state[idx + 1])
        a = q - cfg.x_left
        e_plate = 0.5 * cfg.plate_mass * v**2
        e_spring = 0.5 * cfg.spring_k * (q - cfg.q_eq) ** 2
        if a > 0.0:
            ms = self._state[:idx]
            e_field = energy_mod.field_energy(
                ms, n, a, cfg.g_n, cfg.ns_pi,
                cfg.boundary, cfg.mode_degeneracy,
            )
            n_total = mode_space.total_particle_number(
                ms, n, a, cfg.g_n, cfg.ns_pi, cfg.mode_degeneracy,
            )
            e_total = e_plate + e_spring + e_field
        else:
            e_field = e_total = n_total = float("nan")
        cb({
            "t": self._t_current,
            "t_end": t_end,
            "pct": 100.0 * self._t_current / t_end if t_end else 100.0,
            "segment": self._seg_idx,
            "E_plate": e_plate,
            "E_spring": e_spring,
            "E_field": e_field,
            "E_total": e_total,
            "N_total": n_total,
            "wall": time.time() - wall_start,
            "status": status,
        })

    def run(self) -> Path:
        """Execute the full experiment, returning the output file path."""
        rc = self.run_cfg
        cfg = self.sim_cfg
        t_end = rc.total_time
        wall_start = time.time()

        # Create or open output file. Append whenever we were built from a checkpoint,
        # even if the only checkpoint was the initial seg-0 one: keying off seg_idx>0
        # would truncate streamed data on a crash-before-first-checkpoint restart.
        is_restart = self._is_resume
        if is_restart:
            f = h5py.File(self.output_path, "a")
            logger.info("Resuming from t=%.1f (segment %d)", self._t_current, self._seg_idx)
        else:
            f = _create_output_file(self.output_path, cfg, rc)
            # Save initial checkpoint
            _save_checkpoint(f, self._t_current, self._state, 0)
            # Save initial observables (t=0)
            t0_arr = np.array([self._t_current])
            y0_arr = self._state.reshape(-1, 1)
            _append_observables(f, t0_arr, y0_arr, cfg)
            logger.info("Created output file: %s", self.output_path)

        try:
            while self._t_current < t_end:
                seg_t_start = self._t_current
                seg_t_end = min(self._t_current + rc.segment_time, t_end)

                # Build t_eval for this segment
                n_samples = max(int((seg_t_end - seg_t_start) * rc.samples_per_unit_time), 2)
                # Exclude first point (already saved from previous segment or initial)
                t_eval = np.linspace(seg_t_start, seg_t_end, n_samples + 1)[1:]

                # Build RHS and events
                rhs = make_rhs(cfg)
                events = [_make_cavity_collapse_event(cfg)]

                # Solve this segment
                sol = solve_ivp(
                    rhs,
                    t_span=(seg_t_start, seg_t_end),
                    y0=self._state,
                    method=cfg.method,
                    rtol=cfg.rtol,
                    atol=cfg.atol,
                    max_step=cfg.max_step,
                    t_eval=t_eval,
                    dense_output=False,
                    events=events,
                )

                if sol.status == 1:
                    # Terminal event (cavity collapse)
                    logger.warning("Cavity collapse at t=%.4f, terminating.", sol.t[-1])
                    _append_observables(f, sol.t, sol.y, cfg)
                    self._state = sol.y[:, -1]
                    self._t_current = float(sol.t[-1])
                    _save_checkpoint(f, self._t_current, self._state, self._seg_idx)
                    f.attrs["status"] = "collapsed"
                    self._emit_progress(t_end, wall_start, "collapsed")
                    break

                if sol.status != 0:
                    logger.error("Solver failed: %s", sol.message)
                    _save_checkpoint(f, self._t_current, self._state, self._seg_idx)
                    f.attrs["status"] = "failed"
                    f.attrs["error"] = sol.message
                    self._emit_progress(t_end, wall_start, "failed")
                    break

                # Append observables
                _append_observables(f, sol.t, sol.y, cfg)

                # Update state
                self._state = sol.y[:, -1]
                self._t_current = float(sol.t[-1])
                self._seg_idx += 1

                # Checkpoint if needed
                if self._t_current - self._last_checkpoint_t >= rc.checkpoint_interval:
                    _save_checkpoint(f, self._t_current, self._state, self._seg_idx)
                    self._last_checkpoint_t = self._t_current
                    f.flush()

                # Update file attrs
                f.attrs["t_current"] = self._t_current
                f.attrs["segment_index"] = self._seg_idx
                f.attrs["wall_time_seconds"] = time.time() - wall_start

                # Push live progress to the callback (every segment)
                self._emit_progress(t_end, wall_start, "running")

                # Log progress
                if self._seg_idx % rc.log_interval_segments == 0:
                    elapsed = time.time() - wall_start
                    pct = 100.0 * self._t_current / t_end
                    e_plate = 0.5 * cfg.plate_mass * float(self._state[4 * cfg.n_modes + 1])**2
                    logger.info(
                        "Segment %d: t=%.1f/%.0f (%.1f%%) | E_plate=%.6e | wall=%.1fs | rhs_calls=%d",
                        self._seg_idx, self._t_current, t_end, pct,
                        e_plate, elapsed, rhs.call_count[0],
                    )
            else:
                # Normal completion
                _save_checkpoint(f, self._t_current, self._state, self._seg_idx)
                f.attrs["status"] = "completed"
                self._emit_progress(t_end, wall_start, "completed")

            f.attrs["wall_time_seconds"] = time.time() - wall_start
            logger.info("Run finished: status=%s, t=%.1f, wall=%.1fs",
                        f.attrs["status"], self._t_current,
                        f.attrs["wall_time_seconds"])

        finally:
            f.close()

        return self.output_path
