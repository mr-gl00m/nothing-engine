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

import os
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import h5py
from numpy.typing import NDArray

from nothing_engine.core.bogoliubov import (
    SimulationConfig, SimulationResult, PrecomputedArrays,
    build_initial_state, pack_state, unpack_state,
    make_rhs, _make_cavity_collapse_event, _make_nonfinite_event,
)
from nothing_engine.core import energy as energy_mod
from nothing_engine.core import mode_space
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)

# All fields of SimulationConfig that must survive the HDF5 round trip.
# Anything derived (ns, ns_pi, g_n, q_eq, ...) is rebuilt via
# PrecomputedArrays.from_config and must NOT be persisted under this key.
_CONFIG_FIELDS: tuple = (
    "n_modes", "plate_mass", "spring_k", "q0", "v0", "x_left",
    "boundary", "plate_thickness", "cutoff_shape", "form_factor_tracks_plate",
    "method", "rtol", "atol", "max_step",
    "audit_tolerance_factor", "audit_halt",
)


def _atomic_replace(src: Path, dst: Path) -> None:
    """Cross-platform atomic replace.

    ``os.replace`` is atomic on POSIX and provides the same semantics on
    Windows for files on the same volume. Callers must ensure ``src`` is
    fully flushed and closed before invoking this.
    """
    os.replace(str(src), str(dst))


def _store_config_attrs(cfg_grp: h5py.Group, sim_cfg: SimulationConfig,
                        run_cfg: "RunConfig") -> None:
    """Persist the full SimulationConfig plus RunConfig onto an HDF5 group.

    Every declared field of SimulationConfig is written. Derived fields
    are intentionally omitted — they are recomputed on restore. ``t_span``
    and ``t_eval`` are runtime-only and not persisted either.
    """
    for name in _CONFIG_FIELDS:
        value = getattr(sim_cfg, name)
        cfg_grp.attrs[name] = value
    cfg_grp.attrs["total_time"] = run_cfg.total_time
    cfg_grp.attrs["segment_time"] = run_cfg.segment_time
    cfg_grp.attrs["samples_per_unit_time"] = run_cfg.samples_per_unit_time
    cfg_grp.attrs["checkpoint_interval"] = run_cfg.checkpoint_interval


def _load_config_attrs(cfg_grp: h5py.Group) -> "tuple[SimulationConfig, RunConfig]":
    """Reconstruct a SimulationConfig + RunConfig from HDF5 attrs.

    Round-trip counterpart of :func:`_store_config_attrs`. Missing fields
    fall back to dataclass defaults so that files written before a new
    field was added remain readable.
    """
    kwargs = {}
    defaults = SimulationConfig()
    for name in _CONFIG_FIELDS:
        if name in cfg_grp.attrs:
            raw = cfg_grp.attrs[name]
            default = getattr(defaults, name)
            if isinstance(default, bool):
                kwargs[name] = bool(raw)
            elif isinstance(default, int) and not isinstance(default, bool):
                kwargs[name] = int(raw)
            elif isinstance(default, float):
                kwargs[name] = float(raw)
            elif isinstance(default, str):
                kwargs[name] = raw.decode() if isinstance(raw, bytes) else str(raw)
            else:
                kwargs[name] = raw
    sim_cfg = SimulationConfig(**kwargs)
    run_cfg = RunConfig(
        total_time=float(cfg_grp.attrs["total_time"]),
        segment_time=float(cfg_grp.attrs["segment_time"]),
        samples_per_unit_time=int(cfg_grp.attrs["samples_per_unit_time"]),
        checkpoint_interval=float(cfg_grp.attrs["checkpoint_interval"]),
    )
    return sim_cfg, run_cfg


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
                        run_cfg: RunConfig, overwrite: bool = False) -> h5py.File:
    """Create the HDF5 output file with datasets and metadata.

    Refuses to clobber an existing file unless ``overwrite=True``; callers
    should opt into destructive behavior explicitly.
    """
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {path}. Pass overwrite=True to replace."
        )
    f = h5py.File(path, "w")

    # Store the full config (round-trippable). Derived fields (ns, g_n,
    # q_eq, ...) are NOT persisted — they are recomputed on restore via
    # PrecomputedArrays.from_config, so there is a single source of truth.
    g = f.create_group("config")
    _store_config_attrs(g, sim_cfg, run_cfg)

    # Scalar time-series datasets (extensible along axis 0)
    ts = f.create_group("timeseries")
    maxshape = (None,)
    ts.create_dataset("t", shape=(0,), maxshape=maxshape, dtype="f8")
    ts.create_dataset("plate_q", shape=(0,), maxshape=maxshape, dtype="f8")
    ts.create_dataset("plate_v", shape=(0,), maxshape=maxshape, dtype="f8")
    ts.create_dataset("E_plate", shape=(0,), maxshape=maxshape, dtype="f8")
    ts.create_dataset("E_spring", shape=(0,), maxshape=maxshape, dtype="f8")
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
                        sim_cfg: SimulationConfig, pre: PrecomputedArrays):
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
    e_spring = 0.5 * sim_cfg.spring_k * (q_arr - pre.q_eq)**2

    e_field = np.empty(n_pts)
    n_total = np.empty(n_pts)
    f_field = np.empty(n_pts)
    pn_arr = np.empty((n_pts, n_modes))

    ns_pi_sq_g = pre.ns_pi_sq_g
    g_n = pre.g_n
    ns_pi = pre.ns_pi

    for i in range(n_pts):
        ms = y_pts[:idx, i]
        a = q_arr[i] - sim_cfg.x_left
        e_field[i] = energy_mod.field_energy(ms, n_modes, a, g_n, ns_pi)
        pn = mode_space.particle_number(ms, n_modes, a, g_n, ns_pi)
        pn_arr[i] = pn
        n_total[i] = float(np.sum(pn))
        # Force = sum g_n * (n^2*pi^2 / a^3) * |f_n|^2
        f_sq = mode_space.extract_mode_amplitudes_squared(ms, n_modes)
        f_field[i] = float(np.dot(ns_pi_sq_g, f_sq)) / (a**3)

    e_total = e_plate + e_spring + e_field

    # Append to datasets
    old_len = ts["t"].shape[0]
    new_len = old_len + n_pts

    for name, data in [("t", t_pts), ("plate_q", q_arr), ("plate_v", v_arr),
                       ("E_plate", e_plate), ("E_spring", e_spring),
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
    """Runs a long Track B simulation in segments, streaming to HDF5."""

    def __init__(self, sim_cfg: SimulationConfig, run_cfg: RunConfig,
                 output_path: str,
                 initial_state: Optional[NDArray] = None,
                 start_time: float = 0.0,
                 start_segment: int = 0,
                 overwrite: bool = False):
        self.sim_cfg = sim_cfg
        self.run_cfg = run_cfg
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._overwrite = overwrite
        self._pre = PrecomputedArrays.from_config(sim_cfg)

        if initial_state is not None:
            self._state = initial_state
        else:
            self._state = build_initial_state(sim_cfg, self._pre)

        self._t_current = start_time
        self._seg_idx = start_segment
        self._last_checkpoint_t = start_time

    @classmethod
    def from_checkpoint(cls, hdf5_path: str) -> "ExperimentRunner":
        """Resume a run from the last checkpoint in an existing HDF5 file.

        Reads the full SimulationConfig back from the HDF5 attrs (every
        declared field, via :func:`_load_config_attrs`). Derived arrays
        are recomputed via ``PrecomputedArrays.from_config``. A field that
        existed at write time but was later removed from the dataclass is
        silently dropped; a field that was added later falls back to its
        current default.
        """
        with h5py.File(hdf5_path, "r") as f:
            cfg_grp = f["config"]
            sim_cfg, run_cfg = _load_config_attrs(cfg_grp)

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
        )
        return runner

    def run(self) -> Path:
        """Execute the full experiment, returning the output file path.

        Writes go to ``<output_path>.tmp`` during the run and are atomically
        replaced onto ``output_path`` only at clean completion. An existing
        destination is preserved unless ``overwrite=True`` was passed to
        the constructor, and a crash mid-run leaves the previous file
        untouched.
        """
        rc = self.run_cfg
        cfg = self.sim_cfg
        pre = self._pre
        t_end = rc.total_time
        wall_start = time.time()

        is_restart = self._seg_idx > 0
        if is_restart:
            # Restart writes in-place (the destination is the file being
            # resumed). Atomic-replace semantics don't help here because
            # the prior data is the state we want to continue from.
            work_path = self.output_path
            f = h5py.File(work_path, "a")
            logger.info("Resuming from t=%.1f (segment %d)", self._t_current, self._seg_idx)
        else:
            work_path = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
            if work_path.exists():
                work_path.unlink()
            f = _create_output_file(work_path, cfg, rc, overwrite=True)
            _save_checkpoint(f, self._t_current, self._state, 0)
            t0_arr = np.array([self._t_current])
            y0_arr = self._state.reshape(-1, 1)
            _append_observables(f, t0_arr, y0_arr, cfg, pre)
            logger.info("Created output file: %s", work_path)

        clean_exit = False
        try:
            while self._t_current < t_end:
                seg_t_start = self._t_current
                seg_t_end = min(self._t_current + rc.segment_time, t_end)

                # Build t_eval for this segment
                n_samples = max(int((seg_t_end - seg_t_start) * rc.samples_per_unit_time), 2)
                # Exclude first point (already saved from previous segment or initial)
                t_eval = np.linspace(seg_t_start, seg_t_end, n_samples + 1)[1:]

                # Build RHS and events
                rhs = make_rhs(cfg, pre)
                events = [_make_cavity_collapse_event(cfg, pre), _make_nonfinite_event(cfg)]

                # Solve this segment
                sol = solve_ivp(
                    rhs,
                    t_span=(seg_t_start, seg_t_end),
                    y0=self._state,
                    method=cfg.method,
                    rtol=cfg.rtol,
                    atol=cfg.atol,
                    max_step=cfg.effective_max_step(),
                    t_eval=t_eval,
                    dense_output=False,
                    events=events,
                )

                if sol.status == 1:
                    # Terminal event: either cavity collapse or non-finite state.
                    # Both invalidate further evolution; persist what we have.
                    reason = "cavity collapse or non-finite state"
                    logger.warning("Terminal event at t=%.4f (%s), terminating.",
                                   sol.t[-1], reason)
                    _append_observables(f, sol.t, sol.y, cfg, pre)
                    self._state = sol.y[:, -1]
                    self._t_current = float(sol.t[-1])
                    _save_checkpoint(f, self._t_current, self._state, self._seg_idx)
                    f.attrs["status"] = "collapsed"
                    break

                if sol.status != 0:
                    logger.error("Solver failed: %s", sol.message)
                    _save_checkpoint(f, self._t_current, self._state, self._seg_idx)
                    f.attrs["status"] = "failed"
                    f.attrs["error"] = sol.message
                    break

                # Append observables
                _append_observables(f, sol.t, sol.y, cfg, pre)

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

            f.attrs["wall_time_seconds"] = time.time() - wall_start
            logger.info("Run finished: status=%s, t=%.1f, wall=%.1fs",
                        f.attrs["status"], self._t_current,
                        f.attrs["wall_time_seconds"])
            clean_exit = True

        finally:
            f.close()

        # Atomic publish — only overwrite the destination on a clean run.
        if not is_restart and clean_exit:
            if self.output_path.exists() and not self._overwrite:
                raise FileExistsError(
                    f"Destination exists after run: {self.output_path}. "
                    f"Temp file left at {work_path}. Pass overwrite=True to replace."
                )
            _atomic_replace(work_path, self.output_path)

        return self.output_path
