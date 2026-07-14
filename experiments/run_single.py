"""
Generic single-run CLI for Track B simulations.

Exposes every meaningful SimulationConfig and RunConfig parameter as a flag,
runs one ExperimentRunner, and streams machine-readable progress to stdout as
one JSON object per line:

    {"event": "progress", "t": ..., "pct": ..., "E_plate": ..., "N_total": ..., "status": "running"}
    {"event": "done", "status": "completed", "output": "data/experiments/foo.h5"}
    {"event": "error", "message": "..."}

The runner's own human-readable logging stays on stderr, so stdout is a clean
JSON stream. This is the entry point the GUI shells out to, but it is equally
usable on its own:

    python -m nothing_engine.experiments.run_single \
        --output data/experiments/run.h5 --n-modes 64 --total-time 500 --segment-time 50
"""

import sys
import json
import logging
import argparse

import h5py

from nothing_engine.core.bogoliubov import SimulationConfig
from nothing_engine.experiments.runner import ExperimentRunner, RunConfig

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser, defaulting each flag to the dataclass default."""
    sim_d = SimulationConfig()
    run_d = RunConfig()

    p = argparse.ArgumentParser(
        prog="run_single",
        description="Run one diagonal mode simulation with JSON progress output.",
    )
    p.add_argument("--output", required=True,
                   help="Output HDF5 path (parent dirs created if missing).")

    phys = p.add_argument_group("physics")
    phys.add_argument("--n-modes", type=int, default=sim_d.n_modes)
    phys.add_argument("--plate-mass", type=float, default=sim_d.plate_mass)
    phys.add_argument("--spring-k", type=float, default=sim_d.spring_k)
    phys.add_argument("--q0", type=float, default=sim_d.q0)
    phys.add_argument("--v0", type=float, default=sim_d.v0)
    phys.add_argument("--x-left", type=float, default=sim_d.x_left)
    phys.add_argument("--boundary", choices=["closed", "periodic"], default=sim_d.boundary)
    phys.add_argument("--plate-thickness", type=float, default=sim_d.plate_thickness,
                      help=("0 disables the phenomenological spectral weight. "
                            "Positive values set n_cutoff = a0 / plate_thickness."))
    phys.add_argument("--cutoff-shape", choices=["sigmoid", "gaussian"], default=sim_d.cutoff_shape)

    integ = p.add_argument_group("integrator")
    integ.add_argument("--method", default=sim_d.method,
                       help="scipy.integrate.solve_ivp method (RK45, DOP853, Radau, ...).")
    integ.add_argument("--rtol", type=float, default=sim_d.rtol)
    integ.add_argument("--atol", type=float, default=sim_d.atol)
    integ.add_argument("--max-step", type=float, default=sim_d.max_step)

    audit = p.add_argument_group("energy audit")
    audit.add_argument("--audit-tolerance-factor", type=float, default=sim_d.audit_tolerance_factor)
    audit.add_argument("--audit-halt", action=argparse.BooleanOptionalAction, default=sim_d.audit_halt,
                       help="Halt the run if energy conservation drifts past tolerance.")

    run = p.add_argument_group("run control")
    run.add_argument("--total-time", type=float, default=run_d.total_time)
    run.add_argument("--segment-time", type=float, default=run_d.segment_time)
    run.add_argument("--samples-per-unit-time", type=int, default=run_d.samples_per_unit_time)
    run.add_argument("--checkpoint-interval", type=float, default=run_d.checkpoint_interval)
    run.add_argument("--log-interval-segments", type=int, default=run_d.log_interval_segments)

    return p


def build_configs(args: argparse.Namespace) -> tuple[SimulationConfig, RunConfig]:
    """Map parsed args onto the two engine config dataclasses."""
    sim_cfg = SimulationConfig(
        n_modes=args.n_modes,
        plate_mass=args.plate_mass,
        spring_k=args.spring_k,
        q0=args.q0,
        v0=args.v0,
        x_left=args.x_left,
        boundary=args.boundary,
        plate_thickness=args.plate_thickness,
        cutoff_shape=args.cutoff_shape,
        method=args.method,
        rtol=args.rtol,
        atol=args.atol,
        max_step=args.max_step,
        audit_tolerance_factor=args.audit_tolerance_factor,
        audit_halt=args.audit_halt,
    )
    run_cfg = RunConfig(
        total_time=args.total_time,
        segment_time=args.segment_time,
        samples_per_unit_time=args.samples_per_unit_time,
        checkpoint_interval=args.checkpoint_interval,
        log_interval_segments=args.log_interval_segments,
    )
    return sim_cfg, run_cfg


def _emit(obj: dict) -> None:
    """Write one JSON event to stdout and flush, so the GUI sees it immediately."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _read_final_status(path) -> str:
    """Read the run's terminal status attr from the output file."""
    try:
        with h5py.File(path, "r") as f:
            return str(f.attrs.get("status", "unknown"))
    except OSError:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Keep stdout a clean JSON stream; the runner's logging goes to stderr.
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    sim_cfg, run_cfg = build_configs(args)
    runner = ExperimentRunner(
        sim_cfg, run_cfg,
        output_path=args.output,
        progress_callback=lambda d: _emit({"event": "progress", **d}),
    )

    try:
        path = runner.run()
    except Exception as exc:  # surface any failure to the GUI as a structured event
        logger.exception("run_single failed")
        _emit({"event": "error", "message": f"{type(exc).__name__}: {exc}"})
        return 1

    _emit({"event": "done", "status": _read_final_status(path), "output": str(path)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
