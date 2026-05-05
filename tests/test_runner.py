"""Tests for experiments/runner.py — HDF5 streaming, atomicity, and checkpoint round-trip.

These tests exercise the persistence layer that was the source of the
critical silent-reinterpretation bug (RT-2026-04-16-001). The key invariant
is: every SimulationConfig field round-trips through HDF5, and
``PrecomputedArrays`` rebuild from the restored config is bit-identical to
rebuild from the original config.
"""

from dataclasses import fields
from pathlib import Path

import h5py
import numpy as np
import pytest

from nothing_engine.core.bogoliubov import (
    SimulationConfig,
    PrecomputedArrays,
    build_initial_state,
)
from nothing_engine.experiments.runner import (
    ExperimentRunner,
    RunConfig,
    _CONFIG_FIELDS,
    _load_config_attrs,
    _store_config_attrs,
)


def _declared_field_names() -> set:
    return {f.name for f in fields(SimulationConfig)}


def test_config_fields_cover_every_dataclass_field_except_runtime_only():
    """The persistence allowlist must include every SimulationConfig field.

    ``t_span`` and ``t_eval`` are intentionally runtime-only (they shape a
    single invocation, not the physical simulation) and are excluded.
    Anything else missing from ``_CONFIG_FIELDS`` is RT-2026-04-16-001
    waiting to happen again.
    """
    declared = _declared_field_names()
    runtime_only = {"t_span", "t_eval", "dense_output"}
    persisted = set(_CONFIG_FIELDS)
    missing = declared - runtime_only - persisted
    assert not missing, (
        f"Fields declared on SimulationConfig but not in _CONFIG_FIELDS: {sorted(missing)}. "
        f"Either add them to the allowlist or justify moving them to runtime_only."
    )


@pytest.fixture
def periodic_sigmoid_cfg() -> SimulationConfig:
    """A config that exercises every field the old allowlist was dropping."""
    return SimulationConfig(
        n_modes=8,
        plate_mass=1.0e3,
        spring_k=1.25,
        q0=2.0,
        v0=3.5e-3,
        x_left=0.0,
        boundary="periodic",       # omitted by the pre-fix allowlist
        plate_thickness=0.02,      # omitted
        cutoff_shape="gaussian",   # omitted
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        max_step=0.0,              # auto
        t_span=(0.0, 2.0),
        audit_tolerance_factor=5e-7,  # omitted
        audit_halt=False,
    )


def test_config_attrs_round_trip_exactly(tmp_path, periodic_sigmoid_cfg):
    """Writing config attrs and reading them back yields an equal config.

    Ensures _load_config_attrs is the inverse of _store_config_attrs for
    every field in the allowlist.
    """
    path = tmp_path / "roundtrip.h5"
    run_cfg = RunConfig(total_time=2.0, segment_time=1.0,
                        samples_per_unit_time=4, checkpoint_interval=1.0)

    with h5py.File(path, "w") as f:
        g = f.create_group("config")
        _store_config_attrs(g, periodic_sigmoid_cfg, run_cfg)

    with h5py.File(path, "r") as f:
        restored_cfg, restored_run = _load_config_attrs(f["config"])

    for name in _CONFIG_FIELDS:
        original = getattr(periodic_sigmoid_cfg, name)
        restored = getattr(restored_cfg, name)
        assert original == restored, (
            f"Field {name!r} lost in round trip: {original!r} -> {restored!r}"
        )
    assert restored_run.total_time == run_cfg.total_time
    assert restored_run.segment_time == run_cfg.segment_time


def test_checkpoint_resume_preserves_precomputed_arrays(tmp_path, periodic_sigmoid_cfg):
    """The resumed runner must rebuild the same PrecomputedArrays as the original.

    This is the architectural invariant: a config that writes g_n, ns_pi,
    q_eq at write time must derive the same arrays at read time. Before
    the fix, boundary/plate_thickness/cutoff_shape were silently dropped
    and the resumed run evolved under a different physical cavity.
    """
    run_cfg = RunConfig(total_time=1.0, segment_time=0.5,
                        samples_per_unit_time=4, checkpoint_interval=0.5,
                        log_interval_segments=100)
    out = tmp_path / "resume_roundtrip.h5"

    runner = ExperimentRunner(periodic_sigmoid_cfg, run_cfg,
                              output_path=str(out), overwrite=True)
    runner.run()
    assert out.exists(), "Runner did not produce its output file"

    resumed = ExperimentRunner.from_checkpoint(str(out))

    for name in _CONFIG_FIELDS:
        assert getattr(resumed.sim_cfg, name) == getattr(periodic_sigmoid_cfg, name), (
            f"Config drift after resume on field {name!r}"
        )

    original_pre = PrecomputedArrays.from_config(periodic_sigmoid_cfg)
    resumed_pre = PrecomputedArrays.from_config(resumed.sim_cfg)
    np.testing.assert_array_equal(original_pre.ns_pi, resumed_pre.ns_pi)
    np.testing.assert_array_equal(original_pre.g_n, resumed_pre.g_n)
    np.testing.assert_array_equal(original_pre.ns_pi_sq_g, resumed_pre.ns_pi_sq_g)
    assert original_pre.q_eq == resumed_pre.q_eq
    assert original_pre.n_cutoff == resumed_pre.n_cutoff


def test_runner_refuses_to_clobber_existing_file(tmp_path, periodic_sigmoid_cfg):
    """Without overwrite=True, the runner leaves the destination untouched."""
    run_cfg = RunConfig(total_time=0.5, segment_time=0.5,
                        samples_per_unit_time=4, checkpoint_interval=1.0,
                        log_interval_segments=100)
    out = tmp_path / "existing.h5"
    out.write_bytes(b"do not touch")

    runner = ExperimentRunner(periodic_sigmoid_cfg, run_cfg,
                              output_path=str(out), overwrite=False)
    with pytest.raises(FileExistsError):
        runner.run()
    assert out.read_bytes() == b"do not touch"


def test_runner_atomic_replace_on_success(tmp_path, periodic_sigmoid_cfg):
    """A successful run leaves no .tmp sibling and replaces any prior file."""
    run_cfg = RunConfig(total_time=0.5, segment_time=0.5,
                        samples_per_unit_time=4, checkpoint_interval=1.0,
                        log_interval_segments=100)
    out = tmp_path / "atomic.h5"

    runner = ExperimentRunner(periodic_sigmoid_cfg, run_cfg,
                              output_path=str(out), overwrite=True)
    runner.run()

    assert out.exists(), "Expected atomic publish to create the final file"
    assert not (tmp_path / "atomic.h5.tmp").exists(), (
        "Temporary file should be gone after a clean run"
    )
