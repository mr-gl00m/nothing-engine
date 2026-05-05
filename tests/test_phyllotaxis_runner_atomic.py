"""Regression tests for RT-2026-05-05-001/002/003.

The phyllotaxis HDF5 writers replicated the destructive-write anti-pattern
that ``runner.py`` originally had: ``h5py.File(out, "w")`` directly on the
destination, no overwrite gate. These tests pin the atomic-write +
no-clobber-default invariants for the shared helper and for the call site
in ``run_phyllotaxis_graph.save_hdf5``.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from nothing_engine.experiments._atomic_h5 import atomic_h5_write
from nothing_engine.experiments import run_phyllotaxis_graph


def _write_sentinel(path: Path, marker: str = "ORIGINAL") -> None:
    with h5py.File(path, "w") as f:
        f.attrs["marker"] = marker
        f.create_dataset("payload", data=np.arange(8, dtype=np.float64))


def _read_marker(path: Path) -> str:
    with h5py.File(path, "r") as f:
        raw = f.attrs["marker"]
        return raw.decode() if isinstance(raw, bytes) else str(raw)


def test_atomic_h5_write_publishes_on_success(tmp_path: Path) -> None:
    dst = tmp_path / "out.h5"
    with atomic_h5_write(dst) as f:
        f.attrs["marker"] = "NEW"
        f.create_dataset("x", data=np.zeros(4))
    assert dst.exists()
    assert _read_marker(dst) == "NEW"
    assert not (tmp_path / "out.h5.tmp").exists()


def test_atomic_h5_write_refuses_to_clobber_by_default(tmp_path: Path) -> None:
    dst = tmp_path / "out.h5"
    _write_sentinel(dst)
    with pytest.raises(FileExistsError):
        with atomic_h5_write(dst) as f:
            f.attrs["marker"] = "SHOULD_NOT_LAND"
    assert _read_marker(dst) == "ORIGINAL"


def test_atomic_h5_write_overwrite_true_replaces(tmp_path: Path) -> None:
    dst = tmp_path / "out.h5"
    _write_sentinel(dst)
    with atomic_h5_write(dst, overwrite=True) as f:
        f.attrs["marker"] = "REPLACED"
    assert _read_marker(dst) == "REPLACED"


def test_atomic_h5_write_preserves_existing_on_exception(tmp_path: Path) -> None:
    dst = tmp_path / "out.h5"
    _write_sentinel(dst)
    with pytest.raises(RuntimeError, match="boom"):
        with atomic_h5_write(dst, overwrite=True) as f:
            f.attrs["marker"] = "PARTIAL"
            f.create_dataset("partial", data=np.zeros(4))
            raise RuntimeError("boom")
    assert _read_marker(dst) == "ORIGINAL"
    assert not (tmp_path / "out.h5.tmp").exists()


def test_save_hdf5_preserves_existing_on_midwrite_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RT-001 attack narrative: SIGINT/crash mid-loop must not destroy prior file."""
    dst = tmp_path / "phyllotaxis_graph.h5"
    _write_sentinel(dst, marker="ORIGINAL_GRAPH")

    rows = [
        {"name": "vogel", "n_points": 4, "elapsed": 0.1, "mean_nn": 1.0},
        {"name": "hex", "n_points": 4, "elapsed": 0.1, "mean_nn": 1.0},
    ]
    lattices = {
        "vogel_N4": np.zeros((4, 2)),
        "hex_N4": np.zeros((4, 2)),
    }

    real_create_group = h5py.File.create_group
    call_count = {"n": 0}

    def flaky_create_group(self, name, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise RuntimeError("simulated mid-write crash")
        return real_create_group(self, name, *args, **kwargs)

    monkeypatch.setattr(h5py.File, "create_group", flaky_create_group)

    with pytest.raises(RuntimeError, match="simulated mid-write crash"):
        run_phyllotaxis_graph.save_hdf5(
            dst, density=1.0, r_cut_scale=4.0, rows=rows, lattices=lattices,
            overwrite=True,
        )

    assert _read_marker(dst) == "ORIGINAL_GRAPH"
    assert not (tmp_path / "phyllotaxis_graph.h5.tmp").exists()


def test_save_hdf5_refuses_to_clobber_by_default(tmp_path: Path) -> None:
    """RT-003 attack narrative: re-running must not silently overwrite."""
    dst = tmp_path / "phyllotaxis_graph.h5"
    _write_sentinel(dst, marker="ORIGINAL_GRAPH")

    rows = [{"name": "vogel", "n_points": 4, "elapsed": 0.1, "mean_nn": 1.0}]
    lattices = {"vogel_N4": np.zeros((4, 2))}

    with pytest.raises(FileExistsError):
        run_phyllotaxis_graph.save_hdf5(
            dst, density=1.0, r_cut_scale=4.0, rows=rows, lattices=lattices,
        )

    assert _read_marker(dst) == "ORIGINAL_GRAPH"
