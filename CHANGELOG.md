# Changelog

All notable changes to this project will be documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-07-13

Scientific correction and postmortem release.

### Changed

- Replaced the original paper with a revised thermodynamic analysis, corrected
  phase change energy budget, software audit, and final disposition of the
  equilibrium harvester claim.
- Added a research bank for halo geometry, fluid mediated repulsive devices, and
  the nonequilibrium concentric cylinder heat engine.
- Narrowed the engine description to a diagonal parametric mode approximation.
  A quantitative moving cavity solver still requires velocity dependent
  intermode coupling.
- Disabled the phenomenological high mode spectral weight by default.
- Relabeled the large finite box and topology scripts to state their actual
  numerical scope.

### Fixed

- Restored the regularized static Casimir energy and matching force after finite
  oscillator vacuum subtraction.
- Added the twofold periodic mode degeneracy to energy, force, occupation, and
  saved observables.
- Changed vacuum cancellation to occur per mode, reducing floating point loss at
  high mode count.
- Replaced the old analytic self comparison in Gate 4.1 with checks against the
  production field energy and ODE acceleration paths.
- Changed saved `force_field` data to the renormalized force used by the ODE.
- Added a model revision marker and reject legacy checkpoint resume across the
  corrected physics boundary.

### Added

- Separate `E_excitation` and `E_casimir` HDF5 time series.
- Static energy and force tests for closed and periodic boundary models.
- Release notes documenting the retracted claims and compatibility changes.

## [0.3.0] - 2026-06-06

Adds a desktop GUI and a machine-readable single-run CLI on top of the existing
engine. No changes to the physics: the ODE evolution, force law, and energy
definitions are untouched. The GUI runs the solver out-of-process, so a crash in
a long run can't take the window down with it.

### Highlights
- **Desktop control panel** (`nothing-engine-gui`): set every parameter in a
  form, launch a run, watch plate energy and particle number stream live, then
  browse ringdown / energy / particle-spectrum / PSD results in the same window.
- **`run_single` CLI**: one experiment, every config field as a flag, progress
  emitted as one JSON object per line on stdout. It's what the GUI shells out to,
  but it stands alone for scripting and batch runs.
- **Packaging fix:** the build backend was set to an invalid value that would
  break a clean build or install. Fixed.

### Added
- **PySide6 control-panel GUI.** Launch with `nothing-engine-gui` or
  `python -m nothing_engine.gui`. A spec-driven parameter form covers every
  `SimulationConfig` / `RunConfig` field (defaults read from the dataclasses, so
  they can't drift), with atomic JSON preset save/load. The Live tab streams
  `E_plate(t)` and `N(t)` during a run; the Results tab shows the ringdown fit,
  energy components with conservation drift, the final per-mode particle
  spectrum, and the post-ringdown velocity PSD. Run/Stop does a graceful
  terminate then a hard kill after a grace period, and existing `.h5` runs load
  straight into the Results view. Dark theme, high-DPI-aware, rotating-file
  logging to `logs/gui.log`, and uncaught exceptions are captured to the log.
  Pulled in via a new `gui` optional-dependency group (`PySide6`, `pyqtgraph`):
  `uv sync --extra gui`.
- **`run_single` experiment entrypoint** (`python -m nothing_engine.experiments.run_single`).
  Exposes every physics, integrator, energy-audit, and run-control parameter as a
  CLI flag, runs one `ExperimentRunner`, and streams one JSON event per line to
  stdout (`progress` / `done` / `error`) while the human-readable log stays on
  stderr: so stdout is a clean machine-readable stream.
- **`progress_callback` hook on `ExperimentRunner`.** An optional
  `Callable[[dict], None]` that receives a per-segment snapshot (time, percent,
  segment index, energy components, total particle number, wall time, status).
  Defaults to `None`, so the scenario CLIs and library callers behave exactly as
  before. Field quantities are reported as `NaN` once the cavity has collapsed
  (`a <= 0`) so a terminal snapshot can't divide by zero.

### Changed
- Packaging metadata modernized to the SPDX form: `license = "MIT"` is now
  canonical, and the deprecated `License :: OSI Approved :: MIT License`
  classifier was removed (it duplicated the license field).

### Fixed
- **Invalid build backend.** `pyproject.toml` declared
  `build-backend = "setuptools.backends._legacy:_Backend"`, which is not a real
  backend: a clean package build would fail. Now
  `setuptools.build_meta`.
- **`EnergyAuditor.check` could fail with a confusing error if `set_reference`
  was skipped.** The guard now also checks the tolerance was set, so the clear
  "`set_reference() not called`" message fires instead of a later `TypeError`.

### Internal
- Type-safety pass: added `pyrightconfig.json`, corrected `NDArray = None`
  defaults to `NDArray | None = None` across `energy.py` and `mode_space.py`, and
  confined `# pyright:` suppressions to the h5py/scipy/pyqtgraph stub-driven I/O
  glue: the physics core keeps full type checking.
- New tests: `test_gui_smoke.py` (GUI bootstrap), `test_run_single.py` (JSON
  stream + CLI), `test_runner_callback.py` (progress hook).
- `logs/` added to `.gitignore`.

[Unreleased]: https://github.com/mr-gl00m/nothing-engine/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/mr-gl00m/nothing-engine/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/mr-gl00m/nothing-engine/compare/v0.2.1...v0.3.0

## [0.1.1] - 2026-06-03

Correctness patch. No new features and no changes to the ODE evolution, force law, or
energy definitions: those were audited and left untouched. Every fix is in the
checkpoint/restart path, the standalone validation-gate scripts, the topology summary,
or the post-ringdown analysis windowing. Each fix ships with a regression test.

### Highlights
- Validation Gate 4.7 (static-plate residual baseline) now actually passes: it was
  reporting a spurious ~1e-5 "vacuum particle" floor caused by a form-factor mismatch in
  the gate script, not by the physics.
- Resuming a long run from a checkpoint no longer silently changes physics or truncates
  already-streamed data.
- Post-ringdown analysis no longer collapses its window to the first oscillation turning
  point on spring-restored (oscillating) runs.

### Fixed
- **Validation gates counted particles against the wrong frequencies.** The vacuum state
  is built with the plate-thickness form factor (`g_n`), but `val_residual_baseline` (Gate
  4.7), `val_dynamic_casimir`, and two unit tests called `particle_number` without it,
  measuring against ideal-Dirichlet frequencies and reporting a spurious ~1e-5 particle
  floor (N=64): enough to falsely fail Gate 4.7. Saved HDF5 data was unaffected; the
  production streaming pipeline already passed the form factor.
- **Checkpoint resume silently reverted physics.** `ExperimentRunner.from_checkpoint`
  rebuilt `SimulationConfig` without `boundary`, `plate_thickness`, or `cutoff_shape`, so a
  resumed `periodic` topology run reverted to the `closed` default (`omega_n = n*pi/a`
  instead of `2*n*pi/a`). All three fields now round-trip, and output files now persist
  `cutoff_shape`.
- **Checkpoint resume could truncate streamed data.** A crash before the first checkpoint
  interval left only the initial (segment 0) checkpoint; resuming then reopened the output
  file in write mode and discarded all previously streamed timeseries. Resume now always
  appends.
- **Post-ringdown window collapsed on oscillating runs.** `find_post_ringdown_start` and
  `select_fitting_window` thresholded the instantaneous plate kinetic energy
  `E_plate = ½Mv²`, which dips to ~0 at every velocity turning point: so a spring-restored
  plate truncated the analysis window to the first quarter-period. Both now threshold the
  decay envelope (suffix-max). Monotonic (k=0) closed/open ringdown windows are unchanged.
- **Topology fit table always showed `fit_failed`.** `run_topology_comparison`'s summary
  iterated the `RingdownResults` dataclass as a list of dicts, raising a `TypeError` that a
  broad `except` swallowed. It now uses the dataclass API, and the `except` is narrowed so
  future breakage surfaces.
- **Stale spring-equilibrium test.** A test asserted `q_eq = q0 + F_trunc/k`; under
  renormalization the field force is zero at the vacuum state, so `q_eq = q0` (the engine
  was already correct). Test and docstring corrected.

### Internal
- Added a regression suite under `nothing_engine/tests/regression/` covering all six fixes.
  Test count: 41 → 50, all green.

[0.1.1]: https://github.com/mr-gl00m/nothing-engine/compare/v0.1.0...v0.1.1
