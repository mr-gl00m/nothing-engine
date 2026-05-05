# Changelog

All notable changes to Nothing Engine are recorded here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is [SemVer](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-05-05

Honest-snapshot release. Adds the phyllotaxis Casimir-graph extension and discloses one of its validation gates as a documented physics failure. Also lands the atomic-write fixes from the 2026-05-05 red-team pass (see Security).

### Added
- `experiments/run_phyllotaxis_graph.py` — Casimir-graph runner over Vogel and shell phyllotaxis lattices.
- `experiments/val_phyllotaxis_consistency.py` — gate **P.4**, analytic force vs `-∇E` consistency check. Passes on three lattices with full pair sum (Vogel max relative error 3.55e-8).
- `experiments/val_phyllotaxis_relax.py` — gate **P.3**, overdamped relaxation under the pure 1+1D pair kernel. Ships failing; see Known Issues.
- `experiments/plot_phyllotaxis_graph.py`, `experiments/plot_phyllotaxis_shells.py` — visualization helpers, generated artifacts under `data/experiments/`.
- `experiments/run_topology_v2.py`, `experiments/analyze_convergence.py`, `experiments/run_analysis.py` — secondary tooling for the phyllotaxis study.
- `CHANGELOG.md` (this file).
- README "Known limitations" section disclosing the P.3 result.
- `.gitignore` covering Python build artifacts, virtual envs, generated experiment output, and the `exp_halo_ring_casimir/` research working directory.

### Changed
- `pyproject.toml` — version bumped `0.1.0` → `0.2.0`; `Repository` URL replaced with the real `mr-gl00m/nothing-engine` URL; `authors` set to `Nathan 'Cid' Seals`.
- `LICENSE` — copyright holder updated to match `pyproject.toml`.
- `README.md` — phyllotaxis experiment commands added under "Running experiments"; stale "(This will be released very soon!)" parenthetical on the paper link removed (the paper repo is public).

### Removed
- `nothing_engine/gui/` — empty package directory removed. Nothing Engine is a CLI/library tool by charter; there is no planned GUI.

### Security
- New `nothing_engine/experiments/_atomic_h5.py` shared helper exposes `atomic_h5_write(dst, overwrite=False)` — opens `<dst>.tmp`, publishes via `os.replace` on clean exit, removes the partial temp on exception. Refuses to clobber an existing destination unless `overwrite=True`. Matches the publish semantics of `runner._atomic_replace` / `runner._create_output_file`.
- `experiments/run_phyllotaxis_graph.save_hdf5` and `experiments/val_phyllotaxis_relax.run` go through the helper. A SIGINT or crash mid-write no longer destroys the prior valid output. (RT-2026-05-05-001, RT-2026-05-05-002)
- `--force` argparse flag added to all four phyllotaxis scripts (`run_phyllotaxis_graph`, `val_phyllotaxis_relax`, `plot_phyllotaxis_graph`, `plot_phyllotaxis_shells`). Default behavior now refuses to clobber an existing output file. (RT-2026-05-05-003)
- `.gitignore` snapshot-archive pattern broadened from the literal `nothing_engine.rar` to the glob `nothing_engine*.rar`, so versioned snapshots like `nothing_engine_v0.2.0.rar` are caught by `git add .`. (RT-2026-05-05-004)
- New regression test module `tests/test_phyllotaxis_runner_atomic.py` pins the helper invariants and the real `save_hdf5` entry point against the mid-write and clobber narratives. Suite total: 54 passing, 1 skipped (was 48 / 1).

### Known Issues
- **Gate P.3 fails by design of the test, not a code defect.** Overdamped relaxation under `E_ij = -π/(24 r_ij)` collapses on all three test lattices (final `max|F|` = 1e13 to 1e20). The pure 1+1D parallel-plate pair kernel has no repulsive core, so on a 2D point arrangement no local energy minimum exists — collapse is the correct answer for that kernel. The runner and validator are retained in v0.2.0 for reproducibility of the negative result and as the launching point for the v0.3 kernel study (1/r^3, 1/r^5, 1/r^7 candidates, or a Lennard-Jones-style core). Gates P.1 and P.2 are deferred behind that kernel decision.
- **`nothing_engine/cli/`** is intentionally empty in v0.2.0. The `nothing-engine` console command lands in v0.3.
- The companion paper [_The Geometry of Nothing_](https://github.com/mr-gl00m/paper-geometry-of-nothing) carries a 2026-04-16 corrigendum retracting Findings 1 and 3 (per-mode vacuum-subtraction bug, RT-009). Finding 2 survived. The engine code in this release is independent of that retraction; the affected analyses lived in the paper, not in `nothing_engine/`.

## [0.1.0] — 2026-04-03

Initial release.

### Added
- Core 1+1D Bogoliubov engine: `core/bogoliubov.py` (mode-function ODE evolution), `core/mode_space.py` (cavity mode frequencies and functions), `core/energy.py` and `core/energy_audit.py` (energy totals and conservation monitoring), `core/plate.py` and `core/radiation_pressure.py` and `core/flux.py` (plate dynamics with self-consistent radiation pressure), `core/constants.py` (SI and natural-unit constants).
- Validation gates: `val_static_casimir`, `val_dynamic_casimir`, `val_conservation`, `val_adiabatic`, `val_residual_baseline`.
- Experiment runners: `run_closed_ringdown`, `run_open_ringdown`, `run_convergence`, `run_topology_comparison`.
- Analysis module: `analysis/ringdown_fit.py`, `analysis/psd_analysis.py`, `analysis/residual_motion.py`.
- Unit tests: `test_bogoliubov`, `test_energy_audit`, `test_mode_space`, `test_radiation_pressure`, `test_runner`.
- Packaging: `pyproject.toml` with `dev` and `analysis` extras, MIT `LICENSE`, README with installation and quick-start.
