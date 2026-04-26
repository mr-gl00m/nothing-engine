# Nothing Engine

A Casimir vacuum friction simulator built on Bogoliubov mode-function evolution coupled to dynamical plate motion. I created this for an experiment to see if quantum vacuum friction was topology-dependent. Releasing this for anyone who wants to use this, or will find it useful.

For the scientific context and results, see [The Geometry of Nothing](https://github.com/mr-gl00m/paper-geometry-of-nothing).

## What it does

Nothing Engine solves the quantum field theory of a scalar field inside a 1+1D cavity with one dynamical (moving) boundary. It evolves:

- **Cavity mode functions** `f_n(t)` via `f_n'' + omega_n^2(q(t)) * f_n = 0`
- **Plate dynamics** `M * q'' = -k*(q - q_eq) + F_field` with self-consistent radiation pressure feedback

This captures dynamical Casimir effect physics: a moving mirror creates real photons from the quantum vacuum, and those photons exert back-reaction on the mirror.

## Installation

```bash
pip install -e ".[dev,analysis]"
```

Or with just the core dependencies:

```bash
pip install -e .
```

## Quick start

```python
from nothing_engine.core.bogoliubov import SimulationConfig, run_simulation

config = SimulationConfig(
    n_modes=64,
    plate_mass=1e4,
    v0=1e-3,
    q0=1.0,
    t_span=(0.0, 100.0),
)

result = run_simulation(config)
print(f"Total particles created at end: {result.total_particle_number_at(-1):.6e}")
```

## Running experiments

```bash
# Validation gates (run these first)
python -m nothing_engine.experiments.val_static_casimir
python -m nothing_engine.experiments.val_dynamic_casimir
python -m nothing_engine.experiments.val_conservation
python -m nothing_engine.experiments.val_adiabatic
python -m nothing_engine.experiments.val_residual_baseline

# Physics experiments
python -m nothing_engine.experiments.run_closed_ringdown
python -m nothing_engine.experiments.run_open_ringdown
python -m nothing_engine.experiments.run_convergence
python -m nothing_engine.experiments.run_topology_comparison

# Phyllotaxis Casimir-graph (v0.2.0 — research-in-progress, see Known limitations)
python -m nothing_engine.experiments.run_phyllotaxis_graph
python -m nothing_engine.experiments.val_phyllotaxis_consistency   # Gate P.4 — passes
python -m nothing_engine.experiments.val_phyllotaxis_relax         # Gate P.3 — fails informatively
python -m nothing_engine.experiments.plot_phyllotaxis_graph
python -m nothing_engine.experiments.plot_phyllotaxis_shells
```

## Running tests

```bash
pytest
```

## Package structure

```
nothing_engine/
  core/             # Simulation engine
    bogoliubov.py   # ODE evolution (mode functions + plate)
    constants.py    # Physical constants (SI & natural units)
    mode_space.py   # Cavity mode frequencies & functions
    energy.py       # Energy density & totals
    energy_audit.py # Conservation monitoring
    plate.py        # DynamicalPlate class
    radiation_pressure.py
    flux.py
  analysis/         # Post-simulation analysis
    ringdown_fit.py # Exponential decay fitting
    psd_analysis.py # Power spectral density
    residual_motion.py
  experiments/      # Simulation runners & validation
    runner.py       # ExperimentRunner orchestration
    run_*.py        # Experiment scripts
    val_*.py        # Validation gate scripts
  config/           # Default parameters & validation criteria
  tests/            # Unit tests
```
## Known limitations

- **Phyllotaxis Casimir-graph relaxation (gate P.3) does not converge.** The pure 1+1D pair kernel `E_ij = -π/(24 r_ij)` has no repulsive core, so a 2D point cloud under overdamped relaxation collapses — final `max|F|` grows to 1e13–1e20 across all three test lattices. This is a physics result, not a code bug: the kernel is borrowed from a 1+1D parallel-plate calculation and does not regularize on 2D points. The runner and validator are kept in v0.2.0 for reproducibility of the negative result and as the starting point for the kernel study deferred to v0.3 (1/r^3, 1/r^5, 1/r^7 candidates, or a Lennard-Jones-style core).
- **Phyllotaxis consistency gate (P.4) passes.** Analytic force matches `-∇E` to machine precision (Vogel max relative error 3.55e-8 across three lattices with full pair sum). The graph-construction and force-evaluation code is sound; the open question is which kernel makes the energy landscape physically meaningful on 2D arrangements.
- **`cli/` package directory.** Intentionally empty in v0.2.0 — will be populated in v0.3 with a `nothing-engine` console command wrapping the existing `python -m` invocations. There is no planned GUI; Nothing Engine is a CLI/library tool by charter.

## License
MIT

## Support Me
If you find this useful, consider supporting me and my research:

[![Ko-fi](https://img.shields.io/badge/Ko--fi-F16061?style=for-the-badge&logo=ko-fi&logoColor=white)](https://ko-fi.com/mr_gl00m)
[![GitHub Sponsors](https://img.shields.io/badge/GitHub_Sponsors-EA4AAA?style=for-the-badge&logo=github&logoColor=white)](https://ko-fi.com/mr_gl00m)

**Crypto:**
- BTC: `bc1qnedeq3dr2dmlwgmw2mr5mtpxh45uhl395prr0d`
- ETH: `0x1bCbBa9854dA4Fc1Cb95997D5f42006055282e3c`
- SOL: `3Wm8wS93UpG2CrZsMWHSspJh7M5gQ6NXBbgLHDFXmAdQ`
