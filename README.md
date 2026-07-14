# Nothing Engine

Nothing Engine is a finite dimensional diagonal mode model for a 1+1D massless
scalar field coupled to one mechanical coordinate. The project accompanies a
postmortem of a proposed toroidal Casimir energy harvester.

The equilibrium harvester claim is withdrawn. The engine is maintained as a
transparent reduced model for static Casimir bookkeeping, independent
parametric oscillators, conservation tests, and future solver regression cases.

Read the [revised paper](whitepaper_geometry_of_nothing.md) and the
[banked research directions](BANKED_CASIMIR_DIRECTIONS.md) before interpreting
simulation output.

## Model scope

Each stored mode obeys

```text
f_n'' + omega_n(q)^2 f_n = 0
```

and the mechanical coordinate obeys

```text
M q'' = -k(q - q_eq) + F_field
```

The field energy subtracts the finite oscillator vacuum term mode by mode and
restores the analytic regularized static interaction:

- closed Dirichlet interval: `E_C = -pi/(24a)`, degeneracy 1;
- periodic scalar circle: `E_C = -pi/(6L)`, degeneracy 2.

The matching static force is included in the ODE and in saved observables.

### Limits

This solver omits the velocity dependent intermode couplings of a physical
moving cavity. It cannot produce quantitative moving mirror photon spectra,
vacuum friction coefficients, or toroidal device performance. The periodic
option is a compact scalar circle. The large finite cavity script has reflecting
walls and is not an open radiation boundary.

The optional high mode spectral weight is phenomenological. It is disabled by
default. A physical partially transmitting mirror needs reflection amplitudes
and an open scattering model.

## Installation

Core and development dependencies:

```bash
uv sync --extra dev
```

Analysis and desktop GUI dependencies:

```bash
uv sync --extra dev --extra analysis --extra gui
```

## Quick start

```python
from nothing_engine.core.bogoliubov import SimulationConfig, run_simulation

config = SimulationConfig(
    n_modes=32,
    plate_mass=1.0e4,
    spring_k=1.0,
    q0=1.0,
    v0=1.0e-3,
    t_span=(0.0, 10.0),
    max_step=0.01,
)

result = run_simulation(config)
print(result.energy_at(-1))
print(result.total_particle_number_at(-1))
```

## Validation and tests

```bash
uv run python -m nothing_engine.experiments.val_static_casimir
uv run python -m nothing_engine.experiments.val_dynamic_casimir
uv run python -m nothing_engine.experiments.val_conservation
uv run python -m nothing_engine.experiments.val_adiabatic
uv run python -m nothing_engine.experiments.val_residual_baseline
uv run pytest -q
```

Gate 4.1 exercises the production field energy and ODE force paths for both
boundary conditions across 8 to 512 modes. Gate 4.2 validates the diagonal
parametric oscillator equation against its own resonance prediction. It is not
a validation of the full moving boundary field theory.

## Entry points

```bash
# One run with JSON progress output
uv run python -m nothing_engine.experiments.run_single --help

# Desktop control panel
uv run nothing-engine-gui

# Reduced-model studies
uv run python -m nothing_engine.experiments.run_closed_ringdown --quick
uv run python -m nothing_engine.experiments.run_topology_v2 --quick

# Historical large finite box proxy
uv run python -m nothing_engine.experiments.run_open_ringdown --quick
```

## Output

Long runs stream HDF5 data and checkpoints. Version 0.4 records:

- plate position, velocity, kinetic energy, and spring energy;
- excitation energy and analytic static Casimir energy separately;
- combined field and total energy;
- occupation per stored mode family with periodic degeneracy included;
- the renormalized field force used by the mechanical ODE.

## Repository layout

```text
nothing_engine/
  core/          Reduced model, energy, force, and unit conversion
  experiments/   Validation gates and reproducible run entry points
  analysis/      Ringdown, spectrum, and residual diagnostics
  gui/           PySide6 desktop control panel
  config/        Default parameters and validation criteria
  tests/         Unit and regression tests
```

## License

MIT
