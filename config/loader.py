"""Load validation criteria and default physics parameters from YAML.

These YAML files were load-bearing in spirit only: ``pyyaml`` was listed
in requirements but nothing imported it, and every validator hardcoded
its own thresholds. That let the gates pass under a softer tolerance
than the declared source-of-truth advertised (RT-2026-04-16-002).

This module is now the single entry point. Validation scripts call
:func:`get_gate_criterion` with their gate name and a key, and the
declared value is what actually gates the pass/fail.
"""

from __future__ import annotations

from dataclasses import fields
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from nothing_engine.core.bogoliubov import SimulationConfig


_CONFIG_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def _load_yaml(name: str) -> dict:
    """Load a YAML file from the config directory. Cached per process."""
    path = _CONFIG_DIR / name
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at top level of {path}, got {type(data)!r}")
    return data


def load_validation_criteria() -> dict:
    """Return the validation_criteria.yaml contents.

    The returned dict is a shared, cached reference — treat it as
    read-only.
    """
    return _load_yaml("validation_criteria.yaml")


def load_default_params() -> dict:
    """Return the default_params.yaml contents (cached, read-only)."""
    return _load_yaml("default_params.yaml")


def get_gate_criterion(gate: str, key: str, default: Any = None) -> Any:
    """Fetch a single criterion field for a named gate.

    Parameters
    ----------
    gate : str
        Gate name, e.g. ``"gate_4_3_energy_conservation"``.
    key : str
        Field under that gate, e.g. ``"tolerance_relative"``.
    default :
        Returned if the gate or key is missing. When ``None`` (the
        default) a missing entry raises KeyError — callers that want
        silent fallback must pass one explicitly.
    """
    criteria = load_validation_criteria()
    if gate not in criteria:
        if default is not None:
            return default
        raise KeyError(f"Gate {gate!r} not declared in validation_criteria.yaml")
    block = criteria[gate]
    if key not in block:
        if default is not None:
            return default
        raise KeyError(f"Key {key!r} not declared under {gate!r}")
    return block[key]


def default_simulation_config(**overrides) -> SimulationConfig:
    """Build a SimulationConfig from default_params.yaml, with overrides.

    The YAML keys are mapped to SimulationConfig fields:
      - ``physics.cavity_width_a0`` -> ``q0``
      - ``physics.plate_mass_M``    -> ``plate_mass``
      - ``physics.spring_k``        -> ``spring_k``
      - ``physics.initial_velocity_v0`` -> ``v0``
      - ``physics.x_left``          -> ``x_left``
      - ``modes.N_modes``           -> ``n_modes``
      - ``integrator.method|rtol|atol|max_step`` -> same-named fields
      - ``energy_audit.tolerance_factor`` -> ``audit_tolerance_factor``
      - ``energy_audit.halt_on_violation`` -> ``audit_halt``

    Any keyword in ``overrides`` replaces the YAML value.
    """
    params = load_default_params()
    physics = params.get("physics", {})
    modes = params.get("modes", {})
    integ = params.get("integrator", {})
    audit = params.get("energy_audit", {})

    mapped: dict = {
        "q0": physics.get("cavity_width_a0"),
        "plate_mass": physics.get("plate_mass_M"),
        "spring_k": physics.get("spring_k"),
        "v0": physics.get("initial_velocity_v0"),
        "x_left": physics.get("x_left"),
        "n_modes": modes.get("N_modes"),
        "method": integ.get("method"),
        "rtol": integ.get("rtol"),
        "atol": integ.get("atol"),
        "max_step": integ.get("max_step"),
        "audit_tolerance_factor": audit.get("tolerance_factor"),
        "audit_halt": audit.get("halt_on_violation"),
    }
    # Drop any keys the YAML did not supply — let the dataclass defaults win.
    mapped = {k: v for k, v in mapped.items() if v is not None}
    mapped.update(overrides)

    declared = {f.name for f in fields(SimulationConfig)}
    unknown = set(mapped) - declared
    if unknown:
        raise ValueError(
            f"Unknown SimulationConfig fields in default_params.yaml or overrides: {sorted(unknown)}"
        )
    return SimulationConfig(**mapped)
