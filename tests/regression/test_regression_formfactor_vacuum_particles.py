"""Regression for BH-2026-06-03-002.

particle_number must be evaluated with the SAME form factor the vacuum state was built
with (cfg.g_n / cfg.ns_pi). Gate 4.7 (val_residual_baseline), val_dynamic_casimir, and two
unit tests previously omitted those arguments, so they measured the form-factor vacuum
against ideal-Dirichlet frequencies and reported a spurious ~1e-5 "vacuum particle" floor
(N=64), tripping Gate 4.7's own 1e-9 threshold. Audit evidence: including the original
footgun demonstration: is at .bugs/BH-2026-06-03-002/.
"""

import numpy as np

from nothing_engine.core.bogoliubov import (
    SimulationConfig, build_initial_state, unpack_state, run_simulation,
)
from nothing_engine.core import mode_space


def test_formfactor_consistent_vacuum_is_zero():
    # N=64 with the phenomenological spectral weight explicitly active.
    N = 64
    cfg = SimulationConfig(
        n_modes=N, q0=1.0, v0=0.0, plate_thickness=0.01,
    )
    assert not np.allclose(cfg.g_n, 1.0), "form factor must be genuinely active for this test"

    y0 = build_initial_state(cfg)
    ms, q, v = unpack_state(y0, N)
    a0 = q - cfg.x_left

    beta = mode_space.particle_number(ms, N, a0, cfg.g_n, cfg.ns_pi)
    assert np.max(np.abs(beta)) < 1e-12, (
        f"form-factor-consistent vacuum should be ~0, got {np.max(np.abs(beta)):.3e}"
    )


def test_result_particle_number_at_uses_form_factor():
    # The canonical SimulationResult path that the validation gates must mirror.
    N = 48
    cfg = SimulationConfig(
        n_modes=N, q0=1.0, v0=0.0,
        plate_thickness=0.01,
        t_span=(0.0, 1.0), max_step=0.05,
    )
    result = run_simulation(cfg, prescribed_motion=(lambda t: cfg.q0, lambda t: 0.0))
    pn0 = result.particle_number_at(0)
    assert np.max(np.abs(pn0)) < 1e-12, (
        f"vacuum particle_number_at(0) should be ~0, got {np.max(np.abs(pn0)):.3e}"
    )
