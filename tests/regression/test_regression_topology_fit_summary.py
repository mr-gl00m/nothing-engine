"""Regression for BH-2026-06-03-004.

run_topology_comparison.summarize_results consumed the RingdownResults dataclass returned by
fit_ringdown as if it were an iterable of dicts (`min(fits, key=lambda x: x.get("aic", inf))`),
which raises TypeError. The surrounding `except Exception` swallowed it, so the paper's central
topology-comparison experiment always printed "fit_failed" for the model fit. This asserts the
corrected RingdownResults API the fix relies on, and guards against reverting to an iterate/index
pattern. Audit evidence at .bugs/BH-2026-06-03-004/.
"""

import numpy as np
import pytest

from nothing_engine.analysis.ringdown_fit import fit_ringdown, RingdownResults, FitResult


def test_ringdown_results_api_is_usable_for_summary():
    t = np.linspace(0.0, 200.0, 400)
    e_plate = 1e-3 * np.exp(-t / 40.0) + 1e-6
    fits = fit_ringdown(t, e_plate)

    assert isinstance(fits, RingdownResults)
    assert fits.best_model in {"Exponential", "Stretched Exponential", "Power Law"}

    best_fr = {
        "Exponential": fits.exponential,
        "Stretched Exponential": fits.stretched_exp,
        "Power Law": fits.power_law,
    }[fits.best_model]
    assert isinstance(best_fr, FitResult)
    assert np.isfinite(best_fr.aic)


def test_ringdown_results_not_iterable():
    # Documents why min(fits, ...) / fits[i] fail — guards against reintroducing them.
    t = np.linspace(0.0, 10.0, 50)
    fits = fit_ringdown(t, np.exp(-t))
    with pytest.raises(TypeError):
        iter(fits)
