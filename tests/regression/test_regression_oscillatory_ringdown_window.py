"""Regression for BH-2026-06-03-005.

find_post_ringdown_start and select_fitting_window thresholded the instantaneous plate
kinetic energy E_plate = 1/2 M v^2. For an oscillating (spring-restored) plate v crosses zero
every half-period, so E_plate dips to ~0 at the first velocity turning point and the analysis
window collapsed to the first quarter-period. The fix thresholds the suffix-max envelope, which
is non-increasing and equals e_plate exactly for monotonic decay. Audit evidence at
.bugs/BH-2026-06-03-005/.
"""

import numpy as np

from nothing_engine.analysis.psd_analysis import find_post_ringdown_start
from nothing_engine.analysis.ringdown_fit import select_fitting_window


def _oscillatory_decay():
    t = np.linspace(0.0, 2000.0, 4000)
    envelope = np.exp(-t / 200.0)                      # crosses 1% near t~921
    e_plate = 0.5 * (np.cos(2.0 * np.pi * t / 50.0) ** 2) * envelope + 1e-12
    return t, e_plate


def test_post_ringdown_onset_tracks_envelope_not_first_turning_point():
    t, e_plate = _oscillatory_decay()
    # Buggy behaviour returned the first velocity turning point (~12). The envelope only
    # falls below 1% of E_plate(0) near t~921.
    t_post = find_post_ringdown_start(t, e_plate, frac=0.01)
    assert t_post > 500.0, f"post-ringdown onset collapsed to t={t_post:.2f}"
    assert t_post < 1100.0, f"post-ringdown onset implausibly late: t={t_post:.2f}"


def test_fit_window_spans_the_decay_on_oscillatory_data():
    t, e_plate = _oscillatory_decay()
    t_win, e_win = select_fitting_window(t, e_plate, threshold_frac=0.05)
    assert t_win[-1] > 400.0, f"fit window collapsed to [0, {t_win[-1]:.2f}]"


def test_monotonic_decay_window_unchanged():
    # On monotonic data the suffix-max envelope equals e_plate, so the window is identical
    # to a raw instantaneous threshold: the k=0 ringdown results must not move.
    t = np.linspace(0.0, 1000.0, 2000)
    e_plate = np.exp(-t / 100.0)
    frac = 0.01
    t_post = find_post_ringdown_start(t, e_plate, frac=frac)
    naive_idx = int(np.argmax(e_plate < frac * e_plate[0]))
    assert t_post == float(t[naive_idx])
