"""
Residual motion analysis for post-ringdown plate dynamics.

Computes RMS velocity in the post-ringdown window and compares
against the Gate 4.7 static-plate baseline using a KS test.

Per PRE_REGISTRATION.md §5.5-5.6:
  - Compare experiment PSD vs control (Gate 4.7 baseline)
  - Compare open vs closed PSD
  - KS test with alpha = 0.01 (Bonferroni-corrected to 0.0125)
"""

import numpy as np
from numpy.typing import NDArray
from scipy.stats import ks_2samp
from dataclasses import dataclass
from typing import Optional
import h5py


@dataclass
class ResidualStats:
    """Statistics of residual plate motion."""
    rms_velocity: float
    mean_velocity: float
    std_velocity: float
    n_points: int
    t_start: float
    t_end: float


@dataclass
class ComparisonResult:
    """Result of comparing two residual motion samples."""
    ks_statistic: float
    p_value: float
    significant: bool  # at Bonferroni-corrected alpha = 0.0125
    label_a: str
    label_b: str
    stats_a: ResidualStats
    stats_b: ResidualStats

    def summary(self) -> str:
        lines = [
            f"=== Residual Motion Comparison: {self.label_a} vs {self.label_b} ===",
            f"  {self.label_a}: RMS v = {self.stats_a.rms_velocity:.6e} "
            f"({self.stats_a.n_points} points)",
            f"  {self.label_b}: RMS v = {self.stats_b.rms_velocity:.6e} "
            f"({self.stats_b.n_points} points)",
            f"  KS statistic: {self.ks_statistic:.6f}",
            f"  p-value: {self.p_value:.6e}",
            f"  Significant (alpha=0.0125): {self.significant}",
        ]
        return "\n".join(lines)


def compute_residual_stats(t: NDArray, v: NDArray,
                           t_start: float,
                           window_duration: float = 1.0e4) -> ResidualStats:
    """Compute residual motion statistics in a time window.

    Parameters
    ----------
    t : time array
    v : velocity array
    t_start : start of post-ringdown window
    window_duration : duration of window
    """
    mask = (t >= t_start) & (t <= t_start + window_duration)
    v_win = v[mask]

    if len(v_win) == 0:
        raise ValueError(f"No data points in window [{t_start}, {t_start + window_duration}]")

    return ResidualStats(
        rms_velocity=float(np.sqrt(np.mean(v_win**2))),
        mean_velocity=float(np.mean(v_win)),
        std_velocity=float(np.std(v_win)),
        n_points=len(v_win),
        t_start=t_start,
        t_end=float(t[mask][-1]),
    )


def compare_residuals(t_a: NDArray, v_a: NDArray,
                      t_b: NDArray, v_b: NDArray,
                      t_start_a: float, t_start_b: float,
                      window_duration: float = 1.0e4,
                      label_a: str = "A",
                      label_b: str = "B",
                      alpha: float = 0.0125) -> ComparisonResult:
    """Compare residual motion between two runs using KS test.

    Parameters
    ----------
    t_a, v_a : time and velocity for run A
    t_b, v_b : time and velocity for run B
    t_start_a, t_start_b : post-ringdown start times
    window_duration : window length for both
    label_a, label_b : labels for reporting
    alpha : significance level (Bonferroni-corrected, default 0.0125)
    """
    mask_a = (t_a >= t_start_a) & (t_a <= t_start_a + window_duration)
    mask_b = (t_b >= t_start_b) & (t_b <= t_start_b + window_duration)
    va_win = v_a[mask_a]
    vb_win = v_b[mask_b]

    stats_a = compute_residual_stats(t_a, v_a, t_start_a, window_duration)
    stats_b = compute_residual_stats(t_b, v_b, t_start_b, window_duration)

    ks_stat, p_val = ks_2samp(va_win, vb_win)

    return ComparisonResult(
        ks_statistic=ks_stat,
        p_value=p_val,
        significant=p_val < alpha,
        label_a=label_a,
        label_b=label_b,
        stats_a=stats_a,
        stats_b=stats_b,
    )


def load_and_compare(path_a: str, path_b: str,
                     t_start_a: float, t_start_b: float,
                     label_a: str = "A", label_b: str = "B",
                     window_duration: float = 1.0e4) -> ComparisonResult:
    """Load two HDF5 experiment files and compare residual motion."""
    with h5py.File(path_a, "r") as f:
        t_a = f["timeseries/t"][:]
        v_a = f["timeseries/plate_v"][:]
    with h5py.File(path_b, "r") as f:
        t_b = f["timeseries/t"][:]
        v_b = f["timeseries/plate_v"][:]

    return compare_residuals(
        t_a, v_a, t_b, v_b,
        t_start_a, t_start_b,
        window_duration=window_duration,
        label_a=label_a, label_b=label_b,
    )
