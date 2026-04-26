"""
Power Spectral Density analysis of post-ringdown plate motion.

Computes Welch PSD of plate velocity in the post-ringdown window and
compares against the FDT prediction at T=0.

Per PRE_REGISTRATION.md §5.5:
  - Post-ringdown = after fitted exponential decays to 1% of initial E_plate
  - Observable = PSD of plate velocity over 10^4 cavity crossing times
  - Comparisons: open vs closed, both vs FDT at T=0

Usage:
    from analysis.psd_analysis import compute_psd, load_velocity_data
    t, v = load_velocity_data("data/experiments/closed_ringdown.h5")
    freqs, psd = compute_psd(t, v, post_ringdown_t=5e4)
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch
from dataclasses import dataclass
from typing import Optional
import h5py


def load_velocity_data(hdf5_path: str) -> tuple[NDArray, NDArray]:
    """Load time and plate velocity from experiment HDF5."""
    with h5py.File(hdf5_path, "r") as f:
        t = f["timeseries/t"][:]
        v = f["timeseries/plate_v"][:]
    return t, v


def find_post_ringdown_start(t: NDArray, e_plate: NDArray,
                             frac: float = 0.01) -> float:
    """Find time at which E_plate first drops below frac * E_plate(0).

    Parameters
    ----------
    t : time array
    e_plate : plate kinetic energy array
    frac : fraction of initial energy (default 1%)

    Returns
    -------
    float — time of post-ringdown onset
    """
    threshold = frac * e_plate[0]
    below = np.where(e_plate < threshold)[0]
    if len(below) == 0:
        raise ValueError(
            f"E_plate never dropped below {frac*100:.0f}% of initial. "
            f"Min ratio: {e_plate.min()/e_plate[0]:.4f}"
        )
    return float(t[below[0]])


def select_post_ringdown_window(t: NDArray, v: NDArray,
                                t_start: float,
                                window_duration: float = 1.0e4
                                ) -> tuple[NDArray, NDArray]:
    """Extract the post-ringdown velocity window.

    Parameters
    ----------
    t : full time array
    v : full velocity array
    t_start : start of post-ringdown region
    window_duration : duration of window (default 10^4)

    Returns
    -------
    t_win, v_win — windowed arrays
    """
    mask = (t >= t_start) & (t <= t_start + window_duration)
    if np.sum(mask) < 64:
        raise ValueError(
            f"Too few points ({np.sum(mask)}) in post-ringdown window "
            f"[{t_start:.1f}, {t_start + window_duration:.1f}]"
        )
    return t[mask], v[mask]


@dataclass
class PSDResult:
    """Result of PSD computation."""
    freqs: NDArray       # frequency array (Hz in natural units)
    psd: NDArray         # power spectral density S_vv(f)
    t_start: float       # start of post-ringdown window
    t_end: float         # end of window
    n_points: int        # number of points in window
    dt: float            # sampling interval
    nperseg: int         # Welch segment length


def compute_psd(t: NDArray, v: NDArray,
                post_ringdown_t: Optional[float] = None,
                e_plate: Optional[NDArray] = None,
                window_duration: float = 1.0e4,
                nperseg: Optional[int] = None) -> PSDResult:
    """Compute Welch PSD of plate velocity in post-ringdown window.

    Either provide post_ringdown_t directly, or provide e_plate to
    auto-detect the onset.

    Parameters
    ----------
    t : time array
    v : velocity array
    post_ringdown_t : start time (if known)
    e_plate : plate energy (for auto-detection of post-ringdown)
    window_duration : length of PSD window
    nperseg : Welch segment length (default: N/8 for good freq resolution)
    """
    if post_ringdown_t is None:
        if e_plate is None:
            raise ValueError("Provide either post_ringdown_t or e_plate")
        post_ringdown_t = find_post_ringdown_start(t, e_plate)

    t_win, v_win = select_post_ringdown_window(
        t, v, post_ringdown_t, window_duration
    )

    dt = float(np.median(np.diff(t_win)))
    fs = 1.0 / dt

    if nperseg is None:
        nperseg = max(len(v_win) // 8, 64)

    freqs, psd = welch(v_win, fs=fs, nperseg=nperseg,
                       window="hann", scaling="density")

    return PSDResult(
        freqs=freqs,
        psd=psd,
        t_start=post_ringdown_t,
        t_end=post_ringdown_t + window_duration,
        n_points=len(v_win),
        dt=dt,
        nperseg=nperseg,
    )


def fdt_prediction_T0(freqs: NDArray, mass: float,
                      gamma: float) -> NDArray:
    """FDT prediction for PSD at T=0 (zero-point fluctuations).

    S_vv(omega) = (hbar / M) * Im[chi(omega)] * coth(hbar*omega / 2kT)

    At T=0, coth -> sgn(omega), and for a damped harmonic oscillator:
        Im[chi(omega)] = gamma * omega / [(omega_0^2 - omega^2)^2 + gamma^2 * omega^2]

    For a free plate (k=0, omega_0=0):
        Im[chi(omega)] = gamma / [omega^2 + gamma^2] * (1/omega) ... simplified

    In natural units (hbar=1):
        S_vv(f) = (1/M) * gamma / [(2*pi*f)^2 + gamma^2]  (Lorentzian)

    This is a rough approximation — the actual chi depends on the
    full plate-field coupled response. See EXECUTION_PLAN.md gap #5.

    Parameters
    ----------
    freqs : frequency array (in natural units, not angular)
    mass : plate mass
    gamma : fitted damping rate

    Returns
    -------
    NDArray — predicted PSD
    """
    omega = 2.0 * np.pi * freqs
    # Avoid division by zero at f=0
    omega = np.where(omega == 0, 1e-30, omega)
    return (1.0 / mass) * gamma / (omega**2 + gamma**2)
