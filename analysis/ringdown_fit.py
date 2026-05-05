"""
Ringdown curve fitting for Track B experiments.

Fits plate kinetic energy E_plate(t) to three models per PRE_REGISTRATION.md:

1. Exponential:           E0 * exp(-gamma * t) + E_residual
2. Stretched exponential: E0 * exp(-(t/tau)^beta)
3. Power-law tail:        E0 * (1 + t/tau)^(-alpha)

Reports AIC comparison and 95% CI on damping rate gamma.

Usage:
    from analysis.ringdown_fit import fit_ringdown, load_ringdown_data
    t, e_plate = load_ringdown_data("data/experiments/closed_ringdown.h5")
    results = fit_ringdown(t, e_plate)
    print(results.summary())
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from dataclasses import dataclass, field
from typing import Optional
import h5py


# --- Models ---

def exponential_model(t: NDArray, E0: float, gamma: float,
                      E_residual: float) -> NDArray:
    """E(t) = E0 * exp(-gamma * t) + E_residual."""
    return E0 * np.exp(-gamma * t) + E_residual


def stretched_exponential_model(t: NDArray, E0: float, tau: float,
                                beta: float) -> NDArray:
    """E(t) = E0 * exp(-(t/tau)^beta)."""
    return E0 * np.exp(-np.power(t / tau, beta))


def power_law_model(t: NDArray, E0: float, tau: float,
                    alpha: float) -> NDArray:
    """E(t) = E0 * (1 + t/tau)^(-alpha)."""
    return E0 * np.power(1.0 + t / tau, -alpha)


# --- AIC ---

def aic(n: int, k: int, rss: float) -> float:
    """Akaike Information Criterion.

    AIC = n * ln(RSS/n) + 2k

    Parameters
    ----------
    n : int  — number of data points
    k : int  — number of free parameters
    rss : float — residual sum of squares
    """
    return n * np.log(rss / n) + 2 * k


# --- Data loading ---

def load_ringdown_data(hdf5_path: str) -> tuple[NDArray, NDArray]:
    """Load time and E_plate from an experiment HDF5 file."""
    with h5py.File(hdf5_path, "r") as f:
        t = f["timeseries/t"][:]
        e_plate = f["timeseries/E_plate"][:]
    return t, e_plate


# --- Fitting window ---

def select_fitting_window(t: NDArray, e_plate: NDArray,
                          threshold_frac: float = 0.05
                          ) -> tuple[NDArray, NDArray]:
    """Select fitting window per pre-registration: t=0 until E_plate < 5% of E_plate(0).

    Returns
    -------
    t_win, e_win : masked arrays within the fitting window
    """
    e0 = e_plate[0]
    threshold = threshold_frac * e0
    mask = e_plate >= threshold
    # Include at least up to the first crossing
    if not np.any(~mask):
        # Never dropped below threshold — use all data
        return t, e_plate
    first_below = np.argmax(~mask)
    return t[:first_below], e_plate[:first_below]


# --- Result container ---

@dataclass
class FitResult:
    """Result of fitting a single model."""
    name: str
    params: dict
    param_errors: dict  # 1-sigma uncertainties from covariance
    rss: float
    aic: float
    n_points: int
    n_params: int
    converged: bool
    prediction: Optional[NDArray] = None

    def gamma_with_ci(self, confidence: float = 0.95) -> Optional[tuple[float, float, float]]:
        """Return (gamma, lower_95, upper_95) for exponential model.

        Returns None for non-exponential models.
        """
        if "gamma" not in self.params:
            return None
        from scipy.stats import norm
        z = norm.ppf(1.0 - (1.0 - confidence) / 2.0)
        g = self.params["gamma"]
        se = self.param_errors.get("gamma", 0.0)
        return (g, g - z * se, g + z * se)


@dataclass
class RingdownResults:
    """Results from fitting all three models."""
    exponential: Optional[FitResult] = None
    stretched_exp: Optional[FitResult] = None
    power_law: Optional[FitResult] = None
    best_model: str = ""
    fitting_window: tuple = (0.0, 0.0)

    def summary(self) -> str:
        lines = ["=== Ringdown Fit Results ===",
                 f"Fitting window: t = {self.fitting_window[0]:.1f} to {self.fitting_window[1]:.1f}"]

        for fr in [self.exponential, self.stretched_exp, self.power_law]:
            if fr is None:
                continue
            lines.append(f"\n--- {fr.name} ---")
            lines.append(f"  Converged: {fr.converged}")
            lines.append(f"  Params: {fr.params}")
            lines.append(f"  Errors: {fr.param_errors}")
            lines.append(f"  RSS: {fr.rss:.6e}")
            lines.append(f"  AIC: {fr.aic:.2f}")

            ci = fr.gamma_with_ci()
            if ci:
                lines.append(f"  gamma = {ci[0]:.6e} [{ci[1]:.6e}, {ci[2]:.6e}] (95% CI)")

        lines.append(f"\nBest model (lowest AIC): {self.best_model}")
        return "\n".join(lines)


# --- Main fitting function ---

def fit_ringdown(t: NDArray, e_plate: NDArray,
                 threshold_frac: float = 0.05) -> RingdownResults:
    """Fit ringdown data to all three pre-registered models.

    Parameters
    ----------
    t : NDArray — time array
    e_plate : NDArray — plate kinetic energy
    threshold_frac : float — fitting window cutoff (default 5%)

    Returns
    -------
    RingdownResults
    """
    t_win, e_win = select_fitting_window(t, e_plate, threshold_frac)
    # Shift time to start at 0 for fitting
    t_fit = t_win - t_win[0]
    n = len(t_fit)
    e0_guess = float(e_win[0])

    results = RingdownResults(
        fitting_window=(float(t_win[0]), float(t_win[-1]))
    )

    # 1. Exponential: E0 * exp(-gamma*t) + E_residual
    try:
        p0 = [e0_guess, 1e-5, e0_guess * 0.001]
        bounds = ([0, 0, 0], [np.inf, np.inf, e0_guess])
        popt, pcov = curve_fit(exponential_model, t_fit, e_win,
                               p0=p0, bounds=bounds, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        resid = e_win - exponential_model(t_fit, *popt)
        rss = float(np.sum(resid**2))

        results.exponential = FitResult(
            name="Exponential",
            params={"E0": popt[0], "gamma": popt[1], "E_residual": popt[2]},
            param_errors={"E0": perr[0], "gamma": perr[1], "E_residual": perr[2]},
            rss=rss, aic=aic(n, 3, rss), n_points=n, n_params=3,
            converged=True,
            prediction=exponential_model(t_fit, *popt),
        )
    except (RuntimeError, ValueError) as e:
        results.exponential = FitResult(
            name="Exponential", params={}, param_errors={},
            rss=np.inf, aic=np.inf, n_points=n, n_params=3,
            converged=False,
        )

    # 2. Stretched exponential: E0 * exp(-(t/tau)^beta)
    try:
        p0 = [e0_guess, 1e5, 1.0]
        bounds = ([0, 1e-10, 0.01], [np.inf, np.inf, 10.0])
        popt, pcov = curve_fit(stretched_exponential_model, t_fit, e_win,
                               p0=p0, bounds=bounds, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        resid = e_win - stretched_exponential_model(t_fit, *popt)
        rss = float(np.sum(resid**2))

        results.stretched_exp = FitResult(
            name="Stretched Exponential",
            params={"E0": popt[0], "tau": popt[1], "beta": popt[2]},
            param_errors={"E0": perr[0], "tau": perr[1], "beta": perr[2]},
            rss=rss, aic=aic(n, 3, rss), n_points=n, n_params=3,
            converged=True,
            prediction=stretched_exponential_model(t_fit, *popt),
        )
    except (RuntimeError, ValueError) as e:
        results.stretched_exp = FitResult(
            name="Stretched Exponential", params={}, param_errors={},
            rss=np.inf, aic=np.inf, n_points=n, n_params=3,
            converged=False,
        )

    # 3. Power law: E0 * (1 + t/tau)^(-alpha)
    try:
        p0 = [e0_guess, 1e5, 1.0]
        bounds = ([0, 1e-10, 0.01], [np.inf, np.inf, 100.0])
        popt, pcov = curve_fit(power_law_model, t_fit, e_win,
                               p0=p0, bounds=bounds, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        resid = e_win - power_law_model(t_fit, *popt)
        rss = float(np.sum(resid**2))

        results.power_law = FitResult(
            name="Power Law",
            params={"E0": popt[0], "tau": popt[1], "alpha": popt[2]},
            param_errors={"E0": perr[0], "tau": perr[1], "alpha": perr[2]},
            rss=rss, aic=aic(n, 3, rss), n_points=n, n_params=3,
            converged=True,
            prediction=power_law_model(t_fit, *popt),
        )
    except (RuntimeError, ValueError) as e:
        results.power_law = FitResult(
            name="Power Law", params={}, param_errors={},
            rss=np.inf, aic=np.inf, n_points=n, n_params=3,
            converged=False,
        )

    # Best model by AIC
    models = {"Exponential": results.exponential,
              "Stretched Exponential": results.stretched_exp,
              "Power Law": results.power_law}
    best = min(models.items(), key=lambda x: x[1].aic if x[1] else np.inf)
    results.best_model = best[0]

    return results
