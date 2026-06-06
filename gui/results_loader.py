"""Off-thread HDF5 + analysis loader for the results view.

Given a completed `.h5`, this reads the timeseries, runs the ringdown fit and the
post-ringdown PSD (each guarded — short runs legitimately have neither), and
packages everything as plain numpy arrays so the GUI thread only plots. Runs on
the global QThreadPool; results arrive on the `loaded` signal.
"""

# h5py ships loose type stubs (Group.__getitem__ -> Group|Dataset|Datatype), so
# correct dataset indexing/attr access here trips the type checker. Suppress those
# stub-driven rules; the physics core keeps full type checking.
# pyright: reportIndexIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportOperatorIssue=false, reportAttributeAccessIssue=false

import logging

import h5py

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from nothing_engine.analysis import ringdown_fit as rf
from nothing_engine.analysis import psd_analysis as psd

logger = logging.getLogger(__name__)

_TS_KEYS = ("t", "plate_q", "plate_v", "E_plate", "E_spring",
            "E_field", "E_total", "total_particles")


def _load(path: str) -> dict:
    """Read timeseries + run analysis. Returns a dict of plain arrays/values."""
    data: dict = {"errors": {}}
    with h5py.File(path, "r") as f:
        ts = f["timeseries"]
        for key in _TS_KEYS:
            data[key] = ts[key][:] if key in ts else None
        pn = ts["particle_number"]
        data["particle_spectrum"] = pn[-1, :] if pn.shape[0] > 0 else None
        data["n_modes"] = int(pn.shape[1]) if pn.ndim == 2 else 0
        if "config" in f:
            data["config"] = {k: f["config"].attrs[k] for k in f["config"].attrs}

    t = data.get("t")
    e_plate = data.get("E_plate")

    # Ringdown fit (guarded: needs enough decay to converge)
    data["ringdown"] = None
    if t is not None and e_plate is not None and len(t) >= 4:
        try:
            rr = rf.fit_ringdown(t, e_plate)
            best = {
                "Exponential": rr.exponential,
                "Stretched Exponential": rr.stretched_exp,
                "Power Law": rr.power_law,
            }.get(rr.best_model)
            fit_t = fit_y = None
            if best is not None and best.prediction is not None:
                t0, t1 = rr.fitting_window
                mask = (t >= t0) & (t <= t1)
                fit_t = t[mask]
                if len(fit_t) != len(best.prediction):
                    fit_t = fit_y = None
                else:
                    fit_y = best.prediction
            gamma_ci = rr.exponential.gamma_with_ci(0.95) if rr.exponential else None
            data["ringdown"] = {
                "best_model": rr.best_model,
                "summary": rr.summary(),
                "fit_t": fit_t,
                "fit_y": fit_y,
                "gamma_ci": gamma_ci,
            }
        except Exception as exc:  # fit is best-effort; never block the view
            data["errors"]["ringdown"] = str(exc)
            logger.info("ringdown fit skipped: %s", exc)

    # Post-ringdown PSD (guarded: needs a settled window after ringdown)
    data["psd"] = None
    try:
        t_v, v = psd.load_velocity_data(path)
        if e_plate is not None:
            t_start = psd.find_post_ringdown_start(t, e_plate)
            res = psd.compute_psd(t_v, v, post_ringdown_t=t_start)
            if len(res.freqs) > 0:
                data["psd"] = {"freqs": res.freqs, "psd": res.psd}
    except Exception as exc:
        data["errors"]["psd"] = str(exc)
        logger.info("PSD skipped: %s", exc)

    return data


class _LoaderSignals(QObject):
    loaded = Signal(object)
    failed = Signal(str)


class _LoaderTask(QRunnable):
    def __init__(self, path: str, signals: _LoaderSignals):
        super().__init__()
        self._path = path
        self._signals = signals

    def run(self) -> None:
        try:
            self._signals.loaded.emit(_load(self._path))
        except Exception as exc:
            logger.exception("results load failed")
            self._signals.failed.emit(f"{type(exc).__name__}: {exc}")


class ResultsLoader(QObject):
    """Kick off an off-thread load; subscribe to `loaded`/`failed`."""

    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._signals = _LoaderSignals(self)
        self._signals.loaded.connect(self.loaded)
        self._signals.failed.connect(self.failed)

    def load(self, path: str) -> None:
        QThreadPool.globalInstance().start(_LoaderTask(path, self._signals))
