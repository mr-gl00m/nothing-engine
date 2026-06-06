"""pyqtgraph widgets: a live streaming view and a tabbed results view.

LivePlot is fed dicts from the progress stream during a run. ResultsView is fed
a single packaged dict from results_loader after a run (or a loaded .h5).
"""

# pyqtgraph delegates GraphicsLayoutWidget.addPlot via __getattr__, which its stubs
# don't expose, so the type checker flags real calls. Suppress that one rule here.
# pyright: reportAttributeAccessIssue=false

import os

# Bind pyqtgraph to PySide6 before it auto-detects a Qt binding.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QLabel  # noqa: E402
import pyqtgraph as pg  # noqa: E402

from .theme import ACCENT, ACCENT_2, BG, FG, FG_MUTED  # noqa: E402

pg.setConfigOptions(antialias=True, background=BG, foreground=FG)

# Energy-curve palette (E_plate, E_spring, E_field, E_total)
_ENERGY_COLORS = {
    "E_plate": ACCENT,
    "E_spring": "#81c784",
    "E_field": ACCENT_2,
    "E_total": "#ce93d8",
}


def _pen(color, width=2):
    return pg.mkPen(color=color, width=width)


class LivePlot(QWidget):
    """Two stacked plots updated live: E_plate(t) and N_total(t)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        glw = pg.GraphicsLayoutWidget()
        layout.addWidget(glw)

        self._p_e = glw.addPlot(row=0, col=0, title="Plate kinetic energy  E_plate(t)")
        self._p_e.setLabel("bottom", "t")
        self._p_e.setLabel("left", "E_plate")
        self._p_e.showGrid(x=True, y=True, alpha=0.2)
        self._c_e = self._p_e.plot([], [], pen=_pen(ACCENT))

        self._p_n = glw.addPlot(row=1, col=0, title="Particle number  N(t)")
        self._p_n.setLabel("bottom", "t")
        self._p_n.setLabel("left", "N_total")
        self._p_n.showGrid(x=True, y=True, alpha=0.2)
        self._p_n.setXLink(self._p_e)
        self._c_n = self._p_n.plot([], [], pen=_pen(ACCENT_2))

        self._t: list[float] = []
        self._e: list[float] = []
        self._n: list[float] = []

    def reset(self) -> None:
        self._t.clear()
        self._e.clear()
        self._n.clear()
        self._c_e.setData([], [])
        self._c_n.setData([], [])

    def append(self, d: dict) -> None:
        self._t.append(d["t"])
        self._e.append(d["E_plate"])
        self._n.append(d["N_total"])
        self._c_e.setData(self._t, self._e)
        self._c_n.setData(self._t, self._n)


class ResultsView(QTabWidget):
    """Ringdown / Energy / Particles / PSD tabs, populated from a results dict."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ringdown = pg.PlotWidget(title="Ringdown")
        self._ringdown.setLogMode(y=True)
        self._ringdown.setLabel("bottom", "t")
        self._ringdown.setLabel("left", "E_plate (log)")
        self._ringdown.addLegend()
        self.addTab(self._ringdown, "Ringdown")

        self._energy = pg.PlotWidget(title="Energy components")
        self._energy.setLabel("bottom", "t")
        self._energy.setLabel("left", "Energy")
        self._energy.addLegend()
        self.addTab(self._energy, "Energy")

        self._particles_glw = pg.GraphicsLayoutWidget()
        self._p_ntot = self._particles_glw.addPlot(row=0, col=0, title="N(t)")
        self._p_ntot.setLabel("bottom", "t")
        self._p_ntot.setLabel("left", "N_total")
        self._p_spec = self._particles_glw.addPlot(row=1, col=0, title="Final per-mode spectrum")
        self._p_spec.setLabel("bottom", "mode n")
        self._p_spec.setLabel("left", "|β_n|²")
        self.addTab(self._particles_glw, "Particles")

        self._psd = pg.PlotWidget(title="Post-ringdown velocity PSD")
        self._psd.setLogMode(x=True, y=True)
        self._psd.setLabel("bottom", "frequency")
        self._psd.setLabel("left", "S_vv(f)")
        self.addTab(self._psd, "PSD")

        self._title = QLabel("")

    def clear_all(self) -> None:
        self._ringdown.clear()
        self._energy.clear()
        self._p_ntot.clear()
        self._p_spec.clear()
        self._psd.clear()

    def show_results(self, data: dict) -> None:
        self.clear_all()
        t = data.get("t")
        if t is None or len(t) == 0:
            return

        # Ringdown
        self._ringdown.plot(t, data["E_plate"], pen=_pen(ACCENT), name="E_plate")
        rd = data.get("ringdown")
        if rd is not None:
            self._ringdown.plot(rd["fit_t"], rd["fit_y"], pen=_pen(ACCENT_2, 2), name=f"fit: {rd['best_model']}")
            title = f"Ringdown — best model: {rd['best_model']}"
            ci = rd.get("gamma_ci")
            if ci is not None:
                title += f"   γ = {ci[0]:.4g}  (95% CI {ci[1]:.4g}–{ci[2]:.4g})"
            self._ringdown.setTitle(title)
        else:
            self._ringdown.setTitle("Ringdown — fit unavailable")

        # Energy components + conservation drift
        for key in ("E_plate", "E_spring", "E_field", "E_total"):
            arr = data.get(key)
            if arr is not None:
                self._energy.plot(t, arr, pen=_pen(_ENERGY_COLORS[key]), name=key)

        # Particles
        self._p_ntot.plot(t, data["total_particles"], pen=_pen(ACCENT_2))
        spec = data.get("particle_spectrum")
        if spec is not None and len(spec) > 0:
            modes = list(range(1, len(spec) + 1))
            self._p_spec.plot(modes, spec, pen=_pen(ACCENT))

        # PSD
        psd = data.get("psd")
        if psd is not None and len(psd["freqs"]) > 0:
            self._psd.plot(psd["freqs"], psd["psd"], pen=_pen(ACCENT))
            self._psd.setTitle("Post-ringdown velocity PSD")
        else:
            self._psd.setTitle("PSD — no post-ringdown window")
