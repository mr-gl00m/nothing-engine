"""Parameter form for SimulationConfig + RunConfig.

Spec-driven: FIELD_SPECS is the single list that defines every widget, its type,
and its default (defaults read from the engine dataclasses so they can't drift).
get_config()/set_config() round-trip a plain dict; presets are atomic JSON.
"""

import os
import json
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QFormLayout, QVBoxLayout, QHBoxLayout, QGroupBox, QLineEdit,
    QSpinBox, QComboBox, QCheckBox, QPushButton, QLabel, QFileDialog,
)

from nothing_engine.core.bogoliubov import SimulationConfig
from nothing_engine.experiments.runner import RunConfig

_SIM = SimulationConfig()
_RUN = RunConfig()

# (group, key, label, kind, default, extra)
#   kind "int"    -> QSpinBox,   extra = (min, max)
#   kind "float"  -> QLineEdit,  extra = None
#   kind "choice" -> QComboBox,  extra = [options...]
#   kind "bool"   -> QCheckBox,  extra = None
FIELD_SPECS = [
    ("Physics", "n_modes", "Mode count N", "int", _SIM.n_modes, (1, 8192)),
    ("Physics", "plate_mass", "Plate mass M", "float", _SIM.plate_mass, None),
    ("Physics", "spring_k", "Spring k", "float", _SIM.spring_k, None),
    ("Physics", "q0", "Initial position q0", "float", _SIM.q0, None),
    ("Physics", "v0", "Initial velocity v0", "float", _SIM.v0, None),
    ("Physics", "x_left", "Left wall x_L", "float", _SIM.x_left, None),
    ("Physics", "boundary", "Boundary", "choice", _SIM.boundary, ["closed", "periodic"]),
    ("Physics", "plate_thickness", "Plate thickness (0=auto)", "float", _SIM.plate_thickness, None),
    ("Physics", "cutoff_shape", "Cutoff shape", "choice", _SIM.cutoff_shape, ["sigmoid", "gaussian"]),

    ("Integrator", "method", "Method", "choice", _SIM.method,
     ["RK45", "DOP853", "Radau", "BDF", "LSODA"]),
    ("Integrator", "rtol", "rtol", "float", _SIM.rtol, None),
    ("Integrator", "atol", "atol", "float", _SIM.atol, None),
    ("Integrator", "max_step", "Max step", "float", _SIM.max_step, None),
    ("Integrator", "audit_tolerance_factor", "Audit tol. factor", "float", _SIM.audit_tolerance_factor, None),
    ("Integrator", "audit_halt", "Halt on energy drift", "bool", _SIM.audit_halt, None),

    ("Run", "total_time", "Total time", "float", _RUN.total_time, None),
    ("Run", "segment_time", "Segment time", "float", _RUN.segment_time, None),
    ("Run", "samples_per_unit_time", "Samples / unit time", "int", _RUN.samples_per_unit_time, (1, 10000)),
    ("Run", "checkpoint_interval", "Checkpoint interval", "float", _RUN.checkpoint_interval, None),
    ("Run", "log_interval_segments", "Log every N segments", "int", _RUN.log_interval_segments, (1, 100000)),
]


def _fmt(v) -> str:
    """Compact float formatting: 1e-10 stays 1e-10, 10000.0 -> 10000."""
    return f"{v:g}"


def _atomic_write_json(path: Path, obj: dict) -> None:
    """Write JSON via a temp file + os.replace so a crash can't truncate the preset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


class ConfigPanel(QWidget):
    """Left-hand parameter form plus the output-path row."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._widgets: dict[str, QWidget] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        groups: dict[str, QFormLayout] = {}
        for group, key, label, kind, default, extra in FIELD_SPECS:
            if group not in groups:
                box = QGroupBox(group)
                form = QFormLayout(box)
                form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
                groups[group] = form
                root.addWidget(box)
            widget = self._make_widget(kind, default, extra)
            self._widgets[key] = widget
            groups[group].addRow(QLabel(label), widget)

        # Output path row
        out_box = QGroupBox("Output")
        out_layout = QHBoxLayout(out_box)
        self.output_edit = QLineEdit(self._default_output())
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_output)
        out_layout.addWidget(self.output_edit, 1)
        out_layout.addWidget(browse)
        root.addWidget(out_box)
        root.addStretch(1)

    # -- widget factory -----------------------------------------------------
    def _make_widget(self, kind: str, default, extra):
        if kind == "int":
            w = QSpinBox()
            lo, hi = extra
            w.setRange(lo, hi)
            w.setValue(int(default))
            return w
        if kind == "choice":
            w = QComboBox()
            w.addItems(extra)
            w.setCurrentText(str(default))
            return w
        if kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(default))
            return w
        # float
        w = QLineEdit(_fmt(default))
        return w

    # -- public API ---------------------------------------------------------
    def get_config(self) -> dict:
        """Read every widget into a typed dict. Raises ValueError on bad float text."""
        out: dict = {}
        for key, w in self._widgets.items():
            if isinstance(w, QSpinBox):
                out[key] = int(w.value())
            elif isinstance(w, QComboBox):
                out[key] = w.currentText()
            elif isinstance(w, QCheckBox):
                out[key] = bool(w.isChecked())
            elif isinstance(w, QLineEdit):  # float
                text = w.text().strip()
                try:
                    out[key] = float(text)
                except ValueError as exc:
                    raise ValueError(f"{key}: '{text}' is not a number") from exc
        return out

    def set_config(self, cfg: dict) -> None:
        """Set widgets from a dict; unknown keys ignored, missing keys left as-is."""
        for key, value in cfg.items():
            w = self._widgets.get(key)
            if isinstance(w, QSpinBox):
                w.setValue(int(value))
            elif isinstance(w, QComboBox):
                w.setCurrentText(str(value))
            elif isinstance(w, QCheckBox):
                w.setChecked(bool(value))
            elif isinstance(w, QLineEdit):
                w.setText(_fmt(value))

    def output_path(self) -> str:
        return self.output_edit.text().strip()

    def save_preset(self, path: str) -> None:
        _atomic_write_json(Path(path), self.get_config())

    def load_preset(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            self.set_config(json.load(fh))

    # -- helpers ------------------------------------------------------------
    def _default_output(self) -> str:
        boundary = _SIM.boundary
        return str(Path("data") / "experiments" / f"gui_{boundary}_N{_SIM.n_modes}.h5")

    def _browse_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Output HDF5", self.output_edit.text(), "HDF5 files (*.h5 *.hdf5)"
        )
        if path:
            self.output_edit.setText(path)
