"""Main control-panel window: config → run → live → results."""

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QMainWindow, QSplitter, QScrollArea, QTabWidget, QToolBar, QProgressBar,
    QLabel, QDockWidget, QPlainTextEdit, QFileDialog,
)

from .config_panel import ConfigPanel
from .plots import LivePlot, ResultsView
from .run_controller import RunController
from .results_loader import ResultsLoader

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nothing Engine: Control Panel")
        self.resize(1280, 820)

        self.config_panel = ConfigPanel()
        self.live = LivePlot()
        self.results = ResultsView()
        self.controller = RunController(self)
        self.loader = ResultsLoader(self)

        self._build_layout()
        self._build_toolbar()
        self._build_statusbar()
        self._wire()
        self._set_running(False)

    # -- construction -------------------------------------------------------
    def _build_layout(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.config_panel)
        scroll.setMinimumWidth(340)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.live, "Live")
        self.tabs.addTab(self.results, "Results")

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 920])
        self.setCentralWidget(splitter)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        dock = QDockWidget("Log", self)
        dock.setWidget(self.log_view)
        dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self._log_dock = dock

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        self.act_run = QAction("▶ Run", self)
        self.act_stop = QAction("■ Stop", self)
        self.act_load = QAction("Load .h5…", self)
        self.act_save_preset = QAction("Save preset…", self)
        self.act_load_preset = QAction("Load preset…", self)

        self.act_run.triggered.connect(self._on_run)
        self.act_stop.triggered.connect(self._on_stop)
        self.act_load.triggered.connect(self._on_load_h5)
        self.act_save_preset.triggered.connect(self._on_save_preset)
        self.act_load_preset.triggered.connect(self._on_load_preset)

        tb.addAction(self.act_run)
        tb.addAction(self.act_stop)
        tb.addSeparator()
        tb.addAction(self.act_load)
        tb.addSeparator()
        tb.addAction(self.act_save_preset)
        tb.addAction(self.act_load_preset)

    def _build_statusbar(self) -> None:
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setFixedWidth(260)
        self.status_label = QLabel("Idle")
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.progress)

    def _wire(self) -> None:
        self.controller.progress.connect(self._on_progress)
        self.controller.log.connect(self._append_log)
        self.controller.finished.connect(self._on_finished)
        self.controller.failed.connect(self._on_failed)
        self.loader.loaded.connect(self._on_results_loaded)
        self.loader.failed.connect(self._on_failed)

    # -- run lifecycle ------------------------------------------------------
    def _on_run(self) -> None:
        if self.controller.is_running():
            return
        try:
            params = self.config_panel.get_config()
        except ValueError as exc:
            self._set_status(f"Bad input: {exc}")
            return
        out = self.config_panel.output_path()
        if not out:
            self._set_status("Set an output path first")
            return

        self.live.reset()
        self.tabs.setCurrentWidget(self.live)
        self.progress.setValue(0)
        self._set_running(True)
        self._set_status("Starting…")
        self._append_log(f"$ run_single --output {out}")
        self.controller.start(params, out)

    def _on_stop(self) -> None:
        if self.controller.is_running():
            self._set_status("Stopping…")
            self.controller.stop()

    def _on_progress(self, d: dict) -> None:
        self.live.append(d)
        self.progress.setValue(int(d.get("pct", 0)))
        self._set_status(
            f"t={d['t']:.1f}/{d['t_end']:.0f}  "
            f"E_plate={d['E_plate']:.4e}  N={d['N_total']:.4e}  "
            f"seg={d['segment']}  wall={d['wall']:.1f}s"
        )

    def _on_finished(self, status: str, path: str) -> None:
        self._set_running(False)
        self.progress.setValue(100 if status == "completed" else self.progress.value())
        self._set_status(f"Run {status}")
        if path and Path(path).exists():
            self._set_status(f"Run {status}: loading results…")
            self.loader.load(path)

    def _on_failed(self, message: str) -> None:
        self._set_running(False)
        self._set_status(f"Failed: {message}")
        self._append_log(f"ERROR: {message}")

    # -- results ------------------------------------------------------------
    def _on_load_h5(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open experiment HDF5", "data/experiments", "HDF5 files (*.h5 *.hdf5)"
        )
        if path:
            self._set_status("Loading results…")
            self.loader.load(path)

    def _on_results_loaded(self, data: dict) -> None:
        self.results.show_results(data)
        self.tabs.setCurrentWidget(self.results)
        errs = data.get("errors") or {}
        for stage, msg in errs.items():
            self._append_log(f"[{stage}] {msg}")
        self._set_status("Results loaded")

    # -- presets ------------------------------------------------------------
    def _on_save_preset(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save preset", "preset.json", "JSON (*.json)")
        if path:
            self.config_panel.save_preset(path)
            self._set_status(f"Preset saved: {path}")

    def _on_load_preset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load preset", "", "JSON (*.json)")
        if path:
            self.config_panel.load_preset(path)
            self._set_status(f"Preset loaded: {path}")

    # -- helpers ------------------------------------------------------------
    def _set_running(self, running: bool) -> None:
        self.act_run.setEnabled(not running)
        self.act_stop.setEnabled(running)
        self.act_load.setEnabled(not running)
        self.act_load_preset.setEnabled(not running)
        self.config_panel.setEnabled(not running)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _append_log(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    def closeEvent(self, event) -> None:
        if self.controller.is_running():
            self.controller.stop()
        event.accept()
