"""GUI smoke test: construct the window offscreen and round-trip the config form.

Skipped entirely if PySide6 isn't installed (the gui extra is optional).
"""

import os

import pytest

pytest.importorskip("PySide6")

# Must be set before any QApplication is created.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_main_window_config_roundtrip():
    from PySide6.QtWidgets import QApplication
    from nothing_engine.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])  # noqa: F841 (kept alive)
    win = MainWindow()

    cfg = win.config_panel.get_config()
    assert {"n_modes", "boundary", "audit_halt", "total_time"} <= set(cfg)

    win.config_panel.set_config({"n_modes": 77, "boundary": "periodic",
                                 "v0": 2.5e-3, "audit_halt": False})
    cfg2 = win.config_panel.get_config()
    assert cfg2["n_modes"] == 77
    assert cfg2["boundary"] == "periodic"
    assert abs(cfg2["v0"] - 2.5e-3) < 1e-12
    assert cfg2["audit_halt"] is False

    # argv mapping turns the bool into the right BooleanOptionalAction flag.
    from nothing_engine.gui.run_controller import _params_to_argv
    argv = _params_to_argv(cfg2, "out.h5")
    assert "--no-audit-halt" in argv
    assert "--boundary" in argv
