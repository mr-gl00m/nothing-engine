"""Application bootstrap: high-DPI, dark theme, rotating-file logging, main window."""

import sys
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from .theme import apply_theme
from .main_window import MainWindow


def _setup_logging() -> None:
    """Root logging to logs/gui.log (rotating) + stderr, with uncaught-exception capture."""
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        file_handler = RotatingFileHandler(
            logs_dir / "gui.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        root.addHandler(stream)

    def excepthook(exc_type, exc, tb):
        logging.getLogger("nothing_engine.gui").critical(
            "Uncaught exception", exc_info=(exc_type, exc, tb)
        )
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = excepthook


def main() -> int:
    _setup_logging()
    # High-DPI rounding policy must be set before the QApplication is created.
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("Nothing Engine")
    apply_theme(app)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
