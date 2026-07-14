"""Dark theme and accent color for the control panel.

A single QSS string plus an ACCENT constant. Change ACCENT to recolor the
highlights (buttons, selection, progress bar, plot curves) in one place.
"""

from PySide6.QtGui import QColor

# Single source of truth for the highlight color. Plots read ACCENT too.
ACCENT = "#4fc3f7"      # cyan
ACCENT_2 = "#ffb74d"    # amber: secondary plot curves
BG = "#1e1f22"
BG_ELEVATED = "#2b2d31"
FG = "#e4e6eb"
FG_MUTED = "#9aa0a6"
BORDER = "#3a3d42"

STYLESHEET = f"""
QWidget {{
    background-color: {BG};
    color: {FG};
    font-size: 13px;
}}
QMainWindow, QDialog {{ background-color: {BG}; }}
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {FG_MUTED};
}}
QLabel {{ background: transparent; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {{
    background-color: {BG_ELEVATED};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 3px 6px;
    selection-background-color: {ACCENT};
    selection-color: {BG};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {ACCENT};
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background-color: {BG_ELEVATED};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    selection-color: {BG};
}}
QPushButton, QToolButton {{
    background-color: {BG_ELEVATED};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 5px 14px;
}}
QPushButton:hover, QToolButton:hover {{ border: 1px solid {ACCENT}; }}
QPushButton:pressed, QToolButton:pressed {{ background-color: {BORDER}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {FG_MUTED}; border-color: {BORDER}; }}
QToolBar {{ background-color: {BG_ELEVATED}; border-bottom: 1px solid {BORDER}; spacing: 4px; padding: 4px; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 4px; }}
QTabBar::tab {{
    background: {BG_ELEVATED};
    border: 1px solid {BORDER};
    padding: 6px 14px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}
QTabBar::tab:selected {{ background: {BG}; border-bottom-color: {ACCENT}; color: {ACCENT}; }}
QProgressBar {{
    background-color: {BG_ELEVATED};
    border: 1px solid {BORDER};
    border-radius: 4px;
    text-align: center;
    height: 18px;
}}
QProgressBar::chunk {{ background-color: {ACCENT}; border-radius: 3px; }}
QScrollArea {{ border: none; }}
QScrollBar:vertical {{ background: {BG}; width: 12px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 6px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {FG_MUTED}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QSplitter::handle {{ background: {BORDER}; }}
QStatusBar {{ background: {BG_ELEVATED}; border-top: 1px solid {BORDER}; }}
"""


def apply_theme(app) -> None:
    """Apply the dark stylesheet and a matching palette to the QApplication."""
    app.setStyleSheet(STYLESHEET)


def accent_color() -> QColor:
    return QColor(ACCENT)
