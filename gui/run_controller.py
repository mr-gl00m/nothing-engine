"""Drives a `run_single` subprocess and converts its output into Qt signals.

The GUI never imports the heavy solver. It launches

    <python> -m nothing_engine.experiments.run_single --output ... [flags]

reads the JSON progress stream off stdout, and re-emits it as typed signals.
The runner's human log lands on stderr and is forwarded verbatim as `log`.
"""

import os
import sys
import json

from PySide6.QtCore import QObject, QProcess, QTimer, Signal


def _decode(qba) -> str:
    """QByteArray -> str. PySide6 stubs mistype .data() as memoryview[int]; at
    runtime it returns bytes, so .decode works. The ignore documents that mismatch."""
    return qba.data().decode("utf-8", "replace")  # pyright: ignore[reportAttributeAccessIssue]


def _params_to_argv(params: dict, output_path: str) -> list[str]:
    """Map a config dict onto run_single CLI flags."""
    argv = ["-m", "nothing_engine.experiments.run_single", "--output", output_path]
    for key, value in params.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            # BooleanOptionalAction: --audit-halt / --no-audit-halt
            argv.append(flag if value else "--no-" + key.replace("_", "-"))
        else:
            argv.extend([flag, str(value)])
    return argv


class RunController(QObject):
    progress = Signal(object)        # dict from a {"event":"progress"} line
    finished = Signal(str, str)      # (status, output_path) on clean exit
    failed = Signal(str)             # error message
    log = Signal(str)                # stderr text
    started = Signal()

    KILL_GRACE_MS = 3000

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc: QProcess | None = None
        self._stdout_buf = ""
        self._final_status = "unknown"
        self._final_path = ""
        self._done = False
        self._errored = False
        self._stopping = False

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.state() != QProcess.ProcessState.NotRunning

    def start(self, params: dict, output_path: str) -> None:
        if self.is_running():
            return
        self._stdout_buf = ""
        self._final_status = "unknown"
        self._final_path = output_path
        self._done = False
        self._errored = False
        self._stopping = False

        proc = QProcess(self)
        proc.setProgram(sys.executable)
        proc.setArguments(_params_to_argv(params, output_path))
        proc.setWorkingDirectory(os.getcwd())
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        proc.readyReadStandardOutput.connect(self._on_stdout)
        proc.readyReadStandardError.connect(self._on_stderr)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)
        self._proc = proc
        proc.start()
        self.started.emit()

    def stop(self) -> None:
        """Ask the process to terminate, then hard-kill if it lingers."""
        if not self.is_running():
            return
        self._stopping = True
        proc = self._proc
        assert proc is not None
        proc.terminate()
        QTimer.singleShot(self.KILL_GRACE_MS, self._force_kill)

    # -- internals ----------------------------------------------------------
    def _force_kill(self) -> None:
        if self.is_running():
            assert self._proc is not None
            self._proc.kill()

    def _on_stdout(self) -> None:
        assert self._proc is not None
        chunk = _decode(self._proc.readAllStandardOutput())
        self._stdout_buf += chunk
        while "\n" in self._stdout_buf:
            line, self._stdout_buf = self._stdout_buf.split("\n", 1)
            line = line.strip()
            if line:
                self._handle_line(line)

    def _handle_line(self, line: str) -> None:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # Not JSON (stray print) — forward to the log so nothing is lost.
            self.log.emit(line)
            return
        event = obj.get("event")
        if event == "progress":
            self.progress.emit(obj)
        elif event == "done":
            self._done = True
            self._final_status = obj.get("status", "unknown")
            self._final_path = obj.get("output", self._final_path)
        elif event == "error":
            self._errored = True
            self.failed.emit(obj.get("message", "run failed"))

    def _on_stderr(self) -> None:
        assert self._proc is not None
        text = _decode(self._proc.readAllStandardError())
        if text:
            self.log.emit(text.rstrip("\n"))

    def _on_error(self, err) -> None:
        if not self._errored:
            self._errored = True
            self.failed.emit(f"process error: {err}")

    def _on_finished(self, exit_code: int, _status) -> None:
        # Drain any tail still in the buffer.
        if self._stdout_buf.strip():
            self._handle_line(self._stdout_buf.strip())
            self._stdout_buf = ""
        self._proc = None
        if self._errored:
            return
        if self._stopping:
            self.finished.emit("stopped", self._final_path)
        elif self._done:
            self.finished.emit(self._final_status, self._final_path)
        elif exit_code != 0:
            self.failed.emit(f"run_single exited with code {exit_code}")
        else:
            # Clean exit but no done event — treat as finished with unknown status.
            self.finished.emit(self._final_status, self._final_path)
