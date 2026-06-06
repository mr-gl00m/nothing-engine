"""run_single CLI: arg→config mapping and the stdout JSON progress contract."""

import sys
import json
import subprocess

from nothing_engine.experiments.run_single import build_parser, build_configs


def test_arg_to_config_mapping():
    args = build_parser().parse_args([
        "--output", "x.h5", "--n-modes", "32", "--boundary", "periodic",
        "--no-audit-halt", "--total-time", "100", "--method", "DOP853",
    ])
    sim, run = build_configs(args)
    assert sim.n_modes == 32
    assert sim.boundary == "periodic"
    assert sim.audit_halt is False
    assert sim.method == "DOP853"
    assert run.total_time == 100.0


def test_subprocess_json_stream(tmp_path):
    out = tmp_path / "run.h5"
    proc = subprocess.run(
        [sys.executable, "-m", "nothing_engine.experiments.run_single",
         "--output", str(out), "--n-modes", "16",
         "--total-time", "40", "--segment-time", "20",
         "--samples-per-unit-time", "4", "--checkpoint-interval", "40"],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr

    events = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    assert events, "no JSON events emitted"
    assert all("event" in e for e in events)

    progress = [e for e in events if e["event"] == "progress"]
    assert progress
    for key in ("t", "t_end", "pct", "E_plate", "N_total", "status"):
        assert key in progress[0]

    done = [e for e in events if e["event"] == "done"]
    assert len(done) == 1
    assert done[0]["status"] == "completed"
    assert out.exists()
