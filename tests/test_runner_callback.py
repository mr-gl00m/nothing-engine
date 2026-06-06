"""The ExperimentRunner progress_callback hook fires per segment with a full payload."""

from nothing_engine.core.bogoliubov import SimulationConfig
from nothing_engine.experiments.runner import ExperimentRunner, RunConfig

REQUIRED_KEYS = {
    "t", "t_end", "pct", "segment", "E_plate", "E_spring",
    "E_field", "E_total", "N_total", "wall", "status",
}


def test_progress_callback_fires_per_segment(tmp_path):
    sim = SimulationConfig(n_modes=16, v0=1.0e-3)
    run = RunConfig(
        total_time=60.0, segment_time=20.0,
        samples_per_unit_time=4, checkpoint_interval=40.0,
    )
    events: list[dict] = []
    runner = ExperimentRunner(
        sim, run,
        output_path=str(tmp_path / "run.h5"),
        progress_callback=events.append,
    )
    path = runner.run()

    assert path.exists()
    assert len(events) >= 3  # 60 / 20 = 3 segments
    for e in events:
        assert REQUIRED_KEYS <= set(e), f"missing keys: {REQUIRED_KEYS - set(e)}"
    final = events[-1]
    assert final["status"] == "completed"
    assert final["pct"] == 100.0


def test_no_callback_is_backward_compatible(tmp_path):
    """Omitting the callback must leave behavior unchanged (no crash, file written)."""
    sim = SimulationConfig(n_modes=16, v0=1.0e-3)
    run = RunConfig(total_time=40.0, segment_time=20.0, samples_per_unit_time=4)
    runner = ExperimentRunner(sim, run, output_path=str(tmp_path / "run.h5"))
    path = runner.run()
    assert path.exists()
