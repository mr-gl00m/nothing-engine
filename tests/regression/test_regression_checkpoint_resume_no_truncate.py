# Invariant: resuming a run via from_checkpoint must never destroy already-streamed data.
#   run() chooses recreate('w') vs append('a') via is_restart = self._seg_idx > 0.
# Violation: if the only checkpoint in the file is the initial seg-0 one (a crash before the
#   first checkpoint_interval elapsed), from_checkpoint yields _seg_idx=0, so is_restart is
#   False and run() reopens the existing file with mode 'w', truncating the streamed timeseries.
# Predicted failure: after from_checkpoint(...).run(), the sentinel attribute written before the
#   "crash" is gone -> the file was recreated (truncated), prior data lost.
import numpy as np
import h5py
from nothing_engine.core.bogoliubov import SimulationConfig, build_initial_state
from nothing_engine.experiments.runner import (
    ExperimentRunner, RunConfig, _create_output_file, _save_checkpoint, _append_observables,
)


def test_repro_seg0_resume_truncates(tmp_path):
    cfg = SimulationConfig(n_modes=4, q0=1.0, v0=1e-3)
    rc = RunConfig(total_time=2.0, segment_time=1.0, checkpoint_interval=1e9)
    path = tmp_path / "run.h5"

    f = _create_output_file(path, cfg, rc)
    state = build_initial_state(cfg)
    _save_checkpoint(f, 0.0, state, 0)            # ONLY the initial seg-0 checkpoint
    t_pts = np.array([0.0, 0.5, 1.0])
    y_pts = np.tile(state.reshape(-1, 1), (1, 3))
    _append_observables(f, t_pts, y_pts, cfg)     # stream some data, as a run would
    f.attrs["sentinel"] = 4242                    # survives only if the file is NOT recreated
    n_before = int(f["timeseries/t"].shape[0])
    f.close()
    assert n_before >= 3

    runner = ExperimentRunner.from_checkpoint(str(path))
    runner.run()

    with h5py.File(path, "r") as g:
        assert "sentinel" in g.attrs, (
            "output file was recreated (mode 'w') on resume from a seg-0 checkpoint: "
            "previously streamed data was truncated/lost"
        )
