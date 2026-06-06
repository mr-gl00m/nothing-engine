# Invariant: SimulationConfig persisted to the HDF5 'config' group records boundary,
#   plate_thickness and cutoff_shape (runner._create_output_file writes them). A run resumed
#   via ExperimentRunner.from_checkpoint must reconstruct the SAME physics.
# Violation: from_checkpoint passes only n_modes, plate_mass, spring_k, q0, v0, x_left, method,
#   rtol, atol, max_step to SimulationConfig — it omits boundary, plate_thickness and cutoff_shape.
#   A resumed "periodic" run silently reverts to the "closed" default (omega_n = n*pi/a instead of
#   2*n*pi/a), corrupting the physics of the paper's central topology experiments on restart.
# Predicted failure: AssertionError — reconstructed cfg.boundary == "closed", not "periodic"
#   (and ns_pi differs by a factor of 2).
import numpy as np
from nothing_engine.core.bogoliubov import SimulationConfig, build_initial_state
from nothing_engine.experiments.runner import (
    ExperimentRunner, RunConfig, _create_output_file, _save_checkpoint,
)


def test_repro_checkpoint_drops_boundary(tmp_path):
    cfg = SimulationConfig(n_modes=8, boundary="periodic", q0=2.0, v0=1e-3,
                           plate_thickness=0.0)
    rc = RunConfig(total_time=10.0, segment_time=5.0, checkpoint_interval=5.0)
    path = tmp_path / "run.h5"

    f = _create_output_file(path, cfg, rc)
    state = build_initial_state(cfg)
    # seg_idx >= 1 so from_checkpoint -> run() is treated as a true restart
    _save_checkpoint(f, 5.0, state, 1)
    f.attrs["segment_index"] = 1
    f.close()

    runner = ExperimentRunner.from_checkpoint(str(path))

    # The file recorded boundary="periodic"; the reconstructed config must match.
    assert runner.sim_cfg.boundary == "periodic", (
        f"from_checkpoint silently changed boundary to "
        f"{runner.sim_cfg.boundary!r}; ns_pi[0]={runner.sim_cfg.ns_pi[0]:.6f} "
        f"(expected {2*np.pi:.6f} for periodic, got the closed value {np.pi:.6f})"
    )
