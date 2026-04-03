"""Tests for core/bogoliubov.py — the main Track B simulation engine.

These tests verify:
1. State packing/unpacking roundtrip
2. Static plate produces zero particles (Gate 4.7)
3. Energy conservation for coupled dynamics (Gate 4.3)
4. Dynamic Casimir effect with prescribed motion (Gate 4.2)
5. Adiabatic limit (Gate 4.6)
6. Wronskian conservation (independent integrator check)
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import mode_space
from core import energy as energy_mod
from core.bogoliubov import (
    SimulationConfig, SimulationResult,
    pack_state, unpack_state, build_initial_state,
    make_rhs, run_simulation, audit_result,
)

PI = np.pi


class TestStatePacking:
    def test_pack_unpack_roundtrip(self):
        N = 16
        mode_state = np.random.randn(4 * N)
        q, v = 1.23, -0.456
        y = pack_state(mode_state, q, v)
        assert y.shape == (4 * N + 2,)

        ms_out, q_out, v_out = unpack_state(y, N)
        np.testing.assert_array_equal(ms_out, mode_state)
        assert q_out == q
        assert v_out == v

    def test_state_dimension(self):
        for N in [1, 10, 64, 256]:
            cfg = SimulationConfig(n_modes=N, t_span=(0, 1))
            y0 = build_initial_state(cfg)
            assert y0.shape == (4 * N + 2,)

    def test_initial_state_vacuum(self):
        """Initial mode state has zero particles."""
        N = 32
        cfg = SimulationConfig(n_modes=N, q0=1.0, v0=0.01, t_span=(0, 1))
        y0 = build_initial_state(cfg)
        ms, q, v = unpack_state(y0, N)
        a0 = q - cfg.x_left
        beta_sq = mode_space.particle_number(ms, N, a0)
        np.testing.assert_allclose(beta_sq, 0.0, atol=1e-15)
        assert q == cfg.q0
        assert v == cfg.v0


class TestStaticPlate:
    """Gate 4.7: Truly static plate (prescribed motion) should create zero particles.

    NOTE: With v0=0 and k=0, the unbalanced truncated vacuum force
    (~N^2) accelerates the plate, creating particles. This is a UV
    artifact, not a physics bug. To test the true "static plate"
    condition, we use prescribed motion to hold the plate fixed.
    """

    def test_prescribed_static_zero_particles(self):
        """Plate held fixed by prescribed motion creates zero particles."""
        N = 32
        a0 = 1.0
        cfg = SimulationConfig(
            n_modes=N,
            plate_mass=1e4,
            spring_k=0.0,
            q0=a0,
            v0=0.0,
            t_span=(0.0, 50.0),
            rtol=1e-13,
            atol=1e-15,
            max_step=0.005,
            audit_halt=False,
        )

        def q_const(t):
            return a0

        def v_zero(t):
            return 0.0

        result = run_simulation(cfg, prescribed_motion=(q_const, v_zero))

        # Check particle number at multiple time points
        for i in [0, len(result.t) // 4, len(result.t) // 2, -1]:
            ms = result.mode_state_at(i)
            beta_sq = mode_space.particle_number(ms, N, a0)
            # Threshold 1e-10: numerical noise from high-n modes accumulates
            # in the particle number formula (small difference of O(1) terms).
            # This is integrator error, not physics.
            assert np.max(np.abs(beta_sq)) < 1e-10, \
                f"Particles created at t={result.t[i]:.2f}: max |beta|^2 = {np.max(np.abs(beta_sq)):.2e}"

    def test_prescribed_static_wronskian_conserved(self):
        """Wronskian should be -1/2 throughout for static plate."""
        N = 16
        a0 = 1.0
        cfg = SimulationConfig(
            n_modes=N,
            plate_mass=1e4,
            q0=a0,
            v0=0.0,
            t_span=(0.0, 20.0),
            rtol=1e-13,
            atol=1e-15,
            max_step=0.005,
            audit_halt=False,
        )

        result = run_simulation(cfg, prescribed_motion=(lambda t: a0, lambda t: 0.0))

        for i in range(0, len(result.t), max(1, len(result.t) // 20)):
            ms = result.mode_state_at(i)
            w = mode_space.wronskian(ms, N)
            np.testing.assert_allclose(w, -0.5, rtol=1e-9,
                err_msg=f"Wronskian drift at t={result.t[i]:.2f}")


class TestEnergyConservation:
    """Gate 4.3: Total energy must be conserved in closed system."""

    def test_energy_conserved_coupled(self):
        """Energy conservation with coupled plate-field dynamics.

        Uses fewer modes (16) for tighter conservation, and tight
        integrator tolerances to minimize numerical drift.
        """
        N = 16
        cfg = SimulationConfig(
            n_modes=N,
            plate_mass=1e4,
            spring_k=0.0,
            q0=1.0,
            v0=1e-3,
            t_span=(0.0, 30.0),
            rtol=1e-13,
            atol=1e-15,
            max_step=0.005,
            audit_tolerance_factor=1e-4,
            audit_halt=True,
        )
        result = run_simulation(cfg)
        assert len(result.t) > 10, "Simulation too short"

        # Compute energy at all time points
        E0 = result.energy_at(0)["E_total"]
        E_plate_0 = result.energy_at(0)["E_plate"]

        max_drift = 0.0
        for i in range(len(result.t)):
            E = result.energy_at(i)["E_total"]
            drift = abs(E - E0)
            if drift > max_drift:
                max_drift = drift

        # Drift relative to plate energy should be small
        relative_to_plate = max_drift / max(E_plate_0, 1e-20)
        assert relative_to_plate < 1e-4, \
            f"Energy drift {max_drift:.2e} exceeds threshold (plate E = {E_plate_0:.2e}, relative = {relative_to_plate:.2e})"

    def test_audit_passes(self):
        """The formal EnergyAuditor should pass."""
        N = 16
        cfg = SimulationConfig(
            n_modes=N,
            plate_mass=1e4,
            v0=1e-3,
            t_span=(0.0, 20.0),
            rtol=1e-12,
            atol=1e-14,
            max_step=0.005,
            audit_tolerance_factor=1e-4,
            audit_halt=True,
        )
        result = run_simulation(cfg)
        auditor = audit_result(result, check_every=10)
        s = auditor.summary()
        assert s["passed"], f"Audit failed: max_drift={s['max_drift']:.2e}"


class TestDynamicCasimirEffect:
    """Gate 4.2: Prescribed oscillation at 2*omega_1 produces parametric amplification."""

    def test_parametric_growth(self):
        """Fundamental mode should show exponential growth for resonant driving."""
        N = 16
        a0 = 1.0
        omega_1 = PI / a0
        eps = 1e-3  # oscillation amplitude
        Omega = 2 * omega_1  # resonant frequency

        cfg = SimulationConfig(
            n_modes=N,
            plate_mass=1e4,
            q0=a0,
            v0=0.0,
            t_span=(0.0, 30.0),
            rtol=1e-12,
            atol=1e-14,
            max_step=0.002,
            audit_halt=False,
        )

        def q_func(t):
            return a0 + eps * np.sin(Omega * t)

        def v_func(t):
            return eps * Omega * np.cos(Omega * t)

        result = run_simulation(cfg, prescribed_motion=(q_func, v_func))

        # Extract |beta_1|^2 over time
        beta1_sq = []
        for i in range(len(result.t)):
            ms = result.mode_state_at(i)
            a = q_func(result.t[i]) - cfg.x_left
            pn = mode_space.particle_number(ms, N, a)
            beta1_sq.append(pn[0])  # mode 1

        beta1_sq = np.array(beta1_sq)

        # Should grow from 0. Check that final value is significantly above zero
        assert beta1_sq[-1] > 1e-8, \
            f"No particle creation: final |beta_1|^2 = {beta1_sq[-1]:.2e}"

        # Check that mode 1 dominates (resonance selectivity)
        ms_final = result.mode_state_at(-1)
        a_final = q_func(result.t[-1]) - cfg.x_left
        pn_final = mode_space.particle_number(ms_final, N, a_final)
        assert pn_final[0] > 10 * np.max(pn_final[2:]), \
            "Mode 1 should dominate for 2*omega_1 driving"


class TestAdiabaticLimit:
    """Gate 4.6: Very heavy/slow plate should create negligible particles."""

    def test_heavy_plate_few_particles(self):
        N = 16
        cfg = SimulationConfig(
            n_modes=N,
            plate_mass=1e8,  # Very heavy
            spring_k=0.0,
            q0=1.0,
            v0=1e-5,  # Very slow
            t_span=(0.0, 50.0),
            rtol=1e-12,
            atol=1e-14,
            max_step=0.005,
            audit_halt=False,
        )
        result = run_simulation(cfg)

        # Total particles should be tiny
        total_N = result.total_particle_number_at(-1)
        assert total_N < 1e-6, \
            f"Adiabatic limit violated: N_total = {total_N:.2e}"


class TestEquilibrium:
    def test_free_plate_q_eq(self):
        """For k=0, q_eq = q0."""
        cfg = SimulationConfig(spring_k=0.0, q0=1.5, t_span=(0, 1))
        assert cfg.q_eq == 1.5

    def test_spring_equilibrium_offset(self):
        """For k>0, q_eq shifts to balance truncated Casimir force."""
        N = 32
        k = 100.0
        q0 = 1.0
        cfg = SimulationConfig(n_modes=N, spring_k=k, q0=q0, t_span=(0, 1))
        from core.radiation_pressure import static_casimir_force_truncated
        F_trunc = static_casimir_force_truncated(N, q0)
        expected_q_eq = q0 + F_trunc / k
        np.testing.assert_allclose(cfg.q_eq, expected_q_eq, rtol=1e-12)
