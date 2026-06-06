"""Tests for core/mode_space.py — cavity mode decomposition."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import mode_space

PI = np.pi


class TestModeFrequencies:
    def test_fundamental_mode_unit_cavity(self):
        """omega_1 = pi/a = pi for a=1."""
        omegas = mode_space.mode_frequencies(1, cavity_width=1.0)
        assert len(omegas) == 1
        np.testing.assert_allclose(omegas[0], PI, rtol=1e-14)

    def test_multiple_modes(self):
        """omega_n = n*pi/a."""
        omegas = mode_space.mode_frequencies(5, cavity_width=2.0)
        expected = np.array([1, 2, 3, 4, 5]) * PI / 2.0
        np.testing.assert_allclose(omegas, expected, rtol=1e-14)

    def test_inverse_cavity_scaling(self):
        """Doubling cavity width halves frequencies."""
        w1 = mode_space.mode_frequencies(10, cavity_width=1.0)
        w2 = mode_space.mode_frequencies(10, cavity_width=2.0)
        np.testing.assert_allclose(w2, w1 / 2.0, rtol=1e-14)

    def test_frequencies_squared(self):
        w = mode_space.mode_frequencies(8, 1.5)
        w2 = mode_space.mode_frequencies_squared(8, 1.5)
        np.testing.assert_allclose(w2, w**2, rtol=1e-14)


class TestModeFrequencyDerivatives:
    def test_derivative_sign(self):
        """d(omega_n)/dq < 0: frequency decreases as cavity widens."""
        dw = mode_space.mode_frequency_derivatives(5, cavity_width=1.0)
        assert np.all(dw < 0)

    def test_derivative_finite_difference(self):
        """Compare to numerical finite difference."""
        N = 10
        a = 1.0
        h = 1e-7
        dw_analytic = mode_space.mode_frequency_derivatives(N, a)
        w_plus = mode_space.mode_frequencies(N, a + h)
        w_minus = mode_space.mode_frequencies(N, a - h)
        dw_numeric = (w_plus - w_minus) / (2 * h)
        np.testing.assert_allclose(dw_analytic, dw_numeric, rtol=1e-6)


class TestVacuumInitialConditions:
    def test_shape(self):
        ic = mode_space.vacuum_initial_conditions(10, 1.0)
        assert ic.shape == (40,)

    def test_particle_number_zero(self):
        """Vacuum state has zero particles in all modes."""
        N = 32
        a0 = 1.0
        ic = mode_space.vacuum_initial_conditions(N, a0)
        beta_sq = mode_space.particle_number(ic, N, a0)
        np.testing.assert_allclose(beta_sq, 0.0, atol=1e-15)

    def test_wronskian(self):
        """w_n = u_n * v_dot_n - v_n * u_dot_n = -1/2 for all n."""
        N = 64
        a0 = 1.5
        ic = mode_space.vacuum_initial_conditions(N, a0)
        w = mode_space.wronskian(ic, N)
        np.testing.assert_allclose(w, -0.5, rtol=1e-14)

    def test_mode_energy_is_half_omega(self):
        """Each mode has zero-point energy omega_n / 2."""
        N = 16
        a0 = 1.0
        ic = mode_space.vacuum_initial_conditions(N, a0)
        omegas = mode_space.mode_frequencies(N, a0)
        omegas_sq = omegas**2

        idx = 4 * N
        u = ic[0:idx:4]
        u_dot = ic[1:idx:4]
        v = ic[2:idx:4]
        v_dot = ic[3:idx:4]

        f_sq = u**2 + v**2
        fdot_sq = u_dot**2 + v_dot**2
        E_n = 0.5 * (fdot_sq + omegas_sq * f_sq)
        np.testing.assert_allclose(E_n, omegas / 2.0, rtol=1e-14)


class TestParticleNumber:
    def test_vacuum_zero(self):
        N = 8
        a0 = 1.0
        ic = mode_space.vacuum_initial_conditions(N, a0)
        beta_sq = mode_space.particle_number(ic, N, a0)
        assert np.all(np.abs(beta_sq) < 1e-15)

    def test_total_particle_number_vacuum(self):
        N = 16
        a0 = 2.0
        ic = mode_space.vacuum_initial_conditions(N, a0)
        total = mode_space.total_particle_number(ic, N, a0)
        assert abs(total) < 1e-14


class TestExtractors:
    def test_amplitudes_squared(self):
        N = 4
        state = np.zeros(16)
        state[0] = 3.0   # u_1
        state[2] = 4.0   # v_1
        state[4] = 1.0   # u_2
        state[6] = 0.0   # v_2
        f_sq = mode_space.extract_mode_amplitudes_squared(state, N)
        np.testing.assert_allclose(f_sq[0], 25.0)
        np.testing.assert_allclose(f_sq[1], 1.0)

    def test_velocities_squared(self):
        N = 4
        state = np.zeros(16)
        state[1] = 2.0   # u_dot_1
        state[3] = 3.0   # v_dot_1
        fdot_sq = mode_space.extract_mode_velocities_squared(state, N)
        np.testing.assert_allclose(fdot_sq[0], 13.0)
