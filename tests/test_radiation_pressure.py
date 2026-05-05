"""Tests for core/radiation_pressure.py — field force computation."""

import numpy as np
import pytest

from nothing_engine.core import mode_space, radiation_pressure as rp

PI = np.pi


class TestStaticCasimirForce:
    def test_truncated_formula(self):
        """F_trunc = pi * N * (N+1) / (4 * a^2)."""
        N = 100
        a = 1.0
        F = rp.static_casimir_force_truncated(N, a)
        expected = PI * N * (N + 1) / (4.0 * a**2)
        np.testing.assert_allclose(F, expected, rtol=1e-14)

    def test_truncated_scales_with_a(self):
        """F ~ 1/a^2."""
        N = 50
        F1 = rp.static_casimir_force_truncated(N, 1.0)
        F2 = rp.static_casimir_force_truncated(N, 2.0)
        np.testing.assert_allclose(F2, F1 / 4.0, rtol=1e-14)

    def test_regularized_formula(self):
        """F_reg = -pi / (24 * a^2)."""
        a = 1.5
        F = rp.static_casimir_force_regularized(a)
        expected = -PI / (24.0 * a**2)
        np.testing.assert_allclose(F, expected, rtol=1e-14)

    def test_regularized_negative(self):
        """Physical Casimir force is attractive (negative)."""
        assert rp.static_casimir_force_regularized(1.0) < 0


class TestFieldForce:
    def test_vacuum_force_matches_truncated(self):
        """At t=0 (vacuum), field force = truncated Casimir force."""
        N = 32
        a0 = 1.0
        ic = mode_space.vacuum_initial_conditions(N, a0)
        ns_pi_sq = (np.arange(1, N + 1) * PI) ** 2
        F = rp.field_force_track_b(ic, N, a0, ns_pi_sq)
        F_trunc = rp.static_casimir_force_truncated(N, a0)
        np.testing.assert_allclose(F, F_trunc, rtol=1e-12)

    def test_force_positive_in_vacuum(self):
        """Truncated vacuum force is positive (repulsive before regularization)."""
        N = 16
        a0 = 1.0
        ic = mode_space.vacuum_initial_conditions(N, a0)
        ns_pi_sq = (np.arange(1, N + 1) * PI) ** 2
        F = rp.field_force_track_b(ic, N, a0, ns_pi_sq)
        assert F > 0

    def test_force_inverse_cube_scaling(self):
        """F ~ 1/a^3 for fixed mode amplitudes."""
        N = 8
        # Use arbitrary (not vacuum) mode state
        state = np.ones(4 * N) * 0.01
        ns_pi_sq = (np.arange(1, N + 1) * PI) ** 2
        F1 = rp.field_force_track_b(state, N, 1.0, ns_pi_sq)
        F2 = rp.field_force_track_b(state, N, 2.0, ns_pi_sq)
        np.testing.assert_allclose(F2, F1 / 8.0, rtol=1e-12)

    def test_from_fsq_matches(self):
        """field_force_track_b_from_fsq gives same result as full function."""
        N = 16
        a0 = 1.0
        ic = mode_space.vacuum_initial_conditions(N, a0)
        ns_pi_sq = (np.arange(1, N + 1) * PI) ** 2
        f_sq = mode_space.extract_mode_amplitudes_squared(ic, N)
        F1 = rp.field_force_track_b(ic, N, a0, ns_pi_sq)
        F2 = rp.field_force_track_b_from_fsq(f_sq, a0, ns_pi_sq)
        np.testing.assert_allclose(F1, F2, rtol=1e-14)
