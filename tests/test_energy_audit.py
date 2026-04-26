"""Tests for core/energy_audit.py — conservation monitoring."""

import numpy as np
import pytest

from nothing_engine.core.energy_audit import PhysicalIntegrityError, EnergyAuditor


class TestPhysicalIntegrityError:
    def test_error_message(self):
        err = PhysicalIntegrityError(
            absolute_drift=1.5e-3, tolerance=1e-6, timestep=42, t=3.14
        )
        assert "42" in str(err)
        assert "1.50e-03" in str(err)

    def test_attributes(self):
        err = PhysicalIntegrityError(1e-3, 1e-6, 10, 1.0)
        assert err.absolute_drift == 1e-3
        assert err.tolerance == 1e-6
        assert err.timestep == 10


class TestEnergyAuditor:
    def test_passes_conserved_system(self):
        """Constant energy passes audit."""
        auditor = EnergyAuditor(tolerance_factor=1e-6, halt_on_violation=True)
        auditor.set_reference(E_total_0=100.0, E_plate_0=5.0)

        for i in range(100):
            auditor.check(t=i * 0.1, E_plate=5.0, E_spring=0.0,
                         E_field=95.0, step=i)

        s = auditor.summary()
        assert s["passed"]
        assert s["max_drift"] < 1e-12

    def test_trips_on_violation(self):
        """Growing energy triggers error."""
        auditor = EnergyAuditor(tolerance_factor=1e-6, halt_on_violation=True)
        auditor.set_reference(E_total_0=100.0, E_plate_0=5.0)

        with pytest.raises(PhysicalIntegrityError):
            # Inject drift larger than 5e-6 (= 1e-6 * 5.0)
            auditor.check(t=1.0, E_plate=5.0, E_spring=0.0,
                         E_field=95.001, step=1)

    def test_no_halt_mode(self):
        """With halt_on_violation=False, records violation but doesn't raise."""
        auditor = EnergyAuditor(tolerance_factor=1e-6, halt_on_violation=False)
        auditor.set_reference(E_total_0=100.0, E_plate_0=5.0)

        record = auditor.check(t=1.0, E_plate=5.0, E_spring=0.0,
                              E_field=96.0, step=1)
        assert record.absolute_drift == 1.0
        assert not auditor.summary()["passed"]

    def test_records_stored(self):
        auditor = EnergyAuditor(tolerance_factor=1.0, halt_on_violation=False)
        auditor.set_reference(E_total_0=10.0, E_plate_0=1.0)

        for i in range(50):
            auditor.check(t=i * 0.01, E_plate=1.0, E_spring=0.0,
                         E_field=9.0, step=i)

        assert len(auditor.records) == 50

    def test_zero_plate_energy_guard(self):
        """Static plate (E_plate=0) doesn't cause division by zero."""
        auditor = EnergyAuditor(tolerance_factor=1e-6, halt_on_violation=True)
        auditor.set_reference(E_total_0=1000.0, E_plate_0=0.0)
        assert auditor.tolerance > 0  # guard uses 1e-20 floor

    def test_reference_required(self):
        auditor = EnergyAuditor()
        with pytest.raises(RuntimeError):
            auditor.check(t=0.0, E_plate=1.0, E_spring=0.0,
                         E_field=1.0, step=0)
