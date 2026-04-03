"""
Track C -- Energy audit and conservation framework.

Provides continuous verification of the First Law of Thermodynamics.
Monitors energy balance and halts simulation on violation.

Conservation identity (closed system):
    E_plate + E_spring + E_field = E_0 (constant)

CRITICAL: Energy tolerance is measured in ABSOLUTE units scaled to the
plate energy, NOT relative to total energy. The total energy is dominated
by the (divergent) vacuum zero-point energy sum ~ N^2, which dwarfs the
plate's kinetic energy. Relative tolerance against total energy would be
meaningless -- a 1e-6 relative drift on E_total ~ 50000 is ~0.05, which
is 10x larger than E_plate ~ 0.005 for typical parameters.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


class PhysicalIntegrityError(Exception):
    """Raised when energy conservation is violated beyond tolerance."""

    def __init__(self, absolute_drift: float, tolerance: float,
                 timestep: int, t: float):
        self.absolute_drift = absolute_drift
        self.tolerance = tolerance
        self.timestep = timestep
        self.t = t
        super().__init__(
            f"Energy conservation violated at step {timestep} (t={t:.6f}): "
            f"absolute drift {absolute_drift:.2e} > tolerance {tolerance:.2e}"
        )


@dataclass
class AuditRecord:
    """Single energy audit measurement."""
    t: float
    E_plate: float
    E_spring: float
    E_field: float
    E_total: float
    absolute_drift: float


class EnergyAuditor:
    """Monitors energy conservation throughout a simulation run.

    Tracks absolute drift |E(t) - E(0)| against a tolerance scaled
    to the plate's initial kinetic energy.

    Parameters
    ----------
    tolerance_factor : float
        Drift tolerance = tolerance_factor * E_plate_initial.
        Default 1e-6.
    halt_on_violation : bool
        If True, raises PhysicalIntegrityError on violation.
    """

    def __init__(self, tolerance_factor: float = 1e-6,
                 halt_on_violation: bool = True):
        self.tolerance_factor = tolerance_factor
        self.halt_on_violation = halt_on_violation
        self._E0: Optional[float] = None
        self._tolerance: Optional[float] = None
        self._records: list[AuditRecord] = []
        self._max_drift: float = 0.0
        self._step_count: int = 0

    def set_reference(self, E_total_0: float, E_plate_0: float):
        """Set the reference energy and compute tolerance.

        Must be called before any check() calls.

        Parameters
        ----------
        E_total_0 : float
            Total energy at t=0.
        E_plate_0 : float
            Plate kinetic energy at t=0 (for tolerance scaling).
        """
        self._E0 = E_total_0
        # Guard against zero plate energy (static plate tests)
        scale = max(abs(E_plate_0), 1e-20)
        self._tolerance = self.tolerance_factor * scale

    def check(self, t: float, E_plate: float, E_spring: float,
              E_field: float, step: int) -> AuditRecord:
        """Check energy conservation at a single time point.

        Parameters
        ----------
        t : float
            Current time.
        E_plate, E_spring, E_field : float
            Current energy components.
        step : int
            Current integration step number.

        Returns
        -------
        AuditRecord

        Raises
        ------
        PhysicalIntegrityError
            If drift exceeds tolerance and halt_on_violation is True.
        """
        if self._E0 is None:
            raise RuntimeError("EnergyAuditor.set_reference() not called")

        E_total = E_plate + E_spring + E_field
        drift = abs(E_total - self._E0)

        record = AuditRecord(
            t=t, E_plate=E_plate, E_spring=E_spring,
            E_field=E_field, E_total=E_total, absolute_drift=drift,
        )
        self._records.append(record)
        self._step_count += 1

        if drift > self._max_drift:
            self._max_drift = drift

        if self.halt_on_violation and drift > self._tolerance:
            raise PhysicalIntegrityError(
                absolute_drift=drift,
                tolerance=self._tolerance,
                timestep=step,
                t=t,
            )

        return record

    @property
    def max_drift(self) -> float:
        """Maximum absolute energy drift observed."""
        return self._max_drift

    @property
    def tolerance(self) -> Optional[float]:
        """Current absolute tolerance."""
        return self._tolerance

    @property
    def records(self) -> list[AuditRecord]:
        """All audit records."""
        return self._records

    def summary(self) -> dict:
        """Return summary statistics of the audit."""
        if not self._records:
            return {"n_checks": 0}

        drifts = [r.absolute_drift for r in self._records]
        return {
            "n_checks": len(self._records),
            "max_drift": self._max_drift,
            "mean_drift": float(np.mean(drifts)),
            "tolerance": self._tolerance,
            "passed": self._max_drift <= (self._tolerance or float('inf')),
        }
