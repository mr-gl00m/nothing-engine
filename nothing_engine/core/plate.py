"""
DynamicalPlate — the mechanically free oscillating plate.

Tracks position q(t), velocity q̇(t), and applies forces.
The plate is the right boundary of the Casimir cavity.
The left plate is fixed at x_L.

Equation of motion:
    M q̈(t) = -k·q(t) + F_field(t)

where F_field comes from radiation_pressure.py (Track A or Track B).
"""


class DynamicalPlate:
    """A 1D plate with mass, spring restoring force, and field coupling.

    Parameters
    ----------
    mass : float
        Plate mass M (natural units).
    spring_k : float
        Spring constant k. Set to 0 for a free plate.
    q0 : float
        Initial position (equilibrium cavity width a₀).
    v0 : float
        Initial velocity.
    x_left : float
        Position of the fixed left plate.
    """

    def __init__(self, mass: float, spring_k: float, q0: float, v0: float,
                 x_left: float = 0.0):
        self.mass = mass
        self.spring_k = spring_k
        self.q = q0
        self.v = v0
        self.x_left = x_left

        # Equilibrium position (for spring force)
        self.q_eq = q0

    @property
    def cavity_width(self) -> float:
        """Current cavity width: q - x_L."""
        return self.q - self.x_left

    @property
    def kinetic_energy(self) -> float:
        """E_plate = ½Mv²."""
        return 0.5 * self.mass * self.v**2

    @property
    def spring_energy(self) -> float:
        """E_spring = ½k(q - q_eq)²."""
        return 0.5 * self.spring_k * (self.q - self.q_eq)**2

    def spring_force(self) -> float:
        """Restoring force: -k(q - q_eq)."""
        return -self.spring_k * (self.q - self.q_eq)

    def acceleration(self, field_force: float) -> float:
        """Compute q̈ = (F_spring + F_field) / M."""
        return (self.spring_force() + field_force) / self.mass

    def get_state(self) -> dict:
        """Return current state as a dictionary for checkpointing."""
        return {
            "q": self.q,
            "v": self.v,
            "mass": self.mass,
            "spring_k": self.spring_k,
            "q_eq": self.q_eq,
            "x_left": self.x_left,
        }

    def set_state(self, state: dict):
        """Restore state from checkpoint dictionary."""
        self.q = state["q"]
        self.v = state["v"]
