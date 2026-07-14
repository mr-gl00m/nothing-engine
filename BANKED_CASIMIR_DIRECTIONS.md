# Banked Casimir Research Directions

Status: 2026-07-13

This document preserves the three project branches that remain scientifically
interesting after the equilibrium vacuum harvester was closed. Each branch must
name its free energy source, use a material model on the imaginary frequency
axis, and pass an energy ledger before any claim about power or storage.

| Branch | Legitimate role of Casimir physics | Energy source | Current status |
| --- | --- | --- | --- |
| Halo geometry | Passive alignment, force shaping, or torque coupling | External drive, stored rotor energy, or a temperature gradient | Concept and modeling program |
| Fluid mediated device | Repulsive support, low load bearing, or tunable equilibrium | External actuator or stored mechanical energy | Experimentally grounded, engineering unresolved |
| Concentric cylinder heat engine | Contactless levitation, angular momentum transfer, and friction | Heat flow between maintained temperatures | Recent first principles proposal |

The equilibrium vacuum alone is excluded as a cyclic energy source. A closed
cycle returns the field and device to the same state, so the available free
energy change is zero. Geometry can change forces and equilibria. It cannot
change that ledger.

## 1. Halo and spiral geometry

### Preserved idea

A toroidal or concentric geometry may reduce alignment sensitivity and provide a
useful platform for distributed Casimir forces. A spiral or chiral pattern may
convert a normal fluctuation induced force into a torque when another resource
drives the system away from equilibrium.

The strongest surviving version of the halo idea is a passive mechanical
component:

1. A concentric annulus supplies radial force shaping.
2. Fluid mediated repulsion prevents direct contact over a designed gap range.
3. Patterning supplies stiffness, torque coupling, or a spectral response.
4. An external actuator, thermal gradient, or initially energized rotor supplies
   the work.

### Corrections to the original intuition

Rotational symmetry does not guarantee self centering. A displacement changes
the near and far gaps unequally, and the sign of the resulting stiffness depends
on the full material and geometry response. Stability requires a positive
mechanical stiffness matrix at the proposed operating point.

A spiral does not create a one way equilibrium cycle. An equilibrium Casimir
torque follows the derivative of a free energy and relaxes toward an angular
minimum. Sustained rotation requires time dependent driving, multiple
temperatures, nonreciprocity, or another nonequilibrium resource.

### Required model

The next calculation should use a scattering formulation or boundary element
solver for the actual three dimensional geometry. Pairwise plate formulas and
the proximity force approximation can screen candidates, though they cannot
establish stability for the final device.

Minimum outputs:

- radial force and torque as functions of displacement and angle;
- the full translational and rotational stiffness matrix;
- sensitivity to roughness, eccentricity, finite thickness, and temperature;
- a separate actuator or thermal power ledger;
- a comparison against electrostatic, magnetic, and hydrodynamic alternatives.

### Kill criteria

Bank this branch again if any of the following holds:

- no stable operating point survives the full stiffness calculation;
- fabrication tolerance consumes more than half the stable gap;
- conventional actuation produces the same force or torque with lower loss and
  lower fabrication risk;
- the proposed cycle has no external free energy term.

## 2. Fluid mediated Casimir devices

### Preserved idea

Repulsive Casimir-Lifshitz forces have been measured between suitable materials
across a liquid. This supports a serious research path for contact avoidance,
passive spacing, and very low load bearings. It does not establish a zero drag
bearing or an energy source.

For planar materials 1 and 2 separated by medium m, the familiar ordering

$$
\varepsilon_1(i\xi) > \varepsilon_m(i\xi) > \varepsilon_2(i\xi)
$$

is a useful screening rule when it holds across the frequencies that dominate
the Lifshitz sum. Static dielectric constants alone are insufficient. The
calculation needs the full functions $\varepsilon(i\xi)$, including temperature,
phase, and uncertainty.

### Main engineering question

The useful figure is support per unit drag. A liquid that enables repulsion also
introduces viscous shear, squeeze film damping, contamination risk, and altered
rheology under nanoconfinement. A flywheel claim therefore needs measured
rotational loss at the intended gap and surface speed.

The first credible device should be a static or slowly moving demonstrator:

1. Measure force versus gap for one material-fluid stack.
2. Map lateral and angular stability.
3. Measure approach and withdrawal hysteresis.
4. Measure rotational or translational drag in the same fabricated gap.
5. Compare lifetime and standby loss with a conventional bearing of equal size
   and load.

### Material screening package

For every candidate stack, archive:

- tabulated optical data and the transform to $\varepsilon(i\xi)$;
- the Lifshitz force with propagated optical uncertainty;
- double layer, capillary, patch potential, and van der Waals backgrounds;
- fluid viscosity versus temperature and confinement;
- roughness and adsorbed layer measurements;
- chemical compatibility and aging data.

### Kill criteria

- repulsion disappears within optical data uncertainty;
- electrostatic or double layer backgrounds cannot be separated from the target
  force;
- confined fluid drag removes the claimed storage or bearing advantage;
- the equilibrium is unstable in any uncontrolled degree of freedom;
- surface aging changes the force sign or stable gap on the test timescale.

## 3. Concentric cylinder Casimir heat engine

### Preserved idea

Shah, Asheichyk, Gelbwaser-Klimovsky, Graham, Kardar, and Kruger proposed a
contactless engine with two concentric cylinders held at different temperatures.
Repulsive nonequilibrium Casimir forces levitate the inner cylinder.
Nonreciprocal dielectric response breaks the cancellation between positive and
negative angular momentum channels, producing torque. Fluctuation induced
friction sets a steady rotation rate, and efficiency remains bounded by Carnot.

This is the closest rigorous descendant of the halo concept. Its fuel is heat
flow between maintained reservoirs. The Casimir field is the working medium and
contactless coupling mechanism.

### Immediate reproduction target

Reproduce the paper before extending the geometry:

1. Rebuild the angular momentum resolved heat flux $\Phi_n(\omega)$.
2. Verify cancellation of torque in the reciprocal limit.
3. Reproduce levitation force, driving torque, friction torque, steady angular
   speed, heat flow, output power, and efficiency.
4. Sweep cylinder radii, gap, temperature difference, loss, and nonreciprocal
   strength.
5. Check every result against energy conservation and the Carnot bound.

Only after that reproduction should the project test a toroidal closure, spiral
patterning, or alternative nonreciprocal media.

### Main risks

- the proposal is a recent preprint and has no experimental demonstration yet;
- nonreciprocal material response may require an applied magnetic field with a
  substantial system power cost;
- near field heat transfer may dominate the thermal design;
- fabrication tolerances may spoil concentric stability;
- useful torque and power may remain far below readout and bearing losses.

### Kill criteria

- the reciprocal limit retains torque, indicating an implementation error;
- total mechanical output exceeds absorbed heat or violates the Carnot bound;
- required bias, heating, cooling, and readout power overwhelm gross output;
- material loss removes levitation or torque in the realizable parameter range;
- no parameter set clears a measurable signal threshold with fabrication
  uncertainty included.

## 4. Shared research sequence

| Gate | Required result | Decision |
| --- | --- | --- |
| 0. Resource ledger | Every state change names its work, heat, chemical, or bias source | Reject cycles with an unnamed source |
| 1. Material response | Reproducible $\varepsilon(i\xi)$ and uncertainty | Reject sign claims based on static permittivity |
| 2. Static mechanics | Force, torque, full stiffness matrix, and parasitic backgrounds | Reject unstable operating points |
| 3. Dynamics | Drag, noise, dissipation, actuator cost, and thermal load | Reject negative net utility |
| 4. Device | Measurable output with all support systems counted | Advance to fabrication |

## 5. Boundary of the bank

The following claim is closed and should stay closed: an equilibrium Casimir
device that returns its geometry, material state, temperature, and field state
to their initial values can supply positive net work indefinitely.

Reopen that claim only with a derivation that identifies a failure in passivity
or the free energy cycle, plus an independently reproducible experiment whose
complete energy ledger excludes hidden drive, heat flow, chemical change, and
stored mechanical energy.

## References

1. J. N. Munday, F. Capasso, and V. A. Parsegian, [Measured long-range repulsive Casimir-Lifshitz forces](https://doi.org/10.1038/nature07610), Nature 457, 170 to 173 (2009).
2. S. J. Rahi, M. Kardar, and T. Emig, [Constraints on stable equilibria with fluctuation-induced forces](https://arxiv.org/abs/0911.5364), Physical Review Letters 105, 070404 (2010).
3. T. Emig, N. Graham, R. L. Jaffe, and M. Kardar, [Casimir forces between arbitrary compact objects](https://arxiv.org/abs/0707.1862), Physical Review Letters 99, 170403 (2007).
4. G. Torricelli et al., [Switching Casimir forces with phase change materials](https://arxiv.org/abs/1006.4065), Physical Review A 82, 010101 (2010).
5. D. Shah et al., [A contactless heat engine driven by nonreciprocal fluctuation-induced torques](https://arxiv.org/abs/2606.25053), arXiv:2606.25053 (2026).
