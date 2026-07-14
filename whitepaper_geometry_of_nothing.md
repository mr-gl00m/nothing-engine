# Geometry of Nothing

## Revised Analysis and Postmortem of a Toroidal Casimir Harvester

Nathan Seals  
[GitHub](https://github.com/mr-gl00m) | [Website](https://www.nathanseals.com)

Revision date: 2026-07-13

## Abstract

This paper revisits a proposed toroidal Casimir energy harvester and the
Nothing Engine simulator built to test it. The original concept combined
Casimir attraction, fluid mediated repulsion, a phase change switch, and a
closed halo geometry. A later variant considered spiral patterning inspired by
sunflower phyllotaxis. The intended result was a long lived cyclic source whose
working body returned to its initial state after each stroke.

The revised result is decisive. An equilibrium ground state or thermal state is
passive. For a quasistatic cyclic change of geometry and material parameters,
the Casimir force is the derivative of a free energy, so its closed cycle work
is zero. A rapid boundary change can create dynamical Casimir excitations, while
the drive supplies their energy. Material switching adds a free energy and
dissipation cost. Geometry, including a torus or spiral, changes the interaction
landscape without creating an energy source.

A generous ideal plate calculation gives $3.03\times10^{-18}$ J of Casimir work
for a $1\ \mu\mathrm{m}^2$ element closing from 100 nm to 50 nm. Heating and
transforming a 10 nm VO2 film over the same area requires about
$1.89\times10^{-12}$ J under representative bulk assumptions. The switching
budget is roughly $6.2\times10^5$ times the ideal mechanical work before
piezoelectric, thermal, fluid, and control losses.

The software audit also retracts the original quantitative claims about vacuum
friction and topology. Version 0.4 restores the analytic static Casimir energy
and force, includes periodic mode degeneracy, fixes production validation, and
labels the solver accurately as a diagonal parametric mode approximation. It
does not include the velocity dependent intermode coupling required for a full
moving cavity calculation.

Three research directions remain banked: halo geometry as a passive force
shaping platform, fluid mediated repulsive Casimir devices, and a
nonequilibrium concentric cylinder heat engine powered by a temperature
gradient. The equilibrium vacuum harvester is closed.

## 1. The original question

The project asked whether two Casimir surfaces could be arranged as a ring so
that a useful force continued around a closed path. The first proposed cycle
used four ingredients:

1. attractive Casimir force to perform a mechanical stroke;
2. a phase change material to alter the optical response;
3. a dielectric fluid to permit a repulsive Lifshitz regime;
4. a toroidal geometry to remove the edges and alignment problems of flat
   plates.

The target was long lasting energy rather than an infinite power claim. This
distinction changes engineering expectations, though it leaves the thermodynamic
test unchanged. A device that repeats a cycle and returns every controlled
degree of freedom to its initial state is an engine. Its net work must be traced
to a change in stored free energy or to heat, chemical potential, radiation,
external drive, or another reservoir.

The spiral variant attempted to use a natural packing pattern as a directional
coupling geometry. Spiral and chiral structures can produce Casimir torques.
They still follow an equilibrium energy landscape unless an external resource
maintains nonequilibrium conditions.

## 2. The thermodynamic result

Let $\lambda$ collect the controllable geometry and material parameters. At
temperature $T$, the equilibrium field and matter system has Helmholtz free
energy

$$
\mathcal{F}(\lambda,T)=-k_B T\ln Z(\lambda,T).
$$

The generalized equilibrium force is

$$
X(\lambda,T)=-\frac{\partial \mathcal{F}}{\partial\lambda}.
$$

For a closed isothermal quasistatic path $C$,

$$
W_{\mathrm{out}}=\oint_C X\,d\lambda
=-\oint_C d\mathcal{F}=0.
$$

At zero temperature, $\mathcal{F}$ approaches the ground state energy and the
same result applies. Dissipation, hysteresis, finite rate heat flow, fluid drag,
and imperfect conversion make the practical cycle work negative unless an
external reservoir supplies energy.

This is stronger than an unfavorable component estimate. Pusz and Woronowicz
proved that ground states and thermal equilibrium states are completely
passive. No cyclic unitary operation can extract positive work from such a
state, including from any number of copies. The proposed halo changes the
Hamiltonian through geometry and materials. Returning the device and field to
their initial states returns the free energy as well.

### 2.1 One stroke can release energy

Casimir attraction can perform work while two surfaces close. That work is a
decrease in interaction energy. Restoring the initial separation costs at least
the same amount in a reversible idealization. Real reset work is larger.

This permits a one shot actuator or a stored energy element. It does not produce
a self resetting source. Long retention would describe storage lifetime, and
the charging source would remain part of the ledger.

### 2.2 Switching moves the cost

Changing a dielectric function, phase, temperature, carrier population, or
magnetic bias changes the Hamiltonian. The operation therefore performs work or
exchanges heat. The resulting Casimir force change can be useful as a control
mechanism. It cannot be entered as a free switch in the cycle.

Experiments with phase change materials have measured controllable Casimir force
variation of about 20 percent near 100 nm. That establishes force modulation,
not a zero cost sign reversal. Any proposed switching cycle needs the full
enthalpy, hysteresis, heat leakage, optical or electrical drive, and recovery
cost.

### 2.3 Dynamical Casimir radiation has a source

A time dependent boundary can parametrically excite the field. In a physical
dynamical Casimir experiment, the boundary drive or modulation source loses the
energy carried by the created photons. Back reaction appears as an additional
load on that source. The effect converts supplied mechanical or electromagnetic
work into field excitations.

The dynamical Casimir effect is therefore relevant to transduction, squeezing,
and radiation from driven boundaries. It supplies no exception to passivity.

## 3. Quantitative audit of the phase change cycle

Consider ideal perfectly conducting parallel plates with area $A$ and separation
$d$. Their zero temperature interaction energy is

$$
U_C(d)=-\frac{\pi^2\hbar c}{720}\frac{A}{d^3}.
$$

The largest reversible work available while closing from $d_2=100$ nm to
$d_1=50$ nm is

$$
\begin{aligned}
W_C
&=\frac{\pi^2\hbar cA}{720}
\left(\frac{1}{d_1^3}-\frac{1}{d_2^3}\right) \\
&=3.0336\times10^{-18}\ \mathrm{J}
\end{aligned}
$$

for $A=1\ \mu\mathrm{m}^2$. This is a generous upper benchmark because real
materials, roughness, finite temperature, fluid separation, and incomplete
force modulation reduce the available difference.

For a 10 nm VO2 film over that area,

$$
V=At=10^{-20}\ \mathrm{m}^3,
$$

and a representative density $\rho=4570\ \mathrm{kg\,m^{-3}}$ gives

$$
m=\rho V=4.57\times10^{-17}\ \mathrm{kg}.
$$

Using a baseline specific heat $c_p=700\ \mathrm{J\,kg^{-1}\,K^{-1}}$ and a
5 K temperature excursion,

$$
Q_{\mathrm{sensible}}=mc_p\Delta T
=1.60\times10^{-13}\ \mathrm{J}.
$$

An early calorimetric measurement reported a VO2 transition heat near
750 cal/mol, equivalent to $37.8\ \mathrm{kJ\,kg^{-1}}$. That gives

$$
Q_{\mathrm{latent}}=mL
=1.73\times10^{-12}\ \mathrm{J}.
$$

The gross heating input is then

$$
Q_{\mathrm{switch}}
=Q_{\mathrm{sensible}}+Q_{\mathrm{latent}}
=1.89\times10^{-12}\ \mathrm{J},
$$

and

$$
\frac{Q_{\mathrm{switch}}}{W_C}=6.23\times10^5.
$$

This enthalpy is gross thermal throughput rather than an irreducible loss. An
ideal regenerator could return some heat from the cooling stroke to the next
heating stroke. To clear the ideal Casimir work scale, the unrecovered fraction
would need to stay below $1.61\times10^{-6}$ before piezoelectric, conduction,
fluid, control, and hysteresis losses. Regeneration also leaves the passivity
result unchanged.

Thin film values depend on stoichiometry, strain, grain structure, pulse length,
and hysteresis. Sensible heating alone exceeds the ideal Casimir work by about
$5.3\times10^4$. A partial force modulation widens the useful work deficit.

The old calculation used a volume comparable to $(10\ \mathrm{nm})^3$ while
assigning Casimir work to $1\ \mu\mathrm{m}^2$ of plate area. The corrected film
volume is area times thickness. The old estimate also omitted the transition
enthalpy.

## 4. Why the ring and spiral do not close the ledger

### 4.1 Toroidal closure

Closing a cavity into a ring changes the allowed spectrum and spatial mode
structure. It can alter force magnitude, stability, and coupling to an
actuator. These changes are legitimate design variables.

The free energy remains single valued for an equilibrium configuration. A full
trip around a cyclic parameter path has zero reversible work. Calling the path
a halo does not remove the reset segment.

### 4.2 Spiral patterning

Anisotropic and patterned objects can experience an equilibrium Casimir torque,

$$
\tau(\theta)=-\frac{\partial\mathcal{F}}{\partial\theta}.
$$

The object rotates toward a free energy minimum and then stops. A spiral may
increase torque, select handedness, or couple translation to rotation. Sustained
rotation needs a drive or a nonequilibrium flux.

### 4.3 Stability is a separate calculation

Rotational symmetry does not prove self centering. Stable levitation requires
the Hessian of the total interaction energy to have the appropriate signs in
every uncontrolled translational and rotational coordinate. General constraints
exclude broad classes of stable vacuum Casimir equilibria for ordinary
materials. An intervening fluid or nonequilibrium force can change the allowed
regime, and each candidate still needs a full stiffness calculation.

## 5. Software postmortem

The original Nothing Engine was presented as a self consistent moving boundary
quantum field solver. The implementation solved a smaller model:

$$
\ddot f_n+\omega_n^2(q)f_n=0
$$

for independent instantaneous oscillators, with one mechanical coordinate
coupled through the derivative of their oscillator energies.

That reduced model can demonstrate parametric resonance and test a conservative
energy ledger. A physical moving cavity expansion also produces velocity
dependent couplings between instantaneous modes. Law's effective Hamiltonian
contains these moving basis terms. Their absence removes intermode scattering
and prevents quantitative identification of the simulated occupation with the
full dynamical Casimir output of a moving mirror.

### 5.1 Retracted software claims

The following earlier claims are withdrawn:

- the code solved the full self consistent moving cavity quantum field theory;
- the ringdown curves measured physical vacuum friction;
- closed and periodic runs isolated a topology advantage or disadvantage;
- the large box run represented an open radiating boundary;
- the original static validation independently confirmed the implementation.

The reasons are concrete:

1. Vacuum subtraction removed the finite oscillator zero point force without
   restoring the analytic static Casimir force. The initialized vacuum therefore
   exerted zero force in the ODE.
2. The old static gate compared an analytic formula with the same formula. It did
   not inspect the production energy or force path.
3. A real periodic scalar field has sine and cosine modes for every positive
   mode number. The original accounting omitted that factor of two.
4. Equal closed interval length and periodic circumference do not give matched
   spectra. Matching $a$ with $L=2a$ gives equal frequencies, while the same
   absolute coordinate velocity gives different fractional drive rates.
5. The large box script retained reflecting walls and had no outgoing boundary,
   continuum, or flux observable.
6. A smooth high mode weight was described as finite plate thickness. A physical
   partially transmitting mirror requires frequency dependent reflection data
   and an open scattering model.

The old quantitative friction exponents, depletion ratios, and claimed
topological rankings have no surviving physical interpretation.

## 6. Corrected Nothing Engine model

Version 0.4 describes its scope as a finite dimensional diagonal mode
approximation. For boundary type $b$, define

$$
\kappa_n^{(\mathrm{closed})}=n\pi,
\qquad
\kappa_n^{(\mathrm{periodic})}=2n\pi,
$$

with degeneracy

$$
d_{\mathrm{closed}}=1,
\qquad
d_{\mathrm{periodic}}=2.
$$

The oscillator frequency is

$$
\omega_n^2(q)=g_n\frac{\kappa_n^2}{q^2},
$$

where $g_n=1$ by default. A user selected $g_n$ is a phenomenological spectral
weight for sensitivity studies.

The field energy reported by the engine is

$$
E_{\mathrm{field}}
=d_b\sum_{n=1}^{N}
\frac{1}{2}
\left(
|\dot f_n|^2+\omega_n^2|f_n|^2-\omega_n
\right)
+E_C^{(b)}(q).
$$

The first term is excitation energy above the finite instantaneous oscillator
vacuum. The restored analytic static terms are

$$
E_C^{(\mathrm{closed})}(a)=-\frac{\pi}{24a},
\qquad
E_C^{(\mathrm{periodic})}(L)=-\frac{\pi}{6L}.
$$

The matching force on the mechanical coordinate is

$$
F_{\mathrm{field}}
=d_b\sum_{n=1}^{N}
\left[
g_n\frac{\kappa_n^2}{q^3}|f_n|^2
-\frac{\kappa_n\sqrt{g_n}}{2q^2}
\right]
+F_C^{(b)}(q),
$$

with

$$
F_C^{(\mathrm{closed})}(a)=-\frac{\pi}{24a^2},
\qquad
F_C^{(\mathrm{periodic})}(L)=-\frac{\pi}{6L^2}.
$$

The continuous reduced equations conserve

$$
E_{\mathrm{total}}
=\frac{1}{2}M\dot q^2
+\frac{1}{2}k(q-q_{\mathrm{eq}})^2
+E_{\mathrm{field}}.
$$

The production static validation now initializes the actual finite mode state,
calls the production energy function, calls the production ODE force, and checks
both supported boundaries across 8 to 512 modes. The unit suite checks static
energy, static force, periodic degeneracy, vacuum occupation, Wronskian
preservation, conservative evolution, checkpoint reconstruction, and HDF5
observables.

### 6.1 Valid uses

- testing mode by mode vacuum subtraction and regularized static bookkeeping;
- studying independent parametric oscillator response;
- checking numerical conservation and adiabatic behavior;
- producing regression cases for a future multimode implementation;
- demonstrating why a conservative closed model cannot generate net energy.

### 6.2 Invalid uses

- predicting a laboratory dynamical Casimir photon spectrum from a moving wall;
- extracting a physical vacuum friction coefficient;
- modeling a partially transmitting plate from thickness alone;
- claiming an open boundary from a larger finite box;
- treating the periodic coordinate as a fabricated toroidal cavity without a
  three dimensional electromagnetic model;
- using scalar 1+1D outputs as quantitative 3+1D device performance.

## 7. Research directions that survive

### 7.1 Halo geometry as a passive component

The halo remains worth studying as a geometry for force shaping, alignment, and
torque coupling. Its energy must come from an actuator, a charged storage
element, a temperature gradient, or another named source. Full electromagnetic
scattering calculations and a stability matrix are required.

### 7.2 Fluid mediated repulsive devices

Repulsive Casimir-Lifshitz forces across liquids have experimental support.
Possible applications include passive spacing, contact avoidance, and low load
bearings. The decisive engineering quantities are stable load capacity and
fluid drag. Candidate screening needs $\varepsilon(i\xi)$ across the dominant
frequency range, plus electrostatic, double layer, roughness, contamination, and
nanoconfinement effects.

### 7.3 Concentric cylinder heat engine

A June 2026 preprint describes two concentric cylinders at different
temperatures. Repulsive nonequilibrium Casimir forces levitate the inner
cylinder. Nonreciprocal media break the cancellation of opposite angular
momentum channels and generate torque. Heat flow supplies the energy, fluctuation
induced friction limits the speed, and efficiency remains below Carnot.

This proposal preserves the contactless cylinder and fluctuation induced torque
ideas within a valid thermodynamic engine. The immediate task is independent
reproduction of its force, torque, heat flux, friction, steady speed, and
efficiency before adding a toroidal closure or spiral pattern.

Detailed research gates and kill criteria are recorded in
[BANKED_CASIMIR_DIRECTIONS.md](BANKED_CASIMIR_DIRECTIONS.md).

## 8. Final disposition

The original equilibrium vacuum harvester fails for two independent reasons.
The thermodynamic cycle has zero reversible net work, and the proposed phase
change implementation handles gross thermal energy five to six orders of
magnitude above the generous ideal plate work scale. Its required recovery
fraction leaves no credible margin for the known losses.

No rearrangement into a ring, halo, sunflower spiral, or other equilibrium
geometry changes the closed cycle free energy result. A future device can use
Casimir forces as springs, bearings, switches, torque couplers, or working
interactions. It must also identify a fuel, drive, heat gradient, or stored
energy source.

The correct postmortem is therefore specific. The equilibrium Casimir energy
source is closed. The geometry, fluid mediated force control, and
nonequilibrium cylinder engine remain legitimate research material.

## References

1. H. B. G. Casimir, [On the attraction between two perfectly conducting plates](https://doi.org/10.1016/S0031-8914(48)80072-5), Proceedings of the Royal Netherlands Academy of Arts and Sciences 51, 793 to 795 (1948).
2. W. Pusz and S. L. Woronowicz, [Passive states and KMS states for general quantum systems](https://doi.org/10.1007/BF01614224), Communications in Mathematical Physics 58, 273 to 290 (1978).
3. G. T. Moore, [Quantum theory of the electromagnetic field in a variable-length one-dimensional cavity](https://doi.org/10.1063/1.1665432), Journal of Mathematical Physics 11, 2679 to 2691 (1970).
4. C. K. Law, [Effective Hamiltonian for the radiation in a cavity with a moving mirror and a time-varying dielectric medium](https://doi.org/10.1103/PhysRevA.49.433), Physical Review A 49, 433 to 437 (1994).
5. C. M. Wilson et al., [Observation of the dynamical Casimir effect in a superconducting circuit](https://doi.org/10.1038/nature10561), Nature 479, 376 to 379 (2011).
6. J. N. Munday, F. Capasso, and V. A. Parsegian, [Measured long-range repulsive Casimir-Lifshitz forces](https://doi.org/10.1038/nature07610), Nature 457, 170 to 173 (2009).
7. S. J. Rahi, M. Kardar, and T. Emig, [Constraints on stable equilibria with fluctuation-induced forces](https://arxiv.org/abs/0911.5364), Physical Review Letters 105, 070404 (2010).
8. T. Emig, N. Graham, R. L. Jaffe, and M. Kardar, [Casimir forces between arbitrary compact objects](https://arxiv.org/abs/0707.1862), Physical Review Letters 99, 170403 (2007).
9. G. Torricelli et al., [Switching Casimir forces with phase change materials](https://arxiv.org/abs/1006.4065), Physical Review A 82, 010101 (2010).
10. T. Kawakubo and T. Nakagawa, [Phase transition in VO2](https://doi.org/10.1143/JPSJ.19.517), Journal of the Physical Society of Japan 19, 517 to 519 (1964).
11. D. Shah et al., [A contactless heat engine driven by nonreciprocal fluctuation-induced torques](https://arxiv.org/abs/2606.25053), arXiv:2606.25053 (2026).
