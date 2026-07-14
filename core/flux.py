"""Design note for a future open-boundary flux implementation.

No production code in this module computes flux, and the current solver has no
open boundary. A future scattering or absorbing-boundary model should measure
radiated energy by direct time integration of the stress-energy flux:

    E_radiated(t) = ∫₀ᵗ T₀₁(x_boundary, t') dt'

where T₀₁(x,t) = -∂ₜφ · ∂ₓφ

Computing this quantity requires spatial field derivatives that the current
diagonal oscillator state does not provide.
"""
