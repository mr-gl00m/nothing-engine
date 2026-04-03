"""
Stress-energy flux T₀₁ integration at boundaries.

For open systems, radiated energy is measured via direct
time-integration of the stress-energy flux:

    E_radiated(t) = ∫₀ᵗ T₀₁(x_boundary, t') dt'

where T₀₁(x,t) = -∂ₜφ · ∂ₓφ

This mirrors the Poynting vector approach and maintains
machine-precision accuracy for the energy audit.

IMPORTANT: Do NOT compute radiated energy by summing field
energy in the outer region.
"""
