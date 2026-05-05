"""
Topology comparison v2: addresses methodological concerns.

Fixes from v1:
1. Frequency matching: periodic ring uses L=2*a so fundamental
   frequencies match (omega_1 = pi/a for both topologies).
2. Spring restoring force (k>0) keeps plate near equilibrium,
   preventing enormous displacement and catastrophic cancellation.
3. Sigmoid form factor for sharper UV cutoff.
4. Reports N_total (particle number) as clean observable.
5. Runs at N=64 and N=128 to check ratio convergence.

Usage:
    python -m nothing_engine.experiments.run_topology_v2 [--quick]
"""

import sys
import logging
from pathlib import Path

import numpy as np
import h5py

from nothing_engine.core.bogoliubov import (
    SimulationConfig, PrecomputedArrays, build_initial_state,
)
from nothing_engine.core import mode_space, energy as energy_mod
from nothing_engine.experiments.runner import ExperimentRunner, RunConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def make_configs(n_modes: int, total_time: float, tag: str):
    """Build matched closed/periodic configs.

    For fair frequency matching:
      closed:   a = q0 = 1.0, omega_n = n*pi/a
      periodic: L = q0 = 2.0, omega_n = 2*n*pi/L = n*pi/1 = n*pi/a
    So the mode spectra are identical.

    Spring k chosen so oscillation period ~ 100 time units:
      omega_osc = sqrt(k/M) = 2*pi/T => k = M*(2*pi/T)^2
      T=100, M=100 => k = 100*(2*pi/100)^2 ≈ 0.3948
    """
    mass = 100.0
    v0 = 0.01
    T_osc = 100.0
    spring_k = mass * (2.0 * np.pi / T_osc) ** 2

    run_cfg = RunConfig(
        total_time=total_time,
        segment_time=min(500.0, total_time / 5),
        samples_per_unit_time=8,
        checkpoint_interval=max(total_time / 5, 500.0),
        log_interval_segments=2,
    )

    configs = []
    for boundary in ["closed", "periodic"]:
        # Frequency matching: periodic gets L=2*a
        if boundary == "periodic":
            q0 = 2.0  # L = 2, so omega_n = 2*n*pi/2 = n*pi = same as closed
        else:
            q0 = 1.0  # a = 1, omega_n = n*pi/1 = n*pi

        sim_cfg = SimulationConfig(
            n_modes=n_modes,
            plate_mass=mass,
            spring_k=spring_k,
            q0=q0,
            v0=v0,
            x_left=0.0,
            boundary=boundary,
            cutoff_shape="sigmoid",
            rtol=1.0e-10,
            atol=1.0e-12,
            max_step=0.01,
        )
        output_path = f"data/experiments/topov2_{boundary}_N{n_modes}_{tag}.h5"
        configs.append((boundary, sim_cfg, run_cfg, output_path))

    return configs


def analyze_run(path: str, sim_cfg: SimulationConfig, label: str):
    """Extract key observables from a completed run."""
    with h5py.File(path, 'r') as f:
        t = f['timeseries']['t'][:]
        e_plate = f['timeseries']['E_plate'][:]
        e_spring = f['timeseries']['E_spring'][:]
        e_field = f['timeseries']['E_field'][:]
        e_total = f['timeseries']['E_total'][:]
        n_part = f['timeseries']['total_particles'][:]
        q_arr = f['timeseries']['plate_q'][:]
        v_arr = f['timeseries']['plate_v'][:]
        ckpts = sorted(f['checkpoints'].keys())
        state = f['checkpoints'][ckpts[-1]]['state'][:]

    pre = PrecomputedArrays.from_config(sim_cfg)
    idx = 4 * sim_cfg.n_modes
    ms_final = state[:idx]
    w = mode_space.wronskian(ms_final, sim_cfg.n_modes)
    w_err = float(np.max(np.abs(w + 0.5)))

    # Particle number from checkpoint (more reliable than timeseries)
    q_final = state[idx]
    a_final = q_final - sim_cfg.x_left
    pn_final = mode_space.particle_number(ms_final, sim_cfg.n_modes, a_final,
                                           pre.g_n, pre.ns_pi)
    n_total_final = float(np.sum(pn_final))
    n_positive = float(np.sum(np.maximum(pn_final, 0)))

    return {
        'label': label,
        'E_plate_0': e_plate[0],
        'E_plate_f': e_plate[-1],
        'depletion': 1.0 - e_plate[-1] / e_plate[0],
        'E_spring_f': e_spring[-1],
        'E_field_f': e_field[-1],
        'E_total_0': e_total[0],
        'E_total_f': e_total[-1],
        'E_conserve': abs(e_total[-1] - e_total[0]) / abs(e_total[0]),
        'q_range': (float(np.min(q_arr)), float(np.max(q_arr))),
        'N_total': n_total_final,
        'N_positive': n_positive,
        'Wronskian_err': w_err,
        't_final': t[-1],
        'n_modes': sim_cfg.n_modes,
        'boundary': sim_cfg.boundary,
    }


def print_results(results):
    """Print comparison table."""
    print()
    print("=" * 80)
    print("  TOPOLOGY COMPARISON v2 — frequency-matched, spring-restored")
    print("=" * 80)

    # Group by n_modes
    by_n = {}
    for r in results:
        n = r['n_modes']
        if n not in by_n:
            by_n[n] = {}
        by_n[n][r['boundary']] = r

    for n in sorted(by_n.keys()):
        group = by_n[n]
        print(f"\n  N_modes = {n}")
        print(f"  {'':>12} | {'E_plate(0)':>12} | {'E_plate(f)':>12} | {'Depletion':>9} | {'N_total':>10} | {'Wronsk err':>10} | {'q range':>16}")
        print("  " + "-" * 95)
        for b in ['closed', 'periodic']:
            if b not in group:
                continue
            r = group[b]
            q_lo, q_hi = r['q_range']
            print(f"  {b:>12} | {r['E_plate_0']:>12.6e} | {r['E_plate_f']:>12.6e} | {r['depletion']:>8.1%} | {r['N_total']:>10.4e} | {r['Wronskian_err']:>10.2e} | [{q_lo:.3f}, {q_hi:.3f}]")

        # Ratio
        if 'closed' in group and 'periodic' in group:
            rc = group['closed']
            rp = group['periodic']
            if rc['depletion'] > 0:
                ratio = rp['depletion'] / rc['depletion']
                print(f"  Depletion ratio (periodic/closed) = {ratio:.2f}x")

    # Cross-N ratio stability
    ns = sorted(by_n.keys())
    if len(ns) >= 2:
        print(f"\n  RATIO CONVERGENCE CHECK:")
        for n in ns:
            if 'closed' in by_n[n] and 'periodic' in by_n[n]:
                rc = by_n[n]['closed']
                rp = by_n[n]['periodic']
                ratio = rp['depletion'] / rc['depletion'] if rc['depletion'] > 0 else float('inf')
                print(f"    N={n:>3}: ratio = {ratio:.2f}x  (closed={rc['depletion']:.1%}, periodic={rp['depletion']:.1%})")


def main():
    quick = "--quick" in sys.argv

    if quick:
        mode_counts = [32]
        total_time = 500.0
        tag = "quick"
    else:
        mode_counts = [64, 128]
        total_time = 5000.0
        tag = "light"

    all_results = []

    for n_modes in mode_counts:
        configs = make_configs(n_modes, total_time, tag)

        for label, sim_cfg, run_cfg, output_path in configs:
            pre = PrecomputedArrays.from_config(sim_cfg)
            e0 = 0.5 * sim_cfg.plate_mass * sim_cfg.v0**2
            a0 = sim_cfg.q0 - sim_cfg.x_left
            omega1 = pre.ns_pi[0] / a0
            print("=" * 60)
            print(f"  {label.upper()} | N={sim_cfg.n_modes} | q0={sim_cfg.q0}")
            print(f"  omega_1 = {omega1:.4f}, k={sim_cfg.spring_k:.4f}")
            print(f"  E_plate(0) = {e0:.6e}")
            print(f"  cutoff: {sim_cfg.cutoff_shape}, n_cutoff={pre.n_cutoff}")
            mid = min(99, sim_cfg.n_modes - 1)
            print(f"  g_n[1]={pre.g_n[0]:.4f}, g_n[{mid+1}]={pre.g_n[mid]:.4f}, g_n[-1]={pre.g_n[-1]:.2e}")
            print(f"  Output: {output_path}")
            print("=" * 60)

            runner = ExperimentRunner(sim_cfg, run_cfg,
                                      output_path=output_path, overwrite=True)
            runner.run()
            print(f"  Done -> {output_path}")
            print()

            result = analyze_run(output_path, sim_cfg, label)
            all_results.append(result)

    print_results(all_results)


if __name__ == "__main__":
    main()
