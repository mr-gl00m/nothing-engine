"""
Phase 2, Tasks 2.3-2.5 — Run all pre-registered analyses on completed experiments.

Loads closed and/or open ringdown HDF5 files and runs:
  1. Ringdown fitting (3 models + AIC)
  2. PSD analysis (post-ringdown velocity spectrum)
  3. Residual motion comparison (KS test)
  4. Generates summary figures (matplotlib)

Usage:
    python -m experiments.run_analysis [--closed PATH] [--open PATH] [--no-plots]
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import h5py

from nothing_engine.analysis.ringdown_fit import fit_ringdown, load_ringdown_data
from nothing_engine.analysis.psd_analysis import (
    compute_psd, load_velocity_data, find_post_ringdown_start,
    fdt_prediction_T0,
)
from nothing_engine.analysis.residual_motion import load_and_compare, compute_residual_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_single(hdf5_path: str, label: str, make_plots: bool = True):
    """Run full analysis pipeline on a single experiment."""
    print(f"\n{'='*60}")
    print(f"  Analysis: {label}")
    print(f"  File: {hdf5_path}")
    print(f"{'='*60}\n")

    # 1. Ringdown fit
    t, e_plate = load_ringdown_data(hdf5_path)
    print(f"Data: {len(t)} time points, t=[{t[0]:.1f}, {t[-1]:.1f}]")
    print(f"E_plate: initial={e_plate[0]:.6e}, final={e_plate[-1]:.6e}")
    print(f"Depletion: {e_plate[-1]/e_plate[0]*100:.4f}% remaining\n")

    results = fit_ringdown(t, e_plate)
    print(results.summary())

    # 2. PSD (if ringdown completed)
    try:
        t_post = find_post_ringdown_start(t, e_plate, frac=0.01)
        print(f"\nPost-ringdown starts at t = {t_post:.1f}")

        t_v, v = load_velocity_data(hdf5_path)
        psd_result = compute_psd(t_v, v, post_ringdown_t=t_post)
        print(f"PSD computed: {psd_result.n_points} points, "
              f"dt={psd_result.dt:.4f}, nperseg={psd_result.nperseg}")
        print(f"Peak PSD frequency: {psd_result.freqs[np.argmax(psd_result.psd[1:]) + 1]:.4f}")

        # 3. Residual stats
        stats = compute_residual_stats(t_v, v, t_post)
        print(f"\nResidual motion: RMS v = {stats.rms_velocity:.6e}")

    except ValueError as e:
        print(f"\nPost-ringdown analysis skipped: {e}")
        t_post = None
        psd_result = None
        stats = None

    # 4. Plots
    if make_plots:
        try:
            _make_plots(hdf5_path, label, t, e_plate, results,
                        psd_result, t_post)
        except Exception as e:
            logger.warning("Plot generation failed: %s", e)

    return results, psd_result, stats


def _make_plots(hdf5_path, label, t, e_plate, fit_results,
                psd_result, t_post):
    """Generate summary figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(hdf5_path).parent / "figures"
    out_dir.mkdir(exist_ok=True)
    prefix = label.lower().replace(" ", "_")

    # Fig 1: Ringdown curve + fit
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(t, e_plate, "k-", alpha=0.5, label="Data", linewidth=0.5)

    if fit_results.exponential and fit_results.exponential.converged:
        fr = fit_results.exponential
        t_win_start, t_win_end = fit_results.fitting_window
        t_mask = (t >= t_win_start) & (t <= t_win_end)
        t_fit = t[t_mask] - t[t_mask][0]
        from nothing_engine.analysis.ringdown_fit import exponential_model
        pred = exponential_model(t_fit, **fr.params)
        ax.semilogy(t[t_mask], pred, "r-", label=f"Exp fit (γ={fr.params['gamma']:.2e})",
                     linewidth=1.5)

    ax.set_xlabel("Time (cavity crossing times)")
    ax.set_ylabel("E_plate")
    ax.set_title(f"Ringdown: {label}")
    ax.legend()
    fig.savefig(out_dir / f"{prefix}_ringdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / f'{prefix}_ringdown.png'}")

    # Fig 2: Energy components
    with h5py.File(hdf5_path, "r") as f:
        ts = f["timeseries"]
        t_all = ts["t"][:]
        e_plate_all = ts["E_plate"][:]
        e_field = ts["E_field"][:]
        e_total = ts["E_total"][:]
        n_particles = ts["total_particles"][:]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].semilogy(t_all, e_plate_all, label="E_plate")
    axes[0, 0].set_ylabel("E_plate")
    axes[0, 0].set_title("Plate kinetic energy")

    axes[0, 1].plot(t_all, (e_total - e_total[0]) / max(e_plate_all[0], 1e-30))
    axes[0, 1].set_ylabel("(E_total - E_0) / E_plate(0)")
    axes[0, 1].set_title("Energy conservation drift")

    axes[1, 0].semilogy(t_all, n_particles)
    axes[1, 0].set_ylabel("N_particles")
    axes[1, 0].set_title("Total photon number")

    if psd_result is not None:
        axes[1, 1].semilogy(psd_result.freqs, psd_result.psd)
        axes[1, 1].set_xlabel("Frequency")
        axes[1, 1].set_ylabel("PSD(v)")
        axes[1, 1].set_title("Post-ringdown velocity PSD")
    else:
        axes[1, 1].text(0.5, 0.5, "No post-ringdown data",
                        transform=axes[1, 1].transAxes, ha="center")

    for ax in axes.flat:
        ax.set_xlabel("Time")

    fig.suptitle(label, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / f'{prefix}_dashboard.png'}")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 2 analysis")
    parser.add_argument("--closed", type=str, default=None,
                        help="Path to closed ringdown HDF5")
    parser.add_argument("--open", type=str, default=None,
                        help="Path to open ringdown HDF5")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.closed is None and args.open is None:
        # Try default paths, then quick variants
        for suffix in ["", "_quick"]:
            c = f"data/experiments/closed_ringdown{suffix}.h5"
            o = f"data/experiments/open_ringdown{suffix}.h5"
            if Path(c).exists() or Path(o).exists():
                args.closed = c
                args.open = o
                break
        else:
            args.closed = "data/experiments/closed_ringdown.h5"
            args.open = "data/experiments/open_ringdown.h5"

    plots = not args.no_plots

    closed_results = None
    open_results = None

    found_any = False
    if args.closed and Path(args.closed).exists():
        found_any = True
        closed_results = analyze_single(args.closed, "Closed System", plots)

    if args.open and Path(args.open).exists():
        found_any = True
        open_results = analyze_single(args.open, "Open System", plots)

    if not found_any:
        print("No experiment data found.")
        print(f"  Looked for: {args.closed}")
        print(f"              {args.open}")
        print("  Run an experiment first, or pass --closed / --open paths.")
        sys.exit(1)

    # Cross-comparison
    if closed_results and open_results:
        print(f"\n{'='*60}")
        print("  Cross-System Comparison")
        print(f"{'='*60}\n")

        cr_fit, cr_psd, cr_stats = closed_results
        or_fit, or_psd, or_stats = open_results

        # Damping rate comparison
        if (cr_fit.exponential and cr_fit.exponential.converged and
                or_fit.exponential and or_fit.exponential.converged):
            g_closed = cr_fit.exponential.gamma_with_ci()
            g_open = or_fit.exponential.gamma_with_ci()
            if g_closed and g_open:
                print(f"gamma_closed = {g_closed[0]:.6e} [{g_closed[1]:.6e}, {g_closed[2]:.6e}]")
                print(f"gamma_open   = {g_open[0]:.6e} [{g_open[1]:.6e}, {g_open[2]:.6e}]")
                diff = abs(g_open[0] - g_closed[0])
                # Combined sigma (rough)
                se_c = cr_fit.exponential.param_errors.get("gamma", 0)
                se_o = or_fit.exponential.param_errors.get("gamma", 0)
                sigma = np.sqrt(se_c**2 + se_o**2) if (se_c + se_o) > 0 else 1e-30
                print(f"|gamma_open - gamma_closed| = {diff:.6e}")
                print(f"Combined sigma = {sigma:.6e}")
                print(f"Difference in sigma units: {diff/sigma:.2f}")
                print(f"Significant (>3 sigma): {diff > 3 * sigma}")

        # Residual motion KS test
        if cr_stats and or_stats:
            print("\nResidual motion KS test:")
            comp = load_and_compare(
                args.closed, args.open,
                t_start_a=cr_psd.t_start if cr_psd else 0,
                t_start_b=or_psd.t_start if or_psd else 0,
                label_a="Closed", label_b="Open",
            )
            print(comp.summary())


if __name__ == "__main__":
    main()
