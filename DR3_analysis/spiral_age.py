#!/usr/bin/env python3
"""
Estimate how long ago a galactic spiral feature formed.

Given a rotation curve and the galactocentric radii where the spiral
crosses a reference azimuth (phi = 0), the script infers the spiral's
age from the accumulated differential rotation.

Physics
-------
If a coherent disturbance created the spiral T years ago and all affected
stars started at phi = 0, then after time T a star at radius R has rotated by

    phi(R) = Omega(R) * T,    Omega(R) = V_c(R) / R

The spiral crosses phi = 0 again wherever phi(R) is a multiple of 2*pi:

    Omega(R_i) * T = 2*pi * k_i    (k_i a positive integer)

For two crossing radii R_1 < R_2 (so Omega_1 > Omega_2, k_1 > k_2):

    T = 2*pi * (k_1 - k_2) / (Omega(R_1) - Omega(R_2))

With only the *relative* winding difference between consecutive crossings
(default delta_k = 1) you can estimate T without knowing absolute orbit counts.
If you also know the absolute winding numbers you get one independent T per
radius as a cross-check.

Providing more than two radii yields multiple pairwise estimates that are
compared as a consistency check.

Examples
--------
Two adjacent crossings at 7.5 and 9.0 kpc, using an AGAMA potential:
    python spiral_age.py --radii 7.5 9.0 \\
        --potential /path/to/PriceWhelan22.ini

Three crossings, consecutive crossings separated by 1 winding:
    python spiral_age.py --radii 6.5 8.0 9.5 \\
        --potential /path/to/PriceWhelan22.ini

Same but non-adjacent crossings (2 windings between each pair):
    python spiral_age.py --radii 6.5 9.5 --delta-k 2 \\
        --potential /path/to/PriceWhelan22.ini

Explicit absolute winding numbers (enables per-radius age estimates too):
    python spiral_age.py --radii 6.5 8.0 9.5 --winding-numbers 5 4 3 \\
        --potential /path/to/PriceWhelan22.ini

Precomputed two-column rotation-curve file (R_kpc, Vc_kms, whitespace/comma):
    python spiral_age.py --radii 7.5 9.0 --vc-file rotation_curve.csv

V_c values supplied directly at the crossing radii (no external data needed):
    python spiral_age.py --radii 7.5 9.0 --vc-values 228.3 221.1
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 1 kpc / (km/s) expressed in Gyr
# Derivation: 1 kpc = 3.0857e16 km, 1 Gyr = 3.1558e16 s
# => 1 kpc/(km/s) = 3.0857e16 / 3.1558e16 Gyr ≈ 0.9778 Gyr
_KPC_PER_KMS_TO_GYR = 3.0857e16 / 3.1558e16


def rotation_curve_from_agama(potential, R_kpc, n_grid=5000):
    """Return V_c (km/s) at each radius in R_kpc using an AGAMA potential."""
    import agama
    agama.setUnits(mass=1.0, length=1.0, velocity=1.0)

    R_grid = np.linspace(max(0.1, 0.5 * R_kpc.min()), 1.5 * R_kpc.max() + 5, n_grid)
    dR = 1e-3
    Vc_grid = np.array([
        np.sqrt(Ri * (potential.potential(Ri + dR, 0, 0) - potential.potential(Ri - dR, 0, 0)) / (2 * dR))
        for Ri in R_grid
    ])
    interp = interp1d(R_grid, Vc_grid, kind="cubic", fill_value="extrapolate")
    return interp(R_kpc)


def rotation_curve_from_file(path, R_kpc):
    """Interpolate V_c (km/s) at R_kpc from a two-column file (R, V_c)."""
    try:
        data = np.loadtxt(path, delimiter=",")
    except ValueError:
        data = np.loadtxt(path)
    if data.shape[1] < 2:
        raise ValueError(f"{path}: expected at least two columns (R_kpc, Vc_kms)")
    R_file, Vc_file = data[:, 0], data[:, 1]
    interp = interp1d(R_file, Vc_file, kind="cubic", fill_value="extrapolate")
    return interp(R_kpc)


def compute_age(R_kpc, Vc_kms, winding_numbers, delta_k):
    """
    Compute spiral-age estimates from intersection radii and rotation curve.

    Parameters
    ----------
    R_kpc : array, shape (N,)  — crossing radii, sorted ascending
    Vc_kms : array, shape (N,) — circular velocity at each radius
    winding_numbers : array or None — absolute k_i values; if None, only
                      pairwise estimates using delta_k are returned
    delta_k : int — winding-number difference between consecutive crossings

    Returns
    -------
    pairwise_ages : list of (label, T_Gyr) for each consecutive pair
    absolute_ages : list of (label, T_Gyr) when winding_numbers is given
    """
    Omega = Vc_kms / R_kpc  # km/s / kpc

    pairwise_ages = []
    for i in range(len(R_kpc) - 1):
        dOmega = Omega[i] - Omega[i + 1]
        if dOmega <= 0:
            print(f"  Warning: Omega(R={R_kpc[i]:.2f}) <= Omega(R={R_kpc[i+1]:.2f}); "
                  "check that radii are sorted and rotation curve is declining.")
            continue
        T_raw = 2 * np.pi * delta_k / dOmega   # kpc / (km/s)
        T_Gyr = T_raw * _KPC_PER_KMS_TO_GYR
        label = f"R = {R_kpc[i]:.2f} – {R_kpc[i+1]:.2f} kpc  (Δk = {delta_k})"
        pairwise_ages.append((label, T_Gyr))

    absolute_ages = []
    if winding_numbers is not None:
        for i, (Ri, Oi, ki) in enumerate(zip(R_kpc, Omega, winding_numbers)):
            T_raw = 2 * np.pi * ki / Oi
            T_Gyr = T_raw * _KPC_PER_KMS_TO_GYR
            label = f"R = {Ri:.2f} kpc  (k = {ki})"
            absolute_ages.append((label, T_Gyr))

    return pairwise_ages, absolute_ages


def print_results(pairwise_ages, absolute_ages):
    col = 55
    print()
    print("=" * 70)
    print("  SPIRAL AGE ESTIMATES")
    print("=" * 70)

    if pairwise_ages:
        print("\n  Pairwise estimates (from consecutive crossing pairs):")
        print(f"  {'Pair':<{col}} {'Age (Gyr)':>10}")
        print("  " + "-" * (col + 12))
        vals = []
        for label, T in pairwise_ages:
            print(f"  {label:<{col}} {T:>10.3f}")
            vals.append(T)
        if len(vals) > 1:
            print("  " + "-" * (col + 12))
            print(f"  {'Mean':>{col}} {np.mean(vals):>10.3f}")
            print(f"  {'Std dev':>{col}} {np.std(vals):>10.3f}")

    if absolute_ages:
        print("\n  Per-radius estimates (from absolute winding numbers):")
        print(f"  {'Radius':>{col}} {'Age (Gyr)':>10}")
        print("  " + "-" * (col + 12))
        vals = []
        for label, T in absolute_ages:
            print(f"  {label:<{col}} {T:>10.3f}")
            vals.append(T)
        if len(vals) > 1:
            print("  " + "-" * (col + 12))
            print(f"  {'Mean':>{col}} {np.mean(vals):>10.3f}")
            print(f"  {'Std dev':>{col}} {np.std(vals):>10.3f}")

    print("=" * 70)
    print()


def plot_summary(R_kpc, Vc_kms, pairwise_ages, absolute_ages,
                 R_grid=None, Vc_grid=None):
    """
    Two-panel figure:
      left  — rotation curve with crossing radii marked
      right — bar chart of all age estimates
    """
    fig, (ax_rc, ax_age) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    # Left: rotation curve
    if R_grid is not None and Vc_grid is not None:
        ax_rc.plot(R_grid, Vc_grid, color="steelblue", lw=2, label=r"$V_c(R)$")
    ax_rc.scatter(R_kpc, Vc_kms, color="crimson", zorder=5,
                  s=80, label="Crossing radii")
    for Ri, Vci in zip(R_kpc, Vc_kms):
        ax_rc.axvline(Ri, color="crimson", lw=0.8, ls="--", alpha=0.5)
    ax_rc.set_xlabel(r"$R$ [kpc]")
    ax_rc.set_ylabel(r"$V_c$ [km s$^{-1}$]")
    ax_rc.set_title("Rotation curve")
    ax_rc.legend()

    # Right: age estimates as a bar chart
    all_ages = pairwise_ages + absolute_ages
    labels = [lbl.split("(")[0].strip() for lbl, _ in all_ages]
    ages = [T for _, T in all_ages]
    colors = ["steelblue"] * len(pairwise_ages) + ["seagreen"] * len(absolute_ages)

    y = np.arange(len(ages))
    ax_age.barh(y, ages, color=colors, edgecolor="k", lw=0.6)
    ax_age.set_yticks(y)
    ax_age.set_yticklabels(labels, fontsize=8)
    ax_age.axvline(np.mean(ages), color="crimson", lw=1.5, ls="--", label="Mean")
    ax_age.set_xlabel("Age [Gyr]")
    ax_age.set_title("Age estimates")
    ax_age.legend(fontsize=8)

    # Legend for bar colours
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="steelblue", label="Pairwise (Δk)"),
               Patch(facecolor="seagreen", label="Absolute (k)")]
    ax_age.legend(handles=handles, fontsize=8, loc="lower right")

    plt.savefig("spiral_age.pdf", dpi=150)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the age of a spiral feature from differential rotation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--radii", nargs="+", type=float, required=True, metavar="R",
        help="Galactocentric radii where the spiral crosses phi=0, in kpc "
             "(list two or more, will be sorted ascending).",
    )

    vc_group = parser.add_mutually_exclusive_group(required=True)
    vc_group.add_argument(
        "--potential", metavar="PATH",
        help="AGAMA potential .ini file used to compute the rotation curve.",
    )
    vc_group.add_argument(
        "--vc-file", metavar="PATH",
        help="Two-column text/CSV file with columns R (kpc) and V_c (km/s).",
    )
    vc_group.add_argument(
        "--vc-values", nargs="+", type=float, metavar="V",
        help="V_c values in km/s at each crossing radius (same order as --radii).",
    )

    winding_group = parser.add_mutually_exclusive_group()
    winding_group.add_argument(
        "--winding-numbers", nargs="+", type=int, metavar="K",
        help="Absolute number of full orbits each crossing represents "
             "(same order as --radii, must be decreasing with radius). "
             "Enables per-radius age estimates in addition to pairwise ones.",
    )
    winding_group.add_argument(
        "--delta-k", type=int, default=1, metavar="N",
        help="Winding-number difference between consecutive crossings (default 1). "
             "Ignored when --winding-numbers is given.",
    )

    parser.add_argument("--plot", action="store_true",
                        help="Show and save a summary figure (spiral_age.pdf).")
    args = parser.parse_args()

    # --- Validate and sort radii ---
    R_kpc = np.sort(np.array(args.radii))
    N = len(R_kpc)
    if N < 2:
        parser.error("At least two crossing radii are required.")

    # --- Validate winding numbers ---
    winding_numbers = None
    if args.winding_numbers is not None:
        wn = np.array(args.winding_numbers)
        if len(wn) != N:
            parser.error("--winding-numbers must have the same length as --radii.")
        # Re-sort to match sorted radii (outer radius → smaller k)
        sort_idx = np.argsort(args.radii)
        wn_sorted = wn[sort_idx]
        if not np.all(np.diff(wn_sorted) < 0):
            parser.error(
                "Winding numbers should decrease with radius "
                "(inner orbits more than outer orbits)."
            )
        winding_numbers = wn_sorted
        delta_k = 1  # will derive Δk per pair from winding_numbers directly

    # --- Get rotation curve ---
    R_grid = Vc_grid = None  # kept for plotting only

    if args.potential is not None:
        try:
            import agama
        except ImportError:
            sys.exit("agama is not installed; use --vc-file or --vc-values instead.")
        potential = agama.Potential(args.potential)
        Vc_kms = rotation_curve_from_agama(potential, R_kpc)
        if args.plot:
            R_grid = np.linspace(0.5, max(R_kpc) * 1.5 + 3, 500)
            Vc_grid = rotation_curve_from_agama(potential, R_grid)

    elif args.vc_file is not None:
        Vc_kms = rotation_curve_from_file(args.vc_file, R_kpc)
        if args.plot:
            try:
                data = np.loadtxt(args.vc_file, delimiter=",")
            except ValueError:
                data = np.loadtxt(args.vc_file)
            R_grid, Vc_grid = data[:, 0], data[:, 1]

    else:  # --vc-values
        if len(args.vc_values) != N:
            parser.error("--vc-values must have the same number of entries as --radii.")
        sort_idx = np.argsort(args.radii)
        Vc_kms = np.array(args.vc_values)[sort_idx]

    Omega_kms_kpc = Vc_kms / R_kpc

    # --- Echo inputs ---
    print()
    print("Crossing radii and angular velocities:")
    print(f"  {'R (kpc)':>10}  {'V_c (km/s)':>12}  {'Omega (km/s/kpc)':>18}"
          + ("  winding k" if winding_numbers is not None else ""))
    print("  " + "-" * (46 + (12 if winding_numbers is not None else 0)))
    for i in range(N):
        extra = f"  {winding_numbers[i]:>10}" if winding_numbers is not None else ""
        print(f"  {R_kpc[i]:>10.3f}  {Vc_kms[i]:>12.2f}  {Omega_kms_kpc[i]:>18.4f}{extra}")

    # --- Pairwise estimates ---
    if winding_numbers is not None:
        # Use the actual winding differences between each consecutive pair
        pairwise_ages = []
        for i in range(N - 1):
            dk = int(winding_numbers[i] - winding_numbers[i + 1])
            dOmega = Omega_kms_kpc[i] - Omega_kms_kpc[i + 1]
            if dOmega <= 0:
                print(f"  Warning: non-declining Omega between "
                      f"R={R_kpc[i]:.2f} and R={R_kpc[i+1]:.2f} kpc; skipping.")
                continue
            T_Gyr = (2 * np.pi * dk / dOmega) * _KPC_PER_KMS_TO_GYR
            label = f"R = {R_kpc[i]:.2f} – {R_kpc[i+1]:.2f} kpc  (Δk = {dk})"
            pairwise_ages.append((label, T_Gyr))
        absolute_ages = [
            (f"R = {R_kpc[i]:.2f} kpc  (k = {winding_numbers[i]})",
             (2 * np.pi * winding_numbers[i] / Omega_kms_kpc[i]) * _KPC_PER_KMS_TO_GYR)
            for i in range(N)
        ]
    else:
        pairwise_ages, absolute_ages = compute_age(
            R_kpc, Vc_kms, winding_numbers, args.delta_k
        )

    if not pairwise_ages and not absolute_ages:
        sys.exit("No valid age estimates could be computed. Check your inputs.")

    print_results(pairwise_ages, absolute_ages)

    if args.plot:
        plot_summary(R_kpc, Vc_kms, pairwise_ages, absolute_ages, R_grid, Vc_grid)


if __name__ == "__main__":
    main()
