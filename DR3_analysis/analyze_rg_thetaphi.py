#!/usr/bin/env python3
"""
Phase-spiral analysis in guiding-radius (Rg) vs azimuthal-angle (theta_phi) space.

Reads the AGAMA action-angle HDF5 file, computes guiding radii from the
circular-velocity curve, then runs LaguerreSnails decomposition across a grid
of (Rg, theta_phi) bins and produces summary amplitude maps.

Input:  Gaia_data_Agama_1kpc_v1.h5  (or --input)
Output: plots saved as PDFs + shown interactively

Run with py311 environment.
"""

import argparse
import sys
import os
import copy

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cmasher as cmr
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from astropy.table import Table
import h5py
import agama

sys.path.append(os.path.dirname(__file__))
from df_helpers_data import *  # noqa: F401,F403  (provides LaguerreSnails)


def load_data(path):
    with h5py.File(path, "r") as f:
        Jr = np.array(f["Jr"])
        Jphi = np.array(f["Jphi"])
        Jz = np.array(f["Jz"])
        Omega_r = np.array(f["Omega_r"])
        Omega_phi = np.array(f["Omega_phi"])
        Omega_z = np.array(f["Omega_z"])
        theta_r = np.array(f["theta_r"])
        theta_phi = np.array(f["theta_phi"])
        theta_z = np.array(f["theta_z"])
        x = np.array(f["x"])
        y = np.array(f["y"])
        z = np.array(f["z"])
        vx = np.array(f["vx"])
        vy = np.array(f["vy"])
        vz = np.array(f["vz"])
        S = np.array(f["S"])

    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    theta_phi = np.mod(theta_phi, 2 * np.pi)
    theta_z = np.mod(theta_z, 2 * np.pi)
    phi = np.mod(phi, 2 * np.pi)

    return dict(
        Jr=Jr, Jphi=Jphi, Jz=Jz,
        Omega_r=Omega_r, Omega_phi=Omega_phi, Omega_z=Omega_z,
        theta_r=theta_r, theta_phi=theta_phi, theta_z=theta_z,
        x=x, y=y, z=z, vx=vx, vy=vy, vz=vz,
        S=S, R=R, phi=phi,
    )


def compute_guiding_radius(potential, Jphi):
    """Map each star's Jphi to a guiding radius via the circular-velocity curve."""
    R_grid = np.linspace(0.1, 30, 5000)
    V_c_grid = np.zeros_like(R_grid)
    dR = 1e-2
    for i, Ri in enumerate(R_grid):
        dPhi_dR = (potential.potential(Ri + dR, 0, 0) - potential.potential(Ri - dR, 0, 0)) / (2 * dR)
        V_c_grid[i] = np.sqrt(Ri * dPhi_dR)

    J_grid = R_grid * V_c_grid
    idx = np.searchsorted(J_grid, np.abs(Jphi))
    idx = np.clip(idx, 0, len(R_grid) - 1)
    Rg = R_grid[idx]
    Vc = V_c_grid[idx]
    return Rg, Vc


def plot_rotation_curve(Rg, Vc):
    import pandas as pd
    df = pd.DataFrame({"Rg": Rg, "Vc": Vc})
    Rg_vc_bins = np.linspace(0, 20, 50)
    bin_centers = 0.5 * (Rg_vc_bins[1:] + Rg_vc_bins[:-1])
    bin_idx = np.digitize(df["Rg"], Rg_vc_bins) - 1
    median_Vc = np.array([
        df["Vc"][bin_idx == i].median() if np.any(bin_idx == i) else np.nan
        for i in range(len(bin_centers))
    ])
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, median_Vc, color="red", lw=2, label="Median Vc")
    plt.xlabel(r"Guiding radius $R_g$ [kpc]")
    plt.ylabel(r"Circular velocity $V_c$ [km/s]")
    plt.title("Galaxy Rotation Curve: Gaia Data")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/rotation_curve.pdf", dpi=150)


def plot_spiral_panels(residual_list, n_phi, n_R, xedges, yedges, colorbarlim):
    fig, axes = plt.subplots(n_phi, n_R, figsize=(n_R, n_phi),
                             sharex=True, sharey=True, constrained_layout=True)
    fig.suptitle(r"Rg vs $\Theta_\phi$ within 1 kpc of the Sun: Gaia Data", fontsize=18)
    rect = patches.Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                              fill=False, edgecolor="black", linewidth=2)
    fig.add_artist(rect)

    for i in range(n_phi):
        for j in range(n_R):
            ax = axes[i, j]
            ind = n_R * i + j
            if ind >= len(residual_list) or residual_list[ind] is None:
                ax.set_visible(False)
                continue
            ax.imshow(
                residual_list[ind].T, origin="lower",
                vmin=colorbarlim[0], vmax=colorbarlim[1],
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                interpolation="gaussian", cmap="inferno_r", aspect="auto", rasterized=True,
            )
            ax.scatter(0, 0, color="cyan", s=10)

    fig.supxlabel("Guiding Radius (kpc)", fontsize=16)
    fig.supylabel(r"$\Theta_{\phi}$", fontsize=16)
    plt.savefig("figures/spirals_rg_thetaphi.pdf", dpi=150)


def plot_amplitude_maps(RG_mesh, Theta_phi_mesh, m0_amp_all, m1_amp_all, m2_amp_all):
    m1m2_amp_max = np.maximum(m1_amp_all, m2_amp_all)
    vmin, vmax = 0.05, 0.7
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

    Rg_centers = RG_mesh[0, :]  # one value per R_g column

    ratios = [
        m1_amp_all / m0_amp_all,
        m2_amp_all / m0_amp_all,
        (m1_amp_all + m2_amp_all) / m0_amp_all,
        m1m2_amp_max / m0_amp_all,
    ]
    titles = [
        "m=1 Amplitude Ratio",
        "m=2 Amplitude Ratio",
        "Sum(m=1,m=2) Amplitude Ratio",
        "Max(m=1,m=2) Amplitude Ratio",
    ]

    fig = plt.figure(figsize=(16, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 2])

    top_axes = [fig.add_subplot(gs[0, k]) for k in range(4)]
    bot_axes = [fig.add_subplot(gs[1, k]) for k in range(4)]

    # Share x-axis between the top and bottom panel of each column
    for tax, bax in zip(top_axes, bot_axes):
        tax.sharex(bax)

    for k, (ratio, title, tax, bax) in enumerate(zip(ratios, titles, top_axes, bot_axes)):
        # Top panel: mean over theta_phi bins at each R_g
        mean_ratio = np.nanmean(ratio, axis=0)
        tax.plot(Rg_centers, mean_ratio, color="k", lw=1.5)
        tax.set_yscale("log")
        tax.set_ylim(vmin, vmax)
        tax.set_title(title)
        plt.setp(tax.get_xticklabels(), visible=False)
        if k == 0:
            tax.set_ylabel("Mean ratio")

        # Bottom panel: 2D amplitude map
        im = bax.pcolormesh(RG_mesh, Theta_phi_mesh, ratio, cmap=cmr.chroma, norm=norm)
        bax.set_xlabel(r"Guiding Radius $R_g$ [kpc]")
        if k == 0:
            bax.set_ylabel(r"$\Theta_\phi$ [radians]")

    fig.colorbar(im, ax=bot_axes, label="Amplitude Ratio")
    plt.savefig("figures/Gaia_m1_m2_amplitude_ratios.pdf", dpi=300)


def main():
    parser = argparse.ArgumentParser(description="Phase-spiral analysis in Rg-theta_phi space")
    parser.add_argument("--input", default="Gaia_data_Agama_1kpc_v1.h5", help="Input HDF5 file")
    parser.add_argument(
        "--potential",
        default="/Users/Tavangar/Work/packages/Agama/data/PriceWhelan22.ini",
        help="Path to AGAMA potential .ini file",
    )
    args = parser.parse_args()

    agama.setUnits(mass=1.0, length=1.0, velocity=1.0)
    potential = agama.Potential(args.potential)

    # --- Load ---
    print(f"Loading {args.input}...")
    d = load_data(args.input)
    print(f"Total stars: {len(d['Jphi'])}")

    # --- Guiding radius ---
    print("Computing guiding radii...")
    Rg, Vc = compute_guiding_radius(potential, d["Jphi"])

    plot_rotation_curve(Rg, Vc)

    # --- Filter NaN ---
    valid = (
        np.isfinite(Rg) & np.isfinite(d["Jphi"]) &
        np.isfinite(d["theta_phi"]) & np.isfinite(d["Jz"]) & np.isfinite(d["theta_z"])
    )
    Rg = Rg[valid]
    theta_phi = d["theta_phi"][valid]
    Jphi = d["Jphi"][valid]
    Jz = d["Jz"][valid]
    theta_z = d["theta_z"][valid]
    S = d["S"][valid]
    print(f"Stars after NaN filter: {len(Rg)}")

    jx = np.sqrt(Jz) * np.cos(theta_z)
    jy = np.sqrt(Jz) * np.sin(theta_z)

    # --- Bin definitions ---
    Rg_bins = np.arange(5.5, 10.5, 1 / 4)
    theta_phi_bins = np.arange(2.7, 3.6 + 1e-5, 0.45)
    Rg_bin_centers = 0.5 * (Rg_bins[1:] + Rg_bins[:-1])
    theta_phi_bin_centers = 0.5 * (theta_phi_bins[1:] + theta_phi_bins[:-1])
    Rg_bin_rad = np.diff(Rg_bins)[0] / 2.0
    theta_phi_bin_rad = np.diff(theta_phi_bins)[0] / 2.0
    n_R = len(Rg_bin_centers)
    n_phi = len(theta_phi_bin_centers)

    xmin, xmax, xstep = -8, 8, 0.16
    ymin, ymax, ystep = -8, 8, 0.16
    xbins = np.arange(xmin, xmax, xstep)
    ybins = np.arange(ymin, ymax, ystep)
    colorbarlim = [-0.3, 0.3]

    # --- Pass 1: build residual panels ---
    print("Building spiral panels (pass 1)...")
    bin_masks = []
    residual_list = []
    xedges = yedges = None

    for i, tpc in enumerate(theta_phi_bin_centers):
        print(f"  theta_phi bin {i + 1}/{n_phi}")
        for j, Rgc in enumerate(Rg_bin_centers):
            bin_mask = (np.abs(Rg - Rgc) < Rg_bin_rad) & (np.abs(theta_phi - tpc) < theta_phi_bin_rad)
            bin_masks.append(bin_mask)

            if np.sum(bin_mask) == 0:
                residual_list.append(None)
                continue

            density, xe, ye, _ = stats.binned_statistic_2d(
                jx[bin_mask], jy[bin_mask], jx[bin_mask], statistic="count", bins=(xbins, ybins)
            )
            if xedges is None:
                xedges, yedges = xe, ye

            weights = gaussian_filter(density, sigma=2)
            if np.sum(weights) == 0:
                residual_list.append(None)
                continue

            xcenters = 0.5 * (xe[:-1] + xe[1:])
            ycenters = 0.5 * (ye[:-1] + ye[1:])
            Xc, Yc = np.meshgrid(xcenters, ycenters, indexing="ij")
            X_c = np.sum(Xc * weights) / np.sum(weights)
            Y_c = np.sum(Yc * weights) / np.sum(weights)

            jx_shift = jx[bin_mask] - X_c
            jy_shift = jy[bin_mask] - Y_c

            density_centered, _, _, _ = stats.binned_statistic_2d(
                jx_shift, jy_shift, jx_shift, statistic="count", bins=(xbins, ybins)
            )
            background = gaussian_filter(density_centered, sigma=3, order=0, mode="reflect")
            residual = gaussian_filter(
                (density_centered / background) - 1, sigma=1, order=0, mode="reflect"
            )
            residual_list.append(residual)

    plot_spiral_panels(residual_list, n_phi, n_R, xedges, yedges, colorbarlim)

    # --- Pass 2: LaguerreSnails decomposition ---
    print("Running LaguerreSnails decomposition (pass 2)...")
    a_all = np.zeros((n_phi, n_R))
    m0_amp_all = np.zeros((n_phi, n_R))
    m1_amp_all = np.zeros((n_phi, n_R))
    m2_amp_all = np.zeros((n_phi, n_R))
    m1_Pt_all = np.zeros((n_phi, n_R))
    m2_Pt_all = np.zeros((n_phi, n_R))
    m1_Ph_all = np.zeros((n_phi, n_R))
    m2_Ph_all = np.zeros((n_phi, n_R))
    m1_Fl_all = np.zeros((n_phi, n_R))
    m2_Fl_all = np.zeros((n_phi, n_R))

    jz_grid = (np.arange(0, 5, 0.1)) ** 2
    thetaz_grid = np.arange(0, 2 * np.pi, np.pi / 48)

    recon_m0_all = [[np.zeros((len(jz_grid), len(thetaz_grid))) for _ in range(n_R)] for _ in range(n_phi)]
    recon_m1_all = [[np.zeros((len(jz_grid), len(thetaz_grid))) for _ in range(n_R)] for _ in range(n_phi)]
    recon_m2_all = [[np.zeros((len(jz_grid), len(thetaz_grid))) for _ in range(n_R)] for _ in range(n_phi)]


    for i, tpc in enumerate(theta_phi_bin_centers):
        print(f"  theta_phi bin {i + 1}/{n_phi}")
        for j, Rgc in enumerate(Rg_bin_centers):
            ind = n_R * i + j
            bin_mask = bin_masks[ind]
            if np.sum(bin_mask) == 0:
                m0_amp_all[i, j] = m1_amp_all[i, j] = m2_amp_all[i, j] = np.nan
                a_all[i, j] = np.nan
                continue

            Jz_bin = Jz[bin_mask]
            theta_z_bin = theta_z[bin_mask]
            Jphi_bin = Jphi[bin_mask]
            theta_phi_bin = theta_phi[bin_mask]
            S_bin = S[bin_mask]

            jx_bin = np.sqrt(Jz_bin) * np.cos(theta_z_bin)
            jy_bin = np.sqrt(Jz_bin) * np.sin(theta_z_bin)

            density, xe, ye, _ = stats.binned_statistic_2d(
                jx_bin, jy_bin, jx_bin, statistic="count", bins=(xbins, ybins)
            )
            weights = gaussian_filter(density, sigma=2)
            if np.sum(weights) == 0:
                m0_amp_all[i, j] = m1_amp_all[i, j] = m2_amp_all[i, j] = np.nan
                a_all[i, j] = np.nan
                continue

            xcenters = 0.5 * (xe[:-1] + xe[1:])
            ycenters = 0.5 * (ye[:-1] + ye[1:])
            Xc, Yc = np.meshgrid(xcenters, ycenters, indexing="ij")
            X_c = np.sum(Xc * weights) / np.sum(weights)
            Y_c = np.sum(Yc * weights) / np.sum(weights)

            jx_shift = jx_bin - X_c
            jy_shift = jy_bin - Y_c
            r_shift = np.sqrt(jx_shift**2 + jy_shift**2)
            theta_z_shift = np.arctan2(jy_shift, jx_shift)
            Jz_shift = r_shift**2

            data = Table(
                [Jphi_bin, Jz_shift, theta_phi_bin, theta_z_shift],
                names=["jphi", "jz", "theta_phi", "theta_z"],
            ).to_pandas()

            lss = LaguerreSnails(data, jz_grid, thetaz_grid, m_max=3, n_max=20)
            if len(lss.sel) == 0:
                m0_amp_all[i, j] = m1_amp_all[i, j] = m2_amp_all[i, j] = np.nan
                a_all[i, j] = np.nan
                continue

            lss.get_coeffs(S_per_star=S_bin)
            m0_amp_all[i, j] = np.linalg.norm(np.abs(lss.coeffs[0]))
            m1_amp_all[i, j] = np.linalg.norm(np.abs(lss.coeffs[1]))
            m2_amp_all[i, j] = np.linalg.norm(np.abs(lss.coeffs[2]))

            m1_pitch, m1_phase, m1_flag, _ = lss.get_pitch_phase_angles(m=1)
            m2_pitch, m2_phase, m2_flag, _ = lss.get_pitch_phase_angles(m=2)

            lss.create_df(lss.n_maxs)
            lss.make_spiral_residual()
            a, recon_m0, recon_m1, recon_m2, _, _, _ = lss.make_spiral_recon(lss.n_maxs)

            a_all[i, j] = a
            m1_Pt_all[i, j] = m1_pitch
            m2_Pt_all[i, j] = m2_pitch
            m1_Ph_all[i, j] = m1_phase
            m2_Ph_all[i, j] = m2_phase
            m1_Fl_all[i, j] = m1_flag
            m2_Fl_all[i, j] = m2_flag
            recon_m0_all[i][j] = recon_m0
            recon_m1_all[i][j] = recon_m1
            recon_m2_all[i][j] = recon_m2

    # --- Reconstruction plots ---
    recon_m0 = np.array(recon_m0_all, dtype=float)
    recon_m1 = np.array(recon_m1_all, dtype=float)
    recon_m2 = np.array(recon_m2_all, dtype=float)

    # --- Amplitude maps ---
    RG_mesh, Theta_phi_mesh = np.meshgrid(Rg_bins[:-1] + 0.2, theta_phi_bins[:-1] + 0.05)
    plot_amplitude_maps(RG_mesh, Theta_phi_mesh, m0_amp_all, m1_amp_all, m2_amp_all)

    # --- Norm maps ---
    
    xgrid = np.arange(-lss.rootjzmax,lss.rootjzmax+1e-5,lss.rootjzstep)
    ygrid = np.arange(-lss.rootjzmax,lss.rootjzmax+1e-5,lss.rootjzstep)
    X_grid, Y_grid = np.meshgrid(xgrid, ygrid)
    good_inds = np.array(np.where(X_grid**2 + Y_grid**2 > 2)).T
    total_norm = np.zeros((n_phi, n_R))
    subset_norm = np.zeros((n_phi, n_R))
    for i in range(n_phi):
        for j in range(n_R):
            ratio = recon_m1[i, j] / recon_m0[i, j]
            total_norm[i, j] = np.linalg.norm(ratio)
            subset_norm[i, j] = np.linalg.norm(ratio[good_inds[:, 0], good_inds[:, 1]])

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
    im = ax.pcolormesh(RG_mesh, Theta_phi_mesh, total_norm,
                        cmap=cmr.chroma, norm=mpl.colors.LogNorm())
    ax.set_title("Norm of m=1 Reconstruction Ratio")
    fig.colorbar(im, label="Amplitude")
    ax.set_xlabel(r"Guiding Radius $R_g$ [kpc]")
    ax.set_ylabel(r"$\Theta_\phi$ [radians]")
    plt.savefig("figures/norm_m1_total_rg_thetaphi.pdf", dpi=150)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
    im = ax.pcolormesh(RG_mesh, Theta_phi_mesh, subset_norm,
                        cmap=cmr.chroma, norm=mpl.colors.LogNorm())
    ax.set_title("Norm of m=1 Reconstruction Ratio (excl. central area)")
    fig.colorbar(im, label="Amplitude")
    ax.set_xlabel(r"Guiding Radius $R_g$ [kpc]")
    ax.set_ylabel(r"$\Theta_\phi$ [radians]")
    plt.savefig("figures/norm_m1_subset_rg_thetaphi.pdf", dpi=150)

    print("All done.")


if __name__ == "__main__":
    main()
