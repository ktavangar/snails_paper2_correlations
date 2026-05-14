#!/usr/bin/env python3
"""
Phase-spiral analysis in cylindrical (R, phi) space.

Reads the AGAMA action-angle HDF5 file and runs LaguerreSnails decomposition
across a grid of (R, phi) bins, then produces summary amplitude maps.

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
from astropy.table import Table
import h5py

sys.path.append(os.path.dirname(__file__))
from df_helpers_data import *  # noqa: F401,F403  (provides LaguerreSnails)


def load_data(path):
    with h5py.File(path, "r") as f:
        Jz = np.array(f["Jz"])
        Jphi = np.array(f["Jphi"])
        theta_z = np.array(f["theta_z"])
        theta_phi = np.array(f["theta_phi"])
        x = np.array(f["x"])
        y = np.array(f["y"])
        S = np.array(f["S"])

    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    phi = np.mod(phi, 2 * np.pi)
    theta_z = np.mod(theta_z, 2 * np.pi)
    theta_phi = np.mod(theta_phi, 2 * np.pi)

    return dict(Jz=Jz, Jphi=Jphi, theta_z=theta_z, theta_phi=theta_phi,
                R=R, phi=phi, S=S)


def main():
    parser = argparse.ArgumentParser(description="Phase-spiral analysis in R-phi space")
    parser.add_argument("--input", default="Gaia_data_Agama_1kpc_v1.h5", help="Input HDF5 file")
    parser.add_argument("--no-show", action="store_true", help="Save figures but do not display them")
    args = parser.parse_args()

    if args.no_show:
        mpl.use("Agg")

    # --- Load ---
    print(f"Loading {args.input}...")
    d = load_data(args.input)
    print(f"Total stars: {len(d['R'])}")

    plt.figure()
    plt.hist(d["phi"], bins=100)
    plt.xlabel(r"$\phi$ [rad]")
    plt.title(r"Azimuthal angle $\phi$ distribution")
    plt.tight_layout()
    plt.savefig("figres/hist_phi.pdf", dpi=150)
    plt.show()

    # --- Filter NaN ---
    valid = (
        np.isfinite(d["R"]) & np.isfinite(d["phi"]) &
        np.isfinite(d["Jz"]) & np.isfinite(d["theta_z"])
    )
    R = d["R"][valid]
    phi = d["phi"][valid]
    Jz = d["Jz"][valid]
    theta_z = d["theta_z"][valid]
    theta_phi_arr = d["theta_phi"][valid]
    S = d["S"][valid]
    print(f"Stars after NaN filter: {len(R)}")

    jx = np.sqrt(Jz) * np.cos(theta_z)
    jy = np.sqrt(Jz) * np.sin(theta_z)

    # --- Bin definitions ---
    R_bins = np.arange(7.2, 9.2, 0.2)
    phi_bins = np.arange(2.95, 3.35 + 1e-5, 0.04)
    R_bin_centers = 0.5 * (R_bins[1:] + R_bins[:-1])
    phi_bin_centers = 0.5 * (phi_bins[1:] + phi_bins[:-1])
    R_bin_rad = np.diff(R_bins)[0] / 2.0
    phi_bin_rad = np.diff(phi_bins)[0] / 2.0
    n_R = len(R_bin_centers)
    n_phi = len(phi_bin_centers)

    xmin, xmax, xstep = -8, 8, 0.16
    ymin, ymax, ystep = -8, 8, 0.16
    xbins = np.arange(xmin, xmax, xstep)
    ybins = np.arange(ymin, ymax, ystep)
    colorbarlim = [-0.3, 0.3]

    # --- Pass 1: residual panels ---
    print("Building spiral panels (pass 1)...")
    bin_masks = []
    residual_list = []
    xedges = yedges = None

    for i, phi_center in enumerate(phi_bin_centers):
        print(f"  phi bin {i + 1}/{n_phi}")
        for j, R_center in enumerate(R_bin_centers):
            bin_mask = (np.abs(R - R_center) < R_bin_rad) & (np.abs(phi - phi_center) < phi_bin_rad)
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

    # Plot residual panels
    fig, axes = plt.subplots(n_phi, n_R, figsize=(n_R, n_phi),
                             sharex=True, sharey=True, constrained_layout=True)
    fig.suptitle(r"$R$ vs $\phi$ within 1 kpc of the Sun: Gaia Data", fontsize=18)
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
    fig.supxlabel("R (kpc)", fontsize=16)
    fig.supylabel(r"$\phi$ [rad]", fontsize=16)
    plt.savefig("figures/spirals_r_phi.pdf", dpi=150)
    plt.show()

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

    for i, phi_center in enumerate(phi_bin_centers):
        print(f"  phi bin {i + 1}/{n_phi}")
        for j, R_center in enumerate(R_bin_centers):
            ind = n_R * i + j
            bin_mask = bin_masks[ind]
            if np.sum(bin_mask) == 0:
                m0_amp_all[i, j] = m1_amp_all[i, j] = m2_amp_all[i, j] = np.nan
                a_all[i, j] = np.nan
                continue

            Jz_bin = Jz[bin_mask]
            theta_z_bin = theta_z[bin_mask]
            R_bin = R[bin_mask]
            phi_bin = phi[bin_mask]
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
                [R_bin, Jz_shift, phi_bin, theta_z_shift],
                names=["R", "jz", "phi", "theta_z"],
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

    recon_m1 = np.array(recon_m1_all, dtype=float)
    recon_m2 = np.array(recon_m2_all, dtype=float)
    recon_m0 = np.array(recon_m0_all, dtype=float)

    # Reconstruction grid
    fig, axes = plt.subplots(n_phi, n_R, figsize=(n_R * 2, n_phi * 2),
                             sharex=True, sharey=True, constrained_layout=True)
    fig.suptitle("Gaia: Reconstruction m=1 (centered)", fontsize=20)
    rect = patches.Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                              fill=False, edgecolor="black", linewidth=2)
    fig.add_artist(rect)
    for i in range(n_phi):
        for j in range(n_R):
            data = recon_m1[i, j]
            nx, ny = data.shape
            x_edges = np.linspace(-4, 4, nx + 1)
            y_edges = np.linspace(-4, 4, ny + 1)
            axes[i, j].pcolormesh(x_edges, y_edges, data.T, cmap=cmr.prinsenvlag, shading="auto", rasterized=True)
    fig.supxlabel("R (kpc)", fontsize=16)
    fig.supylabel(r"$\phi$ [rad]", fontsize=16)
    plt.savefig("figures/recon_m1_r_phi.pdf", dpi=150)

    # --- Amplitude maps ---
    R_mesh, phi_mesh = np.meshgrid(R_bins[:-1] + 0.2, phi_bins[:-1] + 0.05)
    m1m2_amp_max = np.maximum(m1_amp_all, m2_amp_all)
    vmin, vmax = 0.05, 0.7

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes
    ax1.pcolormesh(R_mesh, phi_mesh, m1_amp_all / m0_amp_all,
                   cmap=cmr.chroma, norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))
    ax2.pcolormesh(R_mesh, phi_mesh, m2_amp_all / m0_amp_all,
                   cmap=cmr.chroma, norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))
    ax3.pcolormesh(R_mesh, phi_mesh, (m1_amp_all + m2_amp_all) / m0_amp_all,
                   cmap=cmr.chroma, norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))
    im = ax4.pcolormesh(R_mesh, phi_mesh, m1m2_amp_max / m0_amp_all,
                        cmap=cmr.chroma, norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))
    for ax, title in zip(axes, ["m=1 Amplitude Ratio", "m=2 Amplitude Ratio",
                                 "Sum(m=1,m=2) Amplitude Ratio", "Max(m=1,m=2) Amplitude Ratio"]):
        ax.set_title(title)
        ax.set_xlabel(r"$R$ [kpc]")
    ax1.set_ylabel(r"$\phi$ [radians]")
    fig.colorbar(im, label="Amplitude Ratio")
    plt.savefig("figures/Gaia_m1_m2_amplitude_ratios_r_phi.pdf", dpi=300)

    # --- Norm maps ---
    xgrid = np.arange(-lss.rootjzmax, lss.rootjzmax + 1e-5, lss.rootjzstep)
    ygrid = np.arange(-lss.rootjzmax, lss.rootjzmax + 1e-5, lss.rootjzstep)
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
    im = ax.pcolormesh(R_mesh, phi_mesh, total_norm, cmap=cmr.chroma, norm=mpl.colors.LogNorm())
    ax.set_title("Norm of m=1 Reconstruction Ratio")
    fig.colorbar(im, label="Amplitude")
    ax.set_xlabel(r"$R$ [kpc]")
    ax.set_ylabel(r"$\phi$ [radians]")
    plt.savefig("figures/norm_m1_total_r_phi.pdf", dpi=150)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
    im = ax.pcolormesh(R_mesh, phi_mesh, subset_norm, cmap=cmr.chroma, norm=mpl.colors.LogNorm())
    ax.set_title("Norm of m=1 Reconstruction Ratio (excl. central area)")
    fig.colorbar(im, label="Amplitude")
    ax.set_xlabel(r"$R$ [kpc]")
    ax.set_ylabel(r"$\phi$ [radians]")
    plt.savefig("figures/norm_m1_subset_r_phi.pdf", dpi=150)

    print("All done.")


if __name__ == "__main__":
    main()
