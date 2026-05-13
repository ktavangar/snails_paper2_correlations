#!/usr/bin/env python3
"""
Load Gaia FITS data, apply distance + selection-function cuts, convert to
Galactocentric coordinates, compute AGAMA action-angle variables, and save
everything to an HDF5 file.

Input:  Gaia_data_all_v1.fits  (or --input)
Output: Gaia_data_Agama_1kpc_v1.h5  (or --output)

Run with py311 environment.
"""

import argparse
import sys
import os

import numpy as np
import h5py
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord, Galactocentric, Galactic
from gaiaunlimited.selectionfunctions import DR3RVSSelectionFunction
import agama

sys.path.append(os.path.dirname(__file__))
from df_helpers_data import *  # noqa: F401,F403


def main():
    parser = argparse.ArgumentParser(description="Compute AGAMA actions from Gaia FITS data")
    parser.add_argument("--input", default="Gaia_data_all_v1.fits", help="Input FITS file")
    parser.add_argument("--output", default="Gaia_data_Agama_1kpc_v1.h5", help="Output HDF5 file")
    parser.add_argument(
        "--potential",
        default="/Users/Tavangar/Work/packages/Agama/data/PriceWhelan22.ini",
        help="Path to AGAMA potential .ini file",
    )
    parser.add_argument("--max-dist", type=float, default=1000,
                        help="Max photogeometric distance in pc (default 1000)")
    args = parser.parse_args()

    agama.setUnits(mass=1.0, length=1.0, velocity=1.0)
    potential = agama.Potential(args.potential)
    finder = agama.ActionFinder

    # --- Load and filter ---
    print(f"Loading {args.input}...")
    hdul = fits.open(args.input)
    gaia_data = hdul[1].data
    hdul.info()

    dist_mask = gaia_data["r_med_photogeo"] < args.max_dist
    dr3_data = gaia_data[dist_mask]
    print(f"Stars within {args.max_dist} pc: {len(dr3_data)}")

    # --- Initial coord transform for selection function ---
    dr3_icrs_v0 = SkyCoord(
        ra=dr3_data["ra"] * u.degree,
        dec=dr3_data["dec"] * u.degree,
        distance=dr3_data["r_med_photogeo"] * u.pc,
        pm_ra_cosdec=dr3_data["pmra"] * u.mas / u.yr,
        pm_dec=dr3_data["pmdec"] * u.mas / u.yr,
        radial_velocity=dr3_data["radial_velocity"] * u.km / u.s,
        frame="icrs",
    )
    dr3_galactic_v0 = dr3_icrs_v0.transform_to(Galactic())

    # --- Selection function ---
    print("Computing RVS selection function...")
    rvssf = DR3RVSSelectionFunction()
    S_per_star = rvssf.query(
        coords=dr3_galactic_v0,
        g=dr3_data["phot_g_mean_mag"],
        c=dr3_data["phot_bp_mean_mag"] - dr3_data["phot_rp_mean_mag"],
    )

    finite_mask = np.isfinite(S_per_star)
    cut_data = dr3_data[finite_mask]
    S_cut = S_per_star[finite_mask]
    print(f"Stars after selection-function cut: {len(cut_data)}")

    # --- Final coord transform ---
    dr3_icrs = SkyCoord(
        ra=cut_data["ra"] * u.degree,
        dec=cut_data["dec"] * u.degree,
        distance=cut_data["r_med_photogeo"] * u.pc,
        pm_ra_cosdec=cut_data["pmra"] * u.mas / u.yr,
        pm_dec=cut_data["pmdec"] * u.mas / u.yr,
        radial_velocity=cut_data["radial_velocity"] * u.km / u.s,
        frame="icrs",
    )
    dr3_galcen = dr3_icrs.transform_to(Galactocentric())

    x_ = dr3_galcen.x.to(u.kpc)
    y_ = dr3_galcen.y.to(u.kpc)
    z_ = dr3_galcen.z.to(u.kpc)
    vx_ = dr3_galcen.v_x.to(u.km / u.s)
    vy_ = dr3_galcen.v_y.to(u.km / u.s)
    vz_ = dr3_galcen.v_z.to(u.km / u.s)
    hdul.close()

    galcen_array = np.column_stack([
        x_.value, y_.value, z_.value,
        vx_.value, vy_.value, vz_.value,
    ])

    # --- Action-angle computation ---
    print("Computing action-angle variables with AGAMA...")
    af = finder(potential, interp=False)
    actions, angles, freqs = af(galcen_array, angles=True)

    Jr = actions[:, 0]
    Jz = actions[:, 1]
    Jphi = actions[:, 2]
    Theta_r = angles[:, 0]
    Theta_z = angles[:, 1]
    Theta_phi = angles[:, 2]
    Omega_r = freqs[:, 0]
    Omega_z = freqs[:, 1]
    Omega_phi = freqs[:, 2]

    # --- Save ---
    print(f"Saving to {args.output}...")
    with h5py.File(args.output, "w") as f:
        f.create_dataset("Jr", data=Jr)
        f.create_dataset("Jphi", data=Jphi)
        f.create_dataset("Jz", data=Jz)
        f.create_dataset("Omega_r", data=Omega_r)
        f.create_dataset("Omega_phi", data=Omega_phi)
        f.create_dataset("Omega_z", data=Omega_z)
        f.create_dataset("theta_r", data=Theta_r)
        f.create_dataset("theta_phi", data=Theta_phi)
        f.create_dataset("theta_z", data=Theta_z)
        f.create_dataset("x", data=x_.value)
        f.create_dataset("y", data=y_.value)
        f.create_dataset("z", data=z_.value)
        f.create_dataset("vx", data=vx_.value)
        f.create_dataset("vy", data=vy_.value)
        f.create_dataset("vz", data=vz_.value)
        f.create_dataset("S", data=S_cut)
    print("Done.")


if __name__ == "__main__":
    main()
