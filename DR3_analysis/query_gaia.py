#!/usr/bin/env python3
"""
Download Gaia DR3 stars with radial velocities and photogeometric distances.

Output: Gaia_data_all_v1.fits (or --output path)

Run with py311 environment.
"""

import argparse
from astroquery.gaia import Gaia


def main():
    parser = argparse.ArgumentParser(description="Query Gaia DR3 via async TAP job")
    parser.add_argument("--user", default="ktavan01", help="Gaia archive username")
    parser.add_argument("--password", required=True, help="Gaia archive password")
    parser.add_argument("--output", default="Gaia_data_all_v1.fits", help="Output FITS filename")
    parser.add_argument("--max-dist", type=float, default=2000,
                        help="Maximum photogeometric distance in pc (default 2000)")
    args = parser.parse_args()

    Gaia.login(user=args.user, password=args.password)

    query = f"""
SELECT
    g.source_id, g.ra, g.dec, g.parallax, g.parallax_error,
    g.pmra, g.pmra_error, g.pmdec, g.pmdec_error,
    g.radial_velocity, g.radial_velocity_error,
    g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag,
    d.r_med_photogeo, d.r_lo_photogeo, d.r_hi_photogeo
FROM gaiadr3.gaia_source AS g
JOIN external.gaiaedr3_distance AS d
    ON g.source_id = d.source_id
WHERE g.radial_velocity IS NOT NULL
  AND d.r_med_photogeo IS NOT NULL
  AND d.r_med_photogeo < {args.max_dist}
"""

    print("Launching async job...")
    job = Gaia.launch_job_async(
        query,
        background=True,
        dump_to_file=True,
        output_format="fits",
        output_file=args.output,
    )
    print(f"Job ID:  {job.jobid}")
    print(f"Phase:   {job.get_phase()}")
    print("Waiting for job to complete...")
    job.wait_for_job_end()
    print(f"Done. Results written to {args.output}")


if __name__ == "__main__":
    main()
