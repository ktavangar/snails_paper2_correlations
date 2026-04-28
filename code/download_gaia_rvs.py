"""
Download Gaia DR3 RVS sample + Bailer-Jones 2021 geometric/photogeometric
distances from the Gaia archive, filtered to sensible quality cuts for
phase-spiral / disk dynamics work.

What this gets you
------------------
For each source with a published Gaia DR3 radial velocity:
    source_id, ra, dec, parallax, parallax_error, pmra, pmra_error,
    pmdec, pmdec_error, radial_velocity, radial_velocity_error,
    phot_g_mean_mag, bp_rp, ruwe,
    r_med_geo, r_lo_geo, r_hi_geo,                # Bailer-Jones geometric
    r_med_photogeo, r_lo_photogeo, r_hi_photogeo  # Bailer-Jones photogeometric

The join is on source_id with `external.gaiaedr3_distance`, which is the
Bailer-Jones, Rybizki, Fouesneau, Demleitner & Andrae (2021, AJ 161, 147)
distance catalogue for Gaia EDR3 (source_ids are identical in DR3).

Dependencies
------------
    pip install astroquery pyarrow

Login
-----
For large (>~few million rows) downloads, log in to the Gaia archive so
the results can be written to a user table and retrieved without the
anonymous row limit:

    from astroquery.gaia import Gaia
    Gaia.login(user="your_cosmos_user")    # prompts for password
    # or: Gaia.login_gui()

Or register anonymously by calling `download_rvs_sample(..., async_query=True)`
which will still work for queries up to ~3e6 rows but may be slower.

Usage
-----
    python download_gaia_rvs.py --out gaia_rvs_bj.parquet

    # or from Python:
    from download_gaia_rvs import download_rvs_sample
    tab = download_rvs_sample(out_path="gaia_rvs_bj.parquet",
                              max_plx_over_err=5.0,
                              max_ruwe=1.4)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

# Default ADQL. `JOIN external.gaiaedr3_distance d USING (source_id)` is the
# canonical way; the table name `external.gaiaedr3_distance` is what the
# Gaia archive exposes Bailer-Jones 2021 under.
DEFAULT_ADQL = """
SELECT
    g.source_id,
    g.ra, g.dec,
    g.parallax, g.parallax_error,
    g.pmra, g.pmra_error,
    g.pmdec, g.pmdec_error,
    g.radial_velocity, g.radial_velocity_error,
    g.phot_g_mean_mag, g.bp_rp,
    g.ruwe,
    d.r_med_geo, d.r_lo_geo, d.r_hi_geo,
    d.r_med_photogeo, d.r_lo_photogeo, d.r_hi_photogeo
FROM gaiadr3.gaia_source AS g
JOIN external.gaiaedr3_distance AS d
    USING (source_id)
WHERE g.radial_velocity IS NOT NULL
  AND g.parallax IS NOT NULL
  AND g.parallax > 0
  AND g.parallax_over_error > {plx_snr:.3f}
  AND g.ruwe < {max_ruwe:.3f}
  AND g.radial_velocity_error < {max_rv_err:.1f}
  {extra_cuts}
"""


def build_query(
    plx_snr: float = 5.0,
    max_ruwe: float = 1.4,
    max_rv_err: float = 20.0,
    extra_cuts: str = "",
) -> str:
    """Assemble the ADQL string with the given quality thresholds.

    Parameters
    ----------
    plx_snr : minimum parallax / parallax_error.
    max_ruwe : maximum renormalised unit-weight error (astrometric quality).
    max_rv_err : maximum radial_velocity_error in km/s.
    extra_cuts : raw ADQL appended with a leading `AND`, e.g.
        "AND g.phot_g_mean_mag < 14"  to restrict to brighter stars.
    """
    return DEFAULT_ADQL.format(
        plx_snr=plx_snr, max_ruwe=max_ruwe, max_rv_err=max_rv_err,
        extra_cuts=extra_cuts,
    )


def download_rvs_sample(
    out_path: str = "gaia_rvs_bj.parquet",
    plx_snr: float = 5.0,
    max_ruwe: float = 1.4,
    max_rv_err: float = 20.0,
    extra_cuts: str = "",
    async_query: bool = True,
    row_limit: Optional[int] = None,
    dump_to_file: bool = True,
    user: Optional[str] = None,
    password: Optional[str] = None,
):
    """Run the download and return an astropy Table.

    Parameters
    ----------
    out_path : file to write. Extension determines format:
        .parquet (recommended), .fits, .csv, .ecsv all work.
    async_query : use the async TAP endpoint. Required for >3e6 rows.
    row_limit : if not None, set Gaia.ROW_LIMIT. -1 means unlimited.
    dump_to_file : if False, skip writing to disk and just return the Table.
    user, password : optional Gaia archive credentials. If `user` is given
        but `password` is None, `Gaia.login` will prompt.
    """
    from astroquery.gaia import Gaia

    if user is not None:
        # password=None makes astroquery prompt interactively
        Gaia.login(user=user, password=password)

    if row_limit is not None:
        Gaia.ROW_LIMIT = row_limit
    else:
        # -1 removes the default 2000-row cap; logged-in async is needed for
        # very large pulls.
        Gaia.ROW_LIMIT = -1

    adql = build_query(plx_snr=plx_snr, max_ruwe=max_ruwe,
                       max_rv_err=max_rv_err, extra_cuts=extra_cuts)

    print("Submitting ADQL query to Gaia archive ...")
    print(adql)

    if async_query:
        job = Gaia.launch_job_async(adql, dump_to_file=False)
    else:
        job = Gaia.launch_job(adql, dump_to_file=False)
    tab = job.get_results()
    print(f"Retrieved {len(tab):,} rows, {len(tab.colnames)} columns.")

    if dump_to_file:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        ext = out.suffix.lower()
        if ext == ".parquet":
            # Parquet is a good choice for ~10^7 rows: fast, compressed,
            # and pandas/polars-friendly. Requires pyarrow.
            df = tab.to_pandas()
            df.to_parquet(out, index=False)
        elif ext in (".fits", ".fit"):
            tab.write(out, overwrite=True)
        elif ext == ".csv":
            tab.write(out, format="csv", overwrite=True)
        elif ext in (".ecsv", ".txt"):
            tab.write(out, format="ascii.ecsv", overwrite=True)
        else:
            raise ValueError(f"Unsupported output extension: {ext}")
        print(f"Wrote {out.resolve()}")

    return tab


# -----------------------------------------------------------------------------
# Post-processing helpers
# -----------------------------------------------------------------------------
def pick_distance(tab, prefer: str = "photogeo"):
    """Add a `distance_kpc` column using Bailer-Jones distances.

    `prefer` in {"photogeo", "geo"}: if `photogeo` is NaN (Bailer-Jones
    doesn't always publish it when photometry is unreliable), fall back
    to `geo`.
    """
    import numpy as np
    r_pg = np.asarray(tab["r_med_photogeo"], dtype=float)
    r_g = np.asarray(tab["r_med_geo"], dtype=float)
    if prefer == "photogeo":
        d_pc = np.where(np.isfinite(r_pg), r_pg, r_g)
    elif prefer == "geo":
        d_pc = np.where(np.isfinite(r_g), r_g, r_pg)
    else:
        raise ValueError(f"prefer must be 'photogeo' or 'geo', got {prefer!r}")
    tab["distance_kpc"] = d_pc / 1000.0
    return tab


def load_for_pipeline(path: str, prefer: str = "photogeo"):
    """Read a saved file and return arrays ready for `run_pipeline`.

    Returns a dict with keys ra, dec, parallax, pmra, pmdec,
    radial_velocity, distance (all numpy arrays in Gaia archive units
    and kpc for distance).
    """
    import numpy as np
    import astropy.table as at
    ext = Path(path).suffix.lower()
    if ext == ".parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        tab = at.Table.from_pandas(df)
    else:
        tab = at.Table.read(path)
    tab = pick_distance(tab, prefer=prefer)
    return dict(
        ra=np.asarray(tab["ra"], float),
        dec=np.asarray(tab["dec"], float),
        parallax=np.asarray(tab["parallax"], float),
        pmra=np.asarray(tab["pmra"], float),
        pmdec=np.asarray(tab["pmdec"], float),
        radial_velocity=np.asarray(tab["radial_velocity"], float),
        distance=np.asarray(tab["distance_kpc"], float),
        source_id=np.asarray(tab["source_id"]),
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--out", default="gaia_rvs_bj.parquet",
                   help="Output file (.parquet/.fits/.csv/.ecsv).")
    p.add_argument("--plx-snr", type=float, default=5.0,
                   help="Minimum parallax_over_error (default 5).")
    p.add_argument("--max-ruwe", type=float, default=1.4,
                   help="Maximum RUWE (default 1.4).")
    p.add_argument("--max-rv-err", type=float, default=20.0,
                   help="Maximum radial_velocity_error in km/s.")
    p.add_argument("--extra-cuts", default="",
                   help="Extra ADQL to append, e.g. \"AND g.phot_g_mean_mag < 14\".")
    p.add_argument("--sync", action="store_true",
                   help="Use synchronous query (only for tiny samples).")
    p.add_argument("--user", default=None,
                   help="Gaia archive username (triggers login prompt for password).")
    args = p.parse_args()

    download_rvs_sample(
        out_path=args.out,
        plx_snr=args.plx_snr,
        max_ruwe=args.max_ruwe,
        max_rv_err=args.max_rv_err,
        extra_cuts=args.extra_cuts,
        async_query=not args.sync,
        user=args.user,
    )


if __name__ == "__main__":
    main()