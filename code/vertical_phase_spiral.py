"""
Phase-spiral action-angle pipeline.

Takes Gaia astrometry + RVs, transforms to Galactocentric cylindrical coords,
computes global (J_phi, theta_phi) with Agama using the Price-Whelan 2022
Milky Way potential (Gala's MilkyWayPotential2022, stored in Agama as
PriceWhelan22.ini), bins stars in (J_phi, theta_phi), and then fits a local
Phi(z) non-parametrically in each bin (Widmark+2021 style) to produce a
(J_z, theta_z) for every star.

Dependencies: numpy, scipy, astropy, agama.

Author: written for Kiyan, April 2026.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.optimize import minimize
from scipy.stats import gaussian_kde

import astropy.coordinates as coord
import astropy.units as u

import agama


# -----------------------------------------------------------------------------
# Unit system
# -----------------------------------------------------------------------------
# Agama uses a user-specified unit system. We pick kpc / km/s / Msun, which is
# standard for MW dynamics. With these three, time is ~ 0.9778 Gyr and
# G ~ 4.3e-6 kpc (km/s)^2 / Msun. The actions come out in kpc * km/s.
agama.setUnits(length=1.0, velocity=1.0, mass=1.0)  # kpc, km/s, Msun


# -----------------------------------------------------------------------------
# 1. Gaia -> Galactocentric cylindrical
# -----------------------------------------------------------------------------
# Solar parameters consistent with the Price-Whelan+2022 MW model
# (same values Gala uses as defaults in galactocentric_frame_defaults='v4.0').
GC_FRAME = coord.Galactocentric(
    galcen_distance=8.275 * u.kpc,      # GRAVITY 2021
    galcen_v_sun=coord.CartesianDifferential(
        [8.4, 251.8, 8.4] * u.km / u.s   # Drimmel & Poggio 2018 / GRAVITY 2021
    ),
    z_sun=20.8 * u.pc,                   # Bennett & Bovy 2019
)


def gaia_to_galactocentric(
    ra, dec, parallax, pmra, pmdec, radial_velocity,
    parallax_zero_point: float = -0.017,   # Lindegren+2021, mas
    distance: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Transform Gaia DR3 astrometry + RVs to Galactocentric cylindrical.

    Parameters
    ----------
    ra, dec : deg
    parallax : mas (will be corrected for zero point unless `distance` given)
    pmra, pmdec : mas/yr  (pmra is pmra* = pmra*cos(dec), as in Gaia archive)
    radial_velocity : km/s
    parallax_zero_point : mas, subtracted from parallax before inversion
    distance : optional kpc array; if provided, overrides parallax inversion
        (use e.g. Bailer-Jones geometric distances for faint / high-RUWE stars)

    Returns
    -------
    dict with keys R, phi, z (kpc, rad, kpc) and vR, vphi, vz (km/s),
    plus x, y, z (kpc) and vx, vy, vz_cart (km/s).
    """
    ra = np.atleast_1d(ra) * u.deg
    dec = np.atleast_1d(dec) * u.deg
    pmra = np.atleast_1d(pmra) * u.mas / u.yr
    pmdec = np.atleast_1d(pmdec) * u.mas / u.yr
    rv = np.atleast_1d(radial_velocity) * u.km / u.s

    if distance is None:
        plx = (np.atleast_1d(parallax) - parallax_zero_point) * u.mas
        dist = plx.to(u.kpc, equivalencies=u.parallax())
    else:
        dist = np.atleast_1d(distance) * u.kpc

    icrs = coord.SkyCoord(
        ra=ra, dec=dec, distance=dist,
        pm_ra_cosdec=pmra, pm_dec=pmdec, radial_velocity=rv,
        frame="icrs",
    )
    gc = icrs.transform_to(GC_FRAME)

    x = gc.x.to_value(u.kpc)
    y = gc.y.to_value(u.kpc)
    z = gc.z.to_value(u.kpc)
    vx = gc.v_x.to_value(u.km / u.s)
    vy = gc.v_y.to_value(u.km / u.s)
    vz = gc.v_z.to_value(u.km / u.s)

    R = np.hypot(x, y)
    phi = np.arctan2(y, x)
    vR = (x * vx + y * vy) / R
    vphi = (x * vy - y * vx) / R   # = L_z / R

    return dict(x=x, y=y, z=z, vx=vx, vy=vy, vz_cart=vz,
                R=R, phi=phi, vR=vR, vphi=vphi, vz=vz)


# -----------------------------------------------------------------------------
# 2. Global (J_phi, theta_phi) with Agama + MilkyWayPotential2022
# -----------------------------------------------------------------------------
def load_global_potential(ini_path: str = "PriceWhelan22.ini") -> agama.Potential:
    """Load the global MW potential from an Agama .ini file.

    Agama ships PriceWhelan22.ini (= Gala's MilkyWayPotential2022) in
    `agama/data/`. If your install doesn't have it on the search path,
    pass an absolute path.
    """
    return agama.Potential(ini_path)


def compute_global_actions(
    gc: Dict[str, np.ndarray],
    potential: agama.Potential,
) -> Dict[str, np.ndarray]:
    """Compute (J_R, J_phi, J_z) and (theta_R, theta_phi, theta_z)_global
    using Agama's Staeckel-fudge ActionFinder.

    We only use J_phi and theta_phi downstream; the global J_z is discarded
    in favour of the Widmark-style per-bin estimate.
    """
    # Agama expects 6D phase-space in (x,y,z,vx,vy,vz).
    posvel = np.column_stack([gc["x"], gc["y"], gc["z"],
                              gc["vx"], gc["vy"], gc["vz_cart"]])

    af = agama.ActionFinder(potential)
    # angles=True returns (actions, angles, frequencies)
    actions, angles, freqs = af(posvel, angles=True)

    # Agama returns actions in the order (J_R, J_z, J_phi).
    JR, Jz_global, Jphi = actions.T
    thR, thz_global, thphi = angles.T
    OmR, Omz_global, Omphi = freqs.T

    # Wrap theta_phi into [0, 2pi)
    thphi = np.mod(thphi, 2 * np.pi)

    return dict(J_R=JR, J_phi=Jphi, J_z_global=Jz_global,
                theta_R=thR, theta_phi=thphi, theta_z_global=thz_global,
                Omega_R=OmR, Omega_phi=Omphi, Omega_z_global=Omz_global)


# -----------------------------------------------------------------------------
# 3. Binning in (J_phi, theta_phi)
# -----------------------------------------------------------------------------
@dataclass
class BinGrid:
    """Configurable binning in (J_phi, theta_phi).

    Defaults: 20 uniform bins in J_phi between 1000 and 3000 kpc km/s
    (the solar value is ~2020 with our v_sun), and 16 uniform bins in
    theta_phi over [0, 2pi).
    """
    Jphi_edges: np.ndarray = field(
        default_factory=lambda: np.linspace(1000.0, 3000.0, 21)
    )
    theta_phi_edges: np.ndarray = field(
        default_factory=lambda: np.linspace(0.0, 2 * np.pi, 17)
    )

    def assign(self, J_phi: np.ndarray, theta_phi: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray]:
        """Return integer bin indices (iJ, ith). -1 means out of range."""
        iJ = np.digitize(J_phi, self.Jphi_edges) - 1
        ith = np.digitize(theta_phi, self.theta_phi_edges) - 1
        iJ[(iJ < 0) | (iJ >= len(self.Jphi_edges) - 1)] = -1
        ith[(ith < 0) | (ith >= len(self.theta_phi_edges) - 1)] = -1
        return iJ, ith

    def bin_center(self, iJ: int, ith: int) -> Tuple[float, float]:
        Jc = 0.5 * (self.Jphi_edges[iJ] + self.Jphi_edges[iJ + 1])
        tc = 0.5 * (self.theta_phi_edges[ith] + self.theta_phi_edges[ith + 1])
        return Jc, tc


# -----------------------------------------------------------------------------
# 4. Widmark-style non-parametric local Phi(z) fit
# -----------------------------------------------------------------------------
# Strategy (following Widmark, Laporte & de Salas 2021, ApJ 912, 40):
#
#   Inside a bin we assume the vertical DF is in steady state, so
#   f(z, v_z) = f( E_z ),  E_z = v_z^2 / 2 + Phi(z),  Phi(0) = 0.
#
#   The observed (z, v_z) sample gives us an empirical 2D density
#   hat{n}(z, v_z). For any candidate Phi(z), each star has an E_z, and we
#   can build an empirical f_hat(E_z) from the sample and require it to
#   match the marginal. Equivalently: under the correct Phi(z), the
#   *rank* distribution of E_z should look identical when you select
#   stars at different z. We maximise the equivalent 1D log-likelihood
#
#       log L(Phi) = sum_i log f_hat_Phi( E_z,i ) - N * log Z(Phi)
#
#   where f_hat_Phi is a KDE over E_z evaluated on the same sample, and
#   Z(Phi) = int dz dv_z f_hat_Phi(E_z(z,v_z)) normalises over the
#   selection window in (z, v_z).
#
# We parametrise Phi(z) as Phi(z) = 0.5 * a * z^2 * S(z/h), where S is a
# positive monotone spline of |z|/h with control values p_k >= 0, anchored
# at S(0) = 1. This keeps Phi symmetric and smoothly flattens/steepens at
# large z, following Widmark+2021's flexible form. For very small samples
# (N < ~200) the spline is under-constrained and we fall back to a pure
# harmonic Phi(z) = 0.5 * a * z^2 (E_z = isothermal action, closed form).
# -----------------------------------------------------------------------------
@dataclass
class LocalPotentialFit:
    """Result of a per-bin Phi(z) fit."""
    z_grid: np.ndarray
    Phi_grid: np.ndarray
    Phi_interp: Callable[[np.ndarray], np.ndarray]
    dPhi_interp: Callable[[np.ndarray], np.ndarray]
    zmax_window: float     # |z| window used in the fit
    vmax_window: float     # |v_z| window used in the fit
    n_stars: int
    success: bool
    method: str            # "spline" or "harmonic_fallback"


def _make_phi_from_params(params: np.ndarray, h: float, zmax: float):
    """Build Phi(z), dPhi/dz from non-negative spline control points.

    params = [log_a, p_1, ..., p_K]  with p_k in R, mapped to
    monotone-increasing non-negative shape via cumulative softplus.
    """
    log_a = params[0]
    p = params[1:]
    a = np.exp(log_a)

    # Control knots in |z/h|
    K = len(p)
    u_knots = np.linspace(0.0, zmax / h, K + 1)

    # Softplus and cumulative sum -> monotone non-negative delta. Add 1 at 0.
    deltas = np.log1p(np.exp(p))        # > 0
    S_knots = np.concatenate([[1.0], 1.0 + np.cumsum(deltas)])

    S_interp = interp1d(u_knots, S_knots, kind="cubic",
                        bounds_error=False,
                        fill_value=(S_knots[0], S_knots[-1]))

    def Phi(z):
        uu = np.abs(z) / h
        return 0.5 * a * z ** 2 * S_interp(uu)

    def dPhi(z, eps=1e-4):
        # Central difference; cheap and robust.
        return (Phi(z + eps) - Phi(z - eps)) / (2 * eps)

    return Phi, dPhi, a


def _fit_harmonic(z: np.ndarray, vz: np.ndarray
                  ) -> Tuple[Callable, Callable, float]:
    """Closed-form harmonic fallback: Phi(z) = 0.5 * a * z^2.

    The virial estimator a = <v_z^2> / <z^2> gives the matched vertical
    frequency; more robust is the MLE assuming Gaussian f(E_z):
        sigma_z^2 = <v_z^2>, sigma_h^2 = <z^2>, a = sigma_z^2 / sigma_h^2.
    """
    a = np.var(vz) / max(np.var(z), 1e-12)
    omega = np.sqrt(a)
    Phi = lambda zz: 0.5 * a * zz ** 2
    dPhi = lambda zz: a * zz
    return Phi, dPhi, omega


def fit_local_phi(
    z: np.ndarray,
    vz: np.ndarray,
    zmax: Optional[float] = None,
    vmax: Optional[float] = None,
    n_knots: int = 5,
    h_scale: float = 0.4,      # kpc, rough vertical scale height
    min_stars_for_spline: int = 200,
    rng_seed: int = 0,
) -> LocalPotentialFit:
    """Fit a non-parametric, symmetric Phi(z) to one bin's (z, v_z).

    Uses a KDE-based marginal-likelihood objective in E_z. Falls back to
    a harmonic fit when the sample is too small to constrain the spline.
    """
    z = np.asarray(z, dtype=float)
    vz = np.asarray(vz, dtype=float)
    N = len(z)

    # Define the selection window in (z, v_z). Pick generous defaults if
    # the caller didn't set them; these define the integration domain for
    # the normalisation Z(Phi).
    if zmax is None:
        zmax = np.percentile(np.abs(z), 99.0)
        zmax = max(zmax, 0.1)
    if vmax is None:
        vmax = np.percentile(np.abs(vz), 99.0)
        vmax = max(vmax, 10.0)

    # Restrict to the selection window
    mask = (np.abs(z) <= zmax) & (np.abs(vz) <= vmax)
    z = z[mask]
    vz = vz[mask]
    N = len(z)

    if N < min_stars_for_spline:
        Phi_fn, dPhi_fn, _ = _fit_harmonic(z, vz)
        zgrid = np.linspace(-zmax, zmax, 201)
        return LocalPotentialFit(
            z_grid=zgrid, Phi_grid=Phi_fn(zgrid),
            Phi_interp=Phi_fn, dPhi_interp=dPhi_fn,
            zmax_window=zmax, vmax_window=vmax,
            n_stars=N, success=True, method="harmonic_fallback",
        )

    # --- Objective: -log L(Phi) via KDE in E_z ---
    # We integrate the 2D normalisation on a coarse (z, v_z) grid.
    Nz_grid = 41
    Nv_grid = 41
    zgrid = np.linspace(-zmax, zmax, Nz_grid)
    vgrid = np.linspace(-vmax, vmax, Nv_grid)
    ZZ, VV = np.meshgrid(zgrid, vgrid, indexing="ij")
    dz_cell = zgrid[1] - zgrid[0]
    dv_cell = vgrid[1] - vgrid[0]

    def neg_log_like(params):
        Phi_fn, _, _ = _make_phi_from_params(params, h_scale, zmax)
        Ez_data = 0.5 * vz ** 2 + Phi_fn(z)
        if not np.all(np.isfinite(Ez_data)) or np.any(Ez_data < 0):
            return 1e12

        # KDE in E_z space (1D). Bandwidth via Scott's rule.
        try:
            kde = gaussian_kde(Ez_data)
        except np.linalg.LinAlgError:
            return 1e12
        log_f_data = np.log(kde(Ez_data) + 1e-30)

        # Normalisation over the window
        Ez_grid = 0.5 * VV ** 2 + Phi_fn(ZZ)
        f_grid = kde(Ez_grid.ravel()).reshape(Ez_grid.shape)
        Z_norm = f_grid.sum() * dz_cell * dv_cell
        if Z_norm <= 0 or not np.isfinite(Z_norm):
            return 1e12

        return -(log_f_data.sum() - N * np.log(Z_norm))

    # Initial guess: harmonic fit for a, zeros for spline deltas
    a0 = np.var(vz) / max(np.var(z), 1e-6)
    x0 = np.concatenate([[np.log(a0)], np.zeros(n_knots)])

    res = minimize(neg_log_like, x0, method="Nelder-Mead",
                   options=dict(xatol=1e-3, fatol=1e-3, maxiter=2000))
    Phi_fn, dPhi_fn, _ = _make_phi_from_params(res.x, h_scale, zmax)

    z_out = np.linspace(-zmax, zmax, 401)
    return LocalPotentialFit(
        z_grid=z_out, Phi_grid=Phi_fn(z_out),
        Phi_interp=Phi_fn, dPhi_interp=dPhi_fn,
        zmax_window=zmax, vmax_window=vmax,
        n_stars=N, success=bool(res.success), method="spline",
    )


# -----------------------------------------------------------------------------
# 5. (J_z, theta_z) from a 1D Phi(z)  --- vectorised
# -----------------------------------------------------------------------------
# Strategy: in a given bin we know Phi(z) on a dense symmetric grid. Build
# one 1D table J_z(E) and one 1D table T_quarter(E) (= quarter-period for the
# orbit of energy E, equal to the time to go from z=0 with v_z>0 to the
# turning point z_t(E)). Also build a 2D table t(z, E) = time from z=0 with
# v_z>0 to reach z, restricted to 0 <= z <= z_t(E). Then every star is a
# fast interp/broadcast into these tables.
#
# With Nz ~ 512 grid points in z and the same in E, these tables are tiny
# (<~2 MB/bin) and the full O(N_stars) work becomes a handful of np.interp
# calls. Runtime on 3e7 stars is dominated by the per-bin tabulation (a
# few ms) plus the final interpolation passes (seconds total on one core).
# -----------------------------------------------------------------------------
def _build_bin_tables(fit: LocalPotentialFit, n_E: int = 512, n_z: int = 512
                      ) -> Dict[str, np.ndarray]:
    """Pre-tabulate J_z(E), T_full(E), and t(z, E) for one bin's Phi(z).

    All integrals use cumulative trapezoidal rules on a symmetric z-grid.
    The 2D table t_table[k, j] is time-from-z=0 to reach z_grid[k] at
    energy E_grid[j], clipped to the turning point z_t(E_j).
    """
    Phi = fit.Phi_interp
    zmax = fit.zmax_window

    # Positive-side z grid used for all 1D sub-integrals.
    z_pos = np.linspace(0.0, zmax, n_z)
    Phi_pos = np.asarray(Phi(z_pos), dtype=float)
    # Enforce monotone increasing Phi on the positive side (KDE/spline wiggle
    # can make it slightly non-monotone near the top of the window).
    Phi_pos = np.maximum.accumulate(Phi_pos)
    Phi_max = Phi_pos[-1]

    # Energy grid, strictly below Phi_max so turning points stay in-window.
    E_min = max(Phi_pos[1] * 1.001, 1e-6)
    E_max = Phi_max * 0.999
    E_grid = np.linspace(E_min, E_max, n_E)

    # For each E, the turning point z_t(E) via inverting the monotone Phi_pos.
    # np.interp needs monotone-increasing xp, which we already have.
    z_t = np.interp(E_grid, Phi_pos, z_pos)                       # (n_E,)

    # --- J_z(E) = (2/pi) * int_0^{z_t(E)} sqrt(2 (E - Phi(z))) dz ---
    # Compute integrand[k, j] = sqrt(max(0, 2 (E_j - Phi(z_k)))) on the full
    # (z_pos, E_grid) grid, then trapezoidal-integrate in z up to z_t(E_j)
    # using a fractional-endpoint correction.
    two_diff = 2.0 * (E_grid[None, :] - Phi_pos[:, None])         # (n_z, n_E)
    sJ = np.sqrt(np.maximum(two_diff, 0.0))                       # (n_z, n_E)
    sT = 1.0 / np.sqrt(np.maximum(two_diff, 1e-14))               # (n_z, n_E)
    # At/beyond the turning point, v=0, so sT would diverge; we'll only use
    # cumulative integrals up to the turning-point bin, adding the tail
    # analytically using the local harmonic approximation near the turnaround.

    # Cumulative trapezoid in z for J (well-behaved, integrand -> 0 at zt)
    dz = z_pos[1] - z_pos[0]
    cumJ = np.concatenate([np.zeros((1, n_E)),
                           0.5 * (sJ[:-1] + sJ[1:]) * dz], axis=0).cumsum(axis=0)

    # Cumulative trapezoid in z for T, stopping at the cell just before z_t.
    # Near z -> z_t, integrand ~ 1/sqrt(omega^2 (z_t^2 - z^2)) which has an
    # integrable square-root singularity. We handle the last partial cell
    # below with a closed-form harmonic tail.
    cumT = np.concatenate([np.zeros((1, n_E)),
                           0.5 * (sT[:-1] + sT[1:]) * dz], axis=0).cumsum(axis=0)

    # For each energy column j, find the index k_j with z_pos[k_j] <= z_t[j]
    # < z_pos[k_j+1]. Use searchsorted on z_pos.
    k_idx = np.searchsorted(z_pos, z_t, side="right") - 1          # (n_E,)
    k_idx = np.clip(k_idx, 0, n_z - 2)
    j_arange = np.arange(n_E)

    # Fractional position of z_t inside its cell
    z_lo = z_pos[k_idx]                                            # (n_E,)
    frac = (z_t - z_lo) / dz                                       # in [0,1]

    # ---- J_z tail: trapezoid from z_lo to z_t with integrand 0 at z_t ----
    sJ_lo = sJ[k_idx, j_arange]                                    # (n_E,)
    J_tail = 0.5 * sJ_lo * (z_t - z_lo)                            # endpoint 0
    J_full = cumJ[k_idx, j_arange] + J_tail                        # (n_E,)
    Jz_of_E = (2.0 / np.pi) * J_full                               # (n_E,)

    # ---- T_quarter tail: closed-form harmonic tail from z_lo to z_t ----
    # Near z_t: E - Phi(z) ~ -Phi'(z_t) * (z - z_t)  (with Phi'(z_t) > 0)
    # Actually 2(E - Phi(z)) ~ 2 * Phi'(z_t) * (z_t - z), so
    # int_{z_lo}^{z_t} dz/sqrt(2(E-Phi)) ~ sqrt(2(z_t - z_lo)/Phi'(z_t)).
    # Estimate Phi'(z_t) by finite difference on the cell.
    Phi_lo = Phi_pos[k_idx]
    Phi_hi = Phi_pos[np.clip(k_idx + 1, 0, n_z - 1)]
    dPhi_dz_at_zt = np.maximum((Phi_hi - Phi_lo) / dz, 1e-12)
    T_tail = np.sqrt(2.0 * np.maximum(z_t - z_lo, 0.0) / dPhi_dz_at_zt)
    T_quarter = cumT[k_idx, j_arange] + T_tail                     # (n_E,)
    T_full_arr = 4.0 * T_quarter                                   # (n_E,)
    Omega_z = 2.0 * np.pi / np.maximum(T_full_arr, 1e-30)          # (n_E,)

    # ---- t(z, E): time from z=0, v_z>0 to reach z (0 <= z <= z_t(E)) ----
    # This is cumT[k, j] for z = z_pos[k] <= z_t(E_j), and saturates at
    # T_quarter for z >= z_t(E_j).
    t_table = cumT.copy()                                          # (n_z, n_E)
    # mask of cells past the turning point
    past = z_pos[:, None] > z_t[None, :]
    t_table[past] = 0.0  # will be overwritten when we saturate in the evaluator

    return dict(
        z_pos=z_pos, E_grid=E_grid, z_t=z_t,
        cumT=cumT, T_quarter=T_quarter, T_full=T_full_arr, Omega_z=Omega_z,
        Jz_of_E=Jz_of_E, Phi_pos=Phi_pos,
    )


def jz_theta_z_from_local_phi(
    z: np.ndarray,
    vz: np.ndarray,
    fit: LocalPotentialFit,
    n_E: int = 512,
    n_z: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised (J_z, theta_z) for all stars in one bin.

    Builds per-bin 1D/2D tables once, then uses np.interp / bilinear
    interpolation to evaluate every star simultaneously. Scales as
    O(N_stars + n_E * n_z) per bin.

    Conventions: theta_z in [0, 2*pi), increasing with time. Starting at
    z=0 with v_z>0 gives theta_z=0; quarter turns at z=+z_t, z=0 (v_z<0),
    z=-z_t correspond to pi/2, pi, 3pi/2.
    """
    z = np.ascontiguousarray(z, dtype=float)
    vz = np.ascontiguousarray(vz, dtype=float)
    N = len(z)

    Phi = fit.Phi_interp
    Ez = 0.5 * vz ** 2 + np.asarray(Phi(z), dtype=float)

    tab = _build_bin_tables(fit, n_E=n_E, n_z=n_z)
    z_pos = tab["z_pos"]
    E_grid = tab["E_grid"]
    z_t_grid = tab["z_t"]
    cumT = tab["cumT"]
    T_quarter = tab["T_quarter"]
    Omega_z_grid = tab["Omega_z"]
    Jz_grid = tab["Jz_of_E"]

    # Clip energies to the tabulated range. Stars outside are flagged NaN.
    good = np.isfinite(Ez) & (Ez > E_grid[0]) & (Ez < E_grid[-1])
    Jz_out = np.full(N, np.nan)
    thz_out = np.full(N, np.nan)

    if not np.any(good):
        return Jz_out, thz_out

    Ez_g = Ez[good]
    z_g = z[good]
    vz_g = vz[good]

    # --- J_z via 1D interp in E ---
    Jz_out[good] = np.interp(Ez_g, E_grid, Jz_grid)

    # --- theta_z via 2D interp of t(|z|, E), then quadrant mapping ---
    # Bilinear interpolation on the regular (z_pos, E_grid) grid.
    abs_z = np.abs(z_g)

    # Index into E grid
    jE = np.searchsorted(E_grid, Ez_g) - 1
    jE = np.clip(jE, 0, n_E - 2)
    dE = E_grid[1] - E_grid[0]
    wE = (Ez_g - E_grid[jE]) / dE     # in [0, 1]

    # Turning point for each star (linear in E)
    zt_star = (1 - wE) * z_t_grid[jE] + wE * z_t_grid[jE + 1]
    # Clip |z| to just inside its own turning point to avoid crossing out
    abs_z_c = np.minimum(abs_z, np.maximum(zt_star - 1e-9, 0.0))

    # Index into z grid
    dz = z_pos[1] - z_pos[0]
    kZ = np.minimum((abs_z_c / dz).astype(np.int64), n_z - 2)
    wZ = (abs_z_c - z_pos[kZ]) / dz

    # Look up cumT at the four corners
    t00 = cumT[kZ,     jE]
    t10 = cumT[kZ + 1, jE]
    t01 = cumT[kZ,     jE + 1]
    t11 = cumT[kZ + 1, jE + 1]
    t_z = ((1 - wZ) * (1 - wE) * t00 + wZ * (1 - wE) * t10
           + (1 - wZ) * wE * t01 + wZ * wE * t11)

    # Saturate at the quarter period for the star's energy
    Tq_star = (1 - wE) * T_quarter[jE] + wE * T_quarter[jE + 1]
    t_z = np.minimum(t_z, Tq_star)

    Omega_star = (1 - wE) * Omega_z_grid[jE] + wE * Omega_z_grid[jE + 1]
    phase_q = Omega_star * t_z   # in [0, pi/2]

    theta = np.empty_like(phase_q)
    # Quadrants based on signs of (z, v_z)
    q1 = (z_g >= 0) & (vz_g >= 0)     # z>0, vz>0: 0 -> pi/2
    q2 = (z_g >= 0) & (vz_g < 0)      # z>0, vz<0: pi/2 -> pi
    q3 = (z_g < 0)  & (vz_g < 0)      # z<0, vz<0: pi -> 3pi/2
    q4 = (z_g < 0)  & (vz_g >= 0)     # z<0, vz>0: 3pi/2 -> 2pi
    theta[q1] = phase_q[q1]
    theta[q2] = np.pi - phase_q[q2]
    theta[q3] = np.pi + phase_q[q3]
    theta[q4] = 2 * np.pi - phase_q[q4]

    thz_out[good] = np.mod(theta, 2 * np.pi)
    return Jz_out, thz_out


# -----------------------------------------------------------------------------
# 6. Top-level driver
# -----------------------------------------------------------------------------
@dataclass
class PhaseSpiralResult:
    # Galactocentric coordinates
    gc: Dict[str, np.ndarray]
    # Global action-angles from Agama
    global_aa: Dict[str, np.ndarray]
    # Per-star bin indices
    iJ: np.ndarray
    ith: np.ndarray
    # Per-star local vertical action/angle
    J_z: np.ndarray
    theta_z: np.ndarray
    # Per-bin Phi(z) fits, keyed by (iJ, ith)
    fits: Dict[Tuple[int, int], LocalPotentialFit]
    grid: BinGrid


def run_pipeline(
    ra, dec, parallax, pmra, pmdec, radial_velocity,
    distance: Optional[np.ndarray] = None,
    potential_ini: str = "PriceWhelan22.ini",
    grid: Optional[BinGrid] = None,
    fit_kwargs: Optional[dict] = None,
    verbose: bool = True,
) -> PhaseSpiralResult:
    """End-to-end: Gaia columns -> per-star (J_phi, theta_phi, J_z, theta_z)."""
    if grid is None:
        grid = BinGrid()
    if fit_kwargs is None:
        fit_kwargs = {}

    if verbose:
        print("[1/4] Gaia -> Galactocentric ...")
    gc = gaia_to_galactocentric(ra, dec, parallax, pmra, pmdec,
                                radial_velocity, distance=distance)

    if verbose:
        print(f"[2/4] Loading potential {potential_ini} and computing global (J_phi, theta_phi) ...")
    pot = load_global_potential(potential_ini)
    global_aa = compute_global_actions(gc, pot)

    if verbose:
        print("[3/4] Assigning (J_phi, theta_phi) bins ...")
    iJ, ith = grid.assign(global_aa["J_phi"], global_aa["theta_phi"])

    N = len(iJ)
    Jz = np.full(N, np.nan)
    thz = np.full(N, np.nan)
    fits: Dict[Tuple[int, int], LocalPotentialFit] = {}

    if verbose:
        n_bins = (len(grid.Jphi_edges) - 1) * (len(grid.theta_phi_edges) - 1)
        print(f"[4/4] Fitting local Phi(z) in up to {n_bins} bins ...")

    # Iterate over occupied bins only
    bin_ids = np.stack([iJ, ith], axis=1)
    unique_bins = {tuple(b) for b in bin_ids if b[0] >= 0 and b[1] >= 0}

    for (bi, bj) in sorted(unique_bins):
        sel = (iJ == bi) & (ith == bj)
        n_sel = int(sel.sum())
        if n_sel < 20:
            if verbose:
                print(f"  bin ({bi},{bj}): {n_sel} stars -- skipping (too few)")
            continue
        fit = fit_local_phi(gc["z"][sel], gc["vz"][sel], **fit_kwargs)
        fits[(bi, bj)] = fit
        Jz_b, thz_b = jz_theta_z_from_local_phi(
            gc["z"][sel], gc["vz"][sel], fit
        )
        Jz[sel] = Jz_b
        thz[sel] = thz_b
        if verbose:
            print(f"  bin ({bi},{bj}): N={n_sel}, method={fit.method}")

    return PhaseSpiralResult(
        gc=gc, global_aa=global_aa,
        iJ=iJ, ith=ith,
        J_z=Jz, theta_z=thz,
        fits=fits, grid=grid,
    )


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Replace with your real Gaia arrays (e.g. from astroquery / a local FITS).
    # Columns expected: ra, dec, parallax, pmra, pmdec, radial_velocity.
    import sys
    if len(sys.argv) > 1:
        import astropy.table as at
        tab = at.Table.read(sys.argv[1])
        res = run_pipeline(
            tab["ra"], tab["dec"], tab["parallax"],
            tab["pmra"], tab["pmdec"], tab["radial_velocity"],
        )
        print("J_z finite fraction:", np.mean(np.isfinite(res.J_z)))
        print("theta_z finite fraction:", np.mean(np.isfinite(res.theta_z)))
    else:
        print("Pass a Gaia table path (FITS / ECSV / CSV) as argv[1], "
              "or import run_pipeline() and call it on your arrays.")