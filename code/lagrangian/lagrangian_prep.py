"""
Lagrangian mSSA Table Preparation
==================================
Mirrors mssa_prep.py but uses Lagrangian tracking: bins are defined at a
reference timestep and the same particles are followed at all subsequent
timesteps, regardless of where they have moved in (J_phi, theta_phi) space.

Usage
-----
1. Define bins at the reference timestep and save the index assignments:

    from lagrangian_prep import LagrangianMSSATable
    from load_data import load_data_actions

    data_ref = load_data_actions(data_file=..., action_file=...)
    LT = LagrangianMSSATable()
    LT.assign_bins(data_ref)
    LT.save_bin_indices('bin_indices_t40.npy')

2. At every subsequent timestep, load indices and fill table:

    LT.load_bin_indices('bin_indices_t40.npy')
    t = LT.fill_table(timestep, sim='test',
                      data_file=..., action_file=...)
    t.write('t{}.fits'.format(timestep), overwrite=True)

See make_lagrangian_table.py for the MPI batch runner.
"""

import sys
from pathlib import Path
import scipy
import numpy as np
from astropy.table import Table

# Allow imports from the parent code/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from load_actions import *
from df_helpers import LaguerreSnails


class LaguerreSnailsByIndex(LaguerreSnails):
    """
    LaguerreSnails variant that uses a pre-assigned set of particle indices
    (fixed at a reference timestep) rather than selecting particles by their
    current spatial position.

    All computation methods (get_coeffs, get_pitch_phase_angles, etc.) are
    inherited unchanged — the only difference is how self.sel is populated.

    Parameters
    ----------
    data    : DataFrame for the current timestep (all particles)
    indices : array-like of particle indices assigned to this bin at the
              reference timestep
    center  : [J_phi_center, theta_phi_center] — stored as metadata only,
              not used for particle selection
    radius  : [J_phi_radius, theta_phi_radius] — stored as metadata only
    Jz_grid, thetaz_grid, m_max, n_max, time : same as LaguerreSnails
    """

    def __init__(self, data, indices, center, radius,
                 Jz_grid, thetaz_grid, m_max, n_max, time=None):
        # Replicate the attribute setup from LaguerreSnails.__init__
        # without calling select_action_region.
        self.data = data
        self.time = time
        self.center, self.radius = center, radius
        self.Jz_grid, self.thetaz_grid = Jz_grid, thetaz_grid

        self.rootjzmax  = np.sqrt(np.max(Jz_grid))
        self.rootjzstep = np.sqrt(Jz_grid[1]) - np.sqrt(Jz_grid[0])

        self.m_max, self.n_max = m_max, n_max
        self.n_maxs = [n_max] * m_max
        self.ms = np.arange(m_max)

        # Select by stored indices; guard against any missing particles
        available = data.index.intersection(indices)
        self.sel = data.loc[available]

        # Fit Laguerre scale parameter to this particle set
        _, self.a = scipy.stats.expon.fit(
        self.sel.jz[~np.isnan(self.sel.jz)], floc=0)


class LagrangianMSSATable:
    """
    Mirrors MSSATable but with Lagrangian particle tracking.

    The spatial grid (J_phi × theta_phi bins) is defined once in __init__,
    identical to MSSATable. Particle-to-bin assignments are fixed at a
    reference timestep via assign_bins() and reused at all other timesteps.
    """

    def __init__(self, jphi_bounds=[1000, 4000], nphi_bins=[30, 16],
                 m_max=3, n_max=20):

        jphi_c  = np.linspace(jphi_bounds[0], jphi_bounds[1], nphi_bins[0] + 1)
        tphi_c_ = np.linspace(0, 2 * np.pi, nphi_bins[1] + 1)
        self.rad     = [0.5 * (jphi_c[1]  - jphi_c[0]),
                        0.5 * (tphi_c_[1] - tphi_c_[0])]
        tphi_c       = tphi_c_[:-1] + self.rad[1]
        self.centers = np.array(np.meshgrid(jphi_c, tphi_c)).T.reshape(-1, 2)

        self.m_max, self.n_max = m_max, n_max
        self.ms, self.ns = np.arange(m_max), np.arange(n_max)

        self.colnames = [
            'timestep', 'jphi_cen', 'tphi_cen',
            'm0_amp', 'm1_amp', 'm2_amp',
            'pitch_ang_m1', 'pitch_ang_m2',
            'phase_ang_m1', 'phase_ang_m2',
            'pitch_phase_flag_m1', 'pitch_phase_flag_m2',
            'time_since_int_m1', 'time_since_int_m2',
            'n_particles_ref', 'n_particles_cur',
        ]

        self.jz_grid = (np.arange(0, 10, 0.1)) ** 2
        self.tz_grid = np.arange(0, 2 * np.pi, np.pi / 48)

        self.jphi_bins = np.linspace(jphi_bounds[0]-self.rad[0], 
                                jphi_bounds[1]+self.rad[0], 
                                nphi_bins[0] + 2)
        self.thetaphi_bins = np.linspace(0, 2*np.pi, nphi_bins[1] + 1)
        self.bin_indices = None  # set by assign_bins() or load_bin_indices()

    # ------------------------------------------------------------------
    # Bin assignment
    # ------------------------------------------------------------------

    def assign_bins(self, data_ref):
        """
        Assign particle indices to bins based on their position in data_ref
        (the reference timestep DataFrame). Must be called before fill_table.

        Parameters
        ----------
        data_ref : DataFrame with columns jphi, theta_phi (and standard
                   integer index matching all other timestep DataFrames)
        """
        self.bin_indices = []

        jphi_idx = np.digitize(data_ref['jphi'].values, self.jphi_bins) - 1
        thetaphi_idx = np.digitize(data_ref['theta_phi'].values, self.thetaphi_bins) - 1

        data_ref['jphi_bin'] = jphi_idx
        data_ref['thetaphi_bin'] = thetaphi_idx

        bin_idx = {
            key: list(grp.index) for key, grp in data_ref.groupby(['jphi_bin', 'thetaphi_bin'])
        }

        for key, val in bin_idx.items():
            if (key[0] != -1) & (key[0] != len(self.jphi_bins)-1):
                self.bin_indices.append(np.array(val))

        print(f'Assigned {len(self.bin_indices)} bins '
              f'({sum(len(b) for b in self.bin_indices)} total particle-bin assignments).')

    def save_bin_indices(self, path):
        """Save bin index assignments to a .npy file for reuse."""
        np.save(path, np.array(self.bin_indices, dtype=object))
        print(f'Saved bin indices to {path}')

    def load_bin_indices(self, path):
        """Load previously saved bin index assignments."""
        self.bin_indices = np.load(path, allow_pickle=True)
        print(f'Loaded bin indices from {path} '
              f'({len(self.bin_indices)} bins)')

    # ------------------------------------------------------------------
    # Table construction
    # ------------------------------------------------------------------

    def create_empty_table(self):
        return Table(names=self.colnames)

    def create_table(self):
        empty = np.zeros((len(self.centers), len(self.colnames)))
        return Table(empty, names=self.colnames)

    def fill_table(self, timestep, sim, actions_dir=None):
        """
        Compute BFE statistics for all bins at this timestep using the
        Lagrangian particle assignments from the reference timestep.

        Parameters
        ----------
        timestep    : int
        sim         : 'test' or 'live' or 'B2'
        actions_dir : path to directory containing actions files

        Returns
        -------
        t : astropy Table with one row per spatial bin
        """
        if self.bin_indices is None:
            raise RuntimeError(
                'No bin assignments found. '
                'Call assign_bins() or load_bin_indices() first.')

        self.timestep = int(timestep)

        # Load data for this timestep
        if (sim == 'live') | (sim =='B2'):
            data = load_B2_actions(tstep=self.timestep, actions_dir=actions_dir)
        elif sim == 'test':
            data = load_test_actions(tstep=self.timestep, actions_dir=actions_dir)
        else:
            raise ValueError(f"Unknown sim '{sim}'. Expected 'test', 'live', or 'B2'.")

        print(f'Getting coefficients for timestep {self.timestep}...')

        n_bins = len(self.centers)
        all_coeff_array    = np.zeros((n_bins, self.m_max, self.n_max), dtype=np.complex_)
        pitch_angle_m1     = np.zeros(n_bins)
        phase_angle_m1     = np.zeros(n_bins)
        pitch_angle_m2     = np.zeros(n_bins)
        phase_angle_m2     = np.zeros(n_bins)
        pitch_phase_flag_m1 = np.zeros(n_bins)
        pitch_phase_flag_m2 = np.zeros(n_bins)
        time_since_int_m1  = np.zeros(n_bins)
        time_since_int_m2  = np.zeros(n_bins)
        n_particles_ref    = np.array([len(b) for b in self.bin_indices])
        n_particles_cur    = np.zeros(n_bins, dtype=int)

        for i, (cen, indices) in enumerate(zip(self.centers, self.bin_indices)):
            LS = LaguerreSnailsByIndex(
                data, indices, cen, self.rad,
                self.jz_grid, self.tz_grid,
                self.m_max, self.n_max, self.timestep)

            n_particles_cur[i] = len(LS.sel)
            all_coeff_array[i] = LS.get_coeffs()

            (pitch_angle_m1[i], phase_angle_m1[i],
             pitch_phase_flag_m1[i], time_since_int_m1[i]) = LS.get_pitch_phase_angles(m=1)

            (pitch_angle_m2[i], phase_angle_m2[i],
             pitch_phase_flag_m2[i], time_since_int_m2[i]) = LS.get_pitch_phase_angles(m=2)

        t = self.create_table()

        t['timestep']    = self.timestep * np.ones(n_bins)
        t['jphi_cen']    = self.centers[:, 0]
        t['tphi_cen']    = self.centers[:, 1]

        t['m0_amp'] = np.linalg.norm(np.abs(all_coeff_array[:, 0, :]), axis=1)
        t['m1_amp'] = np.linalg.norm(np.abs(all_coeff_array[:, 1, :]), axis=1)
        t['m2_amp'] = np.linalg.norm(np.abs(all_coeff_array[:, 2, :]), axis=1)

        t['pitch_ang_m1']       = pitch_angle_m1
        t['pitch_ang_m2']       = pitch_angle_m2
        t['phase_ang_m1']       = phase_angle_m1
        t['phase_ang_m2']       = phase_angle_m2
        t['pitch_phase_flag_m1'] = pitch_phase_flag_m1
        t['pitch_phase_flag_m2'] = pitch_phase_flag_m2
        t['time_since_int_m1']  = time_since_int_m1
        t['time_since_int_m2']  = time_since_int_m2

        # Extra columns tracking particle count drift over time
        t['n_particles_ref'] = n_particles_ref
        t['n_particles_cur'] = n_particles_cur

        self.t = t
        return self.t

    def save_table(self, out_dir='data/tables_lagrangian'):
        fname = f'{out_dir}/t{self.timestep}.fits'
        self.t.write(fname, overwrite=True)
