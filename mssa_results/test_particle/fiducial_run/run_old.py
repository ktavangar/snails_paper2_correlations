"""
Test Particle Fiducial mSSA Run

Absolute one-armed amplitude, window length = 1/2 total time series. Run covers t=40-300 (2.6 Gyr).
Converted from notebooks/paper2_tp_nbs/fiducial_run.ipynb.
"""

import sys
import os
from pathlib import Path

import numpy as np
import pyEXP
import matplotlib.pyplot as plt

plt.rc('text', usetex=False)

# --- Paths ---
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parents[2]  # mssa_results/test_particle/fiducial_run -> project root

sys.path.append(str(PROJECT_ROOT / 'code'))
import helper

DATA_DIR = PROJECT_ROOT / 'data' / 'mSSA_channels_Kiyan_test_t40-300'
FIG_DIR = str(SCRIPT_DIR / 'figures')
FACE_ON_DIR = str(SCRIPT_DIR / 'figures' / 'face-on_plots')

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(FACE_ON_DIR, exist_ok=True)

# --- Input files ---
fname_m1_amp = str(DATA_DIR / 'm1_amp_bins_j25_t16.dat')
fname_m1_rel_amp = str(DATA_DIR / 'm1_amp_rel_bins_j25_t16.dat')

# --- Load coefficients ---
print('Loading coefficients...')
coefs_m1_amp0 = pyEXP.coefs.Coefs.factory(fname_m1_amp)
coefs_m1_rel_amp0 = pyEXP.coefs.Coefs.factory(fname_m1_rel_amp)

coefs_m1_amp = coefs_m1_amp0.deepcopy()
coefs_m1_rel_amp = coefs_m1_rel_amp0.deepcopy()

# --- mSSA setup ---
n_channels = int(len(coefs_m1_amp.getAllCoefs()))
times = coefs_m1_amp.Times()

keylst_m1_amp = [[i] for i in range(n_channels)]
config = {"snails_m1_amp": (coefs_m1_amp, keylst_m1_amp, [])}

window = int(len(times) / 2)
npc = 50

flags = """
verbose: false
"""

print(f'Running mSSA: {n_channels} channels, window={window}, npc={npc}')
mssa = pyEXP.mssa.expMSSA(config, window, npc, flags)

# --- Eigenvalues and reconstruction ---
print('Computing eigenvalues...')
ev = mssa.eigenvalues()

coefs_m1_amp.zerodata()
mssa.reconstruct([*range(npc)])

# --- Diagnostic plots ---
helper.plot_eigenvalues(ev, FIG_DIR)
helper.plot_fg_matrices(mssa, FIG_DIR)
helper.plot_wcorr(mssa, FIG_DIR)
helper.plot_pc_time_series(mssa, times, FIG_DIR)

# --- Face-on animations ---
print('Creating face-on animations...')
list_of_pc_lists = [
    [0, 1], [2, 3], [4, 5],
    list(range(4, 14)), list(range(4, 22)),
    list(range(21, 30)), list(range(0, 4)),
    list(range(0, 14)), list(range(0, 17)), list(range(0, 22))
]

helper.make_face_on_movies(
    mssa, fname_m1_amp, times, FACE_ON_DIR, list_of_pc_lists,
    sim_name='test', channel_name='one-armed amplitude',
    jphi_min=1000, jbins=26)

print('Done. All outputs saved to:', FIG_DIR)
