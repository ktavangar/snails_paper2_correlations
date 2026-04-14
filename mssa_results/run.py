"""
mSSA Run Script

Reads run parameters from runs.toml and executes the mSSA pipeline.

Usage
-----
    python run.py <run_name>                  # full run with movies
    python run.py <run_name> --no-movies      # diagnostic plots only
    python run.py <run_name> --suggest-groups # diagnostic plots + PC group suggestions
"""

import sys
import os
import tomllib
import argparse
from pathlib import Path

import numpy as np
import pyEXP
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr

plt.rc('text', usetex=False)

# --- Paths ---
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_FILE = SCRIPT_DIR / 'runs.toml'

from run_helper_funcs import make_dirs, expand_pc_entry
sys.path.append(str(PROJECT_ROOT / 'code/mssa_analysis/'))
from mssa_viz import (
    MakeAnimations,
    RewindMacroSpiral,
    compute_data_limits,
    plot_eigenvalues, plot_fg_matrices, plot_wcorr, plot_pc_time_series,
    plot_diagnostics_summary,
)


# --- Argument parsing ---
parser = argparse.ArgumentParser(description='Run mSSA pipeline for a named run in runs.toml.')
parser.add_argument('run_name',
                    help='Name of the run as defined in runs.toml.')
parser.add_argument('--movies', action='store_true', default=False,
                    help='Generate face-on animations.')
parser.add_argument('--diagnostics', action='store_true', default=False,
                    help='Generate diagnostic plots.')
parser.add_argument('--macro_fitting', action='store_true', default=False,
                    help='Generate macro fitting plots.')
parser.add_argument('--data_macro_fitting', action='store_true', default=False,
                    help='Generate macro fitting plots for data.')
args = parser.parse_args()

# --- Load config ---
with open(CONFIG_FILE, 'rb') as f:
    config = tomllib.load(f)

if args.run_name not in config.get('runs', {}):
    print(f"Error: run '{args.run_name}' not found in runs.toml.")
    print(f"Available runs: {list(config.get('runs', {}).keys())}")
    sys.exit(1)

defaults   = config.get('defaults', {})
run_config = config['runs'][args.run_name]

def get(key, fallback=None):
    return run_config.get(key, defaults.get(key, fallback))

data_file        = get('data_file')
sim_name         = get('sim_name', 'test')

_sim_dir_map = {'test': 'test_particle', 'B2': 'B2', 'live': 'B2'}
_sim_dir = _sim_dir_map.get(sim_name, sim_name)
default_output_dir = str(SCRIPT_DIR / _sim_dir / args.run_name)
FIG_DIR          = get('output_dir', default_output_dir)
npc              = get('npc', 50)
window_frac      = get('window_frac', 0.5)
channel_name     = get('channel_name', 'one-armed amplitude')
jphi_min         = get('jphi_min', 1000.0)
jbins            = get('jbins', 26)
list_of_pc_lists = [expand_pc_entry(e) for e in get('list_of_pc_lists', [])]

FIG_DIR, INDIVIDUAL_WINDING_DIR, MOVIES_DIR, WINDING_DIR = make_dirs(FIG_DIR)

print(f"Run: {args.run_name}")

###############################
## Fitting Data Macro-Spiral ##
###############################
if args.data_macro_fitting:
    data_ = np.loadtxt(data_file)
    data = data_[:, 1:].T # convert to right shape (channels x time) and drop time column
    DataMacroFitting = RewindMacroSpiral(data, None, jphi_min, jbins, sim_name, channel_name, m=1)
    DataMacroFitting.plot_macro_tfit_over_time(threshold=np.pi/2, savefig=True, fig_dir=WINDING_DIR)
    DATA_WINDING_DIR = os.path.join(INDIVIDUAL_WINDING_DIR, 'data')
    os.makedirs(DATA_WINDING_DIR, exist_ok=True)

    for tstep_ind in range(data.shape[1]):
        DataMacroFitting.plot_fit_and_dipole(tstep_ind, threshold=np.pi/2, savefig=True, fig_dir=DATA_WINDING_DIR)

#####################################
## Run M-SSA and Quick Diagnostics ##
#####################################

if args.diagnostics or args.macro_fitting or args.movies:
    # --- Load coefficients ---
    print('Loading coefficients...')
    coefs0 = pyEXP.coefs.Coefs.factory(data_file)
    coefs  = coefs0.deepcopy()

    # --- mSSA setup ---
    n_channels  = int(len(coefs.getAllCoefs()))
    times       = coefs.Times()
    window      = int(len(times) * window_frac)
    config_mssa = {'mssa_channel': (coefs, [[i] for i in range(n_channels)], [])}

    flags = """
    verbose: false
    """

    print(f'Running mSSA: {n_channels} channels, window={window}, npc={npc}')
    mssa = pyEXP.mssa.expMSSA(config_mssa, window, npc, flags)

    # --- Eigenvalues ---
    print('Computing eigenvalues...')
    ev = mssa.eigenvalues()

# --- Diagnostic plots ---
if args.diagnostics:
    # Full reconstruction across all PCs is required before calling
    # mssa.contrib() (F/G matrices) and mssa.wCorrAll() (W-correlation).
    coefs.zerodata()
    mssa.reconstruct([*range(npc)])

    plot_diagnostics_summary(mssa, ev, times, FIG_DIR)   # combined single-figure version
    plot_eigenvalues(ev, FIG_DIR)                         # individual figures kept for detail
    plot_fg_matrices(mssa, FIG_DIR)
    plot_wcorr(mssa, FIG_DIR)
    plot_pc_time_series(mssa, times, FIG_DIR)

###################################################################
## Per-PC-group loop: reconstruct → macro fitting → movies      ##
## All work for a given PC group is done before moving to the   ##
## next, so mssa.reconstruct() is called exactly once per group.##
###################################################################

if args.macro_fitting or args.movies:
    if not list_of_pc_lists:
        print('Warning: list_of_pc_lists is empty in runs.toml — skipping PC loop.')

    # Build the data movie once before the PC loop (movies only).
    if args.movies:
        data_tbl = np.loadtxt(data_file)
        data_vmin, data_vmax = compute_data_limits(data_tbl)
        print(f'Data movie limits: vmin={data_vmin:.1f}, vmax={data_vmax:.1f}')
        MA = MakeAnimations(mssa, sim_name=sim_name, channel_name=channel_name,
                            times=times, jphi_min=jphi_min, jbins=jbins)
        MA.make_dual_data_mov(
            os.path.join(MOVIES_DIR, 'data_dual.mp4'), data_tbl,
            norm_function=mpl.colors.LogNorm, cmap=cmr.sunburst,
            vmin=data_vmin, vmax=data_vmax,
        )

    for pc_list in list_of_pc_lists:
        print(f'\nProcessing PC group {pc_list}...')

        # --- Single shared reconstruction for this PC group ---
        mssa.reconstruct(pc_list)
        recon = mssa.getReconstructed()
        pc_rc = recon[list(recon.keys())[0]].getAllCoefs()

        # --- Macro-spiral fitting ---
        if args.macro_fitting:
            MS = RewindMacroSpiral(pc_rc, pc_list, jphi_min, jbins,
                                   sim_name, channel_name, m=1)
            MS.plot_macro_tfit_over_time(threshold=np.pi/2,
                                         savefig=True, fig_dir=WINDING_DIR)
            PC_WINDING_DIR = os.path.join(INDIVIDUAL_WINDING_DIR, MS.pc_string)
            os.makedirs(PC_WINDING_DIR, exist_ok=True)
            for tstep in range(pc_rc.shape[1]):
                MS.plot_fit_and_dipole(tstep, threshold=np.pi/2,
                                       savefig=True, fig_dir=PC_WINDING_DIR)

        # --- Movies (reuse the reconstruction computed above) ---
        if args.movies:
            MA.load_reconstruction(pc_rc, pc_list)
            MA.make_pc_movie_pair(MOVIES_DIR, data_vmin, data_vmax, dual=True)

print('Done. All outputs saved to:', FIG_DIR)