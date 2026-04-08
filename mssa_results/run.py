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
import matplotlib.pyplot as plt

plt.rc('text', usetex=False)

# --- Paths ---
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_FILE = SCRIPT_DIR / 'runs.toml'

from run_helper_funcs import make_dirs, expand_pc_entry
sys.path.append(str(PROJECT_ROOT / 'code/mssa_analysis/'))
import diagnostics
from macro_rewinding import RewindMacroSpiral


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

FIG_DIR, INDIVIDUAL_WINDING_DIR, DIPOLE_DIR, MOVIES_DIR, WINDING_DIR = make_dirs(FIG_DIR)

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

    DATA_DIPOLE_DIR = os.path.join(DIPOLE_DIR, 'data')
    os.makedirs(DATA_DIPOLE_DIR, exist_ok=True)
    for tstep_ind in range(len(data_)):
        DataMacroFitting.plot_fitting_tstep(tstep_ind, threshold=np.pi/2, savefig=True, fig_dir=DATA_WINDING_DIR)
        DataMacroFitting.make_rewind_dipole_fig(tstep_ind, savefig=True, fig_dir=DATA_DIPOLE_DIR)

#####################################
## Run M-SSA and Quick Diagnostics ##
#####################################

if args.diagnostics | args.macro_fitting | args.movies:
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

    # --- Eigenvalues and reconstruction ---
    print('Computing eigenvalues...')
    ev = mssa.eigenvalues()

    coefs.zerodata()
    mssa.reconstruct([*range(npc)])

# --- Diagnostic plots ---
if args.diagnostics:
    diagnostics.plot_eigenvalues(ev, FIG_DIR)
    diagnostics.plot_fg_matrices(mssa, FIG_DIR)
    diagnostics.plot_wcorr(mssa, FIG_DIR)
    diagnostics.plot_pc_time_series(mssa, times, FIG_DIR)

#################################
## Fitting Macro-Spiral to PCs ##
#################################

if args.macro_fitting:
    for pc_list in list_of_pc_lists:

        mssa.reconstruct(pc_list)
        get_recon = mssa.getReconstructed()
        pc_rc = get_recon[list(get_recon.keys())[0]].getAllCoefs()
        MS = RewindMacroSpiral(pc_rc, pc_list, jphi_min, jbins, sim_name, channel_name, m=1)
        MS.plot_macro_tfit_over_time(threshold=np.pi/2, savefig=True, fig_dir=WINDING_DIR)

        PC_WINDING_DIR = os.path.join(INDIVIDUAL_WINDING_DIR, MS.pc_string)
        PC_DIPOLE_DIR = os.path.join(DIPOLE_DIR, MS.pc_string)
        os.makedirs(PC_WINDING_DIR, exist_ok=True)
        os.makedirs(PC_DIPOLE_DIR, exist_ok=True)

        for tstep in range(pc_rc.shape[1]):
            MS.plot_fitting_tstep(tstep, threshold=np.pi/2, savefig=True, fig_dir=PC_WINDING_DIR)
            MS.make_rewind_dipole_fig(tstep, savefig=True, fig_dir=PC_DIPOLE_DIR)


###################################
## Making Movies of Data and PCs ##
###################################
if args.movies:
    if not list_of_pc_lists:
        print('Warning: list_of_pc_lists is empty in runs.toml — data movies only.')
    print('Creating dual-panel movies...')
    diagnostics.make_dual_movies(
        mssa, data_file, times, MOVIES_DIR, list_of_pc_lists,
        sim_name=sim_name, channel_name=channel_name,
        jphi_min=jphi_min, jbins=jbins)

print('Done. All outputs saved to:', FIG_DIR)