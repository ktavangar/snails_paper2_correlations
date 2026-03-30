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

sys.path.append(str(PROJECT_ROOT / 'code'))
import helper

# --- Argument parsing ---
parser = argparse.ArgumentParser(description='Run mSSA pipeline for a named run in runs.toml.')
parser.add_argument('run_name',
                    help='Name of the run as defined in runs.toml.')
parser.add_argument('--no-movies', action='store_true',
                    help='Skip face-on animation generation.')
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

def expand_pc_entry(entry):
    """Expand a range string like '4:14' to list(range(4, 14)), or pass a list through."""
    if isinstance(entry, str):
        start, stop = entry.split(':')
        return list(range(int(start), int(stop)))
    return list(entry)

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

FACE_ON_DIR = os.path.join(FIG_DIR, 'face-on_plots')

os.makedirs(FIG_DIR, exist_ok=True)
if not args.no_movies:
    os.makedirs(FACE_ON_DIR, exist_ok=True)

print(f"Run: {args.run_name}")

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
helper.plot_eigenvalues(ev, FIG_DIR)
helper.plot_fg_matrices(mssa, FIG_DIR)
helper.plot_wcorr(mssa, FIG_DIR)
helper.plot_pc_time_series(mssa, times, FIG_DIR)

# --- Face-on animations ---
if not args.no_movies:
    if not list_of_pc_lists:
        print('Warning: list_of_pc_lists is empty in runs.toml — skipping movies.')
    else:
        print('Creating face-on animations...')
        helper.make_face_on_movies(
            mssa, data_file, times, FACE_ON_DIR, list_of_pc_lists,
            sim_name=sim_name, channel_name=channel_name,
            jphi_min=jphi_min, jbins=jbins)

print('Done. All outputs saved to:', FIG_DIR)
