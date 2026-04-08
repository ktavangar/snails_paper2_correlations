import os

def expand_pc_entry(entry):
    """Expand a range string like '4:14' to list(range(4, 14)), or pass a list through."""
    if isinstance(entry, str):
        start, stop = entry.split(':')
        return list(range(int(start), int(stop)))
    return list(entry)

def make_dirs(FIG_DIR):
    INDIVIDUAL_WINDING_DIR = os.path.join(FIG_DIR, 'individual_winding_time_fits')
    DIPOLE_DIR = os.path.join(FIG_DIR, 'dipole_figs')
    MOVIES_DIR = os.path.join(FIG_DIR, 'movies')
    WINDING_DIR = os.path.join(FIG_DIR, 'winding_plots')

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(DIPOLE_DIR, exist_ok=True)
    os.makedirs(INDIVIDUAL_WINDING_DIR, exist_ok=True)
    os.makedirs(WINDING_DIR, exist_ok=True)
    os.makedirs(MOVIES_DIR, exist_ok=True)

    return FIG_DIR, INDIVIDUAL_WINDING_DIR, DIPOLE_DIR, MOVIES_DIR, WINDING_DIR
