import numpy as np
from astropy.table import Table, vstack
import glob

if __name__ == '__main__':
    files = glob.glob('../data/cache_kiyan_fast_bins_j30_t16/*')

    all_table = vstack([Table.read(file) for file in files])

    all_table.write('../data/kiyan-fast_mssa_prep_table.fits', format='fits', overwrite=True)
