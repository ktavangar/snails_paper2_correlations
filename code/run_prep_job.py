import warnings
warnings.filterwarnings('ignore')

import numpy as np
from astropy.table import Table, vstack
import pathlib
import sys
sys.path.append('/mnt/ktavangar/home/projects/MSSA_Snails/code/')

from schwimmbad.mpi import MPIPool

from functools import partial
from mpi4py import MPI
from mssa_prep import MSSATable
from load_data_B2 import setup_B2, setup_B2_region
from load_data import load_data_actions, load_data
from df_helpers import LaguerreSnails

import warnings
warnings.filterwarnings('ignore')

def worker(args, TableSetup):
    timestep, region = args

    rank = MPI.COMM_WORLD.Get_rank()
    print(f"Processing timestep {timestep}, region {region} on rank {rank}", flush=True)
    
    print("Loading Data", flush=True)
    data = setup_B2_region(timestep, region, TableSetup.rad, 
            actions_only=True, extras=False)
    # data = load_data_actions(timestep)


    # Your analysis function here
    LS = LaguerreSnails(data, region, TableSetup.rad,
                        TableSetup.jz_grid, TableSetup.tz_grid, 
                        TableSetup.m_max, TableSetup.n_max,
                        TableSetup.timestep)
    coeffs = LS.get_coeffs()
    m0_amp = np.linalg.norm(np.abs(coeffs[0,:]), axis=0)
    m1_amp = np.linalg.norm(np.abs(coeffs[1,:]), axis=0)
    m2_amp = np.linalg.norm(np.abs(coeffs[2,:]), axis=0)

    pitch_ang_m1, phase_ang_m1, pitch_phase_flag_m1 = LS.get_pitch_phase_angles(m=1)
    pitch_ang_m2, phase_ang_m2, pitch_phase_flag_m2 = LS.get_pitch_phase_angles(m=2)

    stats = np.array([timestep, region[0], region[1], 
                          m0_amp, m1_amp, m2_amp,
                          pitch_ang_m1, pitch_ang_m2, phase_ang_m1, phase_ang_m2,
                          pitch_phase_flag_m1, pitch_phase_flag_m2])
    return stats

def stats_arrays_to_table(all_stats, TableSetup):
    stacked_array = np.column_stack(all_stats)

    output_table = Table(stacked_array, names=TableSetup.colnames)
    return output_table

if __name__ == "__main__":
    
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        
        timesteps = np.arange(0,600,1) # timesteps
        
        print("Setting up Table", flush=True)
        TableSetup=MSSATable()
        region_centers = TableSetup.centers

        # Generate list of tasks (timestep, region)
        tasks = [(timestep, region) for timestep in timesteps for region in region_centers]

        # Distribute tasks using MPIPool
        all_stats = pool.map(partial(worker, TableSetup=TableSetup), tasks)

        # Write all statistics to table
        tbl = stats_arrays_to_table(all_stats, TableSetup)

        tbl.write('/mnt/home/ktavangar/ceph/live_sim/mssa_prep_live_sim_bins_j25_t16.fits',
                format='fits', overwrite=True)
