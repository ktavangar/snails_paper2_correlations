import warnings
warnings.filterwarnings('ignore')

import numpy as np
from astropy.table import vstack
import pathlib

from schwimmbad.utils import batch_tasks
from mpi4py import MPI
from functools import partial
from argparse import ArgumentParser
import sys

from mssa_prep import MSSATable

def worker(batch, TableSetup):
    (batch_id, _) , tasks, cache_path = batch
    cache_file = cache_path / f'mssa_table_{batch_id:04d}.fits'

    if cache_file.exists():
        return cache_file
    
    rank = MPI.COMM_WORLD.Get_rank()
    t = TableSetup.create_empty_table()
    for timestep in tasks:
        print(f'Processing timestep {timestep} on rank {rank}', flush=True)
        new_t = TableSetup.fill_table(timestep, sim='test', 
                            data_file = '../Kiyan-fast/full/Kiyan_{}.fits'.format(int(timestep)),
                            action_file = '../Kiyan-fast/full/Actions{}.p'.format(int(timestep)))
        t = vstack([t,new_t]) # add this timestep to the table 
    t.write(cache_file, overwrite=True)
    
    return cache_file

def main(pool):

    times = np.arange(0, 300, 1) # timesteps
    
    TableSetup = MSSATable() # all tunable parameters set to default values

    cache_path = pathlib.Path('../data/cache_kiyan_fast_bins_j30_t16') # directory to save the tables
    cache_path.mkdir(exist_ok=True)

    # Create batched tasks to send out to MPI workers
    batched_tasks = batch_tasks(n_batches=pool.size,
                        arr=times,
                        args=(cache_path, ))
    
    #(batch_id_, _) , tasks_, cache_path_ = batched_tasks
    #print(batched_tasks, flush=True)

    filenames = []
    for filename in pool.map(partial(worker, TableSetup=TableSetup), batched_tasks):
        filenames.append(filename)


if __name__ == '__main__':
    # Define parser object
    parser = ArgumentParser()

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI")

    args = parser.parse_args()
    
    print(args.mpi, args.n_procs, flush=True)
    # deal with multiproc:
    
    # --- Step 2: fill tables ---
    if args.mpi:
        from schwimmbad.mpi import MPIPool
        Pool, kw = MPIPool, {}
    elif args.n_procs > 1:
        from schwimmbad import MultiPool
        Pool, kw = MultiPool, {'processes': args.n_procs}
    else:
        from schwimmbad import SerialPool
        Pool, kw = SerialPool, {}

    with Pool(**kw) as pool:
        print(f'Running with {pool.size} workers', flush=True)
        main(pool, args)
    sys.exit(0)
