"""
Lagrangian mSSA Table — MPI batch runner
=========================================
Mirrors make_mssa_table.py but uses LagrangianMSSATable.

Two-step workflow
-----------------
Step 1: Assign bins at the reference timestep (run once, serial):

    python make_lagrangian_table.py --assign-bins \
        --ref-timestep 40 \
        --data-file  /path/to/Kiyan_40.fits \
        --action-file /path/to/Actions40.p \
        --bin-index-file /path/to/bin_indices_t40.npy

Step 2: Fill tables across all timesteps (run with MPI):

    mpirun -n 50 python make_lagrangian_table.py \
        --bin-index-file /path/to/bin_indices_t40.npy \
        --cache-dir /path/to/cache_lagrangian \
        --mpi

"""

import warnings
warnings.filterwarnings('ignore')

import sys
import pathlib
import numpy as np
from argparse import ArgumentParser
from functools import partial
from astropy.table import vstack

# from mpi4py import MPI
from schwimmbad.utils import batch_tasks

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from load_actions import *
from lagrangian_prep import LagrangianMSSATable


# ---------------------------------------------------------------------------
# MPI worker
# ---------------------------------------------------------------------------

def worker(batch, TableSetup, sim, bin_index_file, action_dir):
    """Process one batch of timesteps on a single MPI rank."""
    (batch_id, _), tasks, cache_path = batch
    cache_file = cache_path / f'lagrangian_table_{batch_id:04d}.fits'

    if cache_file.exists():
        return cache_file

    TableSetup.load_bin_indices(bin_index_file)

    rank = MPI.COMM_WORLD.Get_rank()
    t = TableSetup.create_empty_table()

    tables=[]   
    for timestep in tasks:

        print(f'Processing timestep {timestep} on rank {rank}', flush=True)
        new_t = TableSetup.fill_table(
            timestep,
            sim=sim,
            actions_dir=action_dir
        )
        tables.append(new_t)
    t = vstack(tables)

    t.write(cache_file, overwrite=True)
    return cache_file


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(pool, args):
    if args.sim == 'test':
        TableSetup = LagrangianMSSATable(jphi_bounds=[1000, 3500], nphi_bins=[25, 16])
    elif (args.sim == 'live') | (args.sim == 'B2'):
        TableSetup = LagrangianMSSATable(jphi_bounds=[1000, 4000], nphi_bins=[30, 16])
    cache_path = pathlib.Path(args.cache_dir)
    cache_path.mkdir(exist_ok=True)

    times = np.arange(args.t_start, args.t_end, 1)

    batched_tasks = batch_tasks(
        n_batches=pool.size,
        arr=times,
        args=(cache_path,),
    )

    for _ in pool.map(
        partial(worker,
                TableSetup=TableSetup,
                sim=args.sim,
                bin_index_file=args.bin_index_file,
                action_dir=args.action_dir),
        batched_tasks,
    ):
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = ArgumentParser(description='Build Lagrangian mSSA tables.')

    # Step 1: bin assignment
    parser.add_argument('--assign-bins', action='store_true',
                        help='Assign particle indices to bins at the reference timestep and exit.')
    parser.add_argument('--ref-timestep', type=int, default=40,
                        help='Reference timestep for bin assignment (default: 40).')
    parser.add_argument('--action-dir', default=None,
                        help='Directory containing action files')

    # Shared
    parser.add_argument('--bin-index-file', required=True,
                        help='Path to save/load bin index assignments (.npy).')

    # Step 2: table filling
    parser.add_argument('--sim',        default='test', choices=['test', 'live', 'B2'])
    parser.add_argument('--cache-dir',  default='data/tables_lagrangian',
                        help='Output directory for per-batch FITS tables.')
    parser.add_argument('--t-start',    type=int, default=40)
    parser.add_argument('--t-end',      type=int, default=300)

    # Parallelism
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--mpi',   dest='mpi',    action='store_true', default=False)
    group.add_argument('--procs', dest='n_procs', type=int,           default=1)

    args = parser.parse_args()

    # --- Step 1: assign bins ---
    if args.assign_bins:
        if args.action_dir is None:
            parser.error('--assign-bins requires --action-dir.')
        print(f'Loading reference data (timestep {args.ref_timestep})...')
        if args.sim == 'test':
            data_ref = load_test_actions(tstep=args.ref_timestep, actions_dir=args.action_dir)
            LT = LagrangianMSSATable(jphi_bounds=[1000, 3500], nphi_bins=[25, 16])
        elif (args.sim == 'live') | (args.sim == 'B2'):
            data_ref = load_B2_actions(tstep=args.ref_timestep, actions_dir=args.action_dir)
            LT = LagrangianMSSATable(jphi_bounds=[1000, 4000], nphi_bins=[30, 16])
        
        LT.assign_bins(data_ref)
        LT.save_bin_indices(args.bin_index_file)
        print('Bin Indices saved')
        # sys.exit(0)

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
