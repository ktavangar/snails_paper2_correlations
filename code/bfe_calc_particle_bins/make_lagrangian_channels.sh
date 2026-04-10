#!/usr/bin/env bash
#SBATCH -J kiyan-fast-mssa-prep      
#SBATCH -N 10
#SBATCH --ntasks-per-node=5

#SBATCH -p cca   
#SBATCH -t 2-0
#SBATCH -o /mnt/home/ktavangar/ceph/mssa_prep.out        # Output file name   
#SBATCH -e /mnt/home/ktavangar/ceph/mssa_prep.err        # Output file name
#SBATCH -C rome
# -C rome
# -C skylake

source ~/.bash_profile # probably unnecessary
source /Users/Tavangar/Work/EXP_Projects/paper2_correlations/snails-env/bin/activate

#stdbuf -i0 -o0 -e0 mpirun python3 -m mpi4py.futures /mnt/home/ktavangar/projects/MSSA_Snails/code/make_mssa_table.py --mpi
# srun python3 -m mpi4py.run -rc thread_level='funneled' \
#     /Users/Tavangar/Work/EXP_Projects/paper2_correlations/code/bfe_calc_particle_bins/make_lagrangian_table.py \
#     --assign-bins \
#     --mpi \
#     --ref-timestep 10 \
#     --action-dir /Users/Tavangar/Work/EXP_Projects/paper2_correlations/data/ \
#     --bin-index-file /savefilepath.npy \
#     --sim 'test' \
#     --cache_dir /dirpath/ \
#     --t-start 10 \
#     --t-end 299 

mpirun -n 2 python3 -m mpi4py.run -rc thread_level='funneled' \
    /Users/Tavangar/Work/EXP_Projects/paper2_correlations/code/bfe_calc_particle_bins/make_lagrangian_table.py \
    --assign-bins \
    --ref-timestep 10 \
    --action-dir /Users/Tavangar/Work/EXP_Projects/paper2_correlations/data/ \
    --bin-index-file /Users/Tavangar/Work/EXP_Projects/paper2_correlations/data/bin_indices_t10.npy \
    --sim test \
    --cache-dir /Users/Tavangar/Work/EXP_Projects/paper2_correlations/data/lagrangian_cache/ \
    --t-start 10 \
    --t-end 11 