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

srun python3 -m mpi4py.run -rc thread_level='funneled' \
    /Users/Tavangar/Work/EXP_Projects/paper2_correlations/code/bfe_calc_particle_bins/make_lagrangian_table.py \
    --assign-bins \
    --mpi \
    --ref-timestep 250 \
    --action-dir /mnt/ceph/users/jhunt/Bonsai/r2/B2/FlattenedDiscActions/ \
    --bin-index-file /savefilepath.npy \
    --sim test \
    --cache_dir /dirpath/ \
    --t-start 250 \
    --t-end 600 