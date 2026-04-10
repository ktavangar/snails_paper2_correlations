#!/bin/bash   
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

#stdbuf -i0 -o0 -e0 mpirun python3 -m mpi4py.futures /mnt/home/ktavangar/projects/MSSA_Snails/code/make_mssa_table.py --mpi
srun python3 -m mpi4py.run -rc thread_level='funneled' \
    /mnt/home/ktavangar/projects/MSSA_Snails/code/make_lagrangian_table.py \
    --mpi \
    --ref-timestep 40 \
    --action-dir /../Kiyan-Single-Passage/full/ \
    --bin-index-file /savefilepath.npy \
    --sim 'test' \
    --cache_dir /dirpath/ \
    --t-start 0 \
    --t-end 299 