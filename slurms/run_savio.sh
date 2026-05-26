#!/bin/bash
#SBATCH --job-name=jax_rmhd_cpu
#SBATCH --account=fc_kawturb  
#SBATCH --partition=savio4_htc        
#SBATCH --nodes=1                  
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56        
#SBATCH --time=00:05:00           
#SBATCH --output=rmhd_%j.out
#SBATCH --error=rmhd_%j.err
#SBATCH --mem=0

module purge
module load anaconda3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax_cpu

export OMP_PROC_BIND=spread
export OMP_PLACES=cores

export XLA_FLAGS="--xla_cpu_next_to_by_val=true --xla_cpu_parallel_thread_pool_size=$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

python test.py
