#!/bin/bash
#SBATCH --job-name=jax_rmhd_cpu
#SBATCH --account=fc_kawturb  
#SBATCH --partition=savio3       
#SBATCH --nodes=1                  
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1     
#SBATCH --time=00:30:00           
#SBATCH --output=rmhd_%j.out
#SBATCH --error=rmhd_%j.err
#SBATCH --mem=0

module purge
module load anaconda3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax_cpu

export OMP_PROC_BIND=close
export OMP_PLACES=cores

export XLA_FLAGS="--xla_cpu_next_to_by_val=true" #--xla_cpu_parallel_thread_pool_size=1 --xla_force_host_platform_device_count=$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

srun python test.py
