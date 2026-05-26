#!/bin/bash
#SBATCH --job-name=expl
#SBATCH --account=fc_kawturb  
#SBATCH --partition=savio3       
#SBATCH --nodes=1                  
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1     
#SBATCH --time=00:30:00           
#SBATCH --output=expl.out
#SBATCH --error=expl.err
#SBATCH --mem=0

module purge
module load anaconda3

source activate jax_cpu

export OMP_PROC_BIND=close
export OMP_PLACES=cores

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

time srun python -u /global/home/users/alfredmallet/jax_rmhd/examples/test.py
