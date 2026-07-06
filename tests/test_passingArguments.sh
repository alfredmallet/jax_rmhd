#!/bin/bash
#SBATCH --job-name=test_passingArguments
#SBATCH --account=fc_kawturb  
#SBATCH --partition=savio3       
#SBATCH --nodes=1                  
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1     
#SBATCH --time=01:30:00           
#SBATCH --output=test_file_%j.out
#SBATCH --error=test_file_%j.err
#SBATCH --mem=0

python test_passingArguments.py 5 10
set -euo pipefail

module purge
module load anaconda3 gcc openmpi

export OMP_PROC_BIND=close
export OMP_PLACES=cores

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PY=/global/home/users/esromabraham/.conda/envs/jax_cpu/bin/python 
time srun "$PY" -u test_passingArguments.py


