#!/bin/bash
#SBATCH --job-name=test_advection
#SBATCH --account=fc_kawturb  
#SBATCH --partition=savio3       
#SBATCH --nodes=1                  
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1     
#SBATCH --time=00:30:00           
#SBATCH --output=data/test_advection/test_advection_%j.out
#SBATCH --error=data/test_advection/test_advection_%j.err
#SBATCH --mem=0
module purge
module load anaconda3 gcc openmpi


export OMP_PROC_BIND=close
export OMP_PLACES=cores

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export XLA_CPU_ASYNC_THREAD_COUNT=1
export OMPI_MCA_pml=ucx

PY=/global/home/users/esromabraham/.conda/envs/jax_cpu/bin/python

time mpirun -n $SLURM_NTASKS "$PY" -u /global/home/users/esromabraham/jax_rmhd/tests/test_advection.py