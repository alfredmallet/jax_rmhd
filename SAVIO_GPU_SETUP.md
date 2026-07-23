# Setting up a JAX-GPU conda env on Savio

Mirrors the existing `jax_cpu` env used by `slurms/run_savio_*.sh` (`module load anaconda3`,
`source activate jax_cpu`), just swapping in the CUDA build of jax and a GPU-aware toolchain.
Run this once, on a login node (build steps are lightweight — no need for an interactive job).

```bash
module purge
module load anaconda3 gcc openmpi

conda create -n jax_gpu python=3.11 -y
source activate jax_gpu

# JAX's cuda12 pip wheel bundles its own CUDA/cuDNN runtime via nvidia-* packages,
# so you do NOT need to module-load a system cuda toolkit for JAX itself.
pip install -U "jax[cuda12]"

# mpi4py/mpi4jax are hard requirements of this codebase (config.py, physics/) even for a
# single-GPU/single-rank run -- config.py's init_cluster() always touches MPI.COMM_WORLD.
# Build mpi4py against the loaded system openmpi (not a pip-bundled MPI) so it matches
# what `mpirun` on the compute node actually launches.
pip install mpi4py
pip install mpi4jax

pip install orbax-checkpoint tensorstore numpy matplotlib

# from your checkout:
cd ~/jax_rmhd
pip install -e .
```

Sanity-check on a login node (no GPU there, so this only confirms the import/MPI wiring,
not device visibility):

```bash
python -c "import jax, mpi4py, mpi4jax, orbax.checkpoint, tensorstore; print(jax.__version__)"
```

Confirm the GPU is actually visible — this has to run inside a GPU job (`srun`/`sbatch`,
see the SLURM script), not on the login node:

```bash
srun --pty -A fc_kawturb -p savio3_gpu --gres=gpu:V100:1 --cpus-per-task=4 -t 00:10:00 \
  bash -c "source activate jax_gpu && python -c \"import jax; print(jax.devices())\""
```
You should see something like `[CudaDevice(id=0)]`. If you instead see `[CpuDevice(id=0)]`,
the node has no GPU allocated to it, or the `jax[cuda12]` wheel didn't install cleanly.

Two things specific to this codebase, not generic JAX-on-Savio advice:

- Don't touch `CUDA_VISIBLE_DEVICES` yourself — Slurm's `--gres=gpu:...` sets it for you, and
  `config.py::init_cluster()` never re-reads it.
- Set `RMHD_PRECISION=64` (or 32) as an env var before the process starts, same as on CPU —
  this env var is read once at import time.
