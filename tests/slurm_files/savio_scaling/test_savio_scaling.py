import os
os.environ["RMHD_PRECISION"] = "64"
os.environ["MPI4JAX_NO_WARN_JAX_VERSION"] = "1"
import jax
import jax_rmhd as jr
import jax.numpy as jnp
import jax.numpy.fft as ft
#import matplotlib.pyplot as plt
import time
import argparse

jr.init_cluster()

parser = argparse.ArgumentParser() # Initializes and creates a new argument parser object (like setting up a blank digital clipboard)
parser.add_argument("--nx", type=int, default=256)
parser.add_argument("--ny", type=int, default=256)
parser.add_argument("--nz", type=int, default=256)
args = parser.parse_args()


#parameters
nx = args.nx
ny = args.ny
nz = args.nz
Lx = 2.0 * jnp.pi
Ly = 2.0 * jnp.pi
Lz = 2.0 * jnp.pi
t = 0.0
nsnap = 100
t_snap = 10.0
t_end = 0.01
cfl_safety = 0.5 
spatial_dimensions=3
nblock=100
visc=0.0
res=0.0 # how much resistance to apply to the current (j) in the induction equation
hyper=3

snap_path="data"

mngr=jr.snapshot_manager_setup(snap_path=snap_path,nsnap=nsnap)

def init_fields(x,y,z):
    phi = jnp.cos(x) * jnp.cos(y) * jnp.cos(z)
    psi = phi # z+ wave propagates backwards at vA=1
    return jnp.stack([phi,psi],axis=0)


params=jr.Parameters(nx=nx,ny=ny,nz=nz,Lx=Lx,Ly=Ly,Lz=Lz,diss=(visc,res),
                     hyper=hyper,cfl_safety=cfl_safety,dims=spatial_dimensions)
kgrid = jr.setup_kgrids(params)
state=jr.initialize(init_fields,params)

is_control = (params.rank == 0)

# Warmup phase
e = jr.simulate_scan(state, kgrid, params, nblock=1, t_snap=t_snap, t_end=0.0001, mngr=mngr, save=False)
jax.block_until_ready(e)

# Benchmark phase
start_time = time.perf_counter()
end_state = jr.simulate_scan(state, kgrid, params, nblock, t_snap, t_end, mngr, save=False)
jax.block_until_ready(end_state)

if is_control and jax.process_index() == 0:
    end_time = time.perf_counter()
    # Added flush=True to force the file to save this line immediately
    print("NCPU ", params.size, "TIME", end_time - start_time, flush=True)

