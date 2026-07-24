# Phase 1 performance benchmark under real MPI: times the jitted block_of_steps loop and
# reports steps/sec (rank 0). Same script runs against the old (pre-Phase-1) or new
# package depending on PYTHONPATH -- see slurms/bench_phase1.sh.
# usage: bench_phase1.py <label> <case: 2d|2d_forced|3d|3d_forced> <donate|nodonate> [nx nz] [unroll] [nps]
import sys, time
import numpy as np, jax, jax.numpy as jnp
import jax_rmhd as jr
from jax_rmhd import run as jrun
from jax_rmhd.timestepping import get_scheme

jr.init_cluster()
label, case, donate = sys.argv[1], sys.argv[2], sys.argv[3] == "donate"
nx = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 128
nz = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5].isdigit() else 256
forced = case.endswith("forced")
L = 2 * np.pi
# lsrk54 + elsasser + adaptive dt is the comm-heaviest configuration (plan F1)
common = dict(Lx=L, Ly=L, diss=(1e-4, 1e-4), hyper=2, cfl_safety=0.5,
              forcing=forced, forcing_mode="elsasser", forcing_power_elsasser=(1.0, 1.0),
              forcing_tau=1.0, fshell=(1, 3), forcing_seed=1)
if case.startswith("3d"):
    p = jr.Parameters(nx=nx, ny=nx, dims=3, nz=nz, Lz=L, z_diss=0.25, **common)
    ic = lambda x, y, z: jnp.stack([jnp.cos(x + 1.4) + jnp.cos(y + 2.0) * jnp.cos(z),
                                    jnp.cos(2 * x + 2.3) * jnp.cos(z) + 0.5 * jnp.cos(y)])
else:
    p = jr.Parameters(nx=nx, ny=nx, dims=2, **common)
    ic = lambda x, y: jnp.stack([jnp.cos(x + 1.4) + jnp.cos(y + 2.0),
                                 jnp.cos(2 * x + 2.3) + 0.5 * jnp.cos(y + 6.2)])
# set as plain attributes so the old package (whose __init__ lacks them) just ignores them
p.lsrk_scan = "unroll" not in sys.argv
p.forcing_norm_per_step = "nps" in sys.argv
kg = jr.setup_kgrids(p)
state = jrun.initialize(ic, p)
stepper, scheme = get_scheme("lsrk54")
kwargs = dict(static_argnums=(2, 3, 4, 5))
if donate:
    kwargs["donate_argnums"] = (0,)  # new run.py's production jit config; nodonate = old's
step_jit = jax.jit(jrun.block_of_steps, **kwargs)
nblock, nrep = 25, 4
def barrier():
    # real mpi4py only; the local test stub's fake comm has no Barrier (single rank anyway)
    if p.size > 1:
        p.comm.Barrier()

state, _ = step_jit(state, kg, p, nblock, scheme, stepper)  # compile + warm block
jax.block_until_ready(state.fields)
barrier()
t0 = time.perf_counter()
for _ in range(nrep):
    state, _ = step_jit(state, kg, p, nblock, scheme, stepper)
jax.block_until_ready(state.fields)
barrier()
dt = time.perf_counter() - t0
if p.rank == 0:
    n = nrep * nblock
    tags = ("scan" if p.lsrk_scan else "unroll") + ("+nps" if p.forcing_norm_per_step else "")
    print(f"{label:4s} {case:10s} nx={nx} nz={nz} ranks={p.size} [{tags}] "
          f"{n/dt:8.2f} steps/s  {dt/n*1e3:8.2f} ms/step", flush=True)
