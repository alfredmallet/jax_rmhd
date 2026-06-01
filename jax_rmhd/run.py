import jax
import jax.numpy as jnp
from functools import partial
from .timestepping import get_scheme
from .snapshot_io import save_snapshot
from time import perf_counter
from .physics import equation_registry, construct_rhs
from .types import SimulationState
from .fourier import fft

def initialize(func,params):
    # use this to initialize with some known function.
    # func should be a function that sets ALL fields in the problem, in real space.
    @partial(jax.jit,static_argnums=(0,))
    def _init(f):
        x = jnp.linspace(0, params.Lx, params.nx, endpoint=False).reshape(1,-1,1)
        y = jnp.linspace(0, params.Ly, params.ny, endpoint=False).reshape(1,1,-1)
        if params.spatial_dimensions==3:
            nz_device = params.nz//params.size
            idx_device = params.rank * nz_device + jnp.arange(nz_device)
            z_device = (idx_device * params.dz).reshape(-1,1,1)
            state = SimulationState(t=0.0,fields=fft(f(x,y,z_device),params))
        else:
            state = SimulationState(t=0.0,fields=fft(f(x,y),params))
        return state
    return _init(func)        

#This can be used to estimate a good nblock. You can set the minimum higher.
def estimate_good_nblock(state,kgrid,params,t_snap,t_end,t_last_snap=0,nblock_min=10):
    set_timestep,_,grad = equation_registry[params.eqtype] 
    grads = grad(state,kgrid,params)
    dt = set_timestep(grads,params)
    t_next_snap = min(t_last_snap+t_snap,t_end)
    nblock_estimate = max((t_next_snap-state.t)/dt,nblock_min)
    return int(nblock_estimate)

def block_of_steps(state,kgrid,params,nblock,scheme,stepper):
    def stepping(state,_):
        set_timestep = equation_registry[params.eqtype].set_timestep_func
        rhs = construct_rhs(equation_registry[params.eqtype])
        return stepper(state,kgrid,params,rhs,set_timestep,scheme), None
    final_state,_ = jax.lax.scan(stepping,state,None,nblock)
    return final_state,None

#currently an orbax checkpoint mngr must be set outside of the simulate function
#this makes it a little easier to set up snapshots etc but could be changed

def simulate_scan(state,kgrid,params,nblock,t_snap,t_end,mngr,schemestr='lsrk33',save=True):
    # this simulates repeated fixed number of timesteps
    # for automatic differentiation sometime in the future
    # we should set nblock using the helper function estimate_good_nblock
    t_start = perf_counter()
    stepper,scheme = get_scheme(schemestr)
    block_of_steps_jit = jax.jit(block_of_steps,static_argnums=(2,3,4,5))
    t_last_snapshot = state.t
    snap=0
    if save:
        if params.rank==0:
            print("Saving initial state as snapshot "+str(snap))
        save_snapshot(snap,state,mngr)
    while state.t<t_end:
        state, _ = block_of_steps_jit(state,kgrid,params,nblock,scheme,stepper)
        print(state.t)
        if state.t - t_last_snapshot > t_snap and save:
            snap=snap+1
            if params.rank==0:
                print("Saving snapshot "+str(snap))
            save_snapshot(snap,state,mngr)
            t_last_snapshot=state.t
    snap=snap+1
    if save:
        if params.rank==0:
            print("Saving final state as snapshot "+str(snap))
        save_snapshot(snap,state,mngr)
    mngr.wait_until_finished()
    t_sim = perf_counter()-t_start
    if params.rank==0:
        print("Ending simulation at t = " + str(state.t)+". It took "+str(t_sim)+"s")
    return state

def simulate(initial_state,kgrid,params,t_snap,t_end,mngr,schemestr='lsrk33',save=True):
    t_start = perf_counter()
    stepper,scheme = get_scheme(schemestr)
    set_timestep = equation_registry[params.eqtype].set_timestep_func
    rhs = construct_rhs(equation_registry[params.eqtype])
    def stepper_wrapped(state):
        return stepper(state,kgrid,params,rhs,set_timestep,scheme)
    def sim_to_next_snap(state,target_t):
        def snap_cond(state):
            return state.t<target_t
        return jax.lax.while_loop(snap_cond,stepper_wrapped,state)
    sim_to_next_snap_jit = jax.jit(sim_to_next_snap)
    state=initial_state
    t_last_snapshot = state.t
    snap=0
    if save: 
        if params.rank==0:  
            print("Saving initial state as snapshot "+str(snap))
        save_snapshot(snap,state,mngr)
    while state.t<t_end:
        t_next_snapshot=min(t_last_snapshot+t_snap,t_end)
        state = sim_to_next_snap_jit(state,t_next_snapshot)
        snap=snap+1
        if save:
            if params.rank==0:
                print ("Saving snapshot "+str(snap)+ " at t = "+str(state.t))
            save_snapshot(snap,state,mngr)
            t_last_snapshot=state.t
    mngr.wait_until_finished()
    t_sim = perf_counter()-t_start
    if params.rank==0:
        print(f"Ending simulation at t = "+str(state.t)+". It took "+str(t_sim)+"s")
    return state

