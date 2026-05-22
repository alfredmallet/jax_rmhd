import jax
import jax.numpy as jnp
from .timestepping import get_scheme
from .snapshot_io import save_snapshot
from time import perf_counter
from .physics import equation_registry, construct_rhs

#debug
from jax.debug import inspect_array_sharding

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

def simulate_scan(state,kgrid,params,nblock,t_snap,t_end,mngr,schemestr='lsrk33'):
    # this simulates for a fixed number of timesteps
    # for automatic differentiation sometime in the future
    # we should set nblock using the helper function estimate_good_nblock
    t_start = perf_counter()
    stepper,scheme = get_scheme(schemestr)
    block_of_steps_jit = jax.jit(block_of_steps,static_argnums=(2,3,4,5),
                           in_shardings=(params.state_sharding, None),
                             out_shardings=(params.state_sharding,None))
    t_last_snapshot = state.t
    snap=0
    print("Saving initial state as snapshot "+str(snap))
    save_snapshot(snap,state,mngr)
    while state.t<t_end:
        state, _ = block_of_steps_jit(state,kgrid,params,nblock,scheme,stepper)
        print(state.t)
        if state.t - t_last_snapshot > t_snap:
            snap=snap+1
            print("Saving snapshot "+str(snap))
            save_snapshot(snap,state,mngr)
            t_last_snapshot=state.t
    snap=snap+1
    print("Saving final state as snapshot "+str(snap))
    save_snapshot(snap,state,mngr)
    mngr.wait_until_finished()
    t_sim = perf_counter()-t_start
    print(f"Ending simulation at t = " + str(state.t)+". It took "+str(t_sim)+"s")
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
    sim_to_next_snap_jit = jax.jit(sim_to_next_snap,
                                   in_shardings=(params.state_sharding,None),
                                   out_shardings=params.state_sharding)
    state=initial_state
    t_last_snapshot = state.t
    snap=0
    if save:   
        print("Saving initial state as snapshot "+str(snap))
        save_snapshot(snap,state,mngr)
    while state.t<t_end:
        t_next_snapshot=min(t_last_snapshot+t_snap,t_end)
        print("State sharding entering jit:", state.fields.sharding)
        state = sim_to_next_snap_jit(state,t_next_snapshot)
        print("State sharding exiting jit:", state.fields.sharding)
        snap=snap+1
        if save:
            print ("Saving snapshot "+str(snap)+ " at t = "+str(state.t))
            save_snapshot(snap,state,mngr)
            t_last_snapshot=state.t
    mngr.wait_until_finished()
    t_sim = perf_counter()-t_start
    print(f"Ending simulation at t = "+str(state.t)+". It took "+str(t_sim)+"s")
    return state

