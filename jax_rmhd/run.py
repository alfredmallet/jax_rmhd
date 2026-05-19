import jax
from .timestepping import get_scheme,set_timestep
from .snapshot_io import save_snapshot
from .physics import grad
from time import perf_counter

#This can be used to estimate a good nblock. You can set the minimum higher.
def estimate_good_nblock(state,kgrid,params,t_snap,t_end,t_last_snap=0,nblock_min=10):
    grads = grad(state,kgrid)
    dt = set_timestep(grads,params)
    t_next_snap = min(t_last_snap+t_snap,t_end)
    nblock_estimate = max((t_next_snap-state.t)/dt,nblock_min)
    return int(nblock_estimate)

def block_of_steps(state,kgrid,params,nblock,scheme,stepper):
    def stepping(state,_):
        return stepper(state,kgrid,params,scheme), None
    final_state,_ = jax.lax.scan(stepping,state,None,nblock)
    return final_state,None

#currently an orbax checkpoint mngr must be set outside of the simulate function
#this makes it a little easier to set up snapshots etc but could be changed

def simulate_scan(initial_state,kgrid,params,nblock,t_snap,t_end,mngr,shardings,schemestr='lsrk33'):
    # this simulates for a fixed number of timesteps
    # for automatic differentiation sometime in the future
    # we should set nblock using the helper function estimate_good_nblock
    t_start = perf_counter()
    _,_,state_sharding=shardings
    stepper,scheme = get_scheme(schemestr)
    block_of_steps_jit = jax.jit(block_of_steps,static_argnums=(2,3,4,5),
                           in_shardings=(state_sharding, None),
                             out_shardings=(state_sharding,None))
    state=initial_state
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
    return f"Ending simulation at t = " + str(state.t)+". It took "+str(t_sim)+"s"

def simulate(initial_state,kgrid,params,t_snap,t_end,mngr,shardings,schemestr='lsrk33',save=True):
    t_start = perf_counter()
    _,_,state_sharding = shardings
    stepper,scheme = get_scheme(schemestr)
    stepper_jit=jax.jit(stepper,static_argnums=(2,3),
                           in_shardings=(state_sharding, None),
                             out_shardings=state_sharding)
    def stepping(state):
        return stepper_jit(state,kgrid,params,scheme)
    state=initial_state
    t_last_snapshot = state.t
    snap=0
    if save:   
        print("Saving initial state as snapshot "+str(snap))
        save_snapshot(snap,state,mngr)
    while state.t<t_end:
        def snap_cond(state):
            t_next_snapshot=t_last_snapshot+t_snap
            return state.t<t_next_snapshot
        state = jax.lax.while_loop(snap_cond,stepping,state)
        snap=snap+1
        if save:
            state.fields.phik.block_until_ready()
            print ("Saving snapshot "+str(snap)+ " at t = "+str(state.t))
            save_snapshot(snap,state,mngr)
            t_last_snapshot=state.t
    mngr.wait_until_finished()
    t_sim = perf_counter()-t_start
    print(f"Ending simulation at t = "+str(state.t)+". It took "+str(t_sim)+"s")
    return state

