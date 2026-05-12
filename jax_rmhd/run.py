import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
from .timestepping import rk_advance
from .snapshot_io import save_snapshot

def block_of_steps(state,kgrid,params,nblock):
    def stepping(state,_):
        return rk_advance(state,kgrid,params), None
    final_state,_ = jax.lax.scan(stepping,state,None,nblock)
    return final_state,None

#currently an orbax checkpoint mngr must be set outside of the simulate function
#this makes it a little easier to set up snapshots etc but could be changed

def simulate_scan(initial_state,kgrid,params,nblock,t_snap,t_end,shardings,mngr):
    # this simulates for a fixed number of timesteps
    # for automatic differentiation sometime in the future
    _,_,state_sharding=shardings
    block_of_steps_jit = jax.jit(block_of_steps,static_argnums=(2,3),
                           in_shardings=(state_sharding, None),
                             out_shardings=(state_sharding,None))
    state=initial_state
    t_last_snapshot = state.t
    snap=0
    print("Saving initial state as snapshot "+str(snap))
    save_snapshot(snap,state,mngr)
    while state.t<t_end:
        state, _ = block_of_steps_jit(state,kgrid,params,nblock)
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
    return f"Ending simulation at t = " + str(state.t)

def simulate(initial_state,kgrid,params,t_snap,t_end,mngr,shardings):
    _,_,state_sharding = shardings
    rk_advance_jit=jax.jit(rk_advance,static_argnums=(2,),
                           in_shardings=(state_sharding, None),
                             out_shardings=state_sharding)
    def stepping(state):
        return rk_advance_jit(state,kgrid,params)
    state=initial_state
    t_last_snapshot = state.t
    snap=0   
    print("Saving initial state as snapshot "+str(snap))
    save_snapshot(snap,state,mngr)
    while state.t<t_end:
        snap=snap+1
        def snap_cond(state):
            t_next_snapshot=t_last_snapshot+t_snap
            return state.t<t_next_snapshot
        state = jax.lax.while_loop(snap_cond,stepping,state)
        state.fields.phik.block_until_ready()
        print ("Saving snapshot "+str(snap)+ " at t = "+str(state.t))
        save_snapshot(snap,state,mngr)
        t_last_snapshot=state.t
    mngr.wait_until_finished()
    return f"Ending simulation at t = "+str(state.t)

