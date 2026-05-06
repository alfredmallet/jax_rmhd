from .timestepping import *
from .snapshot_io import *

def simulate(initial_state,kgrid,params,nblock,t_snap,t_end,mngr):
    #currently an orbax checkpoint mngr must be set outside of the simulate function
    #this makes it a little easier to set up snapshots etc but could be changed
    state=initial_state
    t_last_snapshot = state.t
    snap=0
    print("Saving initial state as snapshot "+str(snap))
    save_snapshot(snap,state,mngr)
    while state.t<t_end:
        state, _ = block_of_steps(state,kgrid,params,nblock)
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