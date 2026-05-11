import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import os
from .physics import Fields,SimulationState

def get_precision_types():
    if jax.config.read("jax_enable_x64"):
        return jnp.float64, jnp.complex128
    else:
        return jnp.float32, jnp.complex64
    
# Setting up Orbax stuff
def snapshot_manager_setup(snap_path="data",nsnap=100):
    checkpoint_path = os.path.abspath(snap_path)
    options = ocp.CheckpointManagerOptions(max_to_keep=nsnap, create=True)
    return ocp.CheckpointManager(checkpoint_path,ocp.StandardCheckpointer(),options=options)

def save_snapshot(isnap,state,mngr):
    return mngr.save(isnap, args=ocp.args.StandardSave(state), metrics={"time": float(state.t)})

def load_snapshot(isnap,mngr,params):
    shape_complex = (params.nx, params.ny // 2 + 1)
    ftype, ctype = get_precision_types()
    phik_like = jax.ShapeDtypeStruct(shape_complex, ctype)
    psik_like = jax.ShapeDtypeStruct(shape_complex, ctype)
    fields_like = Fields(phik=phik_like,psik=psik_like)
    state_like = SimulationState(t=jax.ShapeDtypeStruct((), jnp.float64), fields=fields_like)
    return mngr.restore(isnap, args=ocp.args.StandardRestore(state_like))