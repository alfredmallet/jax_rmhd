import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import os
from .types import Fields,SimulationState

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

def load_snapshot(isnap,mngr,params,shardings):
    #This will load the whole snapshot into memory; on a cluster it should work in a distributed way.
    _,_,state_sharding=shardings
    if params.spatial_dimensions==3:
        shape_complex = (params.nz, params.nx, params.ny // 2 + 1)
    else:
        shape_complex = (params.nx, params.ny // 2 + 1)
    ftype, ctype = get_precision_types()
    phik_like = jax.ShapeDtypeStruct(shape_complex, ctype)
    psik_like = jax.ShapeDtypeStruct(shape_complex, ctype)
    fields_like = Fields(phik=phik_like,psik=psik_like)
    state_like = SimulationState(t=jax.ShapeDtypeStruct((), ftype), fields=fields_like)
    if state_sharding is not None:
        state_target = jax.tree_util.tree_map(
            lambda x, s: jax.device_put(x, s),
            state_like, 
            state_sharding
        )
    else:
        state_target = state_like
    return mngr.restore(isnap, args=ocp.args.StandardRestore(state_target))

def load_slice(isnap,iz,nzslice,mngr,params):
    #This loads a slice of a snapshot into memory: useful for laptop diagnostics
    shape_slice = (nzslice, params.nx, params.ny//2 + 1)
    ftype, ctype = get_precision_types()
    phik_like = jax.ShapeDtypeStruct(shape_slice, ctype)
    psik_like = jax.ShapeDtypeStruct(shape_slice, ctype)
    fields_like = Fields(phik=phik_like,psik=psik_like)
    state_like = SimulationState(t=jax.ShapeDtypeStruct((), ftype), fields=fields_like)
    def slicer(full_array):
        return jax.lax.dynamic_slice(full_array,(iz,0,0),shape_slice)
    slice_transforms = {
        "fields": {
            "phik": slicer,
            "psik": slicer
        }
    }
    return mngr.restore(isnap, args=ocp.args.StandardRestore(target=state_like,
                                                             transforms=slice_transforms))