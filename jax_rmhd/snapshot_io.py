import jax
import jax.numpy as jnp
import tensorstore as ts
import orbax.checkpoint as ocp
import os
from .types import SimulationState

def get_precision_types():
    if jax.config.read("jax_enable_x64"):
        return jnp.float64, jnp.complex128
    else:
        return jnp.float32, jnp.complex64
    
# Setting up Orbax stuff
def snapshot_manager_setup(snap_path="data",nsnap=1000):
    checkpoint_path = os.path.abspath(snap_path)
    options = ocp.CheckpointManagerOptions()
    return ocp.CheckpointManager(directory=checkpoint_path,options=options)

def save_snapshot(isnap,state,mngr):
    return mngr.save(isnap,args=ocp.args.StandardSave(state))

def load_snapshot(isnap,mngr,params):
    #This will load the whole snapshot into memory; respects the shardings specified in params
    if params.spatial_dimensions==3:
        nz_device = params.nz // params.size
        shape_complex = (params.nfields, nz_device, params.nx, params.ny // 2 + 1)
    else:
        shape_complex = (params.nfields, 1, params.nx, params.ny // 2 + 1)
    ftype, ctype = get_precision_types()
    fields_like = jax.ShapeDtypeStruct(shape_complex, ctype)
    state_like = SimulationState(t=jax.ShapeDtypeStruct((), ftype), fields=fields_like)
    return mngr.restore(isnap,args=ocp.args.StandardRestore(state_like))

def find_items(isnap,snap_path):
    db_path=os.path.join(snap_path, str(isnap), "default")

    kv_spec = {
        'driver': 'ocdbt',
        'base': {'driver': 'file', 'path': db_path}
    }
    kvs = ts.KvStore.open(kv_spec).result()

    print("Found these keys in the database:")
    for key in kvs.list().result():
        print(f"  {key.decode()}")

def load_slice(isnap,iz,nzslice,snap_path,item='fields'):
    #This loads a slice of a snapshot into memory: useful for laptop diagnostics
    #Use find_items to check what to put as item here.
    db_path = os.path.join(snap_path, str(isnap), "default")
    spec = {'driver': 'zarr', 'kvstore': {
        'driver': 'ocdbt', 'base': {
            'driver': 'file', 'path': db_path,
            }
        },
        'path': item,       
    }
    f = ts.open(spec).result()
    return f[iz:iz+nzslice , :, :].read().result()