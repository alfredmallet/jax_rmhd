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

def get_key_dtype():
    return jax.eval_shape(lambda: jax.random.key(0)).dtype
    
# Setting up Orbax stuff
def snapshot_manager_setup(params,snap_path="data",nsnap=1000):
    if params.size>1:
        checkpoint_path = os.path.abspath(snap_path+f'/{params.rank}')
    else:
        checkpoint_path = os.path.abspath(snap_path)
    options = ocp.CheckpointManagerOptions()
    return ocp.CheckpointManager(directory=checkpoint_path,options=options)

def save_snapshot(isnap,state,mngr):
    return mngr.save(isnap,args=ocp.args.StandardSave(state))

def load_snapshot(isnap,mngr,params):
    if params.size > 1:
        snap_path = os.path.dirname(mngr.directory)
    else:
        snap_path = mngr.directory

    p_save = 1
    if os.path.exists(os.path.join(snap_path, "0", str(isnap))):
        while os.path.exists(os.path.join(snap_path, str(p_save), str(isnap))):
            p_save += 1

    ftype, ctype = get_precision_types()
    
    #grid sizes for the current load and the original save
    nz_load = params.nz // params.size if params.spatial_dimensions == 3 else 1
    nz_save = params.nz // p_save if params.spatial_dimensions == 3 else 1
    assert params.nz % params.size == 0, f"Current z-domain {params.nz} not cleanly divisible by params.size {params.size}"
    assert params.nz % p_save == 0, f"Saved z-domain {params.nz} not cleanly divisible by p_save {p_save}"

    #initialize the restored field array for the current rank (r_l)
    restored_fields = jnp.zeros((params.nfields, nz_load, params.nx, params.ny // 2 + 1), dtype=ctype)
    restored_t = 0.0

    #index range for current rank
    z_start_l = params.rank * nz_load
    z_end_l = (params.rank + 1) * nz_load

    # Setup the expected ShapeDtypeStruct for loading from the saved ranks.
    if params.spatial_dimensions == 3:
        shape_complex_s = (params.nfields, nz_save, params.nx, params.ny // 2 + 1)
    else:
        shape_complex_s = (params.nfields, 1, params.nx, params.ny // 2 + 1)

    nkx, nky = params.nx, params.ny // 2 + 1
    # forcing_state/forcing_key have no z-axis and are identical on every saved rank
    forcing_state_like_s = jax.ShapeDtypeStruct((params.n_ou, 2, nkx, nky), ctype)
    forcing_key_like_s = jax.ShapeDtypeStruct((), get_key_dtype())

    fields_like_s = jax.ShapeDtypeStruct(shape_complex_s, ctype)
    state_like_s = SimulationState(t=jax.ShapeDtypeStruct((), ftype), fields=fields_like_s,
                                    forcing_state=forcing_state_like_s, forcing_key=forcing_key_like_s)
    options = ocp.CheckpointManagerOptions()

    #iterate over saved ranks and extract overlapping z-slices (fields/t only)
    for rank_s in range(p_save):
        z_start_s = rank_s * nz_save
        z_end_s = (rank_s + 1) * nz_save

        #check for overlap
        g_start = max(z_start_l, z_start_s)
        g_end = min(z_end_l, z_end_s)

        if g_start < g_end:
            #overlap exists. Setup a checkpoint manager to read from rank_s
            path_s = os.path.abspath(os.path.join(snap_path, str(rank_s))) if p_save > 1 else os.path.abspath(snap_path)
            mngr_s = ocp.CheckpointManager(path_s, options=options)
            state_s = mngr_s.restore(isnap, args=ocp.args.StandardRestore(state_like_s))

            #slice coordinates
            s_start = g_start - z_start_s
            s_end = g_end - z_start_s
            l_start = g_start - z_start_l
            l_end = g_end - z_start_l

            restored_fields = restored_fields.at[:, l_start:l_end, :, :].set(state_s.fields[:, s_start:s_end, :, :])
            restored_t = state_s.t

    # forcing_state/forcing_key: restore once from rank 0
    path_0 = os.path.abspath(os.path.join(snap_path, "0")) if p_save > 1 else os.path.abspath(snap_path)
    mngr_0 = ocp.CheckpointManager(path_0, options=options)
    state_0 = mngr_0.restore(isnap, args=ocp.args.StandardRestore(state_like_s))
    restored_forcing_state = state_0.forcing_state
    restored_forcing_key = state_0.forcing_key

    return SimulationState(t=restored_t, fields=restored_fields,
                            forcing_state=restored_forcing_state, forcing_key=restored_forcing_key)


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