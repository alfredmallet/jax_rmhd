import jax
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
from .types import Fields,SimulationState

class Parameters():
    #Stores all static parameters for the problem
    def __init__(self,nx,ny,Lx,Ly,visc,res,hyper,cfl_safety,dims=2,nz=0,Lz=0.0,z_diss=0.25,z_diss_hyper=2.0,z_diff_order=4):
        #perpendicular grid
        self.nx=nx
        self.ny=ny
        self.Lx=Lx
        self.Ly=Ly
        self.dx=Lx/nx
        self.dy=Ly/ny
        #perpendicular dissipation
        self.visc=visc
        self.res=res
        self.hyper=hyper
        #timestepping
        self.cfl_safety=cfl_safety
        #dimensions
        self.spatial_dimensions=dims
        if dims==3:
            #z grid parameters
            self.nz = nz
            self.Lz = Lz
            self.dz = Lz/nz
            #z dissipation parameters
            self.z_diss = z_diss #This is in dimensionless units, dissipation coefficient is zdiss*(dz/2)^4
                                 #cf Pueschel et al. 2010
            self.z_diss_hyper = z_diss_hyper #currently unused, set =2
            self.z_diff_order = z_diff_order #currently unused, set =4

def init_cluster():
    # This should be called first when running on more than 1 node. It is optional otherwise.
    try:
        jax.distributed.initialize()
        print(f"Distributed system initialized. Total devices: {jax.device_count()}")
    except (ValueError, RuntimeError):
        print(f"Running in local mode. Total devices: {jax.local_device_count()}")

def setup_sharding(params):
    #Sets up parallelization of fields along the z axis if we're in 3D.
    n_devices = jax.device_count()
    if params.spatial_dimensions==3:
        devices = mesh_utils.create_device_mesh((n_devices,))
        mesh = Mesh(devices, axis_names=('z_axis',))
        z_sharding = NamedSharding(mesh, PartitionSpec('z_axis', None, None))
    else:
        z_sharding = None
        if n_devices > 1:
            print("You probably should only run a 2D run on one device, since this isn't parallelized.")
    fields_sharding = Fields(phik=z_sharding, psik=z_sharding)
    state_sharding = SimulationState(t=None,fields=fields_sharding)
    return (z_sharding,fields_sharding,state_sharding)