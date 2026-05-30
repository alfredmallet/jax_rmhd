import jax
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
from .types import SimulationState

class Parameters():
    #Stores all static parameters for the problem
    def __init__(self,nx,ny,Lx,Ly,diss,hyper,cfl_safety,dt=0.1,adaptive_timestep=True,dims=2,nz=0,Lz=0.0,z_diss=0.25,z_diss_hyper=2.0,z_diff_order=4,eqtype="RMHD"):
        self.eqtype=eqtype
        self.nfields=eqtype_registry[self.eqtype]
        #perpendicular grid
        self.nx=nx
        if ny%2==1:
            print("ny should be even: setting ny=ny-1")
            ny=ny-1
        self.ny=ny
        self.Lx=Lx
        self.Ly=Ly
        self.dx=Lx/nx
        self.dy=Ly/ny
        #perpendicular dissipation
        self.diss = diss # should be a tuple of length nfields
        self.hyper=hyper
        #timestepping
        self.cfl_safety=cfl_safety
        self.dt = dt # Only used if adaptive_timestep==False
        self.adaptive_timestep = adaptive_timestep #Usually we want this to be true
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
        #sharding
        n_devices = jax.device_count()
        if self.spatial_dimensions==3:
            devices = mesh_utils.create_device_mesh((n_devices,))
            self.mesh = Mesh(devices, axis_names=('z_axis',))
            self.z_spec = PartitionSpec('z_axis', None, None)
            self.fields_spec = PartitionSpec(None,'z_axis', None, None)
            self.grads_spec = PartitionSpec(None,None,'z_axis',None,None)
            self.z_sharding = NamedSharding(self.mesh, self.z_spec)
            self.fields_sharding = NamedSharding(self.mesh, self.fields_spec)
            #t is replicated across the mesh
            self.t_sharding = NamedSharding(self.mesh,PartitionSpec())
        else:
            #devices = mesh_utils.create_device_mesh((n_devices,))
            #self.mesh = Mesh(devices,axis_names=('dummy',))
            self.z_spec = None
            self.fields_spec = None
            self.grads_spec = None
            self.z_sharding = None
            self.fields_sharding = None
            self.t_sharding = None
            if n_devices > 1:
                print("You probably should only run a 2D run on one device, since this isn't parallelized.")
        self.state_sharding = SimulationState(t=self.t_sharding,fields=self.fields_sharding)
        
        

# registry to set the # of fields we are solving for
eqtype_registry = {
    "RMHD": 2
}

def init_cluster():
    # This should be called first when running on more than 1 node. It is optional otherwise.
    try:
        jax.distributed.initialize()
        print("Distributed system initialized. Total devices: ",jax.device_count())
    except (ValueError, RuntimeError):
        print("Running in local mode. Total devices:",jax.device_count())


#def setup_sharding(params):
#    #Sets up parallelization of fields along the z axis if we're in 3D.
#    n_devices = jax.device_count()
#    if params.spatial_dimensions==3:
#        devices = mesh_utils.create_device_mesh((n_devices,))
#        mesh = Mesh(devices, axis_names=('z_axis',))
#        z_sharding = NamedSharding(mesh, PartitionSpec('z_axis', None, None))
#        fields_sharding = NamedSharding(mesh, PartitionSpec(None,'z_axis', None, None))
#    else:
#        z_sharding = None
#        fields_sharding = None
#        if n_devices > 1:
#            print("You probably should only run a 2D run on one device, since this isn't parallelized.")
#    state_sharding = SimulationState(t=None,fields=fields_sharding)
#    return (z_sharding,fields_sharding,state_sharding)
