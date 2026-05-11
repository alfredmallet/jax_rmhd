class Parameters():
    #Stores all static parameters for the problem
    def __init__(self,nx,ny,Lx,Ly,visc,res,hyper,cfl_safety,dims=2):
        #grid parameters
        self.nx=nx
        self.ny=ny
        self.Lx=Lx
        self.Ly=Ly
        self.dx=Lx/nx
        self.dy=Ly/ny
        #dissipation
        self.visc=visc
        self.res=res
        self.hyper=hyper
        #timestepping
        self.cfl_safety=cfl_safety
        #dimensions
        self.spatial_dimension=dims