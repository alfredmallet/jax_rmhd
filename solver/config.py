class Parameters():
    #Stores all static parameters for the problem
    def __init__(self,n,Lx,visc,res,hyper,cfl_safety):
        #grid parameters
        self.n=n
        self.Lx=Lx
        self.dx=Lx/n
        #dissipation
        self.visc=visc
        self.res=res
        self.hyper=hyper
        #timestepping
        self.cfl_safety=cfl_safety