#This solves the GDI equations in 2D, with modified Hasegawa-Wakatani parallel diffusion terms
#Write out these equations here
#TODO: solve the linear terms using exact integrating factor methods

import jax.numpy as jnp
from .. import fourier
from .shared_physics import gradk,bracket,z_derivatives
from mpi4py import MPI
import mpi4jax

class GDIParameters():
    def __init__(self,Ln=100.0,nu=0.01,v0=100.0,gammapar=100.0):
        self.Ln=Ln
        self.nu=nu
        self.v0=v0
        self.gammapar=gammapar
    
def grad(state,kgrid,params):
    nk=state.fields[0]
    phik=state.fields[1]
    vortk = -kgrid.ksq()*phik
    fk = jnp.stack([nk,phik,vortk])
    gradients = fourier.ifft(gradk(fk,kgrid),params)
    return gradients

