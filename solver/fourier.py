import jax.numpy as jnp
import jax.numpy.fft as ft
from typing import NamedTuple, Tuple

class K_Grids(NamedTuple):
    # Stores the wavenumber grids and methods
    kx: jnp.ndarray
    ky: jnp.ndarray
    def ksq(self):
        return self.kx**2 + self.ky**2
    def inv_ksq(self):
        return jnp.where(self.kx**2 + self.ky**2 > 0, 1.0/(self.kx**2 + self.ky**2), 0.0)
    def dealias_filter(self):
        return (self.kx**2 + self.ky**2)<(jnp.shape(self.kx)[0]/3.0)**2
    def hvisc_factor(self,params,dt):
        return jnp.exp(-params.visc*self.ksq()**params.hyper * dt)
    def hres_factor(self,params,dt):
        return jnp.exp(-params.res*self.ksq()**params.hyper * dt)

def setup_kgrids(params):
    # Gets the wavenumber grid object from parameters.
    # In the future this should depend on Lx,Ly,nx,ny
    n=params.n
    kx = ft.fftfreq(n) * n
    ky = ft.rfftfreq(n) * n
    kx_grid = kx.reshape(-1, 1)
    ky_grid = ky.reshape(1, -1)    
    return K_Grids(kx=kx_grid,ky=ky_grid)

def fft(x):
    #this wrapper isn't actually used, lol.
    return ft.rfft2(x)