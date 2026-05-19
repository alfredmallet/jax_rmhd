import jax.numpy as jnp
import jax.numpy.fft as ft
from typing import NamedTuple

#nb the code is only spectral in the perpendciular plane, so this is all 2D

class K_Grids(NamedTuple):
    # Stores the wavenumber grids and methods
    kx: jnp.ndarray
    ky: jnp.ndarray
    #code to store these as precomputed arrays: I think this is slower.
    #ksq: jnp.ndarray
    #inv_ksq: jnp.ndarray
    #dealias_filter: jnp.ndarray
    #hvisc_exponent: jnp.ndarray
    #hres_exponent: jnp.ndarray
    def ksq(self):
        return self.kx**2 + self.ky**2
    def inv_ksq(self):
        return jnp.where(self.kx**2 + self.ky**2 > 0, 1.0/(self.kx**2 + self.ky**2), 0.0)
    def dealias_filter(self):
        return (self.kx**2 + self.ky**2)<(jnp.shape(self.kx)[0]/3.0)**2
    def hvisc_exponent(self,params):
        return -params.visc*self.ksq()**params.hyper
    def hres_exponent(self,params):
        return -params.res*self.ksq()**params.hyper

def setup_kgrids(params):
    # Gets the wavenumber grid object from parameters.
    kx = ft.fftfreq(params.nx) * params.nx * 2 * jnp.pi / params.Lx
    ky = ft.rfftfreq(params.ny) * params.ny * 2 * jnp.pi / params.Ly
    kx_grid = kx.reshape(-1, 1)
    ky_grid = ky.reshape(1, -1)
    #code for precalculating these arrays: I think this is slower
    #ksq = kx_grid**2 + ky_grid**2
    #inv_ksq = jnp.where(kx_grid**2 + ky_grid**2 > 0, 1.0/(kx_grid**2 + ky_grid**2), 0.0)
    #dealias_filter = (kx_grid**2 + ky_grid**2)<(jnp.shape(kx_grid)[0]/3.0)**2
    #hvisc_exponent = -params.visc * ksq**params.hyper
    #hres_exponent = -params.res * ksq**params.hyper
    return K_Grids(kx=kx_grid,ky=ky_grid)
                   #,ksq=ksq,inv_ksq=inv_ksq,
                   #dealias_filter=dealias_filter,hvisc_exponent=hvisc_exponent,hres_exponent=hres_exponent)

def fft(x):
    x_contiguous = jnp.array(x,copy=True)
    return ft.rfft2(x_contiguous,axes=(-2,-1))

def ifft(x):
    x_contiguous = jnp.array(x,copy=True)
    return ft.irfft2(x_contiguous,axes=(-2,-1))