import jax
import jax.numpy as jnp
import jax.numpy.fft as ft
from typing import NamedTuple

#nb the code is only spectral in the perpendciular plane, so this is all 2D

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
    def hdiss_exponents(self,params):
        diss=jnp.array(params.diss)
        #if params.spatial_dimensions==3:
        diss_grid = diss.reshape(-1,1,1,1)
        #else:
        #    diss_grid = diss.reshape(-1,1,1)
        return -diss_grid*self.ksq()**params.hyper

def setup_kgrids(params):
    # Gets the wavenumber grid object from parameters.
    # The scaling here respects jax.numpy's fourier transform conventions
    # so that e.g. we can calculate derivatives correctly.
    kx = ft.fftfreq(params.nx) * params.nx * 2 * jnp.pi / params.Lx
    ky = ft.rfftfreq(params.ny) * params.ny * 2 * jnp.pi / params.Ly
    kx_grid = kx.reshape(-1, 1)
    ky_grid = ky.reshape(1, -1)
    return K_Grids(kx=kx_grid, ky=ky_grid)

def fft(f):
    return ft.rfft2(f,axes=(-2,-1))

def ifft(f,params):
    return ft.irfft2(f,s=(params.nx,params.ny),axes=(-2,-1))
