import jax
import jax.numpy as jnp
from . import fourier

def spec(state,kgrid,params,bin_factor=2.0):
    energy_u = 0.5 * kgrid.ksq() * jnp.abs(state.fields.phik)**2.0
    energy_b = 0.5 * kgrid.ksq() * jnp.abs(state.fields.psik)**2.0
    kunit = min(2 * jnp.pi / params.Lx, 2 * jnp.pi / params.Ly)
    kmax = min(params.nx//2,params.ny//2)*kunit
    bins = jnp.arange(0,kmax+kunit*bin_factor,kunit*bin_factor)
    norm= 2 / float(params.nx*params.ny)**2
    spec_u, _ = jnp.histogram(jnp.sqrt(kgrid.ksq()),bins=bins,weights=energy_u*norm)
    spec_b, _ = jnp.histogram(jnp.sqrt(kgrid.ksq()),bins=bins,weights=energy_b*norm)
    return spec_u,spec_b

def energy(fields,kgrid):
    from .physics import gradk
    #These aren't really v and b but the squares are the same..
    vsq=jnp.mean(jnp.array(jax.tree_util.tree_map(lambda gfk: fourier.ifft(gfk),gradk(fields.phik,kgrid)))**2.0)
    bsq=jnp.mean(jnp.array(jax.tree_util.tree_map(lambda gfk: fourier.ifft(gfk),gradk(fields.psik,kgrid)))**2.0)
    return (vsq,bsq)

