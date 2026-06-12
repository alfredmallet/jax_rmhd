import jax
import jax.numpy as jnp
from . import fourier
import jax.numpy.fft as ft

#
#
# THESE ARE LIKELY BROKEN! NEED TO FIX
#
#

def perpspec(state,kgrid,params,bin_factor=2.0):
    phik=state.fields[0]
    psik=state.fields[1]
    rfft2_y_factor = jnp.full(phik.shape[-1],2.0)
    rfft2_y_factor = rfft2_y_factor.at[0].set(1.0)
    rfft2_y_factor = rfft2_y_factor.at[-1].set(1.0)
    energy_u = 0.5 * kgrid.ksq() * jnp.abs(phik)**2.0 * rfft2_y_factor
    energy_b = 0.5 * kgrid.ksq() * jnp.abs(psik)**2.0 * rfft2_y_factor
    energy_u = jnp.sum(energy_u,axis=0)/params.nz
    energy_b = jnp.sum(energy_b,axis=0)/params.nz
    kunit = min(2 * jnp.pi / params.Lx, 2 * jnp.pi / params.Ly)
    kmax = min(params.nx//2,params.ny//2)*kunit
    dk=kunit*bin_factor
    bin_edges = jnp.arange(0,kmax+dk,dk)
    norm = 1 / float(params.nx*params.ny)**2
    spec_u, _ = jnp.histogram(jnp.sqrt(kgrid.ksq()),bins=bin_edges,weights=energy_u*norm/dk)
    spec_b, _ = jnp.histogram(jnp.sqrt(kgrid.ksq()),bins=bin_edges,weights=energy_b*norm/dk)

    bin_centers=(bin_edges[1:] + bin_edges[:-1]) / 2
    return bin_centers,spec_u,spec_b

def parspec(state,kgrid,params,bin_factor=2.0):
    phik = state.fields.phik
    psik = state.fields.psik
    rfft2_y_factor = jnp.full(phik.shape[-1],2.0)
    rfft2_y_factor = rfft2_y_factor.at[0].set(1.0)
    rfft2_y_factor = rfft2_y_factor.at[-1].set(1.0)
    phikkz = ft.fft(phik,axis=0)
    psikkz = ft.fft(psik,axis=0)
    kz = ft.rfftfreq(params.nz) * params.nz * 2 * jnp.pi / params.Lz
    en_u_full = jnp.sum(0.5 * kgrid.ksq() * jnp.abs(phikkz)**2.0 * rfft2_y_factor, axis=(1,2))
    en_b_full = jnp.sum(0.5 * kgrid.ksq() * jnp.abs(psikkz)**2.0 * rfft2_y_factor, axis=(1,2))
    half = params.nz // 2
    energy_u = en_u_full[:half+1].at[1:half].add(en_u_full[half+1:][::-1])
    energy_b = en_b_full[:half+1].at[1:half].add(en_b_full[half+1:][::-1])
    kunit = 2 * jnp.pi / params.Lz
    kmax = params.nz//2 * kunit
    dk=kunit*bin_factor
    bin_edges = jnp.arange(0,kmax+dk,dk)
    norm= 1.0 /dk/float(params.nx*params.ny*params.nz)**2
    spec_u, _ = jnp.histogram(kz,bins=bin_edges,weights=energy_u*norm)
    spec_b, _ = jnp.histogram(kz,bins=bin_edges,weights=energy_b*norm)
    bin_centers=(bin_edges[1:] + bin_edges[:-1]) / 2
    return bin_centers,spec_u,spec_b

def energy(fields,kgrid):
    from .physics import gradk
    #These aren't really v and b but the squares are the same..
    vsq=jnp.mean(jnp.array(jax.tree_util.tree_map(lambda gfk: fourier.ifft(gfk),gradk(fields.phik,kgrid)))**2.0)
    bsq=jnp.mean(jnp.array(jax.tree_util.tree_map(lambda gfk: fourier.ifft(gfk),gradk(fields.psik,kgrid)))**2.0)
    return (vsq,bsq)

