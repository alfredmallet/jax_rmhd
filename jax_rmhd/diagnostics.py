import jax
import jax.numpy as jnp

def spec(state,kgrid,params):
    #assumes square grid
    #check normalization
    n=params.n
    energy_u = 0.5 * kgrid.ksq() * jnp.abs(state.fields.phik)**2.0
    energy_b = 0.5 * kgrid.ksq() * jnp.abs(state.fields.psik)**2.0
    kmax=n
    spec_u=list()
    spec_b=list()
    for kk in range(kmax):
        cond=(kk<=jnp.sqrt(kgrid.ksq()))*(jnp.sqrt(kgrid.ksq())<kk+1)
        spec_u.append(jnp.sum(energy_u*cond)/float(n)**4)
        spec_b.append(jnp.sum(energy_b*cond)/float(n)**4)
    return (jnp.array(spec_u),jnp.array(spec_b))

def energy(fields,kgrid):
    import jax.numpy.fft as ft
    from .physics import gradk
    #These aren't really v and b but the squares are the same..
    vsq=jnp.mean(jnp.array(jax.tree_util.tree_map(lambda gfk: ft.irfft2(gfk),gradk(fields.phik,kgrid)))**2.0)
    bsq=jnp.mean(jnp.array(jax.tree_util.tree_map(lambda gfk: ft.irfft2(gfk),gradk(fields.psik,kgrid)))**2.0)
    return (vsq,bsq)

