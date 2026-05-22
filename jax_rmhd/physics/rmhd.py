import jax.numpy as jnp
from .. import fourier
from .shared_physics import gradk,bracket,z_derivatives

#debug
#from jax.debug import inspect_array_sharding

def grad(state,kgrid,params):
    phik=state.fields[0]
    psik=state.fields[1]
    vortk = -kgrid.ksq()*phik
    jpark = -kgrid.ksq()*psik
    fk = jnp.stack([phik,psik,vortk,jpark])
    #inspect_array_sharding(fk, callback=lambda s: print(f"--- Fields entering grad: {s}"))
    gradients = fourier.ifft_g(gradk(fk,kgrid),params)
    #inspect_array_sharding(gradients, callback=lambda s: print(f"--- grads when created: {s}"))
    return gradients

def set_timestep(grads,params):
    #Sets the timestep according to the CFL condition.
    gphi,gpsi = grads[0:2]
    max_vy_eff = jnp.max(jnp.abs(gphi[0])+jnp.abs(gpsi[0]))
    max_vx_eff = jnp.max(jnp.abs(gphi[1])+jnp.abs(gpsi[1]))
    eps=0.1
    max_eps = jnp.maximum(eps/params.dx,eps/params.dy)
    max_all = jnp.maximum(max_vx_eff/params.dx, max_vy_eff/params.dy)
    max_all = jnp.maximum(max_all,max_eps)
    if params.spatial_dimensions==3:
        max_all = jnp.maximum(max_all,1.0/params.dz)
        max_all = jnp.maximum(max_all,params.z_diss)
    return params.cfl_safety / max_all

def NonlinearTerm(state,grads,kgrid,params):
    gphi,gpsi,gvort,gjpar = grads
    #inspect_array_sharding(gphi, callback=lambda s: print(f"--- gphi when unpacked: {s}"))
    NLTerm_vort = bracket(gpsi,gjpar) - bracket(gphi,gvort)
    NLTerm_psi = - bracket(gphi,gpsi)
    (NLTerm_vort_k , NLTerm_psi_k) = fourier.fft(jnp.stack([NLTerm_vort,NLTerm_psi]),params)
    NLTerm_fields = jnp.stack([-kgrid.inv_ksq()*NLTerm_vort_k,NLTerm_psi_k])*kgrid.dealias_filter()
    #inspect_array_sharding(NLTerm_fields, callback=lambda s: print(f"--- NLTerm when created: {s}"))
    return NLTerm_fields

def LinearTerm(state,grads,kgrid,params):
    #TODO: add a check on z_diff_order and z_diss_hyper here. For now use 4th order centered f.d. 
    # and d_z^4 hyperdissipation for stability
    if params.spatial_dimensions==2:
        return jnp.zeros_like(state.fields)
    dz=params.dz
    diss=params.z_diss * (dz/2)**4
    #inspect_array_sharding(state.fields, callback=lambda s: print(f"--- fields before z deriv: {s}"))
    df_dz,d4f_dz4 = z_derivatives(state.fields,dz)
    #inspect_array_sharding(df_dz, callback=lambda s: print(f"--- df_dz when created: {s}"))
    #RMHD only logic: the z-derivatives belong to the opposite equations
    df_dz_rmhd = jnp.stack([df_dz[1],df_dz[0]])
    #inspect_array_sharding(df_dz_rmhd, callback=lambda s: print(f"--- df_dz_rmhd when created: {s}"))
    return df_dz_rmhd - diss * d4f_dz4