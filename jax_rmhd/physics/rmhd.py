import jax.numpy as jnp
from .. import grids
from . import shared_physics
from .shared_physics import gradk,bracket,z_derivatives
from mpi4py import MPI
import mpi4jax

def grad(state,kgrid,params):
    phik=state.fields[0]
    psik=state.fields[1]
    vortk = -kgrid.ksq()*phik
    jpark = -kgrid.ksq()*psik
    fk = jnp.stack([phik,psik,vortk,jpark])
    gradients = grids.ifft(gradk(fk,kgrid),params)
    return gradients

def set_timestep(grads,params):
    #Sets the timestep according to the CFL condition.
    gphi = grads[0]
    gpsi = grads[1]    
    max_vy_eff = jnp.max(jnp.abs(gphi[0])+jnp.abs(gpsi[0]))
    max_vx_eff = jnp.max(jnp.abs(gphi[1])+jnp.abs(gpsi[1]))
    eps=0.1
    max_eps = jnp.maximum(eps/params.dx,eps/params.dy)
    max_all = jnp.maximum(max_vx_eff/params.dx, max_vy_eff/params.dy)
    max_all = jnp.maximum(max_all,max_eps)
    if params.spatial_dimensions==3:
        max_all = jnp.maximum(max_all,1.0/params.dz)
        max_all = jnp.maximum(max_all,params.z_diss)
    if params.cart_comm is not None:
        max_all = mpi4jax.allreduce(max_all,op=MPI.MAX,comm=params.cart_comm)
    return params.cfl_safety / max_all

def NonlinearTerm(state,grads,kgrid,params):
    gphi,gpsi,gvort,gjpar = grads
    NLTerm_vort = bracket(gpsi,gjpar) - bracket(gphi,gvort)
    NLTerm_psi = - bracket(gphi,gpsi)
    (NLTerm_vort_k , NLTerm_psi_k) = grids.fft(jnp.stack([NLTerm_vort,NLTerm_psi]))
    NLTerm_fields = jnp.stack([-kgrid.inv_ksq()*NLTerm_vort_k,NLTerm_psi_k])*kgrid.dealias_filter()
    return NLTerm_fields

def LinearTerm(state,grads,kgrid,params):
    #TODO: add a check on z_diff_order and z_diss_hyper here. For now use 4th order centered f.d. 
    # and d_z^4 hyperdissipation for stability
    if params.spatial_dimensions==2:
        return jnp.zeros_like(state.fields)
    dz=params.dz
    diss=params.z_diss * (dz/2)**4
    df_dz,d4f_dz4 = z_derivatives(state.fields,params)
    #RMHD only logic: the z-derivatives belong to the opposite equations
    df_dz_rmhd = jnp.stack([df_dz[1],df_dz[0]])
    return df_dz_rmhd - diss * d4f_dz4

def ForcingTerm(state,grads,kgrid,params):
    # RMHD-specific forcing: either in the momentum equation or elsasser forcing
    if not params.forcing:
        return jnp.zeros_like(state.fields)
    z_local = grids.local_z_coords(params) if params.spatial_dimensions == 3 else None
    f_raw = shared_physics.reconstruct_envelope(state.forcing_state,z_local,params)
    phik = state.fields[0]
    psik = state.fields[1]
    if params.forcing_mode == "momentum":
        P = shared_physics.perp_inner_product(phik,f_raw[0],kgrid,params)
        f_phi = f_raw[0] * shared_physics.safe_scale(params.forcing_power,P,params.forcing_scale_max)
        f_psi = jnp.zeros_like(f_phi)
    else:
        zplus = phik + psik
        zminus = phik - psik
        Pp = shared_physics.perp_inner_product(zplus,f_raw[0],kgrid,params)
        Pm = shared_physics.perp_inner_product(zminus,f_raw[1],kgrid,params)
        eps_p, eps_m = params.forcing_power_elsasser
        f_plus = f_raw[0] * shared_physics.safe_scale(eps_p,Pp,params.forcing_scale_max)
        f_minus = f_raw[1] * shared_physics.safe_scale(eps_m,Pm,params.forcing_scale_max)
        f_phi = 0.5*(f_plus+f_minus)
        f_psi = 0.5*(f_plus-f_minus)
    return jnp.stack([f_phi,f_psi])
