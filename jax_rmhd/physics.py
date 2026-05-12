import jax
from jax import jit
import jax.numpy as jnp
from . import fourier
from .types import Fields,Gradients

def gradk(fk,kgrid):
    #takes gradient in fourier space
    return (1j*kgrid.kx*fk,1j*kgrid.ky*fk)

def grad(state,kgrids):
    # takes fourier state input, dealiases, and returns real-space gradients as a tuple of tuples
    # this is specific to each equation set since we need the vorticities etc. currently rmhd
    phik = state.fields.phik * kgrids.dealias_filter()
    psik = state.fields.psik * kgrids.dealias_filter()
    vortk = -kgrids.ksq()*phik
    jpark = -kgrids.ksq()*psik
    fk = (phik, psik, vortk, jpark)
    gfk = jax.tree_util.tree_map(lambda f: gradk(f,kgrids),fk)
    gradients=jax.tree_util.tree_map(lambda p: fourier.ifft(p), gfk)
    return Gradients(*gradients)

def NonlinearTerm(grads,kgrid):
    #takes real-space inputs of gradients of a,b and returns poisson bracket {a,b} in fourier space
    def bracket(a,b):
        return a[0]*b[1] - a[1]*b[0]
    NLTerm_vort = bracket(grads.psi,grads.jpar) - bracket(grads.phi,grads.vort)
    #((grads.psi[0]*grads.jpar[1]-grads.psi[1]*grads.jpar[0]) 
                  #- (grads.phi[0]*grads.vort[1] - grads.phi[1]*grads.vort[0]))
    NLTerm_psi = - bracket(grads.phi,grads.psi)
    #- (grads.phi[0]*grads.psi[1] - grads.phi[1]*grads.psi[0])
    (NLTerm_vort_k , NLTerm_psi_k) = jax.tree_util.tree_map(fourier.fft,(NLTerm_vort,NLTerm_psi))
    return Fields(phik = -kgrid.inv_ksq()*NLTerm_vort_k*kgrid.dealias_filter(), 
                  psik = NLTerm_psi_k*kgrid.dealias_filter())

def LinearTerm(state,params):
    #TODO: add a check on z_diff_order and z_diss_hyper here. For now use 4th order centered f.d. 
    # and d_z^4 hyperdissipation for stability
    dz=params.dz
    diss=params.z_diss * (dz/2)**4
    # tree_map the rolls for forward compatibility with KRMHD/gk/etc
    fields_p1 = jax.tree_util.tree_map(lambda x: jnp.roll(x, -1, axis=0), state.fields)
    fields_p2 = jax.tree_util.tree_map(lambda x: jnp.roll(x, -2, axis=0), state.fields)
    fields_m1 = jax.tree_util.tree_map(lambda x: jnp.roll(x,  1, axis=0), state.fields)
    fields_m2 = jax.tree_util.tree_map(lambda x: jnp.roll(x,  2, axis=0), state.fields)
    df_dz = jax.tree_util.tree_map(lambda p1, p2, m1, m2: (-p2 + 8*p1 - 8*m1 + m2) / (12 * dz),
                                fields_p1, fields_p2, fields_m1, fields_m2)
    d4f_dz4 = jax.tree_util.tree_map(lambda f, p1, p2, m1, m2: (p2 -4*p1 +6*f -4*m1 + m2) / (dz**4),
                                state.fields, fields_p1, fields_p2, fields_m1, fields_m2)
    #RMHD only logic: eventually this should be handled by a matrix function set
    #by the equation type
    dphik_dz = df_dz.psik
    dpsik_dz = df_dz.phik
    df_dz_rmhd = Fields(phik = dphik_dz,psik=dpsik_dz)
    return jax.tree_util.tree_map(lambda fz,f4z4: (fz - diss*f4z4), df_dz_rmhd,d4f_dz4)