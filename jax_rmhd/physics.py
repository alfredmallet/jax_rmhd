import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Tuple
from . import config
from . import fourier

def gradk(fk,kgrid):
    #takes gradient in fourier space
    return (1j*kgrid.kx*fk,1j*kgrid.ky*fk)

#Simulation state should be represented as a tuple (t,fields), with fields a tuple of the primitive fields
class Fields(NamedTuple):
    #This object is for anything shaped like fields, including the RHS of the equations.
    phik: jnp.ndarray
    psik: jnp.ndarray

class SimulationState(NamedTuple):
    t: float
    fields: Fields

@jit
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

#Holds all the gradients needed for the nonlinear terms in real space
class Gradients(NamedTuple):
    phi: jnp.ndarray
    psi: jnp.ndarray
    vort: jnp.ndarray
    jpar: jnp.ndarray

@jit   
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