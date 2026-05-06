import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
from .config import *
from .physics import *

def max_vector(vec):
    return jnp.max(jnp.hypot(vec[0],vec[1]))

def set_timestep(grads,params):
    #Sets the timestep according to the CFL condition.
    max_vel = max_vector(grads.phi)
    max_mag = max_vector(grads.psi)
    max_all = jnp.maximum(max_vel,max_mag)
    max_all = jnp.maximum(max_all, 1.0)
    return params.cfl_safety * params.dx/max_all

#this is currently just standard RK4 with integrating factor.
#problem is it uses a lot of memory on k1-k4.
#TODO: add support for LSRK3 and LSRK54 schemes.
@partial(jax.jit, static_argnums=(2,))
def rk_advance(state,kgrid,params):
    #grad
    grads=grad(state,kgrid,params)
    #cfl
    dt=set_timestep(grads,params)
    #dissipation factors e^(- visc * k^(2 * hyper) * dt) and e^(- visc * k^(2 * hyper) * dt/2)
    viscosity_factor_full = kgrid.hvisc_factor(params,dt)
    viscosity_factor_half = kgrid.hvisc_factor(params,dt/2.0)
    resistivity_factor_full = kgrid.hres_factor(params,dt)
    resistivity_factor_half = kgrid.hres_factor(params,dt/2)
    diss_full = Fields(viscosity_factor_full,resistivity_factor_full)
    diss_half = Fields(viscosity_factor_half,resistivity_factor_half)
    #RK4 substep 1
    k1 = NonlinearTerm(grads,kgrid)
    f1 = jax.tree_util.tree_map(lambda fn, kn, dh : dh * (fn + 0.5 * dt * kn), state.fields, k1,diss_half)
    #RK4 substep 2
    k2 = NonlinearTerm(grad(SimulationState(state.t+dt/2.0,f1),kgrid,params),kgrid)
    f2 = jax.tree_util.tree_map(lambda fn, kn, dh : dh * fn + 0.5 * dt * kn, state.fields, k2,diss_half)
    #RK4 substep 3
    k3 = NonlinearTerm(grad(SimulationState(state.t+dt/2.0,f2),kgrid,params),kgrid)
    f3 = jax.tree_util.tree_map(lambda fn, kn, df, dh : df * fn + dt * dh * kn, state.fields, k3, diss_full, diss_half)
    #RK4 final step
    k4 = NonlinearTerm(grad(SimulationState(state.t+dt,f3),kgrid,params),kgrid)
    def combine(fn,k1n,k2n,k3n,k4n,df,dh,dt):
        f_new = df*fn + (dt/6.0) * (df*k1n + 2.0*dh*k2n + 2.0*dh*k3n + k4n)
        return f_new * kgrid.dealias_filter()
    f_end = jax.tree_util.tree_map(
        lambda fn, k1n, k2n, k3n, k4n, df, dh: combine(fn, k1n, k2n, k3n, k4n, df, dh, dt),
        state.fields, k1, k2, k3, k4, diss_full, diss_half
    )
    return SimulationState(t=state.t + dt, fields=f_end)

@partial(jax.jit, static_argnums=(2, 3))
def block_of_steps(state,kgrid,params,nblock):
    print("---COMPILING---")
    def stepping(state,_):
        return rk_advance(state,kgrid,params), None
    final_state,_ = jax.lax.scan(stepping,state,None,nblock)
    return final_state,None