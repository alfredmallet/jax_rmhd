import jax
from jax import jit
import jax.numpy as jnp
from .physics import grad,NonlinearTerm,LinearTerm
from .types import Fields,SimulationState

def max_vector(vec):
    return jnp.max(jnp.hypot(vec[0],vec[1]))

def set_timestep(grads,params):
    #Sets the timestep according to the CFL condition.
    max_vx_eff = jnp.max(jnp.abs(grads.phi[1])+jnp.abs(grads.psi[1]))
    max_vy_eff = jnp.max(jnp.abs(grads.phi[0])+jnp.abs(grads.psi[0]))
    eps=0.1
    if params.spatial_dimensions==3:
        max_freqs = jnp.array([max_vx_eff/params.dx, max_vy_eff/params.dy,eps/params.dx,1.0/params.dz,eps/params.dy,params.z_diss])
    else:
        max_freqs = jnp.array([max_vx_eff/params.dx, max_vy_eff/params.dy,eps/params.dx,eps/params.dy])
    max_all = jnp.max(max_freqs)
    return params.cfl_safety / max_all

#this is currently just standard RK4 with integrating factor.
#problem is it uses a lot of memory on k1-k4.
#TODO: add support for LSRK3 and LSRK54 schemes.
def rk_advance(state,kgrid,params):
    print("---COMPILING rk_advance---")
    #grad
    grads=grad(state,kgrid)
    #cfl
    dt=set_timestep(grads,params)
    #dissipation factors e^(- visc * k^(2 * hyper) * dt) and e^(- visc * k^(2 * hyper) * dt/2)
    viscosity_factor_full = kgrid.hvisc_factor(params,dt)
    viscosity_factor_half = kgrid.hvisc_factor(params,dt/2.0)
    resistivity_factor_full = kgrid.hres_factor(params,dt)
    resistivity_factor_half = kgrid.hres_factor(params,dt/2)
    diss_full = Fields(viscosity_factor_full,resistivity_factor_full)
    diss_half = Fields(viscosity_factor_half,resistivity_factor_half)
    #Defining helper function for the kn
    def kn(grads,state,kgrid,params):
        if params.spatial_dimensions==3:
            return jax.tree_util.tree_map(jnp.add,NonlinearTerm(grads,kgrid),LinearTerm(state,params))
        else:
            return NonlinearTerm(grads,kgrid)
    #RK4 substep 1
    k1 = kn(grads,state,kgrid,params)
    #k1 = jax.tree_util.tree_map(jnp.add,
    #                            NonlinearTerm(grads,kgrid),
    #                            LinearTerm(state,params))
    f1 = jax.tree_util.tree_map(lambda fn, kn, dh : dh * (fn + 0.5 * dt * kn), state.fields, k1,diss_half)
    #RK4 substep 2
    k2 = kn(grad(SimulationState(state.t+dt/2.0,f1),kgrid),SimulationState(state.t+dt/2.0,f1),kgrid,params)
    #k2 = jax.tree_util.tree_map(jnp.add,
    #                            NonlinearTerm(grad(SimulationState(state.t+dt/2.0,f1),kgrid),kgrid),
    #                            LinearTerm(SimulationState(state.t+dt/2.0,f1),params))
    f2 = jax.tree_util.tree_map(lambda fn, kn, dh : dh * fn + 0.5 * dt * kn, state.fields, k2,diss_half)
    #RK4 substep 3
    k3 = kn(grad(SimulationState(state.t+dt/2.0,f2),kgrid),SimulationState(state.t+dt/2.0,f2),kgrid,params)
    #k3 = jax.tree_util.tree_map(jnp.add,
    #                            NonlinearTerm(grad(SimulationState(state.t+dt/2.0,f2),kgrid),kgrid),
    #                            LinearTerm(SimulationState(state.t+dt/2.0,f2),params))
    f3 = jax.tree_util.tree_map(lambda fn, kn, df, dh : df * fn + dt * dh * kn, state.fields, k3, diss_full, diss_half)
    #RK4 final step
    k4 = kn(grad(SimulationState(state.t+dt,f3),kgrid),SimulationState(state.t+dt,f3),kgrid,params)
    #k4 = jax.tree_util.tree_map(jnp.add,
    #                            NonlinearTerm(grad(SimulationState(state.t+dt,f3),kgrid),kgrid),
    #                            LinearTerm(SimulationState(state.t+dt,f3),params))
    def combine(fn,k1n,k2n,k3n,k4n,df,dh,dt):
        f_new = df*fn + (dt/6.0) * (df*k1n + 2.0*dh*k2n + 2.0*dh*k3n + k4n)
        return f_new * kgrid.dealias_filter()
    f_end = jax.tree_util.tree_map(
        lambda fn, k1n, k2n, k3n, k4n, df, dh: combine(fn, k1n, k2n, k3n, k4n, df, dh, dt),
        state.fields, k1, k2, k3, k4, diss_full, diss_half
    )
    return SimulationState(t=state.t + dt, fields=f_end)