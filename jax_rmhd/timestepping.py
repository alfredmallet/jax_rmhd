import jax
from jax import jit
import jax.numpy as jnp
from .physics import grad,NonlinearTerm,LinearTerm,rhs
from .types import Fields,SimulationState
from typing import NamedTuple,Tuple
from functools import partial


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

#standard RK4 with integrating factor.
#problem is it uses a lot of memory on k1-k4.
def rk_advance(state,kgrid,params,scheme=None):
    #print("---COMPILING rk_advance---") #add this back in to check that jit is working properly
    #grad
    grads=grad(state,kgrid)
    #cfl
    dt=set_timestep(grads,params)
    #dissipation factors e^(- visc * k^(2 * hyper) * dt) and e^(- visc * k^(2 * hyper) * dt/2)
    viscosity_factor_full = jnp.exp(kgrid.hvisc_exponent(params)*dt)
    viscosity_factor_half = jnp.exp(kgrid.hvisc_exponent(params)*dt/2.0)
    resistivity_factor_full = jnp.exp(kgrid.hres_exponent(params)*dt)
    resistivity_factor_half = jnp.exp(kgrid.hres_exponent(params)*dt/2.0)
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

# object defining low-storage Runge-Kutta (lsrk) schemes
class LSRK_Scheme(NamedTuple):
    alphas: Tuple[float,...]
    betas: Tuple[float,...]
    gammas: Tuple[float,...]
    nstages: int
"""
# Old LSRK timestepper: includes an integrating factor for the dissipative terms.
# This one is pretty inefficient when jitted, but can be compared with the fast
# version below if necessary.
def lsrk_advance(state,kgrid,params,scheme):
    delta = None
    stage_state = state
    first_rhs, grads = rhs(stage_state,kgrid,params)
    dt=set_timestep(grads,params)
    visc_exponent = kgrid.hvisc_exponent(params)*dt
    res_exponent = kgrid.hres_exponent(params)*dt
    for stage in range(scheme.nstages):
        alpha = scheme.alphas[stage]
        beta = scheme.betas[stage]
        gamma = scheme.gammas[stage]
        diss_factors = Fields(jnp.exp(visc_exponent*gamma),jnp.exp(res_exponent*gamma))
        if stage==0:
            stage_rhs=first_rhs
            delta = jax.tree_util.tree_map(lambda df, srhs: df * dt * srhs, diss_factors, stage_rhs)
        else:
            stage_rhs,_ = rhs(stage_state,kgrid,params)
            delta = jax.tree_util.tree_map(lambda srhs,df,d: df*(alpha  * d + dt * srhs),
                                           stage_rhs, diss_factors, delta)
        stage_fields = jax.tree_util.tree_map(lambda sf, df, d: df * sf + beta * d,
                                                  stage_state.fields, diss_factors, delta)
        next_t = stage_state.t + gamma*dt
        stage_state = SimulationState(t=next_t,fields=stage_fields)
    return stage_state
"""

# LSRK timestepper: includes an integrating factor for the dissipative terms.
# The loop over stages is now done with jax.lax.scan to enforce boundary between stages
# This seems to avoid CPU memory layout thrashing.
def lsrk_advance(state, kgrid, params, scheme):
    alphas_arr = jnp.array(scheme.alphas)
    betas_arr = jnp.array(scheme.betas)
    gammas_arr = jnp.array(scheme.gammas)

    init_rhs,grads = rhs(state,kgrid,params)
    dt = set_timestep(grads,params)

    visc_exponent = kgrid.hvisc_exponent(params) * dt
    res_exponent = kgrid.hres_exponent(params) * dt

    init_delta = jax.tree_util.tree_map(jnp.zeros_like,state.fields)
    init_carry = (state,init_delta)

    stage_pars = (alphas_arr, betas_arr, gammas_arr, jnp.arange(scheme.nstages))

    def scan_body(carry,stage_vals):
        current_state, delta = carry
        alpha, beta, gamma, istage = stage_vals

        stage_rhs = jax.lax.cond(
            istage == 0,
            lambda: init_rhs,
            lambda: rhs(current_state,kgrid,params)[0]
            )
        
        diss_factors = Fields(jnp.exp(visc_exponent * gamma), jnp.exp(res_exponent * gamma))

        next_delta = jax.tree_util.tree_map(lambda srhs, df, d: df * (alpha*d + dt*srhs),
                                            stage_rhs,diss_factors,delta)
        next_fields = jax.tree_util.tree_map(lambda sf,df,d: df*sf + beta*d, 
                                            current_state.fields, diss_factors, next_delta)
        next_t = current_state.t + gamma*dt
        return (SimulationState(t=next_t,fields=next_fields),next_delta), None
    
    (final_state, _), _ = jax.lax.scan(scan_body,init_carry,stage_pars)

    return final_state

#Code to set up a couple of common lsrk schemes

def _get_LSRK54_gammas():
    c2 = 1432997174477 / 9575080441755
    c3 = 2526269341429 / 6820363962896
    c4 = 2006345519317 / 3224310063776
    c5 = 2802321613138 / 2924317926251
    return (c2, c3 - c2, c4 - c3, c5 - c4, 1.0 - c5)

_scheme_registry = {
    "rk44": (rk_advance, None),
    "lsrk33": (lsrk_advance,
               LSRK_Scheme(
                   alphas = (0.0, -5.0 / 9.0, -153.0 / 128.0),
                   betas  = (1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0),
                   gammas = (1.0 / 3.0, 5.0 / 12.0, 1.0 / 4.0),
                   nstages = 3,
                   )), #This is the original Williamson 1980 RK3 scheme
    "lsrk54": (lsrk_advance,
               LSRK_Scheme(
                   alphas = (
                       0.0,
                       -567301805773 / 1357537059087,
                       -2404267990393 / 2016746695238,
                       -3550918686646 / 2091501179385,
                       -3270041069962 / 2362476162756
                       ),
                   betas = (
                       1432997174477 / 9575080441755,
                       5161836677717 / 13612068292357,
                       1720146321549 / 2090206949498,
                       3134564353537 / 4481467310338,
                       2277821191437 / 14882151754819
                       ),
                   gammas = _get_LSRK54_gammas(),
                   nstages = 5,
                   )) #5-stage, 4th-order scheme from Carpenter & Kennedy 1994
}

def get_scheme(name: str):
    if name not in _scheme_registry:
        raise ValueError(f"Unknown scheme '{name}'. Available: {list(_scheme_registry.keys())}")
    return _scheme_registry[name]