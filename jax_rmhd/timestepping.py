import jax
from jax import jit
import jax.numpy as jnp
from .types import Fields,SimulationState
from typing import NamedTuple,Tuple

# Standard RK4 with integrating factor.
# Problem is it uses a lot of memory on k1-k4.
# I think it is always better to use the LSRK schemes
# This currently is not set up properly to use the correct RHS function constructed in physics

def rk_advance(state,kgrid,params,rhs,set_timestep,scheme=None):
    #print("---COMPILING rk_advance---") #add this back in to check that jit is working properly
    #RK4 substep 1
    k1, grads = rhs(state,kgrid,params)
    dt=set_timestep(grads,params)
    #dissipation factors 
    diss_full = jnp.exp(kgrid.hdiss_exponents(params)*dt)
    diss_half = jnp.exp(kgrid.hdiss_exponents(params)*dt/2)
    f1 = diss_half * (state.fields + 0.5 * dt * k1)
    #f1 = jax.tree_util.tree_map(lambda fn, kn, dh : dh * (fn + 0.5 * dt * kn), state.fields, k1,diss_half)
    #RK4 substep 2
    k2,_ = rhs(SimulationState(state.t+dt/2.0,f1),kgrid,params)
    f2 = diss_half * state.fields + 0.5*dt*k2
    #f2 = jax.tree_util.tree_map(lambda fn, kn, dh : dh * fn + 0.5 * dt * kn, state.fields, k2,diss_half)
    #RK4 substep 3
    k3,_ = rhs(SimulationState(state.t+dt/2.0,f2),kgrid,params)
    f3 = diss_full * state.fields + dt * diss_half * k3
    #f3 = jax.tree_util.tree_map(lambda fn, kn, df, dh : df * fn + dt * dh * kn, state.fields, k3, diss_full, diss_half)
    #RK4 final step
    k4,_ = rhs(SimulationState(state.t+dt,f3),kgrid,params)
    f_end = diss_full * state.fields + (dt/6.0) * (diss_full * k1 + 2.0*diss_half*k2+2.0*diss_half*k3+k4)
    return SimulationState(t=state.t + dt, fields=f_end)

# object defining low-storage Runge-Kutta (lsrk) schemes
class LSRK_Scheme(NamedTuple):
    alphas: Tuple[float,...]
    betas: Tuple[float,...]
    gammas: Tuple[float,...]
    nstages: int

# LSRK timestepper: includes an integrating factor for the dissipative terms.
# The loop over stages is now done with jax.lax.scan to enforce boundary between stages
# This seems to avoid so much CPU memory layout thrashing.
def lsrk_advance(state, kgrid, params, rhs, set_timestep, scheme):
    alphas_arr = jnp.array(scheme.alphas)
    betas_arr = jnp.array(scheme.betas)
    gammas_arr = jnp.array(scheme.gammas)

    init_rhs,grads = rhs(state,kgrid,params)
    dt = set_timestep(grads,params)

    diss_exponents = kgrid.hdiss_exponents(params) * dt

    init_delta = jnp.zeros_like(state.fields)
    init_carry = (state,init_delta)

    stage_pars = (alphas_arr, betas_arr, gammas_arr, jnp.arange(scheme.nstages))

    def scan_stage_func(carry,stage_vals):
        current_state, delta = carry
        alpha, beta, gamma, istage = stage_vals

        stage_rhs = jax.lax.cond(istage == 0,lambda: init_rhs,
                                 lambda: rhs(current_state,kgrid,params)[0])
        
        diss_factors = jnp.exp(diss_exponents*gamma)

        next_delta = diss_factors * (alpha * delta + dt * stage_rhs)
        next_fields = diss_factors * current_state.fields + beta*next_delta

        next_t = current_state.t + gamma*dt
        return (SimulationState(t=next_t,fields=next_fields),next_delta), None
    
    (final_state, _), _ = jax.lax.scan(scan_stage_func,init_carry,stage_pars)

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