import jax
import jax.numpy as jnp
from typing import NamedTuple

#Simulation state should be represented as a tuple (t,fields), with fields an array (nfields,nz,nx,ny)
class SimulationState(NamedTuple):
    t: float
    fields: jnp.ndarray         # shape (nfields, nz_local, nkx, nky), complex
    forcing_state: jnp.ndarray  # shape (n_ou, 2, nkx, nky), complex
                                # axis 0: n_ou = 1 ("momentum" mode) or 2 ("elsasser" mode: z+, z-)
                                # axis 1: [A, B] cosine/sine z-envelope coefficients
    forcing_key: jax.Array      # JAX PRNG key (typed key)
