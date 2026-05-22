import jax.numpy as jnp
from typing import NamedTuple

#Simulation state should be represented as a tuple (t,fields), with fields an array (nfields,nz,nx,ny)
class SimulationState(NamedTuple):
    t: float
    fields: jnp.ndarray