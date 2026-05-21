import jax.numpy as jnp
from typing import NamedTuple

#Simulation state should be represented as a tuple (t,fields), with fields a tuple of the primitive fields
class Fields(NamedTuple):
    #This object is for anything shaped like fields, including the RHS of the equations.
    phik: jnp.ndarray
    psik: jnp.ndarray

class SimulationState(NamedTuple):
    t: float
    fields: jnp.ndarray