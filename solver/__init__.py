import os
import jax

precision = os.environ.get("RMHD_PRECISION","32")

if precision == "64":
    jax.config.update("jax_enable_x64", True)
    print("rmhd-solver has initialized jax in 64bit precision.")
else:
    jax.config.update("jax_enable_x64", False)
    print("rmhd-solver has initialized jax in 32bit precision.")

from .physics import SimulationState, Fields
from .fourier import K_Grids, setup_kgrids
from .config import Parameters
from .snapshot_io import snapshot_manager_setup
