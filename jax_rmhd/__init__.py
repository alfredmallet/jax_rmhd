import os
import jax

precision = os.environ.get("RMHD_PRECISION","32")

if precision == "64":
    jax.config.update("jax_enable_x64", True)
    print("rmhd-solver has initialized jax in 64bit precision.")
else:
    jax.config.update("jax_enable_x64", False)
    print("rmhd-solver has initialized jax in 32bit precision.")

from .types import SimulationState, Fields
from .fourier import K_Grids, setup_kgrids
from .config import Parameters,init_cluster,setup_sharding
from .snapshot_io import snapshot_manager_setup
from .run import simulate,simulate_scan,estimate_good_nblock
