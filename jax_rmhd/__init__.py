import os
import jax

precision = os.environ.get("RMHD_PRECISION","32")

if precision == "64":
    jax.config.update("jax_enable_x64", True)
    print("jax is using 64bit precision.")
else:
    jax.config.update("jax_enable_x64", False)
    print("jax is using 32bit precision.")

from .types import SimulationState
from .fourier import K_Grids, setup_kgrids
from .config import Parameters,init_cluster
from .snapshot_io import snapshot_manager_setup
from .run import simulate,simulate_scan,estimate_good_nblock,initialize
