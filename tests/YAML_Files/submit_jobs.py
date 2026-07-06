import yaml
from simple_slurm import Slurm

with open("/global/home/users/esromabraham/jax_rmhd/tests/config", "r") as f:
    config = yaml.safe_load(f)

# Converts to slurm parameters (like cpus, nodes, etc.)
slurm_settings = {k: v for k, v in config.items() if k != 'test_params'}
slurm = Slurm(**slurm_settings)

# Extract the grid point variables
params = config["test_params"]

# DYNAMICALLY BUILD THE RUN COMMAND
# This command grabs the nx, ny, and nz values, and converts them into command flags
command = (
    f"export JAX_PLATFORMS=cpu && "
    f"time srun python /global/home/users/esromabraham/jax_rmhd/tests/slurm_files/savio_scaling/test_savio_scaling.py "
    f"--nx {params['nx']} "
    f"--ny {params['ny']} "
    f"--nz {params['nz']}"
)

# 5. Generate and print the complete SBATCH script text for your peace of mind
print("="*40)
print(" PREVIEW OF THE GENERATED SLURM SCRIPT ")
print("="*40)
print(slurm.script(command))
print("="*40)

# 6. Prompt for manual confirmation before submission
confirm = input("Does the script look correct? Submit to Savio? (y/n): ").strip().lower()

if confirm == 'y':
    job_id = slurm.sbatch(command)
    print(f"\n🚀 Success! Job submitted to Savio. Job ID: {job_id}")
else:
    print("\n❌ Submission cancelled.")


