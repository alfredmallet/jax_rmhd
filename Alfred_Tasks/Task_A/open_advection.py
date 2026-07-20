import matplotlib.pyplot as plt
import numpy as np

# Open the archive using a context manager
with np.load('/Users/esromabraham/assure-2026/jax_rmhd/Alfred_Tasks/test_advectionnz_test.npz') as data:
    # 1. See what arrays are inside the archive
    nz = data['nz']
    l1 = data['l1']
    l2 = data['l2']

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,10))

ax1.plot(nz, l1, color='red')
ax1.set_ylabel("L1 error", weight='bold')
ax1.set_title("nz vs. L1 error", weight='bold')

ax2.plot(nz, l2, color='green')
ax2.set_xlabel("nz", weight='bold')
ax2.set_ylabel("L2 error", weight='bold')
ax2.set_title("nz vs. L2 error", weight='bold')

plt.savefig("/Users/esromabraham/assure-2026/jax_rmhd/poster_images/test_advection_results.png")