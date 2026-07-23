# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A pseudospectral solver for reduced MHD (RMHD) and related plasma fluid models, written in
JAX. Spectral (rfft2) in the perpendicular (x,y) plane, finite-difference (4th order
centered) in z, domain-decomposed across MPI ranks along z only. Currently only RMHD is
implemented; the architecture is designed to add other equation sets (gradient drift,
compressible RMHD, KRMHD, gyrokinetics, ...) without touching the core solver.

## Setup / running

```
pip install -e .
```

`pyproject.toml`'s `dependencies` list (`jax`, `jaxlib`, `orbax-checkpoint`, `numpy`) is
incomplete — `mpi4py`, `mpi4jax`, and `tensorstore` are hard requirements (used throughout
`config.py`, `physics/`, `snapshot_io.py`), and `matplotlib` is needed for the example
notebooks/test scripts. Install these manually if not already present.

Precision is controlled by an env var read at import time, not a runtime flag:
```
RMHD_PRECISION=64 python your_script.py   # float64/complex128; default is 32
```

There's no pytest suite and no lint/format config in this repo. "Tests" under `tests/` are
standalone scripts (not pytest — no fixtures/assertions framework, they print/plot results
for a human to interpret), run directly:
```
python tests/test_dissipation.py
python tests/test_advection.py
python tests/test_forcing_smoke.py     # does use plain assert-and-print PASS/FAIL checks
```
For MPI-parallel (z-decomposed 3D) runs, launch under `mpirun`, e.g.:
```
mpirun -n 4 python tests/test_advection.py
```
2D runs (`dims=2`) are single-process only — `Parameters.__init__` prints a warning if run
with `size>1`. `tests/savio_scaling/` and `slurms/` are SLURM/cluster scaling-test scripts
for the Savio HPC cluster specifically (hardcoded paths), not general-purpose examples.

`examples/*.ipynb` are worked examples of varying freshness — several predate the current
API (`jr.Fields`, positional `SimulationState(...)` construction) and will error as-is;
`orzag-tang-2D.ipynb`, `orzag-tang-3d.ipynb`, `forced-turbulence-2D.ipynb`, and
`forced-turbulence-3D.ipynb` are current.

## Architecture

### Field representation

`SimulationState` (NamedTuple, `types.py`) is `(t, fields, forcing_state, forcing_key)`.
`fields` has shape `(nfields, nz_local, nkx, nky)`: real-space grid in z, rfft2 spectral in
(x,y). **This axis is never dropped, even in 2D** — `dims=2` still produces
`(nfields, 1, nx, ny//2+1)`, not a 3D-rank array; `run.py::initialize` reshapes the x,y
coordinate grids with a leading axis of 1 unconditionally, 2D or 3D.

rfft2 convention: `kx` (`grids.py::K_Grids`) is full two-sided (`fftfreq`), `ky` is
half/non-negative (`rfftfreq`) — real-space reality is a constraint *between* `(kx,ky)` and
`(-kx,ky)` at `ky=0` and `ky=Nyquist`, not a per-mode constraint. Anything that writes
directly into k-space rather than deriving it via fft of a real field (e.g. stochastic
forcing) must enforce this explicitly or the reconstructed field silently isn't real at
those rows. `jnp.fft.rfft2`/`irfft2` (`grids.py::fft`/`ifft`) are unnormalized transforms:
an O(1) real-space field has raw coefficients of magnitude O(nx*ny), not O(1) — matters for
any synthetic k-space process whose amplitude is meant to be grid-resolution-independent.

### Parameters / physics registry

`Parameters` (`config.py`) is a JAX pytree with **all fields as static aux_data**
(`tree_flatten` returns empty `children`) — every attribute is a compile-time constant
under `jax.jit`, so plain Python `if params.foo:` branching in physics code is correct and
preferred over `jax.lax.cond`. z-related attributes (`dz`, `Lz`, `z_diss`, `cart_comm`,
`left_neighbor`/`right_neighbor`) only exist when `dims==3`; guard access to them.
`z_diff_order`/`z_diss_hyper` are accepted by `Parameters.__init__` and stored, but
`physics/rmhd.py::LinearTerm` doesn't read them back — z-derivatives are hardcoded to 4th
order and z-hyperdissipation to the `z_diss_hyper=2` form regardless of what's passed in
(see the `#TODO` in `LinearTerm`); passing a non-default value silently does nothing.

New equation sets register via `physics/__init__.py`'s `equation_registry`: an
`EquationRecipe(set_timestep_func, term_funcs, grad_func)` per `eqtype`. `term_funcs` are
summed to build the RHS (`construct_rhs`); dissipation is *not* one of them — it's applied
separately as an integrating factor in `timestepping.py` (`K_Grids.hdiss_exponents`), not
as an RHS term. `physics/shared_physics.py` holds equation-agnostic helpers (`gradk`,
`bracket`, z-derivative stencils, the O-U forcing mechanics); `physics/rmhd.py` holds the
RMHD-specific term functions and maps generic building blocks onto the (phi,psi) fields.

### Timestepping

`rk_advance`/`lsrk_advance` (`timestepping.py`) rebuild intermediate `SimulationState`s at
every RK/LSRK sub-stage. **Always use `state._replace(...)`, never positional
`SimulationState(t, fields)` construction** — the latter silently drops/misaligns any
fields beyond the first two now that the tuple has grown (this bit us adding forcing:
`forcing_state`/`forcing_key` must survive unchanged across sub-stages within a timestep,
since they're only updated once per full step, not per sub-stage).

### Stochastic forcing (`params.forcing`)

Ornstein-Uhlenbeck process injecting power into a shell of perpendicular wavenumbers,
sustaining turbulence instead of letting it freely decay. `forcing_state` shape
`(n_ou, 2, nkx, nky)`: axis 0 is 1 (`forcing_mode="momentum"`, forces phi only) or 2
(`"elsasser"`, forces z+ = phi+psi and z- = phi-psi independently, each with its own
`forcing_power_elsasser` target); axis 1 is the [A,B] cosine/sine z-envelope coefficients
(dims=3) or just uses A directly (dims=2, no z to project onto).

- `shared_physics.ou_update`/`reconstruct_envelope`/`perp_inner_product`/
  `perp_mean_square` are equation-agnostic; `rmhd.ForcingTerm` does the RMHD-specific power
  normalization and (phi,psi) mapping.
- Power normalization (`perp_inner_product`, `safe_scale`) targets exact injection power
  regardless of current field amplitude — cap the *scale factor* (`forcing_scale_max`), not
  the denominator (`P`): flooring `P` near zero produces wildly wrong (sign-flipped or
  enormous) results when `P` is small-but-nonzero or exactly 0 (e.g. the very first
  forcing evaluation from an all-zero initial condition), whereas capping the resulting
  scale factor bounds the worst case directly regardless of `P`'s units/scale.
- All `perp_*` energy-like reductions share one normalization convention (rfft2 `ky`
  y-doubling factor, divide by `nz*(nx*ny)^2`) to give a volume-averaged,
  grid-resolution-independent physical quantity — matches `diagnostics.perpspec`'s
  convention (note: `diagnostics.py`'s `energy`/`parspec` functions are marked broken in
  the file itself). Keep any new energy-like diagnostic on this same convention or its
  numbers won't be comparable to `forcing_power`.
- 2D MHD (RMHD's `dims=2` limit) is *not* 2D hydro: with `forcing_mode="momentum"` and a
  quiescent start, `psi` stays exactly zero forever (its only 2D source term,
  `-bracket(gphi,gpsi)`, vanishes identically when `psi=0`) — that's pure hydro, use
  `"elsasser"` for actual MHD turbulence. Energy cascades forward/direct in 2D MHD
  (opposite of 2D hydro's inverse cascade), but mean-square flux function `<psi^2>`
  (`perp_mean_square`) inverse-cascades to large scales regardless — don't read a plateaued
  energy spectrum as "fully saturated" without also checking `<psi^2>` isn't still
  climbing. Per the zeroth law of turbulence / dissipative anomaly, `visc`/`res` set the
  time to reach saturation and the dissipation-range cutoff, not the saturated amplitude
  itself (given adequate resolution).

### Checkpointing

`snapshot_io.py` save/restore is orbax-based, one `CheckpointManager` per MPI rank
(`snapshot_manager_setup`). `snapshot_manager_setup`'s `nsnap` arg is passed straight
through as orbax's `max_to_keep` — it used to be silently unwired (every run kept every
snapshot forever regardless of `nsnap`); it's now honored, so old runs' checkpoint
directories may hold more snapshots than a run made after this fix would produce.
`load_snapshot` supports restoring onto a different rank count
than was saved (`p_save` vs `params.size`) by unioning overlapping z-slices per field —
`forcing_state`/`forcing_key` are **not** part of that union (they have no z-axis and are
identical across all saved ranks by construction, since forcing is perpendicular-only and
kept in sync across ranks); they're restored once, directly from rank 0's checkpoint.
`forcing_key`'s dtype is obtained via `jax.eval_shape(lambda: jax.random.key(0)).dtype`
rather than a guessed public constant. Use `get_saved_steps(snap_path)` rather than
`mngr.all_steps()` to enumerate saved snapshot indices — `get_saved_steps` detects and
correctly handles a resharded (multi-rank-saved) layout, whereas a bare `mngr.all_steps()`
on a top-level manager over such a layout misreads the numbered rank subfolders as step
numbers.
