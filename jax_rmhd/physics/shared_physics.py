import jax
import jax.numpy as jnp
import mpi4jax
from mpi4py import MPI

#Takes gradient in fourier space
#Expects the kx,ky axes to be the last two (-2,-1)
def gradk(fk,kgrid):
    return jnp.stack([1j*kgrid.kx*fk,1j*kgrid.ky*fk],axis=1)

# Poisson bracket of real-space fields A and B. Returns in real space
def bracket(a,b):
    return a[0]*b[1] - a[1]*b[0]

# Gets the necessary z derivatives.
def z_derivatives(f,params):
    #n.b. z axis is assumed to be axis 1
    dz=params.dz
    send_left = f[:,:2,:,:]
    send_right = f[:,-2:,:,:]
    recv_right = mpi4jax.sendrecv(send_left, send_left, dest=params.left_neighbor, source=params.right_neighbor,
                                     comm=params.cart_comm, sendtag=101, recvtag=101)
    recv_left = mpi4jax.sendrecv(send_right, send_right, dest=params.right_neighbor, source=params.left_neighbor,
                                    comm=params.cart_comm, sendtag=102, recvtag=102)
    f_padded = jnp.concatenate([recv_left,f,recv_right],axis=1)
    p2 = f_padded[:,4:,:,:]
    p1 = f_padded[:,3:-1,:,:]
    c = f_padded[:,2:-2,:,:]
    m1 = f_padded[:,1:-3,:,:]
    m2 = f_padded[:,:-4,:,:]
    df_dz = (- p2 + 8*p1 - 8*m1 + m2) / (12 * dz)
    d4f_dz4 = (p2 -4*p1 +6*c -4*m1 + m2) / (dz**4)
    return df_dz, d4f_dz4

############
# FORCING. #
############

def _symmetrize_real_line(col):
    # Enforces hermitian symmetry of the forcing: needed at ky=0 and kymax
    nkx = col.shape[-1]
    mirror_idx = (-jnp.arange(nkx)) % nkx
    return 0.5 * (col + jnp.conj(col[..., mirror_idx]))

def _draw_symmetrized_noise(key, shape, dtype, grid_norm):
    key_real, key_imag = jax.random.split(key)
    noise = (jax.random.normal(key_real, shape) + 1j * jax.random.normal(key_imag, shape)) / jnp.sqrt(2.0)
    noise = noise * grid_norm
    noise = noise.astype(dtype)
    noise = noise.at[..., 0].set(_symmetrize_real_line(noise[..., 0]))
    noise = noise.at[..., -1].set(_symmetrize_real_line(noise[..., -1]))
    return noise

def _shell_mask(kgrid, params):
    # Radially symmetric mask selecting nmin <= |k_perp|/dk < nmax.
    kunit = min(2*jnp.pi/params.Lx, 2*jnp.pi/params.Ly)
    nmin, nmax = params.fshell
    kmag_over_dk = jnp.sqrt(kgrid.ksq()) / kunit
    return (kmag_over_dk >= nmin) & (kmag_over_dk < nmax)

def ou_update(forcing_state, forcing_key, dt, params, kgrid):
    # Ornstein-Uhlenbeck step applied independently at every
    # (kx,ky), then restricted to the forcing shell.
    # Question: could this be made more efficient? or is the compiler able to see the mask
    key, new_key = jax.random.split(forcing_key)
    grid_norm = params.nx * params.ny
    # forcing_state has shape (n_ou, 2, nkx, nky)
    # n_ou is 1 for momentum forcing and 2 for elsasser
    # axis=1 has 2 components to set A and B in reconstruct_envelope below
    noise = _draw_symmetrized_noise(key, forcing_state.shape, forcing_state.dtype, grid_norm)
    decay = jnp.exp(-dt / params.forcing_tau)
    diffusion = jnp.sqrt(1.0 - decay**2)
    new_forcing_state = forcing_state * decay + diffusion * noise
    mask = _shell_mask(kgrid, params)
    new_forcing_state = new_forcing_state * mask[None, None, :, :]
    return new_forcing_state, new_key

def reconstruct_envelope(forcing_state, z_local, params):
    if params.spatial_dimensions == 3:
        cos_env = jnp.cos(2*jnp.pi*z_local/params.Lz)[:, None, None]
        sin_env = jnp.sin(2*jnp.pi*z_local/params.Lz)[:, None, None]
        A = forcing_state[:, 0]  # (n_ou, nkx, nky)
        B = forcing_state[:, 1]
        return A[:, None, :, :] * cos_env + B[:, None, :, :] * sin_env
    else:
        #in 2D just use the A coefficient
        return forcing_state[:, 0][:, None, :, :]  # (n_ou, 1, nkx, nky)

def _perp_yfac(kgrid):
    # rfft2 y-doubling factor
    nky = kgrid.ky.shape[-1]
    return jnp.full((nky,), 2.0).at[0].set(1.0).at[-1].set(1.0)

def _perp_reduce(integrand, params):
    # sum over all axes (z-local, kx, ky, ...), allreduce over z-ranks if applicable
    # normalize: divide by nz*(nx*ny)^2.
    P = jnp.sum(integrand)
    if params.cart_comm is not None:
        P = mpi4jax.allreduce(P, op=MPI.SUM, comm=params.cart_comm)
    return P / (params.nz * (params.nx * params.ny)**2)

def perp_inner_product(field_a_k, field_b_k, kgrid, params):
    # Re( sum_k grad(field_a_k)^* . grad(field_b_k) )
    # useful for e.g. energies and power inputs
    grad_a = gradk(field_a_k, kgrid)  # (nz_local, 2, nkx, nky)
    grad_b = gradk(field_b_k, kgrid)
    integrand = jnp.real(jnp.conj(grad_a) * grad_b) * _perp_yfac(kgrid)
    return _perp_reduce(integrand, params)

def perp_mean_square(field_a_k, field_b_k, kgrid, params):
    # Re( sum_k field_a_k^* . field_b_k )
    # useful for anastrophy etc.
    integrand = jnp.real(jnp.conj(field_a_k) * field_b_k) * _perp_yfac(kgrid)
    return _perp_reduce(integrand, params)

def safe_scale(target, P, scale_max=1.0):
    # used to rescale power inputs
    scale = jnp.where(target == 0.0, 0.0, target / P)
    return jnp.clip(scale, -scale_max, scale_max)
