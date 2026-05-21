import jax.numpy as jnp

#Takes gradient in fourier space
#Expects the kx,ky axes to be the last two (-2,-1)
def gradk(fk,kgrid):
    return jnp.stack([1j*kgrid.kx*fk,1j*kgrid.ky*fk],axis=1)

# Poisson bracket of real-space fields A and B. Returns in real space
def bracket(a,b):
    return a[0]*b[1] - a[1]*b[0]

# Gets the necessary z derivatives. This could be done more elegantly.
def z_derivatives(f,dz):
    p1 = jnp.roll(f,-1,axis=1)
    p2 = jnp.roll(f,-2,axis=1)
    m1 = jnp.roll(f, 1,axis=1)
    m2 = jnp.roll(f, 2,axis=1)
    df_dz = (- p2 + 8*p1 - 8*m1 + m2) / (12 * dz)
    d4f_dz4 = (p2 -4*p1 +6*f -4*m1 + m2) / (dz**4)
    return df_dz, d4f_dz4
