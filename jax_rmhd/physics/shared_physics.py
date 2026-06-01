import jax.numpy as jnp
import mpi4jax

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
    recv_right, _ = mpi4jax.sendrecv(send_left, dest=params.left_neighbor, source=params.right_neighbor,
                                     comm=params.cart_comm, sendtag=101, recvtag=101)
    recv_left, _ = mpi4jax.sendrecv(send_right, dest=params.right_neighbor, source=params.left_neighbor,
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
