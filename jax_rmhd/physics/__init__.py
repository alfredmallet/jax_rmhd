from typing import NamedTuple,Tuple,Callable
import jax.numpy as jnp
from . import rmhd
from ..types import LinearMatrices

class EquationRecipe(NamedTuple):
    set_timestep_func: Callable
    term_funcs: Tuple[Callable,...]
    grad_func: Callable
    linear_matrix: Callable

# Constructs the ideal RHS of the equations using the relevant EquationRecipe.
# This now also returns the matrix of any linear terms to handled exactly via integrating factor.
def construct_rhs(recipe):
    def rhs(state,kgrid,params):
        grads=recipe.grad_func(state,kgrid,params)
        fields_rhs = None
        for term in recipe.term_funcs:
            if fields_rhs is None:
                fields_rhs = term(state,grads,kgrid,params)
            else:
                fields_rhs = fields_rhs + term(state,grads,kgrid,params)
        return fields_rhs, grads
    return rhs
    
def construct_linear_matrices(kgrid,params,recipe):
    matrix=recipe.linear_matrix(kgrid,params)
    #If the coeffs are diagonal, it should be stored as (nfields,1,nkx,nky)
    if matrix.ndim == 4:
        return LinearMatrices(diag=matrix,proj=None,proj_inv=None)
    #If it isn't diagonal, diagonalize it and return everything you need
    matrix_t = jnp.transpose(matrix, (2, 3, 4, 0, 1))
    diag_t, proj_t = jnp.linalg.eig(matrix_t)
    proj_inv_t = jnp.linalg.inv(proj_t)
    diag = jnp.transpose(diag_t, (3,0,1,2))
    proj = jnp.transpose(proj_t, (3,4,0,1,2))
    proj_inv = jnp.transpose(proj_inv_t, (3,4,0,1,2))
    return LinearMatrices(diag=diag,proj=proj,proj_inv=proj_inv)

equation_registry = {
    "RMHD": EquationRecipe(set_timestep_func = rmhd.set_timestep,
                           term_funcs = (rmhd.NonlinearTerm, rmhd.LinearTerm),
                           grad_func = rmhd.grad,
                           linear_matrix = rmhd.diss_matrix),
}