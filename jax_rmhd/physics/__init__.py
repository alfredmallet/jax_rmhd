from typing import NamedTuple,Tuple,Callable
from . import rmhd

class EquationRecipe(NamedTuple):
    set_timestep_func: Callable
    term_funcs: Tuple[Callable,...]
    grad_func: Callable


# Constructs the ideal RHS of the equations using the relevant EquationRecipe.
# NB: The dissipative terms are handled via integrating factor in timestepping.py
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

equation_registry = {
    "RMHD": EquationRecipe(set_timestep_func = rmhd.set_timestep,
                           term_funcs = (rmhd.NonlinearTerm, rmhd.LinearTerm),
                           grad_func = rmhd.grad
                           ),
}