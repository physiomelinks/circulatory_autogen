"""States whose initial value libCellML computes from constants.

A CellML model can initialise a state from another variable (``V = V_init`` with
``V_init = f(constants)``). libCellML then sets that state inside ``compute_computed_constants``
rather than in ``initialise_variables``. The generated function writes such states into its optional
``states`` argument (absent for models that have none).
"""
import inspect
import math


def computed_initial_states(model, variables):
    """``{state index: value}`` of the states ``model.compute_computed_constants`` sets from ``variables``.

    ``variables`` is read and, as always, has its computed constants written. The states array
    starts as NaN (``create_states_array``), so the entries that change are exactly the ones set.
    """
    if "states" not in inspect.signature(model.compute_computed_constants).parameters:
        model.compute_computed_constants(variables)
        return {}
    states = model.create_states_array()
    model.compute_computed_constants(variables, states)
    return {i: v for i, v in enumerate(states) if not (isinstance(v, float) and math.isnan(v))}
