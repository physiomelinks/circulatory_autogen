"""Solver-name aliases.

`CVODE_opencor` is an alias for `CVODE_libopencor`. The user-input parser, solver_info
validation and the solver factory all call `canonical_solver_name`, so an alias resolves the
same way wherever it is given.
"""
import warnings

#: Alias -> the solver it names. `CVODE_opencor` was OpenCOR's Python shell; libOpenCOR is
#: its pip-installable engine.
SOLVER_ALIASES = {
    'CVODE_opencor': 'CVODE_libopencor',
}


def canonical_solver_name(solver):
    """Return `solver`, with an alias replaced by the solver it names (with a FutureWarning)."""
    target = SOLVER_ALIASES.get(solver)
    if target is None:
        return solver
    warnings.warn(f"solver {solver} now runs libOpenCOR; name it {target}",
                  FutureWarning, stacklevel=3)
    return target
