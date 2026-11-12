"""Removed solver names, and the solver that replaces each.

Read by the user-input parser, solver_info validation and the solver factory, so a removed
name fails the same way wherever it is given.
"""

#: Removed solver name -> the solver to use instead.
REMOVED_SOLVERS = {
    # The OpenCOR desktop application's Python shell; libOpenCOR is its pip-installable engine.
    'CVODE_opencor': 'CVODE_libopencor',
}


def removed_solver_message(solver):
    """The error for a removed solver name, naming its replacement."""
    replacement = REMOVED_SOLVERS[solver]
    return (
        f"Solver {solver!r} has been removed: it needed OpenCOR's bundled Python shell. "
        f"Use solver {replacement!r} (libOpenCOR, `pip install \"libcuflynx[libopencor]\"`, "
        "Python 3.12 or later), or 'CVODE_myokit', which is installed with libcuflynx."
    )
