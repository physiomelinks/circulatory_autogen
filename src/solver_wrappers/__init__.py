"""
Solver wrapper factory.

Provides access to OpenCOR, Myokit, and SciPy-based solvers through a common API.
"""
import os
import traceback

from solver_wrappers.python_solver_helper import SimulationHelper as PythonSimulationHelper

#: Why each optional backend failed to import, keyed by backend name. A backend is optional
#: because it may genuinely not be installed -- but the import can also fail for reasons that
#: have nothing to do with installation, and those used to be indistinguishable: every failure
#: became a bare `None` and then the same "X is not available" message, whatever went wrong.
#: A real example is a fresh CI runner where two MPI ranks import myokit at once and race on
#: creating ~/.config/myokit/myokit.ini; the message said myokit was not installed, when it was.
#: Keeping the reason turns that into something a user can act on (see _unavailable_message).
BACKEND_IMPORT_ERRORS = {}


def _record_import_error(name, exc):
    BACKEND_IMPORT_ERRORS[name] = ''.join(
        traceback.format_exception_only(type(exc), exc)).strip()


def _unavailable_message(backend, solver):
    """The error for a solver whose backend did not import, naming the underlying cause."""
    reason = BACKEND_IMPORT_ERRORS.get(backend)
    if not reason:
        return f"{backend} solver requested but {backend} is not available"
    return (f"{solver} solver requested but the {backend} backend failed to import: {reason}. "
            f"If {backend} is installed, this is not an installation problem -- the import "
            f"itself raised, and that error is the one to fix.")


try:
    from solver_wrappers.myokit_helper import SimulationHelper as MyokitSimulationHelper
except Exception as _exc:                                # noqa: BLE001 - reason is recorded
    MyokitSimulationHelper = None
    _record_import_error('Myokit', _exc)

try:
    from solver_wrappers.opencor_helper import SimulationHelper as OpenCORSimulationHelper
except Exception as _exc:
    OpenCORSimulationHelper = None
    _record_import_error('OpenCOR', _exc)

try:
    from solver_wrappers.casadi_python_solver_helper import SimulationHelper as CasADiPythonSimulationHelper
except Exception as _exc:
    CasADiPythonSimulationHelper = None
    _record_import_error('CasADi', _exc)

try:
    from solver_wrappers.aadc_python_solver_helper import SimulationHelper as AadcPythonSimulationHelper
except Exception as _exc:
    AadcPythonSimulationHelper = None
    _record_import_error('AADC', _exc)

# Not `from mpi4py import MPI`. This module picks a solver; it has no collectives
# to run. That import initialised MPI and registered an atexit MPI_Finalize for
# every consumer who merely wanted a forward solve -- and that finalise aborts on
# macOS when a NIC goes away, on machines with no MPI installed (#396).
# mpi_utils answers "is it installed" without opening it.
from utilities.mpi_utils import mpi_available as _mpi_available

_MPI_AVAILABLE = _mpi_available()


def get_simulation_helper(model_path: str = None, solver: str = None, 
                          model_type: str = None, dt: float = None, sim_time: float = None, 
                          solver_info: dict = None, pre_time: float = 0.0):
    """Create a `SimulationHelper` for the requested solver.

    Returns the appropriate backend (OpenCOR, Myokit, SciPy, or CasADi) based on
    ``solver`` and ``model_type``. All backends share the common
    [`SimulationHelper`][solver_wrappers.python_solver_helper.SimulationHelper]
    method surface.

    Args:
        model_path: Path to the generated model file.
        solver: Solver identifier. One of:

            - ``'CVODE_opencor'``: OpenCOR CVODE for CellML models (default).
            - ``'CVODE_myokit'``: Myokit CVODE for CellML models.
            - ``'solve_ivp'``: Python/SciPy solver for ``model_type='python'``
              (method set via ``solver_info``, e.g. RK45, BDF).
            - ``'casadi_integrator'``: CasADi integrator for
              ``model_type='casadi_python'`` (cvodes, idas, collocation, rk).
        model_type: ``'cellml_only'``, ``'python'`` or ``'casadi_python'``.
        dt: Output sampling step (s).
        sim_time: Logged simulation duration (s).
        solver_info: Solver config dict (e.g. ``MaximumStep``, ``method``).
        pre_time: Unlogged steady-state spin-up duration (s).

    Returns:
        SimulationHelper: The backend instance for the requested solver.

    Raises:
        ValueError: If the solver is unknown or incompatible with ``model_type``.
        RuntimeError: If the requested backend is not installed.
    """
    # Define valid solver types
    cellml_solvers = ['CVODE_opencor', 'CVODE_myokit']
    python_solvers = ['solve_ivp']
    solve_ivp_methods = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA', 'RK4', 'forward_euler']
    casadi_solvers = ['casadi_integrator']
    aadc_solvers = ['aadc_semi_implicit']
    user_defined_solvers = ['user_defined']

    # Determine if this is a Python model
    is_python_model = (model_type == 'python')
    is_casadi_python_model = (model_type == 'casadi_python')
    is_aadc_python_model = (model_type == 'aadc_python')
    is_user_defined_model = (model_type == 'python_user_defined')

    # Check for explicit solver specification with validation
    if solver == 'CVODE_opencor':
        if is_python_model:
            raise ValueError("CVODE_opencor solver cannot be used with Python models. Use a solve_ivp method instead.")
        if OpenCORSimulationHelper is not None:
            return OpenCORSimulationHelper(model_path, dt, sim_time, solver_info, pre_time=pre_time)
        else:
            raise RuntimeError(_unavailable_message('OpenCOR', 'CVODE_opencor'))
    elif solver == 'CVODE_myokit':
        if is_python_model:
            raise ValueError("CVODE_myokit solver cannot be used with Python models. Use a solve_ivp method instead.")
        if MyokitSimulationHelper is not None:
            return MyokitSimulationHelper(model_path, dt, sim_time, solver_info, pre_time=pre_time)
        else:
            raise RuntimeError(_unavailable_message('Myokit', 'CVODE_myokit'))
    elif solver in python_solvers:
        if not is_python_model:
            raise ValueError(f"solve_ivp method {solver} can only be used with Python models. Use CVODE_opencor (or legacy CVODE) or CVODE_myokit for CellML models.")
        if not model_path.endswith('.py'):
            raise ValueError(f"model_path {model_path} does not end with .py, which is required for Python models")
        return PythonSimulationHelper(model_path, dt, sim_time, solver_info, pre_time=pre_time)
    elif solver in casadi_solvers:
        if not is_casadi_python_model:
            raise ValueError(f"Solver {solver} can only be used for CasADi Python models.")
        if CasADiPythonSimulationHelper is not None:
            return CasADiPythonSimulationHelper(model_path, dt, sim_time, solver_info, pre_time=pre_time)
        else:
            raise RuntimeError(_unavailable_message('CasADi', solver))
    elif solver in aadc_solvers:
        if not is_aadc_python_model:
            raise ValueError(f"Solver {solver} can only be used for AADC Python models (model_type='aadc_python').")
        if AadcPythonSimulationHelper is not None:
            return AadcPythonSimulationHelper(model_path, dt, sim_time, solver_info, pre_time=pre_time)
        else:
            raise RuntimeError("AADC solver requested but aadc package is not installed. pip install aadc")
    elif solver in user_defined_solvers:
        if not is_user_defined_model:
            raise ValueError(f"Solver {solver} can only be used for user-defined Python models (model_type='python_user_defined').")
        if not model_path.endswith('.py'):
            raise ValueError(f"model_path {model_path} does not end with .py, which is required for python_user_defined models (the wrapper module)")
        # The user wrapper is integrated by the shared SciPy PythonSimulationHelper.
        return PythonSimulationHelper(model_path, dt, sim_time, solver_info, pre_time=pre_time)
    elif solver is not None:
        # Unknown solver type
        raise ValueError(f"Unknown solver {solver}. Valid options are: {cellml_solvers} for CellML models, {python_solvers} for Python models, {casadi_solvers} for CasADi Python models, and {user_defined_solvers} for user-defined Python models.")

    # Backward compatibility logic
    if is_python_model:
        return PythonSimulationHelper(model_path, dt, sim_time, solver_info, pre_time=pre_time)
    # Default to OpenCOR for CellML models
    return OpenCORSimulationHelper(model_path, dt, sim_time, solver_info, pre_time=pre_time)

def get_simulation_helper_from_inp_data_dict(inp_data_dict):
    """Create a `SimulationHelper` from a configuration dict.

    Convenience wrapper around
    [`get_simulation_helper`][solver_wrappers.get_simulation_helper] that reads
    ``model_path``, ``solver_info`` (and its ``solver``), ``model_type``, ``dt``,
    ``sim_time`` and ``pre_time`` from the dict.

    Args:
        inp_data_dict: Configuration dict (see
            [`get_default_inp_data_dict`][utilities.utility_funcs.get_default_inp_data_dict]).

    Returns:
        SimulationHelper: The backend instance for the configured solver.
    """
    return get_simulation_helper(model_path=inp_data_dict["model_path"], solver=inp_data_dict["solver_info"]["solver"], model_type=inp_data_dict["model_type"], dt=inp_data_dict["dt"], sim_time=inp_data_dict["sim_time"], solver_info=inp_data_dict["solver_info"], pre_time=inp_data_dict["pre_time"])

__all__ = [
    "get_simulation_helper",
    "PythonSimulationHelper",
    "MyokitSimulationHelper",
    "OpenCORSimulationHelper",
    "CasADiPythonSimulationHelper",
]
