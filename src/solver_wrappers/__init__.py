"""
Solver wrapper factory.

Provides access to OpenCOR, Myokit, and SciPy-based solvers through a common API.
"""
import os
from solver_wrappers.python_solver_helper import SimulationHelper as PythonSimulationHelper
try:
    from solver_wrappers.myokit_helper import SimulationHelper as MyokitSimulationHelper
except:
    MyokitSimulationHelper = None

try:
    from solver_wrappers.opencor_helper import SimulationHelper as OpenCORSimulationHelper
except Exception:
    OpenCORSimulationHelper = None

try:
    from solver_wrappers.casadi_python_solver_helper import SimulationHelper as CasADiPythonSimulationHelper
except Exception:
    CasADiPythonSimulationHelper = None

try:
    from solver_wrappers.aadc_python_solver_helper import SimulationHelper as AadcPythonSimulationHelper
except Exception:
    AadcPythonSimulationHelper = None

try:
    from mpi4py import MPI
    _MPI_AVAILABLE = True
except Exception:
    _MPI_AVAILABLE = False


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

    # Determine if this is a Python model
    is_python_model = (model_type == 'python')
    is_casadi_python_model = (model_type == 'casadi_python')
    is_aadc_python_model = (model_type == 'aadc_python')

    # Check for explicit solver specification with validation
    if solver == 'CVODE_opencor':
        if is_python_model:
            raise ValueError("CVODE_opencor solver cannot be used with Python models. Use a solve_ivp method instead.")
        if OpenCORSimulationHelper is not None:
            return OpenCORSimulationHelper(model_path, dt, sim_time, solver_info, pre_time=pre_time)
        else:
            raise RuntimeError("OpenCOR solver requested but OpenCOR is not available")
    elif solver == 'CVODE_myokit':
        if is_python_model:
            raise ValueError("CVODE_myokit solver cannot be used with Python models. Use a solve_ivp method instead.")
        if MyokitSimulationHelper is not None:
            return MyokitSimulationHelper(model_path, dt, sim_time, solver_info, pre_time=pre_time)
        else:
            raise RuntimeError("Myokit solver requested but Myokit is not available")
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
            raise RuntimeError("CasADi solver requested but CasADi is not available")
    elif solver in aadc_solvers:
        if not is_aadc_python_model:
            raise ValueError(f"Solver {solver} can only be used for AADC Python models (model_type='aadc_python').")
        if AadcPythonSimulationHelper is not None:
            return AadcPythonSimulationHelper(model_path, dt, sim_time, solver_info, pre_time=pre_time)
        else:
            raise RuntimeError("AADC solver requested but aadc package is not installed. pip install aadc")
    elif solver is not None:
        # Unknown solver type
        raise ValueError(f"Unknown solver {solver}. Valid options are: {cellml_solvers} for CellML models, {python_solvers} for Python models, and {casadi_solvers} for CasADi Python models.")

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
