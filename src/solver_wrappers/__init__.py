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
    from mpi4py import MPI
    _MPI_AVAILABLE = True
except Exception:
    _MPI_AVAILABLE = False


def get_simulation_helper(model_path: str = None, solver: str = None, 
                          model_type: str = None, dt: float = None, sim_time: float = None, 
                          solver_info: dict = None, pre_time: float = 0.0):
    """
    Return the appropriate SimulationHelper class based on solver parameter.

    - CVODE_opencor: Use OpenCOR solver CVODE for CellML models (default)
    - CVODE_myokit: Use Myokit CVODE solver for CellML models
    - solve_ivp: methods (RK45, BDF, etc.): Use Python/SciPy solver for Python models
    - casadi_integrator: methods (cvodes, idas, collocation, rk): Use CasADi integrator for CasADi Python models
    """
    # Define valid solver types
    cellml_solvers = ['CVODE_opencor', 'CVODE_myokit']
    python_solvers = ['solve_ivp']
    solve_ivp_methods = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA', 'RK4', 'forward_euler']
    casadi_solvers = ['casadi_integrator']

    # Determine if this is a Python model
    is_python_model = (model_type == 'python')
    is_casadi_python_model = (model_type == 'casadi_python')

    # Check for explicit solver specification with validation
    if solver in ('CVODE_opencor'):
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
    elif solver is not None:
        # Unknown solver type
        raise ValueError(f"Unknown solver {solver}. Valid options are: {cellml_solvers} for CellML models, {python_solvers} for Python models, and {casadi_solvers} for CasADi Python models.")

    # Backward compatibility logic
    if is_python_model:
        return PythonSimulationHelper(model_path, dt, sim_time, solver_info, pre_time=pre_time)
    # Default to OpenCOR for CellML models
    return OpenCORSimulationHelper(model_path, dt, sim_time, solver_info, pre_time=pre_time)

def get_simulation_helper_from_inp_data_dict(inp_data_dict):
    return get_simulation_helper(model_path=inp_data_dict["model_path"], solver=inp_data_dict["solver_info"]["solver"], model_type=inp_data_dict["model_type"], dt=inp_data_dict["dt"], sim_time=inp_data_dict["sim_time"], solver_info=inp_data_dict["solver_info"], pre_time=inp_data_dict["pre_time"])

__all__ = [
    "get_simulation_helper",
    "PythonSimulationHelper",
    "MyokitSimulationHelper",
    "OpenCORSimulationHelper",
    "CasADiPythonSimulationHelper",
]
