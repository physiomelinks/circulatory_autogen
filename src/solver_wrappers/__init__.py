"""
Solver wrapper factory.

Provides access to OpenCOR, Myokit, and SciPy-based solvers through a common API.
"""
import os
from solver_wrappers.python_solver_helper import SimulationHelper as PythonSimulationHelper
from solver_wrappers.myokit_helper import SimulationHelper as MyokitSimulationHelper

try:
    from solver_wrappers.opencor_helper import SimulationHelper as OpenCORSimulationHelper
except Exception:
    OpenCORSimulationHelper = None


def get_simulation_helper(solver: str = None, model_type: str = None, model_path: str = None):
    """
    Return the appropriate SimulationHelper class based on solver parameter.

    - CVODE: Use OpenCORsolver CVODE for CellML models (default)
    - CVODE_myokit: Use Myokit CVODE solver for CellML models
    - solve_ivp: methods (RK45, BDF, etc.): Use Python/SciPy solver for Python models
    """
    # Define valid solver types
    cellml_solvers = ['CVODE', 'CVODE_myokit']
    python_solvers = ['solve_ivp']
    solve_ivp_methods = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA', 'RK4', 'forward_euler']

    # Determine if this is a Python model
    is_python_model = (model_type == 'python' or
                      (isinstance(model_path, str) and model_path.endswith('.py')))

    # Check for explicit solver specification with validation
    if solver == 'CVODE':
        if is_python_model:
            raise ValueError("CVODE_opencor solver cannot be used with Python models. Use a solve_ivp method instead.")
        if OpenCORSimulationHelper is not None:
            return OpenCORSimulationHelper
        else:
            raise RuntimeError("OpenCOR solver requested but OpenCOR is not available")
    elif solver == 'CVODE_myokit':
        if is_python_model:
            raise ValueError("CVODE solver cannot be used with Python models. Use a solve_ivp method instead.")
        return MyokitSimulationHelper
    elif solver in python_solvers:
        if not is_python_model:
            raise ValueError(f"solve_ivp method {solver} can only be used with Python models. Use CVODE or CVODE_opencor for CellML models.")
        return PythonSimulationHelper
    elif solver is not None:
        # Unknown solver type
        raise ValueError(f"Unknown solver {solver}. Valid options are: {cellml_solvers + python_solvers}")

    # Backward compatibility logic
    if is_python_model:
        return PythonSimulationHelper
    # Default to OpenCOR for CellML models
    return OpenCORSimulationHelper


__all__ = [
    "get_simulation_helper",
    "PythonSimulationHelper",
    "MyokitSimulationHelper",
    "OpenCORSimulationHelper",
]
