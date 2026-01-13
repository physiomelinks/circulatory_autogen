"""
Solver wrapper factory.

Provides access to OpenCOR and SciPy-based solvers through a common API.
"""
from solver_wrappers.opencor_helper import SimulationHelper as OpenCORSimulationHelper
from solver_wrappers.python_solver_helper import SimulationHelper as PythonSimulationHelper


def get_simulation_helper(model_type: str = None, model_path: str = None):
    """
    Return the appropriate SimulationHelper class.
    Chooses Python solver if model_type == 'python' or the model_path ends with '.py'.
    """
    if model_type == 'python' or (isinstance(model_path, str) and model_path.endswith('.py')):
        return PythonSimulationHelper
    return OpenCORSimulationHelper


__all__ = ["get_simulation_helper", "OpenCORSimulationHelper", "PythonSimulationHelper"]



