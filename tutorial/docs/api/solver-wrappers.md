# Simulation

Run forward simulations of a generated model. The factory functions return a
`SimulationHelper` for the configured solver (OpenCOR, Myokit, SciPy, or
CasADi); all backends share the common method surface documented below.

## Factory functions

::: libcuflynx.solver_wrappers.get_simulation_helper

::: libcuflynx.solver_wrappers.get_simulation_helper_from_inp_data_dict

## SimulationHelper

The same interface is implemented by every backend. It is documented here on
the SciPy/Python backend; the OpenCOR, Myokit, and CasADi backends expose the
same methods.

::: libcuflynx.solver_wrappers.python_solver_helper.SimulationHelper
