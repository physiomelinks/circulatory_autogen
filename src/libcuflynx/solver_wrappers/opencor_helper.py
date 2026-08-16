import traceback
import warnings

# `opencor` is the Python module OpenCOR's own interpreter provides. It is not on PyPI and
# cannot be shipped in a wheel, so `pip install libcuflynx` has every solver except this one.
# Importing this module must therefore not be fatal: the factory in solver_wrappers/__init__.py
# imports every backend eagerly, direct importers (plotting helpers, scripts) only want the
# class object, and a user who never asks for CVODE_opencor should never see OpenCOR mentioned.
# The failure belongs at the point the solver is *selected* -- see require_opencor(), which
# SimulationHelper.__init__ calls, i.e. when get_simulation_helper() constructs the backend.
try:
    import opencor as oc
except ImportError as _opencor_exc:
    oc = None
    #: Formatted `ImportError` from the attempted `import opencor`, or None when it succeeded.
    OPENCOR_IMPORT_ERROR = ''.join(
        traceback.format_exception_only(type(_opencor_exc), _opencor_exc)).strip()
else:
    OPENCOR_IMPORT_ERROR = None

from libcuflynx.solver_wrappers.name_resolver import VariableNameResolver

#: Where the docs describe the (deprecated) OpenCOR route.
OPENCOR_DOCS_URL = ('https://physiomelinks.github.io/circulatory_autogen/'
                    'getting-started/#deprecated-opencor-based-setup')


class OpenCORUnavailableError(RuntimeError):
    """`CVODE_opencor` was requested but this interpreter has no `opencor` module.

    A `RuntimeError` subclass so that callers catching the factory's documented
    `RuntimeError` for an unavailable backend keep working unchanged.
    """


def is_opencor_available():
    """True when the `opencor` module imported, i.e. `CVODE_opencor` can actually run."""
    return oc is not None


def opencor_unavailable_message(solver='CVODE_opencor'):
    """The message a user gets when they ask for OpenCOR in an environment without it.

    Names the drop-in alternative rather than only stating the problem: `CVODE_myokit`
    runs the same CellML model through the same CVODE integrator with no OpenCOR involved,
    so for almost every user it is a one-word edit in `user_inputs.yaml`.
    """
    return (
        f"The {solver} solver is not available: this Python cannot import `opencor` "
        f"({OPENCOR_IMPORT_ERROR or 'module not found'}). OpenCOR's Python module is not "
        "distributed on PyPI, so a pip install of libcuflynx ships every solver except this "
        "one.\n"
        "  - Use `solver: CVODE_myokit` instead. It is a drop-in replacement for "
        "CVODE_opencor: the same CellML model, integrated by CVODE, with no OpenCOR "
        "needed.\n"
        "  - Or run inside OpenCOR, whose bundled interpreter provides `opencor`. That "
        f"route is deprecated and documented at {OPENCOR_DOCS_URL}\n"
        "It is expected to be replaced by `pip install libopencor` once libOpenCOR reaches "
        "PyPI."
    )


def require_opencor(solver='CVODE_opencor'):
    """Raise `OpenCORUnavailableError` unless the `opencor` module is importable.

    A no-op when OpenCOR is present, so nothing about a real OpenCOR run changes.
    """
    if oc is None:
        raise OpenCORUnavailableError(opencor_unavailable_message(solver))


warnings.filterwarnings(
    "ignore",
    message=r"The value of the smallest subnormal for <class 'numpy\.float64'> type is zero\.",
    category=UserWarning,
    module=r"numpy\.core\.getlimits",
)

import numpy as np
import os
import sys
from importlib import import_module  # Only used for debugging
# from func_timeout import func_timeout, FunctionTimedOut


class SimulationHelper():
    def __init__(self, cellml_path, dt,
                 sim_time, solver_info=None,
                 pre_time=0.0):

        # Before anything else: constructing this helper is the moment CVODE_opencor is
        # actually selected, and it is the last point at which we can say why it will not work.
        require_opencor()

        # TODO comment this out
        # self.resource_module = import_module('psutil')

        self.cellml_path = cellml_path  # path to cellml file
        self.dt = dt  # time step
        self.protocol_info = None
        if sim_time is not None and pre_time is not None:
            self.stop_time = pre_time + sim_time  # full time of simulation
            self.pre_steps = int(pre_time/dt)  # number of steps to do before storing data (used to reach steady state)
            self.n_steps = int(sim_time/dt)  # number of steps for storing data
        else:
            self.stop_time = None
            self.pre_steps = None
            self.n_steps = None

        self.simulation = oc.open_simulation(cellml_path)
        if not self.simulation.valid():
            raise ValueError(f'simulation object opened from {cellml_path} is not valid')
        self.data = self.simulation.data()
        if solver_info is None:
            solver_info = {'MaximumNumberOfSteps': 5000, 'MaximumStep': 0.0001}
        
        valid_unused_keys=["method", "solver", "dt_solver"]
        for key, value in solver_info.items():
            if key == 'rtol':
                key = 'RelativeTolerance'
            if key == 'atol':
                key = 'AbsoluteTolerance'
            # ignore high-level/legacy keys that aren't part of OpenCOR solver properties
            if key.lower() in valid_unused_keys:
                continue
            if key not in self.data.odeSolverProperties():
                print(f'{key} is not a valid key for CVODE solver properties; valid keys are '
                      f'{list(self.data.odeSolverProperties())}. Skipping.')
                continue
            self.data.set_ode_solver_property(key, value)
        self.data.set_point_interval(self.dt)  # time interval for data storage
        self.data.set_starting_point(0)
        if pre_time is not None and sim_time is not None:
            self.data.set_ending_point(self.stop_time)
            self.tSim = np.linspace(pre_time, self.stop_time, self.n_steps + 1)  # time values for stored part of simulation
        else:
            self.tSim = None
        self._has_run = False
        self._last_results_dict = None
        
    def get_time(self, include_pre_time=False):
        if include_pre_time:
            return self.tSim
        else:
            return self.tSim - self.tSim[0]

    def set_protocol_info(self, protocol_info):
        """Store protocol metadata for a common helper API."""
        self.protocol_info = protocol_info

    def _resolve_name(self, name):
        """Resolve a name to (kind, key) using the unified VariableNameResolver."""
        return VariableNameResolver.resolve_key(
            name,
            [("state", self.data.states()), ("const", self.data.constants())],
            separator="/",
        )

    # inner psutil function # TODO only needed for memory checking
    def process_memory(self):
        process = self.resource_module.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss

    def run(self):
        try:
            self.simulation.run()
        except RuntimeError:
            print("Failed to converge")
            print('restarting simulation object')
            self.reset_and_clear()
            return False

        self._has_run = True
        return True

    def run_offline_pre_and_set_default_state(self, offline_pre_time):
        """Run unlogged warmup once; use end state as default for reset_states()."""
        offline_pre_time = float(offline_pre_time)
        if offline_pre_time <= 0:
            return
        self.update_times(self.dt, 0.0, self.dt, offline_pre_time)
        success = self.run()
        if not success:
            raise RuntimeError("Offline pre-time simulation failed")
        for state_name in self.data.states():
            series = self.simulation.results().states()[state_name].values()
            if len(series) > 0:
                self.data.states()[state_name] = float(series[-1])
        self.simulation.reset(False)
        self.simulation.clear_results()
        self._has_run = False

    def reset_and_clear(self, only_one_exp=-1):
        if self._has_run:
            self._last_results_dict = self._collect_all_results_dict()
        self.simulation.reset(True)
        self.simulation.release_all_values()
        self.simulation.clear_results()
        self._has_run = False

    def reset_states(self):
        self.simulation.reset(False)  # True resets everything, False resets only the states

    def get_all_variable_names(self):
        # get all states, algebraics and constants
        variable_names = list(self.simulation.results().states().keys()) + \
            list(self.simulation.results().algebraic().keys()) + \
            list(self.data.constants().keys())
        return variable_names

    def get_all_results(self, flatten=False):
        variable_names = self.get_all_variable_names()
        results = self.get_results(variable_names, flatten=flatten)
        return results

    def get_all_results_dict(self):
        if self._has_run:
            self._last_results_dict = self._collect_all_results_dict()
            return {name: np.asarray(val).copy() for name, val in self._last_results_dict.items()}
        if self._last_results_dict is not None:
            return {name: np.asarray(val).copy() for name, val in self._last_results_dict.items()}
        raise RuntimeError("Simulation has not been run yet.")

    def _collect_all_results_dict(self):
        variable_names = self.get_all_variable_names()
        values = self.get_results(variable_names, flatten=True)
        return {name: np.asarray(val) for name, val in zip(variable_names, values)}

    def get_results(self, variables_list_of_lists, flatten=False):
        # if the input is a list of variables, turn it into a list of lists
        if type(variables_list_of_lists[0]) is not list:
            variables_list_of_lists = [[entry] for entry in variables_list_of_lists]

        results = []
        for JJ, variables_list in enumerate(variables_list_of_lists):
            results.append([])
            for variable_name in variables_list:
                if variable_name == 'time':
                    results[JJ].append(self.tSim)
                elif variable_name in self.simulation.results().states():
                    results[JJ].append(self.simulation.results().states()[variable_name].values()[-self.n_steps - 1:].copy())
                elif variable_name in self.simulation.results().algebraic():
                    results[JJ].append(self.simulation.results().algebraic()[variable_name].values()[-self.n_steps-1:].copy())
                elif variable_name in self.data.constants():
                    const_val = self.data.constants()[variable_name]
                    results[JJ].append(np.ones_like(self.tSim) * const_val)
                else:
                    print(f'variable {variable_name} is not a model variable. model variables are')
                    print([name for name in self.simulation.results().states()])
                    print([name for name in self.simulation.results().algebraic()])
                    print([name for name in self.data.constants()])
                    print('exiting')
                    exit()

        if flatten:
            results = [item for sublist in results for item in sublist]
        return results

    def get_init_param_vals(self, param_names):
        param_init = []
        for JJ, param_name_or_list in enumerate(param_names):
            if not isinstance(param_name_or_list, list):
                param_name_or_list = [param_name_or_list]

            param_init.append([])
            for param_name in param_name_or_list:
                kind, resolved = self._resolve_name(param_name)
                if kind == "state":
                    param_init[JJ].append(self.data.states()[resolved])
                elif kind == "const":
                    param_init[JJ].append(self.data.constants()[resolved])
                else:
                    raise ValueError(
                        f"parameter name {param_name} not found in OpenCOR states/constants"
                    )

        return param_init

    def set_param_vals(self, param_names, param_vals, change_states=True):
        """Set parameter values, by default including any state initial values they drive.

        ``change_states=False`` is for mid-protocol updates that must preserve the state the
        previous sub-experiment evolved into; naming a state there is an error (see the Myokit
        backend for the full rationale).
        """
        if not change_states:
            offenders = []
            for param_name_or_list in param_names:
                names = (param_name_or_list if isinstance(param_name_or_list, list)
                         else [param_name_or_list])
                for param_name in names:
                    try:
                        kind, _ = self._resolve_name(param_name)
                    except Exception:
                        continue
                    if kind == "state":
                        offenders.append(param_name)
            if offenders:
                raise ValueError(
                    "set_param_vals(change_states=False) cannot set states directly, but was "
                    f"given: {', '.join(offenders)}. change_states=False exists for mid-protocol "
                    "updates that must preserve the evolved state.")

        # ensure param_vals stores state values first, then constant values
        for JJ, param_name_or_list in enumerate(param_names):
            if type(param_vals[JJ]) == str:
                raise NotImplementedError("Setting parameter values by name of protocol trace is not implemented for OpenCOR")
            elif type(param_vals[JJ]) not in [float, np.float64, int]:
                raise ValueError(f"Parameter value {param_vals[JJ]} is not a valid type. {type(param_vals[JJ])}" + \
                                 "must be a float, np.float64, or int.")
                                 
            if not isinstance(param_name_or_list, list):
                param_name_or_list = [param_name_or_list]

            for param_name in param_name_or_list:
                kind, resolved = self._resolve_name(param_name)
                if kind == "state":
                    self.data.states()[resolved] = param_vals[JJ]
                elif kind == "const":
                    self.data.constants()[resolved] = param_vals[JJ]
                    # A constant following the "<state>_init" convention initialises a state;
                    # keep that state in step so setting it actually moves the trajectory.
                    if change_states:
                        self._sync_state_for_init_const(param_name, param_vals[JJ])
                else:
                    raise ValueError(
                        f"parameter name {param_name} not found in OpenCOR states/constants"
                    )

    def _sync_state_for_init_const(self, param_name, val):
        """If ``param_name`` is a ``<state>_init`` constant, write ``val`` to that state too.

        Mirrors the python/CasADi/AADC backends, which already do this. Silently does nothing
        when the name does not follow the convention or the state cannot be resolved -- the
        constant has still been set either way.
        """
        var_part = param_name.split("/")[-1] if "/" in param_name else param_name
        if not var_part.endswith("_init"):
            return
        state_var = var_part[:-len("_init")]
        for candidate in (state_var, param_name.rsplit("/", 1)[0] + "/" + state_var
                          if "/" in param_name else state_var):
            try:
                kind, resolved = self._resolve_name(candidate)
            except Exception:
                continue
            if kind == "state":
                self.data.states()[resolved] = val
                return

    def modify_params_and_run_and_get_results(self, param_names, mod_factors, obs_names, absolute=False):

        if absolute:
            new_param_vals = mod_factors
        else:
            init_param_vals = self.get_init_param_vals(param_names)
            new_param_vals = [a*b for a, b in zip(init_param_vals, mod_factors)]

        print(new_param_vals)
        self.set_param_vals(param_names, new_param_vals)

        success = self.run()
        print('here')
        if success:
            pred_obs_new = self.get_results(obs_names)
            # reset params
            self.reset_and_clear()

        else:
            print('simulation failed ')
            exit()

        return pred_obs_new

    def update_times(self, dt, start_time, sim_time, pre_time):
        self.dt = dt
        self.stop_time = start_time + pre_time + sim_time  # full time of simulation
        self.pre_steps = int(pre_time/self.dt)  # number of steps to do before storing data (used to reach steady state)
        self.n_steps = int(sim_time/self.dt)  # number of steps for storing data
        self.data.set_starting_point(start_time)
        self.data.set_ending_point(self.stop_time)
        self.tSim = np.linspace(start_time + pre_time, self.stop_time, self.n_steps + 1)  # time values for stored part of simulation

    def close_simulation(self):
        oc.close_simulation(self.simulation)

