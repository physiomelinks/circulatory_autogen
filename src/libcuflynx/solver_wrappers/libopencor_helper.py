"""The ``CVODE_libopencor`` backend: CellML models integrated by libOpenCOR's CVODE.

libOpenCOR (https://opencor.ws/libopencor/) is OpenCOR's simulation engine as a library,
installed with ``pip install "libcuflynx[libopencor]"`` (its wheels need Python 3.12 or later).
It compiles the model once per helper and runs it as a SED-ML uniform time course:

- the model is flattened with libCellML first, as the Myokit backend does, so both backends see
  the same ``component/variable`` names;
- parameter values and state values are applied as SED-ML attribute changes on the model, which
  libOpenCOR applies when it instantiates a run, recomputing everything that depends on them
  (a state whose initial value references a ``*_init`` constant follows that constant);
- ``pre_time`` is the gap between the time course's initial time and its output start time, so
  the warm-up is integrated without being stored;
- a run continues from the state the previous run ended in, until ``reset_states`` or
  ``reset_and_clear``, so multi-sub-experiment protocols are continuous.

Not supported, unlike ``CVODE_myokit``: time-varying inputs (``protocol_traces`` /
``protocol_shapes``) and CVODES forward sensitivities (calibrations use finite differences).
"""
import itertools
import os
import traceback
import warnings

import numpy as np

try:
    import libopencor as oc
except ImportError as _libopencor_exc:
    oc = None
    #: Formatted `ImportError` from the attempted `import libopencor`, or None when it succeeded.
    LIBOPENCOR_IMPORT_ERROR = ''.join(
        traceback.format_exception_only(type(_libopencor_exc), _libopencor_exc)).strip()
else:
    LIBOPENCOR_IMPORT_ERROR = None

import libcuflynx.utilities.libcellml_helper_funcs as cellml
from libcuflynx.solver_wrappers.name_resolver import VariableNameResolver
from libcuflynx.solver_wrappers.experiment_time import ExperimentTimeMixin

SOLVER_NAME = 'CVODE_libopencor'

# solver_info key -> libOpenCOR SolverCvode attribute.
_SOLVER_INFO_TO_CVODE = {
    'MaximumStep': 'maximum_step',
    'MaximumNumberOfSteps': 'maximum_number_of_steps',
    'rtol': 'relative_tolerance',
    'atol': 'absolute_tolerance',
}
# Mirrors SOLVER_INFO_FIELDS['CVODE_libopencor'] (a test keeps them equal).
DEFAULT_SOLVER_INFO = {'MaximumStep': 0.001, 'MaximumNumberOfSteps': 5000,
                       'rtol': 1e-8, 'atol': 1e-8}
# Keys every backend accepts and this one has no use for.
_FRAMEWORK_KEYS = {'method', 'solver', 'dt_solver'}

_file_counter = itertools.count()


class LibOpenCORUnavailableError(RuntimeError):
    """`CVODE_libopencor` was requested but `libopencor` could not be imported."""


def is_libopencor_available():
    """True when `libopencor` imported, i.e. `CVODE_libopencor` can run."""
    return oc is not None


def libopencor_unavailable_message(solver=SOLVER_NAME):
    """The message a user gets when they ask for libOpenCOR in an environment without it."""
    return (
        f"The {solver} solver needs libOpenCOR, which this Python cannot import "
        f"({LIBOPENCOR_IMPORT_ERROR or 'module not found'}).\n"
        '  - Install it with `pip install "libcuflynx[libopencor]"` (libOpenCOR\'s wheels need '
        'Python 3.12 or later).\n'
        "  - Or use `solver: CVODE_myokit`, which runs the same CellML model with CVODE and is "
        "installed with libcuflynx."
    )


def require_libopencor(solver=SOLVER_NAME):
    """Raise `LibOpenCORUnavailableError` unless `libopencor` is importable."""
    if oc is None:
        raise LibOpenCORUnavailableError(libopencor_unavailable_message(solver))


def _issue_text(item):
    return '; '.join(issue.description for issue in item.issues)


def _split_name(name):
    component, _, variable = name.rpartition('/')
    return component, variable


class SimulationHelper(ExperimentTimeMixin):
    """libOpenCOR backend with the common SimulationHelper surface (see python_solver_helper)."""

    def __init__(self, cellml_path, dt, sim_time, solver_info=None, pre_time=0.0):
        require_libopencor()
        self.cellml_path = cellml_path
        self.protocol_info = None

        self._file = oc.File(
            f'cuflynx-{os.getpid()}-{next(_file_counter)}-{os.path.basename(cellml_path)}', False)
        self._file.contents = list(self._flattened_model(cellml_path).encode('utf-8'))
        if self._file.has_errors:
            raise ValueError(f'libOpenCOR cannot read {cellml_path}: {_issue_text(self._file)}')
        self._document = oc.SedDocument(self._file)
        if self._document.has_errors:
            raise ValueError(f'libOpenCOR cannot simulate {cellml_path}: '
                             f'{_issue_text(self._document)}')
        self._simulation = self._document.simulation(0)
        if not isinstance(self._simulation, oc.SedUniformTimeCourse):
            raise ValueError(f'{cellml_path} has no differential equations, so it has no time '
                             f'course for {SOLVER_NAME} to run.')
        self._model = self._document.model(0)
        self.solver_info = self._apply_solver_info(solver_info)

        # Values applied as SED-ML changes on each run: constants, and states set directly.
        self._constant_values = {}
        self._state_values = {}
        # The state the next run starts from when no reset came in between.
        self._carried_state = {}
        # The state an offline warm-up ended in, which reset_states() returns to.
        self._offline_state = {}
        self._task = None
        self._instance = None
        self._has_run = False
        self._last_results_dict = None
        self._initial_values = None

        names = self._read_variable_names()
        self._states, self._constants, self._computed_constants, self._algebraics = names
        self._state_set, self._constant_set, self._computed_set, self._algebraic_set = (
            set(group) for group in names)
        self._index = {name: i for group in names for i, name in enumerate(group)}

        self.dt = dt
        if sim_time is not None and pre_time is not None:
            self.update_times(dt, 0.0, sim_time, pre_time)
        else:
            self.stop_time = self.pre_steps = self.n_steps = None
            self.tSim = None

    # --- setup ---------------------------------------------------------------------------

    @staticmethod
    def _flattened_model(path):
        """The model at ``path`` with its imports resolved and flattened, as CellML text."""
        model = cellml.parse_model(path, False)
        importer = cellml.resolve_imports(model, os.path.dirname(os.path.abspath(path)), False)
        return cellml.print_model(cellml.flatten_model(model, importer))

    def _apply_solver_info(self, solver_info):
        merged = dict(DEFAULT_SOLVER_INFO)
        unknown = []
        for key, value in (solver_info or {}).items():
            if key in _FRAMEWORK_KEYS:
                continue
            if key not in _SOLVER_INFO_TO_CVODE:
                unknown.append(key)
                continue
            if value is not None:
                merged[key] = value
        if unknown:
            # validate_solver_info rejects these in a user_inputs file; a caller of the helper
            # itself (a shared solver_info dict) is told and carries on.
            warnings.warn(f'solver_info key(s) {sorted(unknown)} are not settings of '
                          f'{SOLVER_NAME} and are ignored; it takes '
                          f'{sorted(_SOLVER_INFO_TO_CVODE)}.')
        solver = oc.SolverCvode()
        for key, attribute in _SOLVER_INFO_TO_CVODE.items():
            value = merged[key]
            setattr(solver, attribute,
                    int(value) if key == 'MaximumNumberOfSteps' else float(value))
        self._simulation.ode_solver = solver
        self._solver = solver
        return merged

    def _read_variable_names(self):
        instance = self._document.instantiate()
        if instance.has_errors:
            raise ValueError(f'libOpenCOR cannot simulate {self.cellml_path}: '
                             f'{_issue_text(instance)}')
        task = instance.task(0)
        return ([task.state_name(i) for i in range(task.state_count)],
                [task.constant_name(i) for i in range(task.constant_count)],
                [task.computed_constant_name(i) for i in range(task.computed_constant_count)],
                [task.algebraic_variable_name(i) for i in range(task.algebraic_variable_count)])

    def set_protocol_info(self, protocol_info):
        """Store the protocol. Time-varying inputs are refused: libOpenCOR has none."""
        traced = sorted(name for name, values in
                        ((protocol_info or {}).get('params_to_change') or {}).items()
                        if any(isinstance(v, str) for row in values
                               for v in (row if isinstance(row, list) else [row])))
        if traced:
            raise NotImplementedError(
                f'{SOLVER_NAME} has no time-varying inputs, so it cannot run the protocol '
                f'traces set on {traced}. Use solver CVODE_myokit for this obs_data.')
        self.protocol_info = protocol_info

    # --- names ---------------------------------------------------------------------------

    def _resolve_name(self, name):
        """(kind, name) with kind 'state', 'const', 'computed' or 'alg', or (None, None)."""
        return VariableNameResolver.resolve_key(
            name,
            [('state', self._state_set), ('const', self._constant_set),
             ('computed', self._computed_set), ('alg', self._algebraic_set)],
            separator='/',
        )

    def get_all_variable_names(self):
        return (list(self._states) + list(self._algebraics) + list(self._constants)
                + list(self._computed_constants))

    # --- running -------------------------------------------------------------------------

    def update_times(self, dt, start_time, sim_time, pre_time):
        self.dt = dt
        self.stop_time = start_time + pre_time + sim_time
        self.pre_steps = int(pre_time / dt)
        self.n_steps = int(sim_time / dt)
        self._track_experiment_start(start_time, pre_time)
        self.tSim = np.linspace(start_time + pre_time, self.stop_time, self.n_steps + 1)
        self._time_course = (float(start_time), float(start_time + pre_time),
                             float(self.stop_time), max(self.n_steps, 1))

    def _start_state(self):
        """The states the next run starts from: the offline warm-up's, overridden by states set
        with set_param_vals, overridden by where the previous run ended (set_param_vals drops a
        state it sets from that)."""
        state = dict(self._offline_state)
        state.update(self._state_values)
        state.update(self._carried_state)
        return state

    def _instantiate(self, time_course, state):
        initial, output_start, output_end, steps = time_course
        self._simulation.initial_time = initial
        self._simulation.output_start_time = output_start
        self._simulation.output_end_time = output_end
        self._simulation.number_of_steps = steps
        self._model.remove_all_changes()
        for name, value in itertools.chain(self._constant_values.items(), state.items()):
            component, variable = _split_name(name)
            self._model.add_change(oc.SedChangeAttribute(component, variable, repr(float(value))))
        return self._document.instantiate()

    def _run_course(self, time_course, state):
        """Run one time course; return (instance, task) or None when the run failed."""
        instance = self._instantiate(time_course, state)
        if instance.has_errors:
            print(f'{SOLVER_NAME}: the model could not be instantiated: {_issue_text(instance)}')
            return None
        instance.run()
        if instance.has_errors:
            print(f'{SOLVER_NAME}: failed to converge: {_issue_text(instance)}')
            return None
        return instance, instance.task(0)

    def run(self):
        outcome = self._run_course(self._time_course, self._start_state())
        if outcome is None:
            self.reset_and_clear()
            return False
        self._instance, self._task = outcome
        self._has_run = True
        self._carried_state = {name: float(self._task.state(i)[-1])
                               for i, name in enumerate(self._states)}
        self._initial_values = None
        return True

    def run_offline_pre_and_set_default_state(self, offline_pre_time):
        """Run an unlogged warm-up once; reset_states() returns to the state it ends in."""
        offline_pre_time = float(offline_pre_time)
        if offline_pre_time <= 0:
            return
        outcome = self._run_course((0.0, 0.0, offline_pre_time, 1), self._start_state())
        if outcome is None:
            raise RuntimeError('Offline pre-time simulation failed')
        _, task = outcome
        self._offline_state = {name: float(task.state(i)[-1])
                               for i, name in enumerate(self._states)}
        self._carried_state = {}
        self._initial_values = None
        self._has_run = False

    def reset_states(self):
        """Start the next run from the initial (or offline warm-up) state."""
        if self._has_run:
            self._last_results_dict = self._collect_all_results_dict()
        self._carried_state = {}
        self._initial_values = None
        self._has_run = False

    def reset_and_clear(self, only_one_exp=-1):
        """As reset_states, and also drop states set with set_param_vals."""
        self.reset_states()
        self._state_values = {}
        self._initial_values = None
        self._task = self._instance = None

    def close_simulation(self):
        self._task = self._instance = None
        manager = oc.FileManager.instance()
        manager.unmanage(self._file)

    # --- results -------------------------------------------------------------------------

    def _series(self, kind, name):
        task, index = self._task, self._index[name]
        if kind == 'state':
            return np.array(task.state(index))
        if kind == 'alg':
            return np.array(task.algebraic_variable(index))
        if kind == 'const':
            return np.array(task.constant(index))
        return np.array(task.computed_constant(index))

    def get_results(self, variables_list_of_lists, flatten=False):
        if not isinstance(variables_list_of_lists[0], list):
            variables_list_of_lists = [[entry] for entry in variables_list_of_lists]
        if self._task is None:
            raise RuntimeError('Simulation has not been run yet.')
        results = []
        for variables_list in variables_list_of_lists:
            row = []
            for variable_name in variables_list:
                if variable_name == 'time':
                    row.append(self.get_time())
                    continue
                kind, resolved = self._resolve_name(variable_name)
                if kind is None:
                    raise ValueError(f'{variable_name} is not a variable of {self.cellml_path}. '
                                     f'Its variables are {self.get_all_variable_names()}.')
                row.append(self._series(kind, resolved))
            results.append(row)
        if flatten:
            results = [item for sublist in results for item in sublist]
        return results

    def get_all_results(self, flatten=False):
        return self.get_results(self.get_all_variable_names(), flatten=flatten)

    def get_all_results_dict(self):
        if self._has_run:
            self._last_results_dict = self._collect_all_results_dict()
        if self._last_results_dict is None:
            raise RuntimeError('Simulation has not been run yet.')
        return {name: np.asarray(val).copy() for name, val in self._last_results_dict.items()}

    def _collect_all_results_dict(self):
        names = self.get_all_variable_names()
        values = self.get_results(names, flatten=True)
        return {name: np.asarray(val) for name, val in zip(names, values)}

    # --- parameters ----------------------------------------------------------------------

    def _current_initial_values(self):
        """Every state and constant at the start of the next run, by name."""
        if self._initial_values is None:
            outcome = self._run_course((0.0, 0.0, 1e-12, 1), self._start_state())
            if outcome is None:
                raise RuntimeError(f'{SOLVER_NAME}: could not evaluate the initial values of '
                                   f'{self.cellml_path}.')
            _, task = outcome
            values = {name: float(task.state(i)[0]) for i, name in enumerate(self._states)}
            values.update({name: float(task.constant(i)[0])
                           for i, name in enumerate(self._constants)})
            values.update({name: float(task.computed_constant(i)[0])
                           for i, name in enumerate(self._computed_constants)})
            self._initial_values = values
        return self._initial_values

    def get_init_param_vals(self, param_names):
        values = self._current_initial_values()
        param_init = []
        for param_name_or_list in param_names:
            if not isinstance(param_name_or_list, list):
                param_name_or_list = [param_name_or_list]
            row = []
            for param_name in param_name_or_list:
                kind, resolved = self._resolve_name(param_name)
                if kind not in ('state', 'const', 'computed'):
                    raise ValueError(f'parameter name {param_name} is not a state or constant '
                                     f'of {self.cellml_path}')
                row.append(values[resolved])
            param_init.append(row)
        return param_init

    def set_param_vals(self, param_names, param_vals, change_states=True):
        """Set constants and state values for the next run.

        A constant named ``<state>_init`` also sets that state (when ``change_states``), as the
        other backends do. ``change_states=False`` is for mid-protocol updates that keep the
        state the previous sub-experiment ended in; naming a state there is an error.
        """
        updates = []
        for param_name_or_list, value in zip(param_names, param_vals):
            if isinstance(value, str):
                raise NotImplementedError(
                    f'{SOLVER_NAME} has no time-varying inputs, so it cannot set '
                    f'{param_name_or_list} to the protocol trace {value!r}. '
                    'Use solver CVODE_myokit for protocol traces.')
            if not isinstance(value, (float, int, np.floating, np.integer)):
                raise ValueError(f'Parameter value {value} ({type(value)}) must be a float, '
                                 'np.float64 or int.')
            names = (param_name_or_list if isinstance(param_name_or_list, list)
                     else [param_name_or_list])
            for param_name in names:
                kind, resolved = self._resolve_name(param_name)
                if kind not in ('state', 'const'):
                    raise ValueError(f'parameter name {param_name} is not a state or constant '
                                     f'of {self.cellml_path}')
                if kind == 'state' and not change_states:
                    raise ValueError(
                        'set_param_vals(change_states=False) cannot set states directly, but was '
                        f'given: {param_name}. change_states=False exists for mid-protocol '
                        'updates that must preserve the evolved state.')
                updates.append((kind, resolved, float(value)))

        for kind, resolved, value in updates:
            if kind == 'state':
                self._state_values[resolved] = value
                self._carried_state.pop(resolved, None)
                continue
            self._constant_values[resolved] = value
            if change_states:
                state = self._state_for_init_constant(resolved)
                if state is not None:
                    self._state_values[state] = value
                    self._carried_state.pop(state, None)
        self._initial_values = None

    def _state_for_init_constant(self, name):
        """The state ``<component>/<x>`` that a ``<component>/<x>_init`` constant initialises."""
        component, variable = _split_name(name)
        if not variable.endswith('_init'):
            return None
        state_var = variable[:-len('_init')]
        for candidate in (f'{component}/{state_var}' if component else state_var, state_var):
            kind, resolved = self._resolve_name(candidate)
            if kind == 'state':
                return resolved
        return None

    def modify_params_and_run_and_get_results(self, param_names, mod_factors, obs_names,
                                              absolute=False):
        if absolute:
            new_param_vals = mod_factors
        else:
            init_param_vals = self.get_init_param_vals(param_names)
            new_param_vals = [a * b for a, b in zip(init_param_vals, mod_factors)]
        self.set_param_vals(param_names, new_param_vals)
        if not self.run():
            raise RuntimeError(f'{SOLVER_NAME}: simulation failed')
        pred_obs_new = self.get_results(obs_names)
        self.reset_and_clear()
        return pred_obs_new
