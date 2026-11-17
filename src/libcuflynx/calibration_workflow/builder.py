'''
Building a calibration workflow in Python: :class:`WorkflowBuilder` writes the same
``calibration_workflow.json`` a user would write by hand, so a notebook, the
``cuflynx-calibration-workflow`` command and CUFLynx all run one file.

::

    from libcuflynx.calibration_workflow import WorkflowBuilder, library_module

    wf = WorkflowBuilder('chain', target=library_module('chain', 'v1', 'rest'),
                         directory='workflows/chain', module_library_dirs=['modules'],
                         settings={'dt': 0.1, 'param_id_method': 'CMA-ES'})
    wf.add_operation(my_feature)              # a function defined in a notebook cell
    wf.add_calibration('fit', obs_data=obs_data_dict, params_for_id=params_for_id_entries)
    wf.run()                                  # writes the files, then runs what is not current
    wf.result('fit')['values']

The builder edits the workflow's raw dict, whose keys are the schema's
(``schemas/calibration_workflow.schema.json``): each ``add_*`` takes the step's schema keys as
keyword arguments (and the options its kind keeps in ``settings`` as an argument of their own)
and checks the step against the schema at once; :meth:`~WorkflowBuilder.check` runs every check
of the file and names what a run still needs. :meth:`~WorkflowBuilder.add_step` takes any step
as raw schema keys.

What :meth:`~WorkflowBuilder.write` puts in ``directory``::

    calibration_workflow.json         the workflow, its paths relative to this directory
    data/<step>_obs_data.json         an obs_data given as a dict (or an ObsDataCreator)
    data/<step>_params_for_id.json    a params_for_id given as entries or a document
    <name>_operation_funcs.py         the functions given to add_operation (and _cost_funcs.py,
                                      _modifier_funcs.py for add_cost_func, add_modifier_func)

Dicts are copied when they are added: to change a step, add it again with the same id. A file
is rewritten only when its content changes, so the runner's change detection (the sha256 of
each input file) finds exactly the steps whose data or functions changed, and
:meth:`~WorkflowBuilder.run` runs those and the steps after them, keeping the rest. A functions
file carries the checksum of what was generated (``utilities.user_funcs_file``): the builder
regenerates one nobody edited, but refuses to replace one edited by hand, or one holding
functions the builder does not have (added by another writer), unless told to
(``overwrite=True``).
'''

import ast
import builtins
import copy
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
import textwrap
import time

from libcuflynx.calibration_workflow.spec import (ALLOWED_SETTINGS, PATH_SETTINGS,
                                                  WORKFLOW_FILE_NAME, WorkflowError,
                                                  parse_workflow)
from libcuflynx.utilities.user_funcs_file import (FUNCS_SETTINGS, HEADER_NAMES,
                                                  funcs_file_source, is_unedited,
                                                  read_funcs_file, unkept_statements)

#: where a dict input is written, relative to the workflow's directory
DATA_DIR = 'data'
#: the marker each decorator sets on a function: (attribute, decorator), by kind of file. A
#: marker set after the def (``f.series_to_constant = True``) is written as the decorator.
_MARKS = {'operation': [('series_to_constant', 'series_to_constant'),
                        ('_circulatory_differentiable', 'differentiable')],
          'cost': [('is_MLE', 'is_MLE'), ('cost_combiner', 'cost_combiner'),
                   ('_circulatory_differentiable', 'differentiable')],
          'modifier': []}

#: the environment variables an MPI launcher sets for one job (by prefix): run(ranks=N) leaves
#: them out of the environment it gives its own mpiexec. A user's settings (OMPI_MCA_*,
#: OMPI_ALLOW_*, PRTE_MCA_*, HYDRA_*, I_MPI_*) are not among them.
_JOB_ENV_PREFIXES = (
    'PMIX_', 'PMI_', 'HYDI_', 'OMPI_COMM_WORLD_', 'OMPI_APP_CTX_', 'OMPI_MCA_orte_',
    'OMPI_MCA_ess', 'OMPI_MCA_pmix', 'OMPI_MCA_shmem_RUNTIME_QUERY_hint',
    'OMPI_MCA_initial_wdir', 'OMPI_UNIVERSE_SIZE',
    'OMPI_NUM_APP_CTX', 'OMPI_FIRST_RANKS', 'OMPI_ARGV', 'OMPI_COMMAND', 'OMPI_FILE_LOCATION',
    'OPAL_OUTPUT_STDERR_FD', 'PRTE_LAUNCHED', 'MPI_LOCALNRANKS', 'MPI_LOCALRANKID')
#: of those, what a user sets and a launcher only passes on
_USER_ENV = ('OMPI_MCA_orte_tmpdir_base', 'OMPI_MCA_orte_default_hostfile',
             'OMPI_MCA_orte_rsh_agent')
#: the step ids a run's output directory uses for itself (runner.MODELS_DIR, TARGET_VIEW,
#: TARGET_DIR, cuflynx_study.INPUTS_DIR); the schema refuses them too
_RESERVED_IDS = frozenset({'models', 'target', '_target', 'inputs'})
#: the kinds of step that name no model (spec._NO_MODEL_KINDS)
_NO_MODEL_KINDS = ('reduction', 'set')
#: the calibration method of a step whose settings name none (the parser's default)
DEFAULT_PARAM_ID_METHOD = 'genetic_algorithm'
#: the longest step id: it names the step's directory and files
MAX_ID_LENGTH = 80
# the order a built step's keys are written in (the schema's)
_STEP_KEYS = ('id', 'kind', 'description', 'comment', 'model', 'obs_data', 'params_for_id',
              'submodule', 'vessel', 'vessel_map', 'fixed_from', 'priors_from',
              'store_distribution', 'on_target', 'settings', 'emulator_from', 'emulator_dir',
              'result_from', 'start_from', 'sensitivity_from', 'params_for_id_from', 'set',
              'point', 'sweep')
#: the keys of a model that are paths, relative to the workflow file when written
_MODEL_PATH_KEYS = ('module_array', 'parameters', 'external_model_path', 'cellml_model_path')
#: the keys of a step that are paths, relative to the workflow file when written
_STEP_PATH_KEYS = ('obs_data', 'params_for_id', 'emulator_dir')


# --------------------------------------------------------------------------------------------
# schema fragments
# --------------------------------------------------------------------------------------------

def library_module(module_type, version, parameterisation):
    '''A model that is a module of a module library, for ``target=`` or a step's ``model=``.

    Args:
        module_type: the module's type (its directory in the library).
        version: the version of that module.
        parameterisation: the parameterisation of that version.

    Returns:
        ``{'module_type', 'version', 'parameterisation'}``, the schema's form.
    '''
    return {'module_type': module_type, 'version': version, 'parameterisation': parameterisation}


def module_array(path, parameters=None):
    '''A model that is a module array file, for ``target=`` or a step's ``model=``.

    Args:
        path: the ``<prefix>_module_array.json`` (or ``.csv``) file.
        parameters: its parameters file, when that is not ``<prefix>_parameters.json``
            beside it.

    Returns:
        the path, or ``{'module_array', 'parameters'}`` with ``parameters``.
    '''
    if parameters is None:
        return os.fspath(path)
    return {'module_array': os.fspath(path), 'parameters': os.fspath(parameters)}


def external_model(path):
    '''A model that is an external Python model, for ``target=``. A workflow with one has it
    as its target and as every step's model.

    Args:
        path: the ``.py`` file defining ``SIM_HELPER``.

    Returns:
        ``{'external_model_path': path}``, the schema's form.
    '''
    return {'external_model_path': os.fspath(path)}


def cellml_model(path, parameters=None):
    '''A model that is a CellML file, for ``target=``. A workflow with one has it as its
    target and as every step's model.

    Args:
        path: the CellML file.
        parameters: a parameters file whose values replace the CellML's.

    Returns:
        ``{'cellml_model_path': path[, 'parameters']}``, the schema's form.
    '''
    model = {'cellml_model_path': os.fspath(path)}
    if parameters is not None:
        model['parameters'] = os.fspath(parameters)
    return model


def selection(*instances):
    '''A step model that is a selection of the target's instances.

    Args:
        *instances: record names of the target's module array, or paths into a super
            instance such as ``heart/lv`` (or one list of them).

    Returns:
        ``{'select': [...]}``, the schema's form.
    '''
    if len(instances) == 1 and isinstance(instances[0], (list, tuple)):
        instances = instances[0]
    return {'select': list(instances)}


def prior(step, kind=None, inflate=None, parameters=None):
    '''A ``priors_from`` entry: the stored distribution of an earlier step as a prior.

    Args:
        step: the step, which sets ``store_distribution``.
        kind: how the stored samples are represented: ``'mvnormal'`` (the default),
            ``'normal'`` or ``'kde'``.
        inflate: multiplies the distribution's variance (default 1).
        parameters: only these parameters (model names in the source step).

    Returns:
        the step id, or ``{'step', ...}`` with the options given.
    '''
    if kind is None and inflate is None and parameters is None:
        return step
    out = {'step': step}
    if kind is not None:
        out['kind'] = kind
    if inflate is not None:
        out['inflate'] = inflate
    if parameters is not None:
        out['parameters'] = list(parameters)
    return out


def set_value(parameter, value, *, reason, units=None):
    '''An entry of :meth:`WorkflowBuilder.add_set`: a value set by hand.

    Args:
        parameter: ``<vessel>/<parameter>`` in the target.
        value: the value.
        reason: why it is set by hand (required; shown in the provenance).
        units: the value's units; by default those the target gives it.

    Returns:
        the entry, ``{'parameter', 'value', 'reason'[, 'units']}``.
    '''
    entry = {'parameter': parameter, 'value': value, 'reason': reason}
    if units is not None:
        entry['units'] = units
    return entry


def scale_value(parameter, scale, *, of, reason, units=None):
    '''An entry of :meth:`WorkflowBuilder.add_set`: a value scaled by hand.

    Args:
        parameter: ``<vessel>/<parameter>`` in the target.
        scale: the factor.
        of: the step whose value of the parameter is scaled, or ``"inherited"`` for the
            target's own value.
        reason: why (required; shown in the provenance).
        units: the value's units; by default those the target or the ``of`` step gives it.

    Returns:
        the entry, ``{'parameter', 'scale', 'of', 'reason'[, 'units']}``.
    '''
    entry = {'parameter': parameter, 'scale': scale, 'of': of, 'reason': reason}
    if units is not None:
        entry['units'] = units
    return entry


def launcher_environment(environ):
    '''``environ`` without the variables an MPI launcher set for the job this process is in
    (it may be under mpiexec, or have initialised MPI as a singleton), which a new mpiexec must
    not inherit. The user's MPI settings stay.'''
    return {k: v for k, v in environ.items()
            if k in _USER_ENV or not k.startswith(_JOB_ENV_PREFIXES)}


def jsonable(value):
    '''``value`` with numpy and pandas scalars and arrays as Python numbers and lists, tuples
    as lists and paths as strings, so it is written as JSON.'''
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if type(value).__module__.split('.')[0] in ('numpy', 'pandas') and hasattr(value, 'tolist'):
        return jsonable(value.tolist())
    return value


def _json_text(doc, what):
    '''``doc`` as JSON text. A WorkflowError for a NaN or infinity, which JSON does not
    have.'''
    try:
        return json.dumps(doc, indent=1, allow_nan=False) + '\n'
    except ValueError as exc:
        raise WorkflowError(
            f'{what} holds {", ".join(_not_finite(doc)) or "a NaN or infinity"}, which a JSON '
            f'file cannot hold: give a number, or leave the key out.') from exc


def _not_finite(value, where=''):
    '''Where the NaNs and infinities of a JSON-like value are: ``["a.b[0] (nan)", ...]``.'''
    if isinstance(value, dict):
        return [x for k, v in value.items() for x in _not_finite(v, f'{where}.{k}'.lstrip('.'))]
    if isinstance(value, list):
        return [x for i, v in enumerate(value) for x in _not_finite(v, f'{where}[{i}]')]
    if isinstance(value, float) and (value != value or value in (float('inf'), float('-inf'))):
        return [f'{where} ({value})']
    return []


# --------------------------------------------------------------------------------------------
# the builder
# --------------------------------------------------------------------------------------------

class _Inline:
    '''An obs_data or params_for_id given as content; written to ``data/`` by ``write()``.'''

    def __init__(self, key, text):
        self.key = key
        self.text = text


class WorkflowBuilder:
    '''A calibration workflow made by API calls, written as an ordinary
    ``calibration_workflow.json`` and run from Python.

    Creating a builder writes nothing: files are written by :meth:`write` (which
    :meth:`run`, :meth:`plan`, :meth:`status`, :meth:`model` and :meth:`plot_step` call).

    Args:
        name: the workflow's name (``workflow_name``). It names the functions files and the
            run's default output directory.
        target: the model the calibrated values are merged into, and the model of each step
            that names none: a module array path, :func:`module_array`,
            :func:`library_module`, :func:`external_model` or :func:`cellml_model`.
        directory: where the workflow file, its ``data/`` and its functions files are
            written; created by :meth:`write`. Relative paths given to the builder are
            relative to the working directory at the time of the call.
        description: the workflow's description.
        module_library_dirs: the module libraries (directories) the models' modules come
            from. Needed whenever a model names a library module, a module array's too.
        settings: the workflow's settings, ``user_inputs.yaml`` keys
            (``spec.ALLOWED_SETTINGS``: ``param_id_method``, ``optimiser_options``, ``dt``,
            ...), checked at once. A step's own ``settings`` are merged over them.

    Raises:
        WorkflowError: an unknown setting, or a name or target the schema refuses.

    Attributes:
        directory (str): the absolute directory the workflow is written to.
        path (str): the workflow file, ``<directory>/calibration_workflow.json``.
        settings (dict): the workflow's settings, paths absolute. Change them with
            :meth:`update_settings`, which checks them.
    '''

    def __init__(self, name, target, *, directory, description=None, module_library_dirs=(),
                 settings=None):
        self.directory = os.path.abspath(os.path.expanduser(os.fspath(directory)))
        self.path = os.path.join(self.directory, WORKFLOW_FILE_NAME)
        self._head = {'schema_version': 1, 'workflow_name': name}
        if description is not None:
            self._head['description'] = description
        self._head['target'] = self._abs_model(copy.deepcopy(target))
        if isinstance(module_library_dirs, (str, os.PathLike)):
            module_library_dirs = [module_library_dirs]
        self._head['module_library_dirs'] = [self._abs(d) for d in module_library_dirs]
        self.settings = {}
        self._steps = []
        self._order = None
        self._init_funcs()
        self._validate_head(self._abs_settings(copy.deepcopy(dict(settings or {}))))
        self.update_settings(**(settings or {}))

    def _init_funcs(self):
        self._funcs = {kind: {} for kind in FUNCS_SETTINGS}
        self._imports = {kind: [] for kind in FUNCS_SETTINGS}
        # the text of each functions file as load() read it, and what it held that a rewrite
        # would drop
        self._funcs_seen = {kind: None for kind in FUNCS_SETTINGS}
        self._funcs_unkept = {kind: [] for kind in FUNCS_SETTINGS}
        self._funcs_loaded = {kind: None for kind in FUNCS_SETTINGS}
        # the kinds whose file on disk may be replaced whatever it holds (overwrite=True)
        self._overwrite = {kind: False for kind in FUNCS_SETTINGS}

    @classmethod
    def load(cls, path):
        '''A builder for an existing workflow, written back to the same file.

        Args:
            path: the workflow file, or the directory holding its
                ``calibration_workflow.json``.

        The functions files a builder wrote beside it are read back (their functions and
        imports), so functions can be added, replaced or removed. Returns the builder.
        '''
        path = os.path.abspath(os.path.expanduser(os.fspath(path)))
        if os.path.isdir(path):
            path = os.path.join(path, WORKFLOW_FILE_NAME)
        if not os.path.isfile(path):
            raise WorkflowError(f'there is no workflow file {path}.')
        try:
            with open(path, encoding='utf-8-sig') as f:
                raw = json.load(f)
        except json.JSONDecodeError as exc:
            raise WorkflowError(f'{path} is not valid JSON: {exc}') from exc
        directory = os.path.dirname(path)
        self = cls.__new__(cls)
        self.directory, self.path = directory, path
        self._head, self._steps, self._order = {}, [], list(raw)
        self.settings = {}
        for key, value in raw.items():
            if key == 'steps':
                self._steps = [self._abs_step(s, directory) for s in value]
            elif key == 'settings':
                self.settings = self._abs_settings(value, directory)
            elif key == 'target':
                self._head[key] = self._abs_model(value, directory)
            elif key == 'module_library_dirs':
                self._head[key] = [self._abs(d, directory) for d in value]
            else:
                self._head[key] = value
        self._head.setdefault('module_library_dirs', [])
        self._init_funcs()
        for kind in FUNCS_SETTINGS:
            funcs_path = self._funcs_path(kind)
            read = read_funcs_file(funcs_path, kind) if os.path.isfile(funcs_path) else None
            if read is not None:
                self._funcs[kind], self._imports[kind] = read
                self._funcs_seen[kind] = self._funcs_on_disk(kind)
                self._funcs_unkept[kind] = unkept_statements(funcs_path, kind)
                self._funcs_loaded[kind] = (dict(read[0]), list(read[1]))
        return self

    def __repr__(self):
        from libcuflynx.calibration_workflow.runner import STEP_RESULT
        lines = [f'<WorkflowBuilder "{self.name}": target {_model_label(self._head["target"])}, '
                 f'{len(self._steps)} step{"" if len(self._steps) == 1 else "s"}, in '
                 f'{self.directory}>']
        for step in self._steps:
            state = 'not run'
            try:
                with open(os.path.join(self.output_dir, step['id'], STEP_RESULT)) as f:
                    if json.load(f).get('status') == 'done':
                        state = 'done'
            except (OSError, ValueError):
                pass
            lines.append(f'  {step["id"]} ({step.get("kind", "calibration")}): {state}')
        for kind in FUNCS_SETTINGS:
            if self._funcs[kind]:
                lines.append(f'  {kind} functions: {", ".join(self._funcs[kind])}')
        return '\n'.join(lines)

    @property
    def name(self):
        '''The workflow's name (``workflow_name``).'''
        return self._head['workflow_name']

    @property
    def target(self):
        '''The target model, as given (a copy; paths absolute).'''
        return copy.deepcopy(self._head['target'])

    @property
    def module_library_dirs(self):
        '''The module libraries the models come from (absolute paths).'''
        return list(self._head['module_library_dirs'])

    @property
    def output_dir(self):
        '''Where :meth:`run` writes unless told otherwise:
        ``<directory>/workflow_output/<name>`` (a str).'''
        return os.path.join(self.directory, 'workflow_output', self.name)

    @property
    def steps(self):
        '''The step ids (a list), in the order they were added.'''
        return [s['id'] for s in self._steps]

    def update_settings(self, **settings):
        '''Set workflow settings (the keys of ``spec.ALLOWED_SETTINGS``), checked against the
        schema at once. A dict setting is merged one level deep, as a step's are:
        ``update_settings(optimiser_options={'seed': 2})`` keeps the other optimiser options.
        A setting given as None is removed. Nothing is written until :meth:`write`.

        Raises:
            WorkflowError: an unknown setting, or one the schema refuses.
        '''
        merged = copy.deepcopy(self.settings)
        for key, value in self._abs_settings(copy.deepcopy(settings)).items():
            if value is None:
                merged.pop(key, None)
            elif isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        self._validate_head(merged)
        self._check_funcs_settings(merged)
        self.settings = merged

    def _validate_head(self, settings):
        '''Check the workflow's name, target and ``settings`` against the schema, as a
        workflow with one stand-in step.'''
        from libcuflynx.schemas import CALIBRATION_WORKFLOW_SCHEMA
        from libcuflynx.schemas.validation import SchemaValidationError, validate
        unknown = sorted(set(settings) - set(ALLOWED_SETTINGS))
        if unknown:
            raise WorkflowError(
                f'workflow "{self.name}": {unknown} {"is not a setting" if len(unknown) == 1 else "are not settings"} a '
                f'workflow takes. The settings are {", ".join(ALLOWED_SETTINGS)}; the model, '
                f'obs_data, params_for_id and output directories are chosen by the workflow.')
        raw = {'workflow_name': self._head['workflow_name'],
               'target': self._rel_model(self._head['target']),
               'steps': [{'id': 'step', 'kind': 'set',
                          'set': [{'parameter': 'a/b', 'value': 0, 'reason': 'stand-in'}]}]}
        if settings:
            raw['settings'] = self._rel_settings(settings)
        try:
            validate(CALIBRATION_WORKFLOW_SCHEMA, jsonable(raw),
                     source=f'workflow "{self._head["workflow_name"]}"',
                     what='calibration workflow')
        except SchemaValidationError as exc:
            raise WorkflowError(str(exc)) from exc

    # -- steps ----------------------------------------------------------------------------

    def add_calibration(self, id, *, obs_data, params_for_id=None, params_for_id_from=None,
                        model=None, fixed_from=(), priors_from=(), store_distribution=False,
                        start_from=None, emulator_from=None, emulator_dir=None, vessel=None,
                        submodule=None, vessel_map=None, settings=None, description=None,
                        comment=None):
        """Add a calibration: fit parameters of a model to an obs_data.

        Nothing is written or run by this call; the step is checked against the schema, and
        the dicts given are copied. A step with this ``id`` already is replaced in place, so
        to change a step, change what you pass and call this again.

        Args:
            id: the step's name: other steps refer to it and its results go to
                ``<output_dir>/<id>/``. No ``/`` or ``\\``; at most 80 characters; unlike
                every other id whatever the case.
            obs_data: what to fit: a path to an obs_data JSON file, an obs_data dict, or an
                ``ObsDataCreator``. A dict is written to ``data/<id>_obs_data.json`` by
                :meth:`write`.
            params_for_id: what to calibrate: a path to a params_for_id JSON file, a list of
                ``{vessel_name, param_name, min, max}`` entries (as ``CVS0DParamID`` takes),
                a list of ``{targets, min, max}`` entries, or a ``{"version", "params"}``
                document. Not a path: written to ``data/<id>_params_for_id.json``.
            params_for_id_from: instead of ``params_for_id``, the reduction step
                (:meth:`add_reduction`) whose chosen parameters are calibrated.
            model: the model the step runs on: a module array path, :func:`module_array`,
                :func:`library_module` or :func:`selection`. Default: the target.
            fixed_from: calibration or set step ids (one, or a list) whose values are fixed
                in this step's model; this step runs after them.
            priors_from: step ids or :func:`prior` entries whose stored distributions become
                this step's priors (those steps set ``store_distribution``).
            store_distribution: after the fit, sample the posterior by MCMC (needs the
                ``UQ_options`` setting) and store it for later steps.
            start_from: a calibration or set step at whose values this step's own
                parameters start.
            emulator_from: an emulation step (:meth:`add_emulation`) whose emulator this
                step is evaluated on; give the same model, obs_data and params_for_id.
            emulator_dir: instead of ``emulator_from``, the directory of an emulator trained
                outside the workflow.
            vessel: for a library module in a module-array target, the vessel it is (inferred
                when the module occurs once).
            submodule: for a library module in a library supermodule target, where it sits
                (inferred when the module occurs once).
            vessel_map: for a module-array model, ``{its vessel name: the target's}``.
            settings: this step's settings, merged over the workflow's one dict level deep.
            description: free text, shown in reports.
            comment: free text, kept in the file.

        Returns:
            ``id``.

        Raises:
            WorkflowError: the step does not fit the schema (the message names the key).
        """
        return self.add_step(id, **_given(locals()))

    def add_sensitivity(self, id, *, obs_data, params_for_id, sa_options, model=None,
                        fixed_from=(), result_from=None, emulator_from=None, emulator_dir=None,
                        vessel=None, submodule=None, vessel_map=None, settings=None,
                        description=None, comment=None):
        """Add a sensitivity analysis of ``params_for_id`` against ``obs_data``.

        Args:
            sa_options: the analysis options (the ``sa_options`` setting: ``method``,
                ``num_samples``, ``sample_type``, ...), written into the step's settings.
            result_from: a calibration step whose values the model is generated with, so
                the analysis is at that result.

        The other arguments, what is returned and when files are written are as for
        :meth:`add_calibration`.
        """
        given = _given(locals(), without=('sa_options',))
        given['settings'] = dict(settings or {}, sa_options=dict(sa_options))
        return self.add_step(id, kind='sensitivity', **given)

    def add_reduction(self, id, *, sensitivity_from, method='pca', n_params=None,
                      index_type=None, criterion=None, threshold=None, min_mode_std=None,
                      variance_fraction=None, min_index=None, redundancy_check=None,
                      description=None, comment=None):
        """Add a reduction: choose the parameters a later step calibrates
        (its ``params_for_id_from``) from the Sobol indices of a sensitivity step.

        Args:
            id: the step's name (see :meth:`add_calibration`).
            sensitivity_from: the sensitivity step whose results are reduced.
            method: ``'pca'`` (one parameter per informative mode of the Sobol outputs) or
                ``'sobol'`` (the ``n_params`` parameters with the highest index).
            n_params: how many parameters to keep (pca: at most; sobol: required).
            index_type: ``'ST'`` (total, the default) or ``'S1'`` (first-order) indices.
            criterion: sobol: a parameter's score, the ``'max'`` or ``'mean'`` of its index.
            threshold: sobol: drop parameters scoring below this.
            min_mode_std: pca: a mode is informative when its std over the samples is at
                least this many obs_data stds (default 1).
            variance_fraction: pca: only the leading modes explaining this fraction of the
                variance (default 1, no cut).
            min_index: pca: skip a mode whose largest index is below this (default 0.05).
            redundancy_check: the check for parameters with the same effect on the outputs.
            description, comment: free text.

        The options are written as the step's ``settings.param_reduction_options``; those
        left as None take the schema's defaults. Returns ``id``.
        """
        options = {k: v for k, v in dict(
            method=method, n_params=n_params, index_type=index_type, criterion=criterion,
            threshold=threshold, min_mode_std=min_mode_std, variance_fraction=variance_fraction,
            min_index=min_index, redundancy_check=redundancy_check).items() if v is not None}
        return self.add_step(id, kind='reduction', sensitivity_from=sensitivity_from,
                             settings={'param_reduction_options': options},
                             **_given({'description': description, 'comment': comment}))

    def add_emulation(self, id, *, obs_data, emulator_settings=None, params_for_id=None,
                      params_for_id_from=None, model=None, fixed_from=(), vessel=None,
                      submodule=None, vessel_map=None, settings=None, description=None,
                      comment=None):
        """Add an emulator training: a surrogate of the obs_data's features over the
        ``params_for_id`` ranges, for calibration and sensitivity steps that name this step as
        their ``emulator_from``.

        Args:
            emulator_settings: the training options (the ``emulator_settings`` setting),
                written into the step's settings. Default: the workflow's, else the
                engine's defaults.

        The other arguments, what is returned and when files are written are as for
        :meth:`add_calibration`.
        """
        given = _given(locals(), without=('emulator_settings',))
        if emulator_settings is not None:
            given['settings'] = dict(settings or {}, emulator_settings=dict(emulator_settings))
        return self.add_step(id, kind='emulation', **given)

    def add_identifiability(self, id, *, obs_data, params_for_id, ia_options, result_from=None,
                            model=None, fixed_from=(), vessel=None, submodule=None,
                            vessel_map=None, settings=None, description=None, comment=None):
        """Add an identifiability analysis of ``params_for_id`` at a step's values.

        Args:
            ia_options: the analysis options (the ``ia_options`` setting; ``method``,
                ``'Laplace'`` or ``'profile_likelihood'``, is
                required), written into the step's settings.
            result_from: the calibration or set step whose values the analysis is at.
                Default: the model's own values.

        The other arguments, what is returned and when files are written are as for
        :meth:`add_calibration`.
        """
        given = _given(locals(), without=('ia_options',))
        given['settings'] = dict(settings or {}, ia_options=dict(ia_options))
        return self.add_step(id, kind='identifiability', **given)

    def add_uq(self, id, *, obs_data, params_for_id, uq_options={}, result_from=None,
               model=None, fixed_from=(), emulator_from=None, emulator_dir=None, vessel=None,
               submodule=None, vessel_map=None, settings=None, description=None, comment=None):
        """Add an uncertainty quantification of ``params_for_id`` at a step's values, with
        no fit of its own. Its distribution is stored, for a later step's ``priors_from``, and
        checked by a posterior predictive check unless ``posterior_predictive`` is false.

        Args:
            uq_options: the options (the ``UQ_options`` setting; ``method`` is ``'mcmc'`` or
                ``'laplace'``), written into the step's settings (copied, never changed).
                Default ``{}``: the workflow's, else MCMC with the engine's defaults.
            result_from: the calibration or set step whose values it is at. Default: the
                model's own values.

        The other arguments, what is returned and when files are written are as for
        :meth:`add_calibration`.
        """
        given = _given(locals(), without=('uq_options',))
        given['settings'] = dict(settings or {}, UQ_options=dict(uq_options))
        return self.add_step(id, kind='uq', **given)

    def add_set(self, id, entries, *, point=False, description=None, comment=None):
        """Add a set step: values of the target set by hand, each with its reason (which
        goes into the provenance).

        Args:
            id: the step's name (see :meth:`add_calibration`).
            entries: a list of :func:`set_value` and :func:`scale_value` entries.
            point: the values are only a point for ``start_from``, ``result_from`` or
                ``fixed_from``: they are not merged into the target.
            description, comment: free text.

        Returns ``id``.
        """
        return self.add_step(id, kind='set', set=list(entries),
                             **_given({'point': point, 'description': description,
                                       'comment': comment}))

    def add_sweep(self, id, *, parameters, model=None, on_target=False, obs_data=None,
                  design=None, outputs=None, zip=None, fixed_from=(), result_from=None,
                  vessel=None, submodule=None, vessel_map=None, settings=None,
                  description=None, comment=None):
        """Add a parameter sweep: simulate a model over a grid of parameter values.

        Args:
            parameters: ``{name: [min, max] | {min, max, points, scale} | {values} |
                {factors}}``, the ranges swept.
            model: the model swept. Default: the target.
            on_target: sweep the target with every calibrated and set value merged, after
                every calibration and set step. Then give no ``model``, ``fixed_from`` or
                ``result_from``.
            obs_data: gives the protocol, and its prediction items a feature per point;
                without one, the settings give ``dt`` and ``sim_time``.
            design: how the points are laid out (the schema's ``sweep.design``).
            outputs: what each point records (the schema's ``sweep.outputs``).
            zip: parameters varied together rather than crossed.
            result_from: a calibration step whose values the model is generated with.

        The other arguments, what is returned and when files are written are as for
        :meth:`add_calibration`.
        """
        sweep = {k: v for k, v in (('design', design), ('outputs', outputs),
                                   ('parameters', parameters), ('zip', zip)) if v is not None}
        given = _given({k: v for k, v in locals().items()
                        if k not in ('design', 'outputs', 'parameters', 'zip', 'sweep')})
        return self.add_step(id, kind='sweep', sweep=sweep, **given)

    def add_step(self, id, kind='calibration', **raw):
        """Add a step of any kind from its raw schema keys (``model``, ``obs_data``,
        ``settings``, ...; see ``schemas/calibration_workflow.schema.json``). The ``add_*``
        methods call this.

        Args:
            id: the step's name (see :meth:`add_calibration`).
            kind: ``'calibration'``, ``'sensitivity'``, ``'reduction'``, ``'emulation'``,
                ``'identifiability'``, ``'uq'``, ``'set'`` or ``'sweep'``.
            **raw: the step's keys. ``obs_data`` and ``params_for_id`` take what
                :meth:`add_calibration` takes; ``model`` defaults to the target for the
                kinds that run a model (not for a sweep with ``on_target``); a key given
                as None is left out.

        A step with this id already is replaced in place. Returns ``id``.

        Raises:
            WorkflowError: the id cannot be used, or the step does not fit the schema.
        """
        self._check_id(id)
        step = {'id': id}
        if kind != 'calibration':
            step['kind'] = kind
        for key, value in raw.items():
            if value is not None:
                step[key] = copy.deepcopy(value)
        if 'model' not in step and kind not in _NO_MODEL_KINDS and not step.get('on_target'):
            step['model'] = copy.deepcopy(self._head['target'])
        if 'obs_data' in step:
            step['obs_data'] = self._obs_data(id, step['obs_data'])
        if 'params_for_id' in step:
            step['params_for_id'] = self._params_for_id(id, step['params_for_id'])
        if 'emulator_dir' in step:
            step['emulator_dir'] = self._abs(step['emulator_dir'])
        if 'model' in step:
            step['model'] = self._abs_model(step['model'])
        for key in ('fixed_from', 'priors_from'):
            if isinstance(step.get(key), (str, dict)):
                step[key] = [step[key]]
        if 'settings' in step and kind != 'reduction':
            step['settings'] = self._abs_settings(step['settings'])
        step = {k: step[k] for k in _STEP_KEYS if k in step} | \
            {k: v for k, v in step.items() if k not in _STEP_KEYS}
        self._validate_step(step)
        ids = self.steps
        steps = list(self._steps)
        if id in ids:
            steps[ids.index(id)] = step
        else:
            steps.append(step)
        self._check_funcs_settings(steps=steps)
        self._steps = steps
        return id

    def _check_id(self, id):
        """A WorkflowError for an id whose files could not be told from another step's on
        every file system, or whose name is too long for one."""
        if not isinstance(id, str) or not id.strip():
            raise WorkflowError(f'a step id is a non-empty string, not {id!r}.')
        if len(id) > MAX_ID_LENGTH:
            raise WorkflowError(
                f'step id {id[:30]!r}... has {len(id)} characters; at most {MAX_ID_LENGTH}, '
                f'because it names the step\'s output directory and files. Put the detail in '
                f'description=.')
        if id.casefold() in _RESERVED_IDS:
            raise WorkflowError(
                f'a step cannot be named "{id}": the run\'s output directory has its own '
                f'"{id.casefold()}" (reserved: {sorted(_RESERVED_IDS)}).')
        for other in self.steps:
            if other == id:
                continue
            if other.casefold() == id.casefold():
                raise WorkflowError(
                    f'step id "{id}" differs from the step "{other}" only by case; their '
                    f'output directories would be one on Windows and macOS. Give it another '
                    f'name.')
            if _file_stem(other).casefold() == _file_stem(id).casefold():
                raise WorkflowError(
                    f'step ids "{id}" and "{other}" both give the file name '
                    f'"{_file_stem(id)}" for their data files. Give it another name.')

    def remove_step(self, id):
        """Remove step ``id`` from the workflow. Its data files and results stay on disk.

        Raises:
            WorkflowError: there is no such step.
        """
        ids = self.steps
        if id not in ids:
            raise WorkflowError(f'workflow "{self.name}" has no step "{id}" (steps: {ids}).')
        del self._steps[ids.index(id)]

    # -- user functions -------------------------------------------------------------------

    def add_operation(self, *funcs, overwrite=False):
        '''Add functions an obs_data item can name as its ``operation``.

        Each function's source is copied (now) and written by :meth:`write` into
        ``<name>_operation_funcs.py`` beside the workflow, which the workflow's
        ``operation_funcs_external_path`` setting then names, so every rank, the command line
        and CUFLynx load it. A function added again replaces its earlier source: edit it where
        it is defined and add it again.

        Args:
            *funcs: functions defined with ``def`` at the top level of a cell or module. A
                helper another function calls is added too (name it with a leading ``_`` and
                it is not registered as an operation).
            overwrite: replace a functions file that was edited by hand, or that no builder
                wrote. Without it such a file is refused, so its edits are not lost.

        Raises:
            WorkflowError: a function cannot be written to a file (a lambda, a closure, a
                name defined only in the session), the workflow names another functions
                file, or the file on disk would lose hand edits.
        '''
        self._add_funcs('operation', funcs, overwrite)

    def add_cost_func(self, *funcs, overwrite=False):
        '''As :meth:`add_operation`, for cost functions an obs_data item names as its
        ``cost_type``: written to ``<name>_cost_funcs.py``, named by
        ``cost_funcs_external_path``.'''
        self._add_funcs('cost', funcs, overwrite)

    def add_modifier_func(self, *funcs, overwrite=False):
        '''As :meth:`add_operation`, for modifier functions a params_for_id entry names as
        its ``modifier``: written to ``<name>_modifier_funcs.py``, named by
        ``modifier_funcs_external_path``. Each must carry ``@modifier_func``.'''
        self._add_funcs('modifier', funcs, overwrite)

    def remove_operation(self, *names):
        '''Remove operations by name (or by the function), so :meth:`write` leaves them out
        of the functions file. Removing the last one removes the setting naming the file; the
        file itself stays on disk.'''
        self._remove_funcs('operation', names)

    def remove_cost_func(self, *names):
        '''As :meth:`remove_operation`, for cost functions.'''
        self._remove_funcs('cost', names)

    def remove_modifier_func(self, *names):
        '''As :meth:`remove_operation`, for modifier functions.'''
        self._remove_funcs('modifier', names)

    def functions(self, kind='operation'):
        '''The names of the functions added so far, in the order they are written. ``kind``
        is ``'operation'``, ``'cost'`` or ``'modifier'``.'''
        return list(self._funcs[_kind(kind)])

    def _add_funcs(self, kind, funcs, overwrite=False):
        adder = _adder(kind)
        for func in funcs:
            if isinstance(func, str):
                raise WorkflowError(
                    f'{adder} takes a function, not a name ({func!r}): pass the function you '
                    f'defined, wf.{adder}(my_function). A built-in function needs no adding: '
                    f'obs_data or params_for_id just names it.')
        key = FUNCS_SETTINGS[kind]
        path = self._funcs_path(kind)
        named = self._named_funcs_files(key)
        if any(p != path for p in named):
            raise WorkflowError(
                f'workflow "{self.name}" already names {key} {sorted(set(named) - {path})}, so '
                f'{adder} cannot add its '
                f'own file: a workflow takes one {kind} functions file. Add these functions to '
                f'that file, or remove the setting.')
        sources, imports = dict(self._funcs[kind]), list(self._imports[kind])
        known = set(sources) | {f.__name__ for f in funcs}
        for func in funcs:
            source, needed = function_source(func, kind, known)
            sources[func.__name__] = source
            imports = list(dict.fromkeys(imports + needed))
        if overwrite:
            self._overwrite[kind] = True
        if sources:
            self._refuse_edited_funcs_file(kind, self._funcs_text(kind, sources, imports))
            self.settings[key] = path
        self._funcs[kind], self._imports[kind] = sources, imports

    def _remove_funcs(self, kind, names):
        names = [getattr(n, '__name__', n) for n in names]
        missing = [n for n in names if n not in self._funcs[kind]]
        if missing:
            raise WorkflowError(f'workflow "{self.name}" has no {kind} function {missing} '
                                f'(added: {list(self._funcs[kind])}).')
        for name in names:
            del self._funcs[kind][name]
        if not self._funcs[kind]:
            self._imports[kind] = []
            if self.settings.get(FUNCS_SETTINGS[kind]) == self._funcs_path(kind):
                del self.settings[FUNCS_SETTINGS[kind]]

    def _funcs_on_disk(self, kind):
        path = self._funcs_path(kind)
        if not os.path.isfile(path):
            return None
        with open(path, encoding='utf-8') as f:
            return f.read()

    def _refuse_edited_funcs_file(self, kind, text, at_write=False):
        '''A WorkflowError when writing ``text`` as the ``kind`` functions file would lose
        something: the file on disk is neither ``text``, nor as a writer generated it (its
        checksum), nor the file :meth:`load` read with everything in it kept. Not after
        ``overwrite=True``. A file with a valid checksum that holds functions this builder
        lacks is refused only when ``at_write`` (from :meth:`write`): while functions are
        still being added, a later add may supply them. A hand-edited file, or one holding
        statements the builder does not keep, is refused at add time and at write.'''
        disk = self._funcs_on_disk(kind)
        if disk is None or disk == text or self._overwrite[kind]:
            return
        path, adder = self._funcs_path(kind), _adder(kind)
        if is_unedited(disk):
            if disk == self._funcs_seen[kind] or not at_write:
                return
            read = read_funcs_file(path, kind)
            held = {n.name for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
            missing = [n for n in (read[0] if read else {}) if n not in held]
            if missing:
                raise WorkflowError(
                    f'{path} holds {missing}, which this builder does not have (another writer '
                    f'added them), and writing would drop them. Start from '
                    f'WorkflowBuilder.load() to keep them, or call wf.write(overwrite=True) '
                    f'(or wf.{adder}(f, overwrite=True)) to replace the file with the '
                    f'functions given to this builder.')
            return
        if disk == self._funcs_seen[kind]:
            if not self._funcs_unkept[kind]:
                return
            raise WorkflowError(
                f'{path} holds statements that a functions file written by the builder does '
                f'not keep (it writes imports and functions only): '
                f'{"; ".join(self._funcs_unkept[kind])}. Move them into the functions that use '
                f'them, or into a module the functions import. wf.write(overwrite=True) '
                f'replaces the file without them.')
        raise WorkflowError(
            f'{path} was edited by hand, or was not written by a WorkflowBuilder, and writing '
            f'would replace it. Edit the function where it is defined and add it again with '
            f'wf.{adder}(f, overwrite=True), or call wf.write(overwrite=True): either '
            f'replaces the file with the functions given to this builder.')

    def _funcs_path(self, kind):
        return os.path.join(self.directory, f'{_file_stem(self.name, "workflow")}_{kind}_funcs.py')

    def _named_funcs_files(self, key, settings=None, steps=None):
        settings = self.settings if settings is None else settings
        named = [settings[key]] if settings.get(key) else []
        named += [s['settings'][key] for s in (self._steps if steps is None else steps)
                  if (s.get('settings') or {}).get(key)]
        return named

    def _check_funcs_settings(self, settings=None, steps=None):
        for kind, key in FUNCS_SETTINGS.items():
            if self._funcs[kind] and any(p != self._funcs_path(kind)
                                         for p in self._named_funcs_files(key, settings,
                                                                          steps)):
                raise WorkflowError(f'{key} names another file than {self._funcs_path(kind)}, '
                                    f'which holds the functions given to the builder; a '
                                    f'workflow takes one {kind} functions file.')

    def funcs_source(self, kind='operation'):
        '''The text of the ``kind`` (``'operation'``, ``'cost'``, ``'modifier'``) functions
        file as :meth:`write` writes it, or None when no function of that kind was added:
        ``utilities.user_funcs_file``'s layout, which CUFLynx writes too. Writes nothing.'''
        kind = _kind(kind)
        if not self._funcs[kind]:
            return None
        return self._funcs_text(kind, self._funcs[kind], self._imports[kind])

    def _funcs_text(self, kind, sources, imports):
        disk = self._funcs_on_disk(kind)
        if disk is not None and disk == self._funcs_seen[kind] \
                and (sources, imports) == self._funcs_loaded[kind]:
            return disk         # the file load() read, its functions unchanged: left as it is
        return funcs_file_source(kind, sources, imports)

    # -- out ------------------------------------------------------------------------------

    def to_dict(self):
        """The workflow as :meth:`write` writes it: the raw dict of
        ``calibration_workflow.json``, its paths relative to :attr:`directory` (with ``/``)
        and each dict input as the ``data/`` file it goes to. Writes nothing."""
        self._check_funcs_settings()
        if not self._steps:
            raise WorkflowError(f'workflow "{self.name}" has no steps yet: add one, for example '
                                f'with wf.add_calibration(id, obs_data=..., params_for_id=...).')
        order = self._order or ['schema_version', 'workflow_name', 'description', 'target',
                                'module_library_dirs', 'settings', 'steps']
        raw = {k: v for k, v in self._head.items() if k not in ('target', 'module_library_dirs')}
        raw['target'] = self._rel_model(self._head['target'])
        if self._head['module_library_dirs'] or 'module_library_dirs' in order:
            raw['module_library_dirs'] = [self._rel(d) for d in self._head['module_library_dirs']]
        if self.settings or (self._order and 'settings' in order):
            raw['settings'] = self._rel_settings(self.settings)
        _, by_text = self._data_files()
        raw['steps'] = [self._rel_step(s, by_text) for s in self._steps]
        raw = {k: raw[k] for k in order if k in raw} | raw
        return jsonable(raw)

    def check(self):
        """Check the workflow without writing or running anything.

        Runs every check of the file (the schema, and those across steps such as an unknown
        ``fixed_from``), then what a run needs beyond the file: a module library when a model
        names library modules, and the settings the chosen calibration method requires.

        Returns:
            the parsed :class:`~libcuflynx.calibration_workflow.spec.Workflow`.

        Raises:
            WorkflowError: naming each problem and how to fix it.
        """
        workflow = parse_workflow(self.to_dict(), path=self.path)
        problems = self._run_problems(workflow)
        if problems:
            raise WorkflowError(f'workflow "{self.name}" cannot run yet:\n  - '
                                + '\n  - '.join(problems))
        return workflow

    def _run_problems(self, workflow):
        """What stops ``workflow`` from running that the file's own checks do not see."""
        from libcuflynx.parsers.PrimitiveParsers import param_id_method_options
        problems = []
        if not workflow.module_library_dirs:
            for label, model in [('the target', workflow.target)] + \
                    [(f'step "{s.id}"', s.model) for s in workflow.steps if s.model is not None]:
                named = self._library_module_named(workflow, model)
                if named:
                    problems.append(
                        f'{label} ({os.path.basename(model.module_array)}) names '
                        f'parameterisation "{named[1]}" of the module {named[0]}, which comes '
                        f'from a module library, and the workflow names no library: pass '
                        f'module_library_dirs=[<the library\'s directory>] to '
                        f'WorkflowBuilder(...).')
                    break
        for step in workflow.steps:
            if step.kind != 'calibration':
                continue
            settings = workflow.step_settings(step)
            method = settings.get('param_id_method') or DEFAULT_PARAM_ID_METHOD
            where = (f'step "{step.id}": its settings' if 'param_id_method' in step.settings
                     or 'optimiser_options' in step.settings else 'the settings')
            options = dict(settings.get('optimiser_options') or {})
            if settings.get('DEBUG'):
                options.update(settings.get('debug_optimiser_options') or {})
            missing = [o['name'] for o in param_id_method_options(method)
                       if o.get('required') and o.get('default') is None
                       and options.get(o['name']) is None]
            if not missing:
                continue
            chosen = method if settings.get('param_id_method') else \
                f"{method} (the param_id_method when none is set)"
            problem = (f"{where} have no optimiser_options {missing}, which {chosen} requires: "
                       f"pass settings={{'param_id_method': {method!r}, 'optimiser_options': "
                       f"{{{missing[0]!r}: ...}}}} to WorkflowBuilder(...), or call "
                       f"wf.update_settings(optimiser_options={{{missing[0]!r}: ...}}).")
            if problem not in problems:
                problems.append(problem)
        return problems

    @staticmethod
    def _library_module_named(workflow, model):
        """``(module label, parameterisation)`` of the first record of the module array
        ``model`` that names a parameterisation of a module no module source of the workflow
        has; None when there is none, or the array cannot be read here."""
        path = getattr(model, 'module_array', None)
        if not path or not str(path).endswith('.json') or not os.path.isfile(path):
            return None
        try:
            from libcuflynx.calibration_workflow.resolve import library_inputs
            from libcuflynx.utilities.module_library import ModuleSources
            with open(path, encoding='utf-8-sig') as f:
                records = json.load(f)
            registry = ModuleSources(library_inputs(workflow)).component_registry()
            for record in records:
                key = (record.get('module_type'), record.get('module_subtype'))
                if record.get('parameterisation') and key not in registry:
                    return f'{key[0]}/{key[1]}', record['parameterisation']
        except Exception:  # noqa: BLE001 - an advisory check: the run reports the rest
            return None
        return None

    def write(self, overwrite=False):
        """Write the workflow to :attr:`directory`, after checking it.

        Writes ``calibration_workflow.json``, a ``data/<step>_obs_data.json`` or
        ``data/<step>_params_for_id.json`` for each input given as a dict or entries, and a
        ``<name>_<kind>_funcs.py`` for the functions given to :meth:`add_operation`,
        :meth:`add_cost_func` and :meth:`add_modifier_func`. A file whose content is unchanged
        is left as it is, so its steps do not become stale. :meth:`run`, :meth:`plan`,
        :meth:`status`, :meth:`model` and :meth:`plot_step` call this first.

        Args:
            overwrite: replace a functions file that was edited by hand, or that no builder
                wrote, with the functions given to this builder.

        Returns:
            the workflow file's path (a str).

        Raises:
            WorkflowError: the workflow does not pass its checks, a functions file would
                lose hand edits, or a step names as its own a file in ``data/`` that the
                builder writes.
        """
        raw = self.to_dict()
        parse_workflow(copy.deepcopy(raw), path=self.path)
        if overwrite:
            self._overwrite = {kind: True for kind in FUNCS_SETTINGS}
        texts = {kind: self.funcs_source(kind) for kind in FUNCS_SETTINGS}
        for kind, text in texts.items():
            if text is not None:
                self._refuse_edited_funcs_file(kind, text, at_write=True)
        for relative, text in self._data_files()[0].items():
            _write_if_changed(os.path.join(self.directory, relative), text)
        for kind, text in texts.items():
            if text is not None:
                _write_if_changed(self._funcs_path(kind), text)
                if text != self._funcs_seen[kind]:
                    self._funcs_seen[kind], self._funcs_unkept[kind] = text, []
                    self._funcs_loaded[kind] = (dict(self._funcs[kind]),
                                                list(self._imports[kind]))
                self._overwrite[kind] = False
        _write_if_changed(self.path, json.dumps(raw, indent=1, allow_nan=False) + '\n')
        return self.path

    def plan(self):
        """What a run would do, without running anything: :meth:`check`, :meth:`write`, then
        :func:`~libcuflynx.calibration_workflow.plan_workflow`.

        Returns:
            ``{'workflow_name', 'target', 'target_parameters', 'module_library_dirs',
            'steps'}``; ``steps`` is in run order, each ``{'id', 'kind', 'model', 'place',
            'depends_on', 'obs_data', 'params_for_id', 'calibrates' (each parameter and where
            it lands in the target), 'fixed_from', 'priors_from', 'store_distribution',
            'settings'}``.
        """
        from libcuflynx.calibration_workflow.runner import plan_workflow
        self.check()
        return plan_workflow(self.write())

    def run(self, *, only=None, from_step=None, force=False, output_dir=None, dry_run=False,
            write_calibrated=False, ranks=1, progress=None):
        """Check and write the workflow (:meth:`check`, :meth:`write`), then run it.

        Without ``only`` and ``from_step``, the steps that are done and current are kept
        (one line is printed for each) and only those that have not run, or are stale, are
        run: running the same cell or script again repeats no calibration. A step is stale
        when its obs_data, params_for_id, parameters, settings or functions file changed
        since it ran, or a step it takes values from is stale or ran again
        (:meth:`status`).

        Args:
            only: run just this step, reusing the results it depends on (which must be
                current).
            from_step: run this step and every step after it, whether or not they are
                current.
            force: run every step again, current or not.
            output_dir: where the run goes. Default :attr:`output_dir`.
            dry_run: run nothing; return the plan (as :meth:`plan`).
            write_calibrated: also write the merged values beside the target's parameters
                file, as ``<name>_calibrated_parameters.json`` (complete runs only).
            ranks: the number of MPI ranks. More than one runs
                ``mpiexec -n <ranks> python -m libcuflynx.scripts.calibration_workflow_run_script``
                on the written file and prints its output as it comes.
            progress: ``callable(event)`` called for each step event (one rank only).
                Default: print one line per event. False: print nothing (with
                ``ranks > 1``, the launched command's output is not printed).

        Returns:
            the workflow result, also written as ``<output_dir>/workflow_result.json``:
            ``{'workflow_name', 'complete', 'order', 'steps', 'merged_parameters', ...}``.
            With ``dry_run``, the plan.

        The run writes ``<output_dir>/<step id>/`` for each step (``step_result.json``, the
        generated model, the calibration's outputs), ``workflow_result.json``, the
        provenance graph and ``_target/``, the target generated with every merged value.

        Raises:
            WorkflowError: the workflow cannot run (see :meth:`check`), a result ``only``
                needs is missing or stale, or the launched command failed.
        """
        if isinstance(ranks, bool) or not isinstance(ranks, int) or ranks < 1:
            raise WorkflowError(f'ranks is the number of MPI ranks, a whole number of 1 or '
                                f'more, not {ranks!r}.')
        if ranks > 1 and callable(progress):
            raise WorkflowError('a progress callable needs ranks=1: with more ranks the run '
                                'is another process, whose output is printed (progress=False '
                                'prints nothing).')
        self.check()
        path = self.write()
        output_dir = os.path.abspath(os.fspath(output_dir or self.output_dir))
        stale_only = not force and not only and not from_step
        if ranks > 1:
            return self._run_mpi(path, output_dir, only, from_step, stale_only, dry_run,
                                 write_calibrated, ranks, echo=progress is not False)
        from libcuflynx.calibration_workflow.runner import run_calibration_workflow
        if progress is None:
            from libcuflynx.scripts.calibration_workflow_run_script import _print_progress
            progress = _print_progress
        return run_calibration_workflow(path, output_dir=output_dir, from_step=from_step,
                                        only=only, stale_only=stale_only, dry_run=dry_run,
                                        write_calibrated=write_calibrated,
                                        progress=progress or None)

    def _run_mpi(self, path, output_dir, only, from_step, stale_only, dry_run,
                 write_calibrated, ranks, echo=True):
        from libcuflynx.calibration_workflow.runner import WORKFLOW_RESULT, plan_workflow
        command = ['mpiexec', '-n', str(int(ranks)), sys.executable, '-m',
                   'libcuflynx.scripts.calibration_workflow_run_script', path,
                   '--output-dir', output_dir]
        if only:
            command += ['--only', only]
        if from_step:
            command += ['--from-step', from_step]
        if stale_only:
            command.append('--stale-only')
        if write_calibrated:
            command.append('--write-calibrated')
        if dry_run:
            return plan_workflow(path)
        started = time.time()
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1,
                                       env=launcher_environment(os.environ))
        except FileNotFoundError as exc:
            raise WorkflowError(
                f'run(ranks={ranks}) needs mpiexec, which is not on the PATH ({exc}). Install '
                f'an MPI (Open MPI or MPICH) or run with ranks=1.') from exc
        tail = []
        for line in process.stdout:
            if echo:
                print(line, end='')
            tail = (tail + [line])[-40:]
        if process.wait() != 0:
            raise WorkflowError(f'{" ".join(command)} exited with {process.returncode}; its '
                                f'last lines:\n{"".join(tail)}')
        result_path = os.path.join(output_dir, WORKFLOW_RESULT)
        if not os.path.isfile(result_path) or os.path.getmtime(result_path) < started:
            raise WorkflowError(f'{" ".join(command)} wrote no {result_path}.')
        with open(result_path) as f:
            return json.load(f)

    def status(self, output_dir=None):
        """Where each step stands in the run in ``output_dir`` (default :attr:`output_dir`),
        after :meth:`write`; nothing is run (:func:`workflow_status`).

        Returns:
            ``{'workflow_name', 'order', 'complete', 'steps'}``: ``complete`` is True when
            every step is done and current, and ``steps`` is ``{id: {'status' ('done' or
            'not_run'), 'stale', 'library_changed', 'best_cost', 'finished',
            'point_estimate'}}``. ``stale`` is True when the step's obs_data, params_for_id,
            parameters, settings or functions file changed since it ran, or a step it takes
            values from is stale or ran again after it.
        """
        from libcuflynx.calibration_workflow.runner import workflow_status
        return workflow_status(self.write(), os.fspath(output_dir or self.output_dir))

    def model(self, view, output_dir=None, work_dir=None):
        """Generate the model of step ``view``, or of ``'target'``, with the values found so
        far in ``output_dir`` (:func:`workflow_model`), after :meth:`write`.

        Args:
            view: a step id, or ``'target'``.
            output_dir: the run to take values from. Default :attr:`output_dir`.
            work_dir: where the model is generated. Default ``<output_dir>/models``, giving
                ``<output_dir>/models/<view>/``.

        Returns:
            ``{'view', 'kind' ('step' or 'target'), 'model_path', 'flat_model_path' (one
            self-contained CellML file), 'parameters_path', 'obs_data_path',
            'params_for_id_path', 'fixed', 'calibrated', 'waiting_for' (steps whose results
            it should have but that have not run), 'stale', ...}``.
        """
        from libcuflynx.calibration_workflow.runner import MODELS_DIR, workflow_model
        output_dir = os.fspath(output_dir or self.output_dir)
        return workflow_model(self.write(), view, output_dir=output_dir,
                              work_dir=os.fspath(work_dir or os.path.join(output_dir,
                                                                          MODELS_DIR)))

    def step_dir(self, id, output_dir=None):
        """The output directory of step ``id`` in the run in ``output_dir`` (default
        :attr:`output_dir`), as a ``pathlib.Path``; it exists once the step has run."""
        from pathlib import Path
        if id not in self.steps:
            raise WorkflowError(f'workflow "{self.name}" has no step "{id}" (steps: '
                                f'{self.steps}).')
        return Path(os.fspath(output_dir or self.output_dir)) / id

    def result(self, id, output_dir=None):
        """What step ``id`` recorded when it ran into ``output_dir`` (default
        :attr:`output_dir`): the content of its ``step_result.json``, read with
        :func:`~libcuflynx.calibration_workflow.load_workflow_run`. Writes nothing.

        Returns:
            a dict with, for every kind, ``'step'``, ``'kind'``, ``'status'``, ``'started'``,
            ``'finished'``, ``'duration_s'``, ``'inputs'`` and ``'step_dir'``, and the kind's
            own keys, its files as absolute paths:

            - calibration: ``'values'`` (``{"<vessel>/<parameter>": best value}``),
              ``'best_cost'``, ``'calibrated'`` (each value with its units and where it
              lands in the target), ``'fixed'``, ``'point_estimate'``,
              ``'param_id_output_dir'``, ``'distribution_dir'``;
            - sensitivity: ``'sensitivity_dir'``, ``'sobol_indices'``;
            - reduction: ``'parameters'`` (the names chosen), ``'selected'`` (each with its
              score), ``'params_for_id'`` (the file a later step calibrates), ``'report'``
              (``param_reduction.json``);
            - emulation: ``'emulator_dir'``; identifiability: ``'laplace'``;
            - sweep: ``'sweep_dir'``, ``'sweep_results'`` (``sweep_results.npz``).

        Raises:
            WorkflowError: there is no such step, or it has not run into ``output_dir``.
        """
        from libcuflynx.calibration_workflow.runner import STEP_OUTPUTS, load_workflow_run
        step_dir = str(self.step_dir(id, output_dir))
        output_dir = os.path.abspath(os.fspath(output_dir or self.output_dir))
        try:
            result = load_workflow_run(output_dir)['steps'].get(id)
        except WorkflowError:
            result = None
        if not result or result.get('status') != 'done':
            raise WorkflowError(f'step "{id}" has not run into {output_dir}.')
        result = dict(result, step_dir=os.path.abspath(step_dir))
        result.setdefault('kind', 'calibration')
        for key in STEP_OUTPUTS.get(result['kind'], ()) + ('report', 'model_path'):
            if isinstance(result.get(key), str):
                result[key] = os.path.normpath(os.path.join(output_dir, result[key]))
        if result['kind'] == 'calibration':
            result['values'] = {f'{row["vessel"]}/{row["param"]}': row['value']
                                for row in result.get('calibrated', [])}
        if result['kind'] == 'reduction':
            result['parameters'] = [row['parameter'] for row in result.get('selected', [])]
        return result

    def plot_step(self, id, output_dir=None):
        """Plot a calibration step's best fit against its obs_data, after :meth:`write`:
        the model is simulated with the step's best values and ``plot_outputs`` of the
        param_id run writes the figures under the step's ``param_id/`` directory.

        Args:
            id: a calibration step that has run.
            output_dir: the run. Default :attr:`output_dir`.

        Returns:
            the image files written (a sorted list of paths).

        Raises:
            WorkflowError: the step has not run, or is not a calibration.
        """
        from libcuflynx.calibration_workflow.resolve import resolve_workflow
        from libcuflynx.calibration_workflow.runner import _get_comm, _Run
        from libcuflynx.calibration_workflow.spec import load_workflow
        output_dir = os.path.abspath(os.fspath(output_dir or self.output_dir))
        result = self.result(id, output_dir)
        if result['kind'] != 'calibration':
            raise WorkflowError(f'step "{id}" is a {result.get("kind")} step; plot_step plots '
                                f'a calibration.')
        workflow = load_workflow(self.write())
        step = workflow.step(id)
        run = _Run(resolve_workflow(workflow), output_dir, _get_comm(None), None)
        inp = run._inp(step, result['settings'], result['generated'],
                       result['params_for_id_used'])
        from libcuflynx.param_id.paramID import CVS0DParamID
        from libcuflynx.parsers.PrimitiveParsers import YamlFileParser
        inp = YamlFileParser().parse_user_inputs_file(inp, obs_path_needed=True,
                                                      do_generation_with_fit_parameters=False)
        # the step's study manifest is the run's; plotting leaves it alone
        inp.update(write_cuflynx_study=False, one_rank=True)
        started = time.time()
        param_id = CVS0DParamID.init_from_dict(inp)
        try:
            param_id.simulate_with_best_param_vals()
            param_id.plot_outputs()
        finally:
            param_id.close_simulation()
        return sorted(os.path.join(root, name)
                      for root, _, names in os.walk(os.path.join(output_dir, id))
                      for name in names
                      if name.endswith(('.png', '.pdf', '.svg'))
                      and os.path.getmtime(os.path.join(root, name)) >= started)

    # -- inputs ---------------------------------------------------------------------------

    def _obs_data(self, step_id, value):
        if hasattr(value, 'get_obs_data_dict'):
            value = copy.deepcopy(value.get_obs_data_dict())
        if not isinstance(value, dict):
            return self._abs(value)
        from libcuflynx.schemas import OBS_DATA_SCHEMA
        from libcuflynx.schemas.validation import validate
        doc = jsonable(value)
        validate(OBS_DATA_SCHEMA, doc, source=f'step "{step_id}": obs_data', what='obs_data')
        return _Inline('obs_data', _json_text(doc, f'step "{step_id}": obs_data'))

    def _params_for_id(self, step_id, value):
        from libcuflynx.utilities import params_for_id_file
        if isinstance(value, (list, tuple)):
            entries = jsonable(list(value))
            if any(isinstance(e, dict) and 'vessel_name' in e for e in entries):
                value = params_for_id_file.entries_to_document(entries)
            else:
                value = {'version': params_for_id_file.VERSION, 'params': entries}
        if not isinstance(value, dict):
            return self._abs(value)
        from libcuflynx.schemas import PARAMS_FOR_ID_SCHEMA
        from libcuflynx.schemas.validation import validate
        doc = jsonable(value)
        validate(PARAMS_FOR_ID_SCHEMA, doc, source=f'step "{step_id}": params_for_id',
                 what='params_for_id')
        _json_text(doc, f'step "{step_id}": params_for_id')
        return _Inline('params_for_id', params_for_id_file.dump_params_for_id(doc))

    def _data_files(self):
        '''``({path relative to directory: text}, {text: path})`` of the inline inputs; content
        several steps share is written once, named after the first step.'''
        files, by_text, owner = {}, {}, {}
        for step in self._steps:
            for key in ('obs_data', 'params_for_id'):
                value = step.get(key)
                if isinstance(value, _Inline) and value.text not in by_text:
                    relative = f'{DATA_DIR}/{_file_stem(step["id"])}_{key}.json'
                    by_text[value.text] = relative
                    files[relative] = value.text
                    owner[os.path.join(self.directory, DATA_DIR,
                                       os.path.basename(relative))] = (step['id'], key)
        for step in self._steps:
            for key in ('obs_data', 'params_for_id'):
                value = step.get(key)
                if isinstance(value, str) and os.path.normpath(value) in owner:
                    other, other_key = owner[os.path.normpath(value)]
                    raise WorkflowError(
                        f'step "{step["id"]}" names {value} as its {key}, and that is the file '
                        f'the builder writes the {other_key} given as a dict to step "{other}" '
                        f'to, so writing would replace your file. Keep your own files outside '
                        f'{os.path.join(self.directory, DATA_DIR)}, or under another name.')
        return files, by_text

    def _validate_step(self, step):
        '''Check one step against the schema, as a one-step workflow.'''
        from libcuflynx.schemas import CALIBRATION_WORKFLOW_SCHEMA
        from libcuflynx.schemas.validation import SchemaValidationError, validate
        raw = self._rel_step(step, {})
        raw = jsonable({k: (f'{DATA_DIR}/{_file_stem(step["id"])}_{k}.json'
                            if isinstance(v, _Inline)
                            else v) for k, v in raw.items()})
        try:
            validate(CALIBRATION_WORKFLOW_SCHEMA,
                     {'workflow_name': self.name, 'target': self._rel_model(self._head['target']),
                      'steps': [raw]},
                     source=f'step "{step["id"]}"', what='calibration workflow step')
        except SchemaValidationError as exc:
            raise WorkflowError(str(exc)) from exc

    # -- paths ----------------------------------------------------------------------------

    def _abs(self, value, base=None):
        value = os.path.expanduser(os.fspath(value))
        return os.path.normpath(os.path.join(base or os.getcwd(), value))

    def _rel(self, path):
        try:
            return _forward_slashes(os.path.relpath(path, self.directory))
        except ValueError:      # another drive
            return path

    def _abs_model(self, model, base=None):
        if isinstance(model, (str, os.PathLike)):
            return self._abs(model, base)
        if isinstance(model, dict) and set(model) & set(_MODEL_PATH_KEYS):
            return {k: (self._abs(v, base) if k in _MODEL_PATH_KEYS else v)
                    for k, v in model.items()}
        return model

    def _rel_model(self, model):
        if isinstance(model, str):
            return self._rel(model)
        if isinstance(model, dict) and set(model) & set(_MODEL_PATH_KEYS):
            return {k: (self._rel(v) if k in _MODEL_PATH_KEYS else v)
                    for k, v in model.items()}
        return model

    def _abs_settings(self, settings, base=None):
        return {k: (self._abs(v, base) if k in PATH_SETTINGS and v else v)
                for k, v in settings.items()}

    def _rel_settings(self, settings):
        return {k: (self._rel(v) if k in PATH_SETTINGS and v else v)
                for k, v in settings.items()}

    def _abs_step(self, step, base):
        step = copy.deepcopy(step)
        for key in _STEP_PATH_KEYS:
            if isinstance(step.get(key), str):
                step[key] = self._abs(step[key], base)
        if 'model' in step:
            step['model'] = self._abs_model(step['model'], base)
        if isinstance(step.get('settings'), dict) and step.get('kind') != 'reduction':
            step['settings'] = self._abs_settings(step['settings'], base)
        return step

    def _rel_step(self, step, by_text):
        out = {}
        for key, value in step.items():
            if isinstance(value, _Inline):
                value = by_text.get(value.text, value)
            elif key in _STEP_PATH_KEYS and isinstance(value, str):
                value = self._rel(value)
            elif key == 'model':
                value = self._rel_model(value)
            elif key == 'settings' and step.get('kind') != 'reduction':
                value = self._rel_settings(value)
            out[key] = copy.deepcopy(value)
        return out


def _forward_slashes(relative, sep=os.sep):
    """A relative path as a workflow file holds it on every platform: with ``/``, which
    Windows reads too."""
    return relative.replace(sep, '/') if sep != '/' else relative


def _adder(kind):
    return 'add_operation' if kind == 'operation' else f'add_{kind}_func'


def _kind(kind):
    if kind not in FUNCS_SETTINGS:
        raise WorkflowError(f'{kind!r} is not a kind of function; the kinds are '
                            f'{list(FUNCS_SETTINGS)}.')
    return kind


def _file_stem(name, default='step'):
    """``name`` as part of a file name: runs of other characters than letters, digits, ``.``
    and ``-`` become ``_``."""
    return re.sub(r'[^\w.-]+', '_', name).strip('_') or default


def _model_label(model):
    if isinstance(model, str):
        return os.path.basename(model)
    if 'select' in model:
        return f'selection {model["select"]}'
    if 'module_type' in model:
        return f'{model["module_type"]}/{model["version"]}/{model["parameterisation"]}'
    # a model that is a file: its first key names it
    return os.path.basename(str(next(iter(model.values()))))


def _given(values, without=()):
    '''The keyword arguments of an ``add_*`` call that were given, as raw step keys.'''
    out = {}
    for key, value in values.items():
        if key in ('self', 'id') or key in without:
            continue
        if value is None or value is False or (isinstance(value, (tuple, list, dict))
                                               and not value and key != 'parameters'):
            continue
        out[key] = list(value) if isinstance(value, tuple) else value
    return out


def _write_if_changed(path, text):
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            if f.read() == text:
                return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.{os.getpid()}.tmp'    # one per process: several ranks may write at once
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(text)
    os.replace(tmp, path)
    return True


# --------------------------------------------------------------------------------------------
# a function as source
# --------------------------------------------------------------------------------------------

def _code_names(code):
    names = set(code.co_names)
    for const in code.co_consts:
        if inspect.iscode(const):
            names |= _code_names(const)
    return names


def _root_name(node):
    while isinstance(node, (ast.Call, ast.Attribute)):
        if isinstance(node, ast.Attribute) and not isinstance(node.value, (ast.Name,
                                                                           ast.Attribute,
                                                                           ast.Call)):
            break
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def _decorator_names(node):
    '''The names a decorator could be known by: its root name and its last attribute.'''
    names = set()
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Attribute):
            names.add(target.attr)
        root = _root_name(target)
        if root:
            names.add(root)
    return names


def _submodules_used(node, name, module):
    '''The dotted names of the submodules of ``module`` that the function ``node`` reaches
    through the global ``name`` (``name.sub.attr``), deepest first.'''
    found = set()
    for attribute in ast.walk(node):
        if not isinstance(attribute, ast.Attribute):
            continue
        chain, base = [], attribute
        while isinstance(base, ast.Attribute):
            chain.append(base.attr)
            base = base.value
        if not isinstance(base, ast.Name) or base.id != name:
            continue
        obj = module
        for attr in reversed(chain):
            obj = getattr(obj, attr, None)
            if not inspect.ismodule(obj) or not obj.__name__.startswith(module.__name__ + '.'):
                break
            found.add(obj.__name__)
    # a.b.c imports a.b too
    return sorted((f for f in found if not any(o.startswith(f + '.') for o in found)),
                  reverse=True)


def _import_line(name, obj, func_module):
    '''The import that gives ``name`` the object ``obj`` in another process, or None.'''
    if inspect.ismodule(obj):
        module = obj.__name__
        return f'import {module}' if module == name else f'import {module} as {name}'
    module = getattr(obj, '__module__', None)
    attr = getattr(obj, '__name__', None)
    if not module or not attr or module in ('__main__', func_module) \
            or module.startswith(('__', 'ipykernel')):
        return None
    try:
        if getattr(importlib.import_module(module), attr, None) is not obj:
            return None
    except Exception:  # noqa: BLE001 - not importable: refused below
        return None
    return f'from {module} import {attr}' + ('' if attr == name else f' as {name}')


def function_source(func, kind='operation', known=()):
    '''``(source, imports)``: the definition of ``func`` as it is written into a ``kind``
    functions file (``utilities.user_funcs_file``), and the import statements it needs beyond
    the file's header (the modules and importable objects it uses). A marker given to it after
    its definition (``f.series_to_constant = True``, ``exact_tangent``) is written as its
    decorator, so the function is complete in itself.

    ``known`` are the names of the other functions in the file. A lambda, a function that
    closes over local variables, and one that uses a name defined only in the session (a
    variable, or a function not in ``known``) are refused: they would not exist where the
    file is imported.'''
    name = getattr(func, '__name__', None)
    if not inspect.isfunction(func):
        raise WorkflowError(f'{func!r} is not a function defined with def.')
    if name == '<lambda>':
        raise WorkflowError('a lambda cannot be written to a functions file: define it with def '
                            '(its name is the name obs_data uses).')
    if func.__closure__:
        raise WorkflowError(
            f'{name} uses {list(func.__code__.co_freevars)} from the function it is defined in; '
            f'define it at the top level of a cell or module, with what it needs as arguments '
            f'or defaults.')
    try:
        source = textwrap.dedent(inspect.getsource(func))
    except (OSError, TypeError) as exc:
        raise WorkflowError(f'the source of {name} cannot be found ({exc}): define it in a '
                            f'notebook cell or a .py file, not with exec or eval.') from exc
    tree = ast.parse(source)
    node = tree.body[0] if tree.body else None
    if not isinstance(node, ast.FunctionDef) or node.name != name:
        raise WorkflowError(f'the source found for {name} is not its def statement.')
    if kind == 'modifier' and 'modifier_func' not in _decorator_names(node):
        raise WorkflowError(f'{name}: a modifier function carries @modifier_func(inputs=...).')

    header = HEADER_NAMES[kind]
    imports, refused, added = [], [], []
    decorators = _decorator_names(node)
    for attribute, decorator in _MARKS[kind]:
        if getattr(func, attribute, False) and decorator not in decorators:
            added.append(f'@{decorator}')
    if kind == 'operation' and getattr(func, 'exact_tangent', None) is not None \
            and 'exact_tangent' not in decorators:
        rule = func.exact_tangent
        rule_name = getattr(rule, '__name__', None)
        line = None if rule_name in known else _import_line(rule_name or '', rule,
                                                            func.__module__)
        if rule_name not in known and line is None:
            refused.append((rule_name, 'is its exact_tangent rule, defined in this session'))
        else:
            imports += [line] if line else []
            imports.append('from libcuflynx.param_id.operation_funcs import exact_tangent')
            added.append(f'@exact_tangent({rule_name})')

    loaded = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)
              and isinstance(n.ctx, ast.Load)}
    # what the def statement itself evaluates, where the file is imported: its decorators,
    # its default values and its annotations
    arguments = node.args
    evaluated = list(node.decorator_list) + arguments.defaults \
        + [d for d in arguments.kw_defaults if d is not None] \
        + [a.annotation for a in arguments.posonlyargs + arguments.args + arguments.kwonlyargs
           + [arguments.vararg, arguments.kwarg] if a is not None and a.annotation is not None] \
        + ([node.returns] if node.returns is not None else [])
    def_loads = {n.id for part in evaluated for n in ast.walk(part) if isinstance(n, ast.Name)}
    used = (_code_names(func.__code__) & loaded) | def_loads
    for global_name in sorted(used):
        if global_name in known or global_name == name:
            continue
        if global_name not in func.__globals__:
            if hasattr(builtins, global_name) or global_name in header:
                continue
            refused.append((global_name, 'is not defined'))
            continue
        obj = func.__globals__[global_name]
        if global_name in header:
            if _is(obj, header[global_name]):
                continue
            refused.append((global_name, 'is a name the functions file defines itself'))
            continue
        line = _import_line(global_name, obj, func.__module__)
        if line is None:
            refused.append((global_name, f'is a {type(obj).__name__} defined in this session'))
        else:
            imports.append(line)
            if inspect.ismodule(obj):
                # a submodule used through its package (pkg.sub.f) is imported itself:
                # "import pkg" alone does not load it
                imports += [f'import {sub}' for sub in _submodules_used(node, global_name, obj)
                            if f'import {sub}' != line]
    if refused:
        why = '; '.join(f'{n!r} {reason}' for n, reason in refused)
        adder = _adder(kind)
        raise WorkflowError(
            f'{name} cannot be written to a functions file: {why}. The file is imported on its '
            f'own, so a function it uses goes to {adder} '
            f'too (name helpers with a leading _ so they are not registered), a value goes '
            f'into the function, and anything else is imported from a module.')
    return '\n'.join(added + [source.strip('\n')]) + '\n', sorted(set(imports))


def _is(obj, spec):
    module, attr = spec
    if module is None:          # the math backend mb, which libcuflynx rebinds
        return True
    try:
        target = importlib.import_module(module)
    except ImportError:
        return False
    return obj is (getattr(target, attr, None) if attr else target)
