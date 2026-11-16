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
    wf.run()                                  # writes the files, then runs the workflow
    wf.status()

The builder edits the workflow's raw dict, whose keys are the schema's
(``schemas/calibration_workflow.schema.json``): each ``add_*`` takes the step's schema keys as
keyword arguments and checks the step against the schema at once, and :meth:`check` /
:meth:`write` run :func:`~libcuflynx.calibration_workflow.spec.parse_workflow` on the whole
workflow. :meth:`add_step` takes any step as raw schema keys.

What :meth:`write` puts in ``directory``::

    calibration_workflow.json         the workflow, its paths relative to this directory
    data/<step>_obs_data.json         an obs_data given as a dict (or an ObsDataCreator)
    data/<step>_params_for_id.json    a params_for_id given as entries or a document
    <name>_operation_funcs.py         the functions given to add_operation (and _cost_funcs.py,
                                      _modifier_funcs.py for add_cost_func, add_modifier_func)

Dicts are copied when they are added, so changing one afterwards does not change the step. A
file is rewritten only when its content changes, so the runner's change detection (the sha256
of each input file) redoes exactly the steps whose data or functions changed.
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

from libcuflynx.calibration_workflow.spec import (PATH_SETTINGS, WORKFLOW_FILE_NAME,
                                                  WorkflowError, parse_workflow)
from libcuflynx.utilities.user_funcs_file import (FUNCS_SETTINGS, HEADER_NAMES,
                                                  funcs_file_source, read_funcs_file)

#: where a dict input is written, relative to the workflow's directory
DATA_DIR = 'data'
#: the marker each decorator sets on a function: (attribute, decorator), by kind of file. A
#: marker set after the def (``f.series_to_constant = True``) is written as the decorator.
_MARKS = {'operation': [('series_to_constant', 'series_to_constant'),
                        ('_circulatory_differentiable', 'differentiable')],
          'cost': [('is_MLE', 'is_MLE'), ('cost_combiner', 'cost_combiner'),
                   ('_circulatory_differentiable', 'differentiable')],
          'modifier': []}

#: environment variables of an MPI launcher, left out of the environment run(ranks=N) gives
#: mpiexec
_LAUNCHER_ENV_PREFIXES = ('OMPI_', 'PMIX_', 'PMI_', 'HYDRA_', 'OPAL_', 'PRTE_')
# the order a built step's keys are written in (the schema's)
_STEP_KEYS = ('id', 'kind', 'description', 'comment', 'model', 'obs_data', 'params_for_id',
              'submodule', 'vessel', 'vessel_map', 'fixed_from', 'priors_from',
              'store_distribution', 'on_target', 'settings', 'emulator_from', 'result_from',
              'sensitivity_from', 'params_for_id_from', 'set', 'sweep')


# --------------------------------------------------------------------------------------------
# schema fragments
# --------------------------------------------------------------------------------------------

def library_module(module_type, version, parameterisation):
    '''A model that is a module of a module library: one parameterisation of one version.'''
    return {'module_type': module_type, 'version': version, 'parameterisation': parameterisation}


def module_array(path, parameters=None):
    '''A model that is a CA module array; ``parameters`` when its parameters file is not
    ``<prefix>_parameters.json`` beside it.'''
    if parameters is None:
        return os.fspath(path)
    return {'module_array': os.fspath(path), 'parameters': os.fspath(parameters)}


def select(*instances):
    '''A step model that is a selection of the target's instances (record names, or paths into
    a super instance such as ``heart/lv``).'''
    if len(instances) == 1 and isinstance(instances[0], (list, tuple)):
        instances = instances[0]
    return {'select': list(instances)}


def prior(step, kind=None, inflate=None, parameters=None):
    '''A ``priors_from`` entry: the stored distribution of ``step``.'''
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
    '''A set step entry: ``parameter`` (``<vessel>/<parameter>``) set to ``value``.'''
    entry = {'parameter': parameter, 'value': value, 'reason': reason}
    if units is not None:
        entry['units'] = units
    return entry


def scale_value(parameter, scale, *, of, reason, units=None):
    '''A set step entry: ``parameter`` set to ``scale`` times its value in step ``of`` (or
    ``"inherited"``, the target's own value).'''
    entry = {'parameter': parameter, 'scale': scale, 'of': of, 'reason': reason}
    if units is not None:
        entry['units'] = units
    return entry


def jsonable(value):
    '''``value`` with numpy scalars and arrays as Python numbers and lists, tuples as lists and
    paths as strings, so it is written as JSON.'''
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if type(value).__module__ == 'numpy':
        if hasattr(value, 'tolist'):
            return jsonable(value.tolist())
    return value


# --------------------------------------------------------------------------------------------
# the builder
# --------------------------------------------------------------------------------------------

class _Inline:
    '''An obs_data or params_for_id given as content; written to ``data/`` by ``write()``.'''

    def __init__(self, key, text):
        self.key = key
        self.text = text


class WorkflowBuilder:
    '''A calibration workflow made by API calls (see the module docstring).

    Args:
        name: the workflow's ``workflow_name``.
        target: the model the calibrated values are merged into: a module array path,
            :func:`module_array` or :func:`library_module`.
        directory: where the workflow file and its data are written.
        description: the workflow's description.
        module_library_dirs: module libraries the steps' models come from.
        settings: the workflow's settings (user_inputs keys, ``spec.ALLOWED_SETTINGS``).
    '''

    def __init__(self, name, target, *, directory, description=None, module_library_dirs=(),
                 settings=None):
        self.directory = os.path.abspath(os.path.expanduser(os.fspath(directory)))
        self.path = os.path.join(self.directory, WORKFLOW_FILE_NAME)
        self._head = {'schema_version': 1, 'workflow_name': name}
        if description is not None:
            self._head['description'] = description
        self._head['target'] = self._abs_model(copy.deepcopy(target))
        self._head['module_library_dirs'] = [self._abs(d) for d in module_library_dirs]
        self.settings = self._abs_settings(copy.deepcopy(dict(settings or {})))
        self._steps = []
        self._order = None
        self._funcs = {kind: {} for kind in FUNCS_SETTINGS}
        self._imports = {kind: [] for kind in FUNCS_SETTINGS}

    @classmethod
    def load(cls, path):
        '''A builder for the existing workflow file ``path``, written back to the same file.
        A functions file this builder wrote is read back, so more functions can be added.'''
        path = os.path.abspath(os.fspath(path))
        with open(path, encoding='utf-8-sig') as f:
            raw = json.load(f)
        directory = os.path.dirname(path)
        self = cls.__new__(cls)
        self.directory, self.path = directory, path
        self._funcs = {kind: {} for kind in FUNCS_SETTINGS}
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
        self._imports = {kind: [] for kind in FUNCS_SETTINGS}
        for kind in FUNCS_SETTINGS:
            if os.path.isfile(self._funcs_path(kind)):
                self._funcs[kind], self._imports[kind] = \
                    read_funcs_file(self._funcs_path(kind), kind) or ({}, [])
        return self

    @property
    def name(self):
        return self._head['workflow_name']

    @property
    def output_dir(self):
        '''Where :meth:`run` writes unless told otherwise:
        ``<directory>/workflow_output/<name>``.'''
        return os.path.join(self.directory, 'workflow_output', self.name)

    @property
    def steps(self):
        '''The step ids, in the order they were added.'''
        return [s['id'] for s in self._steps]

    def update_settings(self, **settings):
        '''Set workflow settings; a dict setting is merged one level deep, as a step's are.'''
        settings = self._abs_settings(copy.deepcopy(settings))
        for key, value in settings.items():
            if isinstance(value, dict) and isinstance(self.settings.get(key), dict):
                self.settings[key] = {**self.settings[key], **value}
            else:
                self.settings[key] = value
        self._check_funcs_settings()

    # -- steps ----------------------------------------------------------------------------

    def add_calibration(self, id, *, obs_data, params_for_id=None, params_for_id_from=None,
                        model=None, fixed_from=(), priors_from=(), store_distribution=False,
                        emulator_from=None, vessel=None, submodule=None, vessel_map=None,
                        settings=None, description=None, comment=None):
        '''A calibration: fits ``params_for_id`` (or a reduction's choice,
        ``params_for_id_from``) of ``model`` (default the target) to ``obs_data``. Returns
        ``id``.'''
        return self.add_step(id, **_given(locals(), model_default=True))

    def add_sensitivity(self, id, *, obs_data, params_for_id, sa_options, model=None,
                        fixed_from=(), result_from=None, emulator_from=None, vessel=None,
                        submodule=None, vessel_map=None, settings=None, description=None,
                        comment=None):
        '''A sensitivity analysis (``sa_options``) of ``params_for_id`` against ``obs_data``.'''
        given = _given(locals(), model_default=True)
        given['settings'] = dict(given.get('settings') or {}, sa_options=given.pop('sa_options'))
        return self.add_step(id, kind='sensitivity', **given)

    def add_reduction(self, id, *, sensitivity_from, method='pca', n_params=None,
                      index_type=None, criterion=None, threshold=None, min_mode_std=None,
                      variance_fraction=None, min_index=None, description=None, comment=None):
        '''A reduction: the parameters a later step fits (``params_for_id_from``), chosen by
        the Sobol indices of ``sensitivity_from``. Options left as None take the schema's
        defaults.'''
        options = {k: v for k, v in dict(
            method=method, n_params=n_params, index_type=index_type, criterion=criterion,
            threshold=threshold, min_mode_std=min_mode_std, variance_fraction=variance_fraction,
            min_index=min_index).items() if v is not None}
        return self.add_step(id, kind='reduction', sensitivity_from=sensitivity_from,
                             settings={'param_reduction_options': options},
                             **_given({'description': description, 'comment': comment}))

    def add_emulation(self, id, *, obs_data, emulator_settings, params_for_id=None,
                      params_for_id_from=None, model=None, fixed_from=(), vessel=None,
                      submodule=None, vessel_map=None, settings=None, description=None,
                      comment=None):
        '''An emulator training (``emulator_settings``), for steps with ``emulator_from``.'''
        given = _given(locals(), model_default=True)
        given['settings'] = dict(given.get('settings') or {},
                                 emulator_settings=given.pop('emulator_settings'))
        return self.add_step(id, kind='emulation', **given)

    def add_identifiability(self, id, *, obs_data, params_for_id, ia_options, result_from,
                            model=None, fixed_from=(), vessel=None, submodule=None,
                            vessel_map=None, settings=None, description=None, comment=None):
        '''An identifiability analysis (``ia_options``) at the result of ``result_from``.'''
        given = _given(locals(), model_default=True)
        given['settings'] = dict(given.get('settings') or {}, ia_options=given.pop('ia_options'))
        return self.add_step(id, kind='identifiability', **given)

    def add_set(self, id, entries, *, description=None, comment=None):
        '''Values set by hand (:func:`set_value`, :func:`scale_value`), each with a reason.'''
        return self.add_step(id, kind='set', set=list(entries),
                             **_given({'description': description, 'comment': comment}))

    def add_sweep(self, id, *, parameters, model=None, on_target=False, obs_data=None,
                  design=None, outputs=None, zip=None, fixed_from=(), result_from=None,
                  vessel=None, submodule=None, vessel_map=None, settings=None,
                  description=None, comment=None):
        '''A parameter sweep: ``parameters`` ({name: [min, max] | {min, max, points, scale} |
        {values} | {factors}}) over ``model`` (default the target), or with ``on_target`` over
        the target with every calibrated and set value.'''
        sweep = {k: v for k, v in (('design', design), ('outputs', outputs),
                                   ('parameters', parameters), ('zip', zip)) if v is not None}
        given = _given({k: v for k, v in locals().items()
                        if k not in ('design', 'outputs', 'parameters', 'zip', 'sweep')},
                       model_default=not on_target)
        return self.add_step(id, kind='sweep', sweep=sweep, **given)

    def add_step(self, id, kind='calibration', **raw):
        '''A step from its raw schema keys. A step with this id already replaces it in place.
        ``obs_data`` may be a path, a dict or an ObsDataCreator; ``params_for_id`` a path, a
        list of entries (``{vessel_name, param_name, min, max}`` or ``{targets, ...}``) or a
        ``{"version", "params"}`` document. Returns ``id``.'''
        step = {'id': id}
        if kind != 'calibration':
            step['kind'] = kind
        for key, value in raw.items():
            step[key] = copy.deepcopy(value)
        if 'obs_data' in step:
            step['obs_data'] = self._obs_data(id, step['obs_data'])
        if 'params_for_id' in step:
            step['params_for_id'] = self._params_for_id(id, step['params_for_id'])
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
        if id in ids:
            self._steps[ids.index(id)] = step
        else:
            self._steps.append(step)
        self._check_funcs_settings()
        return id

    def remove_step(self, id):
        '''Remove step ``id``.'''
        ids = self.steps
        if id not in ids:
            raise WorkflowError(f'workflow "{self.name}" has no step "{id}" (steps: {ids}).')
        del self._steps[ids.index(id)]

    # -- user functions -------------------------------------------------------------------

    def add_operation(self, *funcs):
        '''Write each function's source into ``<name>_operation_funcs.py`` beside the workflow
        and name that file in the workflow's ``operation_funcs_external_path``, so obs_data
        items can use it as their ``operation`` on every rank, from the command line and in
        CUFLynx. A function added again replaces its earlier source.'''
        self._add_funcs('operation', funcs)

    def add_cost_func(self, *funcs):
        '''As :meth:`add_operation`, for cost functions (``cost_funcs_external_path``).'''
        self._add_funcs('cost', funcs)

    def add_modifier_func(self, *funcs):
        '''As :meth:`add_operation`, for modifier functions (``modifier_funcs_external_path``);
        each must carry ``@modifier_func``.'''
        self._add_funcs('modifier', funcs)

    def _add_funcs(self, kind, funcs):
        key = FUNCS_SETTINGS[kind]
        path = self._funcs_path(kind)
        named = self._named_funcs_files(key)
        if any(p != path for p in named):
            raise WorkflowError(
                f'workflow "{self.name}" already names {key} {sorted(set(named) - {path})}, so '
                f'add_{"operation" if kind == "operation" else kind + "_func"} cannot add its '
                f'own file: a workflow takes one {kind} functions file. Add these functions to '
                f'that file, or remove the setting.')
        if os.path.isfile(path) and not self._funcs[kind] and \
                read_funcs_file(path, kind) is None:
            raise WorkflowError(f'{path} exists and is not a {kind} functions file; move it, '
                                f'or name it in {key} instead.')
        known = set(self._funcs[kind]) | {f.__name__ for f in funcs}
        for func in funcs:
            source, imports = function_source(func, kind, known)
            self._funcs[kind][func.__name__] = source
            self._imports[kind] = list(dict.fromkeys(self._imports[kind] + imports))
        self.settings[key] = path

    def _funcs_path(self, kind):
        safe = re.sub(r'[^\w.-]+', '_', self.name).strip('_') or 'workflow'
        return os.path.join(self.directory, f'{safe}_{kind}_funcs.py')

    def _named_funcs_files(self, key):
        named = [self.settings[key]] if self.settings.get(key) else []
        named += [s['settings'][key] for s in self._steps
                  if (s.get('settings') or {}).get(key)]
        return named

    def _check_funcs_settings(self):
        for kind, key in FUNCS_SETTINGS.items():
            if self._funcs[kind] and any(p != self._funcs_path(kind)
                                         for p in self._named_funcs_files(key)):
                raise WorkflowError(f'{key} names another file than {self._funcs_path(kind)}, '
                                    f'which holds the functions given to the builder; a '
                                    f'workflow takes one {kind} functions file.')

    def funcs_source(self, kind='operation'):
        '''The text of the ``kind`` functions file :meth:`write` writes (None without any):
        ``utilities.user_funcs_file``'s layout, which CUFLynx writes too.'''
        if not self._funcs[kind]:
            return None
        return funcs_file_source(kind, self._funcs[kind], self._imports[kind])

    # -- out ------------------------------------------------------------------------------

    def to_dict(self):
        '''The workflow as written: the raw dict, its paths relative to ``directory``.'''
        self._check_funcs_settings()
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
        '''The parsed :class:`~libcuflynx.calibration_workflow.spec.Workflow`: every check of
        the file (the schema, and those across steps such as unknown ``fixed_from``).'''
        return parse_workflow(self.to_dict(), path=self.path)

    def write(self):
        '''Write the data files, the functions files and the workflow file; returns the
        workflow file's path. A file whose content is unchanged is not rewritten.'''
        raw = self.to_dict()
        parse_workflow(copy.deepcopy(raw), path=self.path)
        for relative, text in self._data_files()[0].items():
            _write_if_changed(os.path.join(self.directory, relative), text)
        for kind in FUNCS_SETTINGS:
            text = self.funcs_source(kind)
            if text is not None:
                _write_if_changed(self._funcs_path(kind), text)
        _write_if_changed(self.path, json.dumps(raw, indent=1) + '\n')
        return self.path

    def plan(self):
        '''What a run would do (:func:`~libcuflynx.calibration_workflow.plan_workflow`).'''
        from libcuflynx.calibration_workflow.runner import plan_workflow
        return plan_workflow(self.write())

    def run(self, *, only=None, from_step=None, output_dir=None, dry_run=False,
            write_calibrated=False, ranks=1, progress=None):
        '''Write the workflow, then run it as ``cuflynx-calibration-workflow`` would.

        Args:
            only: run just this step, reusing the results it depends on.
            from_step: run this step and every step after it.
            output_dir: default :attr:`output_dir`.
            ranks: MPI ranks. More than one runs ``mpiexec -n <ranks>`` on the command,
                with this Python, and streams its output.
            progress: ``callable(event)`` for each step event; default prints one line per
                event; False prints nothing.

        Returns the workflow result (``workflow_result.json``), or the plan with ``dry_run``.
        '''
        path = self.write()
        output_dir = os.path.abspath(os.fspath(output_dir or self.output_dir))
        if ranks > 1:
            return self._run_mpi(path, output_dir, only, from_step, dry_run, write_calibrated,
                                 ranks)
        from libcuflynx.calibration_workflow.runner import run_calibration_workflow
        if progress is None:
            from libcuflynx.scripts.calibration_workflow_run_script import _print_progress
            progress = _print_progress
        return run_calibration_workflow(path, output_dir=output_dir, from_step=from_step,
                                        only=only, dry_run=dry_run,
                                        write_calibrated=write_calibrated,
                                        progress=progress or None)

    def _run_mpi(self, path, output_dir, only, from_step, dry_run, write_calibrated, ranks):
        from libcuflynx.calibration_workflow.runner import WORKFLOW_RESULT, plan_workflow
        command = ['mpiexec', '-n', str(int(ranks)), sys.executable, '-m',
                   'libcuflynx.scripts.calibration_workflow_run_script', path,
                   '--output-dir', output_dir]
        if only:
            command += ['--only', only]
        if from_step:
            command += ['--from-step', from_step]
        if write_calibrated:
            command.append('--write-calibrated')
        if dry_run:
            return plan_workflow(path)
        started = time.time()
        # A process that has initialised MPI (this one, under a launcher or as a singleton)
        # holds launcher state a new mpiexec must not inherit: give it the environment without.
        env = {k: v for k, v in os.environ.items() if not k.startswith(_LAUNCHER_ENV_PREFIXES)}
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env=env)
        tail = []
        for line in process.stdout:
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
        '''Where each step stands in the run (:func:`workflow_status`): done, and stale when
        its inputs changed since.'''
        from libcuflynx.calibration_workflow.runner import workflow_status
        return workflow_status(self.write(), os.fspath(output_dir or self.output_dir))

    def model(self, view, output_dir=None, work_dir=None):
        '''The model of step ``view``, or of ``"target"``, with the values found so far
        (:func:`workflow_model`), generated under ``work_dir`` (default
        ``<output_dir>_models``).'''
        from libcuflynx.calibration_workflow.runner import workflow_model
        output_dir = os.fspath(output_dir or self.output_dir)
        return workflow_model(self.write(), view, output_dir=output_dir,
                              work_dir=os.fspath(work_dir or output_dir + '_models'))

    def step_dir(self, id, output_dir=None):
        '''The output directory of step ``id``.'''
        from pathlib import Path
        if id not in self.steps:
            raise WorkflowError(f'workflow "{self.name}" has no step "{id}" (steps: '
                                f'{self.steps}).')
        return Path(os.fspath(output_dir or self.output_dir)) / id

    def plot_step(self, id, output_dir=None):
        '''Plot a calibration step's best fit against its obs_data (``plot_outputs`` of the
        param_id run, with the user_inputs the step ran with). Returns the image files
        written.'''
        from libcuflynx.calibration_workflow.resolve import resolve_workflow
        from libcuflynx.calibration_workflow.runner import STEP_RESULT, _get_comm, _Run
        from libcuflynx.calibration_workflow.spec import load_workflow
        output_dir = os.path.abspath(os.fspath(output_dir or self.output_dir))
        result_path = os.path.join(output_dir, id, STEP_RESULT)
        if not os.path.isfile(result_path):
            raise WorkflowError(f'step "{id}" has not run into {output_dir}.')
        with open(result_path) as f:
            result = json.load(f)
        if result.get('kind', 'calibration') != 'calibration':
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
        return _Inline('obs_data', json.dumps(doc, indent=1) + '\n')

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
        return _Inline('params_for_id', params_for_id_file.dump_params_for_id(doc))

    def _data_files(self):
        '''``({path relative to directory: text}, {text: path})`` of the inline inputs; content
        several steps share is written once, named after the first step.'''
        files, by_text = {}, {}
        for step in self._steps:
            for key in ('obs_data', 'params_for_id'):
                value = step.get(key)
                if isinstance(value, _Inline) and value.text not in by_text:
                    relative = f'{DATA_DIR}/{step["id"]}_{key}.json'
                    by_text[value.text] = relative
                    files[relative] = value.text
        return files, by_text

    def _validate_step(self, step):
        '''Check one step against the schema, as a one-step workflow.'''
        from libcuflynx.schemas import CALIBRATION_WORKFLOW_SCHEMA
        from libcuflynx.schemas.validation import SchemaValidationError, validate
        raw = self._rel_step(step, {})
        raw = jsonable({k: (f'{DATA_DIR}/{step["id"]}_{k}.json' if isinstance(v, _Inline)
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
            return os.path.relpath(path, self.directory)
        except ValueError:      # another drive
            return path

    def _abs_model(self, model, base=None):
        if isinstance(model, (str, os.PathLike)):
            return self._abs(model, base)
        if isinstance(model, dict) and 'module_array' in model:
            return {k: (self._abs(v, base) if k in ('module_array', 'parameters') else v)
                    for k, v in model.items()}
        return model

    def _rel_model(self, model):
        if isinstance(model, str):
            return self._rel(model)
        if isinstance(model, dict) and 'module_array' in model:
            return {k: (self._rel(v) if k in ('module_array', 'parameters') else v)
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
        for key in ('obs_data', 'params_for_id'):
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
            elif key in ('obs_data', 'params_for_id') and isinstance(value, str):
                value = self._rel(value)
            elif key == 'model':
                value = self._rel_model(value)
            elif key == 'settings' and step.get('kind') != 'reduction':
                value = self._rel_settings(value)
            out[key] = copy.deepcopy(value)
        return out


def _given(values, model_default=False):
    '''The keyword arguments of an ``add_*`` call that were given, as raw step keys.'''
    out = {}
    for key, value in values.items():
        if key in ('self', 'id', 'model_default'):
            continue
        if value is None or value is False or (isinstance(value, (tuple, list, dict))
                                               and not value and key != 'parameters'):
            continue
        out[key] = list(value) if isinstance(value, tuple) else value
    if model_default and 'model' not in out:
        out['model'] = values['self']._head['target']
    return out


def _write_if_changed(path, text):
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            if f.read() == text:
                return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
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
    decorator_loads = {n.id for dec in node.decorator_list for n in ast.walk(dec)
                       if isinstance(n, ast.Name)}
    used = (_code_names(func.__code__) & loaded) | decorator_loads
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
    if refused:
        why = '; '.join(f'{n!r} {reason}' for n, reason in refused)
        adder = 'add_operation' if kind == 'operation' else f'add_{kind}_func'
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
