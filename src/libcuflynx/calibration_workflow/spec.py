'''
Reading and checking a ``calibration_workflow.json`` (schema:
``libcuflynx/schemas/calibration_workflow.schema.json``).

A workflow is an ordered set of steps -- calibrations, and the analyses around them -- whose
calibrated values are merged into one model, the workflow's ``target``. A step's ``kind`` is
one of :data:`STEP_KINDS` (default ``calibration``):

* ``calibration`` -- an ordinary param_id run (with MCMC if ``store_distribution``);
* ``sensitivity`` -- a sensitivity analysis (``sa_options`` in its settings);
* ``reduction`` -- keeps the parameters a Sobol ``sensitivity`` step found most influential
  (``sensitivity_from``), as a params_for_id a calibration takes with ``params_for_id_from``.
  With ``"method": "manual"`` the choice is made by hand: it keeps the ``parameters`` named,
  for the ``reason`` given, of its own ``params_for_id``, and reads no sensitivity step;
* ``emulation`` -- trains an emulator (``emulator_settings``) that later calibration and
  sensitivity steps run on through ``emulator_from``, with the emulation step's model; it may
  be trained on a reduction's parameters (``params_for_id_from``). A calibration may instead
  take a reduction's parameters on the emulator of the params_for_id they were reduced from:
  it calibrates those and holds the emulator's other parameters at the model's values
  (``Workflow.holds_entries``);
* ``identifiability`` -- identifiability analysis (``ia_options``) at a calibration's or set
  step's values (``result_from``), or at the model's own values;
* ``uq`` -- uncertainty quantification at a point, without a fit of its own: MCMC started there,
  or a Laplace approximation there (``UQ_options.method``). The point is a calibration's or set
  step's values (``result_from``), or the model's own values. It stores the distribution it
  finds, so a later calibration may take it with ``priors_from``;
* ``set`` -- declares values or scales by hand, each with a ``reason``. It runs nothing: later
  steps take its values with ``fixed_from``, and they reach the target as calibrated values do,
  unless it has ``"point": true``: then they are only a point for ``start_from``,
  ``result_from`` or ``fixed_from``, and never reach the target. A point step may also give
  the theta of a params_for_id modifier entry (``{"entry", "theta", "reason"}``), which a
  step with that entry starts or is analysed at;
* ``combine`` -- gives one value per parameter from several calibration, set or combine steps
  (``combine_from``): those of the calibration with the lowest cost (``"method": "best_fit"``)
  or each parameter's mean (``"mean"``). It runs no model. Its values pass on and reach the
  target as a calibration's do, in place of its inputs';
* ``sweep`` -- runs a model across chosen parameter values (``sweep``) and keeps every run. Its
  values go no further: no step takes from it and it does not reach the target. With
  ``"on_target": true`` it runs on the target, given every calibration's and set step's values
Every kind but ``reduction``, ``set``, ``combine``, ``sweep``, ``simulate`` and ``figure`` names
the three inputs of an ordinary run: a ``model``, an ``obs_data`` file and a ``params_for_id``
file. A sweep or simulate step names a ``model``, and an
``obs_data`` file when its protocol or features come from one. A model is either

* a module array of CA (``"resources/aorta_module_array.csv"``, or
  ``{"module_array": ..., "parameters": ...}`` when its parameters file is not
  ``<prefix>_parameters.json`` beside it), or
* a module of a module library, ``{"module_type", "version", "parameterisation"}``, or
* an external Python model, ``{"external_model_path": "model.py"}`` (a ``.py`` defining
  ``SIM_HELPER``, run as ``model_type: external_python``). It is never generated: a step runs
  the file with its values in an ``external_parameters_file``. A step with one needs the same
  file as the target, or
* a CellML model, ``{"cellml_model_path": "model.cellml", "parameters": "values.json"}``
  (``parameters`` optional: a parameters file whose values replace the CellML's). Its parameters
  are its constants, named ``component/variable``. A step runs a flattened copy with its values
  written in, or with ``model_type`` python, casadi_python or aadc_python the Python generated
  from that copy. A step with one needs the same file as the target.

Paths are relative to the workflow file. A step may take the calibrated values of earlier steps
as fixed parameter values (``fixed_from``) or, from steps that stored their posterior
(``store_distribution``, or a ``uq`` step), as priors (``priors_from``). A calibration may start
its own parameters from an earlier step's values (``start_from``). A step may run on an emulator
trained by an emulation step (``emulator_from``) or outside the workflow (``emulator_dir``).

Only the file is checked here. Whether its files and modules exist, and where a step's model sits
in the target, is checked in ``resolve.py``.
'''

import copy
import json
import os
from dataclasses import dataclass, field


WORKFLOW_FILE_NAME = 'calibration_workflow.json'

# The user_inputs keys a workflow (or a step) may set. Everything else in a step's
# user_inputs is decided by the workflow: the model, the files, the output directories.
ALLOWED_SETTINGS = (
    'param_id_method', 'optimiser_options', 'debug_optimiser_options',
    'model_type', 'solver', 'solver_info', 'dt', 'pre_time', 'sim_time',
    'DEBUG', 'do_ad', 'UQ_options', 'debug_UQ_options',
    'use_builtin_modules', 'external_modules_dir',
    'operation_funcs_external_path', 'cost_funcs_external_path', 'modifier_funcs_external_path',
    'write_cuflynx_study', 'cuflynx_study_copy_inputs', 'sa_options', 'ia_options',
    'emulator_settings',
)

PRIOR_KINDS = ('mvnormal', 'normal', 'kde')
DEFAULT_PRIOR_KIND = 'mvnormal'

STEP_KINDS = ('calibration', 'sensitivity', 'reduction', 'emulation', 'identifiability', 'set',
              'sweep', 'uq', 'combine', 'simulate', 'figure')
#: the kinds that run a model at chosen points and keep every run (``Step.sweep``)
SWEEP_KINDS = ('sweep', 'simulate')
DEFAULT_KIND = 'calibration'
#: the kinds that name no model, and read no files of their own (but a reduction chosen by
#: hand, which names the params_for_id it keeps entries of)
_NO_MODEL_KINDS = ('reduction', 'set', 'combine', 'figure')
#: the keys only a figure step has
FIGURE_KEYS = ('function', 'inputs', 'files')
#: the reduction method that is a choice made by hand
MANUAL_REDUCTION = 'manual'
#: how a combine step gives one value per parameter, the first the default
COMBINE_METHODS = ('best_fit', 'mean')
#: the kinds whose values pass on to later steps and reach the target
VALUE_KINDS = ('calibration', 'set', 'combine')
#: what a set step's scale ``of`` names to scale the target's own value
INHERITED_BASE = 'inherited'
#: the source keys of a step and the kinds of step each may name
SOURCE_KINDS = {'fixed_from': VALUE_KINDS, 'emulator_from': ('emulation',),
                'result_from': VALUE_KINDS, 'sensitivity_from': ('sensitivity',),
                'params_for_id_from': ('reduction',), 'scale_from': VALUE_KINDS,
                'start_from': VALUE_KINDS, 'priors_from': ('calibration', 'uq'),
                'combine_from': VALUE_KINDS}
#: which kinds may use each source key (priors_from and store_distribution: calibration)
_SOURCE_USERS = {'fixed_from': ('calibration', 'sensitivity', 'emulation', 'identifiability',
                                'sweep', 'simulate', 'uq'),
                 'emulator_from': ('calibration', 'sensitivity', 'identifiability', 'uq'),
                 'emulator_dir': ('calibration', 'sensitivity', 'identifiability', 'uq'),
                 'result_from': ('sensitivity', 'identifiability', 'sweep', 'simulate', 'uq'),
                 'sensitivity_from': ('reduction',),
                 'params_for_id_from': ('calibration', 'emulation'),
                 'start_from': ('calibration',)}
#: the analysis options each kind needs in its settings
_KIND_SETTINGS = {'sensitivity': 'sa_options', 'identifiability': 'ia_options'}
#: the methods of a uq step (its ``UQ_options.method``), the first the default
UQ_METHODS = ('mcmc', 'laplace')
# a module array file's name is <prefix><suffix>
ARRAY_SUFFIXES = ('_module_array.csv', '_module_array.json', '_vessel_array.csv',
                  '_vessel_array.json')


class WorkflowError(ValueError):
    '''A calibration workflow that cannot be run as written.'''


@dataclass(frozen=True)
class Target:
    '''A model that is a module of a module library: one parameterisation of it.'''
    module_type: str
    version: str
    parameterisation: str

    kind = 'library'

    def label(self):
        return f'{self.module_type}/{self.version}/{self.parameterisation}'

    def as_dict(self):
        return {'module_type': self.module_type, 'version': self.version,
                'parameterisation': self.parameterisation}


LibraryModule = Target


@dataclass(frozen=True)
class ModuleArray:
    '''A model that is a module array of CA, with its parameters file.'''
    module_array: str
    parameters: str

    kind = 'array'

    @property
    def prefix(self):
        return array_prefix(self.module_array)

    def label(self):
        return self.prefix

    def as_dict(self):
        return {'module_array': self.module_array, 'parameters': self.parameters}


@dataclass(frozen=True)
class Selection:
    '''Instances of the workflow's module-array target (``names``: record names, or paths into
    a super instance such as ``heart/lv``), calibrated as they are in the target. The resolver
    gives the model the step runs (``select.resolve_selection``).'''
    names: tuple

    kind = 'selection'

    def label(self):
        return 'selection ' + ', '.join(self.names)

    def as_dict(self):
        return {'select': list(self.names)}


@dataclass(frozen=True)
class ExternalModel:
    '''An external Python model: a ``.py`` file defining ``SIM_HELPER``. Its parameters are
    named ``component/variable``; the component is the vessel.'''
    path: str

    kind = 'external'

    @property
    def prefix(self):
        '''The file prefix: ``heat1d`` for ``heat1d_model.py`` (the default name of an external
        model is ``<file_prefix>_model.py``), else the file's stem.'''
        stem = self.label()
        return stem[:-len('_model')] if stem.endswith('_model') and stem != '_model' else stem

    def label(self):
        return os.path.splitext(os.path.basename(self.path))[0]

    def as_dict(self):
        return {'external_model_path': self.path}


@dataclass(frozen=True)
class CellMLModel:
    '''A CellML model file. Its parameters are its constants, named ``component/variable``; the
    component is the vessel. ``parameters`` (optional) is a parameters file whose values replace
    the CellML's.'''
    path: str
    parameters: str = None

    kind = 'cellml'

    @property
    def prefix(self):
        return self.label()

    def label(self):
        return os.path.splitext(os.path.basename(self.path))[0]

    def as_dict(self):
        out = {'cellml_model_path': self.path}
        if self.parameters:
            out['parameters'] = self.parameters
        return out


#: the model kinds that are one file run as it is: a workflow with one has it as its target and
#: as every step's model
FILE_MODEL_KINDS = ('external', 'cellml')

#: The model types a CellML file model runs as: itself, or the Python generated from it
#: (``generate.generate_cellml``). ``cpp`` has no SimulationHelper, so no step can run it.
CELLML_MODEL_TYPES = ('cellml', 'python', 'casadi_python', 'aadc_python')


def array_prefix(path):
    '''The file prefix of a module array file (``aorta`` for ``aorta_module_array.csv``).'''
    name = os.path.basename(path)
    for suffix in ARRAY_SUFFIXES:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return os.path.splitext(name)[0]


@dataclass(frozen=True)
class PriorSource:
    step: str
    kind: str = DEFAULT_PRIOR_KIND
    inflate: float = 1.0
    parameters: tuple = None   # None: every parameter of the stored distribution

    def as_dict(self):
        out = {'step': self.step, 'kind': self.kind, 'inflate': self.inflate}
        if self.parameters is not None:
            out['parameters'] = list(self.parameters)
        return out


@dataclass
class Step:
    id: str
    model: object                  # Target (a library module), ModuleArray or Selection;
                                   # None: a reduction or a set step
    obs_data: str                  # None for a reduction, a set step, or a sweep without one
    params_for_id: str             # None for a set step, a sweep, a step with
                                   # params_for_id_from, or a reduction not chosen by hand
    kind: str = DEFAULT_KIND
    submodule: str = None          # where a library module sits in a supermodule target
    vessel: str = None             # where a library module sits in a module-array target
    vessel_map: dict = field(default_factory=dict)   # module-array step: its vessel -> target's
    depends_on: list = field(default_factory=list)   # fixed_from and priors_from steps
    fixed_from: list = field(default_factory=list)
    # {fixed_from step id: (model names in that step)}: only those of its values are fixed
    fixed_parameters: dict = field(default_factory=dict)
    priors_from: list = field(default_factory=list)
    store_distribution: bool = False
    emulator_from: str = None      # an emulation step whose emulator (and model) this step uses
    emulator_dir: str = None       # an emulator trained outside the workflow, which it runs on
    result_from: str = None        # a calibration or set step whose values this step analyses
    start_from: str = None         # (calibration) the step its own parameters start from
    sensitivity_from: str = None   # (reduction) the Sobol step whose indices it reads
    params_for_id_from: str = None  # (calibration, emulation) the reduction whose params it uses
    reduction: dict = field(default_factory=dict)
    set_values: list = field(default_factory=list)   # (set) the entries it declares
    # (set) its modifier thetas: [{'entry', 'theta', 'reason'}], a point for the steps whose
    # params_for_id has a modifier entry of that name
    set_thetas: list = field(default_factory=list)
    point: bool = False            # (set) its values are a point only, not merged into the target
    combine_from: list = field(default_factory=list)  # (combine) the steps it combines
    combine: dict = field(default_factory=dict)      # (combine) its options: method
    sweep: dict = field(default_factory=dict)        # (sweep, simulate) design, outputs,
                                                     # parameters, zip
    on_target: bool = False        # (sweep, simulate) runs on the target with the merged values
    figure: dict = field(default_factory=dict)       # (figure) script, function, inputs, files
    settings: dict = field(default_factory=dict)
    description: str = ''

    @property
    def chosen_by_hand(self):
        '''(reduction) its parameters are named in the workflow, not chosen from a
        sensitivity step's results.'''
        return self.kind == 'reduction' and self.reduction.get('method') == MANUAL_REDUCTION

    def sources(self):
        '''``{key: [step ids]}`` of every step this one takes something from.'''
        out = {'fixed_from': list(self.fixed_from),
               'priors_from': [p.step for p in self.priors_from]}
        for key in ('emulator_from', 'result_from', 'sensitivity_from', 'params_for_id_from',
                    'start_from'):
            if getattr(self, key):
                out[key] = [getattr(self, key)]
        scaled = scale_sources(self.set_values)
        if scaled:
            out['scale_from'] = scaled
        if self.combine_from:
            out['combine_from'] = list(self.combine_from)
        if self.figure:
            out['inputs_from'] = list(dict.fromkeys(
                source['step'] for source in self.figure['inputs'].values()))
        return out


def scale_sources(entries):
    '''The steps a set step's ``entries`` scale values of, in order.'''
    return list(dict.fromkeys(entry['of'] for entry in entries
                              if entry.get('of') not in (None, INHERITED_BASE)))


@dataclass
class Workflow:
    name: str
    target: object                 # Target or ModuleArray
    steps: list
    settings: dict = field(default_factory=dict)
    module_library_dirs: list = field(default_factory=list)
    description: str = ''
    path: str = None
    raw: dict = None

    def step(self, step_id):
        for step in self.steps:
            if step.id == step_id:
                return step
        raise WorkflowError(f'workflow "{self.name}" has no step "{step_id}" '
                            f'(steps: {[s.id for s in self.steps]}).')

    def step_settings(self, step):
        '''The workflow's settings with the step's on top (one level of dict merge, so a
        step's optimiser_options add to the workflow's rather than replace them).'''
        merged = copy.deepcopy(self.settings)
        for key, value in step.settings.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **copy.deepcopy(value)}
            else:
                merged[key] = copy.deepcopy(value)
        return merged

    def topological_order(self):
        '''The steps, each after everything it depends on; ties keep file order.'''
        done, order = set(), []
        remaining = list(self.steps)
        while remaining:
            ready = [s for s in remaining if all(d in done for d in s.depends_on)]
            if not ready:
                cycle = [s.id for s in remaining]
                raise WorkflowError(f'workflow "{self.name}": fixed_from/priors_from form a cycle among '
                                    f'steps {cycle}.')
            step = ready[0]
            order.append(step)
            done.add(step.id)
            remaining.remove(step)
        return order

    def ancestors(self):
        '''``{step id: ids of every step it takes from}``, directly or through other steps.'''
        out = {}
        for step in self.topological_order():
            out[step.id] = set(step.depends_on).union(*(out[d] for d in step.depends_on))
        return out

    def holds_entries(self, step):
        '''Whether ``step`` calibrates a reduction's parameters on an emulator trained on the
        params_for_id they were reduced from: the emulator's other parameters are held at
        their values in the emulation step's model.'''
        if step.kind != 'calibration' or not step.emulator_from or not step.params_for_id_from:
            return False
        emulation = self.step(step.emulator_from)
        return (step.params_for_id, step.params_for_id_from) != \
            (emulation.params_for_id, emulation.params_for_id_from)

    def reduced_from(self, reduction):
        '''``(params_for_id, params_for_id_from)`` of the parameters ``reduction`` chooses
        from: its sensitivity step's, or its own params_for_id when chosen by hand.'''
        source = self.step(reduction.sensitivity_from) if reduction.sensitivity_from \
            else reduction
        return source.params_for_id, source.params_for_id_from

    def chosen_at_run(self, step):
        '''Whether what ``step`` calibrates is known only when its reduction has run: it fits
        the choice (``params_for_id_from``) of a reduction that reads a sensitivity step. A
        reduction chosen by hand names its parameters in the workflow.'''
        return bool(step.params_for_id_from) and \
            not self.step(step.params_for_id_from).chosen_by_hand


# --------------------------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------------------------

def load_workflow(source, module_library_dirs=None):
    '''The checked :class:`Workflow` in ``source`` (a path to a calibration_workflow.json, or
    its content as a dict). ``module_library_dirs`` are added after the file's own (which
    are relative to the file).'''
    if isinstance(source, dict):
        raw, path = copy.deepcopy(source), None
    else:
        path = os.path.abspath(str(source))
        try:
            with open(path, encoding='utf-8-sig') as f:
                raw = json.load(f)
        except json.JSONDecodeError as exc:
            raise WorkflowError(f'{path} is not valid JSON: {exc}') from exc
    return parse_workflow(raw, path=path, module_library_dirs=module_library_dirs)


def _refuse_reduction_options_outside_reduction_steps(raw, where):
    '''param_reduction_options belong to a reduction step's own settings, one set per step.'''
    if not isinstance(raw, dict):
        return
    hint = ('param_reduction_options are the options of one reduction step: put them in that '
            'step\'s "settings".')
    if isinstance(raw.get('settings'), dict) and 'param_reduction_options' in raw['settings']:
        raise WorkflowError(f'{where}: the workflow\'s settings have no param_reduction_options; '
                            + hint)
    for step in raw.get('steps') or []:
        if (isinstance(step, dict) and step.get('kind', 'calibration') != 'reduction'
                and 'param_reduction_options' in (step.get('settings') or {})):
            raise WorkflowError(f'{where}: step "{step.get("id")}" is not a reduction step and '
                                f'has no param_reduction_options; ' + hint)


def _refuse_external_figures(raw, where):
    '''A figure is a step of the workflow, a Python function libcuflynx runs.'''
    if isinstance(raw, dict) and 'external_figures' in raw:
        names = [entry.get('id') for entry in raw['external_figures'] or []
                 if isinstance(entry, dict)]
        raise WorkflowError(
            f'{where}: "external_figures" {names} are not read: a figure is a step of the '
            f'workflow. Write each as {{"id": ..., "kind": "figure", "function": '
            f'"<script>.py:<function>", "inputs": {{"<name>": {{"<kind>": "<step>"}}}}}}, a '
            f'Python function that draws from the results of the steps its inputs name (a '
            f'"simulate" step runs a model for it); see "Figure steps" in the '
            f'parameter-identification docs.')


def parse_workflow(raw, path=None, module_library_dirs=None):
    where = path or 'calibration workflow'
    # The schema owns the file's structure (keys, types, enums, what each kind of step needs);
    # what follows reads it, and checks what spans keys and steps.
    from libcuflynx.schemas import CALIBRATION_WORKFLOW_SCHEMA
    from libcuflynx.schemas.validation import SchemaValidationError, validate
    _refuse_reduction_options_outside_reduction_steps(raw, where)
    _refuse_external_figures(raw, where)
    try:
        validate(CALIBRATION_WORKFLOW_SCHEMA, raw, source=where, what='calibration workflow')
    except SchemaValidationError as exc:
        raise WorkflowError(str(exc)) from exc

    base_dir = os.path.dirname(path) if path else os.getcwd()
    library_dirs = [_path(d, base_dir) for d in raw.get('module_library_dirs', [])]
    for extra in module_library_dirs or []:
        extra = os.path.abspath(str(extra))
        if extra not in library_dirs:
            library_dirs.append(extra)
    workflow = Workflow(name=raw['workflow_name'].strip(), target=_model(raw['target'], base_dir),
                        steps=[_step(s, where, base_dir) for s in raw['steps']],
                        settings=_settings(raw.get('settings', {}), base_dir),
                        module_library_dirs=library_dirs,
                        description=raw.get('description', '') or '', path=path, raw=raw)
    _place_target_sweeps(workflow)
    check_workflow(workflow)
    order = [s.id for s in workflow.topological_order()]
    for step in workflow.steps:
        if step.on_target:
            step.fixed_from.sort(key=order.index)
    return workflow


#: the kinds whose values an on_target sweep is given, as the target merges them
TARGET_SOURCE_KINDS = VALUE_KINDS


def _place_target_sweeps(workflow):
    '''An on_target step runs on the target, taking every calibration's, set step's and combine
    step's values (not a point step's, which the target does not merge).'''
    sources = [s.id for s in workflow.steps if s.kind in TARGET_SOURCE_KINDS and not s.point]
    for step in workflow.steps:
        if step.on_target:
            step.model = workflow.target
            step.fixed_from = list(sources)
            step.depends_on = list(sources)


def _path(value, base_dir):
    value = os.path.expanduser(value.strip())
    return value if os.path.isabs(value) else os.path.normpath(os.path.join(base_dir, value))


def _model(value, base_dir):
    '''A ``model`` (or the workflow ``target``): a module array path, ``{"module_array",
    "parameters"}``, a library module ``{"module_type", "version", "parameterisation"}``, or
    (a step's) a selection of the target's instances ``{"select": [...]}``, an external
    Python model ``{"external_model_path"}``, or a CellML model ``{"cellml_model_path",
    "parameters"}``.'''
    if isinstance(value, str):
        value = {'module_array': value}
    if 'cellml_model_path' in value:
        return CellMLModel(_path(value['cellml_model_path'], base_dir),
                           _path(value['parameters'], base_dir) if value.get('parameters')
                           else None)
    if 'external_model_path' in value:
        return ExternalModel(_path(value['external_model_path'], base_dir))
    if 'select' in value:
        return Selection(tuple(name.strip() for name in value['select']))
    if 'module_array' in value:
        array = _path(value['module_array'], base_dir)
        parameters = (_path(value['parameters'], base_dir) if 'parameters' in value else
                      os.path.join(os.path.dirname(array), f'{array_prefix(array)}_parameters.json'))
        return ModuleArray(array, parameters)
    return Target(value['module_type'].strip(), value['version'].strip(),
                  value['parameterisation'].strip())


#: settings that name a file or directory, relative to the workflow file like its other paths
PATH_SETTINGS = ('external_modules_dir', 'operation_funcs_external_path',
                 'cost_funcs_external_path', 'modifier_funcs_external_path')


def _settings(value, base_dir):
    settings = copy.deepcopy(value)
    for key in PATH_SETTINGS:
        if settings.get(key):
            settings[key] = _path(settings[key], base_dir)
    return settings


def _reduction_options(options, label):
    '''The full options of a reduction step (every option its method reads, defaults filled
    in), checked by building the reduction. The paths are the workflow's to decide.'''
    from libcuflynx.sensitivity_analysis.parameter_reduction import (PATH_OPTIONS,
                                                                     ParamReduction)
    paths = sorted(set(options) & set(PATH_OPTIONS))
    if paths:
        raise WorkflowError(f'{label}: a workflow reads the sensitivity_from step\'s results and '
                            f'writes where it decides, so param_reduction_options has no '
                            f'{paths}.')
    try:
        return ParamReduction.from_options(options).as_dict()
    except ValueError as exc:
        raise WorkflowError(f'{label}: {exc}') from exc


def _prior_source(value):
    if isinstance(value, str):
        return PriorSource(step=value)
    parameters = value.get('parameters')
    return PriorSource(step=value['step'], kind=value.get('kind', DEFAULT_PRIOR_KIND),
                       inflate=float(value.get('inflate', 1.0)),
                       parameters=tuple(parameters) if parameters is not None else None)


def _fixed_sources(entries, label):
    '''``(step ids, {step id: names})`` of a ``fixed_from``: an entry is a step id (all its
    values) or ``{"step", "parameters"}`` (only those, named as the source step's model names
    them).'''
    ids, subsets = [], {}
    for entry in entries:
        step_id = entry if isinstance(entry, str) else entry['step']
        if step_id in ids:
            raise WorkflowError(f'{label}: "fixed_from" lists step "{step_id}" twice.')
        ids.append(step_id)
        if not isinstance(entry, str):
            subsets[step_id] = tuple(name.strip() for name in entry['parameters'])
    return ids, subsets


def _step(value, where, base_dir):
    step_id = value['id'].strip()
    label = f'{where}: step "{step_id}"'
    kind = value.get('kind', DEFAULT_KIND)
    refs = {key: (value[key].strip() if value.get(key) else None) for key in
            ('emulator_from', 'result_from', 'sensitivity_from', 'params_for_id_from',
             'start_from')}
    emulator_path = _path(value['emulator_dir'], base_dir) if value.get('emulator_dir') else None
    fixed_from, fixed_parameters = _fixed_sources(value.get('fixed_from', []), label)
    priors_from = [_prior_source(p) for p in value.get('priors_from', [])]
    store = value.get('store_distribution', False)
    for key, users in _SOURCE_USERS.items():
        if dict(refs, fixed_from=fixed_from, emulator_dir=emulator_path).get(key) and \
                kind not in users:
            raise WorkflowError(f'{label}: a {kind} step has no "{key}" (it is for '
                                f'{" and ".join(users)} steps).')
    if (priors_from or store) and kind != 'calibration':
        raise WorkflowError(f'{label}: "priors_from" and "store_distribution" are for '
                            f'calibration steps, not a {kind} step.')
    if emulator_path and refs['emulator_from']:
        raise WorkflowError(f'{label}: "emulator_dir" and "emulator_from" both name the '
                            f'emulator it runs on; give one.')
    if refs['start_from'] and refs['emulator_from']:
        raise WorkflowError(f'{label}: a step on an emulator ("emulator_from") runs on the '
                            f'emulation step\'s model as it was given, so it has no "start_from".')
    # A reduction step's options are settings.param_reduction_options; set, sweep and combine
    # steps carry theirs under their own key.
    if (kind == 'combine') != ('combine_from' in value):
        raise WorkflowError(f'{label}: "combine_from" (the steps combined) is for a combine '
                            f'step, and a combine step needs it.')
    for options_key in ('set', 'sweep', 'combine'):
        if (kind == options_key) != (options_key in value):
            raise WorkflowError(f'{label}: "{options_key}" (the options) is for a {options_key} '
                                f'step, and a {options_key} step needs it.')
    if 'simulate' in value and kind != 'simulate':
        raise WorkflowError(f'{label}: "simulate" (the options) is for a simulate step.')
    if 'reduction' in value:
        raise WorkflowError(f'{label}: a reduction step\'s options go in '
                            f'"settings": {{"param_reduction_options": {{...}}}}, not "reduction".')

    by_hand = kind == 'reduction' and isinstance(value.get('settings'), dict) and (
        value['settings'].get('param_reduction_options') or {}).get('method') == MANUAL_REDUCTION
    if kind in _NO_MODEL_KINDS:
        present = sorted(k for k in ('model', 'obs_data', 'params_for_id', 'submodule',
                                     'vessel', 'vessel_map')
                         if k in value and not (by_hand and k == 'params_for_id'))
        if present:
            reads = {'reduction': 'keeps entries of its params_for_id' if by_hand else
                                  'reads a sensitivity step\'s results',
                     'set': 'declares values for the target',
                     'combine': 'combines the values of other steps',
                     'figure': 'reads the results of the steps its inputs name'}[kind]
            raise WorkflowError(f'{label}: a {kind} step {reads} and has no {present}.')
    if 'point' in value and kind != 'set':
        raise WorkflowError(f'{label}: "point" is for a set step, not a {kind} step.')
    if 'point' in value and not isinstance(value['point'], bool):
        raise WorkflowError(f'{label}: "point" is true or false, not {value["point"]!r}.')
    figure_keys = sorted(k for k in FIGURE_KEYS if k in value)
    if figure_keys and kind != 'figure':
        raise WorkflowError(f'{label}: {figure_keys} are for a figure step, not a {kind} step.')
    if kind == 'figure':
        from libcuflynx.calibration_workflow.figures import parse_figure
        refused = sorted(k for k in ('settings', 'on_target') if k in value)
        if refused:
            raise WorkflowError(f'{label}: a figure step draws from the results of the steps '
                                f'its inputs name and has no {refused}.')
        figure = parse_figure(value, base_dir, label)
        return Step(id=step_id, model=None, obs_data=None, params_for_id=None, kind=kind,
                    depends_on=list(dict.fromkeys(s['step'] for s in figure['inputs'].values())),
                    figure=figure, description=value.get('description', '') or '')
    if kind == 'set':
        if 'settings' in value:
            raise WorkflowError(f'{label}: a set step declares values for the target and has '
                                f'no "settings".')
        entries = [_set_entry(entry) for entry in value['set'] if 'entry' not in entry]
        thetas = [{'entry': entry['entry'].strip(), 'theta': entry['theta'],
                   'reason': entry['reason'].strip()}
                  for entry in value['set'] if 'entry' in entry]
        return Step(id=step_id, model=None, obs_data=None, params_for_id=None, kind=kind,
                    depends_on=scale_sources(entries), set_values=entries, set_thetas=thetas,
                    point=value.get('point', False), description=value.get('description', '') or '')
    if kind == 'combine':
        if 'settings' in value:
            raise WorkflowError(f'{label}: a combine step runs nothing and has no "settings".')
        combine_from = [s.strip() for s in value['combine_from']]
        return Step(id=step_id, model=None, obs_data=None, params_for_id=None, kind=kind,
                    depends_on=list(combine_from), combine_from=combine_from,
                    combine={'method': value['combine'].get('method', COMBINE_METHODS[0])},
                    description=value.get('description', '') or '')
    if kind == 'reduction':
        settings = dict(value.get('settings') or {})
        other = sorted(set(settings) - {'param_reduction_options'})
        if other:
            raise WorkflowError(f'{label}: a reduction step\'s only setting is '
                                f'"param_reduction_options", not {other}.')
        options = dict(settings.get('param_reduction_options') or {})
        if 'param_reduction_options' not in settings:
            raise WorkflowError(f'{label}: a reduction step needs "settings": '
                                f'{{"param_reduction_options": {{...}}}}.')
        reduction = _reduction_options(options, label)
        if by_hand and refs['sensitivity_from']:
            raise WorkflowError(f'{label}: a reduction chosen by hand ("method": "manual") '
                                f'reads no sensitivity step, so it has no "sensitivity_from".')
        if by_hand and 'params_for_id' not in value:
            raise WorkflowError(f'{label}: a reduction chosen by hand ("method": "manual") '
                                f'needs "params_for_id": the file whose entries, with their '
                                f'ranges, it keeps for the parameters it names.')
        if not by_hand and not refs['sensitivity_from']:
            raise WorkflowError(f'{label}: a {reduction["method"]} reduction needs '
                                f'"sensitivity_from", the Sobol sensitivity step it reads.')
        return Step(id=step_id, model=None, obs_data=None, kind=kind,
                    params_for_id=_path(value['params_for_id'], base_dir) if by_hand else None,
                    depends_on=[] if by_hand else [refs['sensitivity_from']],
                    sensitivity_from=refs['sensitivity_from'],
                    reduction=reduction,
                    settings={'param_reduction_options': options},
                    description=value.get('description', '') or '')

    if value.get('on_target'):
        if kind not in SWEEP_KINDS:
            raise WorkflowError(f'{label}: "on_target" is for a sweep or simulate step, not a '
                                f'{kind} step.')
        named = sorted(k for k in ('model', 'fixed_from', 'result_from', 'submodule', 'vessel',
                                   'vessel_map') if k in value)
        if named:
            raise WorkflowError(f'{label}: a {kind} "on_target" runs on the target with every '
                                f'calibration\'s and set step\'s values, so it names no {named}.')
        return Step(id=step_id, model=None,
                    obs_data=_path(value['obs_data'], base_dir) if 'obs_data' in value else None,
                    params_for_id=None, kind=kind, on_target=True, sweep=_step_sweep(value),
                    settings=_settings(value.get('settings', {}), base_dir),
                    description=value.get('description', '') or '')
    if refs['emulator_from'] and (fixed_from or refs['result_from']):
        raise WorkflowError(f'{label}: a step that uses an emulator ("emulator_from") runs on '
                            f'the emulation step\'s model, with the values that step was given; '
                            f'it has no "fixed_from" or "result_from" of its own.')
    model = _model(value['model'], base_dir)
    submodule, vessel = value.get('submodule'), value.get('vessel')
    if kind in SWEEP_KINDS and model.kind == 'selection':
        raise WorkflowError(f'{label}: a {kind} names its model as a library module or a module '
                            f'array, not as a selection of the target.')
    if model.kind == 'selection' and (submodule or vessel or value.get('vessel_map')):
        raise WorkflowError(f'{label}: a selection is calibrated where its instances are in the '
                            f'target, so it has no "submodule", "vessel" or "vessel_map".')
    if (submodule or vessel) and model.kind != 'library':
        raise WorkflowError(f'{label}: "submodule" and "vessel" place a library module in the '
                            f'target; a module-array model maps its vessels with "vessel_map".')
    vessel_map = value.get('vessel_map', {})
    if vessel_map and model.kind != 'array':
        raise WorkflowError(f'{label}: "vessel_map" is for a module-array model; a library '
                            f'module is placed with "submodule" or "vessel".')
    depends_on = fixed_from + [p.step for p in priors_from] + [r for r in refs.values() if r]
    return Step(id=step_id, model=model,
                obs_data=_path(value['obs_data'], base_dir) if 'obs_data' in value else None,
                params_for_id=(_path(value['params_for_id'], base_dir)
                               if 'params_for_id' in value else None),
                kind=kind,
                submodule=submodule.strip() if submodule else None,
                vessel=vessel.strip() if vessel else None,
                vessel_map={k.strip(): v.strip() for k, v in vessel_map.items()},
                depends_on=list(dict.fromkeys(depends_on)),
                fixed_from=fixed_from, fixed_parameters=fixed_parameters,
                priors_from=priors_from, store_distribution=store,
                emulator_from=refs['emulator_from'], result_from=refs['result_from'],
                params_for_id_from=refs['params_for_id_from'],
                start_from=refs['start_from'], emulator_dir=emulator_path,
                sweep=_step_sweep(value) if kind in SWEEP_KINDS else {},
                settings=_settings(value.get('settings', {}), base_dir),
                description=value.get('description', '') or '')


def _set_entry(value):
    '''One entry of a set step: its ``parameter`` and ``reason``, with a ``value`` or a ``scale``
    ``of`` a step or the inherited value.'''
    entry = {'parameter': value['parameter'].strip(), 'reason': value['reason'].strip()}
    for key in ('value', 'scale', 'units'):
        if key in value:
            entry[key] = value[key]
    if 'of' in value:
        entry['of'] = value['of'].strip()
    return entry


def _sweep(value):
    return {'design': value.get('design', 'grid'),
            'outputs': [name.strip() for name in value.get('outputs', [])],
            'parameters': {name.strip(): copy.deepcopy(spec)
                           for name, spec in value['parameters'].items()},
            'zip': [[name.strip() for name in group] for group in value.get('zip', [])]}


def raw_sweep(value):
    '''The sweep a sweep or simulate step of the file runs, in a sweep's keys: a simulate step
    is a sweep of no parameters. None for another kind.'''
    kind = value.get('kind', DEFAULT_KIND)
    if kind == 'sweep':
        return value.get('sweep')
    if kind == 'simulate':
        return {'outputs': list((value.get('simulate') or {}).get('outputs', [])),
                'parameters': {}}
    return None


def _step_sweep(value):
    return _sweep(raw_sweep(value))


# --------------------------------------------------------------------------------------------
# checks that need only the file
# --------------------------------------------------------------------------------------------

def check_workflow(workflow):
    ids = [s.id for s in workflow.steps]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise WorkflowError(f'workflow "{workflow.name}": step ids {duplicates} are used more '
                            f'than once.')
    for step in workflow.steps:
        for dep in step.depends_on:
            if dep not in ids:
                raise WorkflowError(f'step "{step.id}" takes from unknown step "{dep}".')
            if dep == step.id:
                raise WorkflowError(f'step "{step.id}" takes from itself.')
    workflow.topological_order()  # raises on a cycle
    selections = [s.id for s in workflow.steps
                  if s.model is not None and s.model.kind == 'selection']
    if selections and workflow.target.kind != 'array':
        raise WorkflowError(f'steps {selections} select instances of the target, which only a '
                            f'module-array target has; this target is the library module '
                            f'{workflow.target.label()}.')
    _check_external_models(workflow)

    for step in workflow.steps:
        for key, sources in step.sources().items():
            wanted = SOURCE_KINDS.get(key)
            for source in sources:
                if wanted and workflow.step(source).kind not in wanted:
                    raise WorkflowError(
                        f'step "{step.id}": "{key}" names "{source}", a '
                        f'{workflow.step(source).kind} step; it takes a '
                        f'{" or ".join(wanted)} step.')
        if step.emulator_from:
            emulation = workflow.step(step.emulator_from)
            differ = [k for k in ('model', 'obs_data', 'params_for_id', 'params_for_id_from')
                      if getattr(step, k) != getattr(emulation, k)]
            if workflow.holds_entries(step) and not set(differ) & {'model', 'obs_data'}:
                # a reduction's parameters, calibrated on the emulator of all of them
                reduction = workflow.step(step.params_for_id_from)
                if workflow.reduced_from(reduction) != (emulation.params_for_id,
                                                        emulation.params_for_id_from):
                    raise WorkflowError(
                        f'step "{step.id}" calibrates the parameters "{reduction.id}" chose on '
                        f'the emulator of "{emulation.id}", holding that emulator\'s other '
                        f'parameters; "{reduction.id}" must choose from the params_for_id '
                        f'"{emulation.id}" was trained on, and it chooses from another.')
                if step.store_distribution or step.priors_from:
                    raise WorkflowError(
                        f'step "{step.id}" calibrates some of the parameters of the emulator '
                        f'of "{emulation.id}" and holds the others, so it has no '
                        f'"store_distribution" or "priors_from": an MCMC samples every '
                        f'parameter the emulator was trained on.')
                differ = []
            if differ:
                raise WorkflowError(
                    f'step "{step.id}" uses the emulator of "{emulation.id}" but names a '
                    f'different {" and ".join(differ)}. An emulator answers only for what it '
                    f'was trained on: give the same model, obs_data and params_for_id (or '
                    f'params_for_id_from).')
        if step.sensitivity_from:
            method = (workflow.step_settings(workflow.step(step.sensitivity_from))
                      .get('sa_options') or {}).get('method', 'sobol')
            if method != 'sobol':
                raise WorkflowError(
                    f'reduction "{step.id}" reads Sobol indices, but "{step.sensitivity_from}" '
                    f'runs {method!r} sensitivity analysis.')
        if workflow.chosen_at_run(step):
            sensitivity = workflow.step(workflow.step(step.params_for_id_from).sensitivity_from)
            if step.model != sensitivity.model:
                raise WorkflowError(
                    f'step "{step.id}" fits the parameters "{step.params_for_id_from}" chose '
                    f'from "{sensitivity.id}"\'s analysis, which is of a different model; give '
                    f'the same model.')
        needed = _KIND_SETTINGS.get(step.kind)
        if needed and not workflow.step_settings(step).get(needed):
            raise WorkflowError(f'{step.kind} step "{step.id}" needs "{needed}" in its settings '
                                f'(or the workflow\'s).')
        if step.kind == 'uq':
            _check_uq_step(workflow, step)
        if step.kind == 'set':
            _check_set_step(workflow, step)
        if step.kind in SWEEP_KINDS:
            _check_sweep_step(workflow, step)
        if step.kind == 'combine':
            _check_combine_step(workflow, step)
        if step.kind == 'figure':
            from libcuflynx.calibration_workflow.figures import check_figure_step
            check_figure_step(workflow, step)

    for step in workflow.steps:
        prior_steps = [p.step for p in step.priors_from]
        for source in prior_steps:
            if workflow.step(source).kind == 'calibration' and \
                    not workflow.step(source).store_distribution:
                raise WorkflowError(
                    f'step "{step.id}" takes priors from "{source}", but "{source}" does not '
                    f'store its distribution. Set "store_distribution": true on "{source}".')
            if source in step.fixed_from:
                raise WorkflowError(
                    f'step "{step.id}" takes "{source}" both as fixed values and as priors; '
                    f'a parameter is either fixed or calibrated, not both.')
        if len(set(prior_steps)) != len(prior_steps):
            raise WorkflowError(f'step "{step.id}" lists a step in priors_from twice.')
        settings = workflow.step_settings(step)
        if step.store_distribution and not settings.get('UQ_options') and \
                not settings.get('debug_UQ_options'):
            raise WorkflowError(
                f'step "{step.id}" stores its distribution, which is sampled by MCMC after '
                f'the fit, but no "UQ_options" are set for it (at least num_steps and '
                f'num_walkers; see the UQ section of the parameter-identification docs).')
        if step.store_distribution and uq_method(settings) != 'mcmc':
            raise WorkflowError(
                f'step "{step.id}" stores its distribution, which is an MCMC chain, but its '
                f'UQ_options.method is {uq_method(settings)!r}. For a Laplace approximation at '
                f'its fit, add a uq step with "result_from": "{step.id}".')
        if step.priors_from and not (step.store_distribution or uses_log_posterior(settings)):
            raise WorkflowError(
                f'step "{step.id}" takes priors, but neither samples a posterior '
                f'("store_distribution": true) nor optimises the log posterior '
                f'(optimiser_options.objective_function: "likelihood"). Its calibration would '
                f'ignore the priors.')
    return workflow


_FILE_MODEL_WORDS = {'external': ('the external Python model', 'external_model_path'),
                     'cellml': ('the CellML model', 'cellml_model_path')}


def _check_external_models(workflow):
    '''An external Python or a CellML model is run as it is, so its steps calibrate the target
    itself: the target is the same file, and every step that runs a model runs that file. A
    CellML model runs as one of CELLML_MODEL_TYPES.'''
    target = workflow.target
    steps = [s for s in workflow.steps if s.model is not None and not s.on_target]
    for kind, (words, key) in _FILE_MODEL_WORDS.items():
        named = [s.id for s in steps if s.model.kind == kind]
        if named and target.kind != kind:
            article = 'an' if kind == 'external' else 'a'
            raise WorkflowError(f'steps {named} run {article}{words[3:]}, so the '
                                f'target must be that file ({{"{key}": ...}}); this target is '
                                f'{target.label()}.')
    if target.kind not in FILE_MODEL_KINDS:
        return
    words = _FILE_MODEL_WORDS[target.kind][0]
    other = [s.id for s in steps if s.model != target]
    if other:
        raise WorkflowError(f'the target is {words} {target.path}, so every step\'s model is '
                            f'that file; steps {other} name another model.')
    if target.kind == 'cellml':
        for where, settings in [('the workflow', workflow.settings)] + \
                [(f'step "{s.id}"', workflow.step_settings(s)) for s in steps]:
            model_type = settings.get('model_type', 'cellml')
            if model_type not in CELLML_MODEL_TYPES:
                raise WorkflowError(f'{where}: the target is the CellML model {target.path}, '
                                    f'which runs as model_type {", ".join(CELLML_MODEL_TYPES)}'
                                    f'; not {model_type!r}.')


def uq_options(settings):
    '''The UQ_options of a step with these (merged) settings: debug_UQ_options when DEBUG is
    set and they are given.'''
    return (settings.get('debug_UQ_options') if settings.get('DEBUG') else None) or \
        settings.get('UQ_options') or {}


def uq_method(settings):
    '''The method of a uq step with these (merged) settings: ``UQ_options.method``
    (:func:`uq_options`), default mcmc.'''
    return uq_options(settings).get('method', UQ_METHODS[0])


def _check_uq_step(workflow, step):
    settings = workflow.step_settings(step)
    # an empty UQ_options is MCMC with the engine's defaults
    if 'UQ_options' not in settings and 'debug_UQ_options' not in settings:
        raise WorkflowError(f'uq step "{step.id}" needs "UQ_options" in its settings (or the '
                            f'workflow\'s): its method, and for mcmc num_steps and num_walkers; '
                            f'{{}} for MCMC with the defaults.')
    method = uq_method(settings)
    if method not in UQ_METHODS:
        raise WorkflowError(f'uq step "{step.id}": UQ_options.method is {method!r}; one of '
                            f'{list(UQ_METHODS)}.')


def _check_set_step(workflow, step):
    names = [entry['parameter'] for entry in step.set_values]
    twice = sorted({name for name in names if names.count(name) > 1})
    if twice:
        raise WorkflowError(f'set step "{step.id}" sets {twice} more than once.')
    names = [entry['entry'] for entry in step.set_thetas]
    twice = sorted({name for name in names if names.count(name) > 1})
    if twice:
        raise WorkflowError(f'set step "{step.id}" gives the theta of {twice} more than once.')
    if step.set_thetas and not step.point:
        raise WorkflowError(f'set step "{step.id}" gives the theta of {names}: a theta is a '
                            f'point for start_from or result_from, not a value of the target, '
                            f'so the step needs "point": true.')
    if any(entry.get('of') == INHERITED_BASE for entry in step.set_values) and \
            any(s.id == INHERITED_BASE for s in workflow.steps):
        raise WorkflowError(f'set step "{step.id}" scales the value "{INHERITED_BASE}", which is '
                            f'also a step id here; rename that step.')


def cost_type(settings):
    '''The cost function of a calibration with these (merged) settings: its
    ``optimiser_options.cost_type``, ``debug_optimiser_options`` over them when DEBUG is set.'''
    options = dict(settings.get('optimiser_options') or {})
    if settings.get('DEBUG'):
        options.update(settings.get('debug_optimiser_options') or {})
    return options.get('cost_type')


def _check_combine_step(workflow, step):
    '''A combine step names each input once. ``best_fit`` compares costs, which are comparable
    only between calibrations of the same obs_data with the same cost function.'''
    twice = sorted({s for s in step.combine_from if step.combine_from.count(s) > 1})
    if twice:
        raise WorkflowError(f'combine step "{step.id}": "combine_from" lists {twice} more than '
                            f'once.')
    if step.combine['method'] != 'best_fit':
        return
    inputs = [workflow.step(s) for s in step.combine_from]
    others = [f'"{s.id}" (a {s.kind} step)' for s in inputs if s.kind != 'calibration']
    if others:
        raise WorkflowError(f'combine step "{step.id}" takes the best fit of calibrations, which '
                            f'have a cost; {", ".join(others)} has none. Combine calibration '
                            f'steps, or use "method": "mean".')
    for what, of in (('obs_data', lambda s: s.obs_data),
                     ('cost_type', lambda s: cost_type(workflow.step_settings(s)))):
        found = {s.id: of(s) for s in inputs}
        if len(set(found.values())) > 1:
            raise WorkflowError(
                f'combine step "{step.id}" takes the best fit of {list(found)}, but their '
                f'{what} differ ({found}). Costs are comparable only between calibrations of '
                f'the same obs_data with the same cost_type; use "method": "mean", or calibrate '
                f'them against the same data.')


def _check_sweep_step(workflow, step):
    from libcuflynx.calibration_workflow.sweep import check_sweep
    settings = workflow.step_settings(step)
    needed = ['dt'] + ([] if step.obs_data else ['sim_time'])
    missing = [key for key in needed if settings.get(key) is None]
    if missing:
        raise WorkflowError(f'{step.kind} step "{step.id}" needs {missing} in its settings (or '
                            f'the workflow\'s).')
    if not step.obs_data and not step.sweep['outputs']:
        raise WorkflowError(f'{step.kind} step "{step.id}" records nothing: give "outputs", or '
                            f'an "obs_data" with prediction items.')
    check_sweep(step.sweep, f'{step.kind} step "{step.id}"')


def uses_log_posterior(settings):
    '''Whether the step's optimiser minimises the negative log posterior (priors included)
    rather than the cost. Only the genetic algorithm reads objective_function.'''
    for key in ('optimiser_options', 'debug_optimiser_options'):
        if (settings.get(key) or {}).get('objective_function') == 'likelihood':
            return True
    return False
