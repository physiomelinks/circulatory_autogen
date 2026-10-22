'''
Reading and checking a ``calibration_workflow.json`` (schema:
``libcuflynx/schemas/calibration_workflow.schema.json``).

A workflow is an ordered set of steps -- calibrations, and the analyses around them -- whose
calibrated values are merged into one model, the workflow's ``target``. A step's ``kind`` is
one of :data:`STEP_KINDS` (default ``calibration``):

* ``calibration`` -- an ordinary param_id run (with MCMC if ``store_distribution``);
* ``sensitivity`` -- a sensitivity analysis (``sa_options`` in its settings);
* ``reduction`` -- keeps the parameters a Sobol ``sensitivity`` step found most influential
  (``sensitivity_from``), as a params_for_id a calibration takes with ``params_for_id_from``;
* ``emulation`` -- trains an emulator (``emulator_settings``) that later calibration and
  sensitivity steps run on through ``emulator_from``, with the emulation step's model; it may
  be trained on a reduction's parameters (``params_for_id_from``);
* ``identifiability`` -- identifiability analysis (``ia_options``) at a calibration's result
  (``result_from``).

Every kind but ``reduction`` names the three inputs of an ordinary run: a ``model``, an
``obs_data`` file and a ``params_for_id`` file. A model is either

* a module array of CA (``"resources/aorta_module_array.csv"``, or
  ``{"module_array": ..., "parameters": ...}`` when its parameters file is not
  ``<prefix>_parameters.json`` beside it), or
* a module of a module library, ``{"module_type", "version", "parameterisation"}``.

Paths are relative to the workflow file. A step may take the calibrated values of earlier steps
as fixed parameter values (``fixed_from``) or, from steps that stored their posterior
(``store_distribution``), as priors (``priors_from``).

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
    'write_cuflynx_study', 'sa_options', 'ia_options', 'emulator_settings',
)

PRIOR_KINDS = ('mvnormal', 'normal', 'kde')
DEFAULT_PRIOR_KIND = 'mvnormal'

STEP_KINDS = ('calibration', 'sensitivity', 'reduction', 'emulation', 'identifiability')
DEFAULT_KIND = 'calibration'
#: the source keys of a step and the kind of step each must name
SOURCE_KINDS = {'fixed_from': 'calibration', 'emulator_from': 'emulation',
                'result_from': 'calibration', 'sensitivity_from': 'sensitivity',
                'params_for_id_from': 'reduction'}
#: which kinds may use each source key (priors_from and store_distribution: calibration)
_SOURCE_USERS = {'fixed_from': ('calibration', 'sensitivity', 'emulation', 'identifiability'),
                 'emulator_from': ('calibration', 'sensitivity'),
                 'result_from': ('sensitivity', 'identifiability'),
                 'sensitivity_from': ('reduction',),
                 'params_for_id_from': ('calibration', 'emulation')}
#: the analysis options each kind needs in its settings
_KIND_SETTINGS = {'sensitivity': 'sa_options', 'identifiability': 'ia_options'}
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
    model: object                  # Target (a library module) or ModuleArray; None: reduction
    obs_data: str                  # None for a reduction
    params_for_id: str             # None for a reduction, or a step with params_for_id_from
    kind: str = DEFAULT_KIND
    submodule: str = None          # where a library module sits in a supermodule target
    vessel: str = None             # where a library module sits in a module-array target
    vessel_map: dict = field(default_factory=dict)   # module-array step: its vessel -> target's
    depends_on: list = field(default_factory=list)   # fixed_from and priors_from steps
    fixed_from: list = field(default_factory=list)
    priors_from: list = field(default_factory=list)
    store_distribution: bool = False
    emulator_from: str = None      # an emulation step whose emulator (and model) this step uses
    result_from: str = None        # a calibration step whose result this step analyses
    sensitivity_from: str = None   # (reduction) the Sobol step whose indices it reads
    params_for_id_from: str = None  # (calibration, emulation) the reduction whose params it uses
    reduction: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    description: str = ''

    def sources(self):
        '''``{key: [step ids]}`` of every step this one takes something from.'''
        out = {'fixed_from': list(self.fixed_from),
               'priors_from': [p.step for p in self.priors_from]}
        for key in ('emulator_from', 'result_from', 'sensitivity_from', 'params_for_id_from'):
            if getattr(self, key):
                out[key] = [getattr(self, key)]
        return out


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


def parse_workflow(raw, path=None, module_library_dirs=None):
    where = path or 'calibration workflow'
    # The schema owns the file's structure (keys, types, enums, what each kind of step needs);
    # what follows reads it, and checks what spans keys and steps.
    from libcuflynx.schemas import CALIBRATION_WORKFLOW_SCHEMA
    from libcuflynx.schemas.validation import SchemaValidationError, validate
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
    check_workflow(workflow)
    return workflow


def _path(value, base_dir):
    value = os.path.expanduser(value.strip())
    return value if os.path.isabs(value) else os.path.normpath(os.path.join(base_dir, value))


def _model(value, base_dir):
    '''A ``model`` (or the workflow ``target``): a module array path, ``{"module_array",
    "parameters"}``, or a library module ``{"module_type", "version", "parameterisation"}``.'''
    if isinstance(value, str):
        value = {'module_array': value}
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


def _prior_source(value):
    if isinstance(value, str):
        return PriorSource(step=value)
    parameters = value.get('parameters')
    return PriorSource(step=value['step'], kind=value.get('kind', DEFAULT_PRIOR_KIND),
                       inflate=float(value.get('inflate', 1.0)),
                       parameters=tuple(parameters) if parameters is not None else None)


def _step(value, where, base_dir):
    step_id = value['id'].strip()
    label = f'{where}: step "{step_id}"'
    kind = value.get('kind', DEFAULT_KIND)
    refs = {key: (value[key].strip() if value.get(key) else None) for key in
            ('emulator_from', 'result_from', 'sensitivity_from', 'params_for_id_from')}
    fixed_from = list(value.get('fixed_from', []))
    priors_from = [_prior_source(p) for p in value.get('priors_from', [])]
    store = value.get('store_distribution', False)
    for key, users in _SOURCE_USERS.items():
        if dict(refs, fixed_from=fixed_from).get(key) and kind not in users:
            raise WorkflowError(f'{label}: a {kind} step has no "{key}" (it is for '
                                f'{" and ".join(users)} steps).')
    if (priors_from or store) and kind != 'calibration':
        raise WorkflowError(f'{label}: "priors_from" and "store_distribution" are for '
                            f'calibration steps, not a {kind} step.')
    if kind == 'identifiability' and not refs['result_from']:
        raise WorkflowError(f'{label}: an identifiability step needs "result_from", the '
                            f'calibration whose result it analyses.')
    if (kind == 'reduction') != ('reduction' in value):
        raise WorkflowError(f'{label}: "reduction" (the options) is for a reduction step, '
                            f'and a reduction step needs it.')

    if kind == 'reduction':
        present = sorted(k for k in ('model', 'obs_data', 'params_for_id', 'submodule',
                                     'vessel', 'vessel_map', 'settings') if k in value)
        if present:
            raise WorkflowError(f'{label}: a reduction step reads a sensitivity step\'s '
                                f'results and has no {present}.')
        options = value['reduction']
        return Step(id=step_id, model=None, obs_data=None, params_for_id=None, kind=kind,
                    depends_on=[refs['sensitivity_from']],
                    sensitivity_from=refs['sensitivity_from'],
                    reduction={'top_n': options['top_n'],
                               'index_type': options.get('index_type', 'ST'),
                               'criterion': options.get('criterion', 'max'),
                               'threshold': options.get('threshold')},
                    description=value.get('description', '') or '')

    if refs['emulator_from'] and (fixed_from or refs['result_from']):
        raise WorkflowError(f'{label}: a step that uses an emulator ("emulator_from") runs on '
                            f'the emulation step\'s model, with the values that step was given; '
                            f'it has no "fixed_from" or "result_from" of its own.')
    model = _model(value['model'], base_dir)
    submodule, vessel = value.get('submodule'), value.get('vessel')
    if (submodule or vessel) and model.kind != 'library':
        raise WorkflowError(f'{label}: "submodule" and "vessel" place a library module in the '
                            f'target; a module-array model maps its vessels with "vessel_map".')
    vessel_map = value.get('vessel_map', {})
    if vessel_map and model.kind != 'array':
        raise WorkflowError(f'{label}: "vessel_map" is for a module-array model; a library '
                            f'module is placed with "submodule" or "vessel".')
    depends_on = fixed_from + [p.step for p in priors_from] + [r for r in refs.values() if r]
    return Step(id=step_id, model=model, obs_data=_path(value['obs_data'], base_dir),
                params_for_id=(_path(value['params_for_id'], base_dir)
                               if 'params_for_id' in value else None),
                kind=kind,
                submodule=submodule.strip() if submodule else None,
                vessel=vessel.strip() if vessel else None,
                vessel_map={k.strip(): v.strip() for k, v in vessel_map.items()},
                depends_on=list(dict.fromkeys(depends_on)),
                fixed_from=fixed_from, priors_from=priors_from, store_distribution=store,
                emulator_from=refs['emulator_from'], result_from=refs['result_from'],
                params_for_id_from=refs['params_for_id_from'],
                settings=_settings(value.get('settings', {}), base_dir),
                description=value.get('description', '') or '')


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

    for step in workflow.steps:
        for key, sources in step.sources().items():
            wanted = SOURCE_KINDS.get(key)
            for source in sources:
                if wanted and workflow.step(source).kind != wanted:
                    raise WorkflowError(
                        f'step "{step.id}": "{key}" names "{source}", a '
                        f'{workflow.step(source).kind} step; it takes a {wanted} step.')
        if step.emulator_from:
            emulation = workflow.step(step.emulator_from)
            differ = [k for k in ('model', 'obs_data', 'params_for_id', 'params_for_id_from')
                      if getattr(step, k) != getattr(emulation, k)]
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
        if step.params_for_id_from:
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

    for step in workflow.steps:
        prior_steps = [p.step for p in step.priors_from]
        for source in prior_steps:
            if not workflow.step(source).store_distribution:
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
        if step.priors_from and not (step.store_distribution or uses_log_posterior(settings)):
            raise WorkflowError(
                f'step "{step.id}" takes priors, but neither samples a posterior '
                f'("store_distribution": true) nor optimises the log posterior '
                f'(optimiser_options.objective_function: "likelihood"). Its calibration would '
                f'ignore the priors.')
    return workflow


def uses_log_posterior(settings):
    '''Whether the step's optimiser minimises the negative log posterior (priors included)
    rather than the cost. Only the genetic algorithm reads objective_function.'''
    for key in ('optimiser_options', 'debug_optimiser_options'):
        if (settings.get(key) or {}).get('objective_function') == 'likelihood':
            return True
    return False
