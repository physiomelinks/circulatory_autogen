'''
Reading and checking a ``calibration_workflow.json`` (schema:
``libcuflynx/schemas/calibration_workflow.schema.json``).

A workflow is an ordered set of calibrations of module *instances* of a module library, each
against that instance's own ``<instance>_obs_data.json`` and ``<instance>_params_for_id.csv``,
whose results are merged into the parameters of one supermodule instance, the workflow's
``target``. A step may take the calibrated values of earlier steps as fixed parameter values
(``fixed_from``) or, from steps that stored their posterior (``store_distribution``), as
priors (``priors_from``).

Only the file is checked here. Whether its instances exist, and where a step's module sits in
the target, needs the module library: see ``resolve.py``.
'''

import copy
import json
import os
from dataclasses import dataclass, field

from libcuflynx.utilities.config_schemas import load_module_config
from libcuflynx.utilities.module_instances import INSTANCES_DIR, check_instance_name

SCHEMA_VERSION = 1
WORKFLOW_FILE_NAME = 'calibration_workflow.json'

# The user_inputs keys a workflow (or a step) may set. Everything else in a step's
# user_inputs is decided by the workflow: the model, the files, the output directories.
ALLOWED_SETTINGS = (
    'param_id_method', 'optimiser_options', 'debug_optimiser_options',
    'model_type', 'solver', 'solver_info', 'dt', 'pre_time', 'sim_time',
    'DEBUG', 'do_ad', 'UQ_options', 'debug_UQ_options',
    'use_builtin_modules', 'external_modules_dir',
    'operation_funcs_external_path', 'cost_funcs_external_path', 'modifier_funcs_external_path',
)

PRIOR_KINDS = ('mvnormal', 'normal', 'kde')
DEFAULT_PRIOR_KIND = 'mvnormal'

_TOP_KEYS = {'schema_version', 'workflow_name', 'description', 'target', 'module_library_dirs',
             'settings', 'steps', 'comment'}
_STEP_KEYS = {'id', 'target', 'submodule', 'depends_on', 'fixed_from', 'priors_from',
              'store_distribution', 'settings', 'description', 'comment'}
_TARGET_KEYS = {'module_type', 'version', 'instance'}
_PRIOR_KEYS = {'step', 'kind', 'inflate', 'parameters'}


class WorkflowError(ValueError):
    '''A calibration workflow that cannot be run as written.'''


@dataclass(frozen=True)
class Target:
    module_type: str
    version: str
    instance: str

    def label(self):
        return f'{self.module_type}/{self.version}/{self.instance}'

    def as_dict(self):
        return {'module_type': self.module_type, 'version': self.version,
                'instance': self.instance}


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
    target: Target
    submodule: str = None
    depends_on: list = field(default_factory=list)
    fixed_from: list = field(default_factory=list)
    priors_from: list = field(default_factory=list)
    store_distribution: bool = False
    settings: dict = field(default_factory=dict)
    description: str = ''


@dataclass
class Workflow:
    name: str
    target: Target
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
                raise WorkflowError(f'workflow "{self.name}": depends_on has a cycle among '
                                    f'steps {cycle}.')
            step = ready[0]
            order.append(step)
            done.add(step.id)
            remaining.remove(step)
        return order

    def ancestors(self, step_id):
        '''Every step ``step_id`` depends on, directly or not.'''
        seen, stack = set(), list(self.step(step_id).depends_on)
        while stack:
            current = stack.pop()
            if current not in seen:
                seen.add(current)
                stack.extend(self.step(current).depends_on)
        return seen


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
    if not isinstance(raw, dict):
        raise WorkflowError(f'{where}: a calibration workflow is a JSON object.')
    _no_unknown_keys(raw, _TOP_KEYS, where)
    version = raw.get('schema_version', SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise WorkflowError(f'{where}: schema_version {version!r} is not supported by this '
                            f'libcuflynx (it reads version {SCHEMA_VERSION}).')
    name = raw.get('workflow_name')
    if not isinstance(name, str) or not name.strip():
        raise WorkflowError(f'{where}: "workflow_name" must be a non-empty string.')

    base_dir = os.path.dirname(path) if path else os.getcwd()
    library_dirs = [d if os.path.isabs(d) else os.path.normpath(os.path.join(base_dir, d))
                    for d in _string_list(raw, 'module_library_dirs', where)]
    for extra in module_library_dirs or []:
        extra = os.path.abspath(str(extra))
        if extra not in library_dirs:
            library_dirs.append(extra)

    if 'target' in raw:
        target = _target(raw['target'], f'{where}: "target"')
    else:
        target = target_from_location(path)
        if target is None:
            raise WorkflowError(
                f'{where}: no "target". Name the supermodule instance whose parameters the '
                f'workflow calibrates, or keep the file in that instance\'s directory '
                f'(<module_type>/versions/<version>/{INSTANCES_DIR}/<instance>/).')

    settings = _settings(raw.get('settings', {}), f'{where}: "settings"')

    raw_steps = raw.get('steps')
    if not isinstance(raw_steps, list) or not raw_steps:
        raise WorkflowError(f'{where}: "steps" must be a non-empty list.')
    steps = [_step(s, i, where) for i, s in enumerate(raw_steps)]

    workflow = Workflow(name=name.strip(), target=target, steps=steps, settings=settings,
                        module_library_dirs=library_dirs,
                        description=raw.get('description', '') or '', path=path, raw=raw)
    check_workflow(workflow)
    return workflow


def target_from_location(path):
    '''The instance whose directory holds ``path``:
    ``.../<type>/versions/<version>/instances/<instance>/calibration_workflow.json``, with
    the module type and version read from the version's ``*_modules_config.json``.'''
    if not path:
        return None
    instance_dir = os.path.dirname(os.path.abspath(path))
    instances = os.path.dirname(instance_dir)
    if os.path.basename(instances) != INSTANCES_DIR:
        return None
    version_dir = os.path.dirname(instances)
    configs = sorted(f for f in os.listdir(version_dir) if f.endswith('_modules_config.json'))
    for config in configs:
        entries = load_module_config(os.path.join(version_dir, config), include_supermodules=True)
        if len(entries) == 1:
            entry = entries[0]
            return Target(entry['vessel_type'], entry['BC_type'], os.path.basename(instance_dir))
    return None


def _no_unknown_keys(obj, allowed, where):
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise WorkflowError(f'{where}: unknown key(s) {unknown}; allowed: {sorted(allowed)}.')


def _string_list(obj, key, where):
    value = obj.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise WorkflowError(f'{where}: "{key}" must be a list of non-empty strings.')
    return list(value)


def _target(value, where):
    if not isinstance(value, dict):
        raise WorkflowError(f'{where} must be an object with {sorted(_TARGET_KEYS)}.')
    _no_unknown_keys(value, _TARGET_KEYS, where)
    missing = sorted(_TARGET_KEYS - set(value))
    if missing:
        raise WorkflowError(f'{where} is missing {missing}.')
    for key in ('module_type', 'version'):
        if not isinstance(value[key], str) or not value[key].strip():
            raise WorkflowError(f'{where}: "{key}" must be a non-empty string.')
    try:
        instance = check_instance_name(value['instance'], where)
    except ValueError as exc:
        raise WorkflowError(str(exc)) from exc
    return Target(value['module_type'].strip(), value['version'].strip(), instance)


def _settings(value, where):
    if not isinstance(value, dict):
        raise WorkflowError(f'{where} must be an object.')
    unknown = sorted(set(value) - set(ALLOWED_SETTINGS))
    if unknown:
        raise WorkflowError(
            f'{where}: {unknown} cannot be set by a workflow. The model, obs_data, '
            f'params_for_id and output directories of each step are chosen by the workflow '
            f'itself; settings may be {list(ALLOWED_SETTINGS)}.')
    return copy.deepcopy(value)


def _prior_source(value, where):
    if isinstance(value, str):
        return PriorSource(step=value)
    if not isinstance(value, dict):
        raise WorkflowError(f'{where}: a priors_from entry is a step id or an object with '
                            f'{sorted(_PRIOR_KEYS)}.')
    _no_unknown_keys(value, _PRIOR_KEYS, where)
    if not isinstance(value.get('step'), str):
        raise WorkflowError(f'{where}: "step" must name a step.')
    kind = value.get('kind', DEFAULT_PRIOR_KIND)
    if kind not in PRIOR_KINDS:
        raise WorkflowError(f'{where}: prior kind "{kind}" is not one of {list(PRIOR_KINDS)}.')
    inflate = value.get('inflate', 1.0)
    if isinstance(inflate, bool) or not isinstance(inflate, (int, float)) or inflate <= 0:
        raise WorkflowError(f'{where}: "inflate" must be a positive number (a variance '
                            f'multiplier), not {inflate!r}.')
    parameters = value.get('parameters')
    if parameters is not None:
        if not isinstance(parameters, list) or not all(isinstance(p, str) for p in parameters):
            raise WorkflowError(f'{where}: "parameters" must be a list of parameter names.')
        parameters = tuple(parameters)
    return PriorSource(step=value['step'], kind=kind, inflate=float(inflate),
                       parameters=parameters)


def _step(value, index, where):
    label = f'{where}: steps[{index}]'
    if not isinstance(value, dict):
        raise WorkflowError(f'{label} must be an object.')
    _no_unknown_keys(value, _STEP_KEYS, label)
    step_id = value.get('id')
    if not isinstance(step_id, str) or not step_id.strip():
        raise WorkflowError(f'{label}: "id" must be a non-empty string.')
    try:
        check_instance_name(step_id, label, key='id')  # it names an output directory
    except ValueError as exc:
        raise WorkflowError(str(exc)) from exc
    label = f'{where}: step "{step_id}"'
    if 'target' not in value:
        raise WorkflowError(f'{label} has no "target".')
    submodule = value.get('submodule')
    if submodule is not None and (not isinstance(submodule, str) or not submodule.strip()):
        raise WorkflowError(f'{label}: "submodule" must be a non-empty string.')
    store = value.get('store_distribution', False)
    if not isinstance(store, bool):
        raise WorkflowError(f'{label}: "store_distribution" must be true or false.')
    priors = value.get('priors_from', [])
    if not isinstance(priors, list):
        raise WorkflowError(f'{label}: "priors_from" must be a list.')
    return Step(id=step_id.strip(), target=_target(value['target'], f'{label}: "target"'),
                submodule=submodule.strip() if submodule else None,
                depends_on=_string_list(value, 'depends_on', label),
                fixed_from=_string_list(value, 'fixed_from', label),
                priors_from=[_prior_source(p, label) for p in priors],
                store_distribution=store,
                settings=_settings(value.get('settings', {}), f'{label}: "settings"'),
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
    targets = {}
    for step in workflow.steps:
        if step.target in targets:
            raise WorkflowError(
                f'workflow "{workflow.name}": steps "{targets[step.target]}" and "{step.id}" '
                f'both calibrate {step.target.label()}. Each step calibrates its own instance, '
                f'with that instance\'s own obs_data and params_for_id; make a second instance '
                f'for the second calibration.')
        targets[step.target] = step.id
        for dep in step.depends_on:
            if dep not in ids:
                raise WorkflowError(f'step "{step.id}" depends_on unknown step "{dep}".')
            if dep == step.id:
                raise WorkflowError(f'step "{step.id}" depends on itself.')
    workflow.topological_order()  # raises on a cycle

    for step in workflow.steps:
        ancestors = workflow.ancestors(step.id)
        for source in step.fixed_from:
            if source not in ancestors:
                raise WorkflowError(
                    f'step "{step.id}" takes fixed values from "{source}", which it does not '
                    f'depend on. Add "{source}" to its depends_on so it runs first.')
        prior_steps = [p.step for p in step.priors_from]
        for source in prior_steps:
            if source not in ancestors:
                raise WorkflowError(
                    f'step "{step.id}" takes priors from "{source}", which it does not depend '
                    f'on. Add "{source}" to its depends_on so it runs first.')
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
