'''
Finding a workflow's instances in the module library, and where each step's module sits in
the workflow's target supermodule.
'''

import os
from dataclasses import dataclass, field

from libcuflynx.calibration_workflow import naming
from libcuflynx.calibration_workflow.spec import WorkflowError
from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser
from libcuflynx.utilities.config_schemas import is_supermodule_entry
from libcuflynx.utilities.module_instances import instances_dir
from libcuflynx.utilities.module_library import ModuleSources


@dataclass
class ResolvedInstance:
    target: object                 # spec.Target
    entry: dict                    # the normalised module config entry (with config_path)
    instance_dir: str

    @property
    def is_supermodule(self):
        return is_supermodule_entry(self.entry)

    def file(self, suffix):
        return os.path.join(self.instance_dir, f'{self.target.instance}_{suffix}')

    @property
    def parameters_path(self):
        return self.file('parameters.csv')

    @property
    def obs_data_path(self):
        return self.file('obs_data.json')

    @property
    def params_for_id_path(self):
        return self.file('params_for_id.csv')


@dataclass
class CalibratedParameter:
    '''One target of a params_for_id row, as named in the step model.'''
    vessel: str
    param: str

    @property
    def model_name(self):
        return naming.model_name(self.vessel, self.param)


@dataclass
class ResolvedStep:
    step: object                   # spec.Step
    instance: ResolvedInstance
    path: str                      # submodule path in the workflow target ('' = the target)
    parameters: list = field(default_factory=list)   # [CalibratedParameter]

    def target_model_name(self, parameter):
        '''``parameter`` (a CalibratedParameter of this step) named in the target's model.'''
        return naming.model_name(naming.map_vessel(parameter.vessel, self.path), parameter.param)

    def target_instance_name(self, parameter):
        '''``parameter`` named in the target instance's parameters CSV.'''
        return naming.instance_name(naming.map_vessel(parameter.vessel, self.path),
                                    parameter.param)


@dataclass
class ResolvedWorkflow:
    workflow: object               # spec.Workflow
    target: ResolvedInstance
    steps: dict                    # step id -> ResolvedStep
    config_files: list
    library_inputs: dict           # the generation keys every step model is built with

    def path_between(self, inner_id, outer_id):
        '''The submodule path of step ``inner_id``'s module inside step ``outer_id``'s.'''
        inner, outer = self.steps[inner_id], self.steps[outer_id]
        path = naming.relative_path(inner.path, outer.path)
        if path is None:
            raise WorkflowError(
                f'step "{outer_id}" takes values from "{inner_id}", but '
                f'{inner.step.target.module_type} (at "{inner.path or "the target"}" in '
                f'{self.workflow.target.label()}) is not part of '
                f'{outer.step.target.module_type} (at "{outer.path or "the target"}"). Values '
                f'can only be passed into a model that contains the module they were '
                f'calibrated on.')
        return path


def library_inputs(workflow, settings=None):
    '''The generation keys that locate modules: the workflow's library dirs, plus any
    use_builtin_modules / external_modules_dir in its settings.'''
    settings = workflow.settings if settings is None else settings
    inputs = {'module_library_dirs': list(workflow.module_library_dirs),
              'use_builtin_modules': settings.get('use_builtin_modules', True)}
    if settings.get('external_modules_dir'):
        inputs['external_modules_dir'] = settings['external_modules_dir']
    return inputs


def resolve_workflow(workflow):
    '''The workflow's instances and submodule paths, checked against the module library.'''
    if not workflow.module_library_dirs:
        raise WorkflowError(
            f'workflow "{workflow.name}": no module library. Pass module_library_dirs (the '
            f'--module-library-dir option of cuflynx-calibration-workflow) or list them in '
            f'the file\'s "module_library_dirs".')
    inputs = library_inputs(workflow)
    try:
        sources = ModuleSources(inputs)
    except FileNotFoundError as exc:
        raise WorkflowError(str(exc)) from exc
    config_files = sources.config_files
    # through ModuleSources, so a type defined in several sources is the one generation uses
    # (the most specific, with a ModuleShadowWarning), not whichever config file comes last
    components = sources.component_registry()
    supermodules = sources.supermodule_registry()

    target = _resolve_instance(workflow.target, components, supermodules,
                               f'workflow "{workflow.name}" target', need_parameters=True)
    resolved = {}
    claimed = {}
    for step in workflow.steps:
        where = f'step "{step.id}"'
        instance = _resolve_instance(step.target, components, supermodules, where,
                                     need_parameters=False)
        for path, what in ((instance.obs_data_path, 'obs_data'),
                           (instance.params_for_id_path, 'params_for_id')):
            if not os.path.isfile(path):
                raise WorkflowError(
                    f'{where}: {step.target.label()} has no {what} file ({path}). Every step '
                    f'is calibrated against its own instance\'s obs_data and params_for_id.')
        path = _submodule_path(step, workflow.target, target.entry, supermodules)
        parameters = read_calibrated_parameters(instance.params_for_id_path, where)
        resolved_step = ResolvedStep(step=step, instance=instance, path=path,
                                     parameters=parameters)
        for parameter in parameters:
            try:
                name = resolved_step.target_model_name(parameter)
            except ValueError as exc:
                raise WorkflowError(f'{where}: {instance.params_for_id_path}: {exc}') from exc
            if name in claimed:
                raise WorkflowError(
                    f'"{name}" in {workflow.target.label()} is calibrated by both step '
                    f'"{claimed[name]}" and step "{step.id}". Each parameter belongs to one '
                    f'step; fix it in the later one with fixed_from instead.')
            claimed[name] = step.id
        resolved[step.id] = resolved_step

    result = ResolvedWorkflow(workflow=workflow, target=target, steps=resolved,
                              config_files=config_files, library_inputs=inputs)
    for step in workflow.steps:
        for source in step.fixed_from + [p.step for p in step.priors_from]:
            result.path_between(source, step.id)   # raises when not nested
    return result


def read_calibrated_parameters(params_for_id_path, where=''):
    '''Every ``(vessel, param)`` a params_for_id file calibrates (a grouped row gives one per
    target).'''
    try:
        info = ObsAndParamDataParser().get_param_id_info(params_for_id_path)
    except Exception as exc:  # the parser raises a variety of errors for a malformed file
        raise WorkflowError(f'{where}: cannot read {params_for_id_path}: {exc}') from exc
    parameters = []
    for names in info['param_names']:
        for qname in (names if isinstance(names, (list, tuple)) else [names]):
            vessel, _, param = str(qname).partition('/')
            if not param:
                raise WorkflowError(f'{where}: {params_for_id_path}: cannot split "{qname}" '
                                    f'into vessel and parameter.')
            parameters.append(CalibratedParameter(vessel=vessel, param=param))
    return parameters


def _resolve_instance(target, components, supermodules, where, need_parameters):
    key = (target.module_type, target.version)
    entry = supermodules.get(key) or components.get(key)
    if entry is None:
        known = sorted(f'{t}/{v}' for t, v in list(supermodules) + list(components))
        raise WorkflowError(
            f'{where}: no module {target.module_type} version {target.version} in the module '
            f'library (known: {known[:40]}{" ..." if len(known) > 40 else ""}).')
    instance_dir = os.path.join(instances_dir(entry['config_path']), target.instance)
    resolved = ResolvedInstance(target=target, entry=entry, instance_dir=instance_dir)
    if not os.path.isdir(instance_dir):
        raise WorkflowError(f'{where}: {target.label()} has no instance directory '
                            f'{instance_dir}.')
    if need_parameters and not os.path.isfile(resolved.parameters_path):
        raise WorkflowError(f'{where}: {target.label()} has no parameters file '
                            f'{resolved.parameters_path}.')
    return resolved


def submodule_paths(container_entry, key, supermodules, prefix=''):
    '''Every path (``a`` or ``a_b``) at which a module of ``key`` = (type, version) sits in the
    supermodule ``container_entry``, nested supermodules included.'''
    found = []
    for sub in container_entry.get('submodules', []):
        sub_key = (sub['vessel_type'], sub['BC_type'])
        path = prefix + sub['name']
        if sub_key == key:
            found.append(path)
        if sub_key in supermodules:
            found += submodule_paths(supermodules[sub_key], key, supermodules, path + '_')
    return found


def _submodule_path(step, workflow_target, target_entry, supermodules):
    if (step.target.module_type, step.target.version) == \
            (workflow_target.module_type, workflow_target.version):
        if step.submodule:
            raise WorkflowError(f'step "{step.id}" calibrates the target module itself, so it '
                                f'has no "submodule".')
        return ''
    key = (step.target.module_type, step.target.version)
    paths = submodule_paths(target_entry, key, supermodules) if \
        is_supermodule_entry(target_entry) else []
    if step.submodule:
        if step.submodule not in paths:
            raise WorkflowError(
                f'step "{step.id}": "submodule" is "{step.submodule}", but '
                f'{step.target.module_type} {step.target.version} is at {paths or "nowhere"} '
                f'in {workflow_target.label()}.')
        return step.submodule
    if not paths:
        raise WorkflowError(
            f'step "{step.id}": {step.target.module_type} {step.target.version} is not a '
            f'submodule of {workflow_target.module_type} {workflow_target.version}, so its '
            f'calibrated values have nowhere to go.')
    if len(paths) > 1:
        raise WorkflowError(
            f'step "{step.id}": {step.target.module_type} {step.target.version} is at '
            f'{paths} in {workflow_target.label()}; say which with "submodule".')
    return paths[0]
