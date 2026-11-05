'''
Checking a workflow's files and modules, and working out how each step's model is named in the
workflow's target model.

A step's values reach the target, and other steps, through a vessel map (``naming.VesselMap``):

* a library module in a library supermodule target sits at a submodule (``Nested``; the step's
  ``submodule``, inferred when its module type occurs once);
* a library module in a module-array target is one of its records (``Placed``; the step's
  ``vessel``, inferred likewise);
* a module array keeps its vessel names, unless its ``vessel_map`` renames them (``Renamed``);
* a selection of the target's instances is one of these (``select.resolve_selection``).
'''

import os
import warnings
from dataclasses import dataclass, field

from libcuflynx.calibration_workflow import naming
from libcuflynx.calibration_workflow.select import resolve_selection
from libcuflynx.utilities.parameter_names import flat_name
from libcuflynx.calibration_workflow.spec import WorkflowError
from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser
from libcuflynx.utilities.config_schemas import is_supermodule_entry, read_module_array_records
from libcuflynx.utilities.parameterisations import parameterisations_dir
from libcuflynx.utilities.supermodules import submodule_path_parts
from libcuflynx.utilities.module_library import ModuleSources
from libcuflynx.utilities.parameters_file import refuse_csv


@dataclass
class ResolvedParameterisation:
    '''A library module: its config entry and its parameterisation directory.'''
    target: object                 # spec.Target
    entry: dict                    # the normalised module config entry (with config_path)
    parameterisation_dir: str
    paths: tuple = ()              # its submodules, as tuples of names (a supermodule)

    kind = 'library'

    @property
    def model(self):
        return self.target

    @property
    def is_supermodule(self):
        return is_supermodule_entry(self.entry)

    def file(self, suffix):
        return os.path.join(self.parameterisation_dir,
                            f'{self.target.parameterisation}_{suffix}')

    @property
    def parameters_path(self):
        return self.file('parameters.json')

    @property
    def vessels(self):
        '''Its vessels when generated alone as ``mod``.'''
        return {naming.STEP_VESSEL} | {flat_name((naming.STEP_VESSEL,) + tuple(p))
                                       for p in self.paths}

    def config_paths(self, components, supermodules):
        return [self.entry.get('config_path')] + _submodule_configs(self.entry, components,
                                                                    supermodules)


@dataclass
class ResolvedArray:
    '''A module array: its records and files.'''
    model: object                  # spec.ModuleArray
    records: list
    supermodules: dict = field(default_factory=dict)

    kind = 'array'

    @property
    def parameters_path(self):
        return self.model.parameters

    def file(self, suffix):
        return os.path.join(os.path.dirname(self.model.parameters),
                            f'{self.model.prefix}_{suffix}')

    @property
    def vessels(self):
        names = set()
        for record in self.records:
            names.add(record['name'])
            nested = self.supermodules.get((record['module_type'], record['module_subtype']))
            if nested is not None:
                names |= {flat_name((record['name'],) + tuple(p))
                          for p in submodule_path_parts(nested, self.supermodules)}
        return names

    def places(self, key):
        '''The vessels at which a module of ``key`` = (type, version) sits: a record of that
        type, or a submodule of a supermodule record (``<record>__<path>``).'''
        found = []
        for record in self.records:
            record_key = (record['module_type'], record['module_subtype'])
            if record_key == key:
                found.append(record['name'])
            nested = self.supermodules.get(record_key)
            if nested is not None:
                found += [flat_name((record['name'],) + tuple(p))
                          for p in submodule_paths(nested, key, self.supermodules)]
        return found

    def config_paths(self, components, supermodules):
        paths = []
        for record in self.records:
            key = (record['module_type'], record['module_subtype'])
            entry = components.get(key) or supermodules.get(key) or {}
            paths.append(entry.get('config_path'))
            paths += _submodule_configs(entry, components, supermodules)
        return [p for p in paths if p]


def _submodule_configs(entry, components, supermodules):
    paths = []
    for sub in entry.get('submodules', []):
        key = (sub['module_type'], sub['module_subtype'])
        nested = components.get(key) or supermodules.get(key) or {}
        paths.append(nested.get('config_path'))
        paths += _submodule_configs(nested, components, supermodules)
    return [p for p in paths if p]


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
    model: object                  # ResolvedParameterisation or ResolvedArray
    vessel_map: object             # naming.VesselMap
    parameters: list = field(default_factory=list)   # [CalibratedParameter]
    target: object = None          # the workflow target, resolved
    selection: object = None       # select.ResolvedSelection of a selection step

    @property
    def base_rows(self):
        '''The parameter rows the step's model starts from, under any fixed values: the
        target's values for a selection, none otherwise.'''
        return list(self.selection.rows) if self.selection else []

    @property
    def renames_inputs(self):
        '''Whether the step's obs_data and params_for_id name vessels as the target does while
        its model names them otherwise (a selection of one instance, generated as ``mod``).'''
        return bool(self.selection and self.selection.vessel)

    @property
    def obs_data_path(self):
        return self.step.obs_data

    @property
    def params_for_id_path(self):
        return self.step.params_for_id

    @property
    def place(self):
        '''Where the step's model sits in the target, for people: a submodule path, a
        vessel, or '' (the target itself, or a module array named as the target is).'''
        vmap = self.vessel_map
        if isinstance(vmap, naming.Nested):
            return vmap.path
        if isinstance(vmap, naming.Placed):
            return vmap.vessel
        return ''

    def to_target(self, vessel):
        return self.vessel_map.to_target(vessel)

    def target_model_name(self, parameter):
        '''``parameter`` (a CalibratedParameter of this step) named in the target's model.'''
        return naming.model_name(self.to_target(parameter.vessel), parameter.param)

    def target_parameterisation_name(self, parameter):
        '''``parameter`` named as a row of the target's parameters file.'''
        vessel = self.to_target(parameter.vessel)
        if self.target.kind == 'library':
            return naming.parameterisation_name(vessel, parameter.param, self.target.paths)
        return parameter.param if naming.is_global(vessel) else f'{vessel}/{parameter.param}'


@dataclass
class ResolvedWorkflow:
    workflow: object               # spec.Workflow
    target: object                 # ResolvedParameterisation or ResolvedArray
    steps: dict                    # step id -> ResolvedStep
    config_files: list
    library_inputs: dict           # the generation keys every model is built with
    components: dict = field(default_factory=dict)
    supermodules: dict = field(default_factory=dict)

    def map_between(self, source_id, receiver_id):
        '''``vessel -> vessel``: a vessel of step ``source_id``'s model named in step
        ``receiver_id``'s model, None where that model has no such vessel.'''
        source, receiver = self.steps[source_id].vessel_map, self.steps[receiver_id].vessel_map
        return lambda vessel: naming.between(source, receiver, vessel)

    def module_configs(self):
        '''The module config files of every module in the target.'''
        return self.target.config_paths(self.components, self.supermodules)


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
    '''The workflow's models and files, checked, and each step's vessel map into the target.'''
    # a selection's modules are the target's, so it needs the library only as the target does
    library = [s.model for s in workflow.steps
               if s.model is not None and s.model.kind != 'selection'] + [workflow.target]
    if any(m.kind == 'library' for m in library) and not workflow.module_library_dirs:
        raise WorkflowError(
            f'workflow "{workflow.name}" uses a library module but names no module library. '
            f'Pass module_library_dirs (the --module-library-dir option of '
            f'cuflynx-calibration-workflow) or list them in the file\'s "module_library_dirs".')
    inputs = library_inputs(workflow)
    try:
        sources = ModuleSources(inputs)
    except FileNotFoundError as exc:
        raise WorkflowError(str(exc)) from exc
    # through ModuleSources, so a type defined in several sources is the one generation uses
    # (the most specific, with a ModuleShadowWarning), not whichever config file comes last
    components = sources.component_registry()
    supermodules = sources.supermodule_registry()

    target = _resolve_model(workflow.target, components, supermodules,
                            f'workflow "{workflow.name}" target')
    resolved = {}
    claimed = {}
    for step in workflow.steps:
        where = f'step "{step.id}"'
        if step.kind == 'reduction':
            # reads a sensitivity step's results; it has no model and calibrates nothing
            resolved[step.id] = ResolvedStep(step=step, model=None, vessel_map=None,
                                             parameters=[], target=target)
            continue
        selection = None
        if step.model.kind == 'selection':
            selection = resolve_selection(step, target, components, supermodules)
            if selection.vessel:
                model = _resolve_model(selection.model, components, supermodules, where)
                vessel_map = naming.Placed(selection.vessel, model.vessels)
            else:
                model = ResolvedArray(model=selection.model, records=selection.model.records,
                                      supermodules=supermodules)
                vessel_map = naming.Renamed({}, model.vessels)
        else:
            model = _resolve_model(step.model, components, supermodules, where)
        # a calibration fitting a reduction's choice is checked against every candidate: the
        # params_for_id of the sensitivity step the reduction reads
        params_for_id = step.params_for_id or workflow.step(
            workflow.step(step.params_for_id_from).sensitivity_from).params_for_id
        for path, what in ((step.obs_data, 'obs_data'), (params_for_id, 'params_for_id')):
            if not os.path.isfile(path):
                raise WorkflowError(f'{where}: its {what} file {path} does not exist.')
        if selection is None:
            vessel_map = _vessel_map(step, model, workflow.target, target, supermodules)
        parameters = read_calibrated_parameters(params_for_id, where)
        if selection is not None and selection.vessel:
            parameters = _in_step_naming(parameters, vessel_map, target.vessels, step)
        resolved_step = ResolvedStep(step=step, model=model, vessel_map=vessel_map,
                                     parameters=parameters, target=target,
                                     selection=selection)
        for parameter in parameters:
            if parameter.vessel not in model.vessels and not naming.is_global(parameter.vessel):
                raise WorkflowError(
                    f'{where}: {params_for_id} names vessel "{parameter.vessel}", which '
                    f'its model {step.model.label()} does not have (its vessels: '
                    f'{sorted(model.vessels)}).')
            if step.kind != 'calibration':
                continue     # only a calibration's values are merged into the target
            try:
                name = resolved_step.target_model_name(parameter)
            except ValueError as exc:
                raise WorkflowError(f'{where}: {step.params_for_id}: {exc}') from exc
            landing = resolved_step.to_target(parameter.vessel)
            if not naming.is_global(landing) and landing not in target.vessels:
                raise WorkflowError(
                    f'{where}: "{parameter.vessel}/{parameter.param}" lands on vessel '
                    f'"{landing}" of the target {workflow.target.label()}, which has no such '
                    f'vessel (its vessels: {sorted(target.vessels)}). Map it with '
                    f'"vessel_map" (a module array) or "vessel"/"submodule" (a library module).')
            claimed.setdefault(name, []).append(step.id)
        resolved[step.id] = resolved_step
    _warn_recalibrated(workflow, {n: ids for n, ids in claimed.items() if len(ids) > 1})

    result = ResolvedWorkflow(workflow=workflow, target=target, steps=resolved,
                              config_files=sources.config_files, library_inputs=inputs,
                              components=components, supermodules=supermodules)
    for step in workflow.steps:
        flows = step.fixed_from + [p.step for p in step.priors_from] + \
            ([step.result_from] if step.result_from else [])
        for source in flows:
            mapped = result.map_between(source, step.id)
            if not any(mapped(p.vessel) for p in resolved[source].parameters):
                raise WorkflowError(
                    f'step "{step.id}" takes values from "{source}", but none of the vessels '
                    f'"{source}" calibrates is in {step.model.label()} '
                    f'({sorted({p.vessel for p in resolved[source].parameters})}, as named in '
                    f'the target: '
                    f'{sorted({resolved[source].to_target(p.vessel) for p in resolved[source].parameters})}). '
                    f'Values can only be passed into a model that contains what they were '
                    f'calibrated on.')
    return result


def _resolve_model(model, components, supermodules, where):
    if model.kind == 'library':
        return _resolve_parameterisation(model, components, supermodules, where)
    try:
        refuse_csv(model.parameters, f'{where}: ')
    except ValueError as exc:
        raise WorkflowError(str(exc)) from exc
    for path, what in ((model.module_array, 'module array'), (model.parameters, 'parameters')):
        if not os.path.isfile(path):
            raise WorkflowError(f'{where}: its {what} file {path} does not exist.')
    try:
        records = read_module_array_records(model.module_array)
    except Exception as exc:  # the readers raise a variety of errors for a malformed file
        raise WorkflowError(f'{where}: cannot read {model.module_array}: {exc}') from exc
    return ResolvedArray(model=model, records=records, supermodules=supermodules)


def _vessel_map(step, model, workflow_target, target, supermodules):
    '''How the step's model is named in the target (see the module docstring).'''
    if model.kind == 'array':
        mapping = dict(step.vessel_map)
        if target.kind == 'library':
            # a library target is generated as ``mod``: name its submodules by their path
            mapping = {k: _library_vessel(v, target, step) for k, v in mapping.items()}
        return naming.Renamed(mapping, model.vessels)
    if target.kind == 'library':
        if step.vessel:
            raise WorkflowError(f'step "{step.id}": "vessel" places a library module in a '
                                f'module-array target; in a supermodule target it is '
                                f'"submodule".')
        path = _submodule_path(step, workflow_target, target.entry, supermodules)
        return naming.Nested(path, model.vessels)
    if step.submodule:
        raise WorkflowError(f'step "{step.id}": "submodule" places a library module in a '
                            f'supermodule target; in a module-array target it is "vessel".')
    key = (step.model.module_type, step.model.version)
    places = target.places(key)
    if step.vessel:
        if step.vessel not in places:
            raise WorkflowError(
                f'step "{step.id}": "vessel" is "{step.vessel}", but {step.model.module_type} '
                f'{step.model.version} is at {places or "no vessel"} in '
                f'{workflow_target.label()}.')
        return naming.Placed(step.vessel, model.vessels)
    if len(places) != 1:
        raise WorkflowError(
            f'step "{step.id}": {step.model.module_type} {step.model.version} is at '
            f'{places or "no vessel"} in {workflow_target.label()}; '
            + ('say which with "vessel".' if places else 'its values have nowhere to go.'))
    return naming.Placed(places[0], model.vessels)


def _library_vessel(path, target, step):
    '''A vessel_map value for a library supermodule target: a submodule path (``A``,
    ``soma/i_M``) as the target names it when generated as ``mod``.'''
    parts = tuple(path.split('/')) if '/' in path else None
    for p in target.paths:
        if p == parts or flat_name(p) == path:
            return flat_name((naming.STEP_VESSEL,) + tuple(p))
    raise WorkflowError(f'step "{step.id}": vessel_map maps to "{path}", which is not a '
                        f'submodule of the target (its submodules: '
                        f'{sorted("/".join(p) for p in target.paths)}).')


def _warn_recalibrated(workflow, claimed):
    '''One UserWarning for the target parameters ``{name: [step ids]}`` that several steps
    calibrate. The merge takes them in run order, so the last step's value is used.'''
    if not claimed:
        return
    order = [s.id for s in workflow.topological_order()]
    lines = []
    for name, ids in sorted(claimed.items()):
        ids = sorted(ids, key=order.index)
        lines.append(f'"{name}" by {", then ".join(repr(i) for i in ids)} '
                     f'(the value of {ids[-1]!r} is used)')
    warnings.warn(
        f'calibration workflow "{workflow.name}": {len(lines)} parameter(s) of '
        f'{workflow.target.label()} are calibrated by more than one step, and the later '
        f'calibration replaces the earlier: ' + '; '.join(lines) + '. To update an earlier '
        f'calibration with new data rather than replace it, calibrate Bayesianly: remove the '
        f'parameter from the later step\'s params_for_id and list the earlier step in its '
        f'"priors_from" (with "store_distribution": true on the earlier step), so the earlier '
        f'posterior becomes the later step\'s prior.', UserWarning, stacklevel=3)


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


def _in_step_naming(parameters, vessel_map, target_vessels, step):
    '''A selection's ``parameters``, written with the target's vessels, named in its model.'''
    out = []
    for parameter in parameters:
        vessel = parameter.vessel
        if vessel in target_vessels:
            vessel = vessel_map.from_target(vessel)
            if vessel is None:
                raise WorkflowError(
                    f'step "{step.id}": {step.params_for_id} calibrates '
                    f'"{parameter.vessel}/{parameter.param}", which is outside its selection '
                    f'{", ".join(step.model.names)}.')
        out.append(CalibratedParameter(vessel=vessel, param=parameter.param))
    return out


def _resolve_parameterisation(target, components, supermodules, where):
    key = (target.module_type, target.version)
    entry = supermodules.get(key) or components.get(key)
    if entry is None:
        known = sorted(f'{t}/{v}' for t, v in list(supermodules) + list(components))
        raise WorkflowError(
            f'{where}: no module {target.module_type} version {target.version} in the module '
            f'library (known: {known[:40]}{" ..." if len(known) > 40 else ""}).')
    parameterisation_dir = os.path.join(parameterisations_dir(entry['config_path']),
                                        target.parameterisation)
    if not os.path.isdir(parameterisation_dir):
        raise WorkflowError(f'{where}: {target.label()} has no parameterisation directory '
                            f'{parameterisation_dir}.')
    paths = tuple(submodule_path_parts(entry, supermodules)) if is_supermodule_entry(entry) \
        else ()
    return ResolvedParameterisation(target=target, entry=entry,
                                    parameterisation_dir=parameterisation_dir, paths=paths)


def submodule_paths(container_entry, key, supermodules, prefix=()):
    '''Every path (``('a',)`` or ``('a', 'b')``) at which a module of ``key`` = (type, version)
    sits in the supermodule ``container_entry``, nested supermodules included.'''
    found = []
    for sub in container_entry.get('submodules', []):
        sub_key = (sub['module_type'], sub['module_subtype'])
        path = prefix + (sub['name'],)
        if sub_key == key:
            found.append(path)
        if sub_key in supermodules:
            found += submodule_paths(supermodules[sub_key], key, supermodules, path)
    return found


def _matches(submodule, path):
    '''Whether a step's ``submodule`` (``soma/i_M``, or ``soma__i_M``) names ``path``.'''
    if '/' in submodule:
        return tuple(submodule.split('/')) == path
    return submodule == flat_name(path)


def _submodule_path(step, workflow_target, target_entry, supermodules):
    '''Where the step's module sits in the target, as the joined name its expanded vessel
    carries (``i_M``, ``soma__i_M``); '' for the target itself.'''
    if (step.model.module_type, step.model.version) == \
            (workflow_target.module_type, workflow_target.version):
        if step.submodule:
            raise WorkflowError(f'step "{step.id}" calibrates the target module itself, so it '
                                f'has no "submodule".')
        return ''
    key = (step.model.module_type, step.model.version)
    paths = submodule_paths(target_entry, key, supermodules) if \
        is_supermodule_entry(target_entry) else []
    shown = ['/'.join(p) for p in paths] or 'nowhere'
    if step.submodule:
        chosen = [p for p in paths if _matches(step.submodule, p)]
        if len(chosen) != 1:
            raise WorkflowError(
                f'step "{step.id}": "submodule" is "{step.submodule}", but '
                f'{step.model.module_type} {step.model.version} is at {shown} '
                f'in {workflow_target.label()}.')
        return flat_name(chosen[0])
    if not paths:
        raise WorkflowError(
            f'step "{step.id}": {step.model.module_type} {step.model.version} is not a '
            f'submodule of {workflow_target.module_type} {workflow_target.version}, so its '
            f'calibrated values have nowhere to go.')
    if len(paths) > 1:
        raise WorkflowError(
            f'step "{step.id}": {step.model.module_type} {step.model.version} is at '
            f'{shown} in {workflow_target.label()}; say which with "submodule".')
    return flat_name(paths[0])
