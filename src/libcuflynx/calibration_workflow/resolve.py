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

import json
import os
import warnings
from dataclasses import dataclass, field

from libcuflynx.calibration_workflow import naming, spec
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


@dataclass
class ResolvedExternal:
    '''An external Python model: its file and the parameters ``SIM_HELPER`` declares
    (``{'component/variable': default}``). Its vessels are the components.'''
    model: object                  # spec.ExternalModel
    parameters: dict

    kind = 'external'

    #: an external model has no parameters file of its own: its values are the class defaults
    parameters_path = None

    def file(self, suffix):
        return os.path.join(os.path.dirname(self.model.path), f'{self.model.prefix}_{suffix}')

    @property
    def vessels(self):
        return {name.partition('/')[0] for name in self.parameters}

    def config_paths(self, components, supermodules):
        return []


@dataclass
class ResolvedCellML:
    '''A CellML model: its file and its constants (``{'component/variable': {value, units}}``),
    with the values of its ``parameters`` file over them. Its vessels are the components.'''
    model: object                  # spec.CellMLModel
    constants: dict

    kind = 'cellml'

    @property
    def parameters(self):
        return {name: c['value'] for name, c in self.constants.items()}

    @property
    def parameters_path(self):
        return self.model.parameters

    def file(self, suffix):
        return os.path.join(os.path.dirname(self.model.path), f'{self.model.prefix}_{suffix}')

    @property
    def vessels(self):
        return {name.partition('/')[0] for name in self.constants}

    def config_paths(self, components, supermodules):
        return []


def read_cellml_parameters(model, where=''):
    '''The constants of the CellML model ``model`` (a spec.CellMLModel) as ResolvedCellML holds
    them, its parameters file's values over the CellML's. A name must be ``component/variable``
    with a plain component, as a workflow names it as vessel and parameter; the parameters file
    sets only constants of the model.'''
    from libcuflynx.calibration_workflow.cellml_files import read_cellml_constants
    from libcuflynx.utilities import parameters_file as pf
    constants = read_cellml_constants(model.path, where)
    for name in constants:
        component = name.partition('/')[0]
        if naming.LEVEL_SEPARATOR in component or naming.is_global(component):
            raise WorkflowError(f'{where}: {model.path} has component "{component}"; a workflow '
                                f'names a CellML model\'s parameters component/variable, with no '
                                f'"{naming.LEVEL_SEPARATOR}" in the component and none named '
                                f'"{naming.GLOBAL_VESSEL}".')
    if model.parameters:
        if not os.path.isfile(model.parameters):
            raise WorkflowError(f'{where}: its parameters file {model.parameters} does not '
                                f'exist.')
        unknown = []
        for place, leaf in pf.leaves(pf.read_parameters_file(model.parameters, kind=pf.MODEL)):
            name = '/'.join(place)
            if name not in constants:
                unknown.append(name)
                continue
            try:
                value = float(leaf.get('value'))
            except (TypeError, ValueError) as exc:
                raise WorkflowError(f'{where}: {model.parameters} gives "{name}" the value '
                                    f'{leaf.get("value")!r}, not a number.') from exc
            constants[name] = dict(constants[name], value=value)
        if unknown:
            raise WorkflowError(f'{where}: {model.parameters} sets {sorted(unknown)}, which are '
                                f'not constants of {model.path} (its constants: '
                                f'{sorted(constants)}).')
    return constants


def read_external_parameters(path, where=''):
    '''``{'component/variable': default}`` of the external model at ``path`` (its
    ``SIM_HELPER.parameters``, checked as the external helper checks them). A name must be
    ``component/variable`` with a plain component, as a workflow names it as vessel and
    parameter.'''
    from libcuflynx.solver_wrappers.external_simulation_helper import (_load_module_from_path,
                                                                       _resolve_user_class)
    if not os.path.isfile(path):
        raise WorkflowError(f'{where}: its external model file {path} does not exist.')
    try:
        _, parameters, _ = _resolve_user_class(_load_module_from_path(path), path)
    except Exception as exc:  # the user's file may raise anything on import
        raise WorkflowError(f'{where}: cannot read the external model {path}: {exc}') from exc
    for name in parameters:
        component, _, variable = name.partition('/')
        if '/' in variable or naming.LEVEL_SEPARATOR in component:
            raise WorkflowError(f'{where}: {path} declares parameter "{name}"; a workflow names '
                                f'an external model\'s parameters component/variable, with no '
                                f'"/" in the variable and no "{naming.LEVEL_SEPARATOR}" in the '
                                f'component.')
    return parameters


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

    def in_model(self, parameter):
        '''``parameter`` (a CalibratedParameter, as a params_for_id names it) as the step's
        model names it: for a CellML model the constant it is (:func:`cellml_parameter`).'''
        if self.model is not None and self.model.kind == 'cellml':
            return cellml_parameter(self.model, parameter, f'step "{self.step.id}"')
        return parameter

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
    #: what is wrong when steps that do not take from each other give the target one parameter
    #: (given_once_problem), or None
    target_overlap: str = None

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


def resolve_workflow(workflow, target_overlap='warn'):
    '''The workflow's models and files, checked, and each step's vessel map into the target.

    ``target_overlap`` says what happens when calibration or combine steps that do not take
    from each other give the target the same parameter (:func:`given_twice`): ``'warn'`` (a
    UserWarning, and the result's ``target_overlap`` holds the message) for reading, planning
    and running one step, ``'error'`` (a WorkflowError) where the target is merged for use.'''
    # a selection's modules are the target's, so it needs the library only as the target does
    library = [s.model for s in workflow.steps
               if s.model is not None and s.model.kind != 'selection'] + [workflow.target]
    if any(m.kind == 'library' for m in library) and not workflow.module_library_dirs:
        raise WorkflowError(
            f'workflow "{workflow.name}" uses a library module but names no module library. '
            f'Pass module_library_dirs (WorkflowBuilder(..., module_library_dirs=[...]) in '
            f'Python, the --module-library-dir option of cuflynx-calibration-workflow) or list '
            f'them in the file\'s "module_library_dirs".')
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
        if step.kind == 'set':
            resolved[step.id] = _resolve_set_step(step, target, claimed)
            continue
        if step.kind == 'combine':
            continue      # after the steps it combines, below
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
        params_for_id = None
        if step.kind != 'sweep':
            params_for_id = step.params_for_id or workflow.step(
                workflow.step(step.params_for_id_from).sensitivity_from).params_for_id
        for path, what in ((step.obs_data, 'obs_data'), (params_for_id, 'params_for_id')):
            if path and not os.path.isfile(path):
                raise WorkflowError(f'{where}: its {what} file {path} does not exist.')
        if selection is None:
            vessel_map = _vessel_map(step, model, workflow.target, target, supermodules)
        parameters = read_calibrated_parameters(params_for_id, where) if params_for_id else []
        if step.kind == 'sweep':
            parameters = _swept_parameters(step, model)
        if model.kind == 'cellml':
            parameters = [cellml_parameter(model, p, f'{where}: {params_for_id or "its sweep"}')
                          for p in parameters]
        if selection is not None and selection.vessel:
            parameters = _in_step_naming(parameters, vessel_map, target.vessels, step)
        resolved_step = ResolvedStep(step=step, model=model, vessel_map=vessel_map,
                                     parameters=parameters, target=target,
                                     selection=selection)
        for parameter in parameters:
            declared = f'{parameter.vessel}/{parameter.param}'
            if model.kind in ('external', 'cellml') and declared not in model.parameters:
                raise WorkflowError(
                    f'{where}: {params_for_id or "its sweep"} names "{declared}", which '
                    f'{_declares(model)}.')
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
            if not step.params_for_id_from:
                # a reduction's candidates are not what it calibrates: the run checks those
                claimed.setdefault(name, []).append(step.id)
        resolved[step.id] = resolved_step
    for step in workflow.topological_order():
        if step.kind == 'combine':
            resolved[step.id] = _resolve_combine_step(workflow, step, resolved, target, claimed)
    resolved = {step.id: resolved[step.id] for step in workflow.steps}
    overlap = check_given_once(workflow, claimed, f'the target {workflow.target.label()}',
                               on_clash=target_overlap)

    result = ResolvedWorkflow(workflow=workflow, target=target, steps=resolved,
                              config_files=sources.config_files, library_inputs=inputs,
                              components=components, supermodules=supermodules,
                              target_overlap=overlap)
    for step in workflow.steps:
        flows = step.fixed_from + [p.step for p in step.priors_from] + \
            [s for s in (step.result_from, step.start_from) if s]
        for source in flows:
            mapped = result.map_between(source, step.id)
            if workflow.step(source).set_thetas and not resolved[source].parameters:
                # a set step of modifier thetas alone names no vessel: the run checks that the
                # step has those modifier entries
                continue
            if not any(mapped(p.vessel) for p in resolved[source].parameters):
                raise WorkflowError(
                    f'step "{step.id}" takes values from "{source}", but none of the vessels '
                    f'"{source}" calibrates is in {step.model.label()} '
                    f'({sorted({p.vessel for p in resolved[source].parameters})}, as named in '
                    f'the target: '
                    f'{sorted({resolved[source].to_target(p.vessel) for p in resolved[source].parameters})}). '
                    f'Values can only be passed into a model that contains what they were '
                    f'calibrated on.')
        for source, names in step.fixed_parameters.items():
            _check_fixed_subset(step, source, names, resolved[source],
                                result.map_between(source, step.id))
    _check_steps_given_once(result)
    return result


def _check_fixed_subset(step, source, names, resolved_source, mapped):
    '''A WorkflowError when ``names``, the values ``step`` fixes from ``source``, are not all
    values ``source`` calibrates or sets that ``step``'s model has.'''
    own = {p.model_name: p for p in resolved_source.parameters}
    unknown = sorted(set(names) - set(own))
    if unknown:
        raise WorkflowError(
            f'step "{step.id}" fixes {unknown} from "{source}", which "{source}" does not '
            f'calibrate or set. Its values, as its model names them: {sorted(own)}.')
    outside = sorted(n for n in names if not mapped(own[n].vessel))
    if outside:
        raise WorkflowError(
            f'step "{step.id}" fixes {outside} from "{source}", but {step.model.label()} has '
            f'no such vessel.')


def split_parameter(name):
    '''``<vessel>/<parameter>`` as a CalibratedParameter; a nested vessel's path (``sys/C/q``) is
    its flat name.'''
    vessel, _, param = name.rpartition('/')
    return CalibratedParameter(vessel=flat_name(tuple(vessel.split('/'))), param=param)


def _resolve_set_step(step, target, claimed):
    '''A set step's parameters, named as the target names them. Its map is the target's own.'''
    resolved = ResolvedStep(step=step, model=None, parameters=[], target=target,
                            vessel_map=naming.Renamed({}, target.vessels))
    for entry in step.set_values:
        parameter = split_parameter(entry['parameter'])
        if target.kind == 'cellml':
            parameter = cellml_parameter(target, parameter, f'set step "{step.id}"')
        if parameter.vessel not in target.vessels and not naming.is_global(parameter.vessel):
            raise WorkflowError(f'set step "{step.id}": "{entry["parameter"]}" names vessel '
                                f'"{parameter.vessel}", which the target does not have (its '
                                f'vessels: {sorted(target.vessels)}).')
        if target.kind == 'external' and entry['parameter'] not in target.parameters:
            raise WorkflowError(f'set step "{step.id}": "{entry["parameter"]}" is not '
                                f'{_declared_by(target)}.')
        resolved.parameters.append(parameter)
        if not step.point:
            # a point step's values never reach the target, so they replace nothing there
            claimed.setdefault(resolved.target_model_name(parameter), []).append(step.id)
    return resolved


def cellml_parameter(model, parameter, where):
    '''``parameter`` (a CalibratedParameter, as a params_for_id, set entry or sweep names it)
    as the constant of the CellML model ``model`` (a ResolvedCellML) it is: its
    ``component/variable``, or the first name the engine would read it as that the model has
    (``calibrated_model.constant_named``: ``global/q_init`` is ``parameters_global/q_init``,
    ``aortic_root/C`` is ``parameters/C_aortic_root``). So a step sets, records and merges the
    constant a direct run of the same file calibrates.'''
    from libcuflynx.param_id.calibrated_model import constant_named
    from libcuflynx.parsers.PrimitiveParsers import model_qname_candidates
    declared = f'{parameter.vessel}/{parameter.param}'
    qname = constant_named(declared, model.constants)
    if qname is None:
        raise WorkflowError(
            f'{where} names "{declared}", which is not a constant of the CellML model '
            f'{model.model.path} under any of the names it is read as '
            f'({model_qname_candidates(declared)}); its constants: {sorted(model.constants)}.')
    component, variable = qname.split('/', 1)
    return CalibratedParameter(vessel=component, param=variable)


def _swept_parameters(step, model):
    parameters = [split_parameter(name) for name in step.sweep['parameters']]
    if model.kind == 'cellml':
        return parameters          # named as cellml_parameter resolves them
    for parameter in parameters:
        if parameter.vessel not in model.vessels and not naming.is_global(parameter.vessel):
            raise WorkflowError(
                f'sweep step "{step.id}" sweeps "{parameter.vessel}/{parameter.param}", but its '
                f'model {step.model.label()} has no such vessel (its vessels: '
                f'{sorted(model.vessels)}).')
    return parameters


def _declares(model):
    '''Why a name is not a parameter of a file model, for an error.'''
    if model.kind == 'cellml':
        return (f'is not a constant of the CellML model {model.model.path} (its constants: '
                f'{sorted(model.parameters)})')
    return (f'the external model {model.model.path} does not declare (SIM_HELPER.parameters: '
            f'{sorted(model.parameters)})')


def _declared_by(model):
    if model.kind == 'cellml':
        return (f'a constant of the CellML model {model.model.path} (its constants: '
                f'{sorted(model.parameters)})')
    return (f'declared by the external model {model.model.path} (SIM_HELPER.parameters: '
            f'{sorted(model.parameters)})')


def _resolve_model(model, components, supermodules, where):
    if model.kind == 'library':
        return _resolve_parameterisation(model, components, supermodules, where)
    if model.kind == 'cellml':
        return ResolvedCellML(model=model, constants=read_cellml_parameters(model, where))
    if model.kind == 'external':
        return ResolvedExternal(model=model, parameters=read_external_parameters(model.path, where))
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
    if model.kind in ('external', 'cellml'):
        return naming.Renamed({}, model.vessels)    # the target is the same file
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


def _resolve_combine_step(workflow, step, resolved, target, claimed):
    '''A combine step's parameters: every parameter its inputs give, named as the target names
    them. Its map is the target's own. It gives the target each of them (the run checks those
    that only a step fitting a reduction's choice gives).'''
    out = ResolvedStep(step=step, model=None, parameters=[], target=target,
                       vessel_map=naming.Renamed({}, target.vessels))
    given = {}      # {target model name: [input ids]}
    for source in step.combine_from:
        rs = resolved[source]
        for parameter in rs.parameters:
            in_target = CalibratedParameter(rs.to_target(parameter.vessel), parameter.param)
            given.setdefault(in_target.model_name, []).append(source)
            if len(given[in_target.model_name]) == 1:
                out.parameters.append(in_target)
    # a calibration fitting a reduction's choice gives what it kept, known when it has run
    known = not any(workflow.step(s).params_for_id_from for s in step.combine_from)
    if step.combine['method'] == 'best_fit' and known:
        partly = {name: ids for name, ids in sorted(given.items())
                  if len(ids) != len(step.combine_from)}
        if partly:
            raise WorkflowError(
                f'combine step "{step.id}" takes the best fit of {step.combine_from}, which '
                f'must calibrate the same parameters, but only some of them calibrate '
                f'{partly}. Use "method": "mean", or give the steps the same params_for_id.')
    for name, ids in given.items():
        if not all(workflow.step(s).params_for_id_from for s in ids):
            claimed.setdefault(name, []).append(step.id)
    return out


def _left(workflow, givers):
    '''The steps of ``givers`` whose value is not replaced by a step that takes from them,
    directly or through other steps, in run order.'''
    ancestors = workflow.ancestors()
    order = [s.id for s in workflow.topological_order()]
    return sorted((g for g in dict.fromkeys(givers)
                   if not any(g in ancestors[other] for other in givers)), key=order.index)


def given_twice(workflow, givers):
    '''The steps of ``givers`` (ids of the steps that give one parameter to one receiver) whose
    values clash, or []. A step's value is replaced by that of a step that takes from it,
    directly or through other steps, and a set step's value never clashes: among set steps and
    unrelated steps the later in run order is used. So the clash is more than one calibration
    or combine step left when the replaced ones are taken out.'''
    left = [g for g in _left(workflow, givers) if workflow.step(g).kind != 'set']
    return left if len(left) > 1 else []


def given_once_problem(workflow, givers_by_name, receiver):
    '''What is wrong, or None, when ``receiver`` (words: the target, or a step) is given the
    parameters of ``givers_by_name`` (``{name: [step ids]}``): the message naming those given by
    steps that clash (:func:`given_twice`).'''
    clashes = {name: given_twice(workflow, ids) for name, ids in sorted(givers_by_name.items())}
    clashes = {name: ids for name, ids in clashes.items() if ids}
    if not clashes:
        return None
    lines = '; '.join(f'"{name}" by {" and ".join(repr(i) for i in ids)}'
                      for name, ids in clashes.items())
    example = next(iter(clashes.values()))
    return (
        f'calibration workflow "{workflow.name}": {receiver} is given {len(clashes)} '
        f'parameter(s) by steps that do not take from each other: {lines}. One value must be '
        f'chosen: let one step take from the other (e.g. "fixed_from" or "start_from"), so the '
        f'later one replaces it, or add a combine step that gives one value from them '
        f'("kind": "combine", "combine_from": {json.dumps(example)}, "combine": {{"method": '
        f'"best_fit"}} or "mean"); its values are used in place of theirs.')


def _warn_set_twice(workflow, givers_by_name, receiver):
    '''One UserWarning for the parameters that set steps which do not take from each other
    both give ``receiver``, naming the step whose value is used.'''
    lines = []
    for name, ids in sorted(givers_by_name.items()):
        left = _left(workflow, ids)
        by_hand = [g for g in left if workflow.step(g).kind == 'set']
        if len(by_hand) > 1:
            lines.append(f'"{name}" by {" and ".join(repr(i) for i in by_hand)} (the value of '
                         f'{left[-1]!r}, the later in run order, is used)')
    if lines:
        warnings.warn(
            f'calibration workflow "{workflow.name}": {receiver} is given {len(lines)} '
            f'parameter(s) by set steps that do not take from each other: ' + '; '.join(lines)
            + '. Set each parameter in one step, or scale one step\'s value in the other '
              '("scale", "of").', UserWarning, stacklevel=4)


def check_given_once(workflow, givers_by_name, receiver, on_clash='error'):
    '''A WorkflowError when ``receiver`` is given a parameter by steps that clash
    (:func:`given_once_problem`), or with ``on_clash='warn'`` a UserWarning that the later in
    run order is used. Either way, a UserWarning for parameters set by hand by unrelated set
    steps. Returns the problem, or None.'''
    _warn_set_twice(workflow, givers_by_name, receiver)
    problem = given_once_problem(workflow, givers_by_name, receiver)
    if problem and on_clash == 'error':
        raise WorkflowError(problem)
    if problem:
        warnings.warn(problem + ' Until then the value of the later step in run order is used; '
                      'a run of every step, and writing the calibrated parameters, are '
                      'refused.', UserWarning, stacklevel=3)
    return problem


def _check_steps_given_once(result):
    '''A WorkflowError when a step is given one parameter by steps that clash: through its
    ``fixed_from`` (a plain id passes on what that step was given too) and ``result_from``. What
    a step fitting a reduction's choice gives is known when it has run, so the run checks it.'''
    workflow, resolved = result.workflow, result.steps
    # a parameter is (vessel, param) in a step's model; a step is not given a parameter it
    # calibrates or analyses itself (a sweep is given the values it sweeps about)
    own = {s.id: {(p.vessel, p.param): p.model_name for p in resolved[s.id].parameters}
           for s in workflow.steps if s.kind != 'sweep'}
    passes = {}     # {step id: {(vessel, param) in the step: [owner ids]}}

    def received(step):
        out = {}
        for source in step.fixed_from + ([step.result_from] if step.result_from else []):
            givers = passes[source]
            if source in step.fixed_parameters:
                givers = {key: [source] for key, name in own[source].items()
                          if name in step.fixed_parameters[source]}
            vessel_of = result.map_between(source, step.id)
            for (vessel, param), owners in givers.items():
                vessel = vessel_of(vessel)
                if vessel is not None:
                    out.setdefault((vessel, param), []).extend(owners)
        return out

    for step in workflow.topological_order():
        if step.on_target:
            continue        # given what the target is, which is checked as the target
        given = {key: owners for key, owners in received(step).items()
                 if key not in own.get(step.id, ())}
        if step.kind in spec.VALUE_KINDS:
            mine = () if step.params_for_id_from else own[step.id]
            passes[step.id] = {**given, **{key: [step.id] for key in mine}}
        check_given_once(workflow, {naming.model_name(*key): owners
                                    for key, owners in given.items()}, f'step "{step.id}"')


def check_merged_once(workflow, claimed, on_clash='error'):
    '''The check :func:`resolve_workflow` leaves to the run for a step fitting a reduction's
    choice: ``claimed`` (``{target name: [step ids]}``, every value the merge took) narrowed to
    the names such a step calibrated. A combine step that has not run yet counts as giving what
    its inputs gave. ``on_clash`` as for :func:`check_given_once`.'''
    claimed = {name: list(ids) for name, ids in claimed.items()}
    for step in workflow.topological_order():
        if step.kind == 'combine':
            for ids in claimed.values():
                if step.id not in ids and set(ids) & set(step.combine_from):
                    ids.append(step.id)
    check_given_once(workflow, {
        name: ids for name, ids in claimed.items()
        if any(workflow.step(i).params_for_id_from for i in ids)},
        f'the target {workflow.target.label()}', on_clash=on_clash)


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
