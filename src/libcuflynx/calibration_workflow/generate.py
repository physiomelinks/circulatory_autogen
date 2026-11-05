'''
Generating a workflow model -- a library module on its own, or a module array -- with some of
its parameters overridden.
'''

import json
import os
import shutil

from libcuflynx.calibration_workflow.naming import STEP_VESSEL, host_model_name
from libcuflynx.utilities import parameters_file

# the value generation writes for a parameter nothing gave one
UNSET_VALUE = 'EMPTY_MUST_BE_FILLED'


class UnsetModelParameters(RuntimeError):
    '''Generation left parameters without a value: ``model_names`` (``x_C``).'''

    def __init__(self, label, model_names):
        self.model_names = list(model_names)
        super().__init__(f'generating {label}: no value for {", ".join(self.model_names)}.')


_MODEL_EXTENSIONS = {'cellml': '.cellml', 'python': '.py', 'casadi_python': '.py',
                     'aadc_python': '.py', 'cpp': '.cpp'}


def read_parameter_rows(path):
    '''The rows of a generated model's parameters file (``<prefix>_parameters.json`` beside the
    generated model), named as the model names them (``parameters_file.model_parameter_rows``).'''
    return parameters_file.model_parameter_rows(path)


def with_overrides(doc, overrides, where=''):
    '''The parameters document ``doc`` with each row of ``overrides`` (``variable_name`` as a
    model's parameters file names it, ``C/q`` or ``mod__A/p``) set over what it has.'''
    for row in overrides:
        parameters_file.set_leaf(doc, parameters_file.name_path(row['variable_name'], where=where),
                                 parameters_file.leaf_from_row(row), where)
    return doc


def generate_parameterisation(target, work_dir, file_prefix, overrides=(), library_inputs=None,
                              model_type='cellml', solver=None, vessel=STEP_VESSEL,
                              protocol=None, instance_values=None):
    '''Generate the module parameterisation ``target`` (a spec.Target) alone, as a single
    module-array record named ``vessel``, under ``work_dir``. A supermodule's record gets
    ``instance_values`` (``{sub: {var: parameter}}``) as its values in the model's parameters
    file, or a new instance's, from its version (``supermodules.instance_values_from_version``).

    ``overrides`` are parameter rows (``variable_name`` as a model's parameters file names it,
    ``mod__A/p``, ``units``, ``value``, ``data_reference``) set in the model's parameters file,
    which wins over every parameterisation and supermodule default. ``library_inputs`` are the
    generation keys that locate the modules (``module_library_dirs`` etc.).

    Returns ``{'model_path', 'flat_model_path', 'parameters_path', 'resources_dir',
    'generated_models_dir', 'inp_data_dict'}`` (``flat_model_path``: the CellML with its
    imports resolved, None for other model types; ``parameters_path``: every value the model
    uses, the generated ``<prefix>_parameters.json``); raises RuntimeError when generation
    fails. Not MPI-aware: call it on one rank.

    ``protocol`` (``{qname: value}``, as :func:`select.protocol_values` reads an obs_data)
    gives the parameters generation leaves without a value: they are set in the model's
    parameters file and the model generated again. Parameters still without one raise
    :class:`UnsetModelParameters`.
    '''
    work_dir = os.path.abspath(work_dir)
    resources_dir = os.path.join(work_dir, 'resources')
    os.makedirs(resources_dir, exist_ok=True)
    record = {'name': vessel, 'module_type': target.module_type,
              'module_subtype': target.version, 'parameterisation': target.parameterisation,
              'inp_instances': [], 'out_instances': []}
    array = os.path.join(resources_dir, f'{file_prefix}_module_array.json')
    with open(array, 'w') as f:
        json.dump([record], f, indent=1)
    path = os.path.join(resources_dir, parameters_file.parameters_file_name(file_prefix))
    doc = parameters_file.new_document()
    values = _instance_values(record, library_inputs, instance_values)
    if values is not None:
        doc[vessel] = values
    parameters_file.write_parameters_file(path, with_overrides(doc, overrides, f'{path}: '))
    return _generate(work_dir, resources_dir, file_prefix, model_type, solver, library_inputs,
                     target.label(), protocol)


def _instance_values(record, library_inputs, instance_values=None):
    '''The values of the module-array ``record`` in its model's parameters file when it is a
    supermodule instance (``instance_values`` if given, else a new instance's, from its version);
    None for a component.'''
    from libcuflynx.utilities.module_library import ModuleSources
    from libcuflynx.utilities.supermodules import instance_values_from_version
    registry = ModuleSources(dict(library_inputs or {})).supermodule_registry()
    key = (record['module_type'], record['module_subtype'])
    if key not in registry:
        return None
    if instance_values is not None:
        return instance_values
    return instance_values_from_version(registry[key], registry, record.get('parameterisation'),
                                        f'"{record["name"]}" {key}')


def generate_module_array(model, work_dir, file_prefix, overrides=(), library_inputs=None,
                          model_type='cellml', solver=None, protocol=None):
    '''Generate the module array ``model`` (a spec.ModuleArray) under ``work_dir``, from a copy
    of its module array and its parameters file. ``overrides`` (rows named as a model's parameters
    file names them) are set over the parameters file's values. A select.SelectedArray is written
    from its records, its parameters file holding the target's entries of its supermodule
    instances with ``overrides`` over them. Returns as :func:`generate_parameterisation`.'''
    work_dir = os.path.abspath(work_dir)
    resources_dir = os.path.join(work_dir, 'resources')
    os.makedirs(resources_dir, exist_ok=True)
    path = os.path.join(resources_dir, parameters_file.parameters_file_name(file_prefix))
    doc = parameters_file.read_parameters_file(model.parameters)
    records = getattr(model, 'records', None)
    if records is not None:
        array = os.path.join(resources_dir, f'{file_prefix}_module_array.json')
        with open(array, 'w') as f:
            json.dump(records, f, indent=1)
        doc = dict(parameters_file.new_document(),
                   **{r['name']: doc[r['name']] for r in records
                      if r['name'] in doc and not parameters_file.is_leaf(doc[r['name']])})
    else:
        extension = os.path.splitext(model.module_array)[1].lower() or '.csv'
        array = os.path.join(resources_dir, f'{file_prefix}_module_array{extension}')
        shutil.copyfile(model.module_array, array)
    parameters_file.write_parameters_file(path, with_overrides(doc, overrides, f'{path}: '))
    return _generate(work_dir, resources_dir, file_prefix, model_type, solver, library_inputs,
                     model.label(), protocol)


def generate_model(model, work_dir, file_prefix, overrides=(), library_inputs=None,
                   model_type='cellml', solver=None, protocol=None, instance_values=None):
    '''Generate a workflow model of either kind: :func:`generate_module_array` for a
    spec.ModuleArray, :func:`generate_parameterisation` for a library module (a supermodule's
    with ``instance_values``, if given, as its instance's values).'''
    if model.kind == 'array':
        return generate_module_array(model, work_dir, file_prefix, overrides, library_inputs,
                                     model_type, solver, protocol=protocol)
    return generate_parameterisation(model, work_dir, file_prefix, overrides, library_inputs,
                                     model_type, solver, protocol=protocol,
                                     instance_values=instance_values)


def _unset(resources_dir, file_prefix):
    '''``[(name, units)]`` the failed generation left without a value, named as the model's
    parameters file names them (``C/x``), from the unfinished parameters file it wrote.'''
    path = os.path.join(resources_dir, f'{file_prefix}_parameters_unfinished.json')
    if not os.path.isfile(path):
        return []
    return [(r['variable_name'], r.get('units', ''))
            for r in parameters_file.read_parameter_rows(path)
            if (r.get('value') or '').strip() == UNSET_VALUE]


def _generate(work_dir, resources_dir, file_prefix, model_type, solver, library_inputs, label,
              protocol=None):
    try:
        return _generate_once(work_dir, resources_dir, file_prefix, model_type, solver,
                              library_inputs, label)
    except RuntimeError:
        unset = _unset(resources_dir, file_prefix)
        if not unset:
            raise
    given = {host_model_name(qname): (qname, value) for qname, value in (protocol or {}).items()}
    filled = [{'variable_name': name, 'units': units,
               'value': repr(float(given[host_model_name(name)][1])),
               'data_reference': 'start of the step\'s protocol (obs_data params_to_change)'}
              for name, units in unset if host_model_name(name) in given]
    missing = [host_model_name(name) for name, _ in unset if host_model_name(name) not in given]
    if missing:
        raise UnsetModelParameters(label, missing)
    path = os.path.join(resources_dir, parameters_file.parameters_file_name(file_prefix))
    parameters_file.write_parameters_file(
        path, with_overrides(parameters_file.read_parameters_file(path), filled, f'{path}: '))
    os.remove(os.path.join(resources_dir, f'{file_prefix}_parameters_unfinished.json'))
    return _generate_once(work_dir, resources_dir, file_prefix, model_type, solver,
                          library_inputs, label)


def _generate_once(work_dir, resources_dir, file_prefix, model_type, solver, library_inputs,
                   label):
    # imported here: generation pulls in libCellML, which nothing else in the workflow needs
    from libcuflynx.scripts.script_generate_with_new_architecture import \
        generate_with_new_architecture
    generated_models_dir = os.path.join(work_dir, 'generated_models')
    config = {'file_prefix': file_prefix, 'input_param_file': f'{file_prefix}_parameters.json',
              'model_type': model_type, 'resources_dir': resources_dir,
              'generated_models_dir': generated_models_dir, 'DEBUG': False}
    if solver:
        config['solver'] = solver
    config.update(library_inputs or {})
    if not generate_with_new_architecture(False, config):
        raise RuntimeError(f'generating {label} failed; the generator\'s messages above say '
                           f'why.')
    model_dir = os.path.join(generated_models_dir, file_prefix)
    flat = os.path.join(model_dir, f'{file_prefix}_flat.cellml')
    return {
        'model_path': os.path.join(model_dir,
                                   file_prefix + _MODEL_EXTENSIONS.get(model_type, '.cellml')),
        # the CellML model with its imports resolved into one file (cellml only)
        'flat_model_path': flat if os.path.isfile(flat) else None,
        # every value the model uses, in the form of a model's parameters file
        'parameters_path': os.path.join(model_dir,
                                        parameters_file.parameters_file_name(file_prefix)),
        'resources_dir': resources_dir,
        'generated_models_dir': generated_models_dir,
        'inp_data_dict': config,
    }
