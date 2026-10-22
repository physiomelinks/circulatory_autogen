'''
Generating a workflow model -- a library module on its own, or a module array -- with some of
its parameters overridden.
'''

import csv
import json
import os
import shutil

from libcuflynx.calibration_workflow.naming import STEP_VESSEL
from libcuflynx.utilities import parameters_file

_MODEL_EXTENSIONS = {'cellml': '.cellml', 'python': '.py', 'casadi_python': '.py',
                     'aadc_python': '.py', 'cpp': '.cpp'}


def read_parameter_rows(path):
    '''The rows of a generated model's parameters CSV (``<prefix>_parameters.csv`` beside the
    generated model, named as the model names them).'''
    with open(path, newline='', encoding='utf-8-sig') as f:
        return [dict(row) for row in csv.DictReader(f)]


def with_overrides(doc, overrides, where=''):
    '''The parameters document ``doc`` with each row of ``overrides`` (``variable_name`` as a
    model's parameters file names it, ``C/q`` or ``mod__A/p``) set over what it has.'''
    for row in overrides:
        parameters_file.set_leaf(doc, parameters_file.name_path(row['variable_name'], where=where),
                                 parameters_file.leaf_from_row(row), where)
    return doc


def generate_parameterisation(target, work_dir, file_prefix, overrides=(), library_inputs=None,
                              model_type='cellml', solver=None, vessel=STEP_VESSEL,
                              instance_values=None):
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
    imports resolved, None for other model types); raises RuntimeError when generation
    fails. Not MPI-aware: call it on one rank.
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
                     target.label())


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
                          model_type='cellml', solver=None):
    '''Generate the module array ``model`` (a spec.ModuleArray) under ``work_dir``, from a copy
    of its module array and its parameters file. ``overrides`` (rows named as a model's parameters
    file names them) are set over the parameters file's values. Returns as
    :func:`generate_parameterisation`.'''
    work_dir = os.path.abspath(work_dir)
    resources_dir = os.path.join(work_dir, 'resources')
    os.makedirs(resources_dir, exist_ok=True)
    extension = os.path.splitext(model.module_array)[1].lower() or '.csv'
    array = os.path.join(resources_dir, f'{file_prefix}_module_array{extension}')
    shutil.copyfile(model.module_array, array)
    path = os.path.join(resources_dir, parameters_file.parameters_file_name(file_prefix))
    doc = parameters_file.read_parameters_file(model.parameters)
    parameters_file.write_parameters_file(path, with_overrides(doc, overrides, f'{path}: '))
    return _generate(work_dir, resources_dir, file_prefix, model_type, solver, library_inputs,
                     model.label())


def generate_model(model, work_dir, file_prefix, overrides=(), library_inputs=None,
                   model_type='cellml', solver=None):
    '''Generate a workflow model of either kind: :func:`generate_module_array` for a
    spec.ModuleArray, :func:`generate_parameterisation` for a library module.'''
    if model.kind == 'array':
        return generate_module_array(model, work_dir, file_prefix, overrides, library_inputs,
                                     model_type, solver)
    return generate_parameterisation(model, work_dir, file_prefix, overrides, library_inputs,
                                     model_type, solver)


def _generate(work_dir, resources_dir, file_prefix, model_type, solver, library_inputs, label):
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
        'parameters_path': os.path.join(model_dir, f'{file_prefix}_parameters.csv'),
        'resources_dir': resources_dir,
        'generated_models_dir': generated_models_dir,
        'inp_data_dict': config,
    }
