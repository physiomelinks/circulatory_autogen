'''
Generating one module instance on its own, with some of its parameters overridden.
'''

import csv
import json
import os

from libcuflynx.calibration_workflow.naming import STEP_VESSEL

PARAMETER_HEADER = ('variable_name', 'units', 'value', 'data_reference')
_MODEL_EXTENSIONS = {'cellml': '.cellml', 'python': '.py', 'casadi_python': '.py',
                     'aadc_python': '.py', 'cpp': '.cpp'}


def write_parameter_rows(path, rows):
    '''Write ``rows`` (dicts with PARAMETER_HEADER keys) as a parameters CSV.'''
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(PARAMETER_HEADER), extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in PARAMETER_HEADER})


def read_parameter_rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return [dict(row) for row in csv.DictReader(f)]


def generate_module_instance(target, work_dir, file_prefix, overrides=(), library_inputs=None,
                             model_type='cellml', solver=None, vessel=STEP_VESSEL):
    '''Generate the module instance ``target`` (a spec.Target) alone, as a single module-array
    record named ``vessel``, under ``work_dir``.

    ``overrides`` are parameter rows (``variable_name`` in the generated model's naming,
    ``units``, ``value``, ``data_reference``) written as the host parameters file, which wins
    over every instance and supermodule default. ``library_inputs`` are the generation keys
    that locate the modules (``module_library_dirs`` etc.).

    Returns ``{'model_path', 'flat_model_path', 'parameters_path', 'resources_dir',
    'generated_models_dir', 'inp_data_dict'}`` (``flat_model_path``: the CellML with its
    imports resolved, None for other model types); raises RuntimeError when generation
    fails. Not MPI-aware: call it on one rank.
    '''
    # imported here: generation pulls in libCellML, which nothing else in the workflow needs
    from libcuflynx.scripts.script_generate_with_new_architecture import \
        generate_with_new_architecture

    work_dir = os.path.abspath(work_dir)
    resources_dir = os.path.join(work_dir, 'resources')
    generated_models_dir = os.path.join(work_dir, 'generated_models')
    os.makedirs(resources_dir, exist_ok=True)
    record = {'name': vessel, 'module_type': target.module_type,
              'module_subtype': target.version, 'instance': target.instance,
              'inp_instances': [], 'out_instances': []}
    with open(os.path.join(resources_dir, f'{file_prefix}_module_array.json'), 'w') as f:
        json.dump([record], f, indent=1)
    write_parameter_rows(os.path.join(resources_dir, f'{file_prefix}_parameters.csv'),
                         list(overrides))

    config = {'file_prefix': file_prefix, 'input_param_file': f'{file_prefix}_parameters.csv',
              'model_type': model_type, 'resources_dir': resources_dir,
              'generated_models_dir': generated_models_dir, 'DEBUG': False}
    if solver:
        config['solver'] = solver
    config.update(library_inputs or {})
    if not generate_with_new_architecture(False, config):
        raise RuntimeError(f'generating {target.label()} failed; the generator\'s messages '
                           f'above say why.')
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
