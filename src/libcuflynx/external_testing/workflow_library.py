"""A tiny module library with calibration workflows, for testing workflows inside and outside
this repository (CUFLynx tests its workflow tab against it).

Every model is algebraic and quick to simulate, and every optimum is known in closed form.

Modules (all version ``v1``):

* ``lin_a`` -- ``y = 2 p^2``, with an exit ``lin_port`` [y].
* ``lin_b`` -- ``y = 3 p^2`` (no ports).
* ``lin_c`` -- ``z = x + q^2``, with an entrance ``lin_port`` [x].
* ``twin`` -- supermodule of ``A`` (lin_a) and ``B`` (lin_b), which are independent.
* ``chain`` -- supermodule of ``A`` (lin_a) feeding ``C`` (lin_c): ``z = 2 p_A^2 + q_C^2``.

Instances:

* ``lin_a/fit_a`` -- y = Y_A: p = sqrt(Y_A / 2) (2 by default).
* ``lin_b/fit_b`` -- y = Y_B: p = sqrt(Y_B / 3) (3 by default).
* ``twin/joint`` -- one obs_data with both items and one params_for_id with both parameters:
  the single calibration the ``twin/split`` workflow must reproduce.
* ``twin/split`` -- ``calibration_workflow.json``: fit_a and fit_b, independent.
* ``chain/rest`` -- z = Z_C with p_A fixed from fit_a: q = sqrt(Z_C - Y_A) (3 by default);
  ``calibration_workflow.json``: fit_a, then rest with fixed_from fit_a.
"""

import json
import os

Y_A, STD_A = 8.0, 0.08
Y_B, STD_B = 27.0, 0.27
Z_C, STD_C = 17.0, 0.17

#: Settings that converge every fit here to ~1e-6 relative: one or two parameters,
#: smooth bowls.
OPTIMISER_SETTINGS = {
    'param_id_method': 'CMA-ES',
    'optimiser_options': {'num_calls_to_function': 400, 'cost_convergence': 1e-14,
                          'max_patience': 60, 'seed': 1, 'cost_type': 'gaussian_MLE'},
    'dt': 0.1,
}

_CELLML_HEADER = ("<?xml version='1.0' encoding='UTF-8'?>\n"
                  '<model name="{name}" xmlns="http://www.cellml.org/cellml/1.1#" '
                  'xmlns:cellml="http://www.cellml.org/cellml/1.1#">\n')


def _var(name, interface):
    return (f'        <variable name="{name}" public_interface="{interface}" '
            f'units="dimensionless"/>\n')


def _component(name, variables, equation):
    return (f'    <component name="{name}">\n'
            '        <variable name="t" public_interface="in" units="second"/>\n'
            + ''.join(_var(n, i) for n, i in variables)
            + '        <math xmlns="http://www.w3.org/1998/Math/MathML">\n'
            f'            {equation}\n'
            '        </math>\n    </component>\n')


def _times_square(gain, var):
    return (f'<apply><times/><cn cellml:units="dimensionless">{gain}</cn>'
            f'<ci>{var}</ci><ci>{var}</ci></apply>')


_MODULES = {
    'lin_a': (_component('lin_a_type', [('y', 'out'), ('p', 'in')],
                         f'<apply><eq/><ci>y</ci>{_times_square(2.0, "p")}</apply>'),
              [], [{'port_type': 'lin_port', 'variables': ['y']}],
              [['y', 'dimensionless', 'access', 'variable'],
               ['p', 'dimensionless', 'access', 'constant']]),
    'lin_b': (_component('lin_b_type', [('y', 'out'), ('p', 'in')],
                         f'<apply><eq/><ci>y</ci>{_times_square(3.0, "p")}</apply>'),
              [], [],
              [['y', 'dimensionless', 'access', 'variable'],
               ['p', 'dimensionless', 'access', 'constant']]),
    'lin_c': (_component('lin_c_type', [('z', 'out'), ('x', 'in'), ('q', 'in')],
                         '<apply><eq/><ci>z</ci><apply><plus/><ci>x</ci>'
                         f'{_times_square(1.0, "q")}</apply></apply>'),
              [{'port_type': 'lin_port', 'variables': ['x']}], [],
              [['z', 'dimensionless', 'access', 'variable'],
               ['x', 'dimensionless', 'access', 'boundary_condition'],
               ['q', 'dimensionless', 'access', 'constant']]),
}


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)


def _write_json(path, obj):
    _write(path, json.dumps(obj, indent=1) + '\n')


def _parameters(path, rows):
    _write(path, 'variable_name,units,value,data_reference\n'
           + ''.join(f'{name},dimensionless,{value},workflow test library\n'
                     for name, value in rows))


def _obs(name, items):
    return {'obs_data_name': name,
            'protocol_info': {'pre_times': [0.0], 'sim_times': [[1.0]],
                              'params_to_change': {}},
            'data_items': [{'data_item_name': item_name, 'data_type': 'constant',
                            'unit': 'dimensionless', 'operands': [operand],
                            'operation': 'mean', 'weight': 1.0, 'value': value, 'std': std}
                           for item_name, operand, value, std in items],
            'prediction_items': []}


def _params_for_id(path, rows):
    _write(path, 'vessel_name,param_name,min,max,name_for_plotting\n'
           + ''.join(f'{vessel},{param},{lo},{hi},{param}_{vessel}\n'
                     for vessel, param, lo, hi in rows))


def _instance(modules, module_type, instance, parameters, obs=None, params_for_id=None):
    directory = os.path.join(modules, module_type, 'versions', 'v1', 'instances', instance)
    _parameters(os.path.join(directory, f'{instance}_parameters.csv'), parameters)
    if obs is not None:
        _write_json(os.path.join(directory, f'{instance}_obs_data.json'), _obs(instance, obs))
    if params_for_id is not None:
        _params_for_id(os.path.join(directory, f'{instance}_params_for_id.csv'), params_for_id)
    return directory


def _target(module_type, instance):
    return {'module_type': module_type, 'version': 'v1', 'instance': instance}


def build_workflow_library(root, settings=None):
    """Write the library under ``root``. Returns a dict of paths: ``modules`` (the module
    library dir), ``twin_split`` and ``chain_rest`` (the two workflow files), and the
    instance directories by ``<type>/<instance>``.

    ``settings`` replaces the workflows' settings (default ``OPTIMISER_SETTINGS``).
    """
    settings = OPTIMISER_SETTINGS if settings is None else settings
    modules = os.path.join(os.path.abspath(root), 'modules')
    for module_type, (component, entrance, exit_, variables) in _MODULES.items():
        version_dir = os.path.join(modules, module_type, 'versions', 'v1')
        _write(os.path.join(version_dir, f'{module_type}_v1_modules.cellml'),
               _CELLML_HEADER.format(name=f'{module_type}_v1_modules') + component + '</model>\n')
        _write_json(os.path.join(version_dir, f'{module_type}_v1_modules_config.json'), [{
            'module_type': module_type, 'module_subtype': 'v1',
            'component_file': f'{module_type}_v1_modules.cellml',
            'component_type': f'{module_type}_type', 'module_format': 'cellml',
            'default_instance': 'default', 'entrance_ports': entrance, 'exit_ports': exit_,
            'general_ports': [], 'variables_and_units': variables}])
        param = 'q' if module_type == 'lin_c' else 'p'
        _instance(modules, module_type, 'default', [(param, 1.0)])

    def submodule(name, module_type, inp=(), out=()):
        return {'name': name, 'module_type': module_type, 'module_subtype': 'v1',
                'instance': 'default', 'inp_instances': list(inp), 'out_instances': list(out)}

    for module_type, submodules in (
            ('twin', [submodule('A', 'lin_a'), submodule('B', 'lin_b')]),
            ('chain', [submodule('A', 'lin_a', out=['C']), submodule('C', 'lin_c', inp=['A'])])):
        _write_json(os.path.join(modules, module_type, 'versions', 'v1',
                                 f'{module_type}_v1_modules_config.json'),
                    [{'module_type': module_type, 'module_subtype': 'v1',
                      'module_format': 'supermodule', 'default_instance': 'default',
                      'submodules': submodules}])

    paths = {'modules': modules}
    paths['lin_a/fit_a'] = _instance(modules, 'lin_a', 'fit_a', [('p', 1.0)],
                                     obs=[('y_a', 'mod/y', Y_A, STD_A)],
                                     params_for_id=[('mod', 'p', 0.1, 5.0)])
    paths['lin_b/fit_b'] = _instance(modules, 'lin_b', 'fit_b', [('p', 1.0)],
                                     obs=[('y_b', 'mod/y', Y_B, STD_B)],
                                     params_for_id=[('mod', 'p', 0.1, 5.0)])
    twin_rows = [('p_A', 1.0), ('p_B', 1.0)]
    _instance(modules, 'twin', 'default', twin_rows)
    paths['twin/joint'] = _instance(
        modules, 'twin', 'joint', twin_rows,
        obs=[('y_a', 'mod_A/y', Y_A, STD_A), ('y_b', 'mod_B/y', Y_B, STD_B)],
        params_for_id=[('mod_A', 'p', 0.1, 5.0), ('mod_B', 'p', 0.1, 5.0)])
    paths['twin/split'] = _instance(modules, 'twin', 'split', twin_rows)
    paths['twin_split'] = os.path.join(paths['twin/split'], 'calibration_workflow.json')
    _write_json(paths['twin_split'], {
        'schema_version': 1, 'workflow_name': 'twin_split',
        'description': 'A and B calibrated apart, then merged into twin',
        'module_library_dirs': [os.path.relpath(modules, paths['twin/split'])],
        'settings': settings,
        'steps': [{'id': 'fit_a', 'target': _target('lin_a', 'fit_a')},
                  {'id': 'fit_b', 'target': _target('lin_b', 'fit_b')}]})

    chain_rows = [('p_A', 1.0), ('q_C', 1.0)]
    _instance(modules, 'chain', 'default', chain_rows)
    paths['chain/rest'] = _instance(modules, 'chain', 'rest', chain_rows,
                                    obs=[('z_c', 'mod_C/z', Z_C, STD_C)],
                                    params_for_id=[('mod_C', 'q', 0.1, 10.0)])
    paths['chain_rest'] = os.path.join(paths['chain/rest'], 'calibration_workflow.json')
    _write_json(paths['chain_rest'], {
        'schema_version': 1, 'workflow_name': 'chain_rest',
        'description': 'A calibrated alone, then C in the chain with A fixed',
        'module_library_dirs': [os.path.relpath(modules, paths['chain/rest'])],
        'settings': settings,
        'steps': [{'id': 'fit_a', 'target': _target('lin_a', 'fit_a')},
                  {'id': 'rest', 'target': _target('chain', 'rest'),
                   'depends_on': ['fit_a'], 'fixed_from': ['fit_a']}]})
    return paths
