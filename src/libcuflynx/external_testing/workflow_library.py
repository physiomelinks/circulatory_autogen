"""A tiny module library with calibration workflows, for testing workflows inside and outside
this repository (CUFLynx tests its workflow tab against it).

Every model is algebraic and quick to simulate, and every optimum is known in closed form.

Modules (all version ``v1``):

* ``lin_a`` -- ``y = 2 p^2``, with an exit ``lin_port`` [y].
* ``lin_b`` -- ``y = 3 p^2`` (no ports).
* ``lin_c`` -- ``z = x + k q^2``, with an entrance ``lin_port`` [x]. ``k`` = 1 is never
  calibrated; its source cites ``K_REFERENCE`` from ``lin_c``'s ``references.bib``.
* ``twin`` -- supermodule of ``A`` (lin_a) and ``B`` (lin_b), which are independent.
* ``chain`` -- supermodule of ``A`` (lin_a) feeding ``C`` (lin_c): ``z = 2 p_A^2 + q_C^2``.

Parameterisations:

* ``lin_a/fit_a`` -- y = Y_A: p = sqrt(Y_A / 2) (2 by default).
* ``lin_b/fit_b`` -- y = Y_B: p = sqrt(Y_B / 3) (3 by default).
* ``twin/joint`` -- one obs_data with both items and one params_for_id with both parameters:
  the single calibration the ``twin/split`` workflow must reproduce.
* ``twin/split`` -- ``calibration_workflow.json``: fit_a and fit_b, independent.
* ``chain/rest`` -- z = Z_C with p_A fixed from fit_a: q = sqrt(Z_C - Y_A) (3 by default);
  ``calibration_workflow.json``: fit_a, then rest with fixed_from fit_a.
* ``chain/screened`` -- ``calibration_workflow.json`` (``chain_analysed``): fit_a; ``screen``, a
  Sobol analysis of z over p_A in [1.9, 2.1] and q_C in [0.1, 10], where q_C dominates;
  ``reduce``, which keeps the one most influential parameter; ``fit``, which calibrates what
  ``reduce`` kept with p_A fixed from fit_a (q = 3, as chain/rest); and ``ia``, a Laplace
  identifiability analysis at ``fit``'s result: std(q) = STD_C / (2 q) = STD_C / 6.
* ``chain/pipeline`` -- ``calibration_workflow.json`` (``chain_pipeline``), a whole study:
  fit_a calibrates the submodule A; ``screen``, a Sobol analysis of z in the chain (p_A fixed)
  over q_C in [0.1, 10] and the gain k_C in [0.9, 1.1], where q_C dominates; ``reduce`` keeps
  it; ``emulate`` trains an emulator of z over q_C alone; ``posterior`` calibrates q_C on the
  emulator and samples its posterior by MCMC: median 3, std about STD_C / 6.

Module arrays (``arrays/``), the same chain without the module library's layout: ``a_only``
(one record ``A``, lin_a) and ``chain`` (``A`` feeding ``C``), with their parameters files and
obs_data / params_for_id named freely. ``chain_array.json``: ``a_only`` calibrates ``A/p``, then
``chain`` calibrates ``C/q`` with it fixed; the target is ``chain``. Same answers as chain/rest.
"""

import json
import os

from libcuflynx.utilities import parameters_file, params_for_id_file

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

#: A Sobol analysis small enough for a test: (2 D + 2) N = 384 runs for D = 2 parameters.
SA_OPTIONS = {'method': 'sobol', 'sample_type': 'saltelli', 'num_samples': 64}
#: An emulator a polynomial in one parameter fits exactly, from 32 runs.
EMULATOR_SETTINGS = {'num_train_samples': 32, 'sample_type': 'sobol', 'random_seed': 0,
                     'min_r2': 0.9, 'n_iter': 2, 'n_splits': 2,
                     'models': 'PolynomialRegression'}
#: MCMC short enough for a test of a one-parameter posterior.
UQ_OPTIONS = {'num_steps': 200, 'num_walkers': 8, 'burn_in': 0.5}
IA_OPTIONS = {'method': 'Laplace', 'gradient_source': 'FD'}

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
    'lin_c': (_component('lin_c_type', [('z', 'out'), ('x', 'in'), ('q', 'in'), ('k', 'in')],
                         '<apply><eq/><ci>z</ci><apply><plus/><ci>x</ci>'
                         '<apply><times/><ci>k</ci><ci>q</ci><ci>q</ci></apply></apply></apply>'),
              [{'port_type': 'lin_port', 'variables': ['x']}], [],
              [['z', 'dimensionless', 'access', 'variable'],
               ['x', 'dimensionless', 'access', 'boundary_condition'],
               ['q', 'dimensionless', 'access', 'constant'],
               ['k', 'dimensionless', 'access', 'constant']]),
}

#: The BibTeX key lin_c's k cites, and its entry (in lin_c's references.bib).
K_REFERENCE = 'doe2020gain'
K_SOURCE = f'{K_REFERENCE}; Table 1: the gain of C'
_BIB = {'lin_c': '@article{' + K_REFERENCE + ''',
  author = {Doe, Jane},
  title = {A gain of one},
  journal = {Journal of Test Libraries},
  year = {2020}
}
'''}


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)


def _write_json(path, obj):
    _write(path, json.dumps(obj, indent=1) + '\n')


def _parameters(path, rows, kind=parameters_file.MODEL):
    '''A parameters file (nested JSON; a parameterisation's when ``kind`` is MODULE) of ``rows``:
    ``(name, value)``, or ``(name, value, data_reference)``.'''
    parameters_file.write_parameter_rows(path, [
        {'variable_name': row[0], 'units': 'dimensionless', 'value': str(row[1]),
         'data_reference': row[2] if len(row) > 2 else 'workflow test library'}
        for row in rows], kind)


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
    params_for_id_file.write_params_for_id(path, [
        {'targets': [params_for_id_file.target_name(vessel, param)], 'min': lo, 'max': hi,
         'name_for_plotting': f'{param}_{vessel}'} for vessel, param, lo, hi in rows])


def _parameterisation(modules, module_type, parameterisation, parameters, obs=None,
                      params_for_id=None):
    directory = os.path.join(modules, module_type, 'versions', 'v1', 'parameterisations',
                             parameterisation)
    _parameters(os.path.join(directory, f'{parameterisation}_parameters.json'), parameters,
                parameters_file.MODULE)
    if obs is not None:
        _write_json(os.path.join(directory, f'{parameterisation}_obs_data.json'),
                    _obs(parameterisation, obs))
    if params_for_id is not None:
        _params_for_id(os.path.join(directory, f'{parameterisation}_params_for_id.json'),
                       params_for_id)
    return directory


def _target(module_type, parameterisation):
    return {'module_type': module_type, 'version': 'v1', 'parameterisation': parameterisation}


def _library_step(step_id, module_type, directory, workflow_dir, **extra):
    '''A step that calibrates the library parameterisation in ``directory``, against its own
    obs_data and params_for_id (paths relative to the workflow file).'''
    name = os.path.basename(directory)
    return dict({'id': step_id, 'model': _target(module_type, name),
                 'obs_data': os.path.relpath(os.path.join(directory, f'{name}_obs_data.json'),
                                             workflow_dir),
                 'params_for_id': os.path.relpath(
                     os.path.join(directory, f'{name}_params_for_id.json'), workflow_dir)},
                **extra)


def build_workflow_library(root, settings=None):
    """Write the library under ``root``. Returns a dict of paths: ``modules`` (the module
    library dir), ``twin_split``, ``chain_rest``, ``chain_analysed``, ``chain_pipeline`` and
    ``chain_array`` (the workflow files), and the parameterisation directories by ``<type>/<parameterisation>``.

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
            'default_parameterisation': 'default', 'entrance_ports': entrance, 'exit_ports': exit_,
            'general_ports': [], 'variables_and_units': variables}])
        rows = [('q', 1.0), ('k', 1.0, K_SOURCE)] if module_type == 'lin_c' else [('p', 1.0)]
        _parameterisation(modules, module_type, 'default', rows)
        if module_type in _BIB:
            _write(os.path.join(version_dir, f'{module_type}_v1_references.bib'), _BIB[module_type])

    def submodule(name, module_type, inp=(), out=()):
        return {'name': name, 'module_type': module_type, 'module_subtype': 'v1',
                'inp_instances': list(inp), 'out_instances': list(out)}

    # each supermodule's default parameterisation: its instances' values, the template a new
    # instance of it is copied from
    template_values = {'twin': {'A': [('p', 1.0)], 'B': [('p', 1.0)]},
                       'chain': {'A': [('p', 1.0)], 'C': [('q', 1.0), ('k', 1.0, K_SOURCE)]}}
    template_rows = {module_type: [(f'{sub}/{row[0]}',) + tuple(row[1:])
                                   for sub, rows in values.items() for row in rows]
                     for module_type, values in template_values.items()}
    for module_type, submodules in (
            ('twin', [submodule('A', 'lin_a'), submodule('B', 'lin_b')]),
            ('chain', [submodule('A', 'lin_a', out=['C']), submodule('C', 'lin_c', inp=['A'])])):
        version_dir = os.path.join(modules, module_type, 'versions', 'v1')
        versions = list({(s['module_type'], s['module_subtype']): None for s in submodules})
        _write_json(os.path.join(version_dir, f'{module_type}_v1_modules_config.json'),
                    [{'module_type': module_type, 'module_subtype': 'v1',
                      'module_format': 'supermodule', 'default_parameterisation': 'default',
                      'submodule_versions': [{'module_type': t, 'module_subtype': s}
                                             for t, s in versions]}])
        _write_json(os.path.join(version_dir, f'{module_type}_v1_module_array.json'), submodules)
        _parameterisation(modules, module_type, 'default', template_rows[module_type])

    paths = {'modules': modules}
    paths['lin_a/fit_a'] = _parameterisation(modules, 'lin_a', 'fit_a', [('p', 1.0)],
                                             obs=[('y_a', 'mod/y', Y_A, STD_A)],
                                             params_for_id=[('mod', 'p', 0.1, 5.0)])
    paths['lin_b/fit_b'] = _parameterisation(modules, 'lin_b', 'fit_b', [('p', 1.0)],
                                             obs=[('y_b', 'mod/y', Y_B, STD_B)],
                                             params_for_id=[('mod', 'p', 0.1, 5.0)])
    # a named supermodule parameterisation replaces the default as its instances' values, so it
    # sets every one of them; these set the default's
    twin_rows = template_rows['twin']
    paths['twin/joint'] = _parameterisation(
        modules, 'twin', 'joint', twin_rows,
        obs=[('y_a', 'mod__A/y', Y_A, STD_A), ('y_b', 'mod__B/y', Y_B, STD_B)],
        params_for_id=[('mod__A', 'p', 0.1, 5.0), ('mod__B', 'p', 0.1, 5.0)])
    paths['twin/split'] = _parameterisation(modules, 'twin', 'split', twin_rows)
    paths['twin_split'] = os.path.join(paths['twin/split'], 'calibration_workflow.json')
    _write_json(paths['twin_split'], {
        'schema_version': 1, 'workflow_name': 'twin_split',
        'description': 'A and B calibrated apart, then merged into twin',
        'module_library_dirs': [os.path.relpath(modules, paths['twin/split'])],
        'settings': settings,
        'target': _target('twin', 'split'),
        'steps': [_library_step('fit_a', 'lin_a', paths['lin_a/fit_a'], paths['twin/split']),
                  _library_step('fit_b', 'lin_b', paths['lin_b/fit_b'], paths['twin/split'])]})

    chain_rows = template_rows['chain']
    paths['chain/rest'] = _parameterisation(modules, 'chain', 'rest', chain_rows,
                                            obs=[('z_c', 'mod__C/z', Z_C, STD_C)],
                                            params_for_id=[('mod__C', 'q', 0.1, 10.0)])
    paths['chain_rest'] = os.path.join(paths['chain/rest'], 'calibration_workflow.json')
    _write_json(paths['chain_rest'], {
        'schema_version': 1, 'workflow_name': 'chain_rest',
        'description': 'A calibrated alone, then C in the chain with A fixed',
        'module_library_dirs': [os.path.relpath(modules, paths['chain/rest'])],
        'settings': settings,
        'target': _target('chain', 'rest'),
        'steps': [_library_step('fit_a', 'lin_a', paths['lin_a/fit_a'], paths['chain/rest']),
                  _library_step('rest', 'chain', paths['chain/rest'], paths['chain/rest'],
                                fixed_from=['fit_a'])]})

    paths['chain/screened'] = _parameterisation(
        modules, 'chain', 'screened', chain_rows, obs=[('z_c', 'mod__C/z', Z_C, STD_C)],
        params_for_id=[('mod__A', 'p', 1.9, 2.1), ('mod__C', 'q', 0.1, 10.0)])
    paths['chain_analysed'] = os.path.join(paths['chain/screened'], 'calibration_workflow.json')
    screened = paths['chain/screened']

    def screened_step(step_id, **extra):
        return _library_step(step_id, 'chain', screened, screened, **extra)

    fit = screened_step('fit', fixed_from=['fit_a'], params_for_id_from='reduce')
    del fit['params_for_id']
    _write_json(paths['chain_analysed'], {
        'schema_version': 1, 'workflow_name': 'chain_analysed',
        'description': 'C is found to matter by Sobol analysis, calibrated with A fixed, and '
                       'its identifiability analysed at the fit',
        'module_library_dirs': [os.path.relpath(modules, screened)],
        'settings': settings,
        'target': _target('chain', 'screened'),
        'steps': [
            _library_step('fit_a', 'lin_a', paths['lin_a/fit_a'], screened),
            screened_step('screen', kind='sensitivity', settings={'sa_options': SA_OPTIONS}),
            {'id': 'reduce', 'kind': 'reduction', 'sensitivity_from': 'screen',
             'reduction': {'top_n': 1}},
            fit,
            dict(screened_step('ia', kind='identifiability', result_from='fit',
                               settings={'ia_options': IA_OPTIONS}),
                 params_for_id=os.path.relpath(
                     os.path.join(paths['chain/rest'], 'rest_params_for_id.json'), screened)),
        ]})

    paths['chain/pipeline'] = _parameterisation(
        modules, 'chain', 'pipeline', chain_rows, obs=[('z_c', 'mod__C/z', Z_C, STD_C)],
        params_for_id=[('mod__C', 'q', 0.1, 10.0), ('mod__C', 'k', 0.9, 1.1)])
    paths['chain_pipeline'] = os.path.join(paths['chain/pipeline'], 'calibration_workflow.json')
    pipeline = paths['chain/pipeline']

    def pipeline_step(step_id, reduced=False, **extra):
        step = _library_step(step_id, 'chain', pipeline, pipeline, **extra)
        if reduced:
            del step['params_for_id']
            step['params_for_id_from'] = 'reduce'
        return step

    _write_json(paths['chain_pipeline'], {
        'schema_version': 1, 'workflow_name': 'chain_pipeline',
        'description': 'A calibrated alone; C screened by Sobol analysis, reduced to the '
                       'parameter that matters, emulated, and its posterior sampled on the '
                       'emulator',
        'module_library_dirs': [os.path.relpath(modules, pipeline)],
        'settings': settings,
        'target': _target('chain', 'pipeline'),
        'steps': [
            _library_step('fit_a', 'lin_a', paths['lin_a/fit_a'], pipeline),
            pipeline_step('screen', kind='sensitivity', fixed_from=['fit_a'],
                          settings={'sa_options': SA_OPTIONS}),
            {'id': 'reduce', 'kind': 'reduction', 'sensitivity_from': 'screen',
             'reduction': {'top_n': 1}},
            pipeline_step('emulate', reduced=True, kind='emulation', fixed_from=['fit_a'],
                          settings={'emulator_settings': EMULATOR_SETTINGS}),
            pipeline_step('posterior', reduced=True, emulator_from='emulate',
                          store_distribution=True, settings={'UQ_options': UQ_OPTIONS}),
        ]})

    arrays = os.path.join(os.path.abspath(root), 'arrays')
    paths['arrays'] = arrays

    def record(name, module_type, inp=(), out=()):
        return {'name': name, 'module_type': module_type, 'module_subtype': 'v1',
                'parameterisation': 'default', 'inp_instances': list(inp),
                'out_instances': list(out)}

    _write_json(os.path.join(arrays, 'a_only_module_array.json'), [record('A', 'lin_a')])
    _parameters(os.path.join(arrays, 'a_only_parameters.json'), [('A/p', 1.0)])
    _write_json(os.path.join(arrays, 'chain_module_array.json'),
                [record('A', 'lin_a', out=['C']), record('C', 'lin_c', inp=['A'])])
    _parameters(os.path.join(arrays, 'chain_parameters.json'),
                [('A/p', 1.0), ('C/q', 1.0), ('C/k', 1.0, K_SOURCE)])
    _write_json(os.path.join(arrays, 'data', 'a_obs.json'), _obs('a', [('y_a', 'A/y', Y_A, STD_A)]))
    _params_for_id(os.path.join(arrays, 'data', 'a_params.json'), [('A', 'p', 0.1, 5.0)])
    _write_json(os.path.join(arrays, 'data', 'chain_obs.json'),
                _obs('chain', [('z_c', 'C/z', Z_C, STD_C)]))
    _params_for_id(os.path.join(arrays, 'data', 'chain_params.json'), [('C', 'q', 0.1, 10.0)])
    paths['chain_array'] = os.path.join(arrays, 'chain_array.json')
    _write_json(paths['chain_array'], {
        'schema_version': 1, 'workflow_name': 'chain_array',
        'description': 'A calibrated alone, then C in the chain with A fixed, as module arrays',
        'module_library_dirs': [os.path.relpath(modules, arrays)],
        'settings': settings,
        'target': 'chain_module_array.json',
        'steps': [{'id': 'a-alone', 'model': 'a_only_module_array.json',
                   'obs_data': 'data/a_obs.json', 'params_for_id': 'data/a_params.json'},
                  {'id': 'chain', 'model': 'chain_module_array.json',
                   'obs_data': 'data/chain_obs.json', 'params_for_id': 'data/chain_params.json',
                   'fixed_from': ['a-alone']}]})
    return paths
