"""Calibration workflows (libcuflynx.calibration_workflow, cuflynx-calibration-workflow).

A workflow runs calibrations in order -- each a model, an obs_data and a params_for_id -- and
merges the results into one target model. A model is a library module or a module array. The
fixture library is
libcuflynx.external_testing.workflow_library (shared with CUFLynx's tests): algebraic modules
whose optima are known in closed form.

* ``twin``: A and B are independent, so calibrating them apart (the ``twin/split`` workflow)
  must give what one calibration of both does (``twin/joint``: one obs_data, one
  params_for_id).
* ``chain``: C reads A's output (z = 2 p_A^2 + q^2), so the value fit_a fixes for p_A decides
  what ``rest`` finds for q: q = sqrt(Z_C - Y_A).
* ``chain_analysed``: the steps that are not calibrations -- a Sobol screen of p_A and q, a
  reduction that keeps q, the calibration of what it kept, and a Laplace analysis at that fit,
  whose std is known in closed form.
* ``chain_pipeline``: a whole study in one workflow -- a submodule calibration, a Sobol screen,
  a reduction, an emulator of what was kept, and MCMC on that emulator.
"""
import copy
import csv
import json
import os

import numpy as np
import pytest

from _supermodules import write_supermodule
from _parameters import MODULE, read_parameters, write_parameters
from _params_for_id import params_for_id_entries, write_params_for_id
from libcuflynx.calibration_workflow import (StoredDistribution, WorkflowError,
                                             generate_parameterisation, load_workflow,
                                             load_workflow_run, plan_workflow, resolve_workflow,
                                             run_calibration_workflow)
from libcuflynx.calibration_workflow import naming
from libcuflynx.calibration_workflow.distributions import (DistributionParameter,
                                                           transform_for_bounds)
from libcuflynx.calibration_workflow.spec import ModuleArray, Target, parse_workflow
from libcuflynx.external_testing.workflow_library import (OPTIMISER_SETTINGS, STD_A, STD_B,
                                                          STD_C, Y_A, Y_B, Z_C,
                                                          build_workflow_library)
from libcuflynx.schemas import CALIBRATION_WORKFLOW_SCHEMA, load_schema


@pytest.fixture
def library(tmp_path):
    return build_workflow_library(str(tmp_path / 'library'))


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)


def _raw(library, which='chain_rest'):
    return _read_json(library[which])


def _merged(result):
    return {r['parameterisation_name']: r['value'] for r in result['merged_parameters']}


# --------------------------------------------------------------------------------------------
# the file
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_a_step_names_its_files_relative_to_the_workflow(library):
    workflow = load_workflow(library['twin_split'])
    assert workflow.target == Target('twin', 'v1', 'split')
    assert [s.id for s in workflow.topological_order()] == ['fit_a', 'fit_b']
    assert workflow.module_library_dirs == [library['modules']]
    fit_a = workflow.step('fit_a')
    assert fit_a.model == Target('lin_a', 'v1', 'fit_a')
    assert fit_a.obs_data == os.path.join(library['lin_a/fit_a'], 'fit_a_obs_data.json')
    array = load_workflow(library['chain_array'])
    assert array.target == ModuleArray(
        os.path.join(library['arrays'], 'chain_module_array.json'),
        os.path.join(library['arrays'], 'chain_parameters.json'))
    assert array.step('a-alone').params_for_id == os.path.join(library['arrays'], 'data',
                                                                'a_params.json')


@pytest.mark.unit
def test_the_shipped_schema_accepts_the_fixture_workflows_and_rejects_unknown_keys(library):
    jsonschema = pytest.importorskip('jsonschema')
    schema = load_schema(CALIBRATION_WORKFLOW_SCHEMA)
    for which in ('twin_split', 'chain_rest', 'chain_array', 'chain_analysed', 'chain_pipeline'):
        jsonschema.validate(_raw(library, which), schema)
    raw = _raw(library)
    raw['steps'][0]['fixed_values'] = {}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(raw, schema)
    with pytest.raises(WorkflowError, match="'fixed_values' was unexpected"):
        parse_workflow(raw, path=library['chain_rest'])


@pytest.mark.unit
def test_a_target_names_its_parameterisation_not_an_instance(library):
    jsonschema = pytest.importorskip('jsonschema')
    raw = _raw(library)
    target = raw['target']
    target['instance'] = target.pop('parameterisation')
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(raw, load_schema(CALIBRATION_WORKFLOW_SCHEMA))
    with pytest.raises(WorkflowError, match=r"(?s)at target: .*'instance' was unexpected.*"
                                            r"'parameterisation' is a required property"):
        parse_workflow(raw, path=library['chain_rest'])


@pytest.mark.unit
def test_a_step_runs_after_the_steps_it_takes_values_from(library):
    raw = _raw(library)
    raw['steps'].reverse()
    workflow = parse_workflow(raw, path=library['chain_rest'])
    assert [s.id for s in workflow.topological_order()] == ['fit_a', 'rest']


def _edit(library, edit):
    raw = _raw(library)
    edit(raw)
    return lambda: parse_workflow(raw, path=library['chain_rest'])


@pytest.mark.unit
@pytest.mark.parametrize('edit, message', [
    (lambda w: w['steps'][0].update(fixed_from=['rest']), 'cycle'),
    (lambda w: w['steps'][1].update(fixed_from=['nope']), 'unknown step "nope"'),
    (lambda w: w['steps'][0].update(fixed_from=['fit_a']), 'from itself'),
    (lambda w: w['steps'][1].update(depends_on=['fit_a']), r"'depends_on' was unexpected"),
    (lambda w: w['steps'].append(dict(w['steps'][0])), 'more than once'),
    (lambda w: w['steps'][0].update(target=w['steps'][0]['model']), r"steps\[0\] \('fit_a'\): .*'target' was unexpected"),
    (lambda w: w['steps'][0].pop('obs_data'), r"'obs_data' is a required property"),
    (lambda w: w.pop('target'), r"'target' is a required property"),
    (lambda w: w['steps'][0].update(vessel_map={'A': 'B'}), '"vessel_map" is for a module-array'),
    (lambda w: w['steps'][0].update(model='a_module_array.csv', vessel='A'),
     '"vessel" place a library module'),
    (lambda w: w['settings'].update(param_id_obs_path='x.json'), 'cannot be set'),
    (lambda w: w['steps'][1].update(priors_from=['fit_a'], fixed_from=[]),
     'does not store its distribution'),
    (lambda w: (w['steps'][0].update(store_distribution=True)), 'UQ_options'),
    (lambda w: (w['steps'][0].update(store_distribution=True, settings={'UQ_options': {
        'num_steps': 10}}), w['steps'][1].update(priors_from=['fit_a'])), 'both as fixed'),
    (lambda w: (w['steps'][0].update(store_distribution=True, settings={'UQ_options': {
        'num_steps': 10}}), w['steps'][1].update(priors_from=['fit_a'], fixed_from=[])),
     'would ignore the priors'),
    (lambda w: w['steps'][1].update(priors_from=[{'step': 'fit_a', 'kind': 'gmm'}]),
     'not one of'),
    (lambda w: w.update(schema_version=2), 'schema_version'),
])
def test_invalid_workflows_say_why(library, edit, message):
    with pytest.raises(WorkflowError, match=message):
        _edit(library, edit)()


@pytest.mark.unit
def test_a_step_with_priors_may_instead_optimise_the_log_posterior(library):
    raw = _raw(library)
    raw['steps'][0].update(store_distribution=True, settings={'UQ_options': {'num_steps': 10}})
    raw['steps'][1].update(priors_from=['fit_a'], fixed_from=[], settings={
        'param_id_method': 'genetic_algorithm',
        'optimiser_options': {'objective_function': 'likelihood'}})
    parse_workflow(raw, path=library['chain_rest'])


# --------------------------------------------------------------------------------------------
# naming
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_names_move_into_the_supermodule_as_supermodule_expansion_renames_them():
    assert naming.map_vessel('mod', 'i_M') == 'mod__i_M'
    assert naming.map_vessel('mod__A', 'soma') == 'mod__soma__A'
    assert naming.map_vessel('mod', '') == 'mod'
    assert naming.map_vessel('global', 'soma') == 'global'
    assert naming.model_name(naming.map_vessel('mod', 'i_M'), 'rho_M') == 'rho_M_mod__i_M'
    assert naming.model_name('global', 'R') == 'R'
    paths = [('soma',), ('soma', 'i_M'), ('soma_i',)]
    assert naming.parameterisation_name('mod__soma__i_M', 'rho_M', paths) == 'soma/i_M/rho_M'
    assert naming.parameterisation_name('mod__soma', 'rho_M', paths) == 'soma/rho_M'
    with pytest.raises(ValueError, match='not a submodule'):
        naming.parameterisation_name('mod__heart', 'E', paths)
    assert naming.parameterisation_name('mod', 'p') == 'p'
    assert naming.relative_path('soma__i_M', 'soma') == 'i_M'
    assert naming.relative_path('i_M', '') == 'i_M'
    assert naming.relative_path('B', 'A') is None
    with pytest.raises(ValueError, match='not part of the step model'):
        naming.map_vessel('aorta', 'i_M')


# --------------------------------------------------------------------------------------------
# resolving against the library (no generation)
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_plan_names_every_parameter_in_the_target(library):
    plan = plan_workflow(library['chain_rest'])
    steps = {s['id']: s for s in plan['steps']}
    assert steps['fit_a']['place'] == 'A'
    assert steps['fit_a']['calibrates'] == [{'model_name': 'p_mod', 'target_model_name': 'p_mod__A',
                                             'target_parameterisation_name': 'A/p'}]
    assert steps['rest']['place'] == ''
    assert steps['rest']['fixed_from'] == [{'from_step': 'fit_a', 'model_name': 'p_mod',
                                            'in_step': 'p_mod__A'}]
    assert run_calibration_workflow(library['chain_rest'], dry_run=True) == plan


@pytest.mark.unit
def test_every_step_needs_its_own_params_for_id(library):
    os.remove(os.path.join(library['lin_b/fit_b'], 'fit_b_params_for_id.json'))
    with pytest.raises(WorkflowError, match='params_for_id file .* does not exist'):
        resolve_workflow(load_workflow(library['twin_split']))


def _library_step(library, step_id, module_type, name):
    directory = library[f'{module_type}/{name}']
    return {'id': step_id,
            'model': {'module_type': module_type, 'version': 'v1', 'parameterisation': name},
            'obs_data': os.path.join(directory, f'{name}_obs_data.json'),
            'params_for_id': os.path.join(directory, f'{name}_params_for_id.json')}


def _with_joint_step(library):
    raw = _raw(library, 'twin_split')
    raw['steps'].append(_library_step(library, 'joint', 'twin', 'joint'))
    path = os.path.join(library['twin/split'], 'recalibrated_workflow.json')
    with open(path, 'w') as f:
        json.dump(raw, f)
    return path


@pytest.mark.unit
def test_a_parameter_calibrated_twice_warns_and_points_to_priors(library):
    with pytest.warns(UserWarning) as caught:
        resolve_workflow(load_workflow(_with_joint_step(library)))
    message = str(caught[0].message)
    assert '2 parameter(s)' in message
    assert '"p_mod__A" by \'fit_a\', then \'joint\' (the value of \'joint\' is used)' in message
    assert '"priors_from"' in message


@pytest.mark.integration
def test_the_later_calibration_of_a_parameter_is_merged(library, tmp_path):
    with pytest.warns(UserWarning, match='more than one step'):
        result = run_calibration_workflow(_with_joint_step(library),
                                          output_dir=str(tmp_path / 'run'))
    assert result['complete']
    sources = {r['parameterisation_name']: r['from_step'] for r in result['merged_parameters']}
    assert sources == {'A/p': 'joint', 'B/p': 'joint'}


@pytest.mark.integration
def test_a_step_calibrates_a_parameter_it_would_be_given_fixed(library, tmp_path):
    # rest takes fit_a's values as fixed but also calibrates p of A itself: its own wins
    params = os.path.join(library['chain/rest'], 'rest_params_for_id.json')
    write_params_for_id(params, params_for_id_entries(params) + [
        {'targets': ['mod/A/p'], 'min': 0.1, 'max': 5.0, 'name_for_plotting': 'p_mod__A'}])
    with pytest.warns(UserWarning, match='more than one step'):
        result = run_calibration_workflow(library['chain_rest'],
                                          output_dir=str(tmp_path / 'run'))
    sources = {r['parameterisation_name']: r['from_step'] for r in result['merged_parameters']}
    assert sources == {'A/p': 'rest', 'C/q': 'rest'}
    step = _read_json(str(tmp_path / 'run' / 'rest' / 'step_result.json'))
    assert step['fixed'] == []


@pytest.mark.unit
def test_a_step_must_land_in_the_target(library):
    raw = _raw(library)
    raw['steps'].append(_library_step(library, 'fit_b', 'lin_b', 'fit_b'))
    with pytest.raises(WorkflowError, match='not a submodule of chain'):
        resolve_workflow(parse_workflow(raw, path=library['chain_rest']))


@pytest.mark.unit
def test_an_ambiguous_submodule_must_be_named(library):
    version_dir = os.path.join(library['modules'], 'double', 'versions', 'v1')
    os.makedirs(os.path.join(version_dir, 'parameterisations', 'default'))
    sub = {'module_type': 'lin_a', 'module_subtype': 'v1', 'inp_instances': [], 'out_instances': []}
    write_supermodule(version_dir, {
        'module_type': 'double', 'module_subtype': 'v1', 'module_format': 'supermodule',
        'default_parameterisation': 'default',
        'submodules': [dict(sub, name='A1'), dict(sub, name='A2')]})
    write_parameters(os.path.join(version_dir, 'parameterisations', 'default',
                                  'default_parameters.json'), [], MODULE)
    raw = {'workflow_name': 'double', 'module_library_dirs': [library['modules']],
           'target': {'module_type': 'double', 'version': 'v1', 'parameterisation': 'default'},
           'steps': [_library_step(library, 'fit_a', 'lin_a', 'fit_a')]}
    with pytest.raises(WorkflowError, match=r"\['A1', 'A2'\] in double/v1/default; say which"):
        resolve_workflow(parse_workflow(raw))
    raw['steps'][0]['submodule'] = 'A2'
    resolved = resolve_workflow(parse_workflow(raw))
    assert resolved.steps['fit_a'].place == 'A2'


@pytest.mark.unit
def test_a_type_in_two_libraries_resolves_to_the_one_generation_uses(library, tmp_path):
    """A (module_type, version) defined in two module sources is taken from the most specific
    one (the first module_library_dir), as generation takes it, with a ModuleShadowWarning. The
    workflow read every config file into one registry before, so the copy it resolved could
    differ from the copy generation used."""
    import shutil
    from libcuflynx.utilities.module_library import ModuleShadowWarning
    first = str(tmp_path / 'first' / 'modules')
    shutil.copytree(library['modules'], first)
    with open(library['twin_split']) as f:
        raw = json.load(f)
    raw['module_library_dirs'] = [first, library['modules']]
    with pytest.warns(ModuleShadowWarning):
        resolved = resolve_workflow(parse_workflow(raw, path=library['twin_split']))
    assert resolved.target.entry['config_path'].startswith(first)
    assert all(step.model.entry['config_path'].startswith(first)
               for step in resolved.steps.values())


# --------------------------------------------------------------------------------------------
# stored distributions and joint priors (no simulation)
# --------------------------------------------------------------------------------------------

def _distribution(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    cov = np.array([[0.04, 0.03], [0.03, 0.09]])
    samples = rng.multivariate_normal([2.0, 3.0], cov, size=n)
    parameters = [DistributionParameter([['mod', 'a']], 0.1, 5.0, transform_for_bounds(0.1, 5.0)),
                  DistributionParameter([['mod', 'b']], 0.0, np.inf,
                                        transform_for_bounds(0.0, np.inf))]
    return StoredDistribution(parameters, samples), samples


@pytest.mark.unit
@pytest.mark.parametrize('kind', ['mvnormal', 'normal', 'kde'])
def test_a_stored_distribution_reproduces_its_samples(kind):
    distribution, samples = _distribution()
    draws = distribution.sample(4000, rng=1, kind=kind)
    assert np.allclose(draws.mean(axis=0), samples.mean(axis=0), atol=0.03)
    assert np.allclose(draws.std(axis=0), samples.std(axis=0), rtol=0.1)
    correlation = np.corrcoef(draws.T)[0, 1]
    if kind == 'normal':
        assert abs(correlation) < 0.1
    else:
        assert correlation == pytest.approx(np.corrcoef(samples.T)[0, 1], abs=0.1)
    # draws respect the bounds, and the density is zero outside them
    assert np.all((draws[:, 0] > 0.1) & (draws[:, 0] < 5.0) & (draws[:, 1] > 0.0))
    assert distribution.logpdf([0.05, 3.0], kind=kind) == -np.inf
    assert distribution.logpdf([2.0, -1.0], kind=kind) == -np.inf
    assert np.isfinite(distribution.logpdf([2.0, 3.0], kind=kind))
    # inflating the variance widens it
    wide = distribution.sample(4000, rng=1, kind=kind, inflate=4.0)
    assert wide[:, 1].std() > 1.5 * draws[:, 1].std()


@pytest.mark.unit
def test_the_density_integrates_to_one_on_the_parameter_scale():
    distribution, _ = _distribution()
    a = np.linspace(0.1, 5.0, 401)[1:-1]
    b = np.linspace(0.0, 8.0, 401)[1:]
    grid = np.array(np.meshgrid(a, b, indexing='ij')).reshape(2, -1).T
    density = np.exp(distribution.logpdf(grid)).reshape(len(a), len(b))
    from scipy.integrate import trapezoid
    total = trapezoid(trapezoid(density, b, axis=1), a)
    assert total == pytest.approx(1.0, abs=0.01)


@pytest.mark.unit
def test_a_stored_distribution_round_trips_and_renames(tmp_path):
    distribution, samples = _distribution(n=50)
    chain = samples.reshape(25, 2, 2)
    distribution.save(str(tmp_path / 'distribution'), chain=chain)
    loaded = StoredDistribution.load(str(tmp_path))
    assert np.array_equal(loaded.samples, samples)
    assert loaded.names == ['a_mod', 'b_mod']
    assert set(_read_json(str(tmp_path / 'distribution' / 'stats.json'))['parameters']) == \
        {'a_mod', 'b_mod'}
    mapped = loaded.mapped_into('soma__i_M')
    assert mapped.names == ['a_mod__soma__i_M', 'b_mod__soma__i_M']
    assert mapped.subset(['b_mod__soma__i_M']).names == ['b_mod__soma__i_M']
    from_chain = StoredDistribution.from_chain(chain, distribution.parameters, burn_in_index=5)
    assert from_chain.samples.shape == (40, 2)


@pytest.mark.unit
def test_joint_priors_add_to_the_log_prior():
    from libcuflynx.param_id.paramID import ParamID
    engine = object.__new__(ParamID)
    engine.param_id_info = {'param_prior_types': None, 'param_mins': np.array([0.0, 0.0, 0.0]),
                            'param_maxs': np.array([10.0, 10.0, 10.0]),
                            'param_names_for_gen': [['x_mod'], ['a_mod'], ['b_mod']]}
    engine.joint_priors = []
    assert engine.get_lnprior_from_params([1.0, 2.0, 3.0]) == 0.0
    seen = []
    engine.set_joint_priors([(['b_mod', 'a_mod'], lambda v: seen.append(list(v)) or -1.5)])
    assert engine.get_lnprior_from_params([1.0, 2.0, 3.0]) == -1.5
    assert seen == [[3.0, 2.0]]
    engine.set_joint_priors([(['a_mod'], lambda v: -np.inf)])
    assert engine.get_lnprior_from_params([1.0, 2.0, 3.0]) == -np.inf
    # the per-parameter bounds still apply
    engine.set_joint_priors([(['a_mod'], lambda v: 0.0)])
    assert engine.get_lnprior_from_params([1.0, 20.0, 3.0]) == -np.inf
    with pytest.raises(ValueError, match='not a calibrated parameter'):
        engine.set_joint_priors([(['nope'], lambda v: 0.0)])


# --------------------------------------------------------------------------------------------
# running (generation + calibration; each run is a few seconds)
# --------------------------------------------------------------------------------------------

def _simulate(model_path, names):
    from libcuflynx.solver_wrappers import get_simulation_helper
    helper = get_simulation_helper(model_path=model_path, model_type='cellml',
                                   solver='CVODE_myokit', dt=0.1, sim_time=1.0, pre_time=0.0)
    helper.run()
    return {n: float(np.asarray(helper.get_results([n], flatten=True)[0]).ravel()[-1])
            for n in names}


def _calibrate_joint(library, out_dir):
    """The twin/joint parameterisation calibrated the ordinary way: one obs_data, one
    params_for_id."""
    from libcuflynx.scripts.param_id_run_script import run_param_id
    target = Target('twin', 'v1', 'joint')
    generated = generate_parameterisation(target, os.path.join(out_dir, 'model'), 'joint',
                                          library_inputs={'module_library_dirs':
                                                          [library['modules']]})
    inp = copy.deepcopy(OPTIMISER_SETTINGS)
    inp.update({'file_prefix': 'joint', 'model_type': 'cellml', 'DEBUG': False, 'do_ad': False,
                'do_ia': False, 'do_uq': False, 'UQ_options': {},
                'resources_dir': generated['resources_dir'],
                'generated_models_dir': generated['generated_models_dir'],
                'input_param_file': 'joint_parameters.json',
                'param_id_obs_path': os.path.join(library['twin/joint'], 'joint_obs_data.json'),
                'params_for_id_file': os.path.join(library['twin/joint'],
                                                   'joint_params_for_id.json'),
                'param_id_output_dir': os.path.join(out_dir, 'param_id')})
    outcome = run_param_id(inp)
    best = np.load(os.path.join(outcome['output_dir'], 'best_param_vals.npy'))
    return {'A/p': float(best[0]), 'B/p': float(best[1])}


@pytest.mark.integration
def test_calibrating_independent_parts_apart_matches_calibrating_them_together(library,
                                                                               tmp_path):
    result = run_calibration_workflow(library['twin_split'],
                                      output_dir=str(tmp_path / 'split'))
    assert result['complete'] and result['order'] == ['fit_a', 'fit_b']
    split = _merged(result)

    joint = _calibrate_joint(library, str(tmp_path / 'joint'))

    assert split.keys() == joint.keys() == {'A/p', 'B/p'}
    for name in split:
        assert split[name] == pytest.approx(joint[name], rel=1e-4)
    assert split['A/p'] == pytest.approx(np.sqrt(Y_A / 2), rel=1e-4)
    assert split['B/p'] == pytest.approx(np.sqrt(Y_B / 3), rel=1e-4)

    # the merged target model and the jointly calibrated model predict the same observables
    calibrated_joint = generate_parameterisation(
        Target('twin', 'v1', 'joint'), str(tmp_path / 'joint_calibrated'), 'joint_calibrated',
        [{'variable_name': f'mod__{sub}/p', 'units': 'dimensionless',
          'value': repr(joint[f'{sub}/p']), 'data_reference': 'joint calibration'}
         for sub in ('A', 'B')],
        library_inputs={'module_library_dirs': [library['modules']]})
    names = ['mod__A/y', 'mod__B/y']
    merged_outputs = _simulate(result['target_model']['model_path'], names)
    joint_outputs = _simulate(calibrated_joint['model_path'], names)
    for name in names:
        assert merged_outputs[name] == pytest.approx(joint_outputs[name], rel=1e-4)
    assert merged_outputs['mod__A/y'] == pytest.approx(Y_A, abs=STD_A * 1e-3)
    assert merged_outputs['mod__B/y'] == pytest.approx(Y_B, abs=STD_B * 1e-3)

    # each step was calibrated with its own params_for_id, and only its own parameter
    for step_id, param_vessel in (('fit_a', 'mod'), ('fit_b', 'mod')):
        step = _read_json(str(tmp_path / 'split' / step_id / 'step_result.json'))
        assert step['inputs']['params_for_id']['path'].endswith(f'{step_id}_params_for_id.json')
        snapshots = [f for f in os.listdir(step['param_id_output_dir'])
                     if f.startswith(f'{step_id}_params_for_id_') and f.endswith('.json')]
        assert len(snapshots) == 1
        entries = params_for_id_entries(
            os.path.join(step['param_id_output_dir'], snapshots[0]))
        assert [e['targets'] for e in entries] == [[f'{param_vessel}/p']]
        assert [c['model_name'] for c in step['calibrated']] == ['p_mod']


@pytest.mark.integration
def test_fixed_values_flow_into_the_supermodule_and_editing_obs_data_redoes_it(library,
                                                                              tmp_path):
    out = str(tmp_path / 'chain')
    events = []
    result = run_calibration_workflow(library['chain_rest'], output_dir=out,
                                      progress=events.append)
    merged = _merged(result)
    assert merged['A/p'] == pytest.approx(np.sqrt(Y_A / 2), rel=1e-4)
    assert merged['C/q'] == pytest.approx(np.sqrt(Z_C - Y_A), rel=1e-4)
    assert [(e['event'], e.get('step')) for e in events] == [
        ('workflow_started', None), ('step_started', 'fit_a'), ('step_finished', 'fit_a'),
        ('step_started', 'rest'), ('step_finished', 'rest'), ('workflow_finished', None)]

    rest = _read_json(os.path.join(out, 'rest', 'step_result.json'))
    assert [(f['model_name'], f['from_step']) for f in rest['fixed']] == [('p_mod__A', 'fit_a')]
    # the generated step model's parameters, named as the model names them
    with open(os.path.join(os.path.dirname(rest['model_path']), 'rest_parameters.csv')) as f:
        generated = {r['variable_name']: float(r['value']) for r in csv.DictReader(f)}
    assert generated['p_mod__A'] == pytest.approx(merged['A/p'], rel=1e-12)

    # change fit_a's data: everything downstream changes with it
    obs_path = os.path.join(library['lin_a/fit_a'], 'fit_a_obs_data.json')
    obs = _read_json(obs_path)
    obs['data_items'][0]['value'] = 4.5
    _write_json(obs_path, obs)
    with pytest.raises(WorkflowError, match='has changed since "fit_a" ran'):
        run_calibration_workflow(library['chain_rest'], output_dir=out, from_step='rest')
    merged = _merged(run_calibration_workflow(library['chain_rest'], output_dir=out))
    assert merged['A/p'] == pytest.approx(1.5, rel=1e-4)
    assert merged['C/q'] == pytest.approx(np.sqrt(Z_C - 4.5), rel=1e-4)

    # rerunning only the last step reuses fit_a; editing rest's own data needs no more
    obs_path = os.path.join(library['chain/rest'], 'rest_obs_data.json')
    obs = _read_json(obs_path)
    obs['data_items'][0]['value'] = 20.5
    _write_json(obs_path, obs)
    events = []
    result = run_calibration_workflow(library['chain_rest'], output_dir=out, only='rest',
                                      progress=events.append, write_calibrated=True)
    assert ('step_skipped', 'fit_a') in [(e['event'], e.get('step')) for e in events]
    assert _merged(result)['C/q'] == pytest.approx(4.0, rel=1e-4)

    # the merged set, in the target parameterisation's naming, next to its parameters
    written = {r['variable_name']: r
               for r in read_parameters(result['calibrated_parameters_file'], MODULE)}
    assert result['calibrated_parameters_file'] == os.path.join(
        library['chain/rest'], 'rest_calibrated_parameters.json')
    assert float(written['A/p']['value']) == pytest.approx(1.5, rel=1e-6)
    assert float(written['C/q']['value']) == pytest.approx(4.0, rel=1e-6)
    assert 'step rest' in written['C/q']['data_reference']

    run = load_workflow_run(out)
    assert run['result']['complete'] and set(run['steps']) == {'fit_a', 'rest'}

    from libcuflynx.calibration_workflow import workflow_status
    status = workflow_status(library['chain_rest'], out)
    assert status['complete'] and status['steps']['rest']['status'] == 'done'
    # fit_a's data changes: fit_a is stale, and so is rest, which took its value
    obs_path = os.path.join(library['lin_a/fit_a'], 'fit_a_obs_data.json')
    obs = _read_json(obs_path)
    obs['data_items'][0]['value'] = 5.0
    _write_json(obs_path, obs)
    status = workflow_status(library['chain_rest'], out)
    assert status['steps']['fit_a']['stale'] and status['steps']['rest']['stale']
    assert not status['complete']


@pytest.mark.integration
def test_a_failing_step_is_reported_and_stops_the_workflow(library, tmp_path):
    write_params_for_id(os.path.join(library['lin_a/fit_a'], 'fit_a_params_for_id.json'),
                        [{'targets': ['mod/not_a_param'], 'min': 0.1, 'max': 5,
                          'name_for_plotting': 'x'}])
    events = []
    with pytest.raises(Exception):
        run_calibration_workflow(library['chain_rest'], output_dir=str(tmp_path / 'fail'),
                                 progress=events.append)
    kinds = [(e['event'], e.get('step')) for e in events]
    assert ('step_failed', 'fit_a') in kinds
    assert ('step_started', 'rest') not in kinds


@pytest.mark.integration
@pytest.mark.slow
def test_a_stored_posterior_becomes_the_prior_of_a_later_step(library, tmp_path):
    uq = {'num_steps': 200, 'num_walkers': 8, 'burn_in': 0.5}
    raw = _raw(library)
    raw['workflow_name'] = 'chain_prior'
    raw['steps'][0].update(store_distribution=True, settings={'UQ_options': uq})
    raw['steps'][1].pop('fixed_from')
    raw['steps'][1].update(priors_from=[{'step': 'fit_a', 'kind': 'mvnormal'}],
                           store_distribution=True, settings={'UQ_options': uq})
    path = os.path.join(library['chain/rest'], 'prior_workflow.json')
    _write_json(path, raw)
    out = str(tmp_path / 'prior')
    result = run_calibration_workflow(path, output_dir=out)

    fit_a = StoredDistribution.load(os.path.join(out, 'fit_a'))
    assert fit_a.names == ['p_mod']
    # y = 2 p^2, std 0.08 at y = 8: sd(p) = 0.08 / (4 p) = 0.01
    assert np.median(fit_a.samples) == pytest.approx(2.0, abs=0.01)
    assert np.std(fit_a.samples) == pytest.approx(0.01, rel=0.5)

    rest = _read_json(os.path.join(out, 'rest', 'step_result.json'))
    assert rest['priors'][0]['parameters'] == ['p_mod__A']
    # with priors and a cost-minimising optimiser, the values are the posterior's medians
    assert rest['point_estimate'] == 'posterior_median'
    targets = [e['targets'] for e in
               params_for_id_entries(os.path.join(out, 'rest', 'rest_params_for_id.json'))]
    assert targets == [['mod/C/q'], ['mod/A/p']]
    posterior = StoredDistribution.load(os.path.join(out, 'rest'))
    p_a = posterior.marginal('p_mod__A')
    # z = 2 p_A^2 + q^2 alone is a ridge; the prior from fit_a is what pins p_A
    assert np.median(p_a) == pytest.approx(2.0, abs=0.03)
    assert np.std(p_a) < 0.03
    merged = _merged(result)
    assert merged['A/p'] == pytest.approx(2.0, abs=0.03)
    assert merged['C/q'] == pytest.approx(3.0, abs=0.1)


@pytest.mark.integration
def test_a_tab_shows_each_model_with_the_values_available_so_far(library, tmp_path):
    from libcuflynx.calibration_workflow import workflow_model

    out, work = str(tmp_path / 'run'), str(tmp_path / 'views')

    def parameters(view):
        with open(view['parameters_path']) as f:
            return {r['variable_name']: float(r['value']) for r in csv.DictReader(f)}

    rest = workflow_model(library['chain_rest'], 'rest', output_dir=out, work_dir=work)
    assert rest['kind'] == 'step' and rest['waiting_for'] == ['fit_a'] and rest['fixed'] == []
    assert rest['obs_data_path'].endswith('rest_obs_data.json')
    assert os.path.isfile(rest['flat_model_path'])
    assert parameters(rest)['p_mod__A'] == 1.0     # the parameterisation default, nothing fixed yet

    run_calibration_workflow(library['chain_rest'], output_dir=out, only='fit_a')
    rest = workflow_model(library['chain_rest'], 'rest', output_dir=out, work_dir=work)
    assert rest['waiting_for'] == [] and rest['calibrated'] == []
    assert [(f['model_name'], f['from_step']) for f in rest['fixed']] == [('p_mod__A', 'fit_a')]
    assert parameters(rest)['p_mod__A'] == pytest.approx(np.sqrt(Y_A / 2), rel=1e-4)

    fit_a = workflow_model(library['chain_rest'], 'fit_a', output_dir=out, work_dir=work)
    assert fit_a['place'] == 'A' and fit_a['param_id_output_dir']
    assert parameters(fit_a)['p_mod'] == pytest.approx(np.sqrt(Y_A / 2), rel=1e-4)

    target = workflow_model(library['chain_rest'], 'target', output_dir=out, work_dir=work)
    assert target['kind'] == 'target' and target['step_id'] == 'rest'
    assert target['waiting_for'] == ['rest']
    assert [c['model_name'] for c in target['calibrated']] == ['p_mod__A']
    assert parameters(target)['p_mod__A'] == pytest.approx(np.sqrt(Y_A / 2), rel=1e-4)

    # an edited input marks the result it fed as stale
    obs_path = os.path.join(library['lin_a/fit_a'], 'fit_a_obs_data.json')
    obs = _read_json(obs_path)
    obs['data_items'][0]['std'] = 0.1
    _write_json(obs_path, obs)
    assert workflow_model(library['chain_rest'], 'rest', output_dir=out,
                          work_dir=work)['stale'] == ['fit_a']


@pytest.mark.integration
def test_a_run_made_step_by_step_writes_its_calibrated_parameters_where_asked(library,
                                                                              tmp_path):
    from libcuflynx.calibration_workflow import write_calibrated_parameters_to

    out, path = str(tmp_path / 'run'), str(tmp_path / 'send' / 'chain_parameters.json')
    run_calibration_workflow(library['chain_array'], output_dir=out, only='a-alone')
    with pytest.raises(WorkflowError, match=r"steps \['chain'\] have no current result"):
        write_calibrated_parameters_to(library['chain_array'], out, path)
    run_calibration_workflow(library['chain_array'], output_dir=out, from_step='chain')
    before = {name: open(os.path.join(library['arrays'], name), 'rb').read()
              for name in sorted(os.listdir(library['arrays']))
              if os.path.isfile(os.path.join(library['arrays'], name))}
    assert write_calibrated_parameters_to(library['chain_array'], out, path) == path
    rows = {r['variable_name']: r for r in read_parameters(path)}
    assert float(rows['A/p']['value']) == pytest.approx(np.sqrt(Y_A / 2), rel=1e-4)
    assert float(rows['C/q']['value']) == pytest.approx(np.sqrt(Z_C - Y_A), rel=1e-4)
    assert rows['C/k']['value'] == '1.0'                # not calibrated: as in the target
    assert 'step chain' in rows['C/q']['data_reference']
    # only path is written: the target's own files are as they were, and nothing is added
    assert {name: open(os.path.join(library['arrays'], name), 'rb').read()
            for name in sorted(os.listdir(library['arrays']))
            if os.path.isfile(os.path.join(library['arrays'], name))} == before

    obs_path = os.path.join(library['arrays'], 'data', 'chain_obs.json')
    obs = _read_json(obs_path)
    obs['data_items'][0]['std'] = 0.2
    _write_json(obs_path, obs)
    with pytest.raises(WorkflowError, match=r"steps \['chain'\]"):
        write_calibrated_parameters_to(library['chain_array'], out, path)


@pytest.mark.unit
def test_calibrated_values_replace_their_rows_and_new_ones_are_added(tmp_path):
    from libcuflynx.calibration_workflow.runner import write_calibrated_parameters

    class _Target:
        kind = 'library'
        paths = (('A',), ('C',))
        parameters_path = str(tmp_path / 'rest_parameters.json')

        def file(self, suffix):
            return str(tmp_path / f'rest_{suffix}')

    write_parameters(_Target.parameters_path, 'variable_name,units,value,data_reference,sourced\n'
                     'A/p,dimensionless,1.0,a,yes\nC/q,dimensionless,1.0,c,\n'
                     'global/T,kelvin,300,t,\n', MODULE)
    before = open(_Target.parameters_path).read()
    merged = [{'model_name': 'p_mod__A', 'parameterisation_name': 'A/p', 'value': 2.0,
               'units': 'dimensionless', 'from_step': 'fit_a'},
              {'model_name': 'q_mod__C', 'parameterisation_name': 'C/q', 'value': 3.0,
               'units': 'dimensionless', 'from_step': 'rest'},
              {'model_name': 'r_mod__A', 'parameterisation_name': 'A/r', 'value': 4.0,
               'units': 'dimensionless', 'from_step': 'fit_a'},
              {'model_name': 'T', 'parameterisation_name': 'global/T', 'value': 310.0,
               'units': 'kelvin', 'from_step': 'rest'}]
    path = write_calibrated_parameters(_Target(), merged, 'wf')
    assert path == str(tmp_path / 'rest_calibrated_parameters.json')
    rows = {r['variable_name']: r for r in read_parameters(path, MODULE)}
    assert {n: float(r['value']) for n, r in rows.items()} == \
        {'global/T': 310.0, 'A/p': 2.0, 'C/q': 3.0, 'A/r': 4.0}
    # a calibrated value is no longer sourced; its data_reference says where it came from
    assert rows['A/p']['sourced'] == 'no' and 'step fit_a' in rows['A/p']['data_reference']
    assert rows['A/r']['units'] == 'dimensionless'
    assert open(_Target.parameters_path).read() == before


@pytest.mark.unit
def test_a_nested_submodule_is_named_as_a_path():
    from types import SimpleNamespace
    from libcuflynx.calibration_workflow.resolve import _submodule_path
    target = {'submodules': [{'name': 'soma', 'module_type': 'S', 'module_subtype': 'v'},
                             {'name': 'x', 'module_type': 'M', 'module_subtype': 'v'}],
              'module_format': 'supermodule'}
    supermodules = {('S', 'v'): {'submodules': [{'name': 'i_M', 'module_type': 'M',
                                                 'module_subtype': 'v'}]}}

    def step(submodule):
        return SimpleNamespace(id='s', submodule=submodule,
                               model=Target('M', 'v', 'p'))

    workflow_target = Target('T', 'v', 'p')
    for name in ('soma/i_M', 'soma__i_M'):
        assert _submodule_path(step(name), workflow_target, target, supermodules) == 'soma__i_M'
    with pytest.raises(WorkflowError, match='"submodule" is "soma_i_M"'):
        _submodule_path(step('soma_i_M'), workflow_target, target, supermodules)
    assert _submodule_path(step('x'), workflow_target, target, supermodules) == 'x'
    with pytest.raises(WorkflowError, match=r"is at \['soma/i_M', 'x'\]"):
        _submodule_path(step('i_M'), workflow_target, target, supermodules)
    with pytest.raises(WorkflowError, match='say which with "submodule"'):
        _submodule_path(step(None), workflow_target, target, supermodules)


# --------------------------------------------------------------------------------------------
# module arrays: a workflow outside the module library's layout
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_module_array_steps_keep_their_vessel_names(library):
    steps = {s['id']: s for s in plan_workflow(library['chain_array'])['steps']}
    assert steps['a-alone']['calibrates'] == [{'model_name': 'p_A', 'target_model_name': 'p_A',
                                               'target_parameterisation_name': 'A/p'}]
    assert steps['chain']['fixed_from'] == [{'from_step': 'a-alone', 'model_name': 'p_A',
                                             'in_step': 'p_A'}]


@pytest.mark.unit
def test_a_vessel_map_renames_a_module_array_into_the_target(library):
    arrays = library['arrays']
    _write_json(os.path.join(arrays, 'aorta_module_array.json'), [
        {'name': 'aorta', 'module_type': 'lin_a', 'module_subtype': 'v1',
         'parameterisation': 'default', 'inp_instances': [], 'out_instances': []}])
    write_parameters(os.path.join(arrays, 'aorta_parameters.json'),
                     'variable_name,units,value,data_reference\naorta/p,dimensionless,1.0,x\n')
    write_params_for_id(os.path.join(arrays, 'data', 'aorta_params.json'),
                        [{'targets': ['aorta/p'], 'min': 0.1, 'max': 5.0,
                          'name_for_plotting': 'p'}])
    raw = _raw(library, 'chain_array')
    raw['steps'][0].update(model='aorta_module_array.json', params_for_id='data/aorta_params.json')
    with pytest.raises(WorkflowError, match='lands on vessel "aorta"'):
        resolve_workflow(parse_workflow(raw, path=library['chain_array']))
    raw['steps'][0]['vessel_map'] = {'aorta': 'A'}
    steps = {s['id']: s for s in plan_workflow(parse_workflow(raw, path=library['chain_array']))
             ['steps']}
    assert steps['a-alone']['calibrates'][0]['target_model_name'] == 'p_A'
    # the chain receives the value under its own name
    assert steps['chain']['fixed_from'][0]['in_step'] == 'p_A'


@pytest.mark.unit
def test_a_library_module_is_placed_in_a_module_array_target(library):
    raw = _raw(library, 'chain_array')
    raw['steps'][0] = _library_step(library, 'a-alone', 'lin_a', 'fit_a')
    workflow = parse_workflow(raw, path=library['chain_array'])
    steps = {s['id']: s for s in plan_workflow(workflow)['steps']}
    # lin_a occurs once in the chain array, as record A
    assert steps['a-alone']['place'] == 'A'
    assert steps['a-alone']['calibrates'][0]['target_model_name'] == 'p_A'
    assert steps['chain']['fixed_from'][0]['in_step'] == 'p_A'
    raw['steps'][0]['vessel'] = 'C'
    with pytest.raises(WorkflowError, match=r"lin_a v1 is at \['A'\]"):
        resolve_workflow(parse_workflow(raw, path=library['chain_array']))


@pytest.mark.integration
def test_a_workflow_of_module_arrays_runs_and_writes_beside_the_target(library, tmp_path):
    result = run_calibration_workflow(library['chain_array'], output_dir=str(tmp_path / 'run'),
                                      write_calibrated=True)
    merged = _merged(result)
    assert merged['A/p'] == pytest.approx(np.sqrt(Y_A / 2), rel=1e-4)
    assert merged['C/q'] == pytest.approx(np.sqrt(Z_C - Y_A), rel=1e-4)
    chain = _read_json(str(tmp_path / 'run' / 'chain' / 'step_result.json'))
    assert [(f['model_name'], f['from_step']) for f in chain['fixed']] == [('p_A', 'a-alone')]
    assert set(chain['inputs']) == {'obs_data', 'params_for_id', 'parameters', 'module_array'}
    assert result['calibrated_parameters_file'] == os.path.join(
        library['arrays'], 'chain_calibrated_parameters.json')
    written = {r['variable_name']: r for r in read_parameters(result['calibrated_parameters_file'])}
    # the target's own rows take the values; the uncalibrated k keeps its own and its source
    assert float(written['A/p']['value']) == pytest.approx(np.sqrt(Y_A / 2), rel=1e-6)
    assert float(written['C/q']['value']) == pytest.approx(np.sqrt(Z_C - Y_A), rel=1e-6)
    assert written['C/k']['value'] == '1.0' and 'doe2020gain' in written['C/k']['data_reference']


@pytest.mark.integration
def test_a_workflow_records_itself_in_one_study_manifest(library, tmp_path):
    from libcuflynx.utilities.cuflynx_study import read_study
    out = str(tmp_path / 'run')
    run_calibration_workflow(library['chain_array'], output_dir=out, write_calibrated=True)
    study = read_study(os.path.join(out, 'chain_array_cuflynx_study.json'))
    workflow = study['workflow']
    assert workflow['name'] == 'chain_array' and workflow['complete']
    assert workflow['file'] == os.path.join(out, 'calibration_workflow.json')
    assert workflow['source'] == library['chain_array']
    assert workflow['result'] == os.path.join(out, 'workflow_result.json')
    # the target, as the workflow names it and as it was generated with the merged values
    assert workflow['target']['model']['module_array'] == os.path.join(
        library['arrays'], 'chain_module_array.json')
    assert workflow['target']['generated_model'] == study['model']
    assert os.path.isfile(study['model'])
    assert workflow['target']['calibrated_parameters'] == os.path.join(
        library['arrays'], 'chain_calibrated_parameters.json')
    steps = {s['id']: s for s in workflow['steps']}
    assert [s['id'] for s in workflow['steps']] == ['a-alone', 'chain']
    assert steps['chain']['fixed_from'] == ['a-alone'] and steps['chain']['status'] == 'done'
    assert steps['a-alone']['obs_data'] == os.path.join(library['arrays'], 'data', 'a_obs.json')
    # each step's own manifest, a single calibration, links back to the workflow's
    for step_id, step in steps.items():
        own = read_study(step['manifest'])
        assert own['workflow'] == {'manifest': os.path.join(out, 'chain_array_cuflynx_study.json'),
                                   'step': step_id}
        assert own['runs'][0]['dir'] == step['run']

    # write_cuflynx_study: false writes no manifest, for the workflow or its steps
    raw = _read_json(library['chain_array'])
    raw['settings']['write_cuflynx_study'] = False
    quiet = os.path.join(library['arrays'], 'quiet.json')
    _write_json(quiet, raw)
    run_calibration_workflow(quiet, output_dir=str(tmp_path / 'quiet'))
    found = [f for _, _, files in os.walk(str(tmp_path / 'quiet')) for f in files
             if f.endswith('_cuflynx_study.json')]
    assert found == []


# --------------------------------------------------------------------------------------------
# steps that are not calibrations: sensitivity, reduction, emulation, identifiability
# --------------------------------------------------------------------------------------------

def _analysed(library, edit):
    raw = _raw(library, 'chain_analysed')
    edit(raw)
    return lambda: parse_workflow(raw, path=library['chain_analysed'])


def _steps(raw):
    return {s['id']: s for s in raw['steps']}


@pytest.mark.unit
@pytest.mark.parametrize('edit, message', [
    (lambda w: _steps(w)['screen'].update(kind='screening'), r"kind: 'screening' is not one of"),
    (lambda w: _steps(w)['reduce'].update(model=_steps(w)['screen']['model']),
     "a reduction step reads a sensitivity step's results and has no \['model'\]"),
    (lambda w: _steps(w)['reduce'].pop('sensitivity_from'), r"'sensitivity_from' is a required property"),
    (lambda w: _steps(w)['reduce'].update(reduction={'top_n': 0}), r"reduction\.top_n: 0 is less than the minimum of 1"),
    (lambda w: _steps(w)['reduce'].update(reduction={'top_n': 1, 'criterion': 'median'}),
     r"reduction\.criterion: 'median' is not one of"),
    (lambda w: _steps(w)['reduce'].update(sensitivity_from='fit_a'),
     '"sensitivity_from" names "fit_a", a calibration step; it takes a sensitivity step'),
    (lambda w: _steps(w)['screen']['settings']['sa_options'].update(method='local'),
     'reads Sobol indices'),
    (lambda w: _steps(w)['screen'].update(settings={}), 'needs "sa_options"'),
    (lambda w: _steps(w)['ia'].pop('result_from'), 'needs "result_from"'),
    (lambda w: _steps(w)['ia'].update(result_from='screen'),
     '"result_from" names "screen", a sensitivity step'),
    (lambda w: _steps(w)['fit'].update(fixed_from=['screen']),
     '"fixed_from" names "screen", a sensitivity step'),
    (lambda w: _steps(w)['fit'].update(params_for_id=_steps(w)['ia']['params_for_id']),
     'not both'),
    (lambda w: _steps(w)['fit'].update(result_from='fit_a'), 'a calibration step has no '
                                                             '"result_from"'),
    (lambda w: _steps(w)['screen'].update(store_distribution=True), 'for calibration steps'),
    (lambda w: _steps(w)['fit'].update(model={'module_type': 'lin_a', 'version': 'v1',
                                              'parameterisation': 'fit_a'}),
     'which is of a different model'),
    (lambda w: _steps(w)['fit'].update(reduction={'top_n': 1}), 'is for a reduction step'),
])
def test_invalid_analysis_steps_say_why(library, edit, message):
    with pytest.raises(WorkflowError, match=message):
        _analysed(library, edit)()


def _emulated(library):
    """chain_rest with ``rest`` calibrated, and Sobol-screened, on an emulator of the chain."""
    raw = _raw(library)
    fit_a, rest = raw['steps']
    on_emulator = {k: v for k, v in rest.items() if k != 'fixed_from'}
    raw['steps'] = [
        fit_a,
        dict(rest, id='emulate', kind='emulation', settings={'emulator_settings': {
            'num_train_samples': 32, 'sample_type': 'sobol', 'random_seed': 0, 'min_r2': 0.9,
            'n_iter': 2, 'n_splits': 2, 'models': 'PolynomialRegression'}}),
        dict(on_emulator, emulator_from='emulate'),
        dict(on_emulator, id='screen', kind='sensitivity', emulator_from='emulate',
             settings={'sa_options': {'method': 'sobol', 'sample_type': 'saltelli',
                                      'num_samples': 16}})]
    return raw


@pytest.mark.unit
@pytest.mark.parametrize('edit, message', [
    (lambda w: w['steps'][2].update(fixed_from=['fit_a']), 'no "fixed_from" or'),
    (lambda w: w['steps'][3].update(result_from='fit_a'), 'no "fixed_from" or "result_from"'),
    (lambda w: w['steps'][2].update(obs_data=w['steps'][0]['obs_data']),
     'names a different obs_data'),
    (lambda w: w['steps'][2].update(emulator_from='fit_a'),
     '"emulator_from" names "fit_a", a calibration step; it takes a emulation step'),
])
def test_a_step_on_an_emulator_runs_on_what_it_was_trained_on(library, edit, message):
    raw = _emulated(library)
    edit(raw)
    with pytest.raises(WorkflowError, match=message):
        parse_workflow(raw, path=library['chain_rest'])


@pytest.mark.unit
def test_a_step_on_a_reduced_emulator_names_the_same_reduction(library):
    raw = _raw(library, 'chain_pipeline')
    posterior = raw['steps'][-1]
    posterior.pop('params_for_id_from')
    posterior['params_for_id'] = os.path.relpath(
        os.path.join(library['chain/pipeline'], 'pipeline_params_for_id.json'),
        library['chain/pipeline'])
    with pytest.raises(WorkflowError, match='different params_for_id and params_for_id_from'):
        parse_workflow(raw, path=library['chain_pipeline'])


@pytest.mark.unit
def test_a_reduction_keeps_the_top_parameters_by_its_options(tmp_path):
    from libcuflynx.sensitivity_analysis.parameter_reduction import SobolReduction
    from libcuflynx.sensitivity_analysis.sobol_ranking import (rank_sobol_parameters,
                                                               read_sobol_indices)
    indices = tmp_path / 'all_outputs_n8_Sobol_indices.csv'
    # a matters most to y1, b on average, c to neither
    indices.write_text('Parameter,S1_y1,ST_y1,S1_y2,ST_y2\n'
                       'm/a,0.9,0.95,0.0,0.01\n'
                       'm/b,0.4,0.5,0.4,0.5\n'
                       'm/c,0.01,0.02,0.0,0.01\n')
    read = read_sobol_indices(str(indices))
    assert read['ST']['y2'] == {'m/a': 0.01, 'm/b': 0.5, 'm/c': 0.01}
    names = lambda ranked: [n for n, _ in ranked]  # noqa: E731
    assert names(rank_sobol_parameters(read, top_n=1)) == ['m/a']
    assert names(rank_sobol_parameters(read, top_n=1, criterion='mean')) == ['m/b']
    assert names(rank_sobol_parameters(read, top_n=3, index_type='S1')) == ['m/a', 'm/b', 'm/c']
    assert names(rank_sobol_parameters(read, top_n=3, threshold=0.1)) == ['m/a', 'm/b']

    source = tmp_path / 'params_for_id.json'
    entry = lambda v, lo, hi: {'targets': [f'm/{v}'], 'min': lo, 'max': hi,  # noqa: E731
                               'name_for_plotting': f'{v}_m'}
    write_params_for_id(source, [entry('a', 0.1, 1.0), entry('b', 0.2, 2.0),
                                 entry('c', 0.3, 3.0)])
    out = tmp_path / 'reduced.json'
    assert SobolReduction.write_params_for_id(str(source), str(out), ['m/c', 'm/a']) == 2
    # whole entries, bounds and all, in the source's order
    assert params_for_id_entries(out) == [entry('a', 0.1, 1.0), entry('c', 0.3, 3.0)]

    # the class a reduction step is: ranks, keeps and writes in one call
    reduction = SobolReduction(top_n=2, criterion='mean')
    assert reduction.reduce(str(indices), str(source), str(out)) == [
        ('m/b', 0.5), ('m/a', pytest.approx(0.48))]
    assert params_for_id_entries(out) == [entry('a', 0.1, 1.0), entry('b', 0.2, 2.0)]
    assert SobolReduction.from_options(reduction.as_dict()).as_dict() == {
        'top_n': 2, 'index_type': 'ST', 'criterion': 'mean', 'threshold': None}
    with pytest.raises(ValueError, match='no parameter scores'):
        SobolReduction(top_n=2, threshold=0.99).reduce(str(indices), str(source), str(out))
    for bad in ({'top_n': 0}, {'top_n': 1, 'index_type': 'S2'},
                {'top_n': 1, 'criterion': 'median'}, {'top_n': 1, 'threshold': -1}):
        with pytest.raises(ValueError):
            SobolReduction(**bad)


@pytest.mark.unit
def test_a_workflow_screens_reduces_calibrates_and_analyses_the_fit(library, tmp_path):
    out = str(tmp_path / 'run')
    plan = {s['id']: s for s in plan_workflow(library['chain_analysed'])['steps']}
    assert [plan[i]['kind'] for i in ('fit_a', 'screen', 'reduce', 'fit', 'ia')] == [
        'calibration', 'sensitivity', 'reduction', 'calibration', 'identifiability']
    assert plan['fit']['params_for_id_from'] == 'reduce' and plan['ia']['result_from'] == 'fit'
    # only calibrations land in the target
    assert plan['screen']['calibrates'] == [] and plan['ia']['calibrates'] == []

    result = run_calibration_workflow(library['chain_analysed'], output_dir=out)
    assert result['complete'] and result['order'] == ['fit_a', 'screen', 'reduce', 'fit', 'ia']
    step = lambda sid: _read_json(os.path.join(out, sid, 'step_result.json'))  # noqa: E731

    # q drives z over its range and p_A barely does: the reduction keeps q alone
    assert os.path.isfile(step('screen')['sobol_indices'])
    reduce = step('reduce')
    assert [s['parameter'] for s in reduce['selected']] == ['mod__C/q']
    assert [e['targets'] for e in params_for_id_entries(reduce['params_for_id'])] == [
        ['mod/C/q']]

    # fit calibrates what was kept, with p_A fixed from fit_a, as chain/rest does
    fit = step('fit')
    assert fit['params_for_id_used'] == reduce['params_for_id']
    assert _merged(result) == pytest.approx({'A/p': np.sqrt(Y_A / 2),
                                             'C/q': np.sqrt(Z_C - Y_A)}, rel=1e-5)
    assert {r['parameterisation_name']: r['from_step'] for r in result['merged_parameters']} \
        == {'A/p': 'fit_a', 'C/q': 'fit'}

    # the Laplace analysis is at fit's q: dz/dq = 2 q, so std(q) = STD_C / (2 q) (to the
    # finite-difference error; at the model's own q = 1 it would be three times as wide)
    ia = step('ia')
    q = np.sqrt(Z_C - Y_A)
    assert ia['at'] == pytest.approx([q], rel=1e-5)
    assert ia['laplace']['std'] == pytest.approx([STD_C / (2 * q)], rel=1e-2)

    # the manifest lists every step's kind, what it took from where, and its outputs
    from libcuflynx.utilities.cuflynx_study import read_study
    study = read_study(os.path.join(out, 'chain_analysed_cuflynx_study.json'))
    steps = {s['id']: s for s in study['workflow']['steps']}
    assert {i: s['kind'] for i, s in steps.items()} == {
        'fit_a': 'calibration', 'screen': 'sensitivity', 'reduce': 'reduction',
        'fit': 'calibration', 'ia': 'identifiability'}
    assert steps['reduce']['sensitivity_from'] == ['screen']
    assert steps['reduce']['reduction'] == {'top_n': 1, 'index_type': 'ST', 'criterion': 'max',
                                            'threshold': None}
    assert steps['fit']['params_for_id_from'] == ['reduce']
    assert steps['ia']['result_from'] == ['fit']
    assert steps['reduce']['outputs'] == {'params_for_id': reduce['params_for_id']}
    assert steps['ia']['outputs']['laplace']['std'] == ia['laplace']['std']

    # editing the screen's params_for_id makes everything downstream of it stale
    from libcuflynx.calibration_workflow.runner import workflow_status
    screened = os.path.join(library['chain/screened'], 'screened_params_for_id.json')
    with open(screened, 'a') as f:
        f.write('\n')
    status = workflow_status(library['chain_analysed'], out)['steps']
    assert {i: s['stale'] for i, s in status.items()} == {
        'fit_a': False, 'screen': True, 'reduce': True, 'fit': True, 'ia': True}


@pytest.mark.unit
def test_steps_on_an_emulator_run_the_emulation_steps_model(library, tmp_path):
    pytest.importorskip('autoemulate')
    raw = _emulated(library)
    path = os.path.join(library['chain/rest'], 'emulated.json')
    _write_json(path, raw)
    out = str(tmp_path / 'run')
    result = run_calibration_workflow(path, output_dir=out)
    step = lambda sid: _read_json(os.path.join(out, sid, 'step_result.json'))  # noqa: E731
    emulate = step('emulate')
    assert os.path.isdir(emulate['emulator_dir'])
    # rest runs a copy of emulate's model, so the emulator's fingerprint of it matches
    rest = step('rest')
    assert os.path.basename(rest['model_path']) == 'emulate.cellml'
    assert rest['model_path'].startswith(os.path.join(out, 'rest') + os.sep)
    with open(rest['model_path'], 'rb') as a, open(emulate['model_path'], 'rb') as b:
        assert a.read() == b.read()
    # rest's own fixed_from was dropped; the emulator was trained with p_A fixed from fit_a
    assert _merged(result)['C/q'] == pytest.approx(np.sqrt(Z_C - Y_A), rel=0.03)
    assert os.path.isfile(step('screen')['sobol_indices'])


# --------------------------------------------------------------------------------------------
# packing: every file a run needs, in one directory
# --------------------------------------------------------------------------------------------

def _files_under(directory):
    return sorted(os.path.relpath(os.path.join(root, f), directory)
                  for root, _, names in os.walk(directory) for f in names)


@pytest.mark.unit
@pytest.mark.parametrize('which, expected', [
    ('chain_analysed', {'A/p': np.sqrt(Y_A / 2), 'C/q': np.sqrt(Z_C - Y_A)}),
    ('chain_array', {'A/p': np.sqrt(Y_A / 2), 'C/q': np.sqrt(Z_C - Y_A)}),
])
def test_a_packed_workflow_runs_without_the_files_it_was_packed_from(tmp_path, which,
                                                                     expected):
    from libcuflynx.calibration_workflow import pack_workflow
    root = tmp_path / 'library'
    library = build_workflow_library(str(root))
    pack = str(tmp_path / 'pack')
    packed = pack_workflow(library[which], pack)
    assert packed['workflow'] == os.path.join(pack, 'calibration_workflow.json')
    assert packed['files'] == _files_under(pack)
    # inputs only: no step output directories, results or study manifests
    assert not any(f.endswith(('step_result.json', 'workflow_result.json',
                               '_cuflynx_study.json')) for f in packed['files'])
    raw = _read_json(packed['workflow'])
    assert raw['module_library_dirs'] == ['modules']
    if which == 'chain_analysed':
        # the module versions the workflow uses, not the whole library
        assert packed['modules'] == [os.path.join('modules', t, 'versions', 'v1')
                                     for t in ('chain', 'lin_a', 'lin_c')]
        steps = {s['id']: s for s in raw['steps']}
        assert steps['screen']['obs_data'] == 'inputs/screen/screened_obs_data.json'
        assert steps['fit']['obs_data'] == steps['screen']['obs_data']   # packed once
        assert 'params_for_id' not in steps['fit'] and 'obs_data' not in steps['reduce']
    else:
        assert raw['target'] == {'module_array': 'inputs/_target/chain_module_array.json',
                                 'parameters': 'inputs/_target/chain_parameters.json'}

    os.rename(str(root), str(tmp_path / 'moved'))      # the originals are gone
    result = run_calibration_workflow(packed['workflow'], output_dir=str(tmp_path / 'run'))
    assert result['complete']
    assert _merged(result) == pytest.approx(expected, rel=1e-5)


@pytest.mark.unit
def test_settings_paths_are_relative_to_the_workflow_and_are_packed(library, tmp_path):
    from libcuflynx.calibration_workflow import pack_workflow
    from libcuflynx.scripts.calibration_workflow_run_script import build_parser, run
    raw = _raw(library)
    funcs = os.path.join(library['chain/rest'], 'funcs', 'my_operations.py')
    os.makedirs(os.path.dirname(funcs))
    with open(funcs, 'w') as f:
        f.write('# user operation funcs\n')
    raw['settings']['operation_funcs_external_path'] = 'funcs/my_operations.py'
    raw['steps'][1]['settings'] = {'cost_funcs_external_path': 'funcs/my_operations.py'}
    path = os.path.join(library['chain/rest'], 'with_funcs.json')
    _write_json(path, raw)
    workflow = load_workflow(path)
    assert workflow.settings['operation_funcs_external_path'] == funcs
    assert workflow.step_settings(workflow.step('rest'))['cost_funcs_external_path'] == funcs

    pack = str(tmp_path / 'pack')
    packed = run(build_parser().parse_args([path, '--pack', pack]))   # the command's own path
    raw = _read_json(packed['workflow'])
    in_pack = 'settings/operation_funcs_external_path/my_operations.py'
    assert raw['settings']['operation_funcs_external_path'] == in_pack
    # one file named twice is packed once
    assert raw['steps'][1]['settings']['cost_funcs_external_path'] == in_pack
    with open(os.path.join(pack, in_pack)) as f:
        assert f.read() == '# user operation funcs\n'
    with pytest.raises(ValueError, match='not empty'):
        pack_workflow(path, pack)


@pytest.mark.unit
def test_the_command_reports_steps_that_are_not_calibrations(capsys):
    from libcuflynx.scripts.calibration_workflow_run_script import _print_progress
    _print_progress({'event': 'step_started', 'step': 'reduce', 'index': 2, 'total': 5,
                     'kind': 'reduction', 'model': None})
    _print_progress({'event': 'step_finished', 'step': 'reduce', 'best_cost': None,
                     'duration_s': 0.25})
    _print_progress({'event': 'step_finished', 'step': 'fit', 'best_cost': 1.5e-3,
                     'duration_s': 2.0})
    assert capsys.readouterr().out.splitlines() == [
        '[workflow] step 3/5: reduce (reduction, no model)',
        '[workflow] step reduce done: in 0.2 s',
        '[workflow] step fit done: best cost 0.0015 in 2.0 s']


@pytest.mark.integration
def test_a_whole_study_runs_from_submodule_calibration_to_mcmc_on_an_emulator(library,
                                                                             tmp_path):
    """Submodule calibration -> Sobol screen -> reduction -> emulator of what was kept ->
    MCMC on the emulator, in one workflow of the algebraic chain (about 10 s)."""
    pytest.importorskip('autoemulate')
    out = str(tmp_path / 'run')
    result = run_calibration_workflow(library['chain_pipeline'], output_dir=out)
    assert result['complete']
    assert result['order'] == ['fit_a', 'screen', 'reduce', 'emulate', 'posterior']
    step = lambda sid: _read_json(os.path.join(out, sid, 'step_result.json'))  # noqa: E731

    # the submodule's value is fixed in the chain for the screen and the emulator
    assert step('screen')['fixed'][0]['model_name'] == 'p_mod__A'
    assert step('screen')['fixed'][0]['value'] == pytest.approx(np.sqrt(Y_A / 2), rel=1e-5)

    # q drives z over its range and the gain k barely does: the reduction keeps q
    reduce = step('reduce')
    assert [s['parameter'] for s in reduce['selected']] == ['mod__C/q']
    assert [e['targets'] for e in params_for_id_entries(reduce['params_for_id'])] == [
        ['mod/C/q']]

    # the emulator is trained over the kept parameter alone, and the posterior is sampled on it
    emulate, posterior = step('emulate'), step('posterior')
    assert emulate['params_for_id_used'] == reduce['params_for_id']
    assert posterior['params_for_id_used'] == reduce['params_for_id']
    from libcuflynx.utilities.cuflynx_study import read_study
    study = read_study(os.path.join(out, 'posterior', 'param_id',
                                    'posterior_cuflynx_study.json'))
    assert study['emulator'] == emulate['emulator_dir']
    assert study['runs'][0]['analyses'] == ['CMA-ES', 'mcmc']

    # dz/dq = 2 q at q = 3, so sd(q) = STD_C / 6; the emulator is good to about 1 %
    distribution = StoredDistribution.load(posterior['distribution_dir'])
    assert distribution.names == ['q_mod__C']
    q = np.sqrt(Z_C - Y_A)
    assert np.median(distribution.samples) == pytest.approx(q, rel=0.03)
    assert np.std(distribution.samples) == pytest.approx(STD_C / (2 * q), rel=0.5)
    assert _merged(result) == pytest.approx({'A/p': np.sqrt(Y_A / 2), 'C/q': q}, rel=0.03)
    assert {r['parameterisation_name']: r['from_step'] for r in result['merged_parameters']} \
        == {'A/p': 'fit_a', 'C/q': 'posterior'}
