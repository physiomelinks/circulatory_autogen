"""Calibration workflows (libcuflynx.calibration_workflow, cuflynx-calibration-workflow).

A workflow calibrates several module instances in order, each against its own obs_data and
params_for_id, and merges the results into one supermodule instance. The fixture library is
libcuflynx.external_testing.workflow_library (shared with CUFLynx's tests): algebraic modules
whose optima are known in closed form.

* ``twin``: A and B are independent, so calibrating them apart (the ``twin/split`` workflow)
  must give what one calibration of both does (``twin/joint``: one obs_data, one
  params_for_id).
* ``chain``: C reads A's output (z = 2 p_A^2 + q^2), so the value fit_a fixes for p_A decides
  what ``rest`` finds for q: q = sqrt(Z_C - Y_A).
"""
import copy
import csv
import json
import os

import numpy as np
import pytest

from libcuflynx.calibration_workflow import (StoredDistribution, WorkflowError,
                                             generate_module_instance, load_workflow,
                                             load_workflow_run, plan_workflow, resolve_workflow,
                                             run_calibration_workflow)
from libcuflynx.calibration_workflow import naming
from libcuflynx.calibration_workflow.distributions import (DistributionParameter,
                                                           transform_for_bounds)
from libcuflynx.calibration_workflow.spec import Target, parse_workflow
from libcuflynx.external_testing.workflow_library import (OPTIMISER_SETTINGS, STD_A, STD_B, Y_A,
                                                          Y_B, Z_C, build_workflow_library)
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
    return {r['instance_name']: r['value'] for r in result['merged_parameters']}


# --------------------------------------------------------------------------------------------
# the file
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_target_defaults_to_the_instance_holding_the_file(library):
    workflow = load_workflow(library['twin_split'])
    assert workflow.target == Target('twin', 'v1', 'split')
    assert [s.id for s in workflow.topological_order()] == ['fit_a', 'fit_b']
    assert workflow.module_library_dirs == [library['modules']]


@pytest.mark.unit
def test_the_shipped_schema_accepts_the_fixture_workflows_and_rejects_unknown_keys(library):
    jsonschema = pytest.importorskip('jsonschema')
    schema = load_schema(CALIBRATION_WORKFLOW_SCHEMA)
    for which in ('twin_split', 'chain_rest'):
        jsonschema.validate(_raw(library, which), schema)
    raw = _raw(library)
    raw['steps'][0]['fixed_values'] = {}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(raw, schema)
    with pytest.raises(WorkflowError, match='unknown key'):
        parse_workflow(raw, path=library['chain_rest'])


@pytest.mark.unit
def test_order_follows_depends_on_not_file_order(library):
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
    (lambda w: w['steps'][0].update(depends_on=['rest']), 'cycle'),
    (lambda w: w['steps'][1].update(depends_on=['nope']), 'unknown step "nope"'),
    (lambda w: w['steps'][1].update(depends_on=[]), 'does not depend on'),
    (lambda w: w['steps'].append(dict(w['steps'][0])), 'more than once'),
    (lambda w: w['steps'].append(dict(w['steps'][0], id='again')), 'both calibrate'),
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
    assert naming.map_vessel('mod', 'i_M') == 'mod_i_M'
    assert naming.map_vessel('mod_A', 'soma') == 'mod_soma_A'
    assert naming.map_vessel('mod', '') == 'mod'
    assert naming.map_vessel('global', 'soma') == 'global'
    assert naming.model_name(naming.map_vessel('mod', 'i_M'), 'rho_M') == 'rho_M_mod_i_M'
    assert naming.model_name('global', 'R') == 'R'
    assert naming.instance_name('mod_soma_i_M', 'rho_M') == 'rho_M_soma_i_M'
    assert naming.instance_name('mod', 'p') == 'p'
    assert naming.relative_path('soma_i_M', 'soma') == 'i_M'
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
    assert steps['fit_a']['submodule_path'] == 'A'
    assert steps['fit_a']['calibrates'] == [{'model_name': 'p_mod', 'target_model_name': 'p_mod_A',
                                             'target_instance_name': 'p_A'}]
    assert steps['rest']['submodule_path'] == ''
    assert steps['rest']['fixed_from'] == [{'from_step': 'fit_a', 'model_name': 'p_mod',
                                            'in_step': 'p_mod_A'}]
    assert run_calibration_workflow(library['chain_rest'], dry_run=True) == plan


@pytest.mark.unit
def test_every_step_needs_its_own_params_for_id(library):
    os.remove(os.path.join(library['lin_b/fit_b'], 'fit_b_params_for_id.csv'))
    with pytest.raises(WorkflowError, match='no params_for_id file'):
        resolve_workflow(load_workflow(library['twin_split']))


@pytest.mark.unit
def test_a_parameter_belongs_to_one_step(library):
    raw = _raw(library, 'twin_split')
    raw['steps'].append({'id': 'joint', 'target': {'module_type': 'twin', 'version': 'v1',
                                                   'instance': 'joint'}})
    workflow = parse_workflow(raw, path=library['twin_split'])
    with pytest.raises(WorkflowError, match='calibrated by both step "fit_a" and step "joint"'):
        resolve_workflow(workflow)


@pytest.mark.unit
def test_a_step_must_land_in_the_target(library):
    raw = _raw(library)
    raw['steps'].append({'id': 'fit_b', 'target': {'module_type': 'lin_b', 'version': 'v1',
                                                   'instance': 'fit_b'}})
    with pytest.raises(WorkflowError, match='not a submodule of chain'):
        resolve_workflow(parse_workflow(raw, path=library['chain_rest']))


@pytest.mark.unit
def test_an_ambiguous_submodule_must_be_named(library):
    version_dir = os.path.join(library['modules'], 'double', 'versions', 'v1')
    os.makedirs(os.path.join(version_dir, 'instances', 'default'))
    sub = {'module_type': 'lin_a', 'module_subtype': 'v1', 'instance': 'default',
           'inp_instances': [], 'out_instances': []}
    _write_json(os.path.join(version_dir, 'double_v1_modules_config.json'), [{
        'module_type': 'double', 'module_subtype': 'v1', 'module_format': 'supermodule',
        'default_instance': 'default',
        'submodules': [dict(sub, name='A1'), dict(sub, name='A2')]}])
    with open(os.path.join(version_dir, 'instances', 'default', 'default_parameters.csv'),
              'w') as f:
        f.write('variable_name,units,value,data_reference\n')
    raw = {'workflow_name': 'double', 'module_library_dirs': [library['modules']],
           'target': {'module_type': 'double', 'version': 'v1', 'instance': 'default'},
           'steps': [{'id': 'fit_a', 'target': {'module_type': 'lin_a', 'version': 'v1',
                                                'instance': 'fit_a'}}]}
    with pytest.raises(WorkflowError, match=r"\['A1', 'A2'\] in double/v1/default; say which"):
        resolve_workflow(parse_workflow(raw))
    raw['steps'][0]['submodule'] = 'A2'
    resolved = resolve_workflow(parse_workflow(raw))
    assert resolved.steps['fit_a'].path == 'A2'


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
    mapped = loaded.mapped_into('soma_i_M')
    assert mapped.names == ['a_mod_soma_i_M', 'b_mod_soma_i_M']
    assert mapped.subset(['b_mod_soma_i_M']).names == ['b_mod_soma_i_M']
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
    """The twin/joint instance calibrated the ordinary way: one obs_data, one params_for_id."""
    from libcuflynx.scripts.param_id_run_script import run_param_id
    target = Target('twin', 'v1', 'joint')
    generated = generate_module_instance(target, os.path.join(out_dir, 'model'), 'joint',
                                         library_inputs={'module_library_dirs':
                                                         [library['modules']]})
    inp = copy.deepcopy(OPTIMISER_SETTINGS)
    inp.update({'file_prefix': 'joint', 'model_type': 'cellml', 'DEBUG': False, 'do_ad': False,
                'do_ia': False, 'do_uq': False, 'UQ_options': {},
                'resources_dir': generated['resources_dir'],
                'generated_models_dir': generated['generated_models_dir'],
                'input_param_file': 'joint_parameters.csv',
                'param_id_obs_path': os.path.join(library['twin/joint'], 'joint_obs_data.json'),
                'params_for_id_file': os.path.join(library['twin/joint'],
                                                   'joint_params_for_id.csv'),
                'param_id_output_dir': os.path.join(out_dir, 'param_id')})
    outcome = run_param_id(inp)
    best = np.load(os.path.join(outcome['output_dir'], 'best_param_vals.npy'))
    return {'p_A': float(best[0]), 'p_B': float(best[1])}


@pytest.mark.integration
def test_calibrating_independent_parts_apart_matches_calibrating_them_together(library,
                                                                               tmp_path):
    result = run_calibration_workflow(library['twin_split'],
                                      output_dir=str(tmp_path / 'split'))
    assert result['complete'] and result['order'] == ['fit_a', 'fit_b']
    split = _merged(result)

    joint = _calibrate_joint(library, str(tmp_path / 'joint'))

    assert split.keys() == joint.keys() == {'p_A', 'p_B'}
    for name in split:
        assert split[name] == pytest.approx(joint[name], rel=1e-4)
    assert split['p_A'] == pytest.approx(np.sqrt(Y_A / 2), rel=1e-4)
    assert split['p_B'] == pytest.approx(np.sqrt(Y_B / 3), rel=1e-4)

    # the merged target model and the jointly calibrated model predict the same observables
    calibrated_joint = generate_module_instance(
        Target('twin', 'v1', 'joint'), str(tmp_path / 'joint_calibrated'), 'joint_calibrated',
        [{'variable_name': f'p_mod_{sub}', 'units': 'dimensionless',
          'value': repr(joint[f'p_{sub}']), 'data_reference': 'joint calibration'}
         for sub in ('A', 'B')],
        library_inputs={'module_library_dirs': [library['modules']]})
    names = ['mod_A/y', 'mod_B/y']
    merged_outputs = _simulate(result['target_model']['model_path'], names)
    joint_outputs = _simulate(calibrated_joint['model_path'], names)
    for name in names:
        assert merged_outputs[name] == pytest.approx(joint_outputs[name], rel=1e-4)
    assert merged_outputs['mod_A/y'] == pytest.approx(Y_A, abs=STD_A * 1e-3)
    assert merged_outputs['mod_B/y'] == pytest.approx(Y_B, abs=STD_B * 1e-3)

    # each step was calibrated with its own params_for_id, and only its own parameter
    for step_id, param_vessel in (('fit_a', 'mod'), ('fit_b', 'mod')):
        step = _read_json(str(tmp_path / 'split' / step_id / 'step_result.json'))
        assert step['inputs']['params_for_id']['path'].endswith(f'{step_id}_params_for_id.csv')
        snapshots = [f for f in os.listdir(step['param_id_output_dir'])
                     if f.startswith(f'{step_id}_params_for_id_') and f.endswith('.csv')]
        assert len(snapshots) == 1
        with open(os.path.join(step['param_id_output_dir'], snapshots[0])) as f:
            rows = list(csv.DictReader(f))
        assert [(r['vessel_name'], r['param_name']) for r in rows] == [(param_vessel, 'p')]
        assert [c['model_name'] for c in step['calibrated']] == ['p_mod']


@pytest.mark.integration
def test_fixed_values_flow_into_the_supermodule_and_editing_obs_data_redoes_it(library,
                                                                              tmp_path):
    out = str(tmp_path / 'chain')
    events = []
    result = run_calibration_workflow(library['chain_rest'], output_dir=out,
                                      progress=events.append)
    merged = _merged(result)
    assert merged['p_A'] == pytest.approx(np.sqrt(Y_A / 2), rel=1e-4)
    assert merged['q_C'] == pytest.approx(np.sqrt(Z_C - Y_A), rel=1e-4)
    assert [(e['event'], e.get('step')) for e in events] == [
        ('workflow_started', None), ('step_started', 'fit_a'), ('step_finished', 'fit_a'),
        ('step_started', 'rest'), ('step_finished', 'rest'), ('workflow_finished', None)]

    rest = _read_json(os.path.join(out, 'rest', 'step_result.json'))
    assert [(f['model_name'], f['from_step']) for f in rest['fixed']] == [('p_mod_A', 'fit_a')]
    with open(os.path.join(os.path.dirname(rest['model_path']), 'rest_parameters.csv')) as f:
        generated = {r['variable_name']: float(r['value']) for r in csv.DictReader(f)}
    assert generated['p_mod_A'] == pytest.approx(merged['p_A'], rel=1e-12)

    # change fit_a's data: everything downstream changes with it
    obs_path = os.path.join(library['lin_a/fit_a'], 'fit_a_obs_data.json')
    obs = _read_json(obs_path)
    obs['data_items'][0]['value'] = 4.5
    _write_json(obs_path, obs)
    with pytest.raises(WorkflowError, match='has changed since "fit_a" ran'):
        run_calibration_workflow(library['chain_rest'], output_dir=out, from_step='rest')
    merged = _merged(run_calibration_workflow(library['chain_rest'], output_dir=out))
    assert merged['p_A'] == pytest.approx(1.5, rel=1e-4)
    assert merged['q_C'] == pytest.approx(np.sqrt(Z_C - 4.5), rel=1e-4)

    # rerunning only the last step reuses fit_a; editing rest's own data needs no more
    obs_path = os.path.join(library['chain/rest'], 'rest_obs_data.json')
    obs = _read_json(obs_path)
    obs['data_items'][0]['value'] = 20.5
    _write_json(obs_path, obs)
    events = []
    result = run_calibration_workflow(library['chain_rest'], output_dir=out, only='rest',
                                      progress=events.append, write_calibrated=True)
    assert ('step_skipped', 'fit_a') in [(e['event'], e.get('step')) for e in events]
    assert _merged(result)['q_C'] == pytest.approx(4.0, rel=1e-4)

    # the merged set, in the target instance's naming, next to its parameters
    with open(result['calibrated_parameters_file']) as f:
        written = {r['variable_name']: r for r in csv.DictReader(f)}
    assert result['calibrated_parameters_file'] == os.path.join(
        library['chain/rest'], 'rest_calibrated_parameters.csv')
    assert float(written['p_A']['value']) == pytest.approx(1.5, rel=1e-6)
    assert float(written['q_C']['value']) == pytest.approx(4.0, rel=1e-6)
    assert 'step rest' in written['q_C']['data_reference']

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
    with open(os.path.join(library['lin_a/fit_a'], 'fit_a_params_for_id.csv'), 'w') as f:
        f.write('vessel_name,param_name,min,max,name_for_plotting\nmod,not_a_param,0.1,5,x\n')
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
    assert rest['priors'][0]['parameters'] == ['p_mod_A']
    # with priors and a cost-minimising optimiser, the values are the posterior's medians
    assert rest['point_estimate'] == 'posterior_median'
    with open(os.path.join(out, 'rest', 'rest_params_for_id.csv')) as f:
        rows = [(r['vessel_name'], r['param_name']) for r in csv.DictReader(f)]
    assert rows == [('mod_C', 'q'), ('mod_A', 'p')]
    posterior = StoredDistribution.load(os.path.join(out, 'rest'))
    p_a = posterior.marginal('p_mod_A')
    # z = 2 p_A^2 + q^2 alone is a ridge; the prior from fit_a is what pins p_A
    assert np.median(p_a) == pytest.approx(2.0, abs=0.03)
    assert np.std(p_a) < 0.03
    merged = _merged(result)
    assert merged['p_A'] == pytest.approx(2.0, abs=0.03)
    assert merged['q_C'] == pytest.approx(3.0, abs=0.1)


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
    assert parameters(rest)['p_mod_A'] == 1.0     # the instance default, nothing fixed yet

    run_calibration_workflow(library['chain_rest'], output_dir=out, only='fit_a')
    rest = workflow_model(library['chain_rest'], 'rest', output_dir=out, work_dir=work)
    assert rest['waiting_for'] == [] and rest['calibrated'] == []
    assert [(f['model_name'], f['from_step']) for f in rest['fixed']] == [('p_mod_A', 'fit_a')]
    assert parameters(rest)['p_mod_A'] == pytest.approx(np.sqrt(Y_A / 2), rel=1e-4)

    fit_a = workflow_model(library['chain_rest'], 'fit_a', output_dir=out, work_dir=work)
    assert fit_a['submodule_path'] == 'A' and fit_a['param_id_output_dir']
    assert parameters(fit_a)['p_mod'] == pytest.approx(np.sqrt(Y_A / 2), rel=1e-4)

    target = workflow_model(library['chain_rest'], 'target', output_dir=out, work_dir=work)
    assert target['kind'] == 'target' and target['step_id'] == 'rest'
    assert target['waiting_for'] == ['rest']
    assert [c['model_name'] for c in target['calibrated']] == ['p_mod_A']
    assert parameters(target)['p_mod_A'] == pytest.approx(np.sqrt(Y_A / 2), rel=1e-4)

    # an edited input marks the result it fed as stale
    obs_path = os.path.join(library['lin_a/fit_a'], 'fit_a_obs_data.json')
    obs = _read_json(obs_path)
    obs['data_items'][0]['std'] = 0.1
    _write_json(obs_path, obs)
    assert workflow_model(library['chain_rest'], 'rest', output_dir=out,
                          work_dir=work)['stale'] == ['fit_a']
