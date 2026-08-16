"""The emulator as a SimulationHelper, and the cost path that runs on it (issue #333).

These build a real ``CVS0DParamID`` over a stub emulator and no solver at all -- which is the
point of putting the emulator behind the helper interface: nothing above it has to change, and
nothing below it has to exist. autoemulate is not needed, so these run everywhere.

The obs_data and params_for_id come from ``Simple_ODE_Benchmark`` (two parameters, two scalar
features), matching test_emulator_training.py. Nothing is simulated here -- they are only a
realistic shape for the parsers to produce.
"""
import os

import numpy as np
import pytest

from libcuflynx.emulators.emulator_bundle import EmulatorBundle, fingerprint
from libcuflynx.param_id.paramID import CVS0DParamID, emulated_feature_labels
from libcuflynx.parsers.PrimitiveParsers import (ObsAndParamDataParser, YamlFileParser,
                                      param_entry_labels)

pytestmark = pytest.mark.unit


class LinearStub:
    """features = x @ w, where x is the unit-box theta the bundle hands over."""

    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)

    def predict(self, x):
        return np.asarray(x, dtype=float) @ self.weights


def _config(base_user_inputs, resources_dir, tmp_path, **overrides):
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': 'Simple_ODE_Benchmark',
        'input_param_file': 'Simple_ODE_Benchmark_parameters.csv',
        'model_type': 'cellml_only',
        'resources_dir': resources_dir,
        'param_id_obs_path': os.path.join(resources_dir,
                                          'Simple_ODE_Benchmark_obs_data.json'),
        'params_for_id_path': os.path.join(resources_dir,
                                           'Simple_ODE_Benchmark_params_for_id.csv'),
        'param_id_output_dir': str(tmp_path / 'param_id_output'),
        'param_id_method': 'genetic_algorithm',
        'DEBUG': True,
        'use_emulator': True,
        'emulator_settings': {'emulator_dir': str(tmp_path / 'emulator'), 'min_r2': 0.5},
    })
    config.update(overrides)
    return YamlFileParser().parse_user_inputs_file(
        config, obs_path_needed=True, do_generation_with_fit_parameters=False)


def _parse_infos(config):
    """obs_info / protocol_info / param_id_info, via the same parser CVS0DParamID uses."""
    # process_obs_info writes the ground-truth npys, so it needs somewhere to write them.
    scratch = os.path.join(config['emulator_settings']['emulator_dir'], '_parsed')
    os.makedirs(scratch, exist_ok=True)
    parser = ObsAndParamDataParser()
    parsed = parser.parse_obs_data_json(
        param_id_obs_path=config['param_id_obs_path'],
        pre_time=config['pre_time'], sim_time=config['sim_time'],
        model_type=config['model_type'], method=config['solver_info'].get('method'))
    obs_info = parser.process_obs_info(gt_df=parsed['gt_df'], output_dir=scratch,
                                       dt=config['dt'])
    protocol_info = parser.process_protocol_and_weights(
        gt_df=parsed['gt_df'], protocol_info=parsed['protocol_info'], dt=config['dt'])
    param_id_info = parser.get_param_id_info(config['params_for_id_path'])
    return obs_info, protocol_info, param_id_info


def _write_bundle(config, weights=None, feature_r2=None, fingerprint_override=None,
                  param_maxs=None):
    """Write a stub bundle that matches this config, the way training would."""
    obs_info, protocol_info, param_id_info = _parse_infos(config)
    if param_maxs is not None:
        param_id_info['param_maxs'] = np.asarray(param_maxs, dtype=float)
    labels = emulated_feature_labels(obs_info)
    num_params = len(param_id_info['param_names'])

    if weights is None:
        # Something varying in every parameter, so a finite-difference gradient is non-trivial.
        weights = np.arange(1, num_params * len(labels) + 1, dtype=float).reshape(
            num_params, len(labels))
    digest = fingerprint(param_id_info, obs_info, protocol_info)
    if fingerprint_override is not None:
        digest = {'inputs_sha256': fingerprint_override}

    meta = {
        'param_entry_labels': param_entry_labels(param_id_info),
        'param_mins': [float(v) for v in param_id_info['param_mins']],
        'param_maxs': [float(v) for v in param_id_info['param_maxs']],
        'param_names': [[str(n) for n in entry] for entry in param_id_info['param_names']],
        'param_defaults': {str(name): 1.0
                           for entry in param_id_info['param_names'] for name in entry},
        'feature_labels': labels,
        'feature_r2': list(feature_r2) if feature_r2 else [0.99] * len(labels),
        'feature_rmse': [0.01] * len(labels),
        'x_scale': {'shift': [float(v) for v in param_id_info['param_mins']],
                    'span': [float(hi - lo) for lo, hi in zip(param_id_info['param_mins'],
                                                              param_id_info['param_maxs'])]},
        'y_scale': {'shift': [0.0] * len(labels), 'span': [1.0] * len(labels)},
        'fingerprint': digest,
    }
    bundle = EmulatorBundle(LinearStub(weights), meta)
    bundle.save(config['emulator_settings']['emulator_dir'])
    return bundle, obs_info, param_id_info


# --------------------------------------------------------------------------- the helper alone

def test_helper_returns_one_predicted_feature_per_data_item(base_user_inputs, resources_dir,
                                                            tmp_path):
    from libcuflynx.solver_wrappers import get_simulation_helper

    config = _config(base_user_inputs, resources_dir, tmp_path)
    bundle, obs_info, param_id_info = _write_bundle(config)

    helper = get_simulation_helper(
        solver='CVODE_myokit', model_type='cellml_only', model_path='not/used.cellml',
        dt=0.01, sim_time=5.0, use_emulator=True,
        emulator_dir=config['emulator_settings']['emulator_dir'])
    assert helper.emulates_features is True

    helper.set_obs_map(obs_info['const_idx_to_obs_idx'], num_obs=obs_info['num_obs'])
    theta = np.asarray(param_id_info['param_mins'], dtype=float)
    helper.set_theta(theta)
    assert helper.run() is True

    results = helper.get_results(obs_info['operands'])
    assert len(results) == obs_info['num_obs']
    expected = bundle.predict(theta)
    for item_idx, operands in enumerate(obs_info['operands']):
        assert len(results[item_idx]) == len(operands)
        for operand_result in results[item_idx]:
            # A length-1 array per operand: the shape a solver returns, holding the feature.
            assert operand_result.shape == (1,)
            assert operand_result[0] == pytest.approx(expected[item_idx])


def test_helper_keeps_the_time_axis_the_executor_relies_on(base_user_inputs, resources_dir,
                                                           tmp_path):
    """Nothing is integrated, but the protocol executor still concatenates tSim per
    sub-experiment and drops the duplicated first sample. A single-point axis would break it."""
    from libcuflynx.solver_wrappers import get_simulation_helper

    config = _config(base_user_inputs, resources_dir, tmp_path)
    _write_bundle(config)
    helper = get_simulation_helper(
        solver='CVODE_myokit', model_type='cellml_only', model_path='not/used.cellml',
        dt=0.01, sim_time=5.0, use_emulator=True,
        emulator_dir=config['emulator_settings']['emulator_dir'])

    helper.update_times(0.01, 0.0, 2.0, 1.0)
    assert len(helper.tSim) == 201
    assert helper.tSim[0] == pytest.approx(1.0)
    assert helper.get_time()[0] == pytest.approx(0.0)
    # A sim_time shorter than dt must still leave something to concatenate.
    helper.update_times(0.1, 0.0, 0.01, 0.0)
    assert len(helper.tSim) >= 2


def test_helper_refuses_to_invent_traces(base_user_inputs, resources_dir, tmp_path):
    from libcuflynx.solver_wrappers import get_simulation_helper

    config = _config(base_user_inputs, resources_dir, tmp_path)
    _write_bundle(config)
    helper = get_simulation_helper(
        solver='CVODE_myokit', model_type='cellml_only', model_path='not/used.cellml',
        dt=0.01, sim_time=5.0, use_emulator=True,
        emulator_dir=config['emulator_settings']['emulator_dir'])

    with pytest.raises(NotImplementedError, match='trace'):
        helper.get_all_results_dict()
    with pytest.raises(NotImplementedError, match='trace'):
        helper.get_all_results()


def test_helper_refuses_a_parameter_it_cannot_see(base_user_inputs, resources_dir, tmp_path):
    """Silently ignoring a set_param_vals it cannot honour would be a wrong answer, not an
    approximation: the caller believes it changed something."""
    from libcuflynx.solver_wrappers import get_simulation_helper

    config = _config(base_user_inputs, resources_dir, tmp_path)
    _write_bundle(config)
    helper = get_simulation_helper(
        solver='CVODE_myokit', model_type='cellml_only', model_path='not/used.cellml',
        dt=0.01, sim_time=5.0, use_emulator=True,
        emulator_dir=config['emulator_settings']['emulator_dir'])

    with pytest.raises(ValueError, match='cannot change'):
        helper.set_param_vals([['some/other_param']], [3.0])


def test_helper_serves_defaults_recorded_at_training_time(base_user_inputs, resources_dir,
                                                          tmp_path):
    """x0 and modifier baselines are read before anything simulates, and the emulator has no
    model to read them from -- so training records them and the helper serves them."""
    from libcuflynx.solver_wrappers import get_simulation_helper

    config = _config(base_user_inputs, resources_dir, tmp_path)
    _, _, param_id_info = _write_bundle(config)
    helper = get_simulation_helper(
        solver='CVODE_myokit', model_type='cellml_only', model_path='not/used.cellml',
        dt=0.01, sim_time=5.0, use_emulator=True,
        emulator_dir=config['emulator_settings']['emulator_dir'])

    values = helper.get_init_param_vals(param_id_info['param_names'])
    assert values == [1.0] * len(param_id_info['param_names'])
    with pytest.raises(ValueError, match='no recorded default'):
        helper.get_init_param_vals([['never/trained']])


# --------------------------------------------------------------- the cost path over an emulator

def test_cost_path_uses_the_predicted_features_and_skips_the_operation(
        base_user_inputs, resources_dir, tmp_path):
    """The whole point of the seam.

    Both benchmark observables are ``steady_state_avg`` of a trace -- the mean of its second
    half. Re-applying that to an already-reduced scalar is invisible (the mean of one value is
    itself), and for an operation like ``max_minus_min`` it would silently be zero. So this has
    to be verified as "the operation did not run", not "the number looked plausible", and
    get_obs_output_dict returning exactly the emulator's vector is that check.
    """
    config = _config(base_user_inputs, resources_dir, tmp_path)
    bundle, _, _ = _write_bundle(config)

    pid = CVS0DParamID.init_from_dict(config)
    engine = pid.param_id
    assert engine.emulates_features is True

    theta = np.asarray(engine.param_id_info['param_mins'], dtype=float) * 1.5
    cost, operands_list, _ = engine.get_cost_obs_and_pred_from_params(theta)
    assert np.isfinite(cost)

    obs_dict = engine.get_obs_output_dict(operands_list[0])
    assert np.asarray(obs_dict['const']) == pytest.approx(bundle.predict(theta))


def test_gradient_over_an_emulator_is_finite_differences(base_user_inputs, resources_dir,
                                                         tmp_path):
    """get_gradient must not raise here, because sp_minimize calls it unconditionally."""
    config = _config(base_user_inputs, resources_dir, tmp_path)
    _write_bundle(config)

    engine = CVS0DParamID.init_from_dict(config).param_id
    mins = np.asarray(engine.param_id_info['param_mins'], dtype=float)
    maxs = np.asarray(engine.param_id_info['param_maxs'], dtype=float)
    theta = 0.5 * (mins + maxs)

    gradient = engine.get_gradient(theta)
    assert gradient.shape == theta.shape
    assert np.all(np.isfinite(gradient))
    # The emulator is linear in theta and the cost is quadratic in the features, so the
    # gradient is non-trivial -- a zero vector would mean the emulator never moved.
    assert np.any(np.abs(gradient) > 0)


def test_simulating_the_best_fit_degrades_instead_of_raising(
        base_user_inputs, resources_dir, tmp_path):
    """Callers pair this with plot_outputs() in one try block.

    Raising here therefore cost the run its observable errors as well as its
    traces -- the whole post-calibration report, for a run that had finished
    perfectly well. There is simply nothing to re-simulate: the emulator's
    features are the result.
    """
    config = _config(base_user_inputs, resources_dir, tmp_path)
    _write_bundle(config)
    pid = CVS0DParamID.init_from_dict(config)
    pid.param_id.best_param_vals = np.asarray(
        pid.param_id.param_id_info['param_mins'], dtype=float)

    assert pid.simulate_with_best_param_vals() is None
    assert pid.simulate_with_best_param_vals(return_series=True) == (None, None)
    assert pid.best_output_calculated is True


def test_a_calibration_on_an_emulator_still_writes_its_observable_errors(
        base_user_inputs, resources_dir, tmp_path):
    """A finished run must have something to show for itself.

    An emulator has no traces, so the reconstruction plots cannot be drawn -- but
    the observable *errors* are a comparison of feature against ground truth, and
    that is exactly what an emulator has. Losing them along with the traces left a
    completed calibration looking like it had produced nothing, because the error
    vectors are what the outputs directory (and anything reading it) shows.
    """
    config = _config(base_user_inputs, resources_dir, tmp_path)
    _write_bundle(config)

    pid = CVS0DParamID.init_from_dict(config)
    engine = pid.param_id
    engine.best_param_vals = np.asarray(engine.param_id_info['param_mins'], dtype=float) * 1.2

    from libcuflynx.param_id.plot_outputs import ParamIDPlotOutputs

    plotter = ParamIDPlotOutputs(pid)
    percent, std = plotter.emulator_error_vectors()

    assert percent.shape == (engine.obs_info['num_obs'],)
    assert np.all(np.isfinite(percent))
    assert np.all(np.isfinite(std))
    # Not all zero: a zero vector is what "we never evaluated anything" looks
    # like, and is indistinguishable from a perfect fit.
    assert np.any(np.abs(percent) > 0)


def test_all_series_degrades_rather_than_raising_on_an_emulator(
        base_user_inputs, resources_dir, tmp_path):
    """get_all_series asks for two things: the features and their traces.

    Refusing outright cost the caller the features too, which is what stopped a
    calibration writing its errors. The traces come back as None; the features
    are the emulator's own.
    """
    config = _config(base_user_inputs, resources_dir, tmp_path)
    bundle, _, _ = _write_bundle(config)
    engine = CVS0DParamID.init_from_dict(config).param_id

    theta = np.asarray(engine.param_id_info['param_mins'], dtype=float) * 1.1
    _, operands_list, _ = engine.get_cost_obs_and_pred_from_params(theta)
    obs_dict, all_series = engine.get_obs_output_dict(operands_list[0], get_all_series=True)

    assert np.asarray(obs_dict['const']) == pytest.approx(bundle.predict(theta))
    assert all(series is None for series in all_series)


def test_gradient_at_a_bound_stays_inside_the_training_box(base_user_inputs, resources_dir,
                                                           tmp_path):
    """An optimiser reaching a bound must not be refused mid-descent.

    A central difference at the upper bound would step outside the box the emulator was
    trained in -- which the emulator (rightly) refuses. The difference becomes one-sided
    there instead, so the gradient stays exact for the function being minimised.
    """
    config = _config(base_user_inputs, resources_dir, tmp_path)
    _write_bundle(config)
    engine = CVS0DParamID.init_from_dict(config).param_id

    at_max = np.asarray(engine.param_id_info['param_maxs'], dtype=float)
    gradient = engine.get_gradient(at_max)
    assert np.all(np.isfinite(gradient))

    at_min = np.asarray(engine.param_id_info['param_mins'], dtype=float)
    assert np.all(np.isfinite(engine.get_gradient(at_min)))


def test_do_ad_is_turned_off_for_an_emulator(base_user_inputs, resources_dir, tmp_path):
    config = _config(base_user_inputs, resources_dir, tmp_path, do_ad=True)
    _write_bundle(config)
    engine = CVS0DParamID.init_from_dict(config).param_id
    # Left on, get_cost_from_operands would push emulator scalars through the CasADi-mode
    # operation funcs, and reset would be suppressed across experiments.
    assert engine.do_ad is False


def test_a_stale_emulator_is_refused_at_setup(base_user_inputs, resources_dir, tmp_path):
    """Refused when the engine is built, not on the thousandth cost evaluation."""
    from libcuflynx.emulators.emulator_bundle import EmulatorQualityError

    config = _config(base_user_inputs, resources_dir, tmp_path)
    _write_bundle(config, fingerprint_override='not-this-model')
    with pytest.raises(EmulatorQualityError, match='stale'):
        CVS0DParamID.init_from_dict(config)


def test_a_poor_emulator_is_refused_at_setup(base_user_inputs, resources_dir, tmp_path):
    from libcuflynx.emulators.emulator_bundle import EmulatorQualityError

    config = _config(base_user_inputs, resources_dir, tmp_path)
    config['emulator_settings']['min_r2'] = 0.95
    _write_bundle(config, feature_r2=[0.99, 0.30])
    with pytest.raises(EmulatorQualityError, match='below the configured min_r2'):
        CVS0DParamID.init_from_dict(config)


def test_series_observables_are_refused_with_a_clear_message(base_user_inputs, resources_dir,
                                                             tmp_path):
    """Series emulation is the deferred follow-up, so it must say so rather than mis-answer."""
    config = _config(base_user_inputs, resources_dir, tmp_path)
    _write_bundle(config)

    engine_config = dict(config)
    pid = CVS0DParamID.init_from_dict(engine_config)
    engine = pid.param_id
    engine.obs_info['data_types'] = ['series'] + list(engine.obs_info['data_types'][1:])
    with pytest.raises(ValueError, match='series'):
        engine._configure_emulator()


def test_sobol_sensitivity_runs_on_the_emulator(base_user_inputs, resources_dir, tmp_path):
    """Sobol is the analysis an emulator most obviously pays for: num_samples*(2M+2)
    evaluations, none of which now touch the solver.

    The sampling manager reduces operands to features in its own loop, separate from the cost
    path's, so it needs its own skip -- and this is what proves both were changed. With the stub
    emulator no model is compiled at all, which is also the point.
    """
    from libcuflynx.sensitivity_analysis.sensitivityAnalysis import SensitivityAnalysis

    config = _config(base_user_inputs, resources_dir, tmp_path)
    bundle, obs_info, param_id_info = _write_bundle(config)
    config['sa_options'] = {
        'method': 'sobol', 'num_samples': 8, 'sample_type': 'saltelli',
        'output_dir': str(tmp_path / 'sa_out'),
    }
    config['model_path'] = 'not/used.cellml'

    sa = SensitivityAnalysis.init_from_dict(config)
    manager = sa.SA_manager
    manager.set_sa_options(config['sa_options'])

    samples = manager.generate_samples()
    num_params = len(param_id_info['param_names'])
    assert len(samples) == 8 * (2 * num_params + 2)

    outputs = manager.generate_outputs_mpi(samples)
    assert manager.sim_helper.emulates_features is True
    assert outputs.shape == (len(samples), obs_info['num_obs'])
    # Every row is the emulator's own prediction for that sample, not a re-reduced trace.
    assert outputs[0] == pytest.approx(bundle.predict(samples[0]))

    # ... and the indices come out of it, which is the analysis the user actually asked for.
    S1, ST, S2 = manager.sobol_index(outputs)
    assert len(S1) == obs_info['num_obs']
