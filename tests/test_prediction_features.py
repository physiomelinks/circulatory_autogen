"""Prediction items with an operation as scalar features (sa_options / emulator_settings
``include_prediction_items``).

Unit tests: the feature selection (a warning for items without an operation, an error for a
non-scalar result), validation of a feature against held-out data, the emulator bundle's
feature lists and fingerprints with and without prediction features, and Sobol / calibration
over a stub emulator. Integration tests (``Simple_ODE_Benchmark``: ``dx/dt = -x + p``,
``dy/dt = -3y + q``): a Sobol and a local SA with prediction features on the real solver, and
emulator training with them.
"""
import copy
import json
import os
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from libcuflynx.emulators.emulator_bundle import (EmulatorBundle, EmulatorQualityError,
                                                  fingerprint)
from libcuflynx.param_id import prediction_features, validation
from libcuflynx.param_id.paramID import CVS0DParamID, emulated_feature_labels
from libcuflynx.parsers.PrimitiveParsers import (ObsAndParamDataParser, YamlFileParser,
                                                 param_entry_labels, scriptFunctionParser)

BENCHMARK_OBS = 'Simple_ODE_Benchmark_obs_data.json'

#: Two features (x_max, y_mean_late), one trace without an operation (x_trace).
PREDICTION_ITEMS = [
    {"data_item_name": "x_max", "operands": ["benchmark/x"], "unit": "dimensionless",
     "operation": "max", "data_type": "constant", "value": 1.0, "std": 0.1},
    {"data_item_name": "x_trace", "operands": ["benchmark/x"], "unit": "dimensionless"},
    {"data_item_name": "y_mean_late", "operands": ["benchmark/y"], "unit": "dimensionless",
     "operation": "steady_state_avg"},
]


def _numpy_funcs():
    return scriptFunctionParser().get_operation_funcs_dict('numpy')


def _parsed_prediction_info(items, sim_times=((2.0,),)):
    doc = {"data_items": [{"data_item_name": "c0", "operands": ["main/c"],
                           "data_type": "constant", "unit": "dimensionless", "value": 1.0,
                           "std": 0.1}],
           "prediction_items": items,
           "protocol_info": {"pre_times": [0.0] * len(sim_times),
                             "sim_times": [list(t) for t in sim_times]}}
    parsed = ObsAndParamDataParser().parse_obs_data_json(obs_data_dict=doc, pre_time=0.0,
                                                          sim_time=2.0)
    return parsed["prediction_info"], parsed["protocol_info"]


# ============================================================================ selection

@pytest.mark.unit
def test_only_items_with_an_operation_are_features_and_the_rest_are_named():
    info, _ = _parsed_prediction_info(copy.deepcopy(PREDICTION_ITEMS))
    with pytest.warns(prediction_features.PredictionFeatureWarning, match="x_trace"):
        indices = prediction_features.prediction_feature_indices(info)
    assert indices == [0, 2]
    assert prediction_features.prediction_feature_names(info, indices) == ["x_max",
                                                                           "y_mean_late"]


@pytest.mark.unit
def test_no_warning_when_every_item_has_an_operation():
    info, _ = _parsed_prediction_info([PREDICTION_ITEMS[0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert prediction_features.prediction_feature_indices(info) == [0]


@pytest.mark.unit
def test_the_option_is_off_by_default():
    assert prediction_features.include_prediction_items({}) is False
    assert prediction_features.include_prediction_items(None) is False
    assert prediction_features.include_prediction_items({"include_prediction_items": True})


@pytest.mark.unit
def test_a_feature_is_reduced_over_the_last_sub_experiment_of_its_experiment():
    items = [dict(PREDICTION_ITEMS[0], experiment_idx=1)]
    info, protocol = _parsed_prediction_info(items, sim_times=((1.0,), (1.0, 2.0, 3.0)))
    assert prediction_features.feature_segments(info, [0], protocol) == [(1, 2)]
    assert prediction_features.labels_with_segments(info, [0], protocol) == ["x_max (Exp1, Sub2)"]


@pytest.mark.unit
def test_a_feature_is_reduced_over_its_subexperiment_idx():
    items = [dict(PREDICTION_ITEMS[0], experiment_idx=1, subexperiment_idx=0),
             dict(PREDICTION_ITEMS[2], experiment_idx=1)]
    info, protocol = _parsed_prediction_info(items, sim_times=((1.0,), (1.0, 2.0, 3.0)))
    assert prediction_features.feature_segments(info, [0, 1], protocol) == [(1, 0), (1, 2)]
    assert prediction_features.feature_experiments(info, [0, 1], protocol) == [1]


@pytest.mark.unit
def test_only_scalar_items_are_features():
    """constant + operation, or no data_type + a reducing operation. A series -- with or
    without an operation -- and an item without an operation are skipped and named."""
    info, _ = _parsed_prediction_info([
        {"data_item_name": "c_op", "operands": ["main/a"], "unit": "-", "operation": "max",
         "data_type": "constant", "value": 1.0},
        {"data_item_name": "reduce", "operands": ["main/a"], "unit": "-", "operation": "mean"},
        {"data_item_name": "sum_trace", "operands": ["main/a", "main/b"], "unit": "-",
         "operation": "addition"},
        {"data_item_name": "series_op", "operands": ["main/a"], "unit": "-", "operation": "max",
         "data_type": "series", "value": [1.0, 2.0], "std": 0.1, "obs_dt": 0.5},
        {"data_item_name": "trace", "operands": ["main/a"], "unit": "-"}])
    with pytest.warns(prediction_features.PredictionFeatureWarning, match="not a scalar") as rec:
        assert prediction_features.prediction_feature_indices(info) == [0, 1]
    message = str(rec[0].message)
    for name in ("sum_trace", "series_op", "trace"):
        assert name in message


@pytest.mark.unit
def test_a_series_prediction_with_an_operation_is_validated_as_a_series():
    """addition of two traces, and a @series_to_constant op's series_output, compared with
    held-out series at k*obs_dt from the start of the segment (here t starts at 5)."""
    info, _ = _parsed_prediction_info([
        {"data_item_name": "sum", "operands": ["main/a", "main/b"], "unit": "-",
         "operation": "addition", "data_type": "series", "value": [1.0, 2.0, 3.0],
         "std": 0.1, "obs_dt": 0.5},
        {"data_item_name": "a_series", "operands": ["main/a"], "unit": "-",
         "operation": "max", "data_type": "series", "value": [0.0, 0.5, 1.0], "std": 0.1,
         "obs_dt": 0.5}])
    t = np.linspace(5.0, 6.0, 11)
    a, b = t - 5.0, 1.0 + (t - 5.0)
    res = validation.validation_results(info, None, [[a, b], [a]],
                                        time_per_item=[t, t])
    by = {item["data_item_name"]: item for item in res["items"]}
    assert by["sum"]["operation"] == "addition"
    assert by["sum"]["t"] == pytest.approx([0.0, 0.5, 1.0])
    assert by["sum"]["model"] == pytest.approx([1.0, 2.0, 3.0])
    assert by["a_series"]["model"] == pytest.approx([0.0, 0.5, 1.0])
    assert by["a_series"]["rmse"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
def test_a_non_scalar_result_is_a_clear_error():
    """subtraction of two traces is a trace, not a feature."""
    info, protocol = _parsed_prediction_info([
        {"data_item_name": "d", "operands": ["main/a", "main/b"], "unit": "-",
         "operation": "subtraction"}])
    outputs = {(0, 0): [[np.arange(5.0), np.ones(5)]]}
    with pytest.raises(prediction_features.NonScalarPredictionFeatureError,
                       match="'d'.*not a scalar"):
        prediction_features.features_from_segments(info, [0], _numpy_funcs(), protocol, outputs)


@pytest.mark.unit
def test_features_read_their_operands_after_the_offset_and_resolve_references():
    """Two operands, an operation_kwargs reference to an earlier prediction, and a failed
    segment giving nan."""
    def mean_diff(x1, x2, scale=1.0):
        return scale * float(np.mean(np.asarray(x1) - np.asarray(x2)))

    funcs = dict(_numpy_funcs(), mean_diff=mean_diff)
    info, protocol = _parsed_prediction_info([
        {"data_item_name": "a_max", "operands": ["main/a"], "unit": "-", "operation": "max"},
        {"data_item_name": "diff", "operands": ["main/a", "main/b"], "unit": "-",
         "operation": "mean_diff", "operation_kwargs": {"scale": "a_max"}},
        {"data_item_name": "other", "operands": ["main/a"], "unit": "-", "operation": "max",
         "experiment_idx": 1},
    ], sim_times=((1.0,), (1.0,)))
    a, b = np.array([1.0, 2.0, 4.0]), np.array([0.0, 1.0, 1.0])
    # entry 0 is a data_item's operands; the features start at offset 1
    outputs = {(0, 0): [["data"], [a], [a, b], [a]], (1, 0): None}
    values = prediction_features.features_from_segments(
        info, [0, 1, 2], funcs, protocol, outputs, offset=1)
    assert values[0] == 4.0
    assert values[1] == pytest.approx(4.0 * np.mean(a - b))
    assert np.isnan(values[2])


@pytest.mark.unit
@pytest.mark.parametrize('referenced, what', [
    ({"data_item_name": "a", "operands": ["main/a"], "unit": "-"}, 'has no operation'),
    ({"data_item_name": "a", "operands": ["main/a"], "unit": "-", "operation": "max",
      "data_type": "series"}, 'is a series'),
])
def test_a_reference_must_name_an_item_with_one_value(referenced, what):
    """Sensitivity analysis and emulator training evaluate only the scalar prediction items,
    so a reference to an item without an operation, or to a series, used to reach the
    operation as the item's name (a TypeError, or a wrong number with no error)."""
    with pytest.raises(ValueError, match=f"references prediction item 'a', which {what}"):
        _parsed_prediction_info([
            referenced,
            {"data_item_name": "b", "operands": ["main/a"], "unit": "-", "operation": "max",
             "data_type": "constant", "operation_kwargs": {"scale": "a"}},
        ])


@pytest.mark.unit
def test_a_reference_to_an_item_that_is_not_a_feature_is_a_clear_error():
    """An item with an operation that does not reduce to one number (no data_type) passes the
    parser but is not a feature; a feature referencing it raises, naming it, rather than
    getting its name as the value."""
    def passthrough(x):
        return np.asarray(x)

    def scaled_mean(x, scale=1.0):
        return scale * float(np.mean(x))

    funcs = dict(_numpy_funcs(), passthrough=passthrough, scaled_mean=scaled_mean)
    info, protocol = _parsed_prediction_info([
        {"data_item_name": "a", "operands": ["main/a"], "unit": "-", "operation": "passthrough"},
        {"data_item_name": "b", "operands": ["main/a"], "unit": "-", "operation": "scaled_mean",
         "data_type": "constant", "operation_kwargs": {"scale": "a"}},
    ])
    indices = prediction_features.prediction_feature_indices(info, funcs, warn=False)
    assert indices == [1]
    outputs = {(0, 0): [[np.array([1.0, 2.0])]]}
    with pytest.raises(ValueError, match="references data_item 'a'"):
        prediction_features.features_from_segments(info, indices, funcs, protocol, outputs)


# ============================================================================ validation

@pytest.mark.unit
def test_validation_compares_the_operation_of_the_operands_with_the_value():
    info, _ = _parsed_prediction_info([
        {"data_item_name": "y_max", "operands": ["main/y"], "unit": "mV", "operation": "max",
         "data_type": "constant", "value": 3.0, "std": 0.5},
        {"data_item_name": "y_end", "operands": ["main/y"], "unit": "mV",
         "data_type": "constant", "value": 1.0, "std": 0.5}])
    t = np.linspace(0.0, 2.0, 21)
    y = np.sin(np.pi * t) + 2.0           # max 3 at t = 0.5, 2 at the end
    res = validation.validation_results(info, {0: t}, [[y], y])
    by_name = {item["data_item_name"]: item for item in res["items"]}
    assert by_name["y_max"]["operation"] == "max"
    assert by_name["y_max"]["model"] == pytest.approx([3.0])
    assert by_name["y_max"]["rmse"] == pytest.approx(0.0, abs=1e-12)
    # no operation: today's rule, the end of the experiment
    assert by_name["y_end"]["operation"] is None
    assert by_name["y_end"]["model"] == pytest.approx([2.0])


@pytest.mark.unit
def test_validation_evaluates_a_two_operand_feature():
    def mean_diff(x1, x2):
        return float(np.mean(np.asarray(x1) - np.asarray(x2)))

    funcs = dict(_numpy_funcs(), mean_diff=mean_diff)
    info, _ = _parsed_prediction_info([
        {"data_item_name": "gap", "operands": ["main/a", "main/b"], "unit": "mV",
         "operation": "mean_diff", "data_type": "constant", "value": 1.5, "std": 0.5}])
    t = np.linspace(0.0, 1.0, 11)
    (item,) = validation.validation_results(
        info, {0: t}, [[3.0 * np.ones_like(t), np.ones_like(t)]],
        operation_funcs_dict=funcs)["items"]
    assert item["model"] == pytest.approx([2.0])
    assert item["rmse"] == pytest.approx(0.5)
    assert item["mean_abs_z"] == pytest.approx(1.0)


# ============================================================================ bundle

def _stub_infos():
    param_id_info = {'param_names': [['benchmark/p'], ['benchmark/q']],
                     'param_mins': [0.0, 0.0], 'param_maxs': [6.0, 6.0]}
    obs_info = {'operands': [['benchmark/x']], 'operations': ['max'],
                'operation_kwargs': [{}], 'data_types': ['constant'],
                'experiment_idxs': [0], 'subexperiment_idxs': [0]}
    protocol_info = {'pre_times': [0.0], 'sim_times': [[8.0]], 'params_to_change': {}}
    return param_id_info, obs_info, protocol_info


@pytest.mark.unit
def test_the_fingerprint_is_unchanged_without_prediction_features():
    param_id_info, obs_info, protocol_info = _stub_infos()
    info, _ = _parsed_prediction_info(copy.deepcopy(PREDICTION_ITEMS))
    legacy = fingerprint(param_id_info, obs_info, protocol_info)
    assert set(legacy) == {'inputs_sha256'}
    assert fingerprint(param_id_info, obs_info, protocol_info, prediction_info=info,
                       prediction_indices=[]) == legacy

    with_pred = fingerprint(param_id_info, obs_info, protocol_info, prediction_info=info,
                            prediction_indices=[0, 2])
    assert with_pred['inputs_sha256'] == legacy['inputs_sha256']
    edited = copy.deepcopy(info)
    edited['operations'][0] = 'min'
    assert fingerprint(param_id_info, obs_info, protocol_info, prediction_info=edited,
                       prediction_indices=[0, 2])['prediction_sha256'] \
        != with_pred['prediction_sha256']


def _meta(feature_labels, prediction_labels=None, r2=None, fp=None):
    meta = {'param_entry_labels': ['p', 'q'], 'param_mins': [0.0, 0.0],
            'param_maxs': [6.0, 6.0], 'feature_labels': list(feature_labels),
            'feature_r2': list(r2 or [0.99] * len(feature_labels)),
            'x_scale': {'shift': [0.0, 0.0], 'span': [6.0, 6.0]},
            'y_scale': {'shift': [0.0] * len(feature_labels),
                        'span': [1.0] * len(feature_labels)},
            'fingerprint': fp or {'inputs_sha256': 'abc'}}
    if prediction_labels:
        meta['prediction_feature_labels'] = list(prediction_labels)
    return meta


@pytest.mark.unit
def test_a_bundle_separates_its_data_and_prediction_features():
    old = EmulatorBundle(None, _meta(['a', 'b']))
    assert old.data_feature_labels == ['a', 'b'] and old.prediction_feature_labels == []

    new = EmulatorBundle(None, _meta(['a', 'b', 'x_max'], ['x_max'],
                                     fp={'inputs_sha256': 'abc', 'prediction_sha256': 'p1'}))
    assert new.data_feature_labels == ['a', 'b']
    # a calibration names only its data_item features, and computes no prediction digest
    new.check_matches({'inputs_sha256': 'abc'}, feature_labels=['a', 'b'])
    with pytest.raises(EmulatorQualityError, match='stale'):
        new.check_matches({'inputs_sha256': 'abc', 'prediction_sha256': 'p2'},
                          prediction_feature_labels=['x_max'])
    with pytest.raises(EmulatorQualityError, match='include_prediction_items: true'):
        old.check_matches({'inputs_sha256': 'abc'}, prediction_feature_labels=['x_max'])


@pytest.mark.unit
def test_a_poor_prediction_feature_does_not_block_a_calibration():
    bundle = EmulatorBundle(None, _meta(['a', 'x_max'], ['x_max'], r2=[0.99, 0.2]))
    bundle.check_quality(0.9)                        # the data_item features only
    with pytest.raises(EmulatorQualityError, match="'x_max'"):
        bundle.check_quality(0.9, labels=['x_max'])


# ============================================================================ trainer

def _benchmark_obs(tmp_path, items=PREDICTION_ITEMS, resources_dir=None):
    with open(os.path.join(resources_dir, BENCHMARK_OBS)) as f:
        doc = json.load(f)
    doc['prediction_items'] = copy.deepcopy(items)
    path = tmp_path / 'benchmark_with_predictions_obs_data.json'
    path.write_text(json.dumps(doc))
    return str(path), doc


def _parsed_benchmark(obs_path, scratch):
    parser = ObsAndParamDataParser()
    parsed = parser.parse_obs_data_json(param_id_obs_path=obs_path, pre_time=0.0, sim_time=8.0)
    os.makedirs(scratch, exist_ok=True)
    obs_info = parser.process_obs_info(gt_df=parsed['gt_df'], output_dir=scratch, dt=0.05)
    protocol_info = parser.process_protocol_and_weights(
        gt_df=parsed['gt_df'], protocol_info=parsed['protocol_info'], dt=0.05)
    return obs_info, protocol_info, parsed['prediction_info']


def _stub_trainer(resources_dir, tmp_path, include):
    from libcuflynx.emulators.emulator_trainer import EmulatorTrainer

    obs_path, _ = _benchmark_obs(tmp_path, resources_dir=resources_dir)
    obs_info, protocol_info, prediction_info = _parsed_benchmark(obs_path,
                                                                 str(tmp_path / 'scratch'))
    param_id_info = ObsAndParamDataParser().get_param_id_info(
        os.path.join(resources_dir, 'Simple_ODE_Benchmark_params_for_id.csv'))
    pid = SimpleNamespace(sim_helper=SimpleNamespace(emulates_features=False),
                          obs_info=obs_info, protocol_info=protocol_info,
                          prediction_info=prediction_info, param_id_info=param_id_info,
                          model_path=None)
    settings = {'include_prediction_items': True} if include else {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', prediction_features.PredictionFeatureWarning)
        return EmulatorTrainer(pid, settings, comm=None), pid


@pytest.mark.unit
def test_trainer_features_and_fingerprint_with_and_without_the_option(resources_dir,
                                                                      tmp_path):
    off, pid = _stub_trainer(resources_dir, tmp_path, include=False)
    data_labels = emulated_feature_labels(pid.obs_info)
    assert off.feature_labels == data_labels
    assert off._fingerprint() == fingerprint(pid.param_id_info, pid.obs_info,
                                             pid.protocol_info, pid.model_path)

    on, _ = _stub_trainer(resources_dir, tmp_path, include=True)
    assert on.feature_labels == data_labels + ['x_max', 'y_mean_late']
    assert on.prediction_feature_labels == ['x_max', 'y_mean_late']
    fp = on._fingerprint()
    assert fp['inputs_sha256'] == off._fingerprint()['inputs_sha256']
    assert 'prediction_sha256' in fp


@pytest.mark.unit
def test_the_trainer_warns_about_items_without_an_operation(resources_dir, tmp_path):
    from libcuflynx.emulators.emulator_trainer import EmulatorTrainer

    _, pid = _stub_trainer(resources_dir, tmp_path, include=False)
    with pytest.warns(prediction_features.PredictionFeatureWarning, match='x_trace'):
        EmulatorTrainer(pid, {'include_prediction_items': True}, comm=None)


@pytest.mark.unit
def test_prediction_items_are_not_emulated_by_default(base_user_inputs, resources_dir, tmp_path):
    """An emulator is trained on prediction items only when asked: the schema, the parser's
    fill-in and the shipped user_inputs.yaml all leave the option off, and a trainer given the
    parsed default settings has no prediction features, although the benchmark's prediction
    items have operations that would make them features."""
    import yaml
    from libcuflynx.emulators.emulator_trainer import EmulatorTrainer
    from libcuflynx.parsers.PrimitiveParsers import ANALYSIS_OPTIONS

    for mode in ('emulation', 'sensitivity_analysis'):
        option = next(d for d in ANALYSIS_OPTIONS[mode]['options']
                      if d['name'] == 'include_prediction_items')
        assert option['default'] is False, mode

    shipped = os.path.join(os.path.dirname(__file__), '..', 'user_run_files', 'user_inputs.yaml')
    with open(shipped) as f:
        user_inputs = yaml.safe_load(f)
    assert user_inputs['emulator_settings'].get('include_prediction_items', False) is False
    assert user_inputs['sa_options'].get('include_prediction_items', False) is False

    config = base_user_inputs.copy()
    config.update({'resources_dir': resources_dir})
    config.pop('emulator_settings', None)
    parsed = YamlFileParser().parse_user_inputs_file(
        config, obs_path_needed=False, do_generation_with_fit_parameters=False)
    assert parsed['emulator_settings']['include_prediction_items'] is False

    _, pid = _stub_trainer(resources_dir, tmp_path, include=False)
    assert prediction_features.prediction_feature_indices(pid.prediction_info, warn=False)
    with warnings.catch_warnings():
        warnings.simplefilter('error', prediction_features.PredictionFeatureWarning)
        trainer = EmulatorTrainer(pid, parsed['emulator_settings'], comm=None)
    assert trainer.prediction_indices == []
    assert trainer.prediction_feature_labels == []
    assert trainer.feature_labels == emulated_feature_labels(pid.obs_info)
    assert 'prediction_sha256' not in trainer._fingerprint()


@pytest.mark.unit
def test_an_off_option_is_not_recorded_in_the_bundle_settings():
    from libcuflynx.emulators.emulator_trainer import _jsonable_settings
    assert 'include_prediction_items' not in _jsonable_settings(
        {'min_r2': 0.9, 'include_prediction_items': False})
    assert _jsonable_settings({'include_prediction_items': True})['include_prediction_items']


# ============================================================================ stub emulator

class LinearStub:
    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)

    def predict(self, x):
        return np.asarray(x, dtype=float) @ self.weights


def _emulator_config(base_user_inputs, resources_dir, tmp_path, obs_path):
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': 'Simple_ODE_Benchmark',
        'input_param_file': 'Simple_ODE_Benchmark_parameters.csv',
        'model_type': 'cellml',
        'resources_dir': resources_dir,
        'param_id_obs_path': obs_path,
        'params_for_id_path': os.path.join(resources_dir,
                                           'Simple_ODE_Benchmark_params_for_id.csv'),
        'param_id_output_dir': str(tmp_path / 'param_id_output'),
        'param_id_method': 'genetic_algorithm',
        'DEBUG': True,
        'use_emulator': True,
        'emulator_settings': {'emulator_dir': str(tmp_path / 'emulator'), 'min_r2': 0.5},
        'model_path': 'not/used.cellml',
    })
    return YamlFileParser().parse_user_inputs_file(
        config, obs_path_needed=True, do_generation_with_fit_parameters=False)


def _write_stub_bundle(config, with_predictions):
    obs_info, protocol_info, prediction_info = _parsed_benchmark(
        config['param_id_obs_path'], os.path.join(config['emulator_settings']['emulator_dir'],
                                                  '_parsed'))
    param_id_info = ObsAndParamDataParser().get_param_id_info(config['params_for_id_path'])
    data_labels = emulated_feature_labels(obs_info)
    indices = [0, 2] if with_predictions else []
    pred_labels = prediction_features.prediction_feature_names(prediction_info, indices)
    labels = data_labels + pred_labels
    weights = np.arange(1, 2 * len(labels) + 1, dtype=float).reshape(2, len(labels))
    meta = {
        'param_entry_labels': param_entry_labels(param_id_info),
        'param_mins': [float(v) for v in param_id_info['param_mins']],
        'param_maxs': [float(v) for v in param_id_info['param_maxs']],
        'param_names': [[str(n) for n in entry] for entry in param_id_info['param_names']],
        'param_defaults': {str(name): 1.0
                           for entry in param_id_info['param_names'] for name in entry},
        'feature_labels': labels,
        'feature_r2': [0.99] * len(labels),
        'x_scale': {'shift': [float(v) for v in param_id_info['param_mins']],
                    'span': [float(hi - lo) for lo, hi in zip(param_id_info['param_mins'],
                                                              param_id_info['param_maxs'])]},
        'y_scale': {'shift': [0.0] * len(labels), 'span': [1.0] * len(labels)},
        'fingerprint': fingerprint(param_id_info, obs_info, protocol_info,
                                   prediction_info=prediction_info, prediction_indices=indices),
    }
    if pred_labels:
        meta['prediction_feature_labels'] = pred_labels
    bundle = EmulatorBundle(LinearStub(weights), meta)
    bundle.save(config['emulator_settings']['emulator_dir'])
    return bundle, obs_info


def _sobol_manager(config, tmp_path, include):
    from libcuflynx.sensitivity_analysis.sensitivityAnalysis import SensitivityAnalysis

    config['sa_options'] = {'method': 'sobol', 'num_samples': 4, 'sample_type': 'saltelli',
                            'output_dir': str(tmp_path / 'sa_out'),
                            'include_prediction_items': include}
    sa = SensitivityAnalysis.init_from_dict(config)
    sa.SA_manager.set_sa_options(config['sa_options'])
    return sa.SA_manager


@pytest.mark.unit
def test_sobol_over_an_emulator_uses_its_prediction_features(base_user_inputs, resources_dir,
                                                             tmp_path):
    obs_path, _ = _benchmark_obs(tmp_path, resources_dir=resources_dir)
    config = _emulator_config(base_user_inputs, resources_dir, tmp_path, obs_path)
    bundle, obs_info = _write_stub_bundle(config, with_predictions=True)
    manager = _sobol_manager(config, tmp_path, include=True)

    samples = manager.generate_samples()
    with pytest.warns(prediction_features.PredictionFeatureWarning, match='x_trace'):
        outputs = manager.generate_outputs_mpi(samples)
    assert outputs.shape == (len(samples), obs_info['num_obs'] + 2)
    assert outputs[0] == pytest.approx(bundle.predict(samples[0]))

    S1, ST, S2 = manager.sobol_index(outputs)
    manager.save_sobol_indices(S1, ST, S2)
    df = pd.read_csv(os.path.join(manager.output_dir, 'all_outputs_n4_Sobol_indices.csv'))
    assert 'ST_x_max (Exp0, Sub0)' in df.columns and 'ST_y_mean_late (Exp0, Sub0)' in df.columns
    with open(os.path.join(manager.output_dir, 'sobol_output_features.json')) as f:
        outputs_meta = json.load(f)['outputs']
    assert [o['kind'] for o in outputs_meta] == ['data_item', 'data_item', 'prediction_item',
                                                 'prediction_item']
    assert outputs_meta[2] == {'output': 'x_max (Exp0, Sub0)', 'kind': 'prediction_item',
                               'data_item_name': 'x_max', 'experiment_idx': 0,
                               'subexperiment_idx': 0}


@pytest.mark.unit
def test_sobol_without_the_option_is_unchanged(base_user_inputs, resources_dir, tmp_path):
    obs_path, _ = _benchmark_obs(tmp_path, resources_dir=resources_dir)
    config = _emulator_config(base_user_inputs, resources_dir, tmp_path, obs_path)
    _write_stub_bundle(config, with_predictions=True)
    manager = _sobol_manager(config, tmp_path, include=False)
    with warnings.catch_warnings():
        warnings.simplefilter('error', prediction_features.PredictionFeatureWarning)
        outputs = manager.generate_outputs_mpi(manager.generate_samples())
    assert outputs.shape[1] == manager.obs_info['num_obs']
    S1, ST, S2 = manager.sobol_index(outputs)
    manager.save_sobol_indices(S1, ST, S2)
    assert not os.path.exists(os.path.join(manager.output_dir, 'sobol_output_features.json'))


@pytest.mark.unit
def test_sa_with_an_emulator_trained_without_the_option_says_to_retrain(
        base_user_inputs, resources_dir, tmp_path):
    obs_path, _ = _benchmark_obs(tmp_path, resources_dir=resources_dir)
    config = _emulator_config(base_user_inputs, resources_dir, tmp_path, obs_path)
    _write_stub_bundle(config, with_predictions=False)
    manager = _sobol_manager(config, tmp_path, include=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', prediction_features.PredictionFeatureWarning)
        with pytest.raises(EmulatorQualityError,
                           match='emulator_settings.include_prediction_items: true'):
            manager.generate_outputs_mpi(manager.generate_samples())


@pytest.mark.unit
def test_calibration_over_an_emulator_with_prediction_features(base_user_inputs,
                                                               resources_dir, tmp_path):
    """The prediction features follow the data_item ones; the cost reads only the latter."""
    obs_path, _ = _benchmark_obs(tmp_path, resources_dir=resources_dir)
    config = _emulator_config(base_user_inputs, resources_dir, tmp_path, obs_path)
    bundle, obs_info = _write_stub_bundle(config, with_predictions=True)

    engine = CVS0DParamID.init_from_dict(config).param_id
    theta = np.asarray(engine.param_id_info['param_mins'], dtype=float) + 1.5
    cost, operands_list, _ = engine.get_cost_obs_and_pred_from_params(theta)
    assert np.isfinite(cost)
    predicted = bundle.predict(theta)
    const = np.asarray(engine.get_obs_output_dict(operands_list[0])['const'])
    assert const == pytest.approx(predicted[:obs_info['num_obs']])
    assert engine.sim_helper.get_predicted_prediction_features(['y_mean_late']) \
        == pytest.approx(predicted[-1:])


# ============================================================================ end to end

def _benchmark_config(base_user_inputs, resources_dir, temp_output_dir,
                      temp_generated_models_dir, obs_path, **overrides):
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': 'Simple_ODE_Benchmark',
        'input_param_file': 'Simple_ODE_Benchmark_parameters.csv',
        'model_type': 'cellml',
        'solver': 'CVODE_myokit',
        'param_id_method': 'genetic_algorithm',
        'pre_time': 0.0,
        'sim_time': 8.0,
        'dt': 0.05,
        'DEBUG': True,
        'do_uq': False,
        'do_ia': False,
        'plot_predictions': False,
        'solver_info': {'MaximumStep': 0.01, 'MaximumNumberOfSteps': 5000},
        'param_id_obs_path': obs_path,
        'params_for_id_path': os.path.join(resources_dir,
                                           'Simple_ODE_Benchmark_params_for_id.csv'),
        'param_id_output_dir': temp_output_dir,
        'resources_dir': resources_dir,
        'generated_models_dir': temp_generated_models_dir,
    })
    config.update(overrides)
    return YamlFileParser().parse_user_inputs_file(
        config, obs_path_needed=True, do_generation_with_fit_parameters=False)


@pytest.fixture
def mpi_comm():
    from libcuflynx.utilities.mpi_utils import get_MPI
    return get_MPI().COMM_WORLD


def _generate(config, comm):
    from libcuflynx.scripts.script_generate_with_new_architecture import \
        generate_with_new_architecture
    if comm.Get_rank() == 0:
        assert generate_with_new_architecture(False, config), 'benchmark generation failed'
    comm.Barrier()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_sobol_sa_reports_prediction_features_end_to_end(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm,
        tmp_path):
    """x_max depends on p only (x -> p), y_mean_late on q only (y -> q/3)."""
    from libcuflynx.sensitivity_analysis.sensitivityAnalysis import SensitivityAnalysis

    obs_path, obs_doc = _benchmark_obs(tmp_path, resources_dir=resources_dir)
    out_dir = os.path.join(temp_output_dir, 'sa_prediction_features')
    config = _benchmark_config(base_user_inputs, resources_dir, temp_output_dir,
                               temp_generated_models_dir, obs_path,
                               sa_options={'method': 'sobol', 'num_samples': 8,
                                           'sample_type': 'saltelli', 'output_dir': out_dir,
                                           'include_prediction_items': True})
    _generate(config, mpi_comm)
    sa = SensitivityAnalysis.init_from_dict(config)
    with pytest.warns(prediction_features.PredictionFeatureWarning, match='x_trace'):
        sa.run_sensitivity_analysis(config['sa_options'])
    if mpi_comm.Get_rank() != 0:
        return
    df = pd.read_csv(os.path.join(out_dir, 'all_outputs_n8_Sobol_indices.csv')).set_index(
        'Parameter')
    assert df.loc['benchmark/p', 'ST_x_max (Exp0, Sub0)'] > 0.9
    assert df.loc['benchmark/q', 'ST_x_max (Exp0, Sub0)'] < 0.1
    assert df.loc['benchmark/q', 'ST_y_mean_late (Exp0, Sub0)'] > 0.9
    assert os.path.exists(os.path.join(out_dir, 'sobol_output_features.json'))
    assert any('x_max' in name for name in os.listdir(out_dir) if name.endswith('.png'))


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_local_sa_reports_prediction_features_end_to_end(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm,
        tmp_path):
    """d(x_max)/dp -> 1 and d(y_mean_late)/dq -> 1/3 on the benchmark (closed form)."""
    from libcuflynx.sensitivity_analysis.sensitivityAnalysis import SensitivityAnalysis

    obs_path, obs_doc = _benchmark_obs(tmp_path, resources_dir=resources_dir)
    out_dir = os.path.join(temp_output_dir, 'local_sa_prediction_features')
    sa_options = {'method': 'local', 'gradient_method': 'FD', 'num_samples': 4,
                  'sample_type': 'saltelli', 'output_dir': out_dir,
                  'include_prediction_items': True}
    config = _benchmark_config(base_user_inputs, resources_dir, temp_output_dir,
                               temp_generated_models_dir, obs_path, sa_options=sa_options)
    _generate(config, mpi_comm)
    sa = SensitivityAnalysis.init_from_dict(config)
    sa.set_ground_truth_data(obs_doc)
    sa.set_params_for_id([
        {'vessel_name': 'benchmark', 'param_name': 'p', 'param_type': 'const', 'min': 0, 'max': 6},
        {'vessel_name': 'benchmark', 'param_name': 'q', 'param_type': 'const', 'min': 0, 'max': 6},
    ])
    with pytest.warns(prediction_features.PredictionFeatureWarning, match='x_trace'):
        sa.run_sensitivity_analysis(sa_options)
    local = sa.get_local_sensitivities()
    assert local['prediction_feature_names'] == ['x_max', 'y_mean_late']
    assert local['output_names'][-2:] == ['x_max', 'y_mean_late']
    raw = local['raw']
    p_label, q_label = local['param_names']
    assert raw['x_max'][p_label] == pytest.approx(1.0, abs=0.02)
    assert raw['y_mean_late'][q_label] == pytest.approx(1.0 / 3.0, abs=0.01)
    assert abs(raw['x_max'][q_label]) < 1e-6
    if mpi_comm.Get_rank() == 0:
        df = pd.read_csv(os.path.join(out_dir, 'local_sensitivity_absolute.csv'),
                         index_col='output')
        assert 'x_max' in df.index and 'y_mean_late' in df.index


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_training_with_prediction_features_end_to_end(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm,
        tmp_path, monkeypatch):
    """Training targets are the data_item features then the prediction features of one run;
    the saved bundle records them. The fit is stubbed (no autoemulate needed)."""
    from libcuflynx.emulators.emulator_trainer import EmulatorTrainer, resolve_emulator_dir

    obs_path, _ = _benchmark_obs(tmp_path, resources_dir=resources_dir)
    config = _benchmark_config(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, obs_path,
        do_emulation=True,
        emulator_settings={'num_train_samples': 6, 'sample_type': 'sobol', 'random_seed': 0,
                           'min_r2': 0.5, 'include_prediction_items': True})
    _generate(config, mpi_comm)
    with pytest.warns(prediction_features.PredictionFeatureWarning, match='x_trace'):
        trainer = EmulatorTrainer.init_from_dict(config, comm=mpi_comm)
    x, y = trainer.evaluate(trainer.design())
    if mpi_comm.Get_rank() == 0:
        assert y.shape[1] == len(trainer.feature_labels) == 4
        # x -> p over 8 s from x0: its max is close to p; y_mean_late close to q/3
        assert y[:, 2] == pytest.approx(x[:, 0], abs=0.05 * 6)
        assert y[:, 3] == pytest.approx(x[:, 1] / 3.0, abs=0.05)

    def fake_fit(x, y, space_filling=None):
        stats = {k: [0.99] * 4 for k in ('r2', 'rmse', 'mae', 'bias', 'max_abs_error', 'nrmse')}
        stats.update(theta=x[:1], y_true=y[:1], y_pred=y[:1])
        return (LinearStub(np.zeros((2, 4))), stats, 'Stub', EmulatorBundle.make_scale(x),
                EmulatorBundle.make_scale(y))

    monkeypatch.setattr(trainer, 'fit', fake_fit)
    monkeypatch.setattr('libcuflynx.emulators.emulator_trainer.require_autoemulate',
                        lambda: None)
    trainer.train()
    if mpi_comm.Get_rank() == 0:
        saved = EmulatorBundle.load(resolve_emulator_dir(config))
        assert saved.prediction_feature_labels == ['x_max', 'y_mean_late']
        assert saved.feature_labels[-2:] == ['x_max', 'y_mean_late']
        assert 'prediction_sha256' in saved.meta['fingerprint']


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_save_prediction_data_validates_a_feature_end_to_end(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm,
        tmp_path):
    """At p = 2 the benchmark's x rises from its initial value towards 2, so x_max is ~2 --
    compared with the held-out value 1.0 -- while the trace item keeps today's rule."""
    obs_path, _ = _benchmark_obs(tmp_path, resources_dir=resources_dir, items=[
        dict(PREDICTION_ITEMS[0]),
        {"data_item_name": "x_end", "operands": ["benchmark/x"], "unit": "dimensionless",
         "data_type": "constant", "value": 2.0, "std": 0.1}])
    config = _benchmark_config(base_user_inputs, resources_dir, temp_output_dir,
                               temp_generated_models_dir, obs_path)
    _generate(config, mpi_comm)
    if mpi_comm.Get_rank() != 0:
        return
    pid = CVS0DParamID.init_from_dict(config)
    pid.set_best_param_vals(np.array([2.0, 1.0]))
    pid.save_prediction_data()
    with open(os.path.join(pid.output_dir, validation.VALIDATION_RESULTS_FILE)) as f:
        items = {item['data_item_name']: item for item in json.load(f)['items']}
    assert items['x_max']['operation'] == 'max'
    assert items['x_max']['model'][0] == pytest.approx(2.0, abs=0.05)
    assert items['x_max']['rmse'] == pytest.approx(abs(items['x_max']['model'][0] - 1.0))
    assert items['x_end']['operation'] is None
    assert items['x_end']['model'][0] == pytest.approx(2.0, abs=0.05)


# ============================================================================ validation-only
# experiments. Benchmark: dx/dt = -x + p, dy/dt = -3y + q, x(0) = x_init. Experiment 0 is fitted
# (x, y steady states). Experiment 1 has no data_items: a different sim_time, and
# params_to_change starting x at 2, so there x(t) = p + (2 - p) exp(-t).

def _x_exp1(p, t):
    return p + (2.0 - p) * np.exp(-np.asarray(t, dtype=float))


def _validation_obs(tmp_path, resources_dir, exp1_sim_times=(3.0,), prediction_items=None,
                    exp1_data=False):
    with open(os.path.join(resources_dir, BENCHMARK_OBS)) as f:
        doc = json.load(f)
    doc['protocol_info'] = {
        'pre_times': [0.0, 0.0], 'sim_times': [[8.0], list(exp1_sim_times)],
        'params_to_change': {'benchmark/x_init': [[0.0], [2.0] * len(exp1_sim_times)]}}
    t_obs = np.arange(7) * 0.5
    doc['prediction_items'] = prediction_items if prediction_items is not None else [
        {"data_item_name": "x_val", "operands": ["benchmark/x"], "unit": "dimensionless",
         "experiment_idx": 1, "data_type": "series", "value": _x_exp1(1.0, t_obs).tolist(),
         "std": 0.05, "obs_dt": 0.5},
        {"data_item_name": "x_mean_val", "operands": ["benchmark/x"], "unit": "dimensionless",
         "experiment_idx": 1, "operation": "mean", "data_type": "constant", "value": 1.3,
         "std": 0.1}]
    if exp1_data:
        doc['data_items'].append(dict(doc['data_items'][0], data_item_name='x1',
                                      experiment_idx=1, value=1.0))
    path = tmp_path / 'validation_only_obs_data.json'
    path.write_text(json.dumps(doc))
    return str(path), doc


class _RunSpy:
    """Records the experiments every protocol run simulates."""

    def __init__(self, monkeypatch):
        from libcuflynx.protocol_runners.protocol_executor import ProtocolExecutor
        self.calls = []
        original = ProtocolExecutor.run_protocol
        spy = self

        def run_protocol(executor, protocol_info, *args, **kwargs):
            idxs = kwargs.get('exp_indices')
            spy.calls.append(sorted(idxs) if idxs is not None
                             else list(range(len(protocol_info['sim_times']))))
            return original(executor, protocol_info, *args, **kwargs)

        monkeypatch.setattr(ProtocolExecutor, 'run_protocol', run_protocol)

    def experiments(self):
        seen = set()
        for call in self.calls:
            seen |= set(call)
        return seen


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_a_validation_only_experiment_is_not_calibrated_but_is_validated(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm,
        tmp_path, monkeypatch, capsys):
    """(a) Calibration, the best-fit check and the plots never simulate experiment 1; the
    validation does, with its own sim_time and params_to_change, and matches the closed form."""
    obs_path, _ = _validation_obs(tmp_path, resources_dir)
    config = _benchmark_config(base_user_inputs, resources_dir, temp_output_dir,
                               temp_generated_models_dir, obs_path,
                               debug_optimiser_options={'num_calls_to_function': 30,
                                                        'max_patience': 30})
    _generate(config, mpi_comm)
    pid = CVS0DParamID.init_from_dict(config)
    assert 'experiment(s) [1] have no data_items' in capsys.readouterr().out
    spy = _RunSpy(monkeypatch)
    pid.run()
    if mpi_comm.Get_rank() != 0:
        return
    pid.simulate_with_best_param_vals()
    pid.plot_outputs()
    assert spy.calls and spy.experiments() == {0}, spy.calls

    spy.calls.clear()
    pid.save_prediction_data()
    assert 1 in spy.experiments()
    p = float(pid.param_id.best_param_vals[0])
    with open(os.path.join(pid.output_dir, validation.VALIDATION_RESULTS_FILE)) as f:
        items = {item['data_item_name']: item for item in json.load(f)['items']}
    x_val = items['x_val']
    assert x_val['t'] == pytest.approx(np.arange(7) * 0.5)
    assert x_val['model'] == pytest.approx(_x_exp1(p, x_val['t']), abs=2e-3)
    t_run = np.arange(0.0, 3.0 + 1e-9, 0.05)
    assert items['x_mean_val']['model'][0] == pytest.approx(np.mean(_x_exp1(p, t_run)),
                                                            abs=2e-3)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_sa_computes_a_feature_from_a_validation_only_experiment(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm,
        tmp_path, monkeypatch):
    """(b) mean x over experiment 1 depends on p only. Without the option experiment 1 is not
    simulated at all."""
    from libcuflynx.sensitivity_analysis.sensitivityAnalysis import SensitivityAnalysis

    obs_path, _ = _validation_obs(tmp_path, resources_dir)
    out_dir = os.path.join(temp_output_dir, 'sa_validation_only')
    sa_options = {'method': 'sobol', 'num_samples': 8, 'sample_type': 'saltelli',
                  'output_dir': out_dir, 'include_prediction_items': True}
    config = _benchmark_config(base_user_inputs, resources_dir, temp_output_dir,
                               temp_generated_models_dir, obs_path, sa_options=sa_options)
    _generate(config, mpi_comm)
    spy = _RunSpy(monkeypatch)
    sa = SensitivityAnalysis.init_from_dict(config)
    with pytest.warns(prediction_features.PredictionFeatureWarning, match='x_val'):
        sa.run_sensitivity_analysis(sa_options)
    assert spy.experiments() == {0, 1}
    if mpi_comm.Get_rank() == 0:
        df = pd.read_csv(os.path.join(out_dir, 'all_outputs_n8_Sobol_indices.csv')).set_index(
            'Parameter')
        assert df.loc['benchmark/p', 'ST_x_mean_val (Exp1, Sub0)'] > 0.9
        assert df.loc['benchmark/q', 'ST_x_mean_val (Exp1, Sub0)'] < 0.1

    spy.calls.clear()
    sa_off = dict(sa_options, include_prediction_items=False,
                  output_dir=os.path.join(temp_output_dir, 'sa_validation_only_off'))
    sa.run_sensitivity_analysis(sa_off)
    assert spy.experiments() == {0}


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_subexperiment_idx_picks_the_segment_for_validation_and_features(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm,
        tmp_path):
    """(c) Experiment 1 runs 2 s then 3 s. Sub-experiment 0 starts at x = 2; sub-experiment 1
    starts at x(2). Series times run from the start of their sub-experiment."""
    t_obs = np.arange(5) * 0.5
    items = [
        {"data_item_name": "x_sub0", "operands": ["benchmark/x"], "unit": "dimensionless",
         "experiment_idx": 1, "subexperiment_idx": 0, "data_type": "series",
         "value": _x_exp1(1.0, t_obs).tolist(), "std": 0.05, "obs_dt": 0.5},
        {"data_item_name": "x_sub1", "operands": ["benchmark/x"], "unit": "dimensionless",
         "experiment_idx": 1, "data_type": "series",
         "value": _x_exp1(1.0, 2.0 + t_obs).tolist(), "std": 0.05, "obs_dt": 0.5},
        {"data_item_name": "x_max_sub0", "operands": ["benchmark/x"], "unit": "dimensionless",
         "experiment_idx": 1, "subexperiment_idx": 0, "operation": "max",
         "data_type": "constant", "value": 2.0, "std": 0.1},
        {"data_item_name": "x_max_sub1", "operands": ["benchmark/x"], "unit": "dimensionless",
         "experiment_idx": 1, "operation": "max", "data_type": "constant",
         "value": 1.1, "std": 0.1}]
    obs_path, _ = _validation_obs(tmp_path, resources_dir, exp1_sim_times=(2.0, 3.0),
                                  prediction_items=items)
    config = _benchmark_config(base_user_inputs, resources_dir, temp_output_dir,
                               temp_generated_models_dir, obs_path)
    _generate(config, mpi_comm)
    if mpi_comm.Get_rank() != 0:
        return
    p, q = 0.5, 1.0                           # x < 2 decays, so max is the segment's start
    pid = CVS0DParamID.init_from_dict(config)
    pid.set_best_param_vals(np.array([p, q]))
    pid.save_prediction_data()
    with open(os.path.join(pid.output_dir, validation.VALIDATION_RESULTS_FILE)) as f:
        got = {item['data_item_name']: item for item in json.load(f)['items']}
    assert got['x_sub0']['model'] == pytest.approx(_x_exp1(p, t_obs), abs=2e-3)
    assert got['x_sub1']['model'] == pytest.approx(_x_exp1(p, 2.0 + t_obs), abs=2e-3)
    assert got['x_max_sub0']['model'][0] == pytest.approx(2.0, abs=2e-3)
    assert got['x_max_sub1']['model'][0] == pytest.approx(_x_exp1(p, 2.0), abs=2e-3)
    # the default (last) sub-experiment keeps its file; the other gets its own
    assert os.path.exists(os.path.join(pid.output_dir, 'prediction_variable_data_exp_0.npy'))
    assert os.path.exists(os.path.join(pid.output_dir,
                                       'prediction_variable_data_exp_1_sub_0.npy'))

    engine = pid.param_id
    info = engine.prediction_info
    indices = prediction_features.prediction_feature_indices(info, warn=False)
    assert indices == [2, 3]
    _, features = prediction_features.simulate_features(engine, [p, q], info, indices)
    assert features == pytest.approx([2.0, _x_exp1(p, 2.0)], abs=2e-3)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_every_experiment_with_data_is_simulated_as_before(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm,
        tmp_path, monkeypatch, capsys):
    """(d) With a data_item in experiment 1 too, the cost simulates both, and says nothing."""
    obs_path, _ = _validation_obs(tmp_path, resources_dir, exp1_data=True)
    config = _benchmark_config(base_user_inputs, resources_dir, temp_output_dir,
                               temp_generated_models_dir, obs_path)
    _generate(config, mpi_comm)
    engine = CVS0DParamID.init_from_dict(config).param_id
    assert 'have no data_items' not in capsys.readouterr().out
    spy = _RunSpy(monkeypatch)
    cost = engine.get_cost_from_params(np.array([1.0, 1.0]))
    assert np.isfinite(cost)
    assert spy.calls == [[0, 1]]
