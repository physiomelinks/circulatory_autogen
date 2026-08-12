"""The emulator artefact and its refusal rules (issue #333).

An emulator that is merely inaccurate is not the danger -- it is an emulator that is
inaccurate, out of its training range, or fitted against a different model, and answers anyway.
Every test here is about one of those refusals firing, and firing with a message that names
what is wrong. They use a stub predictor, so the whole seam is covered without autoemulate (and
therefore without torch) being installed.
"""
import json
import os

import numpy as np
import pytest

from emulators.emulator_bundle import (EmulatorBoundsError, EmulatorBundle,
                                       EmulatorQualityError, fingerprint)
from parsers.PrimitiveParsers import ANALYSIS_OPTIONS, YamlFileParser, gradient_sources

pytestmark = pytest.mark.unit


class LinearStub:
    """A trivial emulator: y = x @ w. Enough to exercise every path except the fit."""

    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)

    def predict(self, x):
        return np.asarray(x, dtype=float) @ self.weights


def make_bundle(tmp_path=None, feature_r2=(0.99, 0.98), mins=(0.0, 0.0), maxs=(1.0, 1.0),
                fingerprint_value='abc123', x_span=(1.0, 1.0), y_span=(1.0, 1.0)):
    meta = {
        'param_entry_labels': ['alpha', 'beta'],
        'param_mins': list(mins),
        'param_maxs': list(maxs),
        'param_names': [['comp/alpha'], ['comp/beta']],
        'param_defaults': {'comp/alpha': 0.5, 'comp/beta': 0.5},
        'feature_labels': ['max (max comp/x)', 'mean (mean comp/y)'],
        'feature_r2': list(feature_r2),
        'feature_rmse': [0.01, 0.02],
        'x_scale': {'shift': [0.0, 0.0], 'span': list(x_span)},
        'y_scale': {'shift': [0.0, 0.0], 'span': list(y_span)},
        'fingerprint': {'inputs_sha256': fingerprint_value},
    }
    model = LinearStub([[2.0, 0.0], [0.0, 3.0]])
    return EmulatorBundle(model, meta)


def test_predict_inverts_the_stored_scaling():
    """The transforms live in the bundle, not in the fitting code.

    CA parameters span orders of magnitude and autoemulate works in float32, so the scaling is
    not optional -- and an emulator reloaded without it would predict in the wrong units while
    looking entirely healthy.
    """
    bundle = make_bundle(mins=(0.0, 0.0), maxs=(10.0, 10.0),
                         x_span=(2.0, 4.0), y_span=(10.0, 100.0))
    # scale_x halves/quarters the input, the stub doubles/triples, unscale_y multiplies back
    features = bundle.predict(np.array([2.0, 4.0]))
    assert features == pytest.approx([2.0 / 2.0 * 2.0 * 10.0, 4.0 / 4.0 * 3.0 * 100.0])


class DistributionStub:
    """A probabilistic emulator: predict returns a distribution, not an array.

    ``.mean`` is a property here, exactly as it is on a torch distribution -- as opposed to a
    tensor or ndarray, where ``.mean`` is a method.
    """

    class _Gaussian:
        def __init__(self, values):
            self.mean = values

    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)

    def predict(self, x):
        return self._Gaussian(np.asarray(x, dtype=float) @ self.weights)


def test_a_probabilistic_emulator_is_read_through_its_mean():
    """Every Gaussian process -- autoemulate's default -- returns a distribution.

    Taking ``.mean`` off it by truthiness would grab the *method* on a plain tensor and turn
    every prediction into a bound method, so the distinction has to be by callability.
    """
    bundle = make_bundle()
    bundle.model = DistributionStub([[2.0, 0.0], [0.0, 3.0]])
    assert bundle.predict(np.array([0.5, 0.5])) == pytest.approx([1.0, 1.5])
    # ... and the deterministic (array-returning) emulator still works
    bundle.model = LinearStub([[2.0, 0.0], [0.0, 3.0]])
    assert bundle.predict(np.array([0.5, 0.5])) == pytest.approx([1.0, 1.5])


def test_a_low_r2_emulator_is_refused_by_name():
    bundle = make_bundle(feature_r2=(0.99, 0.42))
    bundle.check_quality(0.4)   # the threshold is the user's, so a lenient one still passes
    with pytest.raises(EmulatorQualityError) as excinfo:
        bundle.check_quality(0.9)
    message = str(excinfo.value)
    # Naming the feature is the point: "the emulator is bad" is not actionable.
    assert 'mean (mean comp/y)' in message
    assert '0.42' in message


def test_nan_r2_counts_as_unusable():
    """A feature that could not be scored is not a feature that scored well."""
    bundle = make_bundle(feature_r2=(0.99, float('nan')))
    with pytest.raises(EmulatorQualityError):
        bundle.check_quality(0.9)


def test_out_of_box_evaluation_is_refused_by_default():
    bundle = make_bundle(mins=(0.0, 0.0), maxs=(1.0, 1.0))
    with pytest.raises(EmulatorBoundsError) as excinfo:
        bundle.predict(np.array([1.5, 0.5]))
    message = str(excinfo.value)
    assert 'alpha' in message and '1.5' in message
    assert 'beta' not in message, 'only the offending parameter should be named'


def test_out_of_box_policies_warn_and_clip():
    bundle = make_bundle()
    warned = bundle.predict(np.array([1.5, 0.5]), out_of_bounds='warn')
    assert warned[0] == pytest.approx(3.0), 'warn must still extrapolate, not clip'
    clipped = bundle.predict(np.array([1.5, 0.5]), out_of_bounds='clip')
    assert clipped[0] == pytest.approx(2.0), 'clip must evaluate at the boundary'


def test_a_stale_emulator_is_refused():
    """Changed bounds, observables, protocol or model => a different theta -> features map.

    Nothing about the emulator itself changes when those do, which is exactly why this has to
    be checked rather than noticed.
    """
    bundle = make_bundle(fingerprint_value='abc123')
    bundle.check_matches({'inputs_sha256': 'abc123'})
    with pytest.raises(EmulatorQualityError, match='stale'):
        bundle.check_matches({'inputs_sha256': 'deadbeef'})


def test_changed_parameters_or_observables_are_refused():
    bundle = make_bundle()
    with pytest.raises(EmulatorQualityError, match='trained for parameters'):
        bundle.check_matches({}, param_entry_labels=['alpha', 'gamma'])
    with pytest.raises(EmulatorQualityError, match='trained for observables'):
        bundle.check_matches({}, feature_labels=['something else'])


def test_fingerprint_changes_with_bounds_observables_and_protocol():
    param_id_info = {'param_names': [['comp/alpha']], 'param_mins': np.array([0.0]),
                     'param_maxs': np.array([1.0])}
    obs_info = {'operands': [['comp/x']], 'operations': ['max'], 'operation_kwargs': [None],
                'data_types': ['constant'], 'experiment_idxs': [0], 'subexperiment_idxs': [0]}
    protocol_info = {'pre_times': [0.0], 'sim_times': [[5]], 'params_to_change': {}}

    base = fingerprint(param_id_info, obs_info, protocol_info)
    widened = dict(param_id_info, param_maxs=np.array([2.0]))
    assert fingerprint(widened, obs_info, protocol_info) != base
    reoperated = dict(obs_info, operations=['min'])
    assert fingerprint(param_id_info, reoperated, protocol_info) != base
    relengthened = dict(protocol_info, sim_times=[[10]])
    assert fingerprint(param_id_info, obs_info, relengthened) != base
    # ... and is stable for an unchanged setup, or every run would look stale
    assert fingerprint(param_id_info, obs_info, protocol_info) == base


def test_bundle_round_trips_through_disk(tmp_path):
    bundle = make_bundle()
    bundle.x_train = np.array([[0.1, 0.2], [0.3, 0.4]])
    bundle.y_train = np.array([[0.2, 0.6], [0.6, 1.2]])
    bundle.save(str(tmp_path))

    assert os.path.isfile(tmp_path / 'emulator_metadata.json')
    reloaded = EmulatorBundle.load(str(tmp_path))
    assert reloaded.feature_labels == bundle.feature_labels
    assert reloaded.predict(np.array([0.5, 0.5])) == pytest.approx(bundle.predict([0.5, 0.5]))
    # The design is kept so the emulator can be refitted or extended without re-simulating --
    # the expensive half of training is the runs, not the fit.
    assert reloaded.x_train == pytest.approx(bundle.x_train)


def test_loading_from_an_empty_directory_says_how_to_train_one(tmp_path):
    with pytest.raises(FileNotFoundError, match='run_emulator_training'):
        EmulatorBundle.load(str(tmp_path))


def test_metadata_missing_a_required_key_is_rejected():
    with pytest.raises(ValueError, match='missing'):
        EmulatorBundle(LinearStub([[1.0]]), {'param_entry_labels': ['a']})


def test_user_inputs_defaults_match_the_schema(base_user_inputs, resources_dir):
    """The parser's fill-ins and the advertised defaults are the same numbers.

    A settings form shows the schema's default; the run uses the parser's. If they disagree,
    the form tells the user something the run does not do.
    """
    config = base_user_inputs.copy()
    config.update({'resources_dir': resources_dir})
    config.pop('emulator_settings', None)
    config.pop('do_emulation', None)
    config.pop('use_emulator', None)
    parsed = YamlFileParser().parse_user_inputs_file(
        config, obs_path_needed=False, do_generation_with_fit_parameters=False)

    assert parsed['do_emulation'] is False
    assert parsed['use_emulator'] is False
    for descriptor in ANALYSIS_OPTIONS['emulation']['options']:
        assert parsed['emulator_settings'][descriptor['name']] == descriptor['default'], (
            f"emulator_settings.{descriptor['name']} default disagrees with the schema")


def test_gradient_menu_over_an_emulator_offers_finite_differences_only():
    """The analytic arms differentiate the real model, which an emulator run is not evaluating.

    Offering AD there would mean the optimiser descends a different function than the cost it
    reports, so the menu a front-end builds must not contain it.
    """
    for model_type in ('cellml_only', 'casadi_python', 'aadc_python', 'python'):
        values = [s['value'] for s in gradient_sources(model_type, use_emulator=True)]
        assert values == ['FD'], f'{model_type} offered {values} over an emulator'
    # ... and the ordinary menu is untouched
    assert 'AD' in [s['value'] for s in gradient_sources('casadi_python')]
