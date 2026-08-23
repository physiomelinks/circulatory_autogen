"""The posterior predictive check, and the two coverage numbers it reports.

Coverage is the kind of quantity that looks plausible while being wrong, so the
tests here are calibration tests: a correctly-calibrated posterior must come out
at the nominal level, and a biased or over-confident one must come out well below
it. A coverage function that always returned 0.9 would pass a spot check and fail
both of these.
"""
import json
import os

import numpy as np
import pytest

from libcuflynx.param_id import posterior_predictive as pp


# ── the normal quantile ────────────────────────────────────────────────────
@pytest.mark.unit
@pytest.mark.parametrize("level,expected", [(0.8, 1.2816), (0.95, 1.9600),
                                            (0.5, 0.6745), (0.99, 2.5758)])
def test_z_matches_the_normal_quantile(level, expected):
    """Computed by bisection rather than pulling scipy in for one number."""
    assert pp._z_for(level) == pytest.approx(expected, abs=1e-3)


# ── coverage ───────────────────────────────────────────────────────────────
def calibrated_case(n_obs=4000, n_samples=4000, seed=0):
    """Predictions and an observation that are both draws from the same law.

    The observation must be a *draw*, not the centre of the predictions -- put it
    at the centre and every interval contains it, which is a coverage of 1.0 and
    tells you nothing.
    """
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=n_obs) * 10
    std = np.abs(rng.normal(size=n_obs)) + 0.5
    truth = mu + rng.normal(size=n_obs) * std
    preds = mu[None, :] + rng.normal(size=(n_samples, n_obs)) * std[None, :]
    return preds, truth, std


@pytest.mark.unit
@pytest.mark.parametrize("level", [0.8, 0.95])
def test_a_calibrated_posterior_hits_the_nominal_level(level):
    """The observation is a draw, so a calibrated model's interval contains it
    at the nominal rate."""
    preds, truth, std = calibrated_case()
    row = pp.coverage(preds, truth, std, levels=(level,))['levels'][str(level)]

    assert row['predictive_coverage'] == pytest.approx(level, abs=0.03)


def centred_on_the_measurement(n_obs=3000, n_samples=3000, seed=0, spread=1.0):
    """Draws centred on each measurement, with the measurement's own spread.

    The reference case for ``sample_interval_coverage``: the error bar is a
    fixed window, and a model sitting in the middle of it with the same width
    puts the nominal fraction of its draws inside.
    """
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=n_obs) * 10
    std = np.abs(rng.normal(size=n_obs)) + 0.5
    preds = truth[None, :] + rng.normal(size=(n_samples, n_obs)) * std[None, :] * spread
    return preds, truth, std


@pytest.mark.unit
@pytest.mark.parametrize("level", [0.8, 0.95])
def test_the_nominal_fraction_of_draws_lands_in_the_error_bar(level):
    preds, truth, std = centred_on_the_measurement()
    row = pp.coverage(preds, truth, std, levels=(level,))['levels'][str(level)]

    assert row['sample_interval_coverage'] == pytest.approx(level, abs=0.02)


@pytest.mark.unit
def test_a_posterior_that_is_too_wide_puts_fewer_draws_in_the_window():
    """The reason this is counted over draws rather than over medians: a median
    can sit dead centre while most of the posterior is nowhere near the data."""
    preds, truth, std = centred_on_the_measurement(spread=3.0)
    result = pp.coverage(preds, truth, std)

    for level, row in result['levels'].items():
        assert row['sample_interval_coverage'] < float(level) - 0.2, (level, row)


@pytest.mark.unit
def test_every_observable_gets_its_own_fraction():
    """Kept per observable so the weak ones can be named, not just counted."""
    preds, truth, std = centred_on_the_measurement(n_obs=25, n_samples=400)
    row = pp.coverage(preds, truth, std, levels=(0.8,))['levels']['0.8']

    assert len(row['per_observable']) == 25
    assert all(0.0 <= v <= 1.0 for v in row['per_observable'])
    assert row['sample_interval_coverage'] == pytest.approx(
        float(np.mean(row['per_observable'])), abs=1e-9)


@pytest.mark.unit
def test_a_draw_that_did_not_simulate_is_not_counted_as_a_miss():
    preds, truth, std = centred_on_the_measurement(n_obs=10, n_samples=200)
    preds[:100, 3] = np.nan  # half this observable's draws never ran

    row = pp.coverage(preds, truth, std, levels=(0.8,))['levels']['0.8']
    # Scored on the 100 that did run, so it still sits near nominal rather than
    # being halved by the ones that are simply absent.
    assert row['per_observable'][3] == pytest.approx(0.8, abs=0.12)


@pytest.mark.unit
def test_a_biased_over_confident_posterior_is_caught():
    """Shifted two std and five times too narrow -- the case the check exists for."""
    preds, truth, std = calibrated_case()
    rng = np.random.default_rng(1)
    bad = (truth[None, :] + 2 * std[None, :]
           + rng.normal(size=preds.shape) * std[None, :] * 0.2)

    result = pp.coverage(bad, truth, std)
    for level, row in result['levels'].items():
        assert row['predictive_coverage'] < 0.2, (level, row)


@pytest.mark.unit
def test_observables_that_never_simulated_are_skipped_not_counted_as_misses():
    preds, truth, std = calibrated_case(n_obs=100, n_samples=200)
    preds[:, :20] = np.nan  # twenty observables the model never produced

    result = pp.coverage(preds, truth, std)
    assert result['num_observables'] == 80
    assert result['num_observables_skipped'] == 20
    for row in result['levels'].values():
        assert row['predictive_coverage'] == pytest.approx(0.8, abs=0.15)
        assert len(row['per_observable']) == 80


@pytest.mark.unit
def test_coverage_of_nothing_is_reported_not_divided_by_zero():
    preds = np.full((10, 5), np.nan)
    result = pp.coverage(preds, np.zeros(5), np.ones(5))
    assert result['num_observables'] == 0
    assert result['levels'] == {}


# ── sampling the chain ─────────────────────────────────────────────────────
def write_chain(tmp_path, n_steps=200, n_walkers=8, n_params=3):
    rng = np.random.default_rng(0)
    chain = rng.normal(size=(n_steps, n_walkers, n_params))
    chain[: n_steps // 2] += 100.0   # a burn-in that is obvious if it survives
    np.save(os.path.join(str(tmp_path), pp.CHAIN_FILE), chain)
    return chain


@pytest.mark.unit
def test_burn_in_is_dropped(tmp_path):
    write_chain(tmp_path)
    chain = pp.load_chain(str(tmp_path))
    thetas, info = pp.sample_parameters(chain, num_samples=200, burn_in=0.5)

    assert info['burn_in_steps'] == 100
    assert info['pool'] == 100 * 8
    # The +100 offset lives entirely in the dropped half.
    assert np.abs(thetas).max() < 20


@pytest.mark.unit
def test_burn_in_can_be_given_as_a_number_of_steps(tmp_path):
    write_chain(tmp_path)
    chain = pp.load_chain(str(tmp_path))
    _, info = pp.sample_parameters(chain, num_samples=10, burn_in=150)
    assert info['burn_in_steps'] == 150
    assert info['pool'] == 50 * 8


@pytest.mark.unit
def test_sampling_is_reproducible(tmp_path):
    write_chain(tmp_path)
    chain = pp.load_chain(str(tmp_path))
    a, _ = pp.sample_parameters(chain, num_samples=50, random_seed=7)
    b, _ = pp.sample_parameters(chain, num_samples=50, random_seed=7)
    c, _ = pp.sample_parameters(chain, num_samples=50, random_seed=8)

    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.unit
def test_a_short_chain_draws_with_replacement_and_says_so(tmp_path):
    write_chain(tmp_path, n_steps=4, n_walkers=2)
    chain = pp.load_chain(str(tmp_path))
    thetas, info = pp.sample_parameters(chain, num_samples=100, burn_in=0.5)

    assert thetas.shape[0] == 100
    assert info['drawn_with_replacement'] is True


@pytest.mark.unit
def test_a_missing_chain_says_which_stage_to_run(tmp_path):
    with pytest.raises(pp.PosteriorPredictiveError, match="do_uq"):
        pp.load_chain(str(tmp_path))


@pytest.mark.unit
def test_a_chain_of_the_wrong_shape_is_refused(tmp_path):
    np.save(os.path.join(str(tmp_path), pp.CHAIN_FILE), np.zeros((10, 3)))
    with pytest.raises(pp.PosteriorPredictiveError, match="steps, walkers, params"):
        pp.load_chain(str(tmp_path))


# ── the result ─────────────────────────────────────────────────────────────
def make_result(n_samples=100, n_obs=6):
    preds, truth, std = calibrated_case(n_obs=n_obs, n_samples=n_samples, seed=3)
    return pp.PosteriorPredictiveResult(
        thetas=np.zeros((n_samples, 2)), predictions=preds, ground_truth=truth,
        std=std, labels=['obs%d' % i for i in range(n_obs)],
        coverage_summary=pp.coverage(preds, truth, std),
        chain_info={'n_steps': 200, 'burn_in_steps': 100, 'pool': 800,
                    'drawn_with_replacement': False},
        failures=0, used_emulator=False)


@pytest.mark.unit
def test_intervals_are_ordered():
    lo, median, hi = make_result().intervals(0.95)
    assert np.all(lo <= median) and np.all(median <= hi)


@pytest.mark.unit
def test_saving_writes_both_artefacts(tmp_path):
    samples_path, coverage_path = make_result().save(str(tmp_path))

    assert os.path.isfile(samples_path) and os.path.isfile(coverage_path)
    with np.load(samples_path, allow_pickle=True) as data:
        assert set(data.files) >= {'thetas', 'predictions', 'ground_truth',
                                   'std', 'labels'}
    with open(coverage_path) as file:
        saved = json.load(file)
    assert saved['num_samples'] == 100
    assert saved['used_emulator'] is False
    assert '0.95' in saved['coverage']['levels']


@pytest.mark.unit
def test_the_summary_says_when_it_used_the_emulator():
    """An emulator scoring its own predictions cannot report that it is wrong, so
    a reader has to be told which was run."""
    result = make_result()
    assert 'EMULATOR' not in result.summary()

    result.used_emulator = True
    assert 'EMULATOR' in result.summary()


@pytest.mark.unit
def test_the_summary_reports_failed_samples():
    result = make_result()
    assert 'did not simulate' not in result.summary()

    result.failures = 4
    assert '4 sample(s) did not simulate' in result.summary()


# ── handing over an engine that already exists ─────────────────────────────
class FakeEngine:
    """The surface ``simulate_samples`` and the result need, and nothing else."""

    emulates_features = False

    def get_cost_and_obs_from_params(self, theta):
        return 0.0, [object()]

    def get_obs_output_dict(self, operands, get_all_series=False):
        return {"const": [float(theta) for theta in (1.0, 2.0)]}


class FakeClient:
    def __init__(self, output_dir, emulates=False):
        self.output_dir = output_dir
        self.param_id = FakeEngine()
        self.param_id.emulates_features = emulates
        self.obs_info = {
            "ground_truth_const": np.array([1.0, 2.0]),
            "std_const_vec": np.array([0.1, 0.2]),
            "const_idx_to_obs_idx": [0, 1],
            "experiment_idxs": [0, 0],
            "subexperiment_idxs": [0, 0],
            "obs_names": ["a", "b"],
        }
        self.protocol_info = {"num_sub_per_exp": [1]}


@pytest.mark.unit
def test_an_existing_client_is_used_instead_of_building_one(tmp_path):
    """A run that has just sampled already has an engine; building a second one
    compiles the model again."""
    write_chain(tmp_path, n_params=2)
    result = pp.posterior_predictive(
        client=FakeClient(str(tmp_path)), num_samples=5, save=False)

    assert result.predictions.shape == (5, 2)
    assert result.used_emulator is False


@pytest.mark.unit
def test_a_handed_over_emulator_client_is_labelled_as_one(tmp_path):
    """The caller owns what the engine was built with, so use_emulator only
    labels the summary -- but it must label it correctly."""
    write_chain(tmp_path, n_params=2)
    result = pp.posterior_predictive(
        client=FakeClient(str(tmp_path), emulates=True), num_samples=5,
        use_emulator=False, save=False)

    assert result.used_emulator is True
    assert "EMULATOR" in result.summary()


@pytest.mark.unit
def test_neither_a_config_nor_a_client_is_refused():
    with pytest.raises(pp.PosteriorPredictiveError, match="configuration or an already-built"):
        pp.posterior_predictive()
