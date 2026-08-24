"""A UQ run summarises the posterior; it does not replace the calibration's best fit.

``MCMC.run`` used to overwrite ``best_param_vals.npy`` / ``best_cost.npy`` with the
posterior median whenever that median happened to score a lower cost. Two different estimators
were being conflated -- a median summarises a distribution, a calibration best is an argmin --
so a UQ run silently mutated the calibration's answer, and the file gave no clue which estimator
it held. It now writes ``mcmc_statistics.json`` alongside instead.
"""
import json
import os

import numpy as np
import pytest

from libcuflynx.param_id.paramID import MCMC


class _StubMCMC:
    """An MCMC with only what the statistics path touches."""

    def __new__(cls, *args, **kwargs):
        return MCMC.__new__(MCMC)


def _engine(tmp_path, num_params=2, best_param_vals=None, best_cost=None,
            costs=(0.5, 0.7), UQ_options=None, names=None):
    obj = MCMC.__new__(MCMC)
    obj.output_dir = str(tmp_path)
    obj.num_params = num_params
    obj.best_param_vals = best_param_vals
    obj.best_cost = best_cost
    obj.UQ_options = UQ_options if UQ_options is not None else {}
    obj.param_id_info = {'param_names_for_plotting': names
                         or [f'p_{i}' for i in range(num_params)]}
    obj._costs = list(costs)
    obj.get_cost_and_obs_from_params = lambda vals, reset=True: (obj._costs.pop(0), None)
    return obj


def _read(tmp_path):
    with open(os.path.join(tmp_path, 'mcmc_statistics.json')) as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# the calibration's best fit is left alone
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_a_uq_run_does_not_overwrite_the_calibration_best_fit(tmp_path):
    """Even when the posterior median scores lower -- which is exactly when the old code
    replaced it."""
    calibration_best = np.array([1.0, 2.0])
    engine = _engine(tmp_path, best_param_vals=calibration_best, best_cost=0.9,
                     costs=(0.1, 0.2))          # median cost 0.1, far below the calibration's

    engine.save_mcmc_statistics(np.random.default_rng(0).normal(size=(200, 2)))

    assert np.array_equal(engine.best_param_vals, calibration_best)
    assert engine.best_cost == 0.9
    assert not os.path.exists(os.path.join(tmp_path, 'best_param_vals.npy'))
    assert not os.path.exists(os.path.join(tmp_path, 'best_cost.npy'))
    assert _read(tmp_path)['source'] == 'calibration'


@pytest.mark.unit
def test_the_comparison_is_still_reported_just_not_acted_on(capsys, tmp_path):
    """A median that beats the optimum is informative -- it usually means the calibration
    stopped early, or the posterior is skewed. Worth saying; not worth deciding on."""
    engine = _engine(tmp_path, best_param_vals=np.array([1.0, 2.0]), best_cost=0.9,
                     costs=(0.1, 0.2))
    engine.save_mcmc_statistics(np.random.default_rng(0).normal(size=(100, 2)))

    printed = capsys.readouterr().out
    assert 'left' in printed and 'unchanged' in printed
    assert 'lower cost than the calibration' in printed

    document = _read(tmp_path)
    assert document['cost_at_posterior_median'] == 0.1
    assert document['cost_at_posterior_mean'] == 0.2
    assert document['calibration_best_cost'] == 0.9


@pytest.mark.unit
def test_a_uq_only_run_still_writes_a_best_fit(tmp_path):
    """Not an overwrite: nothing else has written one, the median is the only estimate there is,
    and the rest of the pipeline needs it. The file records where it came from."""
    engine = _engine(tmp_path, best_param_vals=None, best_cost=None, costs=(0.3, 0.4))
    samples = np.random.default_rng(1).normal(loc=[5.0, -2.0], size=(300, 2))

    engine.save_mcmc_statistics(samples)

    assert engine.best_param_vals is not None
    assert os.path.isfile(os.path.join(tmp_path, 'best_param_vals.npy'))
    saved = np.load(os.path.join(tmp_path, 'best_param_vals.npy'))
    assert saved == pytest.approx(np.median(samples, axis=0))

    document = _read(tmp_path)
    assert document['source'] == 'posterior_median'
    assert document['calibration_best_cost'] is None


# ---------------------------------------------------------------------------
# the statistics themselves
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_summary_reports_a_spread_not_a_winner(tmp_path):
    """A posterior is a distribution; summarising it as one number is what caused the
    conflation in the first place."""
    rng = np.random.default_rng(2)
    samples = rng.normal(loc=[3.0, 10.0], scale=[1.0, 2.0], size=(20000, 2))
    engine = _engine(tmp_path, costs=(0.1, 0.2), best_param_vals=np.zeros(2), best_cost=1.0)

    stats, _, _ = engine.posterior_statistics(samples)

    assert set(stats) == {'p_0', 'p_1'}
    assert set(stats['p_0']) == {'mean', 'median', 'sd', 'q2.5', 'q25', 'q75', 'q97.5',
                                 'min', 'max'}
    assert stats['p_0']['mean'] == pytest.approx(3.0, abs=0.05)
    assert stats['p_0']['median'] == pytest.approx(3.0, abs=0.05)
    assert stats['p_1']['sd'] == pytest.approx(2.0, abs=0.05)
    # a 95% interval on a normal is about +-1.96 sigma
    assert stats['p_1']['q2.5'] == pytest.approx(10.0 - 1.96 * 2.0, abs=0.15)
    assert stats['p_1']['q97.5'] == pytest.approx(10.0 + 1.96 * 2.0, abs=0.15)
    assert stats['p_0']['q25'] < stats['p_0']['median'] < stats['p_0']['q75']


@pytest.mark.unit
def test_the_statistics_are_written_as_readable_json(tmp_path):
    engine = _engine(tmp_path, best_param_vals=np.array([1.0, 2.0]), best_cost=0.5,
                     costs=(0.6, 0.7))
    engine.save_mcmc_statistics(np.random.default_rng(3).normal(size=(150, 2)))

    document = _read(tmp_path)
    assert document['num_samples'] == 150
    assert set(document['parameters']) == {'p_0', 'p_1'}
    # every value must be a plain float, not a numpy scalar, or json.dump would have failed
    assert isinstance(document['parameters']['p_0']['median'], float)


# ---------------------------------------------------------------------------
# burn_in -- a setting the schema advertised but nothing read
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_burn_in_below_one_is_a_fraction_of_the_chain(tmp_path):
    engine = _engine(tmp_path, UQ_options={'burn_in': 0.25})
    assert engine.burn_in_index(400) == 100


@pytest.mark.unit
def test_burn_in_of_one_or_more_is_a_number_of_steps(tmp_path):
    engine = _engine(tmp_path, UQ_options={'burn_in': 120})
    assert engine.burn_in_index(400) == 120


@pytest.mark.unit
def test_burn_in_defaults_to_half_the_chain(tmp_path):
    """What this used to hardcode as samples[shape[0]//2:]."""
    assert _engine(tmp_path, UQ_options={}).burn_in_index(400) == 200


@pytest.mark.unit
def test_a_burn_in_longer_than_the_run_keeps_the_last_sample(capsys, tmp_path):
    """An empty array would give a stack of nan statistics and no indication why."""
    engine = _engine(tmp_path, UQ_options={'burn_in': 5000})
    assert engine.burn_in_index(400) == 399
    assert 'discards all' in capsys.readouterr().out


@pytest.mark.unit
def test_a_nonsense_burn_in_falls_back_with_a_warning(capsys, tmp_path):
    engine = _engine(tmp_path, UQ_options={'burn_in': 'half'})
    assert engine.burn_in_index(400) == 200
    assert 'not a number' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# parameter naming
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_a_grouped_parameter_is_labelled_by_its_first_member(tmp_path):
    """A grouped row calibrates one value shared across several model variables: one parameter,
    several names."""
    engine = _engine(tmp_path, num_params=2,
                     names=[['heart/C', 'aorta/C'], 'heart/R'])
    assert engine.flat_param_names() == ['heart/C', 'heart/R']


@pytest.mark.unit
def test_missing_names_fall_back_to_indices(tmp_path):
    engine = _engine(tmp_path, num_params=3, names=['only_one'])
    engine.param_id_info = {}
    assert engine.flat_param_names() == ['param_0', 'param_1', 'param_2']


@pytest.mark.unit
def test_names_held_as_a_numpy_array_are_handled(tmp_path):
    """param_id_info holds these as numpy arrays in a real run, and `not array` raises rather
    than answering -- which is exactly how this broke the first time."""
    engine = _engine(tmp_path, num_params=2)
    engine.param_id_info = {
        'param_names_for_plotting': np.array(['q_{sbv}', 'C_{ao}']),
        'param_names': np.array(['a', 'b']),
    }
    assert engine.flat_param_names() == ['q_{sbv}', 'C_{ao}']


@pytest.mark.unit
def test_a_best_cost_stored_as_an_array_is_compared_as_a_number(tmp_path):
    """best_cost comes back off disk as a numpy array, and comparing one with `<` inside an
    `if` is ambiguous the moment it is not 0-d."""
    engine = _engine(tmp_path, best_param_vals=np.array([1.0, 2.0]),
                     best_cost=np.array([0.9]), costs=(0.1, 0.2))

    engine.save_mcmc_statistics(np.random.default_rng(4).normal(size=(100, 2)))

    document = _read(tmp_path)
    assert document['calibration_best_cost'] == pytest.approx(0.9)
    assert document['source'] == 'calibration'


# ---------------------------------------------------------------------------
# an unknown calibration cost
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_an_infinite_best_cost_is_reported_as_unknown_not_as_infinity(capsys, tmp_path):
    """A UQ run is often handed the calibration's *parameters* without its cost, leaving the
    in-memory best_cost at inf. Writing that gives the file a JSON Infinity -- which strict
    parsers reject -- and makes every posterior median look like it beat the calibration,
    because everything is lower than inf."""
    engine = _engine(tmp_path, best_param_vals=np.array([1.0, 2.0]), best_cost=np.inf,
                     costs=(2.1, 2.0))

    engine.save_mcmc_statistics(np.random.default_rng(5).normal(size=(100, 2)))

    document = _read(tmp_path)
    assert document['calibration_best_cost'] is None
    printed = capsys.readouterr().out
    assert 'unknown' in printed
    assert 'lower cost than the calibration' not in printed, \
        'there was no number to beat, so nothing should claim it was beaten'

    # and the file must be strict JSON -- Infinity is not
    with open(os.path.join(tmp_path, 'mcmc_statistics.json')) as handle:
        json.loads(handle.read(), parse_constant=_reject_constant)


def _reject_constant(value):
    raise AssertionError(f'mcmc_statistics.json contains non-standard JSON constant {value!r}')


@pytest.mark.unit
def test_the_calibration_cost_is_read_from_disk_when_memory_has_none(tmp_path):
    """best_cost.npy is what the calibration actually wrote; the in-memory value may never have
    been set on this object."""
    np.save(os.path.join(tmp_path, 'best_cost'), np.array(0.42))
    engine = _engine(tmp_path, best_param_vals=np.array([1.0, 2.0]), best_cost=np.inf,
                     costs=(0.9, 1.0))

    assert engine.calibration_best_cost() == pytest.approx(0.42)

    engine.save_mcmc_statistics(np.random.default_rng(6).normal(size=(80, 2)))
    assert _read(tmp_path)['calibration_best_cost'] == pytest.approx(0.42)
