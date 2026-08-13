"""MCMC convergence diagnostics: split-R-hat, effective sample size, and the chain plots.

Computed from numpy and emcee rather than arviz. #367 imported arviz at module level for these,
which would have made every calibration run depend on it -- and arviz is not a CA dependency, so
the diagnostics would have been unavailable in exactly the environments that need them.

The tests below check the statistics against chains whose answer is known by construction: a
well-mixed chain, and one whose walkers sit in different places and therefore has not converged.
"""
import os

import numpy as np
import pytest

from param_id.paramID import CVS0DParamID


def _plotter(tmp_path=None, num_params=2):
    """A CVS0DParamID with only the attributes the diagnostics touch."""
    obj = CVS0DParamID.__new__(CVS0DParamID)
    obj.rank = 0
    obj.param_id_info = {'param_names_for_plotting': [f'p_{i}' for i in range(num_params)]}
    obj.file_name_prefix = 'test'
    obj.param_id_obs_file_prefix = 'obs'
    if tmp_path is not None:
        obj.plot_dir = str(tmp_path)
    return obj


def _mixed_chain(num_steps=400, num_walkers=8, num_params=2, seed=0):
    """Independent draws from one distribution: R-hat must be ~1."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(num_steps, num_walkers, num_params))


def _unconverged_chain(num_steps=400, num_walkers=8, num_params=2, seed=0):
    """Each walker parked around a different mean: between-chain variance dominates."""
    rng = np.random.default_rng(seed)
    samples = rng.normal(scale=0.05, size=(num_steps, num_walkers, num_params))
    offsets = np.linspace(-5.0, 5.0, num_walkers)
    return samples + offsets[None, :, None]


# ---------------------------------------------------------------------------
# split R-hat
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_rhat_is_about_one_for_a_well_mixed_chain():
    rhat = _plotter().calc_rhat(_mixed_chain())
    assert set(rhat) == {'p_0', 'p_1'}
    for name, value in rhat.items():
        assert 0.99 < value < 1.01, f'{name}={value}'


@pytest.mark.unit
def test_rhat_is_large_when_the_walkers_have_not_mixed():
    """The diagnostic's whole job: walkers exploring different regions must not be reported as
    converged."""
    rhat = _plotter().calc_rhat(_unconverged_chain())
    for name, value in rhat.items():
        assert value > 1.5, f'{name}={value}'


@pytest.mark.unit
def test_rhat_splits_each_walker_so_a_drifting_chain_is_caught():
    """The *split* in split-R-hat. A walker drifting steadily has a large within-chain variance,
    which is exactly what makes an unsplit R-hat look fine -- so a plain ratio cannot see it."""
    num_steps, num_walkers = 400, 6
    rng = np.random.default_rng(1)
    drift = np.linspace(0.0, 20.0, num_steps)[:, None, None]
    samples = rng.normal(scale=0.1, size=(num_steps, num_walkers, 1)) + drift

    rhat = _plotter(num_params=1).calc_rhat(samples)
    assert rhat['p_0'] > 1.5, 'a steadily drifting chain has not converged'


@pytest.mark.unit
def test_rhat_reports_nan_rather_than_guessing_on_too_short_a_chain():
    rhat = _plotter().calc_rhat(_mixed_chain(num_steps=2))
    assert all(np.isnan(v) for v in rhat.values())


# ---------------------------------------------------------------------------
# effective sample size
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_ess_of_independent_draws_is_close_to_the_number_of_draws():
    samples = _mixed_chain(num_steps=600, num_walkers=8)
    ess = _plotter().calc_effective_sample_size(samples)
    total = 600 * 8
    for name, value in ess.items():
        assert 0.5 * total < value <= total, f'{name}={value} of {total}'


@pytest.mark.unit
def test_ess_is_far_below_the_draw_count_for_a_correlated_chain():
    """The number that matters: correlated draws carry less information than their count
    suggests, which is why ESS rather than num_steps*num_walkers belongs next to a posterior."""
    rng = np.random.default_rng(2)
    num_steps, num_walkers = 2000, 4
    samples = np.zeros((num_steps, num_walkers, 1))
    for walker in range(num_walkers):
        value = 0.0
        for step in range(num_steps):
            value = 0.98 * value + rng.normal(scale=0.1)   # strongly autocorrelated
            samples[step, walker, 0] = value

    ess = _plotter(num_params=1).calc_effective_sample_size(samples)['p_0']
    total = num_steps * num_walkers
    assert ess < 0.1 * total, f'ess={ess} of {total} should reflect the correlation'


@pytest.mark.unit
def test_ess_never_exceeds_the_number_of_draws():
    """A chain cannot carry more independent information than it has draws, whatever the
    autocorrelation estimate does on a short trace."""
    ess = _plotter().calc_effective_sample_size(_mixed_chain(num_steps=30, num_walkers=4))
    for value in ess.values():
        assert np.isnan(value) or value <= 30 * 4


# ---------------------------------------------------------------------------
# summary table
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_posterior_stats_recover_a_known_mean_and_sd():
    rng = np.random.default_rng(3)
    samples = rng.normal(loc=5.0, scale=2.0, size=(500, 8, 1))
    stats = _plotter(num_params=1).get_posterior_stats(samples)['p_0']

    assert stats['mean'] == pytest.approx(5.0, abs=0.1)
    assert stats['sd'] == pytest.approx(2.0, abs=0.1)
    assert stats['hdi_3%'] < stats['mean'] < stats['hdi_97%']
    assert set(stats) == {'mean', 'sd', 'hdi_3%', 'hdi_97%', 'ess', 'r_hat'}


@pytest.mark.unit
def test_the_diagnostics_say_plainly_whether_the_chain_converged(capsys):
    """A diagnostic nobody reads is not a diagnostic: R-hat and ESS are only useful against
    their thresholds, so the verdict is stated rather than left to the reader."""
    _plotter().print_convergence_diagnostics(_mixed_chain())
    assert 'r_hat <= 1.01' in capsys.readouterr().out

    _plotter().print_convergence_diagnostics(_unconverged_chain())
    warned = capsys.readouterr().out
    assert 'WARNING' in warned and 'not mixed' in warned


@pytest.mark.unit
def test_diagnostics_fall_back_to_indices_when_names_do_not_match():
    """A chain with more parameters than names must not raise or mislabel."""
    plotter = _plotter(num_params=2)
    stats = plotter.get_posterior_stats(_mixed_chain(num_params=3))
    assert set(stats) == {'param_0', 'param_1', 'param_2'}


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_autocorrelation_plot_is_written_and_reports_whether_it_decayed(tmp_path):
    plotter = _plotter(tmp_path)
    bounded = plotter.plot_autocorrelation(_mixed_chain(num_steps=500))
    assert bounded is True, 'independent draws decay inside the +-0.1 band'
    assert any(f.startswith('mcmc_autocorrelation') for f in os.listdir(tmp_path))


@pytest.mark.unit
def test_autocorrelation_plot_reports_a_chain_that_has_not_decayed(tmp_path):
    rng = np.random.default_rng(4)
    num_steps = 400
    samples = np.cumsum(rng.normal(size=(num_steps, 4, 1)), axis=0)   # random walk
    assert _plotter(tmp_path, num_params=1).plot_autocorrelation(samples) is False


@pytest.mark.unit
def test_chain_average_plot_is_written(tmp_path):
    assert _plotter(tmp_path).plot_chain_avg(_mixed_chain()) is True
    assert any(f.startswith('mcmc_chain_average') for f in os.listdir(tmp_path))


@pytest.mark.unit
def test_chain_average_plot_skips_a_chain_shorter_than_its_window(tmp_path, capsys):
    assert _plotter(tmp_path).plot_chain_avg(_mixed_chain(num_steps=5)) is None
    assert 'skipping' in capsys.readouterr().out


@pytest.mark.unit
def test_a_single_parameter_chain_still_plots(tmp_path):
    """plt.subplots returns a bare Axes rather than an array for one row, which is a routine
    way for per-parameter plotting to break."""
    plotter = _plotter(tmp_path, num_params=1)
    assert plotter.plot_autocorrelation(_mixed_chain(num_params=1)) is not None
    assert plotter.plot_chain_avg(_mixed_chain(num_params=1)) is True
