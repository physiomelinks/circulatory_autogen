"""Posterior-predictive plots and the prior density (from #367).

These answer "does the calibrated model reproduce the data, and with what spread", which a
best-fit line cannot show. The tests below drive them from stub engines, so they exercise the
aggregation, the file writing and the prior maths without running a simulation.
"""
import os

import numpy as np
import pytest

from libcuflynx.param_id.paramID import CVS0DParamID, integrate_trapezoid


class _StubEngine:
    """An ParamID stand-in: returns canned observables and records resets."""

    def __init__(self, values_by_call, param_id_info=None, best_param_vals=None):
        self._values = values_by_call
        self._call = 0
        self.param_id_info = param_id_info or {}
        self.best_param_vals = best_param_vals
        self.sim_helper = self
        self.resets = 0

    def get_cost_and_obs_from_params(self, param_vals, reset=True):
        value = self._values[self._call % len(self._values)]
        self._call += 1
        return 0.0, [{'const': [value]}]

    def get_obs_output_dict(self, obs_item):
        return obs_item

    def reset_and_clear(self):
        self.resets += 1

    # exercised by get_prior_pdf
    def get_lnprior_from_params(self, param_vals):
        mins = np.asarray(self.param_id_info['param_mins'], dtype=float)
        maxs = np.asarray(self.param_id_info['param_maxs'], dtype=float)
        param_vals = np.asarray(param_vals, dtype=float)
        if np.any(param_vals < mins) or np.any(param_vals > maxs):
            return -np.inf
        prior = self.param_id_info.get('param_prior_types', ['uniform'])[0]
        if prior == 'normal':
            mean = 0.5 * (mins[0] + maxs[0])
            std = (maxs[0] - mins[0]) / 6.0
            return -0.5 * ((param_vals[0] - mean) / std) ** 2
        return 0.0


def _plotter(tmp_path, engine, names=('x',), data_types=('constant',), units=('dimensionless',),
             num_experiments=1, num_sub=(1,)):
    obj = CVS0DParamID.__new__(CVS0DParamID)
    obj.rank = 0
    obj.mcmc_instead = False
    obj.param_id = engine
    obj.plot_dir = str(tmp_path)
    obj.output_dir = str(tmp_path)
    obj.obs_info = {
        'names_for_plotting': list(names),
        'data_types': list(data_types),
        'units': list(units),
        'experiment_idxs': [0] * len(names),
        'subexperiment_idxs': [0] * len(names),
        'ground_truth_const': [1.0] * len(names),
        'std_const_vec': [0.1] * len(names),
        'ground_truth_prob_dist_params': [None] * len(names),
    }
    obj.protocol_info = {
        'num_experiments': num_experiments,
        'num_sub_per_exp': list(num_sub),
        'experiment_labels': ['baseline'],
        'experiment_colors': ['C0'],
    }
    return obj


# ---------------------------------------------------------------------------
# predictive values
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_each_posterior_draw_is_simulated_and_collected(tmp_path):
    engine = _StubEngine([1.0, 2.0, 3.0])
    plotter = _plotter(tmp_path, engine)

    values = plotter._posterior_predictive_values(np.arange(3, dtype=float).reshape(3, 1),
                                                  n_sims=3)

    assert sorted(values['x'][0]) == [1.0, 2.0, 3.0]
    assert engine.resets == 3, 'the simulation must be reset between posterior draws'


@pytest.mark.unit
def test_more_requested_draws_than_samples_uses_what_there_is(tmp_path):
    """np.random.choice(replace=False) raises when asked for more than it has."""
    engine = _StubEngine([1.0])
    plotter = _plotter(tmp_path, engine)
    values = plotter._posterior_predictive_values(np.zeros((2, 1)), n_sims=500)
    assert len(values['x'][0]) == 2


@pytest.mark.unit
def test_a_constant_observation_is_expanded_into_a_comparable_spread(tmp_path):
    """A constant observation is a mean and a std, not samples. Drawing from it makes the
    comparison distribution-against-distribution rather than distribution-against-a-line."""
    plotter = _plotter(tmp_path, _StubEngine([1.0]))
    values = {'x': {}}
    plotter._add_measured_values(values)

    measured = values['x']['exp_data']
    assert len(measured) == 20
    assert np.mean(measured) == pytest.approx(1.0, abs=0.15)


@pytest.mark.unit
def test_distribution_measurements_are_used_as_given(tmp_path):
    """An observation stated as a distribution already *is* samples, so it must not be re-drawn
    from a mean and a std -- it has neither (issue #421)."""
    plotter = _plotter(tmp_path, _StubEngine([1.0]), data_types=('constant',))
    plotter.obs_info['ground_truth_prob_dist_params'] = [{'data_points': [1.0, 2.0, 3.0]}]
    values = {'x': {}}
    plotter._add_measured_values(values)
    assert list(values['x']['exp_data']) == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# csv + figures
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_predictions_are_saved_long_format(tmp_path):
    """The csv is what someone re-plots or re-analyses from without paying for the simulations
    again, so it has to distinguish simulated rows from measured ones."""
    import pandas as pd

    plotter = _plotter(tmp_path, _StubEngine([1.0]))
    plotter.save_posterior_predictions({'x': {0: [1.0, 2.0], 'exp_data': [1.1]}})

    df = pd.read_csv(os.path.join(tmp_path, 'posterior_predictions.csv'))
    assert set(df.columns) == {'feature', 'experiment_idx', 'value', 'data_type'}
    assert sorted(df['data_type'].unique()) == ['experimental', 'simulated']
    assert len(df) == 3


@pytest.mark.unit
def test_the_grid_is_drawn_once_not_once_per_feature(tmp_path, monkeypatch):
    """#367 called plot_distribution_grid inside the per-feature loop, redrawing the whole grid
    once per feature and keeping only the last."""
    pytest.importorskip('seaborn')
    plotter = _plotter(tmp_path, _StubEngine([1.0, 2.0]), names=('a', 'b'),
                       data_types=('constant', 'constant'),
                       units=('dimensionless', 'dimensionless'))

    calls = []
    monkeypatch.setattr(type(plotter), 'plot_distribution_grid',
                        lambda self, values: calls.append(values))

    plotter.plot_boxplots_for_predictions(np.zeros((4, 1)), n_sims=2)
    assert len(calls) == 1


@pytest.mark.unit
def test_a_latex_feature_name_still_produces_a_usable_filename(tmp_path):
    """names_for_plotting are LaTeX-ish (u_{A_{R}}); braces and slashes do not survive as a
    filename, which is the same defect #167 fixed for the sensitivity plots."""
    pytest.importorskip('seaborn')
    plotter = _plotter(tmp_path, _StubEngine([1.0, 2.0]), names=('u_{A_{R}} - exp0',))

    written = plotter.plot_boxplots_for_predictions(np.zeros((3, 1)), n_sims=2)

    assert written, 'a figure should have been written'
    for path in written:
        stem = os.path.basename(path)
        for bad in '{}\\/ ,':
            assert bad not in stem, f'{bad!r} should not survive in {stem!r}'
        assert os.path.isfile(path)


@pytest.mark.unit
def test_the_kde_grid_is_written(tmp_path):
    pytest.importorskip('seaborn')
    plotter = _plotter(tmp_path, _StubEngine([1.0]))
    rng = np.random.default_rng(0)
    values = {'x': {0: list(rng.normal(size=50)), 'exp_data': list(rng.normal(size=50))}}

    path = plotter.plot_distribution_grid(values)
    assert path and os.path.isfile(path)


@pytest.mark.unit
def test_a_feature_with_no_spread_falls_back_to_a_histogram(tmp_path):
    """gaussian_kde needs a non-singular covariance: identical samples raise, and a plot that
    raises loses every other panel in the figure too."""
    pytest.importorskip('seaborn')
    plotter = _plotter(tmp_path, _StubEngine([1.0]))
    values = {'x': {0: [2.0] * 30, 'exp_data': [2.0] * 30}}
    assert plotter.plot_distribution_grid(values) is not None


# ---------------------------------------------------------------------------
# prior pdf
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_uniform_prior_pdf_is_flat_inside_the_bounds_and_zero_outside(tmp_path):
    info = {'param_mins': [0.0], 'param_maxs': [2.0], 'param_prior_types': ['uniform']}
    plotter = _plotter(tmp_path, _StubEngine([1.0], param_id_info=info,
                                             best_param_vals=np.array([1.0])))

    x = np.linspace(-1.0, 3.0, 401)
    pdf = plotter.get_prior_pdf(0, x)

    inside = (x >= 0.0) & (x <= 2.0)
    assert np.allclose(pdf[inside], 0.5, atol=0.02), 'a uniform prior on [0, 2] has density 0.5'
    assert np.all(pdf[~inside] == 0.0)


@pytest.mark.unit
def test_normal_prior_pdf_peaks_at_its_mean_and_integrates_to_one(tmp_path):
    info = {'param_mins': [0.0], 'param_maxs': [6.0], 'param_prior_types': ['normal']}
    plotter = _plotter(tmp_path, _StubEngine([1.0], param_id_info=info,
                                             best_param_vals=np.array([3.0])))

    x = np.linspace(0.0, 6.0, 601)
    pdf = plotter.get_prior_pdf(0, x)

    assert x[int(np.argmax(pdf))] == pytest.approx(3.0, abs=0.05)
    # Imported from paramID rather than called as np.trapz: numpy 2.0 removed that name,
    # and a test that only runs on numpy 1.x is not testing the code CI runs.
    assert integrate_trapezoid(pdf, x) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_the_prior_pdf_comes_from_the_engines_own_prior(tmp_path):
    """Derived from get_lnprior_from_params rather than reimplemented, so params_for_id's
    prior_mean / prior_std / prior_origin / prior_scale and the unbounded flag are honoured --
    #367 restated the old hardcoded defaults here, which a user's hyper-parameters would have
    silently disagreed with."""
    info = {'param_mins': [0.0], 'param_maxs': [2.0], 'param_prior_types': ['uniform']}
    engine = _StubEngine([1.0], param_id_info=info, best_param_vals=np.array([1.0]))
    plotter = _plotter(tmp_path, engine)

    seen = []
    original = engine.get_lnprior_from_params
    engine.get_lnprior_from_params = lambda p: (seen.append(np.copy(p)), original(p))[1]

    plotter.get_prior_pdf(0, np.linspace(0.0, 2.0, 11))
    assert len(seen) == 11, 'the engine prior must be evaluated at every plotted point'
