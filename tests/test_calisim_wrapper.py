"""Unit tests for the calisim calibration backend (param_id/calisim_wrapper.py).

These exercise the CA <-> calisim translation with a stub cost function, so they run without a
model and (except where marked) without calisim installed. The end-to-end runs against a real
generated model live in tests/test_param_id.py.
"""
import os

import numpy as np
import pytest

from param_id.calisim_methods import (
    calisim_method_name,
    calisim_param_id_methods,
    is_calisim_method,
    split_calisim_method_name,
)
from param_id.calisim_wrapper import FAILED_SIMULATION_COST, CalisimOptimiser
from parsers.PrimitiveParsers import PARAM_ID_METHODS, param_id_method_options
from utility_funcs import Normalise_class

pytestmark = pytest.mark.unit

# The three engine/method pairs the integration tests use: the two most common optuna samplers
# plus differential evolution through OpenTURNS/Pagmo.
COMMON_METHODS = ['calisim_optuna_tpes', 'calisim_optuna_cmaes', 'calisim_openturns_de']


class _StubParamID:
    """Stands in for OpencorParamID: a quadratic cost with a known minimum."""

    def __init__(self, optimum=(1.5, 3.0), fail_every=0):
        self.optimum = np.asarray(optimum, dtype=float)
        self.fail_every = fail_every
        self.num_calls = 0
        self.best_param_vals_set = None

    def get_cost_from_params(self, param_vals, reset=True):
        self.num_calls += 1
        if self.fail_every and self.num_calls % self.fail_every == 0:
            return np.inf
        return float(np.sum((np.asarray(param_vals) - self.optimum) ** 2))

    def set_best_param_vals(self, param_vals):
        self.best_param_vals_set = param_vals


def _make_optimiser(tmp_path, param_id_method='calisim_optuna_tpes', stub=None, **options):
    param_id_info = {
        'param_names': [['heart/R_a', 'aorta/R_a'], ['venous/C']],
        'param_mins': np.array([0.0, 0.0]),
        'param_maxs': np.array([5.0, 5.0]),
    }
    norm_obj = Normalise_class(param_id_info['param_mins'], param_id_info['param_maxs'])
    # OpencorParamID.run() writes the param-name header before the optimiser starts.
    with open(os.path.join(tmp_path, 'best_param_vals_history.csv'), 'w') as f:
        f.write('heart R_a, venous C\n')
    opts = {'num_calls_to_function': 20, 'n_init': 4, 'cost_convergence': 1e-6}
    opts.update(options)
    optimiser = CalisimOptimiser(
        stub if stub is not None else _StubParamID(), param_id_info, norm_obj, 2, str(tmp_path),
        optimiser_options=opts, param_id_method=param_id_method)
    return optimiser


# -------------------------------------------------------------------------------------------
# schema / method names
# -------------------------------------------------------------------------------------------
def test_calisim_methods_are_registered_in_the_param_id_schema():
    """CUFLynx builds its calibration menu from PARAM_ID_METHODS, so the calisim engines must be
    merged in automatically -- without that, none of them are selectable in the GUI."""
    for method in COMMON_METHODS:
        assert method in PARAM_ID_METHODS, f'{method} missing from PARAM_ID_METHODS'
        meta = PARAM_ID_METHODS[method]
        assert meta['label'] and meta['description']
        assert meta['gradient_based'] is False
        assert meta['calisim']['engine'] and 'method' in meta['calisim']
        # the per-method settings form is populated from here
        assert {opt['name'] for opt in param_id_method_options(method)} >= {
            'num_calls_to_function', 'cost_convergence', 'method_kwargs'}


def test_calisim_method_names_round_trip():
    for engine, method in [('optuna', 'tpes'), ('openturns', 'simulated_annealing'),
                           ('emukit', '')]:
        name = calisim_method_name(engine, method)
        assert is_calisim_method(name)
        assert split_calisim_method_name(name) == (engine, method)

    assert not is_calisim_method('genetic_algorithm')
    with pytest.raises(ValueError):
        split_calisim_method_name('genetic_algorithm')


def test_calisim_schema_covers_every_engine_and_is_well_formed():
    schema = calisim_param_id_methods()
    engines = {meta['calisim']['engine'] for meta in schema.values()}
    assert {'optuna', 'openturns', 'emukit', 'botorch'} <= engines
    # surrogate-based engines expose the acquisition function; samplers do not
    assert 'acquisition_func' in {o['name'] for o in schema['calisim_emukit']['options']}
    assert 'acquisition_func' not in {
        o['name'] for o in schema['calisim_optuna_tpes']['options']}


# -------------------------------------------------------------------------------------------
# params_for_id / obs_data -> calisim inputs
# -------------------------------------------------------------------------------------------
def test_param_names_are_calisim_safe_and_ordered(tmp_path):
    """CA parameter names are `component/variable` and may be shared across components, which is
    not a usable calisim/optuna key, so they are sanitised while keeping param_id_info order."""
    optimiser = _make_optimiser(tmp_path)
    assert optimiser.calisim_param_names == ['p0_heart_R_a', 'p1_venous_C']
    # and the mapping back to CA's ordered vector is the exact inverse
    values = {'p0_heart_R_a': 1.25, 'p1_venous_C': 4.0}
    assert np.array_equal(optimiser._params_dict_to_array(values), np.array([1.25, 4.0]))


def test_bounds_from_params_for_id_become_uniform_distributions(tmp_path):
    """The min/max columns of {prefix}_params_for_id.csv are the calisim search box, in real
    (un-normalised) parameter space."""
    pytest.importorskip('calisim')
    optimiser = _make_optimiser(tmp_path)
    spec = optimiser._parameter_spec()
    assert [p.name for p in spec.parameters] == optimiser.calisim_param_names
    for parameter, lower, upper in zip(spec.parameters, [0.0, 0.0], [5.0, 5.0]):
        assert parameter.distribution_name == 'uniform'
        assert parameter.distribution_args == [lower, upper]


# -------------------------------------------------------------------------------------------
# calisim outputs -> CA outputs
# -------------------------------------------------------------------------------------------
def test_evaluate_tracks_best_and_writes_ca_artifacts(tmp_path):
    optimiser = _make_optimiser(tmp_path)
    # deliberately never hits the optimum exactly, so cost_convergence does not cut the run short
    optimiser._evaluate({'p0_heart_R_a': 0.0, 'p1_venous_C': 0.0})
    optimiser._evaluate({'p0_heart_R_a': 1.4, 'p1_venous_C': 3.1})  # best of the three
    optimiser._evaluate({'p0_heart_R_a': 5.0, 'p1_venous_C': 5.0})

    assert optimiser.best_cost == pytest.approx(0.02)
    assert np.allclose(optimiser.best_param_vals, [1.4, 3.1])
    # CA's usual artefacts, so plotting/simulate_with_best_param_vals need no special-casing
    assert np.load(os.path.join(tmp_path, 'best_cost.npy')) == pytest.approx(0.02)
    assert np.allclose(np.load(os.path.join(tmp_path, 'best_param_vals.npy')), [1.4, 3.1])

    costs = np.atleast_1d(np.loadtxt(os.path.join(tmp_path, 'best_cost_history.csv'),
                                     delimiter=','))
    assert len(costs) == 3 and np.all(np.diff(costs) <= 0), 'best-cost history must be monotone'
    params = np.loadtxt(os.path.join(tmp_path, 'best_param_vals_history.csv'), delimiter=',',
                        skiprows=1)
    # history is in normalised space, like every other optimiser writes it
    assert params.shape == (3, 2)
    assert np.allclose(params[-1], [1.4 / 5.0, 3.1 / 5.0])


def test_failed_simulation_becomes_a_finite_penalty(tmp_path):
    """np.inf poisons the surrogate fits the calisim engines build, so failures are reported as a
    large finite cost and never become the incumbent."""
    stub = _StubParamID(fail_every=1)
    optimiser = _make_optimiser(tmp_path, stub=stub)
    assert optimiser._evaluate({'p0_heart_R_a': 1.0, 'p1_venous_C': 1.0}) == FAILED_SIMULATION_COST
    assert optimiser.best_param_vals is None
    assert optimiser.best_cost == np.inf


def test_budget_is_enforced_even_if_the_engine_overshoots(tmp_path):
    """calisim owns the loop and some engines ask for more points than n_iterations, so the
    wrapper stops simulating once num_calls_to_function is spent."""
    stub = _StubParamID()
    optimiser = _make_optimiser(tmp_path, stub=stub, num_calls_to_function=5)
    for i in range(20):
        optimiser._evaluate({'p0_heart_R_a': 0.1 * i, 'p1_venous_C': 0.1 * i})
    assert stub.num_calls == 5


def test_cost_convergence_stops_the_run_early(tmp_path):
    stub = _StubParamID()
    optimiser = _make_optimiser(tmp_path, stub=stub, cost_convergence=1e3)
    optimiser._evaluate({'p0_heart_R_a': 1.5, 'p1_venous_C': 3.0})
    assert optimiser._stopped_early
    optimiser._evaluate({'p0_heart_R_a': 0.0, 'p1_venous_C': 0.0})
    assert stub.num_calls == 1, 'no further simulations after convergence'


def test_missing_calisim_raises_a_helpful_import_error(tmp_path, monkeypatch):
    """calisim is an optional dependency: CA must import fine without it and only complain when a
    calisim_* method is actually selected."""
    import param_id.calisim_wrapper as wrapper
    monkeypatch.setattr(wrapper, 'CALISIM_AVAILABLE', False)
    optimiser = _make_optimiser(tmp_path)
    with pytest.raises(ImportError, match='pip install calisim'):
        optimiser.run()


# -------------------------------------------------------------------------------------------
# end to end through calisim itself (still no model: the stub is the "simulator")
# -------------------------------------------------------------------------------------------
@pytest.mark.parametrize('param_id_method', COMMON_METHODS)
def test_calisim_engines_optimise_the_stub_cost(tmp_path, param_id_method):
    """The three most common methods really drive CA's cost function through calisim and come
    back with a better-than-random optimum, in CA's format."""
    pytest.importorskip('calisim')
    stub = _StubParamID()
    optimiser = _make_optimiser(tmp_path, param_id_method=param_id_method, stub=stub,
                                num_calls_to_function=30, n_init=5, random_seed=7)
    optimiser.run()

    assert stub.num_calls <= 30
    assert np.isfinite(optimiser.best_cost)
    # a uniform draw over the 5x5 box averages a cost of ~8; anything sensible beats 2
    assert optimiser.best_cost < 2.0, f'{param_id_method} did not optimise: {optimiser.best_cost}'
    assert optimiser.best_param_vals.shape == (2,)
    assert stub.best_param_vals_set is not None, 'the best params must be pushed back to param_id'
    assert np.load(os.path.join(tmp_path, 'best_cost.npy')) == pytest.approx(optimiser.best_cost)
