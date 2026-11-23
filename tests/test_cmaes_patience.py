"""CMA-ES counts ``max_patience`` in generations, not in single cost evaluations.

On one rank the run used to stop after ``max_patience`` *candidates* without an improvement --
fewer than three generations of nevergrad's 4-candidate population on a 2-parameter problem --
so a windkessel fit with a 600-call budget stopped after 42 evaluations, far from the optimum.
These run the optimiser on analytic costs (no model), on whatever MPI rank count the suite has.
"""
import types

import numpy as np
import pytest

pytest.importorskip("nevergrad", reason="CMA-ES needs nevergrad")

from libcuflynx.param_id.optimisers import CMAESOptimiser

TRUTH = np.array([1e8, 1e-8])
MINS, MAXS = np.array([2e7, 2e-9]), np.array([5e8, 5e-8])


def _run(cost, options):
    """Run CMAESOptimiser from a poor start (3x and 0.3x the truth) and return it."""
    param_id_obj = types.SimpleNamespace(
        param_init=np.array([3e8, 3e-9]),
        get_cost_from_params=lambda x: float(cost(np.asarray(x))))
    info = {"param_names": [["R"], ["C"]], "param_mins": MINS, "param_maxs": MAXS}
    opt = CMAESOptimiser(param_id_obj, info, None, 2, None, dict(options))
    opt.run()
    return opt


def _log_misfit(x):
    return float(np.sum(np.log(x / TRUTH) ** 2))


@pytest.mark.unit
def test_patience_counts_generations_of_the_population():
    """A cost that never improves after the first evaluation stops after max_patience
    generations, not max_patience candidates. nevergrad's population for 2 bounded parameters
    is 4, or ranks + 1 when that is larger, and one generation is asked per loop."""
    opt = _run(lambda x: 1.0, {"num_calls_to_function": 600, "max_patience": 10, "seed": 1})
    popsize = max(4, opt.num_procs + 1)
    if opt.rank == 0:
        # the start value, then max_patience generations that never beat it
        assert opt.num_cost_evals == 1 + 10 * popsize


@pytest.mark.unit
def test_an_unapplied_sigma0_is_reported_not_printed_as_the_step(capsys):
    """nevergrad picks the initial step; the run must not print a sigma it does not use."""
    opt = _run(_log_misfit, {"num_calls_to_function": 8, "sigma0": 0.1, "seed": 1})
    out = capsys.readouterr().out
    assert "Initial sigma" not in out
    assert opt.rank != 0 or "sigma0=0.1 is not applied" in out


@pytest.mark.unit
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_cmaes_reaches_the_optimum_of_a_badly_scaled_two_parameter_problem(seed):
    """Parameters 16 orders of magnitude apart, as R and C of a windkessel. With patience
    counted per candidate this stopped within ~30 evaluations at a cost of order 0.1-1."""
    opt = _run(_log_misfit, {"num_calls_to_function": 600, "max_patience": 10,
                             "cost_convergence": 1e-6, "seed": seed})
    assert opt.best_cost < 1e-6
    assert np.allclose(opt.best_param_vals, TRUTH, rtol=2e-3)


@pytest.mark.unit
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a_run_started_at_the_optimum_never_ends_worse(seed):
    """CMA samples around its start without evaluating it, so a two-parameter run started at
    the optimum (a warm start, a profile likelihood point) could end with a worse cost. The
    start is evaluated first, as one of the budget, so the run ends exactly there."""
    points = []

    def cost(x):
        points.append(np.array(x, dtype=float))
        return _log_misfit(x)

    param_id_obj = types.SimpleNamespace(param_init=TRUTH.copy(),
                                         get_cost_from_params=lambda x: float(cost(np.asarray(x))))
    info = {"param_names": [["R"], ["C"]], "param_mins": MINS, "param_maxs": MAXS}
    opt = CMAESOptimiser(param_id_obj, info, None, 2, None,
                         {"num_calls_to_function": 9, "seed": seed})
    opt.run()
    if opt.rank == 0:
        np.testing.assert_array_equal(points[0], TRUTH)
        assert opt.best_cost == 0.0
        np.testing.assert_array_equal(opt.best_param_vals, TRUTH)
        assert opt.num_cost_evals <= 9  # within the budget, the start included
