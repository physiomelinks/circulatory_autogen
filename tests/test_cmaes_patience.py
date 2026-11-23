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
        assert opt.num_cost_evals == 10 * popsize


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
def test_a_cmaes_run_given_sigma0_is_refused(capsys):
    """sigma0 was never applied (nevergrad picks the initial step) and has been removed: a run
    given it stops with a message saying to delete the line, and the schema no longer lists it.
    A run without it prints no initial step either."""
    from libcuflynx.parsers.PrimitiveParsers import param_id_method_options
    with pytest.raises(ValueError, match="'sigma0' has been removed.*Delete the sigma0 line "
                                         "from optimiser_options"):
        _run(_log_misfit, {"num_calls_to_function": 8, "sigma0": 0.1, "seed": 1})
    _run(_log_misfit, {"num_calls_to_function": 8, "seed": 1})
    assert "sigma" not in capsys.readouterr().out.lower()
    assert "sigma0" not in [o["name"] for o in param_id_method_options("CMA-ES")]


@pytest.mark.unit
@pytest.mark.parametrize("where", ["optimiser_options", "debug_optimiser_options"])
def test_user_inputs_naming_sigma0_are_refused(where):
    from libcuflynx.parsers.PrimitiveParsers import refuse_removed_user_inputs
    with pytest.raises(ValueError, match=f"Delete the sigma0 line from {where}"):
        refuse_removed_user_inputs({where: {"num_calls_to_function": 8, "sigma0": 0.1}})
    refuse_removed_user_inputs({where: {"num_calls_to_function": 8}})
