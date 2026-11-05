"""`seed` on the population optimisers, and what it is and is not worth (#344).

The 3compartment scaling sweep reported CMA-ES speeding up 22.8x on 8 cores, past the physical
ceiling. The cause is that the population methods stop when they reach `cost_convergence`, so
the work they do is whatever the search happened to need -- and unseeded, that is a fresh draw
every run. A wall-clock ratio across core counts was then measuring how soon each one happened
to converge.

`seed` fixes the draw. It does **not** fix the rank dependence, and these tests pin both halves
so the claim in the README cannot quietly drift into the stronger one.
"""
import numpy as np
import pytest

from libcuflynx.parsers.PrimitiveParsers import param_id_method_options

ng = pytest.importorskip("nevergrad", reason="CMA-ES needs nevergrad")


def _candidates(num_workers, batch, seed, n=16, dim=4):
    """The first ``n`` candidates CMA-ES proposes, asked in batches of ``batch``."""
    import warnings
    import cma.evolution_strategy as ces
    warnings.simplefilter("ignore", ces.InjectionWarning)

    param = ng.p.Array(init=np.zeros(dim), lower=np.full(dim, -1.0), upper=np.full(dim, 1.0))
    opt = ng.optimizers.CMA(parametrization=param, budget=1000, num_workers=num_workers)
    if seed is not None:
        opt.parametrization.random_state = np.random.RandomState(seed)
    out = []
    while len(out) < n:
        asked = [opt.ask() for _ in range(batch)]
        out += [np.asarray(c.value).copy() for c in asked]
        for c in asked:
            opt.tell(c, float(np.sum(np.asarray(c.value) ** 2)))
    return np.array(out[:n])


@pytest.mark.unit
def test_a_seed_makes_two_runs_at_the_same_rank_count_identical():
    """Without this a benchmark cannot be compared with itself, let alone across core counts."""
    assert np.allclose(_candidates(1, 1, seed=0), _candidates(1, 1, seed=0))
    assert np.allclose(_candidates(4, 4, seed=0), _candidates(4, 4, seed=0))


@pytest.mark.unit
def test_unseeded_runs_differ_from_each_other():
    """The behaviour the seed exists to remove -- guards the test above against being vacuous."""
    assert not np.allclose(_candidates(1, 1, seed=None), _candidates(1, 1, seed=None))


@pytest.mark.unit
def test_a_seed_does_not_make_the_search_rank_independent():
    """The half that is easy to assume and is not true.

    CMA-ES asks for one candidate per MPI rank, so the ask/tell interleaving -- and with it the
    trajectory, and the number of evaluations before it stops -- changes with the rank count
    even at a fixed seed. Until the population is decoupled from the rank count, a scaling
    speedup is not a throughput measurement, which is why the sweep records and reports the
    evaluation counts instead of asserting they match.
    """
    reference = _candidates(1, 1, seed=0)
    differing = [w for w in (2, 4, 8)
                 if not np.allclose(_candidates(w, w, seed=0), reference)]
    assert differing == [2, 4, 8], (
        f"only {differing} differed from the 1-rank sequence -- if seeding now does make the "
        f"search rank-independent, the README and the seed option's description say it does "
        f"not, and both need updating")


@pytest.mark.unit
def test_fixing_the_batch_size_is_what_would_make_the_ranks_agree():
    """Names the actual fix, so it is on record rather than in a commit message.

    Asking a fixed population per generation and spreading it over whatever ranks exist makes
    1, 2 and 4 ranks draw the same candidates. 8 still differs because nevergrad derives
    internal settings from ``num_workers`` too, so that has to be pinned as well.
    """
    reference = _candidates(1, 8, seed=0)
    assert np.allclose(_candidates(2, 8, seed=0), reference)
    assert np.allclose(_candidates(4, 8, seed=0), reference)
    assert not np.allclose(_candidates(8, 8, seed=0), reference)


@pytest.mark.unit
@pytest.mark.parametrize("method", ["genetic_algorithm", "CMA-ES"])
def test_the_seed_is_published_in_the_schema(method):
    """CUFLynx builds its settings form from this, so an option it cannot see does not exist."""
    options = {o["name"]: o for o in param_id_method_options(method)}
    assert "seed" in options
    assert options["seed"]["default"] is None, "seeding must stay opt-in"
    # the description must not overclaim, since the test above shows it would be wrong
    assert "rank-independent" in options["seed"]["description"]
