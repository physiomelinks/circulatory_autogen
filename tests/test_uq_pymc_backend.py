"""The pyMC UQ backend's adapter layer (issue #195, from #367).

pymc is an optional [uq] extra, so these cover the parts that must be right *without* it
installed: the emcee-shaped chain conversion, the chain-count arithmetic, the sampler dispatch,
and the failure message a user gets when they select a backend they have not installed.

The sampling itself needs pymc and is exercised by the slow posterior-recovery tests.
"""
import numpy as np
import pytest

from libcuflynx.param_id.pymc_backend import PyMCSampler, _INSTALL_HINT, _import_pymc

pymc_installed = True
try:
    import pymc  # noqa: F401
except ImportError:
    pymc_installed = False


class _FakePosterior:
    """Stands in for arviz's posterior group: (chain, draw) values per named variable."""

    def __init__(self, data):
        self._data = {name: _FakeVar(values) for name, values in data.items()}

    def __contains__(self, name):
        return name in self._data

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, name):
        return self._data[name]


class _FakeVar:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)


class _FakeTrace:
    def __init__(self, data):
        self.posterior = _FakePosterior(data)


# ---------------------------------------------------------------------------
# trace conversion -- the contract with every downstream consumer
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_trace_is_converted_to_emcees_steps_walkers_params_shape():
    """pyMC stores (chain, draw); emcee -- and therefore mcmc_chain.npy, the corner plots and
    the R-hat diagnostic -- expects (steps, walkers, params). Getting the two axes the wrong way
    round does not raise, it silently reports walkers as steps."""
    num_chains, num_draws = 3, 50
    trace = _FakeTrace({
        'a': np.arange(num_chains * num_draws).reshape(num_chains, num_draws),
        'b': np.zeros((num_chains, num_draws)),
    })

    chain = PyMCSampler.trace_to_emcee_chain(trace, ['a', 'b'])

    assert chain.shape == (num_draws, num_chains, 2)
    # element (draw d, chain c, param 0) must be the value pyMC stored at (chain c, draw d)
    assert chain[7, 2, 0] == 2 * num_draws + 7


@pytest.mark.unit
def test_trace_conversion_keeps_parameters_in_the_requested_order():
    """The chain has no column names downstream -- position *is* the identity of a parameter."""
    trace = _FakeTrace({'x': np.full((2, 4), 1.0), 'y': np.full((2, 4), 2.0)})

    chain = PyMCSampler.trace_to_emcee_chain(trace, ['y', 'x'])
    assert np.all(chain[:, :, 0] == 2.0) and np.all(chain[:, :, 1] == 1.0)


@pytest.mark.unit
def test_a_trace_missing_a_parameter_raises_instead_of_returning_none():
    """#367 printed a warning and returned None here. A None becomes an unreadable failure much
    further downstream, after the sampling has already been paid for."""
    trace = _FakeTrace({'a': np.zeros((2, 4))})

    with pytest.raises(ValueError, match='missing sampled parameters'):
        PyMCSampler.trace_to_emcee_chain(trace, ['a', 'b'])


@pytest.mark.unit
def test_a_trace_with_no_posterior_raises():
    class _Empty:
        pass

    with pytest.raises(ValueError, match='no posterior'):
        PyMCSampler.trace_to_emcee_chain(_Empty(), ['a'])


# ---------------------------------------------------------------------------
# chain arithmetic
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_chains_are_split_across_ranks():
    assert PyMCSampler.chains_for_rank(num_walkers=32, num_procs=4) == 8
    assert PyMCSampler.chains_for_rank(num_walkers=32, num_procs=1) == 32


@pytest.mark.unit
def test_more_ranks_than_walkers_still_runs_one_chain_per_rank():
    """num_walkers // num_procs is 0 there, and pyMC asked for zero chains either raises or
    returns an empty trace -- an MPI run that quietly produces no samples."""
    assert PyMCSampler.chains_for_rank(num_walkers=4, num_procs=16) == 1
    assert PyMCSampler.chains_for_rank(num_walkers=1, num_procs=64) == 1


# ---------------------------------------------------------------------------
# optional dependency
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.skipif(pymc_installed, reason='pymc is installed, so the import cannot fail')
def test_selecting_pymc_without_it_installed_names_the_install_command():
    """The failure a user actually hits. A bare ModuleNotFoundError for 'pytensor' does not tell
    anyone which CA setting caused it or what to install."""
    with pytest.raises(ImportError) as excinfo:
        _import_pymc()
    message = str(excinfo.value)
    assert '[uq]' in message and 'pymc' in message
    assert 'library: emcee' in message, 'the message should name the way to carry on without it'


@pytest.mark.unit
def test_the_install_hint_names_the_extra_that_actually_exists():
    """Pins the hint against pyproject, so the extra cannot be renamed and leave the message
    telling users to install something that is not there."""
    import pathlib
    pyproject = (pathlib.Path(__file__).resolve().parent.parent / 'pyproject.toml').read_text()
    assert 'uq = [' in pyproject and 'pymc' in pyproject
    assert '[uq]' in _INSTALL_HINT


@pytest.mark.unit
def test_an_unknown_pymc_method_is_rejected_up_front():
    """Before anything expensive: a typo'd method must not surface after a model has been built
    and an MPI pool opened."""
    with pytest.raises(ValueError, match='unknown pyMC method'):
        PyMCSampler(4, 2, lambda p: 0.0, method='nuts_but_misspelled')


# ---------------------------------------------------------------------------
# dispatch from UQ_options
# ---------------------------------------------------------------------------
def _mcmc_engine(library):
    from libcuflynx.param_id.paramID import MCMC
    obj = MCMC.__new__(MCMC)
    obj.UQ_options = {'library': library, 'num_walkers': 8, 'num_steps': 10}
    obj.num_params = 2
    obj.param_id_info = {'param_names_for_plotting': ['a', 'b']}
    return obj


@pytest.mark.unit
def test_emcee_is_still_the_default_backend():
    import emcee
    sampler = _mcmc_engine('emcee')._build_sampler()
    assert isinstance(sampler, emcee.EnsembleSampler)


@pytest.mark.unit
def test_an_unknown_library_is_rejected_naming_the_valid_ones():
    with pytest.raises(ValueError, match="unknown UQ_options library"):
        _mcmc_engine('stan')._build_sampler()


@pytest.mark.unit
def test_every_backend_exposes_the_same_sampler_surface():
    """The point of the adapter: the sampling loop and everything downstream of mcmc_chain.npy
    are identical whichever library was chosen."""
    for method_name in ('run_mcmc', 'get_chain'):
        assert callable(getattr(PyMCSampler, method_name, None)), method_name
        import emcee
        assert callable(getattr(emcee.EnsembleSampler, method_name, None)), method_name
