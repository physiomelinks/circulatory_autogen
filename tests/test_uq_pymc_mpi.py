"""A multi-rank UQ run with ``library: pymc`` used to hang forever.

The two backends parallelise in opposite directions. emcee and zeus advance one ensemble in one
process and farm the likelihood out to a ``schwimmbad.MPIPool``, where every other rank sits in
``pool.wait()`` until the master closes the pool. pyMC has no such hook: ``PyMCSampler`` gives
each rank chains of its own and gathers them at the end, so every rank has to reach
``run_mcmc`` and its ``comm.Barrier()``/``comm.gather`` are collectives over COMM_WORLD.

``OpencorMCMC.run`` opened the pool for both. So under ``mpiexec -n >1`` with pymc, the workers
were parked in a receive that only the master's ``pool.close()`` ends, and the master waited on
a barrier they could never reach: the run hung after sampling, with no error and no chain. It
went unnoticed because nothing exercised it -- the pyMC tests all run on one rank, where the
pool branch is not taken.

The dispatch tests below run everywhere. The last one actually launches ``mpiexec -n 2`` and is
the regression test proper: before the fix it times out, which is what a hang looks like when
something is watching it.
"""
import os
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from libcuflynx.param_id.paramID import OpencorMCMC
from libcuflynx.utilities.mpi_utils import LAUNCHER_ENV_VARS

pymc_installed = True
try:
    import pymc  # noqa: F401
except ImportError:
    pymc_installed = False

pytestmark = pytest.mark.unit

NUM_PARAMS = 2


class _StubNorm:
    """param_norm_obj, which only has to round-trip here."""

    @staticmethod
    def normalise(vals):
        return np.asarray(vals, dtype=float)

    @staticmethod
    def unnormalise(vals):
        return np.asarray(vals, dtype=float)


def _engine(library='emcee', num_walkers=8, best_param_vals=None):
    engine = OpencorMCMC.__new__(OpencorMCMC)
    engine.UQ_options = {'library': library, 'num_walkers': num_walkers, 'num_steps': 10}
    engine.num_params = NUM_PARAMS
    engine.param_norm_obj = _StubNorm()
    engine.best_param_vals = best_param_vals
    return engine


# ---------------------------------------------------------------------------
# which arrangement each backend gets
# ---------------------------------------------------------------------------
def test_emcee_and_zeus_want_the_worker_pool():
    """They drive one ensemble from one process; the ranks are there to evaluate likelihoods."""
    assert _engine('emcee').sampler_needs_a_worker_pool() is True
    assert _engine('zeus').sampler_needs_a_worker_pool() is True


def test_the_default_backend_still_wants_the_worker_pool():
    """UQ_options with no library at all is emcee -- the MPI behaviour must not change for it."""
    engine = _engine()
    del engine.UQ_options['library']
    assert engine.sampler_needs_a_worker_pool() is True


def test_pymc_does_not_want_a_worker_pool():
    """The fix. A pool for pyMC parks the ranks where they can never reach its collectives."""
    assert _engine('pymc').sampler_needs_a_worker_pool() is False


# ---------------------------------------------------------------------------
# splitting the ensemble across ranks
# ---------------------------------------------------------------------------
def test_each_rank_gets_its_own_walkers():
    """Every rank taking the same first few walkers would sample the same starts repeatedly and
    report them as an ensemble of independent chains."""
    positions = np.arange(8 * NUM_PARAMS, dtype=float).reshape(8, NUM_PARAMS)

    slices = [OpencorMCMC._walkers_for_rank(positions, rank, 4) for rank in range(4)]

    assert [s.shape for s in slices] == [(2, NUM_PARAMS)] * 4
    np.testing.assert_allclose(np.concatenate(slices), positions, err_msg='the ensemble should '
                               'be covered exactly once across the ranks')


def test_the_split_matches_the_number_of_chains_the_backend_will_run():
    """_walkers_for_rank and PyMCSampler.chains_for_rank decide from the same two numbers; if
    they disagree a rank starts chains from another rank's positions."""
    from libcuflynx.param_id.pymc_backend import PyMCSampler

    positions = np.zeros((32, NUM_PARAMS))
    for num_procs in (1, 2, 4, 5, 64):
        for rank in range(num_procs):
            assert len(OpencorMCMC._walkers_for_rank(positions, rank, num_procs)) == \
                PyMCSampler.chains_for_rank(32, num_procs), (num_procs, rank)


def test_more_ranks_than_walkers_wraps_rather_than_starving_a_rank():
    """chains_for_rank never returns zero, so every rank runs a chain and needs a start."""
    positions = np.arange(3 * NUM_PARAMS, dtype=float).reshape(3, NUM_PARAMS)

    slices = [OpencorMCMC._walkers_for_rank(positions, rank, 6) for rank in range(6)]

    assert all(s.shape == (1, NUM_PARAMS) for s in slices)
    np.testing.assert_allclose(slices[3][0], positions[0], err_msg='rank 3 should wrap to the '
                               'start of the ensemble rather than run off the end')


# ---------------------------------------------------------------------------
# the starting ensemble
# ---------------------------------------------------------------------------
def test_walker_positions_are_walkers_by_params():
    """The orientation the samplers take. Transposed, it is a chain of the wrong length in the
    wrong number of parameters, which does not raise -- it samples something else."""
    assert _engine(num_walkers=8)._initial_walker_positions(0.1).shape == (8, NUM_PARAMS)


def test_positions_cluster_around_the_calibrated_fit_when_there_is_one():
    best = np.array([0.5, 0.25])
    positions = _engine(num_walkers=200, best_param_vals=best)._initial_walker_positions(0.01)

    np.testing.assert_allclose(positions.mean(axis=0), best, atol=0.01)
    assert positions.std(axis=0).max() < 0.05, 'a tight ball, not a draw over the whole box'


def test_positions_spread_over_the_prior_box_when_there_is_no_fit():
    positions = _engine(num_walkers=400)._initial_walker_positions(0.01)

    assert positions.min() >= 0.0 and positions.max() <= 1.0
    assert positions.std(axis=0).min() > 0.2, 'should cover the box, not sit in a ball'


# ---------------------------------------------------------------------------
# what run() actually does with more than one rank
# ---------------------------------------------------------------------------
class _FakeComm:
    def __init__(self, rank, size, broadcast=None):
        self._rank, self._size, self._broadcast = rank, size, broadcast

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return self._size

    def bcast(self, obj, root=0):
        return obj if self._rank == root else self._broadcast


class _FakeMPI:
    def __init__(self, comm):
        self.COMM_WORLD = comm


class _StubSampler:
    def get_chain(self):
        return np.zeros((10, 2, NUM_PARAMS))


def _run_with(monkeypatch, engine, rank, size, broadcast=None, tmp_path=None):
    """Drive OpencorMCMC.run with a fake COMM_WORLD, recording what it chose to do.

    The pool is faked rather than forbidden outright: ``run`` wraps ``MPIPool()`` in a bare
    ``except: return``, so a fake that raised would be swallowed and read as "no pool opened" --
    the very thing under test.
    """
    import schwimmbad

    import libcuflynx.param_id.paramID as paramID

    pools = []

    class _FakePool:
        def __init__(self, *args, **kwargs):
            pools.append(self)

        def is_master(self):
            return True

        def close(self):
            pass

    monkeypatch.setattr(paramID, 'MPI', _FakeMPI(_FakeComm(rank, size, broadcast)))
    monkeypatch.setattr(schwimmbad, 'MPIPool', _FakePool)

    sampled = {'pools': pools}
    engine.sampler = None
    engine.output_dir = str(tmp_path) if tmp_path is not None else None
    engine.save_mcmc_statistics = lambda flat_samples: sampled.update(stats=flat_samples.shape)

    def build(pool=None):
        sampled['pool'] = pool
        engine.sampler = _StubSampler()
        return engine.sampler

    engine._build_sampler = build
    engine._sample = lambda positions, **kwargs: sampled.update(positions=np.asarray(positions),
                                                                kwargs=kwargs)
    engine.run()
    return sampled


def test_a_multi_rank_pymc_run_opens_no_pool_and_samples_on_every_rank(monkeypatch, tmp_path):
    """The regression, at the point the decision is made. Opening the pool here is what parked
    the workers where the sampler's collectives could not reach them."""
    positions = np.zeros((8, NUM_PARAMS))

    for rank in range(4):
        sampled = _run_with(monkeypatch, _engine('pymc'), rank, 4, broadcast=positions,
                            tmp_path=tmp_path)
        assert sampled['pools'] == [], 'no worker pool may be opened for pyMC'
        assert sampled['pool'] is None, 'the sampler must not be given a pool'
        assert sampled['positions'].shape == (2, NUM_PARAMS), "this rank's own walkers"


def test_a_multi_rank_emcee_run_still_opens_the_pool(monkeypatch, tmp_path):
    """The fix must not change how the default backend runs under MPI."""
    sampled = _run_with(monkeypatch, _engine('emcee'), 0, 4, tmp_path=tmp_path)

    assert len(sampled['pools']) == 1, 'emcee still parallelises through a worker pool'
    assert sampled['pool'] is sampled['pools'][0], 'and the sampler is given it'
    assert sampled['positions'].shape == (8, NUM_PARAMS), 'the master drives the whole ensemble'


def test_a_single_rank_run_opens_no_pool_whichever_backend(monkeypatch, tmp_path):
    for library in ('emcee', 'pymc'):
        sampled = _run_with(monkeypatch, _engine(library), 0, 1, tmp_path=tmp_path)
        assert sampled['pools'] == [] and sampled['pool'] is None
        assert sampled['positions'].shape == (8, NUM_PARAMS), 'one rank runs every walker'


# ---------------------------------------------------------------------------
# the real thing: two ranks, no model, and a clock on it
# ---------------------------------------------------------------------------
#: Runs OpencorMCMC.run on an analytic posterior, so the MPI arrangement is exercised without a
#: CellML model. Before the fix this hangs at the barrier inside PyMCSampler.run_mcmc; the test
#: gives it a deadline so a hang is reported as a failure rather than by never finishing.
_TWO_RANK_RUN = '''
import sys
sys.path.insert(0, {src!r})
LIBRARY, NUM_WALKERS, NUM_STEPS = {library!r}, {num_walkers}, 6

import numpy as np
import libcuflynx.param_id.paramID as paramID
from libcuflynx.param_id.paramID import OpencorMCMC


class Norm:
    def normalise(self, vals):
        return np.asarray(vals, dtype=float)

    def unnormalise(self, vals):
        return np.asarray(vals, dtype=float)


class Engine(OpencorMCMC):
    def __init__(self):
        self.UQ_options = {{'library': LIBRARY, 'num_walkers': NUM_WALKERS,
                           'num_steps': NUM_STEPS,
                           'num_tune': 2, 'chain_save_every': 2, 'burn_in': 0.5}}
        self.num_params = 2
        self.param_norm_obj = Norm()
        self.best_param_vals = None
        self.param_id_info = {{'param_names_for_plotting': ['a', 'b'],
                              'param_mins': [-5.0, -5.0], 'param_maxs': [5.0, 5.0]}}
        self.output_dir = {output_dir!r}
        self.saved = None

    def get_lnlikelihood_lnprior_from_params(self, param_vals):
        return -0.5 * float(np.sum(np.asarray(param_vals) ** 2))

    def save_mcmc_statistics(self, flat_samples):
        self.saved = flat_samples.shape


engine = Engine()
paramID.mcmc_object = engine
engine.run()

from mpi4py import MPI
if MPI.COMM_WORLD.Get_rank() == 0:
    chain = np.load(engine.mcmc_chain_path())
    assert chain.shape == (NUM_STEPS, NUM_WALKERS, 2), chain.shape
    assert engine.saved is not None, 'rank 0 never reached the end of run()'
    print('OK', chain.shape)
'''


@pytest.mark.parametrize('library, num_walkers', [
    pytest.param('pymc', 2, marks=pytest.mark.skipif(not pymc_installed,
                                                     reason='needs the optional [uq] extra')),
    ('emcee', 4),
])
@pytest.mark.skipif(not sys.executable, reason='no interpreter to launch (OpenCOR pythonshell)')
@pytest.mark.skipif(shutil.which('mpiexec') is None, reason='no mpiexec')
def test_two_ranks_finish_instead_of_hanging(tmp_path, library, num_walkers):
    """The bug itself: two ranks and a run that has to end.

    ``pymc`` is the regression -- before the fix this hangs, and the deadline below is what turns
    a hang into a failure. ``emcee`` is here because nothing else covers it: the pool branch only
    runs with more than one rank, and every UQ test in CI runs on one, so the branch this change
    refactors would otherwise be exercised by nothing at all.

    Deliberately not run under the suite's own ranks -- a nested launch inherits the parent job's
    launcher variables and confuses the child. The environment is cleaned the way
    tests/test_mpi_utils.py cleans it.
    """
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
    code = textwrap.dedent(_TWO_RANK_RUN).format(src=src, output_dir=str(tmp_path),
                                                 library=library, num_walkers=num_walkers)
    env = {key: value for key, value in os.environ.items() if key not in LAUNCHER_ENV_VARS}

    try:
        proc = subprocess.run(['mpiexec', '-n', '2', sys.executable, '-c', code],
                              capture_output=True, text=True, timeout=600, env=env)
    except subprocess.TimeoutExpired:
        pytest.fail('a two-rank pyMC UQ run did not finish: the ranks are deadlocked, which is '
                    'the bug this test exists for')

    assert proc.returncode == 0, f'stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}'
    assert 'OK' in proc.stdout, proc.stdout
