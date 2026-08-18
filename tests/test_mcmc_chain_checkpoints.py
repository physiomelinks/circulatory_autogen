"""The MCMC chain has to be on disk *during* the run, not only after it (issue #417).

``run_mcmc`` is one blocking call, so the chain used to appear once, hours in. That makes a long
run unobservable -- there is no way to see whether the walkers are mixing or stuck until it is
too late to stop it -- and makes a cancelled or killed run a total loss, when a partial chain is
a perfectly good chain with fewer steps.

What is asserted here is what a reader polling the file actually needs: that it appears early,
that it grows, that it is always loadable, and that the finished chain is not changed by any of
it.
"""

import os
import threading

import numpy as np
import pytest

from param_id.paramID import sample_with_checkpoints, save_chain_atomically

pytestmark = pytest.mark.unit

NUM_WALKERS = 4
NUM_PARAMS = 2


class _StubSampler:
    """A sampler with emcee's two relevant surfaces: ``sample()`` and ``get_chain()``.

    emcee's ``sample`` is a generator that yields once per step and accumulates the chain as it
    goes, which is the property the checkpointing relies on.
    """

    def __init__(self, seed=0):
        self.steps = []
        self.run_mcmc_calls = 0
        self.sample_kwargs = None
        self._rng = np.random.default_rng(seed)

    def sample(self, initial_state, iterations, **kwargs):
        self.sample_kwargs = kwargs
        for _ in range(iterations):
            self.steps.append(self._rng.normal(size=(NUM_WALKERS, NUM_PARAMS)))
            yield 'state'

    def get_chain(self):
        return np.array(self.steps)

    def run_mcmc(self, initial_state, num_steps, **kwargs):
        self.run_mcmc_calls += 1
        for _ in self.sample(initial_state, num_steps, **kwargs):
            pass


class _RunMcmcOnlySampler:
    """A backend that cannot be stepped -- it only offers the blocking call."""

    def __init__(self):
        self.run_mcmc_calls = 0

    def get_chain(self):
        return np.zeros((1, NUM_WALKERS, NUM_PARAMS))

    def run_mcmc(self, initial_state, num_steps, **kwargs):
        self.run_mcmc_calls += 1


class _SelfCheckpointingSampler:
    """A backend that cannot be stepped but checkpoints itself -- pyMC's shape.

    ``pm.sample`` is one blocking call with no generator form, but it does take a per-draw
    callback, so the backend can write the partial chain from inside its own run_mcmc.
    """

    saves_own_checkpoints = True

    def __init__(self):
        self.run_mcmc_kwargs = None

    def get_chain(self):
        return np.zeros((1, NUM_WALKERS, NUM_PARAMS))

    def run_mcmc(self, initial_state, num_steps, save_chain=None, save_every=0, **kwargs):
        self.run_mcmc_kwargs = kwargs
        for step in range(1, num_steps + 1):
            if save_every and step % save_every == 0:
                save_chain(np.zeros((step, 1, NUM_PARAMS)))


def _saver(path, seen):
    """Record the shape of every chain written, and write it, so both can be asserted."""
    def save(samples):
        save_chain_atomically(path, samples)
        seen.append(np.load(path).shape)
    return save


def test_the_chain_appears_and_grows_while_sampling(tmp_path):
    """The point of the change: a reader polling this file sees the run progress."""
    path = str(tmp_path / 'mcmc_chain.npy')
    sampler = _StubSampler()
    shapes = []

    written = sample_with_checkpoints(sampler, 'x0', 20, _saver(path, shapes), save_every=5)

    # Steps 5, 10, 15 -- not 20, which the caller saves as the finished chain.
    assert written == 3
    assert [s[0] for s in shapes] == [5, 10, 15]
    assert all(s[1:] == (NUM_WALKERS, NUM_PARAMS) for s in shapes)


def test_each_checkpoint_is_the_chain_so_far_not_a_placeholder(tmp_path):
    """A file that exists but holds the wrong steps is worse than no file: it would plot."""
    path = str(tmp_path / 'mcmc_chain.npy')
    sampler = _StubSampler()

    sample_with_checkpoints(sampler, 'x0', 9, lambda s: save_chain_atomically(path, s),
                            save_every=3)

    on_disk = np.load(path)
    # The last checkpoint is step 6 (9 is the final step, left to the caller).
    np.testing.assert_allclose(on_disk, sampler.get_chain()[:6])


def test_the_finished_chain_is_exactly_what_it_was_before(tmp_path):
    """Checkpointing must not change the answer -- it writes what sampling already produced."""
    path = str(tmp_path / 'mcmc_chain.npy')

    checkpointed = _StubSampler(seed=7)
    sample_with_checkpoints(checkpointed, 'x0', 12, lambda s: save_chain_atomically(path, s),
                            save_every=4)

    straight_through = _StubSampler(seed=7)
    sample_with_checkpoints(straight_through, 'x0', 12, lambda s: None, save_every=0)

    np.testing.assert_allclose(checkpointed.get_chain(), straight_through.get_chain())


def test_checkpointing_off_falls_back_to_the_blocking_call(tmp_path):
    """0 is 'save only at the end' -- the behaviour before this change, still reachable."""
    sampler = _StubSampler()
    saves = []

    written = sample_with_checkpoints(sampler, 'x0', 10, saves.append, save_every=0)

    assert written == 0
    assert saves == []
    assert sampler.run_mcmc_calls == 1
    assert sampler.get_chain().shape[0] == 10


def test_a_backend_that_cannot_be_stepped_still_runs():
    """zeus goes through this path too, and a backend offering only run_mcmc must not raise."""
    sampler = _RunMcmcOnlySampler()
    saves = []

    written = sample_with_checkpoints(sampler, 'x0', 10, saves.append, save_every=5)

    assert written == 0
    assert sampler.run_mcmc_calls == 1


def test_a_backend_that_checkpoints_itself_is_given_the_hook(tmp_path):
    """The pyMC case (#417 follow-up). Before this it matched 'cannot be stepped' and took the
    blocking fallback, so its chain appeared only at the end however chain_save_every was set --
    the file a live progress view polls stayed absent for the whole run."""
    path = str(tmp_path / 'mcmc_chain.npy')
    sampler = _SelfCheckpointingSampler()
    shapes = []

    written = sample_with_checkpoints(sampler, 'x0', 20, _saver(path, shapes), save_every=5)

    assert written == 4, 'the count is the calls to the hook, whichever route made them'
    assert [s[0] for s in shapes] == [5, 10, 15, 20]
    assert np.load(path).shape == (20, 1, NUM_PARAMS)


def test_a_self_checkpointing_backend_still_honours_checkpointing_off():
    """save_every 0 means 'only at the end' for every backend, so the hook must not be handed
    over -- a backend given save_every=0 would write nothing anyway, but it must not be asked."""
    sampler = _SelfCheckpointingSampler()
    saves = []

    written = sample_with_checkpoints(sampler, 'x0', 10, saves.append, save_every=0)

    assert written == 0 and saves == []


def test_self_checkpointing_backends_get_the_sampler_kwargs_too():
    sampler = _SelfCheckpointingSampler()

    sample_with_checkpoints(sampler, 'x0', 4, lambda s: None, save_every=2,
                            progress=True, tune=True)

    assert sampler.run_mcmc_kwargs == {'progress': True, 'tune': True}


def test_sampler_kwargs_are_passed_through():
    """The MPI path asks for progress and tuning; routing through sample() must not drop them."""
    sampler = _StubSampler()

    sample_with_checkpoints(sampler, 'x0', 4, lambda s: None, save_every=2,
                            progress=True, tune=True)

    assert sampler.sample_kwargs == {'progress': True, 'tune': True}


def test_a_partial_write_is_never_visible(tmp_path):
    """Written beside the file and renamed, so a poller never loads a truncated array.

    Checked by overwriting a large chain with a small one while reading in a loop: an in-place
    write leaves a window where the file is half of each, and np.load raises on it.
    """
    path = str(tmp_path / 'mcmc_chain.npy')
    save_chain_atomically(path, np.ones((400, NUM_WALKERS, NUM_PARAMS)))

    for _ in range(25):
        save_chain_atomically(path, np.zeros((400, NUM_WALKERS, NUM_PARAMS)))
        loaded = np.load(path)          # raises if it ever catches a partial file
        assert loaded.shape == (400, NUM_WALKERS, NUM_PARAMS)

    # and nothing is left lying beside it
    assert os.listdir(tmp_path) == ['mcmc_chain.npy']


def test_a_concurrent_reader_never_sees_a_broken_chain(tmp_path):
    """The real arrangement: CA writes the chain while something else polls it, uncoordinated.

    A writer runs alongside a reader with no lock between them. np.save releases the GIL while
    it writes, so an in-place write genuinely loses this race -- the reader catches a truncated
    array and np.load raises, or the halves of two different chains and the values disagree.
    Both are asserted, because a file that loads but holds half of each would plot.

    A thread rather than a process on purpose: the suite runs under mpiexec, where forking is
    unwise and OpenCOR's embedded interpreter leaves sys.executable empty so spawn cannot start
    a child at all.
    """
    path = str(tmp_path / 'mcmc_chain.npy')
    save_chain_atomically(path, np.zeros((300, NUM_WALKERS, NUM_PARAMS)))
    stop = threading.Event()
    failures = []

    def write_until_stopped():
        step = 0
        while not stop.is_set():
            step += 1
            try:
                save_chain_atomically(path, np.full((300, NUM_WALKERS, NUM_PARAMS), float(step)))
            except Exception as exc:  # noqa: BLE001 - surfaced through `failures`
                failures.append(exc)
                return

    writer = threading.Thread(target=write_until_stopped)
    writer.start()
    try:
        for _ in range(80):
            samples = np.load(path)     # the assertion is that this does not raise
            assert samples.shape == (300, NUM_WALKERS, NUM_PARAMS)
            # every element comes from one write, never half of two
            assert len(np.unique(samples)) == 1
    finally:
        stop.set()
        writer.join(timeout=30)
    assert not failures, f'the writer failed: {failures[0]!r}'
