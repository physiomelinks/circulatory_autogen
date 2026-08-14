"""The pyMC backend writes its chain while it samples, not only at the end (issue #417).

#418 gave emcee a growing ``mcmc_chain.npy`` by stepping its generator. pyMC has no generator:
``pm.sample`` is one blocking call, so it matched "a backend that cannot be stepped" and took
the fallback -- its chain appeared once, at the end, whatever ``chain_save_every`` said, and a
live progress view of a pyMC run stayed empty for the whole run. It does take a per-draw
callback, which is what the backend now checkpoints from.

The part worth testing hardest is the one that looks right and is not: **draw ordering across
chains**. pyMC with ``cores=1`` samples chain 0 to completion, then chain 1, and reports draws
through a callback that carries the chain index and a tuning flag. A partial chain that mixed
those up, or that included the tuning draws pyMC discards, would still load and still plot --
it would just be showing something other than the posterior being sampled.

The fake-trace tests run everywhere; the ones that actually sample are guarded on the optional
[uq] extra and are the reason this file is in the UQ CI job.
"""
import os
import sys
from collections import namedtuple

import numpy as np
import pytest

from param_id.paramID import (
    drop_unsampled_draws,
    sample_with_checkpoints,
    save_chain_atomically,
)
from param_id.pymc_backend import PyMCSampler, _LiveChainWriter, _progressbar_wanted

pymc_installed = True
try:
    import pymc  # noqa: F401
except ImportError:
    pymc_installed = False

PARAM_NAMES = ['a', 'b']


# ---------------------------------------------------------------------------
# a stand-in for pyMC's per-chain trace and its callback
# ---------------------------------------------------------------------------
#: pyMC's ``Draw``, in field order (pymc.sampling.parallel.Draw).
Draw = namedtuple('Draw', 'chain is_last draw_idx tuning stats point')


class _FakeTrace:
    """pyMC's NDArray chain backend, in the two respects the writer touches.

    The preallocation matters and is the reason this is not just a list: pyMC sizes the arrays
    to the whole run up front and fills them as it goes, so ``get_values`` on a chain that is
    still sampling returns real draws followed by zeros. A writer that did not cut it to
    ``len(trace)`` would publish a chain padded with zeros -- values no sampler produced.
    """

    def __init__(self, total_draws):
        self.samples = {name: np.zeros(total_draws) for name in PARAM_NAMES}
        self.recorded = 0

    def record(self, values):
        for name, value in zip(PARAM_NAMES, values):
            self.samples[name][self.recorded] = value
        self.recorded += 1

    def __len__(self):
        return self.recorded

    def get_values(self, varname):
        return self.samples[varname]


def _replay_pymc(writer, draws_per_chain, num_tune, num_chains, start=0.0):
    """Drive ``writer`` with the callback sequence pyMC produces at ``cores=1``.

    Chain by chain, tuning draws first, one callback per recorded draw. Returns the post-tuning
    draws per chain, ``[chain][draw][param]`` -- the shape the live file is expected to hold,
    one column per chain, before any NaN padding is applied.
    """
    total = num_tune + draws_per_chain
    per_chain = []
    value = start
    for chain in range(num_chains):
        trace = _FakeTrace(total)
        kept = []
        for idx in range(total):
            value += 1.0
            point = [value, -value]
            trace.record(point)
            if idx >= num_tune:
                kept.append(point)
            writer(trace=trace, draw=Draw(chain, False, idx, idx < num_tune, {}, {}))
        per_chain.append(np.asarray(kept, dtype=float).reshape(-1, len(PARAM_NAMES)))
    return per_chain


def _padded(per_chain):
    """``per_chain`` as the ``(draws, chains, params)`` rectangle, NaN where a chain is short."""
    longest = max((len(c) for c in per_chain), default=0)
    out = np.full((longest, len(per_chain), len(PARAM_NAMES)), np.nan)
    for idx, block in enumerate(per_chain):
        out[:len(block), idx, :] = block
    return out


# ---------------------------------------------------------------------------
# what lands in the partial chain
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_chain_is_written_every_save_every_draws_and_grows():
    """The point of the change: a reader polling the file sees the run progress."""
    written = []
    writer = _LiveChainWriter(written.append, save_every=5, param_names=PARAM_NAMES, num_tune=2)

    _replay_pymc(writer, draws_per_chain=20, num_tune=2, num_chains=1)

    assert [chunk.shape for chunk in written] == [(5, 1, 2), (10, 1, 2), (15, 1, 2), (20, 1, 2)]


@pytest.mark.unit
def test_each_chain_keeps_its_own_column():
    """The fix. Chains sampled one after another must not be concatenated into one trace.

    Joined end to end, the trace steps discontinuously where one chain ends and the next
    begins, and the autocorrelation and running mean are then computed *across* that join --
    so a healthy pyMC run was drawn as one badly-mixing chain.
    """
    written = []
    writer = _LiveChainWriter(written.append, save_every=4, param_names=PARAM_NAMES, num_tune=3)

    per_chain = _replay_pymc(writer, draws_per_chain=8, num_tune=3, num_chains=2)

    final = written[-1]
    assert final.shape == (8, 2, 2), 'two chains of eight draws, side by side'
    np.testing.assert_allclose(final, _padded(per_chain))
    # column index is the chain index: chain 1's draws are not in chain 0's column
    np.testing.assert_allclose(final[:, 0, :], per_chain[0])
    np.testing.assert_allclose(final[:, 1, :], per_chain[1])


@pytest.mark.unit
def test_a_chain_that_has_not_caught_up_is_padded_with_nan():
    """Mid-run, chain 1 has draws chain 2 does not. The gap is NaN -- not zero, not the last
    value, not the mean -- because it is a draw no sampler has produced yet, and anything put
    there would be plotted and averaged as though it had been."""
    written = []
    writer = _LiveChainWriter(written.append, save_every=3, param_names=PARAM_NAMES, num_tune=0)

    # 6 draws for chain 0, then only 3 for chain 1 -> the writer is called mid-chain-1
    _replay_pymc(writer, draws_per_chain=6, num_tune=0, num_chains=1)
    trace = _FakeTrace(6)
    for idx in range(3):
        trace.record([100.0 + idx, -(100.0 + idx)])
        writer(trace=trace, draw=Draw(1, False, idx, False, {}, {}))

    partial = written[-1]
    assert partial.shape == (6, 2, 2), 'full height, one column per started chain'
    assert not np.isnan(partial[:, 0, :]).any(), 'the finished chain has no gaps'
    assert not np.isnan(partial[:3, 1, :]).any(), "chain 1's real draws are present"
    assert np.isnan(partial[3:, 1, :]).all(), 'and the draws it has not reached are NaN'
    # a NaN-aware reduction still gets the right answer for the short chain
    np.testing.assert_allclose(np.nanmean(partial[:, 1, 0]), 101.0)


@pytest.mark.unit
def test_a_cancelled_runs_partial_chain_is_cut_to_its_complete_draws():
    """A killed run leaves the partial file where the finished one would be, so whatever reads
    it must not turn a NaN into a NaN mean for the whole parameter."""
    partial = np.arange(24, dtype=float).reshape(4, 2, 3)
    partial[2:, 1, :] = np.nan            # chain 1 stopped two draws in

    kept = drop_unsampled_draws(partial)

    assert kept.shape == (2, 2, 3), 'cut to the draws both chains reached'
    assert not np.isnan(kept).any()
    np.testing.assert_allclose(kept, partial[:2])


@pytest.mark.unit
def test_a_finished_chain_is_returned_untouched():
    """The common case: nothing to drop, and no copy of a chain that may be very large."""
    dense = np.arange(24, dtype=float).reshape(4, 2, 3)
    assert drop_unsampled_draws(dense) is dense


@pytest.mark.unit
def test_a_chain_with_no_complete_draw_comes_back_empty_rather_than_nan():
    """Cancelled before the second chain produced anything. An empty chain reads as 'no
    samples'; one full of NaN reads as samples whose every statistic is NaN."""
    nothing = np.full((3, 2, 2), np.nan)
    nothing[:, 0, :] = 1.0
    assert drop_unsampled_draws(nothing).shape[0] == 0


@pytest.mark.unit
def test_the_finished_chain_carries_no_nan():
    """The padding is a property of a *partial* chain only. Once every chain has run its full
    length the rectangle is dense, and a consumer that forgot about NaN still gets a number."""
    written = []
    writer = _LiveChainWriter(written.append, save_every=5, param_names=PARAM_NAMES, num_tune=1)

    _replay_pymc(writer, draws_per_chain=5, num_tune=1, num_chains=3)

    assert written[-1].shape == (5, 3, 2)
    assert not np.isnan(written[-1]).any()


@pytest.mark.unit
def test_tuning_draws_are_left_out_because_pymc_leaves_them_out():
    """They are recorded in the same trace but are not in the posterior. A live view holding
    them would disagree with the chain that lands at the end -- and the tuning draws are the
    least converged ones there are, so it would disagree in the direction that looks worst."""
    written = []
    writer = _LiveChainWriter(written.append, save_every=2, param_names=PARAM_NAMES, num_tune=6)

    per_chain = _replay_pymc(writer, draws_per_chain=4, num_tune=6, num_chains=1)

    assert written[-1].shape[0] == 4, 'four post-tuning draws, not ten recorded ones'
    np.testing.assert_allclose(written[-1], _padded(per_chain))


@pytest.mark.unit
def test_the_preallocated_tail_of_a_running_trace_is_never_published():
    """pyMC sizes a chain's arrays to the whole run up front. Publishing the array as-is would
    pad the chain with zeros, which plot as a walker that has collapsed onto zero."""
    written = []
    writer = _LiveChainWriter(written.append, save_every=3, param_names=PARAM_NAMES, num_tune=1)

    _replay_pymc(writer, draws_per_chain=9, num_tune=1, num_chains=1)

    assert not np.any(written[0] == 0.0), 'the unfilled tail leaked into the chain'
    assert written[0].shape[0] == 3


@pytest.mark.unit
def test_nothing_is_written_before_there_is_a_post_tuning_draw():
    """A file that exists but holds no draws is worse than no file: it would plot."""
    written = []
    writer = _LiveChainWriter(written.append, save_every=1, param_names=PARAM_NAMES, num_tune=4)

    _replay_pymc(writer, draws_per_chain=0, num_tune=4, num_chains=2)

    assert written == []


@pytest.mark.unit
def test_a_checkpoint_that_cannot_be_written_does_not_kill_the_run(capsys):
    """A callback that raises takes pm.sample down with it. Losing hours of sampling because a
    progress nicety could not write a file is far worse than not having the file, so the first
    failure is reported and turns checkpointing off for the rest of the run."""
    calls = []

    def exploding_save(samples):
        calls.append(samples)
        raise OSError('no space left on device')

    writer = _LiveChainWriter(exploding_save, save_every=2, param_names=PARAM_NAMES, num_tune=0)

    _replay_pymc(writer, draws_per_chain=10, num_tune=0, num_chains=1)   # must not raise

    assert len(calls) == 1, 'it should stop trying after the first failure'
    output = capsys.readouterr().out
    assert 'no space left on device' in output and 'sampling continues' in output


# ---------------------------------------------------------------------------
# when checkpointing applies at all
# ---------------------------------------------------------------------------
def _sampler(method='mcmc', num_tune=5):
    """A PyMCSampler without pymc installed -- these paths are decided before any of it runs."""
    sampler = PyMCSampler.__new__(PyMCSampler)
    sampler.method = method
    sampler.num_tune = num_tune
    sampler.num_params = len(PARAM_NAMES)
    sampler.param_id_info = {'param_names_for_plotting': PARAM_NAMES}
    return sampler


@pytest.mark.unit
def test_only_rank_zero_writes_the_partial_chain():
    """Every rank samples its own chains into the same output directory, and the chain file is
    one path. Ranks all writing it would each overwrite the others with a chain that is only
    their own, so the file would flicker between ranks instead of growing."""
    assert _sampler()._make_checkpointer(lambda s: None, 10, rank=0) is not None
    assert _sampler()._make_checkpointer(lambda s: None, 10, rank=1) is None
    assert _sampler()._make_checkpointer(lambda s: None, 10, rank=7) is None


@pytest.mark.unit
def test_checkpointing_off_means_no_callback():
    assert _sampler()._make_checkpointer(lambda s: None, 0, rank=0) is None
    assert _sampler()._make_checkpointer(None, 50, rank=0) is None


@pytest.mark.unit
def test_smc_says_it_saves_only_at_the_end_rather_than_looking_broken(capsys):
    """sample_smc advances a whole population per stage and takes no callback, so there is no
    per-draw hook to write from. Silence there would look identical to the bug this fixes."""
    assert _sampler(method='smc')._make_checkpointer(lambda s: None, 10, rank=0) is None

    output = capsys.readouterr().out
    assert 'chain_save_every' in output and 'smc' in output


@pytest.mark.unit
def test_the_backend_declares_that_it_checkpoints_itself():
    """The flag sample_with_checkpoints dispatches on. Without it the backend matches 'cannot be
    stepped' and silently goes back to writing the chain once, at the end."""
    assert PyMCSampler.saves_own_checkpoints is True


@pytest.mark.unit
def test_the_progress_bar_is_kept_out_of_a_redirected_log():
    """pyMC's bar is a rich table that repaints; in a log file that is thousands of lines of
    ANSI escapes around a redrawn table, interleaved with the run's own output."""
    assert _progressbar_wanted(True, rank=0) == bool(
        getattr(sys.stdout, 'isatty', lambda: False)())
    assert _progressbar_wanted(False, rank=0) is False
    assert _progressbar_wanted(True, rank=3) is False


# ---------------------------------------------------------------------------
# the real thing
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.skipif(not pymc_installed, reason='needs the optional [uq] extra')
def test_a_real_pymc_run_writes_a_growing_chain_that_matches_what_it_returns(tmp_path):
    """End to end against pyMC itself: the fakes above encode what pyMC's callback does, and
    this is what proves that encoding right -- the draws pooled from the callback are exactly
    the chains pm.sample went on to return, in order.
    """
    num_steps, num_chains, num_tune = 6, 2, 3
    path = str(tmp_path / 'mcmc_chain.npy')
    shapes = []

    def save(samples):
        save_chain_atomically(path, samples)
        shapes.append(np.load(path).shape)      # loadable at every checkpoint, or this raises

    sampler = PyMCSampler(
        num_walkers=num_chains, num_params=2,
        log_posterior_fn=lambda params: -0.5 * float(np.sum(np.asarray(params) ** 2)),
        param_id_info={'param_names_for_plotting': PARAM_NAMES,
                       'param_mins': [-5.0, -5.0], 'param_maxs': [5.0, 5.0]},
        num_tune=num_tune)

    initial_state = np.array([[0.1, -0.1], [0.2, -0.2]])
    checkpoints = sample_with_checkpoints(sampler, initial_state, num_steps, save, save_every=2)
    chain = sampler.get_chain()

    # the finished chain is the shape everything downstream expects, and is not disturbed by any
    # of the checkpointing
    assert chain.shape == (num_steps, num_chains, 2)

    # a checkpoint every 2 post-tuning draws over 2 chains x 6 draws, and never the tuning ones
    assert checkpoints == 6
    # chain 0 grows to its full 6 draws, then chain 1 appears as a second column and grows
    assert shapes == [(2, 1, 2), (4, 1, 2), (6, 1, 2), (6, 2, 2), (6, 2, 2), (6, 2, 2)]

    # ...and the last partial chain is exactly the chains pyMC went on to return, per column
    np.testing.assert_allclose(np.load(path), chain)


@pytest.mark.unit
@pytest.mark.skipif(not pymc_installed, reason='needs the optional [uq] extra')
def test_a_real_pymc_run_with_checkpointing_off_writes_nothing_until_the_end(tmp_path):
    """0 still means "only at the end" -- the behaviour before this change, still reachable for
    a very wide chain on a slow shared filesystem."""
    path = str(tmp_path / 'mcmc_chain.npy')

    sampler = PyMCSampler(
        num_walkers=1, num_params=2,
        log_posterior_fn=lambda params: -0.5 * float(np.sum(np.asarray(params) ** 2)),
        param_id_info={'param_names_for_plotting': PARAM_NAMES,
                       'param_mins': [-5.0, -5.0], 'param_maxs': [5.0, 5.0]},
        num_tune=2)

    checkpoints = sample_with_checkpoints(
        sampler, np.array([[0.1, -0.1]]), 5,
        lambda samples: save_chain_atomically(path, samples), save_every=0)

    assert checkpoints == 0
    assert not os.path.exists(path)
    assert sampler.get_chain().shape == (5, 1, 2)
