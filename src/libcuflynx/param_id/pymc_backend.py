"""pyMC as an alternative MCMC backend, behind emcee's sampler interface (issue #195).

emcee's affine-invariant ensemble sampler is a good default, but it is one algorithm. pyMC brings
Metropolis, NUTS and sequential Monte Carlo, and SMC in particular can cross the low-probability
regions between separated modes that an ensemble sampler gets stuck on.

``PyMCSampler`` exposes exactly the surface ``MCMC.run`` already uses -- ``run_mcmc`` and
``get_chain``, with a ``(steps, walkers, params)`` chain -- so selecting a library changes one
line of construction and nothing in the sampling loop or in anything downstream (the diagnostics,
the corner plots and the saved ``mcmc_chain.npy`` all keep working unchanged).

pymc and pytensor are imported lazily, inside the sampler. They are not CA dependencies:
``paramID.py`` is imported by every calibration run, so importing them at module level (as the
original patch did) would break every user who has not installed them, to no benefit for anyone
not doing UQ. Install with ``pip install "libcuflynx[uq]"``.

The one place the two backends are not interchangeable is *how* the chain reaches disk while it
is being sampled (#417). emcee's ``sample()`` is a generator, so the caller checkpoints between
steps; ``pm.sample`` is a single blocking call with no generator equivalent, and a pyMC run
therefore wrote nothing until it finished -- a live progress plot of it stayed empty for the
whole run. ``pm.sample`` does take a per-draw ``callback``, so this module does its own
checkpointing through that and declares ``saves_own_checkpoints`` so the shared loop hands it
the same ``save_chain`` hook rather than falling back to the blocking call.
"""
import sys

import numpy as np

try:
    from mpi4py import MPI
except ImportError:                                            # pragma: no cover - MPI is a dep
    MPI = None


_INSTALL_HINT = (
    "The pyMC UQ backend needs pymc and pytensor, which are not installed. "
    "Install them with:  pip install \"libcuflynx[uq]\"  (about 65 MB; from a checkout, "
    "pip install -e \".[uq]\"). Or set UQ_options: library: emcee to use the built-in "
    "ensemble sampler instead."
)


def _import_pymc():
    """Import pymc/pytensor, or raise with the install command rather than a bare ImportError."""
    try:
        import pymc as pm
        import pytensor.tensor as pt
        from pytensor.compile.ops import as_op
    except ImportError as exc:
        raise ImportError(f"{_INSTALL_HINT} (underlying error: {exc})") from exc
    return pm, pt, as_op


def _progressbar_wanted(progress, rank):
    """Whether to let pyMC draw its progress bar.

    Only rank 0 has anything to draw, and only when the caller asked for progress at all -- but
    also only onto a terminal. pyMC's bar is a rich table that repaints, so redirected into a
    log file it becomes thousands of lines of ANSI escapes wrapped around a redrawn table
    (measured: ~8 kB for a 1000-draw run, growing with the run), interleaved with whatever else
    is writing to that log. That is the run log a user is left reading when a run goes wrong,
    and the bar is worth nothing in it: nobody watches a progress bar in a file. A terminal
    still gets it.
    """
    return bool(progress) and rank == 0 and bool(getattr(sys.stdout, 'isatty', lambda: False)())


class _LiveChainWriter:
    """pyMC's per-draw callback, writing the draws so far the way #417 writes emcee's.

    pyMC hands a callback ``(trace, draw)`` after every recorded draw, where ``trace`` is the
    per-chain backend it is filling and ``draw`` carries the chain index and whether the draw is
    a tuning one. That is enough to write the same growing ``mcmc_chain.npy`` an emcee run
    writes, without pretending ``pm.sample`` can be stepped.

    **The live file keeps each chain in its own column, NaN where it has not got there yet.**
    With ``cores=1`` -- which is what CA asks for, because the parallelism is MPI ranks, not
    pyMC processes -- chains are sampled one after another, not advanced together, so there is
    no full ``(steps, chains, params)`` rectangle until the last one finishes: chain 2 has one
    draw while chain 1 has five thousand.

    This first shipped concatenating them into a single walker, on the reasoning that a
    ragged set of chains has no honest rectangle and the concatenation at least invented
    nothing. That was wrong in a way only a plot shows. The joined trace steps discontinuously
    where one chain ends and the next begins, and every statistic computed along it -- the
    autocorrelation, the running mean -- is taken *across* those joins, so a live view of a
    healthy pyMC run looked like one badly-mixing chain. emcee advances all its walkers
    together and never has this shape, so only pyMC looked broken.

    Padding with NaN is the honest rectangle: each column is one real chain, and a draw that
    has not happened is absent rather than fabricated. Consumers must use NaN-aware reductions
    (``np.nanmean`` and friends) or drop the NaNs -- which is the right requirement, because
    the alternative is a number no sampler produced. The finished chain, written by the caller
    once sampling returns, is the real dense ``(steps, chains, params)`` from every rank and
    carries no NaN at all.

    Tuning draws are dropped, because pyMC drops them: they are recorded in the same trace but
    do not appear in the posterior, and a live view that included them would disagree with the
    chain that lands at the end.

    Nothing in here may raise. A callback that throws takes the whole ``pm.sample`` call down
    with it, and losing hours of sampling because a progress nicety could not write a file is a
    far worse failure than not having the file. A first failure is reported and turns the
    checkpointing off for the rest of the run.
    """

    def __init__(self, save_chain, save_every, param_names, num_tune):
        self.save_chain = save_chain
        self.save_every = save_every
        self.param_names = list(param_names)
        self.num_tune = int(num_tune)
        # chain index -> pyMC's trace for it, in the order sampling reached them; the traces are
        # the same objects for the life of the run, so a finished chain stays readable here.
        self.traces = {}
        self.num_draws = 0
        self.disabled = False

    def __call__(self, trace, draw):
        self.traces[draw.chain] = trace
        if self.disabled or draw.tuning:
            return
        self.num_draws += 1
        if self.num_draws % self.save_every:
            return
        try:
            samples = self.chain_so_far()
            if samples is not None:
                self.save_chain(samples)
        except Exception as exc:                       # noqa: BLE001 - see the class docstring
            self.disabled = True
            print(f'WARNING: could not write the partial MCMC chain ({exc}); sampling continues '
                  'and the chain will be saved at the end of the run.')

    def chain_so_far(self):
        """The post-tuning draws so far as ``(draws, chains, params)``, or None before any.

        One column per chain pyMC has started, in chain order, padded with NaN where a chain
        has not reached that draw yet. A chain that has not started at all contributes no
        column, so the array widens as sampling moves on to the next chain.

        NaN rather than a fabricated number: "not sampled yet" is not a value, and anything
        substituted for it -- zero, the last draw, the mean -- is a datum no sampler produced
        and would be plotted and averaged as though it were real.
        """
        blocks = {}
        for chain_idx, trace in self.traces.items():
            recorded = len(trace)
            if recorded <= self.num_tune:
                continue
            # A trace still being filled is preallocated to its full length, so it has to be cut
            # to what has actually been recorded -- the tail is zeros, not draws.
            blocks[chain_idx] = np.stack(
                [np.asarray(trace.get_values(name))[self.num_tune:recorded]
                 for name in self.param_names], axis=-1)
        if not blocks:
            return None
        # Column index *is* the chain index, so a trace keeps its identity as the run goes on
        # rather than shifting left when an earlier chain is skipped.
        num_chains = max(blocks) + 1
        longest = max(len(block) for block in blocks.values())
        samples = np.full((longest, num_chains, len(self.param_names)), np.nan)
        for chain_idx, block in blocks.items():
            samples[:len(block), chain_idx, :] = block
        return samples


class PyMCSampler:
    """A pyMC sampler wearing emcee's interface.

    Args:
        num_walkers: Total chains to run across all MPI ranks.
        num_params: Number of calibrated parameters.
        log_posterior_fn: ``f(param_vals) -> float``, the **full** log posterior
            (log likelihood + log prior). See ``_build_model`` for why the whole thing goes in
            rather than the likelihood alone.
        param_id_info: The engine's parameter info (names, mins, maxs, unbounded flags).
        num_tune: Tuning (burn-in) draws discarded before sampling.
        method: ``'mcmc'`` for Metropolis sampling, or ``'smc'`` for sequential Monte Carlo.
    """

    #: Read by ``paramID.sample_with_checkpoints``. ``pm.sample`` is one blocking call with no
    #: generator form, so this backend cannot be driven a step at a time -- but it does not have
    #: to fall back to writing the chain only at the end either: it checkpoints itself from
    #: pyMC's per-draw callback, given the same ``save_chain`` hook.
    saves_own_checkpoints = True

    def __init__(self, num_walkers, num_params, log_posterior_fn, param_id_info=None,
                 num_tune=1000, method='mcmc'):
        if method not in ('mcmc', 'smc'):
            raise ValueError(f"unknown pyMC method {method!r}; expected 'mcmc' or 'smc'")
        self.num_walkers = num_walkers
        self.num_params = num_params
        self.log_posterior_fn = log_posterior_fn
        self.param_id_info = param_id_info or {}
        self.num_tune = num_tune
        self.method = method
        self.chain = None
        # Fail here rather than after a model has already been built and a pool opened.
        self.pm, self.pt, self._as_op = _import_pymc()

    # ------------------------------------------------------------------
    # model
    # ------------------------------------------------------------------
    def _param_names(self):
        names = self.param_id_info.get('param_names_for_plotting')
        if names is None or len(names) != self.num_params:
            return [f'param_{idx}' for idx in range(self.num_params)]
        return list(names)

    def _build_model(self):
        """A pyMC model whose only density term is CA's own log posterior.

        The prior is deliberately *not* re-declared as pyMC distributions. CA's
        ``get_lnprior_from_params`` already implements the full params_for_id prior vocabulary --
        uniform / exponential / normal, the user's ``prior_mean``, ``prior_std``,
        ``prior_origin`` and ``prior_scale`` hyper-parameters, and the ``unbounded`` flag that
        suppresses the range check. Restating that in pyMC terms would mean maintaining a second
        implementation of it, and the original patch shows what that costs: it hardcoded the old
        defaults (lambda=1, sigma=(max-min)/6, mu=(max+min)/2) and so silently ignored every
        prior hyper-parameter a user set.

        Worse, it declared those pyMC priors *and* put the full log posterior in the Potential,
        so the prior was counted twice and the sampled posterior was not the one CA defines.

        Here each parameter is a bare support declaration -- Uniform over its box, or Flat when
        the parameter is unbounded -- which contributes only a constant inside that support and
        so leaves the posterior shape entirely to the Potential.
        """
        pm = self.pm
        names = self._param_names()
        mins = self.param_id_info.get('param_mins')
        maxs = self.param_id_info.get('param_maxs')
        unbounded = self.param_id_info.get('param_unbounded')

        log_posterior_op = self._as_op(
            itypes=[self.pt.dvector], otypes=[self.pt.dscalar])(self._evaluate_log_posterior)

        with pm.Model() as model:
            variables = []
            for idx, name in enumerate(names):
                is_unbounded = bool(unbounded[idx]) if unbounded is not None and \
                    idx < len(unbounded) else False
                if is_unbounded or mins is None or maxs is None:
                    variables.append(pm.Flat(name))
                else:
                    variables.append(
                        pm.Uniform(name, lower=float(mins[idx]), upper=float(maxs[idx])))
            pm.Potential('log_posterior', log_posterior_op(pm.math.stack(variables)))
        return model

    def _evaluate_log_posterior(self, param_vals):
        """The pytensor-facing wrapper: always a 0-d float64, whatever the cost returns."""
        value = np.asarray(self.log_posterior_fn(np.asarray(param_vals, dtype=float)),
                           dtype=float)
        if value.shape != ():
            value = np.sum(value)
        return np.array(float(value))

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    @staticmethod
    def chains_for_rank(num_walkers, num_procs):
        """How many chains this rank runs, given the total walker budget.

        Never zero: with more ranks than walkers, ``num_walkers // num_procs`` is 0 and pyMC is
        asked for no chains at all, which either raises or silently returns an empty trace.
        """
        if num_procs is None or num_procs <= 1:
            return max(1, int(num_walkers))
        return max(1, int(num_walkers) // int(num_procs))

    def run_mcmc(self, initial_state, num_steps, progress=False, save_chain=None, save_every=0,
                 **kwargs):
        """Sample, and return the chain as ``(steps, walkers, params)``.

        ``initial_state`` is emcee's ``(walkers, params)`` starting positions; it seeds the
        per-chain initial values for ``method='mcmc'`` and is unused for SMC, which draws its
        own initial population from the prior.

        ``save_chain``/``save_every`` are the checkpoint hook from
        ``paramID.sample_with_checkpoints`` (issue #417): call ``save_chain(samples)`` every
        ``save_every`` draws so the run can be watched and a cancelled one is not lost. Routed
        into ``pm.sample``'s per-draw callback -- see ``_LiveChainWriter`` for what the partial
        chain contains and why.
        """
        comm = MPI.COMM_WORLD if MPI is not None else None
        rank = comm.Get_rank() if comm is not None else 0
        num_procs = comm.Get_size() if comm is not None else 1
        num_chains = self.chains_for_rank(self.num_walkers, num_procs)
        checkpointer = self._make_checkpointer(save_chain, save_every, rank)

        model = self._build_model()
        with model:
            if self.method == 'smc':
                trace = self.pm.sample_smc(draws=num_steps, chains=num_chains, cores=1,
                                           progressbar=_progressbar_wanted(progress, rank))
            else:
                trace = self.pm.sample(
                    draws=num_steps, tune=self.num_tune, chains=num_chains, cores=1,
                    step=self.pm.Metropolis(), progressbar=_progressbar_wanted(progress, rank),
                    callback=checkpointer,
                    initvals=self._initial_values(initial_state, num_chains))

        local_chain = self.trace_to_emcee_chain(trace, self._param_names())

        if comm is None:
            self.chain = local_chain
            return self.chain

        comm.Barrier()
        gathered = comm.gather(local_chain, root=0)
        if rank != 0:
            return None
        # Each rank sampled its own chains, so they concatenate along the walker axis.
        self.chain = np.concatenate([c for c in gathered if c is not None], axis=1)
        return self.chain

    def _make_checkpointer(self, save_chain, save_every, rank):
        """The per-draw callback that writes the partial chain, or None if nothing should.

        Nothing should when checkpointing is off, when this is not rank 0, or under SMC:

        * **Not rank 0.** Every rank samples its own chains into the same output directory, and
          the chain file is one path. Ranks all writing it would each overwrite the others with
          a chain that is only their own -- so the file would flicker between ranks rather than
          grow. Only rank 0 writes, which means the live file shows rank 0's chains; the other
          ranks' are gathered into the finished chain at the end, as they always were.
        * **SMC.** ``pm.sample_smc`` takes no callback -- it advances a whole population per
          stage rather than drawing one sample at a time, so there is no per-draw hook to hang
          this on. Rather than leave that looking broken, say it: the chain arrives at the end.
        """
        if save_chain is None or save_every <= 0 or rank != 0:
            return None
        if self.method == 'smc':
            print("NOTE: pyMC's sequential Monte Carlo (pymc_method 'smc') has no per-draw hook, "
                  'so the chain is written once, when sampling finishes. UQ_options '
                  "chain_save_every takes effect for pymc_method 'mcmc' only.")
            return None
        return _LiveChainWriter(save_chain, save_every, self._param_names(), self.num_tune)

    def _initial_values(self, initial_state, num_chains):
        """emcee's (walkers, params) start positions as pyMC's per-chain initval dicts."""
        if initial_state is None:
            return None
        initial_state = np.asarray(initial_state, dtype=float)
        names = self._param_names()
        return [
            {name: float(initial_state[chain_idx % initial_state.shape[0], param_idx])
             for param_idx, name in enumerate(names)}
            for chain_idx in range(num_chains)
        ]

    @staticmethod
    def trace_to_emcee_chain(trace, param_names):
        """Convert an arviz InferenceData posterior to emcee's ``(steps, walkers, params)``.

        pyMC stores ``(chain, draw)`` per variable; emcee -- and therefore every CA consumer of
        ``mcmc_chain.npy``, from the corner plots to the R-hat diagnostic -- expects draws first
        and walkers second.

        Raises rather than returning None on a trace that does not contain the parameters: a
        silent None becomes an unreadable failure much further downstream, after the sampling
        has already been paid for.
        """
        posterior = getattr(trace, 'posterior', None)
        if posterior is None:
            raise ValueError('pyMC returned a trace with no posterior group')

        missing = [name for name in param_names if name not in posterior]
        if missing:
            raise ValueError(
                f'pyMC trace is missing sampled parameters {missing}; '
                f'it contains {list(posterior)}')

        # (chain, draw) per parameter -> (chain, draw, param) -> (draw, chain, param)
        stacked = np.stack([np.asarray(posterior[name].values) for name in param_names], axis=-1)
        return np.swapaxes(stacked, 0, 1)

    def get_chain(self):
        """The sampled chain, ``(steps, walkers, params)`` -- emcee's accessor."""
        return self.chain
