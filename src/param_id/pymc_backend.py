"""pyMC as an alternative MCMC backend, behind emcee's sampler interface (issue #195).

emcee's affine-invariant ensemble sampler is a good default, but it is one algorithm. pyMC brings
Metropolis, NUTS and sequential Monte Carlo, and SMC in particular can cross the low-probability
regions between separated modes that an ensemble sampler gets stuck on.

``PyMCSampler`` exposes exactly the surface ``OpencorMCMC.run`` already uses -- ``run_mcmc`` and
``get_chain``, with a ``(steps, walkers, params)`` chain -- so selecting a library changes one
line of construction and nothing in the sampling loop or in anything downstream (the diagnostics,
the corner plots and the saved ``mcmc_chain.npy`` all keep working unchanged).

pymc and pytensor are imported lazily, inside the sampler. They are not CA dependencies:
``paramID.py`` is imported by every calibration run, so importing them at module level (as the
original patch did) would break every user who has not installed them, to no benefit for anyone
not doing UQ. Install with ``pip install -e ".[uq]"``.
"""
import numpy as np

try:
    from mpi4py import MPI
except ImportError:                                            # pragma: no cover - MPI is a dep
    MPI = None


_INSTALL_HINT = (
    "The pyMC UQ backend needs pymc and pytensor, which are not installed. "
    "Install them with:  pip install -e \".[uq]\"  (or set UQ_options: library: emcee to use "
    "the built-in ensemble sampler instead)."
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

    def run_mcmc(self, initial_state, num_steps, progress=False, **kwargs):
        """Sample, and return the chain as ``(steps, walkers, params)``.

        ``initial_state`` is emcee's ``(walkers, params)`` starting positions; it seeds the
        per-chain initial values for ``method='mcmc'`` and is unused for SMC, which draws its
        own initial population from the prior.
        """
        comm = MPI.COMM_WORLD if MPI is not None else None
        rank = comm.Get_rank() if comm is not None else 0
        num_procs = comm.Get_size() if comm is not None else 1
        num_chains = self.chains_for_rank(self.num_walkers, num_procs)

        model = self._build_model()
        with model:
            if self.method == 'smc':
                trace = self.pm.sample_smc(draws=num_steps, chains=num_chains, cores=1,
                                           progressbar=(progress and rank == 0))
            else:
                trace = self.pm.sample(
                    draws=num_steps, tune=self.num_tune, chains=num_chains, cores=1,
                    step=self.pm.Metropolis(), progressbar=(progress and rank == 0),
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
