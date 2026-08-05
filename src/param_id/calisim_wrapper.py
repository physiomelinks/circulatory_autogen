"""calisim as a calibration backend for circulatory_autogen.

calisim (https://github.com/Plant-Food-Research-Open/calisim) wraps many optimisation libraries
(optuna, emukit, openturns, botorch, ...) behind one `specify() / execute() / analyze()`
interface. `CalisimOptimiser` plugs that interface into CA's existing `Optimiser` contract, so
every calisim optimisation engine becomes usable as a `param_id_method` without writing a new
optimiser loop per library. The available engine/method pairs are surfaced to CUFLynx by
`param_id/calisim_methods.py`.

How CA's inputs become calisim inputs (deliberately thin -- no new user input files):

* `{prefix}_params_for_id.csv` is already parsed into `param_id_info` (`param_names`,
  `param_mins`, `param_maxs`). Each calibrated parameter becomes one calisim `DistributionModel`
  with a uniform distribution over ``[min, max]``, i.e. calisim samples in *real* (un-normalised)
  parameter space, in the same order as `param_id_info['param_names']`.
* `{prefix}_obs_data.json` is *not* re-parsed here. calisim's `calibration_func` is a thin adapter
  over `OpencorParamID.get_cost_from_params()`, so the observation weights, protocols,
  multi-experiment setup, cost type and any user-defined operations all keep working exactly as
  they do for the built-in optimisers.

How calisim's outputs become CA's outputs: the best (cost, params) pair is tracked inside the
objective rather than read back from the engine, which is both engine-independent and immune to
calisim's `analyze()` step being optional/absent. The usual CA artefacts are written --
`best_cost.npy`, `best_param_vals.npy`, and the `best_cost_history.csv` /
`best_param_vals_history.csv` progress files in the same format as the other optimisers -- so
plotting, `simulate_with_best_param_vals()` and everything downstream is unchanged. Whatever
calisim itself emits (trial CSVs, plots) lands in a `calisim/` subdirectory of the output dir.

MPI: calisim owns its optimisation loop and evaluates one parameter set at a time, so this first
implementation runs the loop on rank 0 and leaves the other ranks idle, exactly like
`sp_minimize`. Rank-0 failures are broadcast so no rank is left waiting.
"""

import os
import re
import traceback

import numpy as np

from param_id.calisim_methods import split_calisim_method_name
from param_id.optimisers import Optimiser

try:
    import calisim  # noqa: F401
    CALISIM_AVAILABLE = True
except ImportError:
    CALISIM_AVAILABLE = False

# Cost returned to calisim when a simulation fails. The built-in optimisers hand np.inf straight
# to their samplers, but optuna/botorch surrogates fit a model through the returned values and a
# non-finite one poisons that fit, so failures are reported as a large finite penalty instead.
FAILED_SIMULATION_COST = 1e10


class CalisimOptimiser(Optimiser):
    """Run any calisim optimisation engine against CA's cost function."""

    def __init__(self, param_id_obj, param_id_info, param_norm_obj, num_params, output_dir,
                 optimiser_options=None, param_id_method=None, DEBUG=False):
        super().__init__(param_id_obj, param_id_info, param_norm_obj, num_params, output_dir,
                         optimiser_options=optimiser_options, DEBUG=DEBUG)

        self.engine, self.method = split_calisim_method_name(param_id_method)

        self.param_mins = np.asarray(param_id_info['param_mins'], dtype=float)
        self.param_maxs = np.asarray(param_id_info['param_maxs'], dtype=float)

        opts = self.optimiser_options
        self.num_calls_to_function = int(opts.get('num_calls_to_function', 100))
        if self.DEBUG:
            # Keep a debug run short even if the user left a production budget in place.
            self.num_calls_to_function = min(self.num_calls_to_function, 20)
        self.n_init = int(opts.get('n_init', 10))
        self.random_seed = opts.get('random_seed', 0)
        self.n_jobs = int(opts.get('n_jobs', 1))
        self.method_kwargs = opts.get('method_kwargs', None) or {}
        self.acquisition_func = opts.get('acquisition_func', 'ei')
        self.cost_convergence = opts.get('cost_convergence', 1e-4)

        # An initial design bigger than the whole budget makes the engines either error or spend
        # the entire run sampling at random, so clamp it.
        self.n_init = max(1, min(self.n_init, max(1, self.num_calls_to_function // 2)))

        self.calisim_param_names = self._make_calisim_param_names()
        self.num_evaluations = 0
        self._stopped_early = False

    # -------------------------------------------------------------------------------------
    # CA <-> calisim translation
    # -------------------------------------------------------------------------------------
    def _make_calisim_param_names(self):
        """One calisim-safe, unique name per calibrated parameter, in param_id_info order.

        CA parameter names are ``component/variable`` (and one calibrated parameter may be shared
        by several components), which is not usable as a calisim/optuna parameter key, so they are
        sanitised and index-prefixed to guarantee uniqueness while staying readable in calisim's
        own trial output.
        """
        names = []
        for idx, name_or_list in enumerate(self.param_id_info['param_names']):
            name = name_or_list[0] if isinstance(name_or_list, (list, tuple)) else name_or_list
            safe = re.sub(r'[^0-9a-zA-Z_]', '_', str(name))[:60]
            names.append(f'p{idx}_{safe}')
        return names

    def _parameter_spec(self):
        """`param_id_info` bounds -> a calisim ParameterSpecification (uniform over [min, max])."""
        from calisim.data_model import (DistributionModel, ParameterDataType,
                                        ParameterSpecification)

        parameters = []
        for idx, name in enumerate(self.calisim_param_names):
            lower = float(self.param_mins[idx])
            upper = float(self.param_maxs[idx])
            parameters.append(DistributionModel(
                name=name,
                distribution_name='uniform',
                distribution_args=[lower, upper],
                data_type=ParameterDataType.CONTINUOUS,
            ))
        return ParameterSpecification(parameters=parameters)

    def _params_dict_to_array(self, parameters):
        """calisim's ``{name: value}`` -> CA's ordered parameter vector (real space)."""
        return np.array([float(parameters[name]) for name in self.calisim_param_names])

    def _evaluate(self, parameters):
        """Evaluate one calisim parameter dict with CA's cost function."""
        if self._stopped_early or self.num_evaluations >= self.num_calls_to_function:
            # calisim owns the loop and some engines overshoot n_iterations (initial designs,
            # batched asks). Once the budget is spent, or the cost has converged, stop simulating
            # and hand back the incumbent so the remaining asks cost nothing.
            self._stopped_early = True
            return float(min(self.best_cost, FAILED_SIMULATION_COST))

        param_vals = self._params_dict_to_array(parameters)
        cost = self.param_id_obj.get_cost_from_params(param_vals)
        self.num_evaluations += 1

        if cost is None or not np.isfinite(cost):
            return FAILED_SIMULATION_COST

        cost = float(cost)
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_param_vals = param_vals
            self._save_best_params()
        self._write_history()

        if self.best_cost < self.cost_convergence:
            print(f'[calisim] cost {self.best_cost:.6e} is below cost_convergence='
                  f'{self.cost_convergence}; stopping early.')
            self._stopped_early = True
        return cost

    def _calibration_func(self, parameters, simulation_id, observed_data, **kwargs):
        """The function calisim calls. Returns the CA cost (a scalar loss to minimise).

        calisim passes a list of parameter dicts instead of one when a vectorised engine runs with
        `batched=True`; that is handled so a future switch to batched engines does not silently
        break.
        """
        if isinstance(parameters, (list, tuple)):
            return [self._evaluate(one) for one in parameters]
        return self._evaluate(parameters)

    def _write_history(self):
        """Append to the same progress files the other optimisers write, so the existing plotting
        (plot_param_id) works unchanged: one best-cost per row, and the best params normalised."""
        if self.output_dir is None or self.best_param_vals is None:
            return
        with open(os.path.join(self.output_dir, 'best_cost_history.csv'), 'a') as file:
            np.savetxt(file, np.array([[self.best_cost]]), fmt='%1.9f', delimiter=', ')
        best_norm = self.param_norm_obj.normalise(np.asarray(self.best_param_vals).reshape(-1, 1))
        with open(os.path.join(self.output_dir, 'best_param_vals_history.csv'), 'a') as file:
            np.savetxt(file, np.asarray(best_norm).reshape(1, -1), fmt='%.5e', delimiter=', ')

    # -------------------------------------------------------------------------------------
    # run
    # -------------------------------------------------------------------------------------
    def _build_calibrator(self):
        from calisim.optimisation import OptimisationMethod, OptimisationMethodModel

        calisim_outdir = os.path.join(self.output_dir, 'calisim')
        os.makedirs(calisim_outdir, exist_ok=True)

        method_kwargs = dict(self.method_kwargs)
        if self.engine == 'optuna' and self.random_seed is not None:
            # calisim's optuna wrapper passes method_kwargs to the sampler constructor and never
            # looks at specification.random_seed, so seed the sampler here to make the run
            # reproducible. An explicit user seed in method_kwargs still wins.
            method_kwargs.setdefault('seed', int(self.random_seed))

        specification = OptimisationMethodModel(
            experiment_name=f'CA_{self.engine}_{self.method or "default"}',
            parameter_spec=self._parameter_spec(),
            observed_data=None,
            outdir=calisim_outdir,
            method=self.method,
            directions=['minimize'],
            output_labels=['cost'],
            n_iterations=self.num_calls_to_function,
            n_init=self.n_init,
            n_jobs=self.n_jobs,
            random_seed=self.random_seed,
            acquisition_func=self.acquisition_func,
            method_kwargs=method_kwargs,
        )
        return OptimisationMethod(calibration_func=self._calibration_func,
                                  specification=specification, engine=self.engine)

    def _run_on_rank_0(self):
        print(f'Running calisim optimisation: engine={self.engine!r}, '
              f'method={self.method or "<engine default>"!r}')
        print(f'  Budget: {self.num_calls_to_function} cost-function calls, '
              f'n_init={self.n_init}, seed={self.random_seed}')

        calibrator = self._build_calibrator()
        calibrator.specify().execute()

        # analyze() is what writes calisim's own plots/CSV artefacts. It is decorative for CA (the
        # best point is tracked in the objective), and it pulls in plotting backends, so a failure
        # here must not lose a completed calibration.
        try:
            calibrator.analyze()
            artifacts = calibrator.get_artifacts()
            if artifacts:
                print('[calisim] artifacts:\n  ' + '\n  '.join(str(a) for a in artifacts))
        except Exception as analyze_error:
            print(f'[calisim] WARNING: analyze() failed ({analyze_error}); the calibration result '
                  'is unaffected, only calisim\'s own plots/tables were not written.')

        if self.best_param_vals is None:
            raise RuntimeError(
                f'calisim engine {self.engine!r} finished without a successful simulation; every '
                'cost evaluation failed. Check the parameter bounds and the model setup.')

    def run(self):
        if not CALISIM_AVAILABLE:
            raise ImportError(
                'The calisim package is required for the calisim_* param_id_methods but is not '
                'installed. Install it with `pip install calisim` (the botorch engine also needs '
                'the `calisim[torch]` extra). Note calisim requires python >=3.10,<3.13.')

        comm = self.comm
        if self.rank == 0 and self.num_procs > 1:
            print(f'WARNING calisim drives its own optimisation loop, so it runs on rank 0 only; '
                  f'the other {self.num_procs - 1} rank(s) will be idle.')

        error = None
        if self.rank == 0:
            try:
                self._run_on_rank_0()
            except Exception:
                error = traceback.format_exc()

        error = comm.bcast(error, root=0)
        if error is not None:
            raise RuntimeError(f'calisim optimisation failed on rank 0:\n{error}')

        self.best_param_vals, self.best_cost = comm.bcast(
            (self.best_param_vals, self.best_cost), root=0)
        self._save_best_params()

        if hasattr(self.param_id_obj, 'set_best_param_vals'):
            self.param_id_obj.set_best_param_vals(self.best_param_vals)

        if self.rank == 0:
            print(f'calisim optimisation finished after {self.num_evaluations} cost evaluations; '
                  f'best cost {self.best_cost}')
