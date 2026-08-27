'''
@author: Finbar J. Argus
'''

import contextlib
import numpy as np
import os
import sys
from sys import exit
import math as math
try:
    import opencor as oc
    opencor_available = True
except:
    opencor_available = False
    pass
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from libcuflynx.utilities import paperPlotSetup
from libcuflynx.utilities import stat_distributions
from libcuflynx.utilities import diagnostics
from libcuflynx.utilities import utility_funcs
import traceback
from libcuflynx.utilities.utility_funcs import Normalise_class
from libcuflynx.utilities.paths import default_param_id_output_dir, default_resources_dir
paperPlotSetup.Setup_Plot(3)
from libcuflynx.solver_wrappers import get_simulation_helper
from libcuflynx.protocol_runners.protocol_executor import ProtocolExecutor
from libcuflynx.parsers.PrimitiveParsers import scriptFunctionParser
# Not `from mpi4py import MPI`: that import initialises MPI and registers an
# atexit MPI_Finalize, and with no launcher present that finalise is what aborts
# on macOS when a NIC goes away (#396). get_MPI hands back the real
# mpi4py.MPI under mpiexec -- a multi-rank run is unchanged -- and a one-rank
# stub otherwise, so a serial run never opens MPI at all.
from libcuflynx.utilities.mpi_utils import get_MPI as _get_MPI

MPI = _get_MPI()
import re
from numpy import genfromtxt
from importlib import import_module
# import tqdm # TODO this needs to be installed for corner plot but doesnt need an import here
# Which sampler a UQ run uses is UQ_options['library'], read in MCMC._build_sampler -- not
# the module-level constant this used to be, which meant editing the source to change sampler.
# Both are imported optionally: emcee is a CA dependency but zeus is not, and pymc is imported
# only inside its own backend module (it is an optional [uq] extra, and this module is imported
# by every calibration run, UQ or not).
try:
    import emcee
except ImportError:
    emcee = None
try:
    import zeus
except ImportError:
    zeus = None
# corner is NOT imported here, deliberately. It is a core dependency, so this was once a
# plain module-level import -- but it is used at exactly four call sites, all of them inside
# plot_mcmc(), and importing it drags in arviz and xarray on every rank of every calibration,
# the vast majority of which never draw a corner plot.
#
# It also cannot be guarded with `except ImportError`. arviz 0.23.4 writes a once-a-day stamp
# file at import time through a fixed temporary name, so concurrent MPI ranks race and the
# losers get FileNotFoundError -- which sails straight through an ImportError guard and takes
# the rank, and then via MPI_ABORT the whole job, down with it. See
# utilities/lazy_imports.py, which carries the measurements and the reasoning.
from libcuflynx.utilities.lazy_imports import require_corner
import csv
import shutil
from datetime import date, datetime
# from skopt import gp_minimize, Optimizer
from libcuflynx.parsers.PrimitiveParsers import (CSVFileParser, ObsAndParamDataParser, PARAM_ID_METHODS,
                                      PARAM_PRIOR_TYPES, prior_param_default)
from libcuflynx.param_id.optimisers import GeneticAlgorithmOptimiser, BayesianOptimiser, CMAESOptimiser, \
    SciPyMinimizeOptimiser, MultiStartSciPyMinimizeOptimiser
from libcuflynx.param_id.differentiable import (
    assert_casadi_differentiable,
    assert_mle_cost_for_bayesian,
    is_circulatory_differentiable,
)
from libcuflynx.param_id.operation_funcs import resolve_operation_kwargs, validate_operation_kwargs
from libcuflynx.param_id.cost_kwargs import call_cost_func, ground_truth_param_name, validate_cost_kwargs
from libcuflynx.parsers.PrimitiveParsers import (apply_modifier_identity_nominals,
                                      expand_modifier_param_vals,
                                      param_entry_labels,
                                      resolve_modifier_baselines,
                                      save_param_modifiers)
from libcuflynx.param_id.plot_outputs import ParamIDPlotOutputs
from libcuflynx.param_id import casadi_backend
from libcuflynx.param_id import fsa_backend
from libcuflynx.param_id import aadc_backend
from libcuflynx.param_id import fd_backend
import pandas as pd
try:
    import casadi as ca
except ImportError:
    ca = None
import json
import math
import scipy.linalg as la
# from scipy.optimize import curve_fit
import warnings
from libcuflynx.utilities.obs_data_helpers import obs_item_names, obs_item_labels
warnings.filterwarnings( "ignore", module = "matplotlib/..*" )
# TODO maybe remove matplotlib warnings as above

# set resource limit to inf to stop seg fault problem #TODO remove this, I don't think it does much
# import resource
# curlimit = resource.getrlimit(resource.RLIMIT_STACK)
# resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY,resource.RLIM_INFINITY))

# This mcmc_object will be an instance of the ParamID class
# it needs to be global so that it can be used in calculate_lnlikelihood()
# without having its attributes pickled. opencor simulation objects
# can't be pickled because they are pyqt.
mcmc_object = None

#: The UQ backends that parallelise across MPI ranks by farming likelihood evaluations out to a
#: worker pool, rather than by giving each rank chains of its own. See
#: ``MCMC.sampler_needs_a_worker_pool`` for why the two arrangements cannot be mixed.
_POOL_BACKED_UQ_LIBRARIES = ('emcee', 'zeus')


# numpy 2.0 renamed trapz to trapezoid and removed the old name; numpy 1.x has only the old one,
# and this project supports both (CI runs 2.2 while the OpenCOR shell ships 1.26). Bound once here
# rather than branched at the call site, and named for what it does rather than for either
# spelling.
try:
    from numpy import trapezoid as integrate_trapezoid       # numpy >= 2.0
except ImportError:                                          # pragma: no cover - numpy < 2.0
    from numpy import trapz as integrate_trapezoid


def _resolve_UQ_options(UQ_options, mcmc_options):
    """Accept the deprecated ``mcmc_options=`` kwarg wherever ``UQ_options=`` is now taken.

    MCMC is one method of uncertainty quantification rather than the whole of it, so the options
    moved to ``UQ_options`` with a ``method`` key. Callers passing the old name keep working and
    are told once. Passing both is refused: the two can disagree, and picking a winner would
    silently discard one from a caller who believes it is in effect.
    """
    if mcmc_options is None:
        return UQ_options
    if UQ_options is not None:
        raise ValueError(
            "pass either UQ_options or the deprecated mcmc_options, not both -- they are the "
            "same setting and their values can disagree.")
    print("WARNING: the 'mcmc_options' argument is deprecated; use 'UQ_options' instead "
          "(MCMC is now selected with UQ_options={'method': 'mcmc', ...}).")
    return mcmc_options


def ensure_mle_cost_type_for_bayesian_inner(inner, inp_data_dict):
    """
    Make sure every observable uses an ``@is_MLE`` cost (required for ``ln L = -cost`` in
    MCMC / Laplace), **without discarding the ones that already do**.

    An observable whose own ``cost_type`` is already an MLE cost keeps it. Only the rest are
    replaced, with the first ``cost_type`` found in the optimiser / UQ option dicts of
    ``inp_data_dict`` that names an ``@is_MLE`` cost, or ``gaussian_MLE``.

    It used to overwrite the whole vector with that one name, which quietly undid every
    per-data_item choice the obs_data made -- and worse than quietly. A ``poisson_MLE`` count
    is scored against ``prob_dist_params`` and deliberately carries no ``value``, so rewriting
    it to ``gaussian_MLE`` compared the model against a ground truth of ``nan`` and every
    sample in the chain came back ``nan``. A ``gaussian_MLE_robust`` item simply lost its
    outlier component and went back to paying hundreds of nats for a wrong branch. Both are
    MLE costs; neither needed replacing.

    The consequence for the legacy spelling: a ``cost_type`` named in ``UQ_options`` no
    longer forces a cost onto observables that already name a valid one. It is the fallback
    for those that do not, which is what the key is for now that ``cost_type`` lives per
    data_item.
    """
    if inner is None or getattr(inner, "obs_info", None) is None:
        return
    costs = getattr(inner, "cost_funcs_dict", None) or {}
    chosen = None
    option_dicts = []
    if inp_data_dict.get("DEBUG"):
        option_dicts.append(inp_data_dict.get("debug_optimiser_options") or {})
        option_dicts.append(inp_data_dict.get("debug_UQ_options")
                            or inp_data_dict.get("debug_mcmc_options") or {})
    option_dicts.append(inp_data_dict.get("optimiser_options") or {})
    # The legacy spelling is still read here: parse_user_inputs_file normalises it, but this is
    # also reachable with a hand-built dict that never went through the parser.
    option_dicts.append(inp_data_dict.get("UQ_options")
                        or inp_data_dict.get("mcmc_options") or {})
    for src in option_dicts:
        if not isinstance(src, dict):
            continue
        ct = src.get("cost_type")
        fn = costs.get(ct) if ct else None
        if fn is not None and getattr(fn, "is_MLE", False):
            chosen = ct
            break
    if chosen is None:
        chosen = "gaussian_MLE"
    n = inner.obs_info["num_obs"]
    existing = list(inner.obs_info.get("cost_type") or [])
    resolved = []
    for idx in range(n):
        current = existing[idx] if idx < len(existing) else None
        func = costs.get(current) if current else None
        keep = func is not None and getattr(func, "is_MLE", False)
        resolved.append(current if keep else chosen)
    inner.obs_info["cost_type"] = resolved
    inner.cost_type = resolved


# Re-exported for backwards compatibility; the canonical definition is in param_id.aadc_backend.
TAPE_CONSISTENT_AADC_METHODS = aadc_backend.TAPE_CONSISTENT_METHODS

# The CasADi symbolic cost/gradient/observable machinery lives in param_id.casadi_backend; the
# methods below delegate to it. require_casadi/as_casadi_column are re-bound to their previous
# private names because the *generic* cost-assembly layer that stays here (cost_calc,
# get_obs_output_dict) still builds SX expressions directly and calls them.
_require_casadi = casadi_backend.require_casadi
_as_casadi_column = casadi_backend.as_casadi_column


#: The model types parameter identification can actually run. Deliberately not every entry
#: in SOLVER_SCHEMA's model types: `cpp` is a valid model_type for *generation*, but neither
#: this module nor solver_wrappers can simulate one, so it cannot be calibrated. Naming the
#: set here keeps the check and the error message reading from one list.
PARAM_ID_MODEL_TYPES = ('cellml', 'python', 'casadi_python', 'aadc_python', 'external_python')


class CVS0DParamID():
    """Parameter identification (calibration) for a 0D CVS model.

    This is the main user-facing entry point for calibration. It wraps an inner
    optimisation engine ([`ParamID`][param_id.paramID.ParamID], or
    [`MCMC`][param_id.paramID.MCMC] when ``mcmc_instead=True``) and
    coordinates loading observation data, selecting parameters, running the
    optimiser, and writing/plotting results. It is MPI-aware: rank 0 handles all
    file I/O and output directory creation.

    Construct it either directly, or from a config dict with
    [`init_from_dict`][param_id.paramID.CVS0DParamID.init_from_dict]. A typical
    flow is::

        pid = CVS0DParamID.init_from_dict(inp)
        pid.set_ground_truth_data(obs_data_dict)
        pid.set_params_for_id(params_for_id_dict)
        pid.set_param_id_method("genetic_algorithm")
        pid.run()
        pid.simulate_with_best_param_vals()
        pid.plot_outputs()

    Args:
        model_path: Path to the generated model file (CellML/Python/CasADi).
        model_type: One of ``'cellml'``, ``'python'``, ``'casadi_python'``.
        param_id_method: Optimiser to use, e.g. ``'genetic_algorithm'``,
            ``'CMA-ES'``, ``'bayesian'``, ``'sp_minimize'``.
        mcmc_instead: If True, build an MCMC sampler instead of an optimiser.
        file_name_prefix: Model name prefix; ties together the resource files
            and names the output case directory.
        params_for_id_path: Optional path to a ``{prefix}_params_for_id.csv``.
            Alternatively call
            [`set_params_for_id`][param_id.paramID.CVS0DParamID.set_params_for_id].
        param_id_obs_path: Optional path to an ``obs_data.json``. Alternatively
            call
            [`set_ground_truth_data`][param_id.paramID.CVS0DParamID.set_ground_truth_data].
        sim_time: Logged simulation duration (s).
        pre_time: Unlogged steady-state spin-up duration (s).
        dt: Output sampling step (s); must be <= every dt in the obs data.
        solver_info: Solver config dict (defaults to ``{"solver": "CVODE_myokit"}``).
        UQ_options: Options dict for uncertainty quantification (used when
            ``mcmc_instead=True``): ``method`` (only ``'mcmc'`` so far), ``library``,
            ``num_steps``, ``num_walkers``, ``burn_in``. ``mcmc_options`` is accepted as a
            deprecated alias.
        optimiser_options: Options dict for the optimiser (e.g. ``cost_convergence``,
            ``max_patience``, ``num_calls_to_function``, ``cost_type``). Sensible
            defaults are used if omitted.
        do_ad: Enable automatic differentiation (CasADi backend).
        DEBUG: Enable debug behaviour and the debug optimiser options.
        param_id_output_dir: Root directory for results; defaults to
            ``param_id_output/`` in the repo.
        resources_dir: Directory holding input resources; defaults to
            ``resources/`` in the repo.
        one_rank: If True, skip the MPI barrier (single-rank usage).

    Attributes:
        output_dir: Directory (under ``param_id_output_dir``) where results and
            plots for this case are written (rank 0 only).
    """
    def __init__(self, model_path, model_type, param_id_method, mcmc_instead=False, file_name_prefix='no_name',
                 params_for_id_path=None,
                 param_id_obs_path=None, sim_time=2.0, pre_time=20.0, dt=0.01,
                 solver_info=None, UQ_options=None, optimiser_options=None,
                 do_ad=False, DEBUG=False,
                 param_id_output_dir=None, resources_dir=None, one_rank=False,
                 operation_funcs_external_path=None, cost_funcs_external_path=None,
                 modifier_funcs_external_path=None, mcmc_options=None,
                 use_emulator=False, emulator_dir=None, emulator_settings=None):
        self.model_path = model_path
        self.param_id_method = param_id_method
        self.mcmc_instead = mcmc_instead
        self.model_type = model_type
        self.file_name_prefix = file_name_prefix
        # Emulator mode (#333): the analyses evaluate a trained surrogate instead of the
        # solver. `solver_info['solver']` still names the solver it was trained against.
        self.use_emulator = bool(use_emulator)
        self.emulator_dir = emulator_dir
        self.emulator_settings = dict(emulator_settings or {})
        # Optional external user-func files (issue #303), threaded into the param-id engine so its
        # operation/cost dicts merge them in alongside the built-ins. modifier_funcs (issue #383)
        # follow the same pattern via the params_for_id parser.
        self.operation_funcs_external_path = operation_funcs_external_path
        self.cost_funcs_external_path = cost_funcs_external_path
        self.modifier_funcs_external_path = modifier_funcs_external_path

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.num_procs = self.comm.Get_size()

        self.UQ_options = _resolve_UQ_options(UQ_options, mcmc_options)
        if solver_info is None:
            self.solver_info = {"solver": "CVODE_myokit"}
        else:
            self.solver_info = solver_info
        self.dt = dt
        self.sim_time = sim_time
        self.pre_time = pre_time

        if param_id_obs_path is None:
            date_str = date.today().strftime("%Y%m%d")
            self.param_id_obs_file_prefix = f"obs_{date_str}"
        else:
            self.param_id_obs_file_prefix = re.sub('.json', '', os.path.split(param_id_obs_path)[1])
        case_type = f'{param_id_method}_{file_name_prefix}_{self.param_id_obs_file_prefix}'
        if self.rank == 0:
            if param_id_output_dir is None:
                self.param_id_output_dir = default_param_id_output_dir()
            else:
                self.param_id_output_dir = param_id_output_dir
            
            if not os.path.exists(self.param_id_output_dir):
                os.mkdir(self.param_id_output_dir)
            self.output_dir = os.path.join(self.param_id_output_dir, f'{case_type}')
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)
            self.plot_dir = os.path.join(self.output_dir, 'plots_param_id')
            if not os.path.exists(self.plot_dir):
                os.mkdir(self.plot_dir)
            # Archive the input files (timestamped) used for this run so the user can
            # later check exactly what params_for_id / obs_data produced these outputs.
            self._archive_input_files(params_for_id_path, param_id_obs_path)
        else:
            self.output_dir = None
        
        if resources_dir is None:
            self.resources_dir = default_resources_dir()
        else:
            self.resources_dir = resources_dir

        if one_rank is False:
            self.comm.Barrier()

        self.DEBUG = DEBUG
        # if self.DEBUG:
        #     import resource

        # TODO I should have a separate class for parsing the observable info from param_id_obs_path
        #  and param info from params_for_id_path
        # param names
        self.param_id_info = None
        self.gt_df = None
        self.protocol_info = None
        self.obs_info = None
        self.prediction_info = None
        self.params_for_id_path = params_for_id_path
        self.optimiser_options = optimiser_options
        self.obs_and_param_parser = ObsAndParamDataParser(
            modifier_funcs_external_path=modifier_funcs_external_path)
        if param_id_obs_path:
            # self.__set_obs_names_and_df(param_id_obs_path, sim_time=sim_time, pre_time=pre_time)
            parsed_data = self.obs_and_param_parser.parse_obs_data_json(
                param_id_obs_path=param_id_obs_path,
                pre_time=pre_time,
                sim_time=sim_time,
                model_type=model_type,
                method=(solver_info or {}).get('method'),
            )
            self.gt_df = parsed_data["gt_df"]
            self.protocol_info = parsed_data["protocol_info"]
            self.prediction_info = parsed_data["prediction_info"]

            self.obs_info = self.obs_and_param_parser.process_obs_info(gt_df=self.gt_df, output_dir=self.output_dir, dt=self.dt)
            self.protocol_info = self.obs_and_param_parser.process_protocol_and_weights(
                gt_df=self.gt_df,
                protocol_info=self.protocol_info,
                dt=self.dt
            )

        if self.params_for_id_path:
            self.param_id_info = self.obs_and_param_parser.get_param_id_info(self.params_for_id_path)
            self.obs_and_param_parser.save_param_names(self.param_id_info, self.output_dir)

        if self.optimiser_options is None:
            print("No optimiser options provided, using default options")
            self.optimiser_options = {
                'cost_convergence': 0.0001,
                'max_patience': 10,
                'num_calls_to_function': 10000
            }
            print(f'Default optimiser options: {self.optimiser_options}')

        if self.mcmc_instead:
            # This mcmc_object will be an instance of the ParamID class
            # it needs to be global so that it can be used in calculate_lnlikelihood()
            # without having its attributes pickled. opencor simulation objects
            # can't be pickled because they are pyqt.
            global mcmc_object 
            mcmc_object = MCMC(self.model_path,
                                           self.obs_info, self.param_id_info,
                                           self.protocol_info, self.prediction_info, self.solver_info, dt=self.dt,
                                           UQ_options=self.UQ_options,
                                           DEBUG=self.DEBUG, model_type=self.model_type,
                                           use_emulator=self.use_emulator,
                                           emulator_dir=self.emulator_dir,
                                           emulator_settings=self.emulator_settings)
            self.n_steps = mcmc_object.n_steps
        else:
            if model_type in PARAM_ID_MODEL_TYPES:
                self.param_id = ParamID(self.model_path, self.param_id_method,
                                               self.obs_info, self.param_id_info, self.protocol_info,
                                               self.prediction_info, self.solver_info, dt=self.dt,
                                               optimiser_options=self.optimiser_options,
                                               do_ad=do_ad, DEBUG=self.DEBUG,
                                               model_type=self.model_type,
                                               operation_funcs_external_path=self.operation_funcs_external_path,
                                               cost_funcs_external_path=self.cost_funcs_external_path,
                                               use_emulator=self.use_emulator,
                                               emulator_dir=self.emulator_dir,
                                               emulator_settings=self.emulator_settings)
                self.n_steps = self.param_id.n_steps
            else:
                # Say so here rather than leaving self.param_id unset. set_output_dir()
                # dereferences it a few lines below, so an unsupported model_type used to
                # surface as `'CVS0DParamID' object has no attribute 'param_id'` -- a
                # message about an attribute, several frames from the config key that
                # actually caused it (CUFLynx #270). `cpp` is the one that reaches here by
                # being *valid*: it is a real model_type for generation, but nothing in
                # param_id or solver_wrappers can run one, so calibrating it was never
                # going to work and should say which part is missing.
                raise ValueError(
                    f'model_type {model_type!r} cannot be used for parameter '
                    f'identification. Supported: {sorted(PARAM_ID_MODEL_TYPES)}. '
                    f'(cpp models can be generated, but not calibrated or simulated by '
                    f'libcuflynx -- build and run the generated code yourself.)')
        if self.rank == 0:
            self.set_output_dir(self.output_dir)
        
        self.best_output_calculated = False
        self.sensitivity_calculated = False

    def _archive_input_files(self, params_for_id_path, param_id_obs_path):
        """Copy the params_for_id and obs_data input files into the run output_dir.

        A ``_<yymmdd>_<HHMMSS>`` timestamp is inserted before the extension (a single
        timestamp shared by both files), so a user inspecting ``param_id_output/<case>/``
        can see exactly which inputs were used for that run. Missing/None paths are
        skipped; multiple runs into the same case dir accumulate timestamped copies.
        """
        if self.output_dir is None:
            return
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        for src in (params_for_id_path, param_id_obs_path):
            if not src or not os.path.isfile(src):
                continue
            base, ext = os.path.splitext(os.path.basename(src))
            dst = os.path.join(self.output_dir, f"{base}_{timestamp}{ext}")
            try:
                shutil.copy2(src, dst)
            except OSError as e:
                print(f"Warning: could not archive input file {src} -> {dst}: {e}")

    @classmethod
    def init_from_dict(cls, inp_data_dict):
        """Build a `CVS0DParamID` from a configuration dict.

        Only the keys relevant to the constructor are consumed. ``file_prefix``
        is accepted as an alias for ``file_name_prefix``.

        Args:
            inp_data_dict: Config dict, e.g. as returned by
                [`get_default_inp_data_dict`][utilities.utility_funcs.get_default_inp_data_dict]
                and then mutated in code.

        Returns:
            CVS0DParamID: A configured instance (observation data and parameters
            still need to be set unless their paths were in the dict).
        """
        # Only pass kwargs that exist in inp_data_dict
        arg_options = [
            'model_path', 'model_type', 'param_id_method', 'mcmc_instead',
            'file_name_prefix', 'params_for_id_path', 'param_id_obs_path',
            'sim_time', 'pre_time', 'dt', 'solver_info', 'UQ_options',
            'optimiser_options', 'DEBUG', 'param_id_output_dir', 'resources_dir',
            'one_rank', 'do_ad',
            'operation_funcs_external_path', 'cost_funcs_external_path',
            'modifier_funcs_external_path', 'use_emulator', 'emulator_settings',
        ]
        kwargs = {key: inp_data_dict[key] for key in arg_options if key in inp_data_dict}

        # Support common naming used elsewhere
        if 'file_name_prefix' not in kwargs and 'file_prefix' in inp_data_dict:
            kwargs['file_name_prefix'] = inp_data_dict['file_prefix']

        # Where this config's emulator lives, resolved from the same dict the trainer used, so
        # a run finds the emulator its own settings produced without naming a path twice.
        if kwargs.get('use_emulator'):
            from libcuflynx.emulators.emulator_trainer import resolve_emulator_dir
            kwargs['emulator_dir'] = resolve_emulator_dir(inp_data_dict)

        return cls(**kwargs)

    @classmethod
    def init_from_all_dicts(cls, inp_data_dict, obs_data_dict, params_for_id_dict):
        """Build a fully configured `CVS0DParamID` in one call.

        Convenience constructor that calls
        [`init_from_dict`][param_id.paramID.CVS0DParamID.init_from_dict] then sets
        the ground-truth data and the parameters to identify.

        Args:
            inp_data_dict: Configuration dict (see `init_from_dict`).
            obs_data_dict: Observation data dict (see
                [`ObsDataCreator`][utilities.obs_data_helpers.ObsDataCreator]).
            params_for_id_dict: List of parameter entries to calibrate (see
                [`set_params_for_id`][param_id.paramID.CVS0DParamID.set_params_for_id]).

        Returns:
            CVS0DParamID: A ready-to-run instance.
        """
        new_object = cls.init_from_dict(inp_data_dict)
        new_object.set_ground_truth_data(obs_data_dict)
        new_object.set_params_for_id(params_for_id_dict)
        return new_object

    def temp_test(self):
        self.param_id.temp_test()
    def temp_test2(self):
        self.param_id.temp_test2()

    def run(self):
        """Run the parameter identification.

        Executes the configured optimiser. Ground-truth data and parameters to
        identify must be set first. On rank 0 the best parameters are written to
        ``best_param_vals.npy`` and per-experiment full-output dumps
        (``all_outputs_with_best_param_vals_exp_*.npz``) are written under
        [`output_dir`][param_id.paramID.CVS0DParamID].

        Raises:
            ValueError: If observation data or parameters for id are not set.
        """
        self._check_info_available()
        self.param_id.run()

        # Some execution paths (or older optimiser flows) can finish without writing
        # the per-experiment full output dumps. Ensure they exist for downstream
        # tooling (e.g. post-processing, debug comparisons, external plotting).
        try:
            if getattr(self, "rank", 0) == 0:
                output_dir = getattr(self.param_id, "output_dir", None)
                protocol_info = getattr(self.param_id, "protocol_info", None)
                best_param_vals = getattr(self.param_id, "best_param_vals", None)

                if output_dir and protocol_info and best_param_vals is not None:
                    expected0 = os.path.join(
                        output_dir, "all_outputs_with_best_param_vals_exp_0.npz"
                    )
                    if not os.path.exists(expected0):
                        print(
                            "[param_id] all_outputs_with_best_param_vals_exp_*.npz "
                            "not found; generating per-experiment output dumps now."
                        )
                        self.param_id.save_all_outputs_per_experiment(
                            best_param_vals, suffix=""
                        )
        except Exception as e:
            # Don't fail an otherwise-successful optimisation because of optional artifacts.
            try:
                print(f"[param_id] WARNING: failed to write all-outputs npz dumps: {e}")
            except Exception:
                pass

    def run_UQ(self, UQ_options=None, mcmc_options=None):
        """Run uncertainty quantification (MCMC) on **this** object.

        Callable whether or not the instance was built with ``mcmc_instead=True``:

        * built with it -- runs the UQ engine constructed up front (unchanged behaviour);
        * built without it -- promotes the calibration engine via
          ``MCMC.from_param_id``, so UQ after a calibration reuses the model already
          compiled instead of building a second CVS0DParamID for it (CUFLynx #217).

        ``UQ_options`` overrides the options the object was built with; omit it to keep them.
        ``mcmc_options`` is a deprecated alias.
        """
        UQ_options = _resolve_UQ_options(UQ_options, mcmc_options)
        global mcmc_object
        if not self.mcmc_instead:
            if getattr(self, 'param_id', None) is None:
                raise RuntimeError(
                    "run_UQ needs either mcmc_instead=True or a built param-id engine; this "
                    "object has neither.")
            mcmc_object = MCMC.from_param_id(
                self.param_id,
                UQ_options if UQ_options is not None else getattr(self, 'UQ_options', None))
        elif UQ_options is not None:
            mcmc_object._init_mcmc(UQ_options, DEBUG=self.DEBUG)
        mcmc_object.run()

    def run_mcmc(self):
        """Deprecated alias of :meth:`run_UQ`, kept so existing scripts keep working."""
        return self.run_UQ()
    
    def _check_info_available(self):
        #new check, need ensure 'operands' or 'operation_kwargs' exist
        def is_nan(x):
            return isinstance(x, float) and math.isnan(x)
        obs_info = self.obs_info
        operands_list = obs_info.get("operands", [])
        operation_kwargs_list = obs_info.get("operation_kwargs", [])
        num_obs = len(operands_list)
        for i in range(num_obs):
            operands = operands_list[i]
            kwargs = operation_kwargs_list[i]
            if not isinstance(operands, (list, tuple)):
                operands = [operands]
            is_empty_operand = (len(operands) == 1 and operands[0] == "") or len(operands) == 0
            if is_empty_operand:
                # Case 2: operation_kwargs must NOT be nan / None / empty dict
                if kwargs is None or is_nan(kwargs) or kwargs == {}:
                    raise ValueError(f"[ERROR] In obs index {i}: operands is empty {operands}, "f"but operation_kwargs is invalid: {kwargs}")

        
        if self.gt_df is None:
            raise ValueError('Ground truth data not set')
        if self.protocol_info is None:
            raise ValueError('Protocol info not set')
        if self.obs_info is None:
            raise ValueError('Obs info not set')
        if self.param_id_info is None:
            raise ValueError('Param id info not set')

    def simulate_with_best_param_vals(self, reset=True, only_one_exp=-1, return_series=False):
        """Simulate the model using the best-fit parameters.

        Args:
            reset: Reset the simulation state before running.
            only_one_exp: If >= 0, only simulate that experiment index;
                ``-1`` simulates all experiments.
            return_series: If True, also return the full time-series arrays.

        Returns:
            If ``return_series`` is False, the observation dict of computed
            feature values. If True, a tuple ``(obs_dicts, obs_arrays)`` where
            ``obs_arrays`` holds the time-series for plotting.
        """
        if getattr(self.param_id, 'emulates_features', False):
            # Nothing to simulate: the emulator's features *are* the result, and
            # they were produced by the run that just finished. Returning rather
            # than raising matters because callers pair this with plot_outputs()
            # in one try block -- raising here cost the run its observable errors
            # too, which an emulator can perfectly well report (#333).
            print('use_emulator is set, so there is no simulation to re-run for the '
                  'best fit; the emulator predicts the observable features directly.')
            self.best_output_calculated = True
            return (None, None) if return_series else None
        if return_series:
            obs_dicts, obs_arrays = self.param_id.simulate_once(reset=reset, only_one_exp=only_one_exp, return_series=return_series)
            self.best_output_calculated = True
            return obs_dicts, obs_arrays
        else:
            obs_dict, _ = self.param_id.simulate_once(reset=reset, only_one_exp=only_one_exp)
            self.best_output_calculated = True
            return obs_dict

    def update_param_range(self, params_to_update_list_of_lists, mins, maxs):
        """Update the min/max bounds of a subset of parameters after construction.

        Args:
            params_to_update_list_of_lists: List of parameter-name groups to
                update; each must match an existing entry in the param-id info.
            mins: New lower bound for each group.
            maxs: New upper bound for each group.
        """
        for params_to_update_list, min, max in zip(params_to_update_list_of_lists, mins, maxs):
            for JJ, param_name_list in enumerate(self.param_id_info["param_names"]):
                if param_name_list == params_to_update_list:
                    self.param_id_info["param_mins"][JJ] = min
                    self.param_id_info["param_maxs"][JJ] = max

    def set_output_dir(self, path):
        """Override the directory where results and plots are written (rank 0 only)."""
        if self.rank != 0:
            return
        self.output_dir = path
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        if self.mcmc_instead:
            mcmc_object.set_output_dir(self.output_dir)
        else:
            self.param_id.set_output_dir(self.output_dir)
    

    def add_user_operation_func(self, func):
        """Register a custom feature-extraction function.

        The function can then be referenced by name in a data item's
        ``operation`` (its operands map to the function args). Set
        ``func.series_to_constant = True`` for series->scalar features so that
        auto-plotting works.

        Args:
            func: The Python callable to register.
        """
        self.param_id.add_user_operation_func(func)

    def add_user_cost_func(self, func):
        """Register a custom cost function (referenced via ``cost_type``)."""
        self.param_id.add_user_cost_func(func)

    def set_param_names(self, param_names):
        """Override the list of parameter names."""
        if self.mcmc_instead:
            mcmc_object.set_param_names(param_names)
        else:
            self.param_id.set_param_names(param_names)

    def set_optimiser_options(self, optimiser_options):
        """Set/update the optimiser options dict.

        Args:
            optimiser_options: e.g. ``cost_convergence``, ``max_patience``,
                ``num_calls_to_function``, ``cost_type``.
        """
        self.optimiser_options = optimiser_options
        self.param_id.set_optimiser_options(optimiser_options)

    def set_param_id_method(self, param_id_method):
        """Change the optimiser method.

        Args:
            param_id_method: e.g. ``'genetic_algorithm'``, ``'CMA-ES'``,
                ``'bayesian'``, ``'sp_minimize'``.
        """
        self.param_id_method = param_id_method
        self.param_id.set_param_id_method(param_id_method)

    def set_ground_truth_data(self, obs_data_dict):
        """Set the observation (ground-truth) data to calibrate against.

        Parses the obs-data structure into the internal ground-truth dataframe,
        protocol info, observation info and prediction info.

        Args:
            obs_data_dict: Observation data dict, e.g. built with
                [`ObsDataCreator`][utilities.obs_data_helpers.ObsDataCreator] or
                loaded from an ``obs_data.json`` file.
        """
        if self.rank == 0:
            print(f'Setting ground truth data: {obs_data_dict}')
        parsed_data = self.obs_and_param_parser.parse_obs_data_json(
            obs_data_dict=obs_data_dict,
            pre_time=self.pre_time,
            sim_time=self.sim_time,
            model_type=self.model_type,
        )
        self.gt_df = parsed_data["gt_df"]
        self.protocol_info = parsed_data["protocol_info"]
        self.prediction_info = parsed_data["prediction_info"]

        self.obs_info = self.obs_and_param_parser.process_obs_info(gt_df=self.gt_df, output_dir=self.output_dir, dt=self.dt)
        self.protocol_info = self.obs_and_param_parser.process_protocol_and_weights(
            gt_df=self.gt_df,
            protocol_info=self.protocol_info,
            dt=self.dt
        )
        self.param_id.set_obs_info(self.obs_info)
        self.param_id.set_protocol_info(self.protocol_info)
        self.param_id.set_prediction_info(self.prediction_info)
        if self.rank == 0:
            print(f'Ground truth data set: {self.obs_info}')
    
    def set_params_for_id(self, params_for_id_dict):
        """Set which parameters to identify and their bounds.

        Args:
            params_for_id_dict: List of entries of the form
                ``{vessel_name, param_name, min, max, name_for_plotting}`` (the
                in-memory equivalent of ``{prefix}_params_for_id.csv``).
                ``vessel_name`` may be a single name or a list of names to share
                one calibrated parameter across many vessels.
        """
        if self.rank == 0:
            print(f'Setting params for id: {params_for_id_dict}')
        self.param_id_info = self.obs_and_param_parser.get_param_id_info_from_entries(params_for_id_dict)
        self.obs_and_param_parser.save_param_names(self.param_id_info, self.output_dir)
        self.param_id.set_param_id_info(self.param_id_info)
        if self.rank == 0:
            print(f'Params for id set: {self.param_id_info["param_names"]}')

    def set_best_param_vals(self, best_param_vals):
        """Manually supply the best-fit parameter vector (e.g. from a previous run).

        Args:
            best_param_vals: Array of parameter values, ordered as
                [`get_param_names`][param_id.paramID.CVS0DParamID.get_param_names].
        """
        if self.mcmc_instead:
            mcmc_object.set_best_param_vals(best_param_vals)
        else:
            self.param_id.set_best_param_vals(best_param_vals)

    def _resolve_best_param_vals_for_outputs(self):
        """Return best-fit parameters for full-output NPZ dumps (memory or disk)."""
        if self.mcmc_instead:
            vals = mcmc_object.best_param_vals
        else:
            vals = self.param_id.best_param_vals
        if vals is not None:
            return vals
        if self.output_dir is not None:
            npy_path = os.path.join(self.output_dir, "best_param_vals.npy")
            if os.path.isfile(npy_path):
                vals = np.load(npy_path)
                self.set_best_param_vals(vals)
                print("[param_id] loaded best_param_vals.npy for NPZ output dump")
                return vals
        print(
            "[param_id] WARNING: best_param_vals not available; "
            "skipping _plot.npz dumps"
        )
        return None

    def plot_outputs(self):
        """Generate and save calibration result plots (under ``output_dir/plots_param_id``)."""
        if self.rank == 0:
            param_vals = self._resolve_best_param_vals_for_outputs()
            if param_vals is not None and not self.mcmc_instead:
                self.param_id.save_all_outputs_per_experiment(
                    param_vals, suffix="_plot"
                )
        ParamIDPlotOutputs(self).plot_outputs()

    def get_mcmc_samples(self):
        """Load and post-process the MCMC chain (burn-in + stuck-walker removal).

        Returns:
            tuple: ``(flat_samples, samples, num_params)``, or None if no chain
            has been written.
        """
        mcmc_chain_path = os.path.join(self.output_dir, 'mcmc_chain.npy')

        if not os.path.exists(mcmc_chain_path):
            print('No mcmc results to get chain')
            return

        samples = np.load(os.path.join(self.output_dir, 'mcmc_chain.npy'))
        samples = drop_unsampled_draws(samples)
        num_steps = samples.shape[0]
        num_walkers = samples.shape[1]
        num_params = samples.shape[2]  #
        if self.mcmc_instead:
            if num_params != mcmc_object.num_params:
                print('num params in mcmc chain doesn\'t equal mcmc_object number of params')
        else:
            if num_params != self.param_id.num_params:
                print('num params in mcmc chain doesn\'t equal param_id number of params')

        # TODO fix the below
        # for some reason some chains get stuck for long times, remove the chains that get stuck
        # I think this occurs when initialisation happens outside of the prior distribution
        walkers_to_remove = []
        for walker_idx in range(num_walkers):
            for param_idx in range(num_params):
                block_size = 200
                for step_block_idx in range(num_steps//block_size):
                    # get std of the block and remove that chain it if is zero
                    block_std = np.std(samples[step_block_idx*block_size:(step_block_idx+1)*block_size, walker_idx, param_idx])
                    if block_std == 0:
                        walkers_to_remove.append(walker_idx)

        walkers_to_remove = list(set(walkers_to_remove))
        if len(walkers_to_remove) > 0:
            print('There is a bug where chains can get stuck, removing walkers with stuck parameters. removed walker idxs:')
            print(walkers_to_remove)
            samples = np.delete(samples, walkers_to_remove, axis=1)

        # discard first num_steps/2 samples
        # TODO include a user defined burn in if we aren't starting from
        samples = samples[samples.shape[0]//2:, :, :]
        # thin = 5
        # samples = samples[::thin, :, :]
        flat_samples = samples.reshape(-1, num_params)

        return flat_samples, samples, num_params

    def plot_mcmc(self):
        """Generate MCMC trace and corner plots from the saved chain (rank 0)."""
        flat_samples, samples, num_params = self.get_mcmc_samples()
        if self.rank != 0:
            return

        # Imported here rather than at module scope: this is the only method in the file
        # that needs it, and on rank 0 only. See the note beside the import at the top.
        corner = require_corner("plot MCMC corner plots")

        means = np.zeros((num_params))
        conf_ivals = np.zeros((num_params, 3))

        for param_idx in range(num_params):
            means[param_idx] = np.mean(flat_samples[:, param_idx])
            conf_ivals[param_idx, :] = np.percentile(flat_samples[:, param_idx], [5, 50, 95])

        print('5th, 50th, and 95th percentile parameter values are:')
        print(conf_ivals)

        fig, axes = plt.subplots(num_params, figsize=(10, 7), sharex=True)
        for i in range(num_params):
            if hasattr(axes, '__len__'):
                ax = axes[i]
            else:
                ax = axes
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(f'${self.param_id_info["param_names_for_plotting"][i]}$')
            ax.yaxis.set_label_coords(-0.1, 0.5)

        ax.set_xlabel("step number")
            
        # plt.savefig(os.path.join(self.output_dir, 'plots_param_id', 'mcmc_chain_plot.eps'))
        plt.savefig(os.path.join(self.output_dir, 'plots_param_id', 'mcmc_chain_plot.pdf'))
        plt.close()

        label_list = [f'${self.param_id_info["param_names_for_plotting"][II]}$' for II in range(len(self.param_id_info["param_names_for_plotting"]))]
        if self.mcmc_instead:
            if mcmc_object.best_param_vals is None:
                best_param_vals = np.load(os.path.join(self.output_dir, 'best_param_vals.npy'))
                mcmc_object.set_best_param_vals(best_param_vals)
        else:
            if self.param_id.best_param_vals is None:
                best_param_vals = np.load(os.path.join(self.output_dir, 'best_param_vals.npy'))
                self.param_id.set_best_param_vals(best_param_vals)

        overwrite_params_to_plot_idxs = [II for II in range(num_params)] # This plots all param distributions
        if self.mcmc_instead:
            fig = corner.corner(flat_samples[:, overwrite_params_to_plot_idxs], bins=20, hist_bin_factor=2, smooth=0.5, quantiles=(0.05, 0.5, 0.95),
                                labels=[label_list[II] for II in overwrite_params_to_plot_idxs],
                                truths=mcmc_object.best_param_vals[overwrite_params_to_plot_idxs],
                                fontsize=20)
        else:
            fig = corner.corner(flat_samples[:, overwrite_params_to_plot_idxs], bins=20, hist_bin_factor=2, smooth=0.5, quantiles=(0.05, 0.5, 0.95),
                                labels=[label_list[II] for II in overwrite_params_to_plot_idxs],
                                truths=self.param_id.best_param_vals[overwrite_params_to_plot_idxs],
                                fontsize=20)
        axes = fig.get_axes()
        for idx, ax in enumerate(axes):
            if idx >= num_params*(num_params - 1):

                ax.tick_params(axis='both', rotation=0)
                formatterx = matplotlib.ticker.ScalarFormatter()
                ax.xaxis.set_major_formatter(formatterx)
                ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            if idx%num_params == 0:

                ax.tick_params(axis='both', rotation=0)
                formattery = matplotlib.ticker.ScalarFormatter()
                ax.yaxis.set_major_formatter(formattery)
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        plt.subplots_adjust(hspace=0.12, wspace=0.1)

        plt.savefig(os.path.join(self.plot_dir, f'mcmc_cornerplot_{self.file_name_prefix}_'
                                                f'{self.param_id_obs_file_prefix}.pdf'))
        plt.close()

        # do another corner plot with just a subset of params
        # overwrite_params_to_plot_idxs = [0,1, 4, 7] # This chooses a subset of params to plot
        if self.mcmc_instead:
            fig = corner.corner(flat_samples[:, overwrite_params_to_plot_idxs], bins=20, hist_bin_factor=2, smooth=0.5, quantiles=(0.05, 0.5, 0.95),
                                labels=[label_list[II] for II in overwrite_params_to_plot_idxs],
                                truths=mcmc_object.best_param_vals[overwrite_params_to_plot_idxs],
                                fontsize=20)
        else:
            fig = corner.corner(flat_samples[:, overwrite_params_to_plot_idxs], bins=20, hist_bin_factor=2, smooth=0.5, quantiles=(0.05, 0.5, 0.95),
                                labels=[label_list[II] for II in overwrite_params_to_plot_idxs],
                                truths=self.param_id.best_param_vals[overwrite_params_to_plot_idxs],
                                fontsize=20)
        axes = fig.get_axes()
        for idx, ax in enumerate(axes):
            if idx >= len(overwrite_params_to_plot_idxs)*(len(overwrite_params_to_plot_idxs) - 1):

                ax.tick_params(axis='both', rotation=0)
                formatterx = matplotlib.ticker.ScalarFormatter()
                ax.xaxis.set_major_formatter(formatterx)
                ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            if idx%len(overwrite_params_to_plot_idxs) == 0:

                ax.tick_params(axis='both', rotation=0)
                formattery = matplotlib.ticker.ScalarFormatter()
                ax.yaxis.set_major_formatter(formattery)
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        plt.subplots_adjust(hspace=0.12, wspace=0.1)

        plt.savefig(os.path.join(self.plot_dir, f'mcmc_cornerplot_subset_{self.file_name_prefix}_'
                                                f'{self.param_id_obs_file_prefix}.pdf'))
        plt.close()

        # Also check autocorrelation times for mcmc chain
        tau = self.calculate_autocorrelation_time(samples)

        # Per-parameter posterior summary with ESS and split-R-hat, and the two chain-diagnostic
        # plots. Printed rather than only returned: R-hat and ESS are the numbers that say
        # whether the posterior above is trustworthy at all, so a run should not be able to
        # finish without stating them.
        self.print_convergence_diagnostics(samples)
        self.plot_autocorrelation(samples)
        self.plot_chain_avg(samples)

        # check geweke convergence
        if not self.DEBUG:
            # the chain is too short when running debug to do geweke diagnostics
            # TODO test this another way
            acceptable = self.calculate_geweke_convergence(samples)
            if acceptable:
                print('chain passed geweke diagnostic with p>0.05')
            else:
                print('chain failed geweke diagnostic with p<0.05, USE CHAIN RESULTS WITH CARE')
        else:
            print("DEBUG mode, skipping geweke diagnostic becuase chain is too short in DEBUG")

    def calculate_autocorrelation_time(self, samples):
        tau = emcee.autocorr.integrated_time(samples, quiet=True)
        return tau

    # -----------------------------------------------------------------------
    # Convergence diagnostics (issue #367)
    # -----------------------------------------------------------------------
    # Computed here from numpy and emcee rather than through arviz. #367 imported arviz at module
    # level for these, which would have made every calibration run depend on it -- and arviz is
    # not a CA dependency, so the diagnostics would then be unavailable in exactly the
    # environments that need them. R-hat and ESS are short, standard formulas and emcee (already
    # a dependency) supplies the autocorrelation, so nothing is gained by the dependency.

    def calc_rhat(self, samples):
        """Split-R-hat (Gelman-Rubin) per parameter, from a ``(steps, walkers, params)`` chain.

        The *split* form: each walker is halved and the halves treated as separate chains, so a
        single walker that drifts steadily is caught. Plain R-hat cannot see that -- a drifting
        chain has a large within-chain variance, which is exactly what makes the ratio look fine.

        Returns ``{param_name: rhat}``. Values near 1 indicate the walkers have mixed; the usual
        working threshold is 1.01.
        """
        samples = np.asarray(samples, dtype=float)
        num_steps, num_walkers, num_params = samples.shape

        half = num_steps // 2
        if half < 2:
            return {name: float('nan') for name in self._param_labels(num_params)}

        # (steps, walkers, params) -> (2*walkers chains, half draws, params)
        chains = np.concatenate([samples[:half], samples[half:2 * half]], axis=1)
        chains = np.swapaxes(chains, 0, 1)
        num_chains = chains.shape[0]

        chain_means = chains.mean(axis=1)
        chain_vars = chains.var(axis=1, ddof=1)

        within = chain_vars.mean(axis=0)
        between = half * chain_means.var(axis=0, ddof=1) if num_chains > 1 else np.zeros(num_params)

        var_plus = ((half - 1) / half) * within + between / half
        with np.errstate(divide='ignore', invalid='ignore'):
            rhat = np.sqrt(np.where(within > 0, var_plus / within, np.nan))

        return dict(zip(self._param_labels(num_params), (float(r) for r in rhat)))

    def calc_effective_sample_size(self, samples):
        """Effective sample size per parameter: ``N / tau``, with tau the integrated
        autocorrelation time over the pooled chain.

        MCMC draws are correlated, so the number of samples overstates how much independent
        information the chain carries. This is the number that should be quoted alongside a
        posterior mean, not ``num_steps * num_walkers``.

        Returns ``{param_name: ess}``.
        """
        samples = np.asarray(samples, dtype=float)
        num_steps, num_walkers, num_params = samples.shape
        total = num_steps * num_walkers

        ess = {}
        for name, idx in zip(self._param_labels(num_params), range(num_params)):
            try:
                tau = float(emcee.autocorr.integrated_time(samples[:, :, idx], quiet=True)[0])
            except Exception:
                tau = float('nan')
            if not np.isfinite(tau) or tau <= 0:
                ess[name] = float('nan')
            else:
                # A chain can never carry more independent information than it has draws.
                ess[name] = float(min(total / tau, total))
        return ess

    def get_posterior_stats(self, samples):
        """Per-parameter posterior summary: mean, sd, the 3%/97% credible bounds, ESS and R-hat.

        One table rather than three separate arviz summary calls (#367 built the same dataset and
        re-ran the summary in each of three accessors, so the expensive part ran three times).
        """
        samples = np.asarray(samples, dtype=float)
        num_steps, num_walkers, num_params = samples.shape
        flat = samples.reshape(num_steps * num_walkers, num_params)

        ess = self.calc_effective_sample_size(samples)
        rhat = self.calc_rhat(samples)

        stats = {}
        for idx, name in enumerate(self._param_labels(num_params)):
            column = flat[:, idx]
            stats[name] = {
                'mean': float(np.mean(column)),
                'sd': float(np.std(column, ddof=1)) if column.size > 1 else float('nan'),
                'hdi_3%': float(np.percentile(column, 3)),
                'hdi_97%': float(np.percentile(column, 97)),
                'ess': ess[name],
                'r_hat': rhat[name],
            }
        return stats

    def print_convergence_diagnostics(self, samples):
        """Print the summary table and say plainly whether the chain has converged.

        A diagnostic nobody reads is not a diagnostic, and R-hat / ESS are only useful against
        their thresholds -- so the verdict is stated rather than left to the reader.
        """
        stats = self.get_posterior_stats(samples)
        print('')
        print(f'{"parameter":<28s}{"mean":>12s}{"sd":>12s}{"3%":>12s}{"97%":>12s}'
              f'{"ess":>10s}{"r_hat":>9s}')
        for name, row in stats.items():
            print(f'{name:<28s}{row["mean"]:>12.4g}{row["sd"]:>12.4g}{row["hdi_3%"]:>12.4g}'
                  f'{row["hdi_97%"]:>12.4g}{row["ess"]:>10.1f}{row["r_hat"]:>9.3f}')

        unconverged = [n for n, r in stats.items()
                       if not np.isfinite(r['r_hat']) or r['r_hat'] > 1.01]
        if unconverged:
            print(f'WARNING: r_hat > 1.01 for {unconverged} -- the walkers have not mixed. '
                  f'Run more steps before trusting the posterior.')
        else:
            print('All parameters have r_hat <= 1.01 (walkers mixed).')
        return stats

    def plot_autocorrelation(self, samples, num_params=None):
        """One autocorrelation-vs-lag panel per parameter, every walker overlaid.

        The +-0.1 guides are what makes the plot readable: a chain whose autocorrelation has
        decayed inside them by the end of the trace is producing near-independent draws, and one
        that has not is still exploring. Returns True when every walker is inside the band over
        the last fifth of the lags, so a caller can act on it rather than only look at it.
        """
        if self.rank != 0:
            return None
        samples = np.asarray(samples, dtype=float)
        if num_params is None:
            num_params = samples.shape[2]

        fig, axes = plt.subplots(num_params, figsize=(10, 2 * num_params), sharex=True,
                                 squeeze=False)
        all_bounded = True
        labels = self._param_labels(num_params)
        autocorr = None
        for idx in range(num_params):
            ax = axes[idx][0]
            for walker in range(samples.shape[1]):
                autocorr = emcee.autocorr.function_1d(samples[:, walker, idx])
                ax.plot(autocorr, alpha=0.3)
                window_size = max(1, int(0.2 * len(autocorr)))
                if np.any(np.abs(autocorr[-window_size:]) > 0.1):
                    all_bounded = False

            ax.axhline(y=0, color='k', linestyle='--', alpha=0.7)
            ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
            ax.axhline(y=-0.1, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
            ax.set_ylabel(f'${labels[idx]}$')
            if autocorr is not None:
                ax.set_xlim(0, len(autocorr))

        axes[-1][0].set_xlabel('Lag')
        fig.tight_layout()
        fig.savefig(os.path.join(self.plot_dir,
                                 f'mcmc_autocorrelation_{self.file_name_prefix}_'
                                 f'{self.param_id_obs_file_prefix}.pdf'))
        plt.close(fig)
        return all_bounded

    def plot_chain_avg(self, samples=None, window_size=10):
        """Running mean of each walker, one panel per parameter.

        Convergence shows up here as the walkers' running means coming together and flattening;
        a walker whose mean is still moving has not finished exploring, which a corner plot of
        the pooled chain hides by averaging it away.
        """
        if self.rank != 0:
            return None
        if samples is None:
            chain = self.get_mcmc_samples()
            if chain is None:
                return None
            _, samples, _ = chain
        samples = np.asarray(samples, dtype=float)

        num_steps, num_chains, num_params = samples.shape
        if window_size >= num_steps:
            print(f'Warning: chain-average window {window_size} is not shorter than the '
                  f'{num_steps} steps available; skipping the chain average plot.')
            return None

        fig, axes = plt.subplots(num_params, figsize=(10, 2 * num_params), sharex=True,
                                 squeeze=False)
        window = np.ones(window_size) / window_size
        labels = self._param_labels(num_params)
        for idx in range(num_params):
            ax = axes[idx][0]
            for chain_idx in range(num_chains):
                moving_avg = np.convolve(samples[:, chain_idx, idx], window, mode='valid')
                ax.plot(np.arange(len(moving_avg)) + window_size - 1, moving_avg, alpha=0.5)
            ax.set_ylabel(f'${labels[idx]}$')

        axes[-1][0].set_xlabel('Step')
        fig.tight_layout()
        fig.savefig(os.path.join(self.plot_dir,
                                 f'mcmc_chain_average_{self.file_name_prefix}_'
                                 f'{self.param_id_obs_file_prefix}.pdf'))
        plt.close(fig)
        return True

    # -----------------------------------------------------------------------
    # Posterior-predictive plots (issue #367)
    # -----------------------------------------------------------------------
    # seaborn is imported lazily inside each of these. It is a CA dependency, but paramID is
    # imported by every calibration run and none of these are on that path, so there is no
    # reason to pay the import for a run that never plots a posterior.

    def get_prior_pdf(self, param_idx, x_values):
        """The prior density for one parameter, over ``x_values``, normalised on that range.

        Derived from CA's own ``get_lnprior_from_params`` rather than reimplemented. The priors
        are independent per parameter, so evaluating the log prior along one axis with the others
        held at their best-fit values recovers that parameter's prior up to a constant, and the
        constant falls out in the normalisation.

        That matters because the prior vocabulary is not simply uniform/exponential/normal any
        more: params_for_id carries ``prior_mean``, ``prior_std``, ``prior_origin`` and
        ``prior_scale``, and an ``unbounded`` flag that suppresses the range truncation. #367
        restated the *old* defaults here (lambda=1, sigma=(max-min)/6, mu=(max+min)/2), so a
        plotted prior would have silently disagreed with the one actually being sampled.
        """
        engine = mcmc_object if self.mcmc_instead else self.param_id
        x_values = np.asarray(x_values, dtype=float)

        centre = getattr(engine, 'best_param_vals', None)
        if centre is None:
            mins = np.asarray(engine.param_id_info['param_mins'], dtype=float)
            maxs = np.asarray(engine.param_id_info['param_maxs'], dtype=float)
            centre = 0.5 * (mins + maxs)
        centre = np.asarray(centre, dtype=float).copy()

        lnprior = np.empty_like(x_values)
        for idx, value in enumerate(x_values):
            trial = centre.copy()
            trial[param_idx] = value
            lnprior[idx] = engine.get_lnprior_from_params(trial)

        finite = np.isfinite(lnprior)
        pdf = np.zeros_like(x_values)
        if not np.any(finite):
            return pdf
        # Subtract the max before exponentiating: the log prior is unnormalised, so its absolute
        # level is arbitrary and can overflow exp() outright.
        pdf[finite] = np.exp(lnprior[finite] - np.max(lnprior[finite]))
        area = integrate_trapezoid(pdf, x_values)
        if area > 0:
            pdf /= area
        return pdf

    def _posterior_predictive_values(self, flat_samples, n_sims=50):
        """Re-simulate ``n_sims`` posterior draws and collect each observable's value.

        Returns ``{name_for_plotting: {experiment_idx: [values], 'exp_data': [values]}}`` -- the
        model's predictive distribution per feature, alongside the measurements it is answerable
        to.
        """
        sim_obj = mcmc_object if self.mcmc_instead else self.param_id
        names = self.obs_info['names_for_plotting']
        values = {name: {} for name in names}

        flat_samples = np.asarray(flat_samples, dtype=float)
        n_actual = int(min(n_sims, len(flat_samples)))
        if n_actual == 0:
            return values
        sample_indices = np.random.choice(len(flat_samples), n_actual, replace=False)

        for count, sample_idx in enumerate(sample_indices, start=1):
            _, obs_list = sim_obj.get_cost_and_obs_from_params(flat_samples[sample_idx, :],
                                                               reset=True)
            subexp_count = 0
            for exp_idx in range(self.protocol_info['num_experiments']):
                for sub_idx in range(self.protocol_info['num_sub_per_exp'][exp_idx]):
                    if subexp_count >= len(obs_list) or obs_list[subexp_count] is None:
                        subexp_count += 1
                        continue
                    obs_proc = sim_obj.get_obs_output_dict(obs_list[subexp_count])
                    subexp_count += 1

                    for obs_idx, name in enumerate(names):
                        if (self.obs_info['experiment_idxs'][obs_idx] != exp_idx
                                or self.obs_info['subexperiment_idxs'][obs_idx] != sub_idx):
                            continue
                        value = self._predictive_value(obs_proc, obs_idx)
                        if value is not None:
                            values[name].setdefault(exp_idx, []).append(value)

            sim_obj.sim_helper.reset_and_clear()
            print(f'Processed {count}/{n_actual} posterior samples for the predictive plots.')

        self._add_measured_values(values)
        return values

    def _predictive_value(self, obs_proc, obs_idx):
        """One observable's scalar out of a simulated obs dict, by its data_type."""
        data_type = self.obs_info['data_types'][obs_idx]
        try:
            if data_type == 'constant':
                return obs_proc['const'][obs_idx]
            if data_type == 'series':
                return np.max(obs_proc['series'][obs_idx])
            if data_type == 'frequency':
                return obs_proc['amp'][obs_idx]
        except (IndexError, KeyError, TypeError):
            return None
        return None

    def _add_measured_values(self, values):
        """Add the measured data each feature is answerable to, under 'exp_data'."""
        for obs_idx, name in enumerate(self.obs_info['names_for_plotting']):
            data_type = self.obs_info['data_types'][obs_idx]
            measured = values[name].setdefault('exp_data', [])
            if data_type == 'constant':
                # A constant scored against a distribution already *is* samples -- use them as
                # given. Otherwise the observation is a mean and a std, so draw from it and the
                # comparison stays distribution against distribution rather than against a line.
                params = self.obs_info['ground_truth_prob_dist_params'][obs_idx]
                if isinstance(params, dict) and 'data_points' in params:
                    measured.extend(np.asarray(params['data_points'], dtype=float))
                else:
                    mean = self.obs_info['ground_truth_const'][obs_idx]
                    std = self.obs_info['std_const_vec'][obs_idx]
                    measured.extend(np.random.normal(mean, std, 20))

    def save_posterior_predictions(self, values):
        """Write the predictive values to posterior_predictions.csv, long-format.

        The plots below are a view of this; the csv is what someone re-plots or re-analyses
        from without paying for the simulations again.
        """
        rows = []
        for feature, by_experiment in values.items():
            for key, vals in by_experiment.items():
                kind = 'experimental' if key == 'exp_data' else 'simulated'
                for value in vals:
                    rows.append({'feature': feature, 'experiment_idx': key,
                                 'value': value, 'data_type': kind})
        path = os.path.join(self.output_dir, 'posterior_predictions.csv')
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f'Saved posterior predictions to {path}')
        return path

    def plot_boxplots_for_predictions(self, flat_samples, n_sims=50, show_points=True):
        """Violin + box + jittered points per feature: the model's predictive spread against
        the measurements, one figure per feature, plus a summary grid.

        This is the plot that answers "does the calibrated model reproduce the data, and with
        what spread" -- which a best-fit line cannot show.
        """
        if self.rank != 0:
            return None
        import seaborn as sns

        values = self._posterior_predictive_values(flat_samples, n_sims=n_sims)
        self.save_posterior_predictions(values)

        written = []
        for feature, by_experiment in values.items():
            ordered_keys = sorted(by_experiment.keys(), key=lambda k: str(k))
            series, labels, colors = [], [], []
            for key in ordered_keys:
                if not by_experiment[key]:
                    continue
                series.append(by_experiment[key])
                if key == 'exp_data':
                    labels.append('Experimental')
                    colors.append('red')
                else:
                    labels.append(self._experiment_label(key))
                    colors.append(self._experiment_color(key))
            if not series:
                continue

            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            sns.violinplot(data=series, ax=ax, palette=colors, cut=3, inner='box',
                           saturation=0.8)
            for idx, collection in enumerate(ax.collections):
                if idx < len(series):
                    collection.set_alpha(0.35)
                    collection.set_edgecolor('none')

            for idx, vals in enumerate(series):
                mean_v, std_v = np.mean(vals), np.std(vals)
                ax.scatter(idx, mean_v, marker='D', color='white', edgecolor='black', s=30,
                           zorder=4)
                spread = np.max(vals) - np.min(vals)
                ax.text(idx, np.max(vals) + 0.05 * spread,
                        fr'${mean_v:.2g} \pm {std_v:.2g}$', ha='center', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5, ec='none'))
                if show_points:
                    ax.scatter(np.random.normal(idx, 0.04, size=len(vals)), vals,
                               color='black', s=5, alpha=0.2, zorder=2)

            obs_idx = self.obs_info['names_for_plotting'].index(feature)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=15)
            ax.set_ylabel(f"{feature} ({self.obs_info['units'][obs_idx]})")
            ax.set_title(feature)
            sns.despine(ax=ax)
            fig.tight_layout()
            # Sanitised, not feature.replace(' ', '_'): a name_for_plotting is LaTeX-ish
            # (u_{A_{R}}), and braces and slashes do not survive as a filename (#167).
            # Imported here rather than at module level: sobolSA pulls in SALib, which a
            # calibration-only install need not have.
            from libcuflynx.sensitivity_analysis.sobolSA import sanitize_for_filename

            path = os.path.join(self.plot_dir, f'posterior_{sanitize_for_filename(feature)}.png')
            fig.savefig(path, dpi=300)
            plt.close(fig)
            written.append(path)

        # Once, after every feature -- #367 called this inside the loop, redrawing the whole
        # grid once per feature and keeping only the last.
        self.plot_distribution_grid(values)
        return written

    def plot_distribution_grid(self, values):
        """One figure of KDE panels, model posterior against measurement, for every feature."""
        if self.rank != 0:
            return None
        import seaborn as sns

        features = list(self.obs_info['names_for_plotting'])
        if not features:
            return None
        cols = min(3, len(features))
        rows = (len(features) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4), squeeze=False)
        flat_axes = axes.flatten()

        for idx, feature in enumerate(features):
            ax = flat_axes[idx]
            by_experiment = values.get(feature, {})
            model_vals = [v for key, vals in by_experiment.items() if key != 'exp_data'
                          for v in vals]
            self._draw_density(ax, model_vals, 'Model posterior', '#1f77b4')
            self._draw_density(ax, by_experiment.get('exp_data', []), 'Experimental', '#d62728')

            ax.set_title(feature, fontweight='bold')
            ax.set_xlabel(f"Value ({self.obs_info['units'][idx]})")
            ax.set_ylabel('Density')
            ax.legend(fontsize=8, frameon=False)
            sns.despine(ax=ax)

        for empty in range(len(features), len(flat_axes)):
            flat_axes[empty].axis('off')

        fig.tight_layout()
        path = os.path.join(self.plot_dir, 'all_features_kde_grid.png')
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path

    @staticmethod
    def _draw_density(ax, data, label, color):
        """A KDE curve, falling back to a histogram when the data has no spread to smooth."""
        data = np.asarray(list(data), dtype=float)
        if data.size < 2:
            return
        try:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(data)
            pad = 0.5 * np.std(data)
            grid = np.linspace(np.min(data) - pad, np.max(data) + pad, 200)
            ax.plot(grid, kde(grid), color=color, lw=2, label=label)
        except (np.linalg.LinAlgError, ValueError):
            # gaussian_kde needs a non-singular covariance: identical samples raise here.
            ax.hist(data, bins=50, density=True, alpha=0.2, color=color, label=label)

    def _experiment_label(self, exp_idx):
        labels = self.protocol_info.get('experiment_labels') or []
        return labels[exp_idx] if exp_idx < len(labels) else f'Exp {exp_idx}'

    def _experiment_color(self, exp_idx):
        colors = self.protocol_info.get('experiment_colors') or []
        return colors[exp_idx] if exp_idx < len(colors) else f'C{exp_idx}'

    def _param_labels(self, num_params=None):
        """Parameter names for the diagnostic tables, falling back to indices."""
        names = None
        info = getattr(self, 'param_id_info', None)
        if isinstance(info, dict):
            names = info.get('param_names_for_plotting')
        if names is None or (num_params is not None and len(names) != num_params):
            return [f'param_{idx}' for idx in range(num_params or 0)]
        return list(names)

    def calculate_geweke_convergence(self, samples):
        d = diagnostics.Diagnostics()
        acceptable = d.geweke(samples, first=0.3, last=0.5)
        return acceptable

    def run_single_sensitivity(self, do_triples_and_quads):
        self.param_id.run_single_sensitivity(self.output_dir, do_triples_and_quads)

    def __get_prediction_data(self):
        # Currently this function saves all prediction variables for all experiments
        # only for the best_param_vals

        if self.rank !=0:
            return

        time_and_pred_per_exp_list = []
        for exp_idx in self.prediction_info['experiment_idxs']:
            self.param_id.simulate_once(reset=False, only_one_exp=exp_idx)
            tSim = self.param_id.sim_helper.tSim - self.param_id.pre_time
            pred_names = [name for II, name in enumerate(self.prediction_info['names']) if 
                                  self.prediction_info['experiment_idxs'][II] == exp_idx]
            pred_output = np.array(self.param_id.sim_helper.get_results(pred_names))
                    
            time_and_pred_per_exp_list.append(np.concatenate((tSim.reshape(1, -1), 
                                                         pred_output[:, 0, :])))
        return time_and_pred_per_exp_list

    def save_prediction_data(self):
        if self.rank !=0:
            return
        if getattr(self.param_id, 'emulates_features', False):
            print('Prediction variables are not saved when use_emulator is set: they are '
                  'traces, and the emulator predicts the scalar observable features only.')
            return
        if self.prediction_info['names'] is not None:
            print('Saving prediction data')
            time_and_pred_per_exp_list = self.__get_prediction_data()

            #save the prediction output
            for exp_idx in range(len(time_and_pred_per_exp_list)):
                time_and_pred = time_and_pred_per_exp_list[exp_idx]
                np.save(os.path.join(self.output_dir, f'prediction_variable_data_exp_{exp_idx}'), 
                        time_and_pred)
                
            # also save the prediction variable names to csv
            with open(os.path.join(self.output_dir, 'prediction_variable_names.csv'), 'w') as wf:
                for name in self.prediction_info['names']:
                    wf.write(name + '\n')
            
            print('prediction data saved')

        else:
            print(f'prediction variables have not been defined, if you want to save predicition variables,',
                  f'create a prediction_items entry in the obs_data.json file')

        return

    def set_bayesian_parameters(self, n_calls, n_initial_points, acq_func, random_state, acq_func_kwargs={}):
        """Configure the Bayesian optimiser.

        Args:
            n_calls: Total number of objective evaluations.
            n_initial_points: Number of random initial points before fitting.
            acq_func: Acquisition function name (e.g. ``'EI'``, ``'LCB'``).
            random_state: Seed for reproducibility.
            acq_func_kwargs: Extra keyword args for the acquisition function.
        """
        self.param_id.set_bayesian_parameters(n_calls, n_initial_points, acq_func, random_state,
                                              acq_func_kwargs=acq_func_kwargs)

    def close_simulation(self):
        """Release the underlying simulation resources."""
        if self.mcmc_instead:
            mcmc_object.close_simulation()
        else:
            self.param_id.close_simulation()



    def get_best_param_vals(self):
        """Return the best-fit parameter vector (ndarray), or None if not yet run."""
        if self.mcmc_instead:
            return mcmc_object.best_param_vals
        else:
            return self.param_id.best_param_vals

    def get_param_names(self):
        """Return the list of identified parameter names (order matches the param vector)."""
        if self.mcmc_instead:
            return mcmc_object.param_id_info["param_names"]
        else:
            return self.param_id.param_id_info["param_names"]

    def get_param_importance(self):
        """Return per-parameter importance scores (computed during sensitivity step)."""
        return self.param_id.param_importance

    def get_collinearity_idx(self):
        """Return the collinearity index of the identified parameter set."""
        return self.param_id.collinearity_idx

    def get_collinearity_idx_pairs(self):
        """Return pairwise collinearity indices for the identified parameters."""
        return self.param_id.collinearity_idx_pairs

    def get_pred_param_importance(self):
        """Return parameter importance scores for the prediction quantities."""
        return self.param_id.pred_param_importance

    def get_pred_collinearity_idx_pairs(self):
        """Return pairwise collinearity indices for the prediction quantities."""
        return self.param_id.pred_collinearity_idx_pairs

    def remove_params_by_idx(self, param_idxs_to_remove):
        """Drop parameters from the identification set by index."""
        self.__set_and_save_param_names(idxs_to_ignore=param_idxs_to_remove)
        if self.mcmc_instead:
            mcmc_object.remove_params_by_idx(param_idxs_to_remove)
        else:
            self.param_id.remove_params_by_idx(param_idxs_to_remove)

    def remove_params_by_name(self, param_names_to_remove):
        """Drop parameters from the identification set by name."""
        param_idxs_to_remove = []
        if self.mcmc_instead:
            num_params = mcmc_object.num_params
        else:
            num_params = self.param_id.num_params

        for II in range(num_params):
            if self.param_id_info["param_names"][II] in param_names_to_remove:
                param_idxs_to_remove.append(II)

        self.remove_params_by_idx(param_idxs_to_remove)

    def postprocess_predictions(self):
        # TODO redo this for new prediction_info in obs_data.json 
        # TODO This should be straight forward
        if self.prediction_info['names'] == None:
            print('no prediction variables, not plotting predictions')
            return 0
        m3_to_cm3 = 1e6
        Pa_to_kPa = 1e-3

        flat_samples, _, _ = self.get_mcmc_samples()
        # this array is of size (num_pred_var, num_samples,
        if self.DEBUG:
            n_sims = 6
        else:
            n_sims = 5 # 20

        pred_list_of_arrays = mcmc_object.calculate_pred_from_posterior_samples(flat_samples, n_sims=n_sims)
        # idxs of pred_list_of_arrays are [exp_idx][sim_idx, pred_idx, time_idx]
        # also get best fit predictions
        best_param_vals = self.get_best_param_vals()

        save_list = []
        for pred_idx in range(len(self.prediction_info['names'])):
            exp_idx = self.prediction_info['experiment_idxs'][pred_idx]
            pred_array = pred_list_of_arrays[pred_idx]
            tSim = self.protocol_info['tSims_per_exp'][exp_idx].flatten()

                        

            fig, axs = plt.subplots()

            #TODO I should include conversion in the prediction_info and use it here
            # also then the units entry can be a unit suitable for plotting
            if self.prediction_info['units'][pred_idx] == 'm3_per_s':
                conversion = m3_to_cm3
                unit_for_plot = '$cm^3/s$'
            elif self.prediction_info['units'][pred_idx] == 'm_per_s':
                conversion = 1.0
                unit_for_plot = '$m/s$'
            elif self.prediction_info['units'][pred_idx] == 'm3':
                conversion = m3_to_cm3
                unit_for_plot = '$cm^3$'
            elif self.prediction_info['units'][pred_idx] == 'J_per_m3':
                conversion = Pa_to_kPa
                unit_for_plot = '$kPa$'
            else:
                conversion = 1.0
                unit_for_plot = f'${self.prediction_info["units"][pred_idx]}$'

            # first plot all arrays on one plot
            fig, axs = plt.subplots()
            for sample_idx in range(pred_array.shape[0]):
                axs.plot(tSim, conversion*pred_array[sample_idx, pred_idx, :], alpha=0.5)
            axs.set_xlabel('Time [$s$]', fontsize=14)
            axs.set_ylabel(f'${self.prediction_info["names_for_plotting"][pred_idx]}$ [{unit_for_plot}]', fontsize=14)
            axs.set_xlim(min(tSim), max(tSim))
            plt.savefig(os.path.join(self.plot_dir,
                                    f'prediction_{self.file_name_prefix}_'
                                    f'{self.param_id_obs_file_prefix}_pred_var_{pred_idx}_all.png'), dpi=500)
            
            # close the figure
            plt.close()
            
            fig, axs = plt.subplots()

            # calculate mean and std of the ensemble
            pred_mean = np.mean(pred_array[:, pred_idx, :], axis=0)
            pred_std = np.std(pred_array[:, pred_idx, :], axis=0)
            # also get the best fit predictions for plotting
            pred_best_fit = mcmc_object.get_pred_array_from_params_per_exp(best_param_vals, exp_idx)[pred_idx, :]

            # get idxs of max min and mean prediction to plot std bars
            idxs_to_plot_std = [np.argmax(pred_mean), np.argmin(pred_mean),
                                np.argmin(np.abs(pred_mean - np.mean(pred_mean)))]
            # TODO put units in prediction file and use it here
            axs.set_xlabel('Time [$s$]', fontsize=14)
            axs.set_ylabel(f'${self.prediction_info["names_for_plotting"][pred_idx]}$ [{unit_for_plot}]', fontsize=14)
            # for sample_idx in range(pred_array.shape[1]):

            # axs.plot(tSim, conversion*pred_mean, 'b', label='mean', linewidth=1.5)
            axs.plot(tSim, conversion*pred_best_fit, 'b', label='best_fit', linewidth=1.5)
            axs.errorbar(tSim[idxs_to_plot_std], conversion*pred_mean[idxs_to_plot_std],
                                yerr=conversion*pred_std[idxs_to_plot_std], ecolor='b', fmt='^', capsize=6, zorder=3)
            axs.set_xlim(min(tSim), max(tSim))
            # z_star = 1.96 for 95% confidence interval. margin_of_error=z_star*std
            z_star = 1.96
            margin_of_error = z_star * pred_std
            conf_ival_up = pred_mean + margin_of_error
            conf_ival_down = pred_mean - margin_of_error
            axs.plot(tSim, conversion*conf_ival_up, 'r--', label='95% CI', linewidth=1.2)
            axs.plot(tSim, conversion*conf_ival_down, 'r--', linewidth=1.2)
            axs.legend()
            # y_max = 1.2*max(conversion*conf_ival_up)
            # axs.set_ylim(ymin=0.0, ymax=y_max)
            # save prediction value, std, and CI of for max, min, and mean
            for idx in idxs_to_plot_std:
                save_list.append(pred_mean[idx])
                save_list.append(pred_std[idx])
                save_list.append(conf_ival_up[idx])
                save_list.append(conf_ival_down[idx])

            # save prediction value, std, and CI of for max, min, and mean
            pred_save_array = conversion*np.array(save_list)
            np.save(os.path.join(self.output_dir, f'prediction_vals_std_ci_{pred_idx}.npy'), pred_save_array)

            plt.savefig(os.path.join(self.plot_dir,
                                    f'prediction_{self.file_name_prefix}_'
                                    f'{self.param_id_obs_file_prefix}_pred_var_{pred_idx}.eps'))
            plt.savefig(os.path.join(self.plot_dir,
                                    f'prediction_{self.file_name_prefix}_'
                                    f'{self.param_id_obs_file_prefix}_pred_var_{pred_idx}.pdf'))
            plt.savefig(os.path.join(self.plot_dir,
                                    f'prediction_{self.file_name_prefix}_'
                                    f'{self.param_id_obs_file_prefix}_pred_var_{pred_idx}.png'))

        # save param standard deviations
        param_std = np.std(flat_samples, axis=0)
        print(param_std)
        np.save(os.path.join(self.output_dir, 'params_std.npy'), param_std)

def observable_base_label(obs_info, obs_idx):
    """The display label for one data_item -- ``item_name_for_plotting``, before disambiguation.

    Since #466 an item states this itself (defaulting to ``<trace name> (<operation>)``), so this
    no longer composes one out of the parts: composing would spell the operation twice for every
    item that took the default. A module function rather than only a method because more than one
    subsystem has to spell an observable the same way.
    """
    return str(obs_item_labels(obs_info)[obs_idx])


def observable_labels(obs_info):
    """One disambiguated label per data_item, in obs_info order. See ``_observable_label``."""
    bases = [observable_base_label(obs_info, idx) for idx in range(obs_info["num_obs"])]
    counts = {}
    for base in bases:
        counts[base] = counts.get(base, 0) + 1
    labels = []
    for idx, base in enumerate(bases):
        if counts[base] == 1:
            labels.append(base)
        else:
            labels.append(f'{base} [exp {obs_info["experiment_idxs"][idx]}, '
                          f'sub {obs_info["subexperiment_idxs"][idx]}]')
    return labels


def emulated_feature_labels(obs_info):
    """The names of the scalar features an emulator is trained on, in emulator output order.

    ``data_item_name``, not a display label: this is what a stored bundle is checked against on
    reload, and a label may be reworded without changing which feature it names -- which would
    make every existing bundle look stale. The name is unique by construction (#466).
    """
    names = obs_item_names(obs_info)
    return [str(names[obs_idx]) for obs_idx in obs_info["const_idx_to_obs_idx"]]


OFFLINE_PRE_TIME_INIT_STATE_ERROR = (
    "varying initial state (quantity) requires doing it from the actual initial state, so "
    "offline_pre_time can't be used. Reformulate to calibrate wrt a constant parameter rather "
    "than a initial state if offline_pre_time is essential. e.g. rather than setting initial LV "
    "volume, set the total volume and calculate the intial LV volume from that."
)


class ParamID():
    """
    Class for doing parameter identification on opencor models
    """

    #: True once the sim helper is an emulator of the scalar observable features (#333). A class
    #: attribute so it is always readable -- the cost, gradient and observable paths branch on
    #: it, and any of them can be reached on an engine built without the full constructor.
    emulates_features = False

    def __init__(self, model_path, param_id_method,
                 obs_info, param_id_info, protocol_info, prediction_info,
                 solver_info, dt=0.01,
                 optimiser_options=None, do_ad=False,
                 DEBUG=False, model_type=None,
                 operation_funcs_external_path=None, cost_funcs_external_path=None,
                 use_emulator=False, emulator_dir=None, emulator_settings=None):

        self.model_path = model_path
        self.param_id_method = param_id_method
        self.output_dir = None
        self.model_type = model_type

        # Emulator mode (#333). `solver` still names the truth solver -- the one the emulator
        # was trained against, and the one to compare it with -- so this is its own flag.
        self.use_emulator = bool(use_emulator)
        self.emulator_dir = emulator_dir
        self.emulator_settings = dict(emulator_settings or {})
        # Set for real once the helper exists; defined here so anything reading it during
        # construction sees False rather than an AttributeError.
        self.emulates_features = False

        self.solver_info = solver_info
        self.obs_info = obs_info
        self.param_id_info = param_id_info
        self.prediction_info = prediction_info # currently not used
        self.optimiser_options = optimiser_options
        if self.param_id_info is not None:
            self.num_params = len(self.param_id_info["param_names"])
            self.param_norm_obj = Normalise_class(self.param_id_info["param_mins"], self.param_id_info["param_maxs"])

        self.protocol_info = protocol_info

        self.sfp = scriptFunctionParser(
            operation_funcs_external_path=operation_funcs_external_path,
            cost_funcs_external_path=cost_funcs_external_path)

        # The operation/cost funcs are backend-dispatched (#199): the casadi-mode funcs build a
        # symbolic graph (e.g. mean is ``ca.sum(x)/x.numel()``) while the numpy-mode funcs operate
        # on arrays. A casadi_python model needs BOTH -- the casadi funcs for the AD-gradient
        # (do_ad) path, whose operands are casadi symbols, and the numpy funcs for the numeric
        # (gradient-free) cost path, whose operands are numpy arrays. Feeding numpy operands to a
        # casadi func raises ``'numpy.ndarray' has no attribute 'numel'`` (#315), so keep one dict
        # of each and select by ``is_symbolic`` at evaluation time (see get_obs_output_dict /
        # cost_calc). For non-casadi models the two are the same numpy dict.
        self.operation_funcs_dict = self.sfp.get_operation_funcs_dict("numpy")
        self.cost_funcs_dict = self.sfp.get_cost_funcs_dict("numpy")
        if self.model_type == "casadi_python":
            mode = "casadi"
            self.operation_funcs_dict_symbolic = self.sfp.get_operation_funcs_dict("casadi")
            self.cost_funcs_dict_symbolic = self.sfp.get_cost_funcs_dict("casadi")
        else:
            # aadc_python uses numpy for passive (non-tape) cost evaluation; all other model types
            # are numeric only. Alias the symbolic dicts to the numpy ones so the selection below
            # is uniform.
            mode = "numpy"
            self.operation_funcs_dict_symbolic = self.operation_funcs_dict
            self.cost_funcs_dict_symbolic = self.cost_funcs_dict

        # set up opencor simulation
        self.dt = dt
        if self.protocol_info is not None:
            if self.protocol_info['sim_times'][0][0] is not None:
                self.sim_time = self.protocol_info['sim_times'][0][0]
            else:
                self.sim_time = None
            if self.protocol_info['pre_times'][0] is not None:
                self.pre_time = self.protocol_info['pre_times'][0]
            else:
                self.pre_time = None
        else:
            self.sim_time = None
            self.pre_time = None

        if self.sim_time is None:
            if 'sim_time' in self.solver_info:
                self.sim_time = self.solver_info['sim_time']
            else:
                self.sim_time = None
        if self.pre_time is None:
            if 'pre_time' in self.solver_info:
                self.pre_time = self.solver_info['pre_time']
            else:
                self.pre_time = None

        #: Cost evaluations performed on this rank. The deterministic referee for "did these
        #: two runs do the same work": wall-clock on this hardware varies 1.3-2.1x on identical
        #: code (benchmarks/PROFILING.md), so a speedup only reads as throughput once the
        #: evaluation counts are known to match (#344).
        self.num_cost_evals = 0

        self.sim_helper = self.initialise_sim_helper()
        # Cached rather than probed per evaluation: this decides, on the hot path, whether the
        # obs `operation` still has to run. getattr keeps every real backend at False untouched.
        self.emulates_features = bool(getattr(self.sim_helper, 'emulates_features', False))
        if self.emulates_features:
            self._configure_emulator()
        self._protocol_executor = ProtocolExecutor(self.sim_helper)

        # Resolve modifier baselines here and nowhere else: the helper exists and nothing has
        # written a parameter yet. Deriving them later would read values the optimiser had
        # already scaled, and theta would compound every iteration.
        resolve_modifier_baselines(self.param_id_info, self.sim_helper)

        if self.sim_time is not None and self.pre_time is not None:
            self.sim_helper.update_times(self.dt, 0.0, self.sim_time, self.pre_time)
            self.n_steps = int(self.sim_time/self.dt)
        else:
            self.n_steps = None

        offline_pre_time = None
        if self.protocol_info is not None:
            offline_pre_time = self.protocol_info.get('offline_pre_time')
        if offline_pre_time is not None and float(offline_pre_time) > 0:
            offenders = self._params_that_set_initial_states()
            if offenders:
                raise ValueError(
                    OFFLINE_PRE_TIME_INIT_STATE_ERROR
                    + f" Offending calibration parameter(s): {', '.join(offenders)}.")
            # Run the offline warm-up ONCE and reuse its end state as the initial condition for
            # every evaluation's pre_time + sim_time. This is the speed-up offline_pre_time exists
            # for: the settled state is reached once instead of on all ~N cost evaluations.
            #
            # It is only sound because the guard above rejects calibration parameters that set a
            # state's initial value. For every remaining parameter the frozen state is genuinely a
            # constant with respect to the parameters being fitted, so its sensitivity is zero and
            # FSA/AD accumulate d/dp correctly across the per-evaluation pre_time. That was the
            # flaw in the original version (issue #269): with a state-init parameter in the set,
            # the frozen state silently absorbed d(steady state)/d(p) and both AD and FD agreed on
            # the wrong gradient, because both perturbed the same frozen state.
            #
            # pre_time still runs per evaluation and must be long enough for the trial parameters
            # to re-settle from the warm start -- how long varies with the parameters, which is
            # what issue #328 is about.
            offline_pre_time = float(offline_pre_time)
            self.sim_helper.run_offline_pre_and_set_default_state(offline_pre_time)

        if self.protocol_info is not None:
            self.sim_helper.set_protocol_info(self.protocol_info)

        # initialise
        self.param_init = None
        self.best_param_vals = None
        self.best_cost = np.inf

        # bayesian optimisation constants TODO add more of the constants to this so they can be modified by the user
        # TODO or remove bayesian optimisation, as it is untested
        self.acq_func = 'EI'  # the acquisition function
        self.n_initial_points = 5
        self.acq_func_kwargs = {}
        self.random_state = 1234 # random seed

        # sensitivity
        self.param_importance = None
        self.collinearity_idx = None
        self.collinearity_idx_pairs = None
        self.pred_param_importance = None
        self.pred_collinearity_idx_pairs = None

        self.do_ad = do_ad
        if self.emulates_features and self.do_ad:
            # Every analytic arm differentiates the real model, so AD over an emulator would
            # descend a different function than the cost reports. get_gradient answers with
            # finite differences on the emulator instead, which costs nothing here.
            print('use_emulator is set, so do_ad is being turned off: gradients over an '
                  'emulator come from finite differences on its own (near-free) evaluations.')
            self.do_ad = False

        if self.obs_info is not None:
            self.cost_type = self.obs_info["cost_type"]
        else:
            self.cost_type = None
        if mode == "casadi":
            assert_casadi_differentiable(
                self.obs_info, self.cost_type,
                self.operation_funcs_dict_symbolic, self.cost_funcs_dict_symbolic
            )
        # Fail fast on a stale obs_data.json rather than part-way through an optimisation (#304).
        validate_operation_kwargs(self.obs_info, self.operation_funcs_dict)
        validate_cost_kwargs(self.obs_info, self.cost_funcs_dict, self.cost_type)
        self.DEBUG = DEBUG

        # Per (experiment, subexperiment) count of observables with non-zero weight. The sum
        # over all subs equals the divisor applied in get_cost_obs_and_pred_from_params and
        # is the exact factor that recovers summed NLL in get_lnlikelihood_from_params.
        self._num_weighted_obs_by_exp_sub = None
        self._lnlikelihood_denorm_factor = 1.0
        self._refresh_num_weighted_obs_tables()

    def _refresh_num_weighted_obs_tables(self):
        """Rebuild weighted-observable counts from protocol weight maps (call after obs/protocol change).

        ``_lnlikelihood_denorm_factor`` is the total number of weighted observable slots
        across all experiments and subexperiments; it matches the denominator used when
        forming the mean cost in ``get_cost_obs_and_pred_from_params`` for a full run.
        """
        if self.protocol_info is None:
            self._num_weighted_obs_by_exp_sub = None
            self._lnlikelihood_denorm_factor = 1.0
            return
        by_exp_sub = []
        total = 0
        for exp_idx in range(self.protocol_info["num_experiments"]):
            row = []
            for sub_idx in range(self.protocol_info["num_sub_per_exp"][exp_idx]):
                wc = self.protocol_info["scaled_weight_const_from_exp_sub"][exp_idx][sub_idx]
                ws = self.protocol_info["scaled_weight_series_from_exp_sub"][exp_idx][sub_idx]
                wa = self.protocol_info["scaled_weight_amp_from_exp_sub"][exp_idx][sub_idx]
                wp = self.protocol_info["scaled_weight_phase_from_exp_sub"][exp_idx][sub_idx]
                n = int(
                    np.sum(wc != 0)
                    + np.sum(ws != 0)
                    + np.sum(wa != 0)
                    + np.sum(wp != 0)
                )
                row.append(n)
                total += n
            by_exp_sub.append(row)
        self._num_weighted_obs_by_exp_sub = by_exp_sub
        self._lnlikelihood_denorm_factor = float(total) if total > 0 else 1.0

    def _params_that_set_initial_states(self):
        """Calibration parameters that set a state's initial value.

        These are incompatible with an offline warm-up: the offline pass freezes one state and
        every evaluation restarts from it, so a parameter whose only role is to initialise a state
        never enters the simulation at all. Its gradient is then exactly zero and it is silently
        unrecoverable -- measured on 3compartment, q_lv_init's response to a 25% perturbation
        drops from 8.49% to 0.0000% once an offline warm-up is used.

        Detected two ways: a name that resolves to a state directly, and the ``<state>_init``
        naming convention that ties a constant to a state's initial value (the convention the
        python/CasADi/AADC backends already key on).
        """
        offenders = []
        param_names = (self.param_id_info or {}).get('param_names') or []
        resolve = getattr(self.sim_helper, '_resolve_name', None)
        for name_or_list in param_names:
            names = name_or_list if isinstance(name_or_list, (list, tuple)) else [name_or_list]
            for name in names:
                var_part = str(name).split('/')[-1]
                if var_part.endswith('_init'):
                    offenders.append(str(name))
                    continue
                if resolve is not None:
                    try:
                        if resolve(name)[0] == 'state':
                            offenders.append(str(name))
                    except Exception:
                        pass
        return offenders

    def initialise_sim_helper(self):
        # Get method from solver_info (check both 'solver' and 'method' for backward compatibility)
        solver = self.solver_info.get('solver')
        helper_cls = get_simulation_helper(solver=solver, model_type=self.model_type,
                                           model_path=self.model_path, dt=self.dt, sim_time=self.sim_time,
                                           solver_info=self.solver_info, pre_time=self.pre_time,
                                           use_emulator=self.use_emulator,
                                           emulator_dir=self.emulator_dir,
                                           # None -> the helper takes what the emulator was
                                           # trained with (see _use_time_setting).
                                           out_of_bounds=self.emulator_settings.get(
                                               'out_of_bounds'))
        return helper_cls

    def _configure_emulator(self):
        """Validate the loaded emulator against this run, and wire its outputs to the obs items.

        Everything here is a refusal that has to happen *before* the first evaluation. An
        emulator asked to answer for a model, parameter set or protocol it was not trained
        against does not fail -- it answers, about something else -- and the resulting costs
        and Sobol indices carry no sign of it (#333).
        """
        from libcuflynx.emulators.emulator_bundle import fingerprint
        bundle = self.sim_helper.bundle

        # Zero-weighted non-scalars are exempt: they are not in the cost, so the emulator is
        # never asked for them. The rule lives in emulator_bundle so this and the trainer's
        # copy cannot drift -- they already had, and a recorded trace carried at weight 0 for
        # plotting was refused here after the trainer had accepted it.
        from libcuflynx.emulators.emulator_bundle import weighted_non_scalar_obs

        bad = weighted_non_scalar_obs(self.obs_info)
        if bad:
            raise ValueError(
                f'use_emulator is set, but obs_data.json has data_type(s) '
                f'{sorted(set(bad.values()))} at data_item index(es) {sorted(bad)}. The emulator '
                f'predicts scalar data_item features only; those need the full simulated trace '
                f'("series") or its FFT ("frequency"). Emulating series outputs is not supported '
                f'yet -- give them weight 0 if they are only there to be plotted, run with '
                f'use_emulator: false, or drop those items.')

        bundle.check_matches(
            fingerprint(self.param_id_info, self.obs_info, self.protocol_info, self.model_path),
            param_entry_labels=param_entry_labels(self.param_id_info),
            feature_labels=emulated_feature_labels(self.obs_info))
        bundle.check_quality(self._use_time_setting('min_r2', 0.9, bundle))
        self.sim_helper.set_obs_map(self.obs_info['const_idx_to_obs_idx'],
                                    num_obs=self.obs_info['num_obs'])

    def _use_time_setting(self, name, default, bundle=None):
        """An ``emulator_settings`` value that is read when the emulator is *used*.

        Most of that block only matters while training, and only training is given
        it. ``min_r2`` and ``fd_rel_step`` are the exceptions: they are read again
        by a calibration / SA / UQ run that evaluates the emulator -- and such a run
        is configured by its own settings, which need say nothing about emulation.
        Falling straight back to the schema default there meant a user who set
        ``min_r2: 0.88`` was refused at 0.9 and told 0.9 was "the configured min_r2".

        So the emulator carries its own configuration: the value comes from this
        run's ``emulator_settings`` when it names one, else from the block saved in
        the bundle when it was trained, else the default. An explicit setting still
        wins, which is what lets one run accept a lower-quality emulator without
        retraining it.
        """
        if name in (self.emulator_settings or {}):
            return self.emulator_settings[name]
        bundle = bundle if bundle is not None else getattr(self.sim_helper, 'bundle', None)
        trained_with = (getattr(bundle, 'meta', None) or {}).get('settings') or {}
        if name in trained_with:
            return trained_with[name]
        return default

    def add_user_operation_func(self, func):
        if self.model_type == "casadi_python" and not is_circulatory_differentiable(func):
            raise ValueError(
                f"User operation {func.__name__!r} must be decorated with @differentiable for casadi_python mode."
            )
        # Register into both the numeric and symbolic dicts so the func is available on whichever
        # path evaluates it (a @differentiable func dispatches through the backend in either mode).
        self.operation_funcs_dict = self.sfp.add_user_operation_func(self.operation_funcs_dict, func)
        if self.operation_funcs_dict_symbolic is not self.operation_funcs_dict:
            self.sfp.add_user_operation_func(self.operation_funcs_dict_symbolic, func)

    def add_user_cost_func(self, func):
        if self.model_type == "casadi_python" and not is_circulatory_differentiable(func):
            raise ValueError(
                f"User cost function {func.__name__!r} must be decorated with @differentiable for casadi_python mode."
            )
        self.cost_funcs_dict = self.sfp.add_user_cost_func(self.cost_funcs_dict, func)
        if self.cost_funcs_dict_symbolic is not self.cost_funcs_dict:
            self.sfp.add_user_cost_func(self.cost_funcs_dict_symbolic, func)
    
    def set_best_param_vals(self, best_param_vals):
        self.best_param_vals = best_param_vals
    
    def set_param_names(self, param_names):
        self.param_id_info["param_names"] = param_names
        self.num_params = len(self.param_id_info["param_names"])
    
    def set_param_id_info(self, param_id_info):
        self.param_id_info = param_id_info
        self.num_params = len(self.param_id_info["param_names"])
        self.param_norm_obj = Normalise_class(self.param_id_info["param_mins"], self.param_id_info["param_maxs"])
        # The constructor resolves baselines when param_id_info is already known; entry points
        # that set it afterwards resolve here instead. Idempotent, and still before any
        # parameter has been written, which is the property that stops theta compounding.
        if getattr(self, 'sim_helper', None) is not None:
            resolve_modifier_baselines(self.param_id_info, self.sim_helper)
        # Re-check the emulator against the parameters it is now being asked about: the
        # programmatic API sets these after construction, and an emulator trained for a
        # different set (or a different box) would answer anyway.
        if self.emulates_features:
            self._configure_emulator()

    
    def set_protocol_info(self, protocol_info):
        self.protocol_info = protocol_info
        # set the protocol_info in the sim_helper so that the protocol traces can be accessed.
        self.sim_helper.set_protocol_info(self.protocol_info)
        self._refresh_num_weighted_obs_tables()

    def set_prediction_info(self, prediction_info):
        self.prediction_info = prediction_info
    
    def set_obs_info(self, obs_info):
        self.obs_info = obs_info
        self.cost_type = self.obs_info["cost_type"]
        validate_operation_kwargs(self.obs_info, self.operation_funcs_dict)
        validate_cost_kwargs(self.obs_info, self.cost_funcs_dict, self.cost_type)
        self._refresh_num_weighted_obs_tables()
        # As in set_param_id_info: the observables just changed, and the emulator's outputs
        # are tied to the ones it was trained on, feature for feature.
        if self.emulates_features:
            self._configure_emulator()

    def set_optimiser_options(self, optimiser_options):
        self.optimiser_options = optimiser_options

    def set_param_id_method(self, param_id_method):
        self.param_id_method = param_id_method
    
    def remove_params_by_idx(self, param_idxs_to_remove):
        if len(param_idxs_to_remove) > 0:
            self.param_id_info["param_names"] = [self.param_id_info["param_names"][II] for II in range(self.num_params) if II not in param_idxs_to_remove]
            self.num_params = len(self.param_id_info["param_names"])
            if self.best_param_vals is not None:
                self.best_param_vals = np.delete(self.best_param_vals, param_idxs_to_remove)
            self.param_id_info["param_mins"] = np.delete(self.param_id_info["param_mins"], param_idxs_to_remove)
            self.param_id_info["param_maxs"] = np.delete(self.param_id_info["param_maxs"], param_idxs_to_remove)
            self.param_id_info["param_prior_types"] = np.delete(self.param_id_info["param_prior_types"], param_idxs_to_remove)
            # Kept in step with the types, or every remaining parameter would read the
            # hyper-parameters of whichever one used to sit at its index.
            if self.param_id_info.get("param_prior_params") is not None:
                self.param_id_info["param_prior_params"] = [
                    p for II, p in enumerate(self.param_id_info["param_prior_params"])
                    if II not in param_idxs_to_remove
                ]
            if self.param_id_info.get("param_unbounded") is not None:
                self.param_id_info["param_unbounded"] = np.delete(
                    self.param_id_info["param_unbounded"], param_idxs_to_remove)
            self.param_norm_obj = Normalise_class(self.param_id_info["param_mins"], self.param_id_info["param_maxs"])
            self.param_init = None

    def save_all_outputs_per_experiment(self, param_vals, suffix=""):
        """
        Simulate each experiment with ``param_vals`` and save all model variables to NPZ.

        Parameters
        ----------
        param_vals : array-like
            Parameter values to apply before each per-experiment simulation.
        suffix : str
            Inserted before ``.npz`` (e.g. ``"_plot"`` for plot-time dumps).
        """
        if MPI.COMM_WORLD.Get_rank() != 0:
            return
        if self.emulates_features:
            # Not a failure -- the run worked, and this is the one thing an emulator of scalar
            # features genuinely cannot give. Said plainly, at the end of a run, rather than
            # raised from inside a save.
            print(
                "[param_id] the all-outputs npz is not written when use_emulator is set: the "
                "emulator predicts the scalar observable features, not the traces they came "
                "from. Re-run with use_emulator: false for simulated outputs."
            )
            return
        if self.output_dir is None or self.protocol_info is None:
            print(
                "[param_id] WARNING: cannot save all-outputs npz "
                "(output_dir or protocol_info missing)"
            )
            return
        num_experiments = int(self.protocol_info.get("num_experiments", 0) or 0)
        for exp_idx in range(num_experiments):
            try:
                self.simulate_once(
                    param_vals, reset=True, only_one_exp=exp_idx
                )
                all_outputs_dict = self.sim_helper.get_all_results_dict()
                path = os.path.join(
                    self.output_dir,
                    f"all_outputs_with_best_param_vals_exp_{exp_idx}{suffix}.npz",
                )
                np.savez(path, **all_outputs_dict)
                print(f"[param_id] saved {os.path.basename(path)}")
            except Exception as e:
                print(
                    f"[param_id] WARNING: failed to write exp {exp_idx} npz "
                    f"(suffix={suffix!r}): {e}"
                )

    def run(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        num_procs = comm.Get_size()
        
        if rank == 0:
            print(f'Running parameter identification across {num_procs} MPI rank(s)')
            if num_procs == 1:
                print('WARNING Running in serial, are you sure you want to be a snail?')
            # save date as identifier for the param_id
            np.save(os.path.join(self.output_dir, 'date'), date.today().strftime("%d_%m_%Y"))

            # delete history files
            if os.path.exists(os.path.join(self.output_dir, 'best_cost_history.csv')):
                # delete file
                os.remove(os.path.join(self.output_dir, 'best_cost_history.csv'))
            if os.path.exists(os.path.join(self.output_dir, 'best_param_vals_history.csv')):
                os.remove(os.path.join(self.output_dir, 'best_param_vals_history.csv'))

            # write column header for best params
            with open(os.path.join(self.output_dir, 'best_param_vals_history.csv'), 'w') as f:
                wr = csv.writer(f)
                new_array_names = np.char.replace(np.array([list_of_names[0] 
                                    for list_of_names in self.param_id_info["param_names"]]), '/', ' ')
                wr.writerows(new_array_names.reshape(1, -1))

        if rank == 0:
            print('Starting param id run (rank 0 coordinating)')

        # ________ Do parameter identification ________

        # Don't remove the get_init_param_vals, this also checks the parameters names are correct.
        raw_init = self.sim_helper.get_init_param_vals(self.param_id_info["param_names"])
        # One x0 slot per calibrated variable (theta), flat. get_init_param_vals returns a
        # *list* of member values for a multi-name entry, and np.asarray over that ragged
        # structure is a crash in every optimiser's x0 handling -- a grouped row starts at its
        # first member's default (the shared value). A modifier's slot is theta, not a model
        # value, so it starts at the operation's identity (scale -> 1.0), where every target
        # sits at its baseline; a member's raw default there (~1e-8 for a compliance) would be
        # taken as a scale factor.
        self.param_init = apply_modifier_identity_nominals(
            self.param_id_info,
            np.array([v[0] if isinstance(v, (list, tuple)) else v for v in raw_init],
                     dtype=float))

        # The param_modifiers.json written at parse time has baselines: None -- no simulation
        # helper existed yet. Re-save now they are resolved; without baselines the recorded
        # theta is uninterpretable, which is the file's whole purpose.
        save_param_modifiers(self.param_id_info, self.output_dir)

        # C_T min and max was 1e-9 and 1e-5 before

        # Use optimiser classes for all methods
        if self.param_id_method == 'bayesian':
            # Use BayesianOptimiser class
            optimiser = BayesianOptimiser(
                self, self.param_id_info, self.param_norm_obj,
                self.num_params, self.output_dir,
                optimiser_options=self.optimiser_options,
                DEBUG=self.DEBUG,
                acq_func=self.acq_func,
                n_initial_points=self.n_initial_points,
                random_state=self.random_state,
                acq_func_kwargs=self.acq_func_kwargs
            )
            optimiser.run()
            self.best_param_vals = optimiser.best_param_vals
            self.best_cost = optimiser.best_cost

        elif self.param_id_method == 'genetic_algorithm':
            # Use GeneticAlgorithmOptimiser class
            optimiser = GeneticAlgorithmOptimiser(
                self, self.param_id_info, self.param_norm_obj,
                self.num_params, self.output_dir,
                optimiser_options=self.optimiser_options,
                DEBUG=self.DEBUG
            )
            optimiser.run()
            self.best_param_vals = optimiser.best_param_vals
            self.best_cost = optimiser.best_cost

        elif self.param_id_method in ['CMA-ES', 'CMAES', 'cmaes']:
            # Use CMAESOptimiser for CMA-ES optimization
            optimiser = CMAESOptimiser(
                self, self.param_id_info, self.param_norm_obj,
                self.num_params, self.output_dir,
                optimiser_options=self.optimiser_options,
                DEBUG=self.DEBUG
            )
            optimiser.run()
            self.best_param_vals = optimiser.best_param_vals
            self.best_cost = optimiser.best_cost

        elif self.param_id_method == 'sp_minimize':
            # Use SciPyMinimizeOptimiser for gradient-based optimization
            optimiser = SciPyMinimizeOptimiser(
                self, self.param_id_info, self.param_norm_obj,
                self.num_params, self.output_dir,
                optimiser_options=self.optimiser_options,
                do_ad=self.do_ad, DEBUG=self.DEBUG
            )
            optimiser.run()
            self.best_param_vals = optimiser.best_param_vals
            self.best_cost = optimiser.best_cost
            self.init_gradient = optimiser.init_gradient
            self.best_gradient = optimiser.best_gradient

        elif self.param_id_method == 'multi_start_sp_minimize':
            # Multi-start L-BFGS-B: gradient descent from many scattered starts, so a
            # multi-modal cost surface doesn't trap us in the basin of the initial params.
            optimiser = MultiStartSciPyMinimizeOptimiser(
                self, self.param_id_info, self.param_norm_obj,
                self.num_params, self.output_dir,
                optimiser_options=self.optimiser_options,
                do_ad=self.do_ad, model_type=self.model_type, DEBUG=self.DEBUG
            )
            optimiser.run()
            self.best_param_vals = optimiser.best_param_vals
            self.best_cost = optimiser.best_cost
            self.init_gradient = optimiser.init_gradient
            self.best_gradient = optimiser.best_gradient

        else:
            print(f"param_id_method '{self.param_id_method}' is not implemented. Valid options: "
                  f"{list(PARAM_ID_METHODS.keys())}")
            exit()

        if rank == 0:
            print('')
            print(f'{self.param_id_method} is complete')
            # print init params and final params
            print('init params     : {}'.format(self.param_init))
            print('best fit params : {}'.format(self.best_param_vals))
            print('best cost       : {}'.format(self.best_cost))

            self.save_all_outputs_per_experiment(self.best_param_vals, suffix="")

            if self.param_id_method in ['sp_minimize', 'multi_start_sp_minimize'] and \
                    self.init_gradient is not None:
                print('init gradients  : {}'.format(self.init_gradient))
                print('best gradients  : {}'.format(self.best_gradient))

        return
    
    def get_cost_obs_and_pred_from_params(self, param_vals, reset=True, 
                                          only_one_exp=-1, pred_names=None, do_ad=False):
        # Every cost evaluation funnels through here -- get_cost_from_params and
        # get_cost_and_obs_from_params both delegate -- so this is the one place that can count
        # them without each optimiser keeping its own tally (#344).
        self.num_cost_evals = getattr(self, 'num_cost_evals', 0) + 1

        # loop through subexperiments
        if only_one_exp == -1:
            # unless the user wants to just one experiment, reset must be true
            reset = True
            exp_idxs_to_run = list(range(self.protocol_info["num_experiments"]))
        else:
            exp_idxs_to_run = [only_one_exp]

        # TODO: Test AD with multiple subexperiments
        if do_ad:
            reset = False

        if self.emulates_features:
            # The emulator's input is theta itself. By the time the executor calls
            # set_param_vals these values have been expanded to one per model parameter, and a
            # modifier entry's expansion (theta * baseline) is not the theta it was trained on
            # -- so theta is handed over here, before any of that.
            self.sim_helper.set_theta(param_vals)

        # Run the protocol loop via the shared ProtocolExecutor.
        # reset_after_experiment mirrors the original `reset` flag: when do_ad=True
        # (reset=False) the solver state must be preserved across experiments.
        sim_success, results_by_sub, extra_by_sub, _ = self._protocol_executor.run_protocol(
            self.protocol_info,
            id_param_names=self.param_id_info["param_names"],
            # A modifier occupies one slot in the optimiser's vector but names N model
            # parameters, so its slot expands to theta * baseline_i here. Everything else passes
            # through, and set_param_vals pairs N names with N values positionally (#376).
            id_param_vals=expand_modifier_param_vals(self.param_id_info, param_vals),
            result_variables=self.obs_info["operands"],
            extra_result_variables=pred_names,
            exp_indices=exp_idxs_to_run,
            continue_on_failure=False,
            reset_after_experiment=reset,
        )

        if not sim_success:
            print('simulation failed with params...')
            print(param_vals)
            return np.inf, [], []

        # Rebuild flat operands_outputs_list indexed by cumulative subexp_count,
        # preserving None entries for skipped experiments (needed by downstream callers).
        num_experiments = self.protocol_info["num_experiments"]
        num_sub_per_exp = self.protocol_info["num_sub_per_exp"]
        operands_outputs_list = []
        pred_outputs_list = []
        for exp_idx in range(num_experiments):
            for sub_idx in range(num_sub_per_exp[exp_idx]):
                operands_outputs_list.append(
                    results_by_sub.get((exp_idx, sub_idx))
                )
                if pred_names is not None:
                    pred_outputs_list.append(
                        extra_by_sub.get((exp_idx, sub_idx))
                    )

        # Update sim_time / pre_time to the last-run values (preserves existing behaviour).
        if exp_idxs_to_run:
            last_exp = exp_idxs_to_run[-1]
            self.sim_time = self.protocol_info["sim_times"][last_exp][-1]
            self.pre_time = self.protocol_info["pre_times"][last_exp]

        cost = 0.0
        weighted_obs_denominator = 0
        # One table for the whole evaluation, so an item may reference one from an earlier
        # (experiment, sub-experiment) and not just its own (#466). The segments are visited in
        # order, so a reference is backward-only. Declared as a block rather than by assigning
        # the attribute here: `evaluating_segment` gives a *standalone* caller its own fresh
        # table, and it can only tell the two apart if this walk says which it is.
        with self.accumulating_temp_results():
            for exp_idx in exp_idxs_to_run:
                for this_sub_idx in range(num_sub_per_exp[exp_idx]):
                    subexp_count = int(np.sum([num_sub for num_sub in
                                               num_sub_per_exp[:exp_idx]]) + this_sub_idx)

                    sub_cost = self.get_cost_from_operands(
                        operands_outputs_list[subexp_count],
                        exp_idx=exp_idx, sub_idx=this_sub_idx, do_ad=do_ad,
                    )
                    cost += sub_cost
                    if self._num_weighted_obs_by_exp_sub is not None:
                        weighted_obs_denominator += self._num_weighted_obs_by_exp_sub[exp_idx][this_sub_idx]
                    else:
                        wc = self.protocol_info["scaled_weight_const_from_exp_sub"][exp_idx][this_sub_idx]
                        ws = self.protocol_info["scaled_weight_series_from_exp_sub"][exp_idx][this_sub_idx]
                        wa = self.protocol_info["scaled_weight_amp_from_exp_sub"][exp_idx][this_sub_idx]
                        wp = self.protocol_info["scaled_weight_phase_from_exp_sub"][exp_idx][this_sub_idx]
                        weighted_obs_denominator += int(
                            np.sum(wc != 0)
                            + np.sum(ws != 0)
                            + np.sum(wa != 0)
                            + np.sum(wp != 0)
                        )

        # Mean NLL contribution per weighted observable slot (summed raw sub costs / global count).
        if weighted_obs_denominator <= 0:
            weighted_obs_denominator = 1
        cost = cost / float(weighted_obs_denominator)

        return cost, operands_outputs_list, pred_outputs_list

    def get_cost_and_obs_from_params(self, param_vals, reset=True, only_one_exp=-1, do_ad=False):
        cost, obs, _ = self.get_cost_obs_and_pred_from_params(param_vals, reset=reset, only_one_exp=only_one_exp, do_ad=do_ad)
        return cost, obs

    def get_cost_from_params(self, param_vals, reset=True):
        cost = self.get_cost_and_obs_from_params(param_vals, reset=reset)[0]
        return cost
    
    def _is_unbounded(self, idx):
        """Whether parameter ``idx`` was marked unbounded in params_for_id.

        Tolerates a param_id_info without the key -- assembled by hand, or from
        before the column existed -- by answering False, which is the behaviour
        every parameter had then.
        """
        flags = self.param_id_info.get("param_unbounded")
        if flags is None or idx >= len(flags):
            return False
        return bool(flags[idx])

    def _resolved_prior_param(self, idx, prior_type, name):
        """A prior hyper-parameter with its default resolved from the schema.

        A stated value wins; otherwise the schema's default_expr is evaluated
        against this parameter's bounds and its sibling values. Raises when the
        result is unusable -- an unbounded exponential with no scale has nothing
        to decay by, and guessing one would silently invent a prior.
        """
        stated = self._prior_param(idx, name)
        if stated is not None:
            return stated
        bounds = {}
        lo = self.param_id_info["param_mins"][idx]
        hi = self.param_id_info["param_maxs"][idx]
        if np.isfinite(lo):
            bounds['min'] = float(lo)
        if np.isfinite(hi):
            bounds['max'] = float(hi)
        siblings = dict(self.param_id_info.get("param_prior_params", [{}] * (idx + 1))[idx] or {})
        for spec in PARAM_PRIOR_TYPES.get(prior_type, {}).get('params', []):
            siblings.setdefault(spec['name'], spec.get('default'))
        value = prior_param_default(prior_type, name, bounds, siblings)
        if value is None:
            raise ValueError(
                f"'{name}' is needed for the {prior_type} prior on parameter index {idx} "
                f"and could not be derived from the parameter's range. State it in "
                f"params_for_id.")
        return value

    def _prior_param(self, idx, name):
        """One prior hyper-parameter for parameter ``idx``, or its declared default.

        Tolerates a param_id_info built without ``param_prior_params`` -- assembled by
        hand, or unpickled from before these columns existed -- by falling back to the
        schema default, so an older config keeps the behaviour it had.
        """
        per_param = self.param_id_info.get("param_prior_params")
        if per_param is not None and idx < len(per_param):
            values = per_param[idx] or {}
            if name in values:
                return values[name]
        for meta in PARAM_PRIOR_TYPES.values():
            for spec in meta['params']:
                if spec['name'] == name:
                    return spec['default']
        return None

    def get_lnprior_from_params(self, param_vals):
        lnprior = 0
        for idx, param_val in enumerate(param_vals):
            if self.param_id_info["param_prior_types"] is not None:
                prior_dist = self.param_id_info["param_prior_types"][idx]
            else:
                prior_dist = None

            # An unbounded parameter has no range of its own -- the range in
            # param_id_info was derived from this very prior, purely so the
            # optimiser and the normalisation have a finite box. Truncating the
            # prior at it would re-impose the bounds the user said were absent.
            bounded = not self._is_unbounded(idx)

            if not prior_dist or prior_dist == 'uniform':
                if bounded and (param_val < self.param_id_info["param_mins"][idx] or param_val > self.param_id_info["param_maxs"][idx]):
                    return -np.inf
                else:
                    #prior += 0
                    pass
            
            elif prior_dist == 'exponential':
                if bounded and (param_val < self.param_id_info["param_mins"][idx] or param_val > self.param_id_info["param_maxs"][idx]):
                    return -np.inf
                else:
                    # -(x - origin)/scale. With both left blank this is exactly the
                    # original -lambda*x/max: origin defaults to 0 and scale to
                    # max/lambda. Stating a scale gives the decay a size in the
                    # parameter's own units, which is what makes an unbounded
                    # exponential meaningful -- the original rate is defined
                    # *relative to max*, and an unbounded parameter has no max.
                    origin = self._resolved_prior_param(idx, 'exponential', 'prior_origin')
                    scale = self._resolved_prior_param(idx, 'exponential', 'prior_scale')
                    lnprior += -(param_val - origin) / scale

            elif prior_dist == 'normal':
                if bounded and (param_val < self.param_id_info["param_mins"][idx] or param_val > self.param_id_info["param_maxs"][idx]):
                    return -np.inf
                else:
                    # Defaults when the user states neither: the centre of the range, and a
                    # sixth of it, which puts [min, max] at +/- 3 sigma. Both are now
                    # params_for_id columns (prior_mean / prior_std), because a prior whose
                    # centre is fixed to the middle of the bounds cannot express most of
                    # what a prior is for.
                    # Both defaults come from the schema's default_expr, so the
                    # number used here and the one a UI shows in the blank field
                    # are the same statement rather than two copies of it.
                    std = self._resolved_prior_param(idx, 'normal', 'prior_std')
                    mean = self._resolved_prior_param(idx, 'normal', 'prior_mean')
                    lnprior += -0.5*((param_val - mean)/std)**2

            else:
                # Unreachable via params_for_id, which validates the column against
                # PARAM_PRIOR_TYPES. Kept because falling through silently is what made a
                # mis-spelled prior drop this parameter's range check -- the walker then
                # left [min, max] with a finite lnprior instead of -inf, and nothing said so.
                raise ValueError(
                    f"unknown prior '{prior_dist}' for parameter index {idx}. "
                    f"Valid priors are: {', '.join(sorted(PARAM_PRIOR_TYPES))}."
                )

        return lnprior

    def get_lnlikelihood_lnprior_from_params(self, param_vals, reset=True):
        lnprior = self.get_lnprior_from_params(param_vals)

        if not np.isfinite(lnprior):
            return -np.inf

        lnlikelihood = self.get_lnlikelihood_from_params(param_vals)

        return lnprior + lnlikelihood

    def get_lnlikelihood_lnprior_from_ensemble(self, ensemble):
        """Log-posterior for a whole ensemble, predicting the surrogate once for all of it.

        The saving is entirely in the emulator call. Evaluating a fitted regressor at
        sixty-four points costs barely more than at one -- 84.8 ms against 355 ms on the
        study this was written for, 15x less per point -- because the per-call overhead
        dominates the arithmetic. An ensemble sampler asks for its whole population at
        every step, so it is exactly the caller that can pay that overhead once.

        Everything after the prediction is unchanged and still per-walker: the protocol
        loop, the cost reduction, the priors. Only the surrogate is batched, which is the
        part that was expensive.

        Walkers whose prior is not finite are dropped before the prediction rather than
        after. They contribute nothing but would still cost a row in the batch, and an
        out-of-bounds row is exactly what ``out_of_bounds='error'`` refuses -- so
        predicting them would turn a walker that is merely being rejected into a failed run.
        """
        ensemble = np.atleast_2d(np.asarray(ensemble, dtype=float))
        lnpriors = np.array([self.get_lnprior_from_params(theta) for theta in ensemble])
        out = np.full(len(ensemble), -np.inf)

        usable = np.flatnonzero(np.isfinite(lnpriors))
        if usable.size == 0:
            return out

        self.sim_helper.predict_ensemble(ensemble[usable])
        for position, walker in enumerate(usable):
            self.sim_helper.select_from_ensemble(position)
            out[walker] = (lnpriors[walker]
                           + self.get_lnlikelihood_from_params(ensemble[walker]))
        return out

    def get_lnlikelihood_from_params(self, param_vals):
        cost = self.get_cost_from_params(param_vals)
        # cost = (sum of raw per-sub costs) / total weighted observable count; recover summed NLL.
        lnlikelihood = -cost * self._lnlikelihood_denorm_factor

        return lnlikelihood
    
    def get_pred_from_params(self, param_vals, reset=True, 
                                          only_one_exp=-1, pred_names=None):
        _, _, pred = self.get_cost_obs_and_pred_from_params(param_vals, reset=reset,
                                          only_one_exp=only_one_exp, pred_names=pred_names)
        return pred

    def get_pred_array_from_params_per_exp(self, param_vals, exp_idx):
                                          
        pred_operand_outputs = self.get_pred_from_params(param_vals=param_vals, reset=False, 
                                                only_one_exp=exp_idx, 
                                                pred_names=self.prediction_info['names'])
    
        # The second index of pred_output is the operand idx
        # TODO currently we don't allow operands for prediction outputs.
        # TODO but we should in the future
        # TODO here is where we would do the operations on the operands
        # for now we just concatenate results for subexperiments 
        pred_output_list = []                           
        for this_sub_idx in range(self.protocol_info["num_sub_per_exp"][exp_idx]):
            if this_sub_idx == 0:
                # the last 3 idxs are, pred_idx, operand_idx, time_idx
                pred_output_list.append(np.array(pred_operand_outputs[this_sub_idx])[:,0,:])
            else:
                pred_output_list.append(np.array(pred_operand_outputs[this_sub_idx])[:,0,1:])
        pred_outputs = np.concatenate(pred_output_list, axis=1)
        return pred_outputs

    def get_cost_from_operands(self, operands_outputs, exp_idx = 0, sub_idx = 0, do_ad=False):

        # The operands are symbolic (casadi SX) only on the AD path of a casadi_python model; every
        # other evaluation -- including gradient-free calibration on casadi_python -- produces numpy
        # operands, which must go through the numpy-mode funcs (#315). model_type alone does NOT
        # imply symbolic: casadi_python evaluated numerically is not.
        is_symbolic = do_ad and self.model_type == 'casadi_python'

        with self.evaluating_segment(exp_idx, sub_idx):
            obs_dict = self.get_obs_output_dict(operands_outputs, is_symbolic=is_symbolic)
        # calculate error between the observables of this set of parameters
        # and the ground truth
        
        cost = self.cost_calc(obs_dict, exp_idx=exp_idx, sub_idx=sub_idx, is_symbolic=is_symbolic)

        return cost

    def _align_series_to_ground_truth(self, series_obj, series_idx):
        """Put a simulated series and its ground truth on a common time grid.

        `series_obj` is either a numpy array or a casadi column vector (symbolic when
        differentiating). When the simulation dt differs from the observation's obs_dt, the
        simulated series is linearly interpolated onto the observation times, so the residuals
        are taken at the times the data was actually measured at.

        Linear interpolation is a multiply by weights that depend only on the two time grids,
        never on the parameters, so this works on a symbolic series too and leaves it
        differentiable. (Interpolating the ground truth up onto the finer simulation grid
        instead would invent data points between the samples, leaving a non-zero cost at the
        true parameters.)

        Returns (series_entry, ground_truth, std), all of the same length, with series_entry the
        same kind of object as `series_obj`.
        """
        is_casadi = not isinstance(series_obj, np.ndarray)

        ground_truth = np.asarray(self.obs_info["ground_truth_series"][series_idx], dtype=float)
        std = np.asarray(self.obs_info["std_series_vec"][series_idx], dtype=float)
        if std.ndim == 0:
            std = np.full(ground_truth.shape, float(std))

        obs_dt = self.obs_info["obs_dt"][series_idx]
        num_sim = series_obj.size1() if is_casadi else series_obj.shape[0]

        if obs_dt == self.dt:
            min_len_series = min(ground_truth.shape[0], num_sim)
            return (series_obj[:min_len_series], ground_truth[:min_len_series],
                    std[:min_len_series])

        if num_sim < 2:
            raise ValueError(
                f'cannot interpolate series observable {series_idx}: the simulation produced '
                f'{num_sim} sample(s).')

        # Sample k of a series is at time k*dt, so the grids are built with arange. (Note
        # linspace(0, n*dt, n) has a spacing of n*dt/(n-1), not dt, which stretches the two grids
        # by different factors and drifts them apart over a long simulation.)
        t_sim = np.arange(num_sim) * self.dt
        t_obs = np.arange(ground_truth.shape[0]) * obs_dt

        # Only compare where the simulation actually reaches: past its end there is nothing to
        # interpolate between, and clamping to the final value would invent a flat tail.
        num_in_range = int(np.count_nonzero(t_obs <= t_sim[-1] + 1e-12 * max(1.0, t_sim[-1])))
        if num_in_range == 0:
            raise ValueError(
                f'series observable {series_idx} has no overlap between the simulated times '
                f'(dt={self.dt}, {num_sim} samples) and the observation times (obs_dt={obs_dt}).')
        t_obs = t_obs[:num_in_range]

        # Each observation time sits between simulation samples lower and lower+1, a fraction
        # `frac` of the way along; interpolated[k] = (1-frac)*sim[lower] + frac*sim[lower+1].
        lower = np.clip(np.floor(t_obs / self.dt).astype(int), 0, num_sim - 2)
        frac = (t_obs - lower * self.dt) / self.dt

        if is_casadi:
            # gathers, so every entry of the symbolic series is preserved and differentiable
            frac_ca = ca.DM(frac.reshape(-1, 1))
            series_entry = ((1.0 - frac_ca) * series_obj[lower.tolist()]
                            + frac_ca * series_obj[(lower + 1).tolist()])
        else:
            series_entry = (1.0 - frac) * series_obj[lower] + frac * series_obj[lower + 1]

        return series_entry, ground_truth[:num_in_range], std[:num_in_range]

    def _cost_kwargs_for(self, obs_idx):
        """The data_item's ``cost_kwargs`` for observable ``obs_idx`` (issue #84).

        Indexed by *observable*, matching cost_type and the weight vectors, so it stays correct
        when the const/series/amp vectors are compacted to their own index spaces.
        """
        raw = self.obs_info.get("cost_kwargs") if self.obs_info else None
        if not raw or obs_idx >= len(raw):
            return None
        return raw[obs_idx] or None

    def _cost_weight_vectors(self, exp_idx, sub_idx):
        """The four per-data_item weight vectors this sub-experiment's cost is built from.

        A single seam so a subclass can change what weighting the cost uses without
        reimplementing cost_calc -- MCMC flattens them, because a weighted likelihood is
        not a posterior (issue #193).

        Returns them in the order (const, series, amp, phase).
        """
        return (
            self.protocol_info["scaled_weight_const_from_exp_sub"][exp_idx][sub_idx],
            self.protocol_info["scaled_weight_series_from_exp_sub"][exp_idx][sub_idx],
            self.protocol_info["scaled_weight_amp_from_exp_sub"][exp_idx][sub_idx],
            self.protocol_info["scaled_weight_phase_from_exp_sub"][exp_idx][sub_idx],
        )

    def cost_calc(self, obs_dict, exp_idx=0, sub_idx=0, is_symbolic=False):

        # Symbolic cost terms use the casadi-mode cost funcs; numeric ones the numpy-mode funcs
        # (#315). For non-casadi models both are the same numpy dict.
        cost_funcs_dict = (self.cost_funcs_dict_symbolic if is_symbolic
                           else self.cost_funcs_dict)

        const = obs_dict['const']
        series = obs_dict['series']
        amp = obs_dict['amp']
        phase = obs_dict['phase']

        # update cost weights for this experiment and subexperiment.
        #
        # These are indexed by *data_item row*, not by the per-type compacted counter:
        # process_protocol_and_weights builds each one full length over all data_items and
        # zeroes the rows that are not its type. So every read below uses obs_idx, the row
        # the observable actually came from, the way cost_type already did. Reading them by
        # const_idx / series_idx / ... only agreed when the items of a type happened to
        # occupy the leading rows; interleave the types and an observable picked up another
        # row's weight -- usually a zero, which dropped it from the cost while
        # _refresh_num_weighted_obs_tables still counted it in the denominator (#349).
        (updated_weight_const_vec, updated_weight_series_vec, updated_weight_amp_vec,
         updated_weight_phase_vec) = self._cost_weight_vectors(exp_idx, sub_idx)

        # get number of obs that don't have zero weights (cached in __init__ / refresh on obs/protocol change)
        if self._num_weighted_obs_by_exp_sub is not None:
            num_weighted_obs = self._num_weighted_obs_by_exp_sub[exp_idx][sub_idx]
        else:
            num_weighted_obs = int(
                np.sum(updated_weight_const_vec != 0)
                + np.sum(updated_weight_series_vec != 0)
                + np.sum(updated_weight_amp_vec != 0)
                + np.sum(updated_weight_phase_vec != 0)
            )
        
        # this subexperiment doesn't have any weighted observables, so no cost
        if num_weighted_obs == 0.0:
            return 0.0
        
        if len(self.obs_info["ground_truth_phase"]) == 0:
            phase = None
        if self.obs_info["ground_truth_phase"].all() == None:
            phase = None

        # TODO: Fix for amp and phase
        if is_symbolic:
            _require_casadi()
            cost = ca.SX(0)
            if const is not None:
                for const_idx in range(const.size1()):
                    obs_idx = self.obs_info['const_idx_to_obs_idx'][const_idx]
                    if updated_weight_const_vec[obs_idx] != 0:
                        cost_func = cost_funcs_dict[self.cost_type[obs_idx]]
                        # A cost that scores against a distribution builds its density from
                        # numbers (scipy's gaussian_kde, a mixture) and cannot take a symbol.
                        # Before #421 these items were data_type prob_dist and were refused
                        # below with amp/phase; now they are constants, so refuse them here --
                        # otherwise the symbolic cost silently picks up the nan standing in for
                        # the `value` such an item deliberately does not have.
                        if ground_truth_param_name(cost_func) == 'prob_dist_params':
                            raise NotImplementedError(
                                f"cost_type '{self.cost_type[obs_idx]}' scores against a "
                                f"distribution and cannot be differentiated symbolically. Use a "
                                f"value/std cost (gaussian_MLE, MSE, AE) for this data_item, or "
                                f"turn off do_ad.")
                        cost += call_cost_func(cost_func,
                                               const[const_idx], self.obs_info["ground_truth_const"][const_idx],
                                               std=self.obs_info["std_const_vec"][const_idx],
                                               weight=updated_weight_const_vec[obs_idx],
                                               cost_kwargs=self._cost_kwargs_for(obs_idx))

            if series is not None:
                for series_idx in range(len(series)):
                    obs_idx = self.obs_info['series_idx_to_obs_idx'][series_idx]
                    weight_entry = updated_weight_series_vec[obs_idx]
                    if weight_entry == 0:
                        continue

                    # this branch is taken for every casadi_python model, not just when
                    # differentiating, so the series is symbolic (SX) under do_ad and a plain
                    # numeric array otherwise. Both become a casadi column vector here.
                    series_col = _as_casadi_column(series[series_idx])

                    series_entry, obs_np, std_np = self._align_series_to_ground_truth(
                        series_col, series_idx)

                    # cast the data to casadi column vectors so the elementwise ops below
                    # don't get broadcast against a numpy row vector
                    obs_entry = ca.DM(obs_np.reshape(-1, 1))
                    std_entry = ca.DM(std_np.reshape(-1, 1))

                    cost += call_cost_func(cost_funcs_dict[self.cost_type[obs_idx]],
                        series_entry, obs_entry, std=std_entry, weight=weight_entry,
                        cost_kwargs=self._cost_kwargs_for(obs_idx))

            # Silently returning a zero cost for observables we can't differentiate would look
            # like a perfectly converged fit, so fail loudly instead.
            if amp is not None or phase is not None:
                raise NotImplementedError(
                    'automatic differentiation of frequency (amp/phase) observables is not '
                    'implemented. Use constant or series data items, or turn off do_ad.')

            return cost

        # # TODO change functionality so the cost type is defined in the obs_data.json not the user_inputs.yaml
        # if self.cost_type == 'MSE':
        #     cost = np.sum(np.power(updated_weight_const_vec*(const -
        #                        self.obs_info["ground_truth_const"])/self.obs_info["std_const_vec"], 2))
        # elif self.cost_type == 'AE':
        #     cost = np.sum(np.abs(updated_weight_const_vec*(const -
        #                                                   self.obs_info["ground_truth_const"])/self.obs_info["std_const_vec"]))
        # else:
        #     print(f'cost type of {self.cost_type} not implemented')
        #     exit()
        cost = 0.0
        if const is not None:
            for const_idx in range(len(const)):
                obs_idx = self.obs_info['const_idx_to_obs_idx'][const_idx]
                if updated_weight_const_vec[obs_idx] != 0:
                    cost_func = cost_funcs_dict[self.cost_type[obs_idx]]
                    cost += call_cost_func(cost_func,
                                           const[const_idx],
                                           self._ground_truth_for(cost_func, const_idx, obs_idx),
                                           std=self.obs_info["std_const_vec"][const_idx],
                                           weight=updated_weight_const_vec[obs_idx],
                                           cost_kwargs=self._cost_kwargs_for(obs_idx))
        
        # TODO debugging a strange error that occurs occasionally in GA
        # assert not np.isnan(cost), 'cost is nan'
        assert isinstance(cost, float), 'cost is not a float'

        series_cost = 0
        if series is not None:
            #print(series)
            # TODO make the above applicable for different length series? If we have different dt for series data

            # calculate sum of squares cost and divide by number data points in series data
            # divide by number data points in series data
            # if self.cost_type == 'MSE':
            #     series_cost = np.sum(np.power((series[:, :min_len_series] -
            #                                    self.obs_info["ground_truth_series"][:,
            #                                    :min_len_series]) * updated_weight_series_vec.reshape(-1, 1) /
            #                                   self.obs_info["std_series_vec"].reshape(-1, 1), 2)) / min_len_series
            # elif self.cost_type == 'AE':
            #     series_cost = np.sum(np.abs((series[:, :min_len_series] -
            #                                  self.obs_info["ground_truth_series"][:,
            #                                  :min_len_series]) * updated_weight_series_vec.reshape(-1, 1) /
            #                                 self.obs_info["std_series_vec"].reshape(-1, 1))) / min_len_series

            for series_idx in range(len(series)):
                obs_idx = self.obs_info['series_idx_to_obs_idx'][series_idx]
                weight_entry = updated_weight_series_vec[obs_idx]
                if weight_entry == 0:
                    # Nothing to add, and nothing to align either. The alignment was
                    # happening first and raising on items whose cost was then thrown
                    # away -- which made a recorded trace carried at weight 0 purely for
                    # plotting break every run that used an emulator, since the emulator
                    # returns scalars and there is no trace to interpolate.
                    continue

                # interpolates the simulated series onto the observation times when
                # dt != obs_dt; shared with the symbolic cost so both agree exactly
                series_entry, obs_entry, std_entry = self._align_series_to_ground_truth(
                    np.asarray(series[series_idx], dtype=float).flatten(), series_idx)

                series_cost += call_cost_func(cost_funcs_dict[self.cost_type[obs_idx]], series_entry, obs_entry,
                                              std=std_entry, weight=weight_entry, cost_kwargs=self._cost_kwargs_for(obs_idx))


        amp_cost = 0
        if amp is not None:
            # calculate sum of squares cost and divide by number data points in freq data
            # divide by number data points in series data
            # if self.cost_type == 'MSE':
            #     amp_cost = np.sum([np.power((amp[JJ] - self.obs_info["ground_truth_amp"][JJ]) *
            #                                  updated_weight_amp_vec[JJ] /
            #                                  self.obs_info["std_amp_vec"][JJ], 2) / len(amp[JJ]) for JJ in range(len(amp))])
            # elif self.cost_type == 'AE':
            #     amp_cost = np.sum([np.abs((amp[JJ] - self.obs_info["ground_truth_amp"][JJ]) *
            #                                  updated_weight_amp_vec[JJ] /
            #                                  self.obs_info["std_amp_vec"][JJ]) / len(amp[JJ]) for JJ in range(len(amp))])
            for amp_idx in range(len(amp)):
                obs_idx = self.obs_info['freq_idx_to_obs_idx'][amp_idx]
                amp_entry = amp[amp_idx]
                obs_entry = self.obs_info["ground_truth_amp"][amp_idx]
                weight_entry = updated_weight_amp_vec[obs_idx]
                std_entry = self.obs_info["std_amp_vec"][amp_idx]
                if hasattr(weight_entry, '__len__'):
                    if not all(val==0 for val in weight_entry):
                        amp_cost += call_cost_func(cost_funcs_dict[self.cost_type[obs_idx]], amp_entry, obs_entry,
                                                   std=std_entry, weight=weight_entry, cost_kwargs=self._cost_kwargs_for(obs_idx))
                else:
                    if weight_entry != 0:
                        amp_cost += call_cost_func(cost_funcs_dict[self.cost_type[obs_idx]], amp_entry, obs_entry,
                                                   std=std_entry, weight=weight_entry, cost_kwargs=self._cost_kwargs_for(obs_idx))

        phase_cost = 0
        if phase is not None:
            # calculate sum of squares cost and divide by number data points in freq data
            # divide by number data points in series data
            # TODO figure out how to properly weight this compared to the frequency weight.
            # if self.cost_type == 'MSE':
            #     phase_cost = np.sum([np.power((phase[JJ] - self.obs_info["ground_truth_phase"][JJ]) *
            #                                  updated_weight_phase_vec[JJ], 2) / len(phase[JJ]) for JJ in
            #                         range(len(phase))])
            # if self.cost_type == 'AE':
            #     phase_cost = np.sum([np.abs((phase[JJ] - self.obs_info["ground_truth_phase"][JJ]) *
            #                                   updated_weight_phase_vec[JJ]) / len(phase[JJ]) for JJ in
            #                          range(len(phase))])
            # TODO should we be inputting in a proper std for the phase? Probably.
            for phase_idx in range(len(phase)):
                obs_idx = self.obs_info['freq_idx_to_obs_idx'][phase_idx]
                phase_entry = phase[phase_idx]
                std_entry = np.ones(len(phase_entry))
                obs_entry = self.obs_info["ground_truth_phase"][phase_idx]
                weight_entry = updated_weight_phase_vec[obs_idx]
                if hasattr(weight_entry, '__len__'):
                    if not all(val==0 for val in weight_entry):
                        phase_cost += call_cost_func(cost_funcs_dict[self.cost_type[obs_idx]], phase_entry, obs_entry,
                                                     std=std_entry, weight=weight_entry, cost_kwargs=self._cost_kwargs_for(obs_idx))
                else:
                    if weight_entry != 0:
                        phase_cost += call_cost_func(cost_funcs_dict[self.cost_type[obs_idx]], phase_entry, obs_entry,
                                                     std=std_entry, weight=weight_entry, cost_kwargs=self._cost_kwargs_for(obs_idx))

        return cost + series_cost + amp_cost + phase_cost

    def _ground_truth_for(self, cost_func, const_idx, obs_idx):
        """What this observable is compared against: a number, or a distribution.

        Chosen from the cost func's signature rather than from the data_item's type. A scalar
        scored against a KDE of measured samples is still a scalar -- ``prob_dist`` used to be a
        fourth data_type for this, which put those observables in a parallel vector and hid them
        from everything that works on scalar features, the emulator included (issue #421).

        ``prob_dist_params`` is read by ``obs_idx``, the data_item row, the way ``cost_type`` and
        the weight vectors are; ``ground_truth_const`` keeps its own compacted ``const_idx``.
        """
        if ground_truth_param_name(cost_func) == 'prob_dist_params':
            params = self.obs_info["ground_truth_prob_dist_params"][obs_idx]
            if params is None:
                raise ValueError(
                    f'cost_type {self.cost_type[obs_idx]!r} scores its data_item against a '
                    f'distribution, so the data_item needs a "prob_dist_params" entry.')
            return params
        return self.obs_info["ground_truth_const"][const_idx]

    def _resolve_operation_kwargs(self, JJ, operation_funcs_dict, operands_outputs,
                                  num_operands=None):
        """Keyword arguments for observable ``JJ``'s operation func, from its ``operation_kwargs``.

        Single entry point for the ``operation_kwargs`` contract (#304) on the param-id / MCMC /
        UQ path, so validation and the "string value naming an earlier observable" substitution
        behave exactly as they do in sensitivity analysis. See
        ``param_id.operation_funcs.resolve_operation_kwargs``.
        """
        operation_name = self.obs_info["operations"][JJ]
        if num_operands is None:
            operands_for_JJ = operands_outputs[JJ]
            num_operands = len(operands_for_JJ) if hasattr(operands_for_JJ, '__len__') else 0
        return resolve_operation_kwargs(
            self.obs_info["operation_kwargs"][JJ],
            operation_funcs_dict[operation_name],
            operation_name=operation_name,
            data_item_name=obs_item_names(self.obs_info)[JJ],
            temp_results=self.temp_results,
            num_operands=num_operands,
            known_item_names=set(obs_item_names(self.obs_info)),
        )

    def cross_segment_reference_items(self):
        """data_items whose ``operation_kwargs`` reference an item in another (exp, sub).

        Returns a list of ``(referencing_name, referenced_name)``. Empty for the ordinary case
        where every reference stays inside its own sub-experiment.
        """
        obs = self.obs_info
        names = list(obs_item_names(obs))
        segment = {name: (int(obs["experiment_idxs"][i]), int(obs["subexperiment_idxs"][i]))
                   for i, name in enumerate(names)}
        found = []
        for i, raw in enumerate(obs.get("operation_kwargs") or []):
            if not isinstance(raw, dict):
                continue
            here = segment.get(names[i])
            for value in raw.values():
                if isinstance(value, str) and value in segment and segment[value] != here:
                    found.append((names[i], value))
        return found

    def _refuse_cross_segment_references(self, source):
        """Cross-segment references are a cost-path feature; say so rather than mis-differentiate.

        The Myokit-FSA and CasADi arms build each observable from one sub-experiment's operands,
        so an item whose value comes from *another* segment is not a function of what they
        differentiate -- they would return a gradient for a different feature than the cost was
        built from, with nothing to show for it. Finite differences and the gradient-free
        methods step the whole protocol, so they are unaffected.
        """
        crossing = self.cross_segment_reference_items()
        if not crossing:
            return
        detail = "; ".join(f"{a!r} -> {b!r}" for a, b in crossing)
        raise NotImplementedError(
            f"{source} cannot differentiate a data_item that references another experiment or "
            f"sub-experiment ({detail}). It builds each observable from one sub-experiment's "
            f"operands, so a cross-segment reference is not part of what it differentiates and "
            f"the gradient would be for a different feature than the cost. Use finite "
            f"differences (do_ad: False) or a gradient-free method, or keep the reference "
            f"inside one sub-experiment (#466).")

    @contextlib.contextmanager
    def accumulating_temp_results(self):
        """One ``temp_results`` table across every segment of a single cost evaluation (#466).

        Only the cost loop wants that. Every other caller evaluates one segment on its own --
        CUFLynx's ``obs_cost``, ``plot_outputs``, the gradient backends, a test double -- and
        must not resolve a reference against values left behind by whatever ran before it.
        ``evaluating_segment`` gives those callers a fresh table, and this is how it knows not
        to.
        """
        previous = getattr(self, '_accumulating_temp_results', False)
        self._accumulating_temp_results = True
        self.temp_results = {}
        try:
            yield
        finally:
            self._accumulating_temp_results = previous

    @contextlib.contextmanager
    def evaluating_segment(self, exp_idx, sub_idx):
        """Mark which (experiment, sub-experiment)'s operands the next evaluations receive.

        Carried as state rather than passed to ``get_obs_output_dict``, so that method keeps the
        signature every other caller already uses -- the gradient backends, plot_outputs,
        CUFLynx, and any test double that stands in for it. Adding parameters to it broke all of
        those for no gain.

        Inside the block an item is only evaluated and recorded when the segment is its own, and
        ``temp_results`` is left alone so the caller can accumulate one table across the whole
        protocol. Outside it -- the default -- every item is evaluated against whatever operands
        are handed over and the table is cleared per call, which is the pre-#466 behaviour.
        """
        previous = getattr(self, '_eval_segment', (None, None))
        # Outside an `accumulating_temp_results` block this is a caller evaluating one segment
        # on its own, so it gets a fresh table -- the pre-#466 behaviour, where the table was
        # cleared on every `get_obs_output_dict` call. Without this the attribute may not exist
        # at all (it was only ever created by the cost loop), and `_record_temp_result` raised
        # AttributeError for every caller entering at `get_cost_from_operands`.
        if not getattr(self, '_accumulating_temp_results', False):
            self.temp_results = {}
        self._eval_segment = (exp_idx, sub_idx)
        try:
            yield
        finally:
            self._eval_segment = previous

    def _item_belongs_to_segment(self, JJ, exp_idx, sub_idx):
        """Whether data_item ``JJ`` was declared for the segment currently being evaluated.

        Always True when the caller did not say which segment it is handing over, which is the
        old behaviour and what the callers that want the whole const vector from one segment's
        operands rely on.
        """
        if exp_idx is None:
            return True
        return (int(self.obs_info["experiment_idxs"][JJ]) == int(exp_idx)
                and int(self.obs_info["subexperiment_idxs"][JJ]) == int(sub_idx))

    def _record_temp_result(self, JJ, obs, exp_idx, sub_idx):
        """Record observable ``JJ``'s value under its name, for a later item to reference.

        Skipped when this is not ``JJ``'s own segment: the value would be this item's operation
        applied to a different experiment's trace, which is a number nothing should read.
        """
        if not self._item_belongs_to_segment(JJ, exp_idx, sub_idx):
            return
        self.temp_results[obs_item_names(self.obs_info)[JJ]] = obs

    def get_obs_output_dict(self, operands_outputs, get_all_series=False, is_symbolic=False):
        """Evaluate every data_item's operation against one sub-experiment's operands.

        ``temp_results`` is the table an ``operation_kwargs`` reference to another item resolves
        against. It used to be cleared on every call, i.e. once per (experiment, sub-experiment),
        which is why a reference could only ever see the segment being evaluated (#466, #127).
        Two arguments change that:

A caller that steps through the segments in order does so inside
        ``evaluating_segment``, which says which segment these operands belong to and leaves the
        table alone so it accumulates across all of them.

        Knowing the segment matters because every data_item is evaluated against every one of
        them -- which item *counts* is decided afterwards by the zeroed weight vectors. Without
        it an item would be recorded under its name from some other experiment's trace, and a
        cross-segment reference would silently read that.
        """
        exp_idx, sub_idx = getattr(self, '_eval_segment', (None, None))
        if exp_idx is None:
            self.temp_results = {}

        # Symbolic (SX) operands go through the casadi-mode operation funcs; numeric operands go
        # through the numpy-mode ones (#315). For non-casadi models both are the same numpy dict.
        operation_funcs_dict = (self.operation_funcs_dict_symbolic if is_symbolic
                                else self.operation_funcs_dict)

        if operands_outputs == None:
            if get_all_series:
                return None, None
            else:
                return None

        if is_symbolic:
            _require_casadi()
            # TODO: Test series, amp and phase
            obs_const_vec = ca.SX.zeros(len(self.obs_info["ground_truth_const"]), 1)
            obs_series_list_of_arrays = [None]*len(self.obs_info["ground_truth_series"])
            obs_amp_list_of_arrays = [None]*len(self.obs_info["ground_truth_amp"])
            obs_phase_list_of_arrays = [None]*len(self.obs_info["ground_truth_phase"])
        else:     
            obs_const_vec = np.zeros((len(self.obs_info["ground_truth_const"]), ))
            obs_series_list_of_arrays = [None]*len(self.obs_info["ground_truth_series"])
            obs_amp_list_of_arrays = [None]*len(self.obs_info["ground_truth_amp"])
            obs_phase_list_of_arrays = [None]*len(self.obs_info["ground_truth_phase"])

        if get_all_series:
            # An emulator has no series to give -- it predicts the scalar the series
            # was reduced to. That used to raise, which was too blunt: the caller
            # (plot_outputs) wants the *features* as well, and those are exactly what
            # an emulator does have. Hand back the features with every series None,
            # and let the caller skip the reconstruction rather than lose the errors
            # with it (#333).
            obs_series_array_all = [None]*len(operands_outputs)


        const_count = 0
        series_count = 0
        freq_count = 0
        for JJ in range(len(operands_outputs)):
            if self.obs_info["data_types"][JJ] == 'frequency':
                pass
            elif get_all_series and not self.emulates_features:
                # An emulator has no series for this item; the None left in place
                # is what says so. Running the operation's series branch on an
                # already-reduced scalar would put a length-1 "trace" here, which
                # a plot would draw as a single point and read as real.
                if self.obs_info["operations"][JJ] is None:
                    obs_series_array_all[JJ] = operands_outputs[JJ][0]
                elif hasattr(operation_funcs_dict[self.obs_info["operations"][JJ]], 'series_to_constant'):
                    kwargs = self._resolve_operation_kwargs(JJ, operation_funcs_dict, operands_outputs)
                    obs_series_array_all[JJ] = operation_funcs_dict[self.obs_info["operations"][JJ]](*operands_outputs[JJ],series_output=True,**kwargs)
                else:
                    kwargs = self._resolve_operation_kwargs(JJ, operation_funcs_dict, operands_outputs)
                    val_or_array = operation_funcs_dict[
                            self.obs_info["operations"][JJ]](*operands_outputs[JJ], **kwargs)
                    if type(val_or_array) == float:
                        print("an operation func that returns a float (constant) "
                              "Is present. This operation_func should have the header @series_to_constant"
                              "and have a kwarg series_output=True if you want to plot the series.")
                        # operation funcs that don't have @series_to_constant and kwarg series_output
                        # will not be plotted
                        obs_series_array_all[JJ] = None
                    else:
                        obs_series_array_all[JJ] = val_or_array

            # use the function defined in the operation_funcs_dict to calculate the observable
            # from the operands
            if self.emulates_features:
                # The emulator predicts the feature itself, i.e. what the operation would have
                # returned. Running the operation again would reduce an already-reduced scalar:
                # harmless for `mean`, but `max_minus_min` of a single value is zero, and the
                # cost would then be fitting zeros without anything looking wrong.
                obs = float(np.asarray(operands_outputs[JJ][0]).reshape(-1)[0])
                self._record_temp_result(JJ, obs, exp_idx, sub_idx)
            elif not self._item_belongs_to_segment(JJ, exp_idx, sub_idx):
                # Every data_item is evaluated against every segment, and which one *counts* is
                # decided afterwards by the zeroed weight vectors. For an item belonging to
                # another (experiment, sub-experiment) the operation would run on the wrong
                # trace, so the value is discarded either way -- and for an item built from
                # other items it cannot run at all, because the items it names live in a
                # segment this one is not (#466).
                #
                # The placeholder has to keep the *shape* the slot expects, not just be falsy: a
                # series observable is interpolated onto its ground-truth times before the
                # weights are applied, and a scalar there fails as "the simulation produced 1
                # sample". So a series gets a zero trace of the operand's own length.
                obs = 0.0
                if self.obs_info["data_types"][JJ] == 'series':
                    trace = operands_outputs[JJ][0] if operands_outputs[JJ] is not None \
                        and len(operands_outputs[JJ]) else None
                    if trace is not None:
                        obs = ca.SX.zeros(trace.shape) if is_symbolic else np.zeros_like(
                            np.asarray(trace, dtype=float))
            elif self.obs_info["operations"][JJ] == None:
                obs = operands_outputs[JJ][0]
            else:
                if self.obs_info["data_types"][JJ] != 'frequency':
                    key_idxt = obs_item_names(self.obs_info)[JJ]
                    kwargs = self._resolve_operation_kwargs(JJ, operation_funcs_dict, operands_outputs)
                    obs = operation_funcs_dict[self.obs_info["operations"][JJ]](*operands_outputs[JJ], **kwargs)
                    #each predict result saved into tmp array
                    self._record_temp_result(JJ, obs, exp_idx, sub_idx)
                else:
                    obs = None
            
            if self.obs_info["data_types"][JJ] == 'constant':
                obs_const_vec[const_count] = obs
                const_count += 1
            if self.obs_info["data_types"][JJ] == 'series':
                obs_series_list_of_arrays[series_count] = obs
                series_count += 1
            elif self.obs_info["data_types"][JJ] == 'frequency':
                # TODO copy this to mcmc
                if self.obs_info["operations"][JJ] == None:

                    # TODO add a hanning window when doing the fft if it is not periodic
                    time_domain_obs = operands_outputs[JJ][0][:-1]
                    # time_domain_obs = np.hanning(len(time_domain_obs)) * time_domain_obs
                    # zero-padding
                    # time_domain_obs = np.concatenate([time_domain_obs, np.zeros(len(time_domain_obs))]) 
                    # N = len(time_domain_obs) //2 # if zero-padding do this
                    N = len(time_domain_obs)

                    # TODO this scaling needs to change if i do more periodic repeats!!
                    complex_num = np.fft.fft(time_domain_obs)/(N)
                    amp = np.abs(complex_num)[0:N]
                    # make sure the first amplitude is negative if it is a negative signal
                    amp[0] = amp[0] * np.sign(np.mean(time_domain_obs))
                    phase = np.angle(complex_num)[0:N]
                    for idx in range(len(phase)):
                        if np.abs(amp[idx]) < 1e-12:
                            phase[idx] = 0
                
                    freqs = np.fft.fftfreq(N, d=self.dt)[:N]
                else:
                    complex_operands = [np.fft.fft(operands_outputs[JJ][KK]) / \
                                       len(operands_outputs[JJ][KK]) for \
                                       KK in range(len(operands_outputs[JJ]))]

                    time_domain_obs = operands_outputs[JJ][0]
                    # operations also apply to complex numbers
                    freq_kwargs = self._resolve_operation_kwargs(
                        JJ, operation_funcs_dict, operands_outputs, num_operands=len(complex_operands))
                    complex_num = operation_funcs_dict[self.obs_info["operations"][JJ]](*complex_operands, **freq_kwargs)
                    # TODO check this works for all cases
                    # I am checking the sign of the mean operated on time domain signal to ensure 
                    # the first amplitude is negative if it is a negative signal
                    # sign_signal = np.sign(operation_funcs_dict[self.obs_info["operations"][JJ]](* \
                    #                             [np.mean(entry) for entry in operands_outputs[JJ]]))

                    amp = np.abs(complex_num)[0:len(time_domain_obs)]
                    # TODO I don't think I should do the below, commenting out
                    # Just make sure ground truth is abs value
                    # make sure the first amplitude is negative if it is a negative signal
                    # amp[0] = amp[0] * sign_signal
                    phase = np.angle(complex_num)[0:len(time_domain_obs)]
                    for idx in range(len(phase)):
                        if np.abs(amp[idx]) < 1e-12:
                            phase[idx] = 0

                    freqs = np.fft.fftfreq(len(time_domain_obs), 
                                           d=self.dt)[:len(time_domain_obs)]


                # now interpolate to defined frequencies
                obs_amp_list_of_arrays[freq_count] = utility_funcs.bin_resample(amp, freqs, self.obs_info["freqs"][JJ])
                # and phase
                obs_phase_list_of_arrays[freq_count] = utility_funcs.bin_resample(phase, freqs, self.obs_info["freqs"][JJ])

                # print(np.mean(amp))
                # TODO remove this plotting
                # fig, ax = plt.subplots()
                # ax.plot(freqs, amp, 'ko')
                # ax.plot(self.obs_freqs[JJ], obs_amp_list_of_arrays[freq_count][:], 'rx')
                # ax.set_xlim([0, 10])
                # ax.set_ylim([0, max(amp)*1.1])
                # ax.set_xlabel('freq Hz')
                # ax.set_ylabel('Impedance $Js/m^6$')

                # # randnum = np.random.randint(100000)
                # plt.savefig(f'/home/farg967/Documents/random/rand_plots/amp.png')
                # plt.close()
                
                # fig, ax = plt.subplots()
                # ax.plot(freqs, phase, 'ko')
                # ax.plot(self.obs_freqs[JJ], obs_phase_list_of_arrays[freq_count][:], 'rx')
                # ax.set_xlim([0, 10])
                # ax.set_xlabel('freq Hz')
                # ax.set_ylabel('Phase')

                # # randnum = np.random.randint(100000)
                # plt.savefig(f'/home/farg967/Documents/random/rand_plots/phase.png')
                # plt.close()

                freq_count += 1

        if const_count == 0:
            obs_const_vec = None
        if series_count == 0:
            obs_series_list_of_arrays = None
        if freq_count == 0:
            obs_amp_list_of_arrays = None
            obs_phase_list_of_arrays = None
        obs_dict = {'const': obs_const_vec, 'series': obs_series_list_of_arrays,
                    'amp': obs_amp_list_of_arrays, 'phase': obs_phase_list_of_arrays,
}

        if get_all_series: 
            return obs_dict, obs_series_array_all
        else:
            return obs_dict

    def get_preds_min_max_mean(self, preds):

        preds_const_vec = np.zeros((preds.shape[0]*3, ))
        for JJ in range(len(preds)):
            preds_const_vec[JJ] = np.min(preds[JJ, :])
            preds_const_vec[JJ + 1] = np.max(preds[JJ, :])
            preds_const_vec[JJ + 2] = np.mean(preds[JJ, :])
        return preds_const_vec
    
    # ---- CasADi symbolic backend (param_id/casadi_backend.py) ----

    def _casadi_functions_cache_key(self, param_names, get_all_series):
        """Signature of everything the CasADi graph is built from. See
        param_id.casadi_backend.functions_cache_key."""
        return casadi_backend.functions_cache_key(self, param_names, get_all_series)

    def build_casadi_functions(self, param_names, param_vals=None, get_all_series=False):
        """Build (and cache) the CasADi cost/gradient/observable Functions. See
        param_id.casadi_backend.build_functions."""
        return casadi_backend.build_functions(self, param_names, param_vals, get_all_series)

    def get_jac_cost_ca(self, param_vals):
        """Gradient dJ/dp from the CasADi symbolic graph. See
        param_id.casadi_backend.get_jac_cost."""
        return casadi_backend.get_jac_cost(self, param_vals)

    def get_cost_ca(self, param_vals):
        """Cost J(p) from the CasADi symbolic graph. See param_id.casadi_backend.get_cost."""
        return casadi_backend.get_cost(self, param_vals)

    # ---- Myokit CVODES forward-sensitivity backend (param_id/fsa_backend.py) ----

    def fsa_gradient_available(self):
        """True when this run can produce an analytic gradient via Myokit CVODES FSA. See
        param_id.fsa_backend.gradient_available."""
        return fsa_backend.gradient_available(self)

    def _ensure_fsa_setup(self):
        """Enable CVODES forward sensitivities on the Myokit sim helper (once). See
        param_id.fsa_backend.ensure_setup."""
        return fsa_backend.ensure_setup(self)

    def _total_weighted_obs_denominator(self):
        """Sum of weighted-observable counts over all experiments/sub-experiments.

        Matches the divisor get_cost_obs_and_pred_from_params uses for the full cost, so a
        gradient assembled from raw per-sub costs divided by this equals d(mean cost)/dp.

        Generic, despite only fsa_backend calling it today: this is the same divisor that
        get_cost_obs_and_pred_from_params and cost_calc each compute inline. It stays here so
        those three can eventually be unified in the cost-assembly layer rather than across a
        module boundary.
        """
        num_experiments = self.protocol_info["num_experiments"]
        num_sub_per_exp = self.protocol_info["num_sub_per_exp"]
        D = 0
        for exp_idx in range(num_experiments):
            for sub_idx in range(num_sub_per_exp[exp_idx]):
                if self._num_weighted_obs_by_exp_sub is not None:
                    D += self._num_weighted_obs_by_exp_sub[exp_idx][sub_idx]
                else:
                    wc = self.protocol_info["scaled_weight_const_from_exp_sub"][exp_idx][sub_idx]
                    ws = self.protocol_info["scaled_weight_series_from_exp_sub"][exp_idx][sub_idx]
                    wa = self.protocol_info["scaled_weight_amp_from_exp_sub"][exp_idx][sub_idx]
                    wp = self.protocol_info["scaled_weight_phase_from_exp_sub"][exp_idx][sub_idx]
                    D += int(np.sum(wc != 0) + np.sum(ws != 0) + np.sum(wa != 0)
                             + np.sum(wp != 0))
        return max(int(D), 1)

    def get_jac_cost_fsa(self, param_vals, return_cost=False):
        """Gradient dJ/dp via Myokit CVODES forward sensitivity, optionally with the cost from
        the same solve. See param_id.fsa_backend.get_jac_cost."""
        return fsa_backend.get_jac_cost(self, param_vals, return_cost)

    def get_cost_and_jac_fsa(self, param_vals):
        """(cost, gradient) from a single Myokit CVODES FSA solve. See
        param_id.fsa_backend.get_cost_and_jac."""
        return fsa_backend.get_cost_and_jac(self, param_vals)

    def _perturb_operands_along_sensitivity(self, operands, sens, pname, h):
        """Operand traces stepped by h along dS/dp. See
        param_id.fsa_backend.perturb_operands_along_sensitivity."""
        return fsa_backend.perturb_operands_along_sensitivity(self, operands, sens, pname, h)

    # ---- Backend-agnostic cost/gradient interface ----

    def get_cost(self, param_vals):
        """Compute cost J(p), dispatching to the emulator, CasADi, AADC or numpy."""
        if self.emulates_features:
            # First, and before the model_type branches: a casadi_python model run on an
            # emulator would otherwise evaluate its real symbolic graph here and quietly
            # ignore the emulator the user asked for.
            return float(self.get_cost_from_params(param_vals))
        if self.model_type == 'casadi_python':
            return float(self.get_cost_ca(param_vals))
        if self.model_type == 'aadc_python' and self.do_ad:
            # When the gradient comes off the tape the cost has to as well, or the optimiser
            # descends a different function than it evaluates. See get_cost_aadc.
            return float(self.get_cost_aadc(param_vals))
        return float(self.get_cost_from_params(param_vals))

    def get_gradient(self, param_vals):
        """Compute gradient ∇J(p), dispatching to the emulator, CasADi, AADC, or Myokit FSA."""
        if self.emulates_features:
            # Finite differences on the emulator's own cost -- the same function get_cost
            # returns, which the analytic arms (all of which differentiate the real model)
            # would not be. 2M evaluations is the wrong trade against a solver and the right
            # one here, where an evaluation is a matrix multiply.
            return fd_backend.cost_gradient(
                self, param_vals, h=float(self._use_time_setting('fd_rel_step', 1e-3)))
        if self.model_type == 'casadi_python':
            self._refuse_cross_segment_references('The CasADi AD gradient')
            return self.get_jac_cost_ca(param_vals)
        elif self.model_type == 'aadc_python':
            self._refuse_cross_segment_references('The AADC AD gradient')
            return self.get_jac_cost_aadc(param_vals)
        elif self.fsa_gradient_available():
            self._refuse_cross_segment_references('The Myokit CVODES FSA gradient')
            return self.get_jac_cost_fsa(param_vals)
        else:
            raise ValueError(f"Gradient not available for model_type={self.model_type}")

    def _observable_base_label(self, obs_idx):
        return observable_base_label(self.obs_info, obs_idx)

    def _ambiguous_observable_labels(self):
        """Base labels shared by more than one observable. Computed once per obs_info."""
        obs_info = self.obs_info
        cached = getattr(self, '_ambiguous_labels_cache', None)
        if cached is not None and cached[0] is obs_info:
            return cached[1]
        counts = {}
        for II in range(obs_info["num_obs"]):
            base = self._observable_base_label(II)
            counts[base] = counts.get(base, 0) + 1
        ambiguous = {base for base, n in counts.items() if n > 1}
        self._ambiguous_labels_cache = (obs_info, ambiguous)
        return ambiguous

    def _observable_label(self, obs_idx):
        """Human-readable, disambiguating label for observable ``obs_idx`` (used as the row key
        of the local-sensitivity matrices). names_for_plotting can repeat across observables that
        share a variable but differ by operation (e.g. mean vs max of the same trace), so the
        operation and operand are folded in.

        The experiment and sub-experiment are folded in too, but only when even that repeats.
        A data_item names both, and two experiments measuring the same feature of the same
        variable is an ordinary obs_data -- SN_simple has exactly that. These labels are the
        keys of the dict get_observable_sensitivities returns, so without this the two share
        one key and the second silently overwrites the first: one experiment's local
        sensitivity reported for both, with nothing to show anything was lost.

        Only when needed, so every unambiguous label -- all of them in a single-experiment
        study -- keeps the spelling it already had."""
        base = self._observable_base_label(obs_idx)
        if base not in self._ambiguous_observable_labels():
            return base
        exp = self.obs_info["experiment_idxs"][obs_idx]
        sub = self.obs_info["subexperiment_idxs"][obs_idx]
        return f"{base} [exp {exp}, sub {sub}]"

    def get_observable_sensitivities(self, param_vals, gradient_method=None, fd_rel_step=None):
        """d(observable feature)/d(param) for the scalar observables -- the backend-agnostic
        local-sensitivity accessor, parallel to ``get_gradient``.

        Returns ``{observable_label: {param_name: d(feature)/d(param)}}``, dispatching by
        model_type to the same analytic machinery the cost gradient uses: the CasADi jacobian
        of the observable vector, or the Myokit CVODES sensitivities with a directional
        derivative of the feature. Every arm reports the identical quantity, so a local
        sensitivity analysis is comparable across backends whichever computed it.

        ``gradient_method`` selects how, in the same FD/AD/FSA vocabulary as
        ``gradient_sources()`` and the Laplace ``gradient_source`` -- the arm's own name, so a
        front-end can offer it, disable it, and report back which one ran:

        * ``None`` / ``'auto'`` / ``'analytic'`` (default) -- the analytic arm for this
          backend, raising when there is none. Deliberately still not a silent fall back to
          FD: a result quietly computed a different way, at a different cost and accuracy,
          is not the same result.
        * ``'AD'`` -- the exact CasADi jacobian; requires ``model_type='casadi_python'``
          (``aadc_python`` names its arm AD too, but its local SA is not implemented yet and
          says so). Any other backend raises naming the mismatch rather than silently
          reinterpreting.
        * ``'FSA'`` -- Myokit CVODES forward sensitivities; requires ``cellml`` +
          ``CVODE_myokit`` + ``do_ad``. Raises naming exactly what is missing otherwise.
        * ``'FD'`` -- central finite differences (``param_id.fd_backend``). Works on any
          backend that runs a forward simulation, which is how AADC and the plain scipy
          backend get a local SA at all (issue #338). Costs 2M simulations for M parameters.

        ``fd_rel_step`` is the FD step, relative to each parameter, and is ignored by the
        analytic arms. It matters more than it looks: on Lotka-Volterra, moving it from
        1e-3 to 1e-2 changes a sensitivity coefficient by up to 48%, because `max` of an
        oscillating trace is a rough functional. So it is the caller's to choose, not a
        constant buried in the backend -- the same reason the prior hyper-parameters
        stopped being hardcoded.
        """
        method = (gradient_method or '').strip().upper()
        if self.emulates_features:
            # Over an emulator, FD is the only honest arm: the analytic ones differentiate the
            # real model, which is not the function being evaluated. It is also free here.
            if method not in ('', 'ANALYTIC', 'AUTO', 'FD'):
                raise ValueError(
                    f"gradient_method '{gradient_method}' differentiates the real model, but "
                    f"this run evaluates an emulator (use_emulator: true). Only 'FD' is "
                    f"available over an emulator -- and it costs 2M emulator evaluations, "
                    f"not 2M simulations.")
            kwargs = {} if fd_rel_step is None else {'h': float(fd_rel_step)}
            return fd_backend.observable_feature_sensitivities(self, param_vals, **kwargs)
        if method == 'FD':
            kwargs = {} if fd_rel_step is None else {'h': float(fd_rel_step)}
            return fd_backend.observable_feature_sensitivities(self, param_vals, **kwargs)
        if method not in ('', 'ANALYTIC', 'AUTO', 'AD', 'FSA'):
            raise ValueError(
                f"unknown gradient_method '{gradient_method}' for local sensitivity analysis. "
                "Valid values are 'AD' (exact CasADi jacobian, casadi_python), 'FSA' (Myokit "
                "CVODES forward sensitivities, cellml + CVODE_myokit + do_ad), 'FD' "
                "(central finite differences, any backend), or None/'auto'/'analytic' (this "
                "backend's analytic arm).")

        solver_info = getattr(self, 'solver_info', None)
        solver = solver_info.get('solver') if isinstance(solver_info, dict) else None
        # An explicit arm name validates against the backend instead of being silently
        # reinterpreted -- a caller that asked for FSA must get FSA or an error, never a
        # different arm with plausible numbers (the same reason FD never stands in above).
        if method == 'AD' and self.model_type not in ('casadi_python', 'aadc_python'):
            raise ValueError(
                f"gradient_method 'AD' needs model_type 'casadi_python' (the exact CasADi "
                f"jacobian); this run is model_type='{self.model_type}', solver='{solver}'. "
                "Use 'FSA' for cellml + CVODE_myokit + do_ad, or 'FD'.")
        if method == 'FSA' and not fsa_backend.gradient_available(self):
            missing = []
            if self.model_type != 'cellml':
                missing.append(f"model_type is '{self.model_type}', needs 'cellml'")
            if not hasattr(getattr(self, 'sim_helper', None), 'enable_fsa'):
                missing.append(f"solver is '{solver}', needs 'CVODE_myokit' (its helper "
                               "provides CVODES forward sensitivities)")
            if not getattr(self, 'do_ad', False):
                missing.append("do_ad must be true")
            raise ValueError(
                "gradient_method 'FSA' is not available for this run: "
                + "; ".join(missing) + ". Use 'AD' for casadi_python, or 'FD'.")

        if self.model_type == 'casadi_python':
            return casadi_backend.get_observable_sensitivities(self, param_vals)
        elif self.model_type == 'aadc_python':
            raise NotImplementedError(
                "Local (derivative-based) sensitivity analysis is not yet implemented for the "
                "AADC backend. Use sa_options gradient_method 'FD', or model_type "
                "'casadi_python', or 'cellml' with solver 'CVODE_myokit', or global "
                "Sobol SA (sa_options method 'sobol').")
        elif fsa_backend.gradient_available(self):
            return fsa_backend.observable_feature_sensitivities(self, param_vals)
        else:
            raise NotImplementedError(
                "Local (derivative-based) sensitivity analysis needs an analytic sensitivity "
                f"backend, not available for model_type={self.model_type} / solver="
                f"{self.solver_info.get('solver') if isinstance(self.solver_info, dict) else None}. "
                "Use sa_options gradient_method 'FD', or model_type 'casadi_python', or "
                "'cellml' with solver 'CVODE_myokit' and do_ad true, or global Sobol SA "
                "(sa_options method 'sobol').")

    def get_cost_and_gradient(self, param_vals):
        """Return ``(cost, gradient)`` in one evaluation.

        L-BFGS-B needs both J(p) and ∇J(p) at every point it visits. For the Myokit CVODES
        FSA path a single augmented solve yields both, so this avoids the separate cost solve
        the optimiser would otherwise do. Other backends fall back to separate calls (CasADi's
        reverse pass and the AADC tape are cheap, so there is little to merge there).
        """
        if self.model_type not in ('casadi_python', 'aadc_python') \
                and self.fsa_gradient_available():
            return self.get_cost_and_jac_fsa(param_vals)
        return float(self.get_cost(param_vals)), self.get_gradient(param_vals)

    # ---- AADC tape backend (param_id/aadc_backend.py) ----

    def _aadc_cost_and_grad(self, param_vals):
        """(cost, gradient) from one AADC tape evaluation. See
        param_id.aadc_backend.cost_and_grad."""
        return aadc_backend.cost_and_grad(self, param_vals)

    def get_jac_cost_aadc(self, param_vals):
        return self._aadc_cost_and_grad(param_vals)[1]

    def get_cost_aadc(self, param_vals):
        """J(p) evaluated on the AADC tape.

        This must be the cost an AADC-gradient optimiser minimises. The forward solver and the
        tape do not integrate the same way -- the tape has to replay a fixed sequence of
        operations, so it uses a fixed-step scheme, while sim_helper.run() may use an adaptive
        one -- and the tape's cost is a separate implementation of the cost function. Taking
        J(p) from get_cost_from_params and dJ/dp from the tape therefore hands L-BFGS-B the
        gradient of a *different function* than the one it is minimising, which breaks the line
        search. Measured on Lotka-Volterra, that mismatch gave AD/FD ratios of
        [1.79, 1.96, 1.32, -0.067] -- the last one has the wrong sign.
        """
        return self._aadc_cost_and_grad(param_vals)[0]
    
    def get_obs_ca(self, param_vals, get_all_series=False):
        """Observables evaluated through the CasADi graph, in the same shape the numpy path
        returns. See param_id.casadi_backend.get_obs."""
        return casadi_backend.get_obs(self, param_vals, get_all_series)

    def simulate_once(self, param_vals=None, reset=True, only_one_exp=-1, return_series=False):
        """

        Setting reset to False and only_one_exp to the experiment number you want to use 
        allows you to use the simulation helper object to investigate all parameters.

        This can be used with reset=False and only_one_exp set to the experiment number
        to have the simulation helper object open and ready to investigate the parameters.

        if param_vals is not set, then the best_param_vals will be used.

        Args:
            only_one_exp (int, optional): If the user wants to only simulate one experiment
                                          change this to the experiment number. Defaults to -1.
            reset (bool, optional): if you want to reset the simulation after running.
                                    Gets changed to True for num_experiments > 1. Defaults to True.
        """
        if self.emulates_features:
            raise NotImplementedError(
                'simulate_once needs the full simulated trace, which a feature emulator cannot '
                'produce -- it predicts the scalar data_item features only. Re-run with '
                'use_emulator: false to simulate, plot or save outputs.')
        if MPI.COMM_WORLD.Get_rank() != 0:
            print('simulate once should only be done on one rank')
            exit()
        else:
            # The sim object has already been opened so the best cost doesn't need to be opened
            pass

        # ___________ Run model with new parameters ________________

        # NOT NEEDED self.sim_helper.update_times(self.dt, 0.0, self.sim_time, self.pre_time)

        # run simulation and check cost
        if param_vals is None:
            if self.best_param_vals is None:
                self.best_param_vals = np.load(os.path.join(self.output_dir, 'best_param_vals.npy'))
                param_vals = self.best_param_vals
            else:
                # The sim object has already been opened so the best cost doesn't need to be opened
                param_vals = self.best_param_vals

        cost_check, obs = self.get_cost_and_obs_from_params(param_vals=param_vals, 
                                                            reset=reset, only_one_exp=only_one_exp)
        
        obs_dicts = []
        obs_arrays = []
        for obs_item in obs:                                                    
            # if return_series:
            obs_dict, obs_array = self.get_obs_output_dict(obs_item, get_all_series=True)
            obs_dicts.append(obs_dict)
            obs_arrays.append(obs_array)
            # else:
            #     obs_dict = self.get_obs_output_dict(obs_item)
            #     obs_dicts.append(obs_dict)
            #     obs_arrays.append(None)

        if self.model_type == 'casadi_python':
            cost_check = self.get_cost_ca(param_vals)
            obs_dicts = self.get_obs_ca(param_vals)

        if only_one_exp != -1:
            # only print out results if doing all experiments, otherwise cost will be strange
            return None, None

        best_cost = np.load(os.path.join(self.output_dir, 'best_cost.npy'))
        print(f'cost should be {best_cost}')
        print('cost check after single simulation is {}'.format(cost_check))

        if abs(best_cost - cost_check) > 1e-3:
            print(f'WARNING: best cost {best_cost} is not close to cost check {cost_check}')
            print(f'Something is wrong with the cost calculation')

            if os.path.exists(os.path.join(self.output_dir, f'all_outputs_with_best_param_vals_exp_0.npz')):
                print('calculating some debug metrics for this issue')

                for exp_idx in range(self.protocol_info["num_experiments"]):
                    print(f'running simulation for experiment {exp_idx} to compare best fit and this run outputs')
                    best_fit_outputs = np.load(os.path.join(self.output_dir, f'all_outputs_with_best_param_vals_exp_{exp_idx}.npz'))
                    _, _ = self.get_cost_and_obs_from_params(self.best_param_vals, reset=True, only_one_exp=exp_idx)
                    this_run_outputs = self.sim_helper.get_all_results_dict()

                    for obs_idx in range(len(obs)):
                        for key in best_fit_outputs.keys():
                            # A diagnostic that raises destroys the information it exists to
                            # print: this block runs *because* something is already wrong, and
                            # the saved run and the live one need not expose the same variables.
                            if key not in this_run_outputs:
                                print(f'parameter {key} is not in this run\'s outputs, skipping')
                                continue
                            print(f'parameter {key}')
                            best_fit_output = best_fit_outputs[key]
                            this_run_output = this_run_outputs[key]
                            print('printing for the first `10 timepoints of the output difference')
                            print(f'best fit output: {best_fit_output[:10]}')
                            print(f'this run output: {this_run_output[:10]}')
                            print(f'difference: {best_fit_output[:10] - this_run_output[:10]}')
                            print(f'relative difference: {np.abs(best_fit_output[:10] - this_run_output[:10]) / (np.abs(best_fit_output[:10]) + 1e-10)}')
            else:
                print('no best fit outputs to compare to. Run calibration to completion',
                      'and there will be automatic comparison of outputs done here')
        
            
        print(f'final obs values :')
        for idx, obs_dict in enumerate(obs_dicts):
            print(f'subexperiment {idx+1}:')
            # TODO make the printing of the obs_dict more informative
            print(obs_dict['const'])
        return obs_dicts, obs_arrays

    def set_bayesian_parameters(self, n_calls, n_initial_points, acq_func, random_state, acq_func_kwargs={}):
        if not self.param_id_method == 'bayesian':
            print('param_id is not set up as a bayesian optimization process')
            exit()
        self.optimiser_options['num_calls_to_function'] = n_calls
        self.n_initial_points = n_initial_points
        self.acq_func = acq_func  # the acquisition function
        self.random_state = random_state  # random seed
        self.acq_func_kwargs = acq_func_kwargs
        # TODO add more of the gen alg constants here so they can be changed by user.

    def close_simulation(self):
        self.sim_helper.close_simulation()

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir

def calculate_lnlikelihood(param_vals):
    """
    This function is a wrapper around the mcmc_object method
    to calculate the lnlikelihood from model simulation.
    It allows the emcee algorithm to only pickle the param_vals
    and not all the attributes of the class instance.
    """
    return mcmc_object.get_lnlikelihood_lnprior_from_params(param_vals)


def calculate_lnlikelihood_ensemble(ensemble):
    """The vectorised form: one call per sampler step, for every walker at once.

    Same wrapper trick as above -- the sampler pickles the array, never the instance.
    """
    return mcmc_object.get_lnlikelihood_lnprior_from_ensemble(ensemble)


def drop_unsampled_draws(samples):
    """A chain cut back to the draws every walker actually reached.

    Only a *partial* pyMC chain has anything to drop. Its chains are sampled one after another
    (``cores=1``), so the file written mid-run carries NaN where a chain has not got to a draw
    yet -- see ``pymc_backend._LiveChainWriter``. A finished chain is dense and this returns it
    untouched.

    It matters because a cancelled or killed run leaves that partial file exactly where the
    finished one would be, and every statistic downstream (``np.mean``, ``np.percentile``,
    arviz's ESS and R-hat) turns a single NaN into a NaN answer for the whole parameter.
    Truncating to the shortest chain is the conservative reading: every walker is then a real
    chain of the same length, which is the rectangle the rest of the code is written against.

    Deliberately not ``np.nan_to_num`` or a per-walker compaction: substituting zeros invents
    draws, and letting walkers have different lengths would push the raggedness into every
    consumer instead of resolving it here.
    """
    samples = np.asarray(samples)
    if samples.ndim != 3 or not np.isnan(samples).any():
        return samples
    complete = ~np.isnan(samples).any(axis=(1, 2))
    if not complete.any():
        return samples[:0]
    # Draws are contiguous from the start, so the first gap ends the usable chain.
    first_gap = np.argmax(~complete) if (~complete).any() else len(complete)
    return samples[:first_gap]


def save_chain_atomically(path, samples):
    """Write ``samples`` to ``path`` so a concurrent reader never sees half an array.

    The chain is written *while* it is being sampled, so something else is expected to be
    polling this file -- a front-end drawing the run, or a user with a notebook open. Writing in
    place would leave a window in which the file is a truncated array, and ``np.load`` on that
    raises rather than returning fewer steps. Writing beside it and renaming closes the window:
    ``os.replace`` is atomic, so a reader sees either the previous chain or the new one.

    ``np.save`` appends ``.npy`` to any path that lacks it, which would turn the temporary name
    into ``...npy.tmp.npy`` and leave it behind; handing it an open file avoids that.
    """
    tmp_path = f'{path}.tmp'
    with open(tmp_path, 'wb') as tmp_file:
        np.save(tmp_file, samples)
    os.replace(tmp_path, path)


def sample_with_checkpoints(sampler, initial_state, num_steps, save_chain, save_every,
                            **sample_kwargs):
    """Run ``sampler`` for ``num_steps``, saving the chain so far every ``save_every`` steps.

    ``run_mcmc`` is one blocking call that returns only when sampling is done, which is why the
    chain used to reach disk once, hours in. ``sample()`` is the same loop as a generator, so
    checkpointing is just doing something on the way round.

    A backend with no generator form can still checkpoint, if it has some hook of its own to do
    it from: one that sets ``saves_own_checkpoints`` is handed ``save_chain`` and ``save_every``
    and is trusted to call the hook itself. pyMC is that case -- ``pm.sample`` cannot be stepped
    but does take a per-draw callback -- and without this it took the fallback below, so a pyMC
    run's chain appeared only at the end however ``chain_save_every`` was set.

    Falls back to ``run_mcmc`` when ``save_every`` is non-positive (checkpointing off) or the
    sampler has neither ``sample`` nor its own checkpointing -- zeus is driven through the same
    code path here, and a backend that only offers ``run_mcmc`` should keep working rather than
    raise.

    Returns the number of checkpoints written, which is what a test can assert on without
    reaching into the filesystem.
    """
    sample = getattr(sampler, 'sample', None)
    if save_every > 0 and getattr(sampler, 'saves_own_checkpoints', False):
        # Counted here rather than trusted to the backend, so "how many were written" means the
        # same thing -- calls to this hook -- whichever route produced them.
        checkpoints = 0

        def counted_save(samples):
            nonlocal checkpoints
            checkpoints += 1
            save_chain(samples)

        sampler.run_mcmc(initial_state, num_steps, save_chain=counted_save,
                         save_every=save_every, **sample_kwargs)
        return checkpoints

    if save_every <= 0 or sample is None:
        sampler.run_mcmc(initial_state, num_steps, **sample_kwargs)
        return 0

    checkpoints = 0
    for step, _state in enumerate(sample(initial_state, iterations=num_steps, **sample_kwargs),
                                  start=1):
        # Not on the last step: the caller saves the finished chain either way, and saving the
        # same array twice in a row is pure I/O on the largest it will ever be.
        if step % save_every == 0 and step < num_steps:
            save_chain(sampler.get_chain())
            checkpoints += 1
    return checkpoints


class MCMC(ParamID):
    """
    Class for doing mcmc on opencor models
    
    # TODO check the parallelisation for this mcmc
    """

    def __init__(self, model_path,
                 obs_info, param_id_info, protocol_info, prediction_info, solver_info,
                 dt=0.01, UQ_options=None, DEBUG=False, model_type=None, mcmc_options=None,
                 use_emulator=False, emulator_dir=None, emulator_settings=None):
        super().__init__(model_path, "MCMC",
                obs_info, param_id_info, protocol_info, prediction_info, solver_info,
                dt=dt, DEBUG=DEBUG, model_type=model_type,
                use_emulator=use_emulator, emulator_dir=emulator_dir,
                emulator_settings=emulator_settings)
        self._init_mcmc(_resolve_UQ_options(UQ_options, mcmc_options), DEBUG=DEBUG)

    @classmethod
    def from_param_id(cls, engine, UQ_options=None, mcmc_options=None):
        """Adopt an already-built ``ParamID`` instead of constructing a second one.

        Building an engine compiles the model. Because ``mcmc_instead`` selects the inner
        class at *construction* time, a UQ run following a calibration had to build a second
        CVS0DParamID and pay that compile again (CUFLynx #217) -- for the same model, the same
        obs_info and the same parameters.

        ``MCMC`` only *adds* to ``ParamID``, so the built engine's state is
        exactly what it needs: adopt its ``__dict__`` (simulation helper, parsed infos,
        output_dir, and the best_param_vals of the calibration that just ran, which is what
        seeds the walkers) and then run only the MCMC-specific tail.

        The result **shares** the engine's simulation helper rather than copying it -- that is
        the point -- which is safe because UQ follows a calibration rather than running beside
        it. Do not use the engine concurrently afterwards.
        """
        obj = cls.__new__(cls)
        obj.__dict__.update(engine.__dict__)
        obj.param_id_method = "MCMC"
        obj._init_mcmc(_resolve_UQ_options(UQ_options, mcmc_options),
                       DEBUG=getattr(engine, 'DEBUG', False))
        return obj

    def _init_mcmc(self, UQ_options, DEBUG=False):
        """The MCMC-specific half of construction, shared by __init__ and from_param_id."""
        # mcmc init stuff
        self.sampler = None
        if UQ_options is not None:
            self.UQ_options = UQ_options
            if 'num_steps' not in self.UQ_options.keys(): 
                self.UQ_options['num_steps'] = 5000
                print('number of mcmc steps is not set, choosing default of 5000')
            if 'num_walkers' not in self.UQ_options.keys():
                self.UQ_options['num_walkers'] = 2*self.num_params
                print('number of mcmc walkers is not set, ',
                    'choosing default of 2*num_params')
        else:
            self.UQ_options = {}
            self.UQ_options['num_steps'] = 5000
            self.UQ_options['num_walkers'] = 2*self.num_params
            print('number of mcmc steps and walkers is not set, ',
                  'choosing defaults of 5000 and 2*num_params')

        self.DEBUG = DEBUG
        self._warned_about_flattened_weights = False
        # Weights are flattened for the likelihood MCMC samples (#193), but not for costs that
        # are reported or compared against the calibration -- see calibration_weighting().
        self._flatten_weights = True
        assert_mle_cost_for_bayesian(
            self.cost_type, self.cost_funcs_dict, "MCMC (log-likelihood uses -cost)"
        )

    def _build_sampler(self, pool=None, vectorize=False):
        """The sampler named by ``UQ_options['library']``, behind emcee's interface.

        Every backend here exposes ``run_mcmc`` and ``get_chain`` returning a
        ``(steps, walkers, params)`` chain, so the sampling loop, the saved ``mcmc_chain.npy``
        and everything downstream of it are the same whichever was chosen.
        """
        library = (self.UQ_options or {}).get('library', 'emcee')
        num_walkers = self.UQ_options['num_walkers']

        if library == 'emcee':
            if vectorize:
                return emcee.EnsembleSampler(num_walkers, self.num_params,
                                             calculate_lnlikelihood_ensemble, vectorize=True)
            if pool is not None:
                return emcee.EnsembleSampler(num_walkers, self.num_params,
                                             calculate_lnlikelihood, pool=pool)
            return emcee.EnsembleSampler(num_walkers, self.num_params, calculate_lnlikelihood)

        if library == 'zeus':
            if vectorize and zeus is not None:
                return zeus.EnsembleSampler(num_walkers, self.num_params,
                                            calculate_lnlikelihood_ensemble, vectorize=True)
            if zeus is None:
                raise ImportError("UQ_options library 'zeus' was selected but zeus is not "
                                  "installed.")
            if pool is not None:
                return zeus.EnsembleSampler(num_walkers, self.num_params,
                                            calculate_lnlikelihood, pool=pool)
            return zeus.EnsembleSampler(num_walkers, self.num_params, calculate_lnlikelihood)

        if library == 'pymc':
            # Imported here, not at module level: pymc is an optional extra, and paramID is
            # imported by every calibration run.
            from libcuflynx.param_id.pymc_backend import PyMCSampler
            return PyMCSampler(
                num_walkers, self.num_params, calculate_lnlikelihood,
                param_id_info=self.param_id_info,
                num_tune=self.UQ_options.get('num_tune', 1000),
                method=self.UQ_options.get('pymc_method', 'mcmc'))

        raise ValueError(
            f"unknown UQ_options library {library!r}. Valid options are 'emcee' and 'pymc'.")

    def _cost_weight_vectors(self, exp_idx, sub_idx):
        """Every feature entering the likelihood carries equal weight (issue #193).

        Calibration weights are a modelling choice: they say which features the optimiser should
        care about most. A posterior is not. Under ``ln L = -cost``, a weight w on a feature
        raises its likelihood term to the power w, which is the same as claiming w independent
        observations of it -- so a feature weighted 10 shrinks the posterior as if it had been
        measured ten times, and the credible intervals that come out are not the ones the data
        supports. The relative weighting between features distorts their trade-off in the same
        way.

        Zero weights are preserved: a zero does not mean "unimportant", it means the observable
        is not part of this sub-experiment at all, and reinstating it would add a feature the
        user excluded. The non-zero count is therefore unchanged, so the cached
        ``_num_weighted_obs_by_exp_sub`` denominator stays correct.

        Warns once when this actually changed something, so a user who tuned weights for a
        calibration and then ran UQ on the same obs_data finds out that they no longer apply.
        """
        vectors = super()._cost_weight_vectors(exp_idx, sub_idx)
        if not getattr(self, '_flatten_weights', True):
            return vectors
        flattened = tuple(np.asarray(vec != 0, dtype=float) for vec in vectors)

        if not getattr(self, '_warned_about_flattened_weights', False):
            for original, flat in zip(vectors, flattened):
                original = np.asarray(original, dtype=float)
                if original.size and not np.allclose(original, flat):
                    print(
                        'WARNING: obs_data weights are ignored for UQ -- every feature entering '
                        'the likelihood is weighted 1. A weighted likelihood is not a posterior: '
                        'a weight w on a feature is the same claim as w independent observations '
                        'of it, so it would shrink the credible intervals by a factor the data '
                        'does not support (issue #193). Weights of 0 still exclude an observable.'
                    )
                    self._warned_about_flattened_weights = True
                    break

        return flattened

    @contextlib.contextmanager
    def calibration_weighting(self):
        """Evaluate costs inside this block with the obs_data weights, not the flat ones.

        The flattening exists for the *likelihood being sampled*: a weighted likelihood is not a
        posterior (#193). It must not follow the cost out into the artifacts, because best_cost
        is a calibration artifact -- plot_param_id and simulate_once re-derive it and compare
        against the saved value, and the calibration itself optimised the weighted cost. A
        best_cost written on the flat scale is simply a different quantity under the same name,
        and the two disagree by whatever the weights were (measured on 3compartment: 0.0377 saved
        against 0.1058 recomputed, which tripped simulate_once's consistency check).
        """
        previous = getattr(self, '_flatten_weights', True)
        self._flatten_weights = False
        try:
            yield
        finally:
            self._flatten_weights = previous

    def run(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        num_procs = comm.Get_size()
        if rank == 0:
            print('Running mcmc')


        if self.can_vectorise_the_ensemble():
            # One process, one batched surrogate call per step. A worker pool here would
            # split sixty-four cheap evaluations across ranks and pay an MPI round trip for
            # each -- measured at 1.4x from seven workers, against 15x from batching the
            # same sixty-four into one call. The ranks are better off not being used.
            if rank != 0:
                # Same contract as a pool worker that is not the master: drop out without
                # entering a collective. Nothing after this is collective over COMM_WORLD.
                return
            init_param_vals = self._initial_walker_positions(ball_scale=0.1)
            self.sampler = self._build_sampler(vectorize=True)
            start_time = time.time()
            self._sample(init_param_vals, progress=True, tune=True)
            print(f'mcmc time = {time.time() - start_time}')

        elif num_procs > 1 and self.sampler_needs_a_worker_pool():
            # from pathos import multiprocessing
            # from pathos.multiprocessing import ProcessPool
            from schwimmbad import MPIPool

            if rank == 0:
                init_param_vals = self._initial_walker_positions(ball_scale=0.1)

            try:
                pool = MPIPool() # workers dont get past this line in this try, they wait for work to do
            except:
                return

            if not pool.is_master():
                pool.wait()
                return

            self.sampler = self._build_sampler(pool=pool)

            start_time = time.time()
            self._sample(init_param_vals, progress=True, tune=True)
            print(f'mcmc time = {time.time() - start_time}')
            pool.close()

        elif num_procs > 1:
            # Every rank samples its own chains and they are gathered at the end, so no rank is
            # a worker and no pool is opened. See sampler_needs_a_worker_pool.
            #
            # The starting ensemble is drawn once and broadcast rather than drawn per rank: the
            # ranks must agree on it (best_param_vals is only guaranteed on rank 0, so drawing
            # locally would put some ranks in a ball around the calibrated fit and others on a
            # uniform draw over the whole prior box), and each rank then takes the slice of
            # walkers it is responsible for so the gathered chain covers the ensemble asked for
            # rather than repeating its first few walkers on every rank.
            init_param_vals = self._initial_walker_positions(ball_scale=0.1) if rank == 0 else None
            init_param_vals = comm.bcast(init_param_vals, root=0)

            self.sampler = self._build_sampler()

            start_time = time.time()
            self._sample(self._walkers_for_rank(init_param_vals, rank, num_procs),
                         progress=True, tune=True)
            if rank == 0:
                print(f'mcmc time = {time.time() - start_time}')

        else:
            init_param_vals = self._initial_walker_positions(ball_scale=0.01)

            self.sampler = self._build_sampler()

            start_time = time.time()
            self._sample(init_param_vals) # , progress=True)
            print(f'mcmc time = {time.time()-start_time}')

        if rank == 0:
            if hasattr(self.sampler, 'acceptance_fraction'):
                print(f'acceptance fraction was {self.sampler.acceptance_fraction}')
            samples = self.sampler.get_chain()
            mcmc_chain_path = self.mcmc_chain_path()
            save_chain_atomically(mcmc_chain_path, samples)
            print('mcmc complete')
            print(f'mcmc chain saved in {mcmc_chain_path}')

            flat_samples = samples[self.burn_in_index(samples.shape[0]):, :, :].reshape(
                -1, self.num_params)
            self.save_mcmc_statistics(flat_samples)

    def can_vectorise_the_ensemble(self):
        """Whether the whole walker population can be evaluated in one call.

        Two conditions, and both are about what the likelihood *is*:

        * The forward model has to be a surrogate. Batching wins because a fitted
          regressor costs almost the same at sixty-four points as at one; a solver
          integrates sixty-four times either way and gains nothing, while losing the
          worker pool that was genuinely parallelising it.
        * The sampler has to advance a whole ensemble at once and accept a vectorised
          log-probability. emcee and zeus both do; pyMC has no such hook and parallelises
          in the opposite direction, by giving each rank chains of its own.
        """
        if not getattr(self, 'emulates_features', False):
            return False
        if not getattr(getattr(self, 'sim_helper', None), 'predict_ensemble', None):
            return False
        return (self.UQ_options or {}).get('library', 'emcee') in ('emcee', 'zeus')

    def sampler_needs_a_worker_pool(self):
        """Whether this backend parallelises across ranks by farming out the likelihood.

        The two backends parallelise in opposite directions, and running one arrangement under
        the other hangs:

        * **emcee and zeus** advance one ensemble in one process. The parallelism is the
          likelihood: the sampler is handed a ``schwimmbad.MPIPool``, and every other rank sits
          in ``pool.wait()`` serving evaluations until the master closes the pool.
        * **pyMC** has no such hook. ``PyMCSampler`` instead gives each rank
          ``chains_for_rank(...)`` chains of its own and gathers them along the walker axis at
          the end -- which means every rank must reach ``run_mcmc``, and its ``comm.Barrier()``
          and ``comm.gather`` are collectives over COMM_WORLD.

        Opening a pool for pyMC therefore deadlocked every ``mpiexec -n >1`` UQ run with
        ``library: pymc``: the workers were parked inside ``pool.wait()``, blocked in a receive
        that only the master's ``pool.close()`` ends, so they could never join the master's
        barrier -- and the master waits on that barrier forever, holding the pool open. Neither
        side can move, and the run hangs after sampling with no error and no chain written. It
        was never seen because nothing exercised it: the pyMC tests run on one rank, where this
        branch is not taken at all.
        """
        return (self.UQ_options or {}).get('library', 'emcee') in _POOL_BACKED_UQ_LIBRARIES

    def _initial_walker_positions(self, ball_scale):
        """Starting positions for the ensemble, ``(num_walkers, num_params)``.

        A gaussian ball of relative width ``ball_scale`` around the calibrated fit, in
        normalised space, or a uniform draw over the prior box when there is no fit to start
        from. Was written out twice, once per branch of ``run``, with the two copies differing
        only in ``ball_scale`` -- a third branch is not worth a third copy.
        """
        num_walkers = self.UQ_options['num_walkers']
        if self.best_param_vals is not None:
            best_param_vals_norm = self.param_norm_obj.normalise(self.best_param_vals)
            # create initial params in gaussian ball around best_param_vals estimate
            init_param_vals_norm = (np.ones((num_walkers, self.num_params))*best_param_vals_norm).T + \
                               ball_scale*np.random.randn(self.num_params, num_walkers)
            init_param_vals_norm = np.clip(init_param_vals_norm, 0.001, 0.999)
        else:
            init_param_vals_norm = np.random.rand(self.num_params, num_walkers)
        return self.param_norm_obj.unnormalise(init_param_vals_norm).T

    @staticmethod
    def _walkers_for_rank(init_param_vals, rank, num_procs):
        """The slice of the starting ensemble this rank is responsible for.

        Mirrors ``PyMCSampler.chains_for_rank``, which decides how many chains the rank runs
        from the same two numbers -- if these two disagree, a rank either starts chains from
        another rank's positions or is handed positions it never uses.

        Indices wrap, because ``chains_for_rank`` never returns zero: with more ranks than
        walkers every rank still runs one chain, and the ranks past the end of the ensemble
        start over at its beginning rather than being handed nothing.
        """
        num_walkers = len(init_param_vals)
        per_rank = max(1, num_walkers // num_procs)
        return init_param_vals[(rank*per_rank + np.arange(per_rank)) % num_walkers]

    def mcmc_chain_path(self):
        """Where the chain is written -- the same path during the run as at the end of it.

        Deliberately one file rather than a partial one that is renamed at the end: everything
        that reads a chain (``load_mcmc_chain``, the plotters, a front-end) then needs no notion
        of "the run has finished", and a run that is cancelled or killed leaves its chain exactly
        where the tooling already looks for it.
        """
        return os.path.join(self.output_dir, 'mcmc_chain.npy')

    def _sample(self, initial_state, **sample_kwargs):
        """Sample, leaving a readable chain behind on the way rather than only at the end."""
        return sample_with_checkpoints(
            self.sampler, initial_state, self.UQ_options['num_steps'],
            lambda samples: save_chain_atomically(self.mcmc_chain_path(), samples),
            self.UQ_options.get('chain_save_every', 50),
            **sample_kwargs)

    def burn_in_index(self, num_steps):
        """The first step to keep, from ``UQ_options['burn_in']``.

        A value below 1 is a fraction of the chain; 1 or above is a number of steps. Defaults to
        half the chain, which is what this used to hardcode. Always leaves at least one step, so
        a burn_in longer than the run degrades to "keep the last sample" rather than producing an
        empty array and a stack of nan statistics.
        """
        burn_in = (self.UQ_options or {}).get('burn_in', 0.5)
        try:
            burn_in = float(burn_in)
        except (TypeError, ValueError):
            print(f"WARNING: UQ_options burn_in {burn_in!r} is not a number; using half the chain.")
            burn_in = 0.5

        index = int(num_steps * burn_in) if burn_in < 1 else int(burn_in)
        if index >= num_steps:
            print(f'WARNING: burn_in discards all {num_steps} steps of the chain; '
                  f'keeping the last one. Run more steps, or lower burn_in.')
            index = num_steps - 1
        return max(0, index)

    def flat_param_names(self):
        """One name per calibrated parameter, for labelling the statistics.

        A grouped row calibrates one value shared across several model variables, so it is one
        parameter with several names; the first stands for the group, as it does elsewhere.
        """
        # len(), never truthiness: param_id_info holds these as numpy arrays, and `not array`
        # raises rather than answering.
        info = self.param_id_info or {}
        names = info.get('param_names_for_plotting')
        if names is None or len(names) != self.num_params:
            names = info.get('param_names')
        if names is None:
            names = []
        flat = []
        for idx in range(self.num_params):
            if idx < len(names):
                entry = names[idx]
                flat.append(str(entry[0] if isinstance(entry, (list, tuple)) else entry))
            else:
                flat.append(f'param_{idx}')
        return flat

    def posterior_statistics(self, flat_samples):
        """Per-parameter summary of the posterior, plus the cost at its point summaries.

        A posterior is a distribution, and the honest summary of one is a spread rather than a
        single number -- so mean *and* median *and* the quartiles and the 95% interval, not a
        winner. The costs are reported for comparison with the calibration's best, deliberately
        without acting on that comparison: see save_mcmc_statistics.
        """
        flat_samples = np.asarray(flat_samples, dtype=float)
        names = self.flat_param_names()

        means = flat_samples.mean(axis=0)
        medians = np.median(flat_samples, axis=0)

        stats = {}
        for idx, name in enumerate(names):
            column = flat_samples[:, idx]
            stats[name] = {
                'mean': float(means[idx]),
                'median': float(medians[idx]),
                'sd': float(np.std(column, ddof=1)) if column.size > 1 else float('nan'),
                'q2.5': float(np.percentile(column, 2.5)),
                'q25': float(np.percentile(column, 25)),
                'q75': float(np.percentile(column, 75)),
                'q97.5': float(np.percentile(column, 97.5)),
                'min': float(np.min(column)),
                'max': float(np.max(column)),
            }
        return stats, means, medians

    def calibration_best_cost(self):
        """The calibration's best cost as a finite float, or None if there isn't one.

        Prefers the value on disk: a UQ run is often handed the calibration's *parameters*
        (set_best_param_vals) without its cost, leaving the in-memory best_cost at inf. Reporting
        inf would put a JSON Infinity in the file -- which strict JSON parsers reject -- and make
        every posterior median look like it had beaten the calibration.
        """
        candidates = [self.best_cost]
        path = os.path.join(self.output_dir, 'best_cost.npy')
        if os.path.isfile(path):
            try:
                candidates.append(np.load(path))
            except Exception:
                pass
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                value = float(np.ravel(candidate)[0])
            except (TypeError, ValueError, IndexError):
                continue
            if np.isfinite(value):
                return value
        return None

    def save_mcmc_statistics(self, flat_samples):
        """Write ``mcmc_statistics.json``, and leave the calibration's best fit alone.

        This used to overwrite best_param_vals.npy and best_cost.npy with the posterior median
        whenever that median scored a lower cost. Two different estimators were being conflated:
        a posterior median summarises a distribution, a calibration best is an argmin, and they
        answer different questions. Silently replacing one with the other meant a UQ run mutated
        the calibration's answer -- and the file gave no clue which estimator it held.

        The comparison is still reported, because it is genuinely informative (a median that
        beats the optimum usually means the calibration stopped early, or that the posterior is
        skewed), but nothing is decided on it. Choosing between the two is the user's call, and
        both are now on disk to choose from.

        The one exception is a UQ run with no calibration behind it at all: nothing else has
        written a best fit, so the median is the only estimate there is, and the rest of the
        pipeline (plotting, predictions) needs one. That is recorded in the file's `source`.
        """
        stats, means, medians = self.posterior_statistics(flat_samples)

        # Weighted like the calibration, not like the likelihood: these sit in the same file as
        # calibration_best_cost and are read against it, so all three have to be the same
        # quantity. The flat weighting (#193) is for the likelihood being sampled, and must not
        # follow the cost out into the artifacts.
        with self.calibration_weighting():
            median_cost = float(np.ravel(
                self.get_cost_and_obs_from_params(medians, reset=True)[0])[0])
            mean_cost = float(np.ravel(
                self.get_cost_and_obs_from_params(means, reset=True)[0])[0])

        document = {
            'parameters': stats,
            'num_samples': int(np.asarray(flat_samples).shape[0]),
            'cost_at_posterior_median': median_cost,
            'cost_at_posterior_mean': mean_cost,
            'calibration_best_cost': self.calibration_best_cost(),
        }

        if self.best_param_vals is None:
            # No calibration behind this run, so this is not an overwrite: it is the only
            # estimate available, and downstream plotting/prediction needs one.
            self.best_param_vals = medians
            self.best_cost = median_cost
            document['source'] = 'posterior_median'
            document['calibration_best_cost'] = None
            np.save(os.path.join(self.output_dir, 'best_cost'), self.best_cost)
            np.save(os.path.join(self.output_dir, 'best_param_vals'), self.best_param_vals)
            print('No calibration best fit existed, so best_param_vals and best_cost were '
                  'written from the posterior median.')
        else:
            document['source'] = 'calibration'
            calibration_cost = document['calibration_best_cost']
            reported = 'unknown' if calibration_cost is None else calibration_cost
            print(f'cost at the posterior median is {median_cost}, at the posterior mean is '
                  f'{mean_cost}; the calibration best fit ({reported}) is left unchanged '
                  f'(a posterior median is not an optimum -- see mcmc_statistics.json for the '
                  f'full posterior summary).')
            # Only when there is a real number to beat. The engine's in-memory best_cost is inf
            # on a UQ run that adopted a calibration's parameters without its cost, and
            # "lower than inf" is true of everything.
            if calibration_cost is not None and median_cost < calibration_cost:
                print('NOTE: the posterior median scores a lower cost than the calibration best '
                      'fit. That usually means the calibration stopped early or the posterior '
                      'is skewed. Both are on disk; choosing between them is yours to make.')

        path = os.path.join(self.output_dir, 'mcmc_statistics.json')
        with open(path, 'w') as write_file:
            json.dump(document, write_file, indent=2)
        print(f'mcmc statistics saved in {path}')
        return document

    def calculate_pred_from_posterior_samples(self, flat_samples, n_sims=100):
        # idxs of output are [exp_idx][sim_idx, pred_idx, time_idx]
        
        pred_arrays_per_exp_list= []
        for exp_idx in list(set(self.prediction_info['experiment_idxs'])):
            pred_list = []
            for sim_idx in range(n_sims):
                rand_idx = np.random.randint(0, len(flat_samples)-1)
                sample_param_vals = flat_samples[rand_idx, :]
                pred_outputs = self.get_pred_array_from_params_per_exp(sample_param_vals, exp_idx)
                
                pred_list.append(pred_outputs)
                    
                # TODO shouldn't fail here because each mcmc sample ran..., 
                # TODO but if it does, we need to catch it
                self.sim_helper.reset_and_clear()
            pred_arrays_per_exp_list.append(np.array(pred_list))
            # can't all be one array because the number of timepoints
            # can be different between experiments.
        
        # idxs of output are [exp_idx][sim_idx, pred_idx, time_idx]
        return pred_arrays_per_exp_list

#: The names these two classes had until they were renamed. Neither ever had anything to
#: do with OpenCOR: they are the parameter-identification and MCMC engines, and they run
#: against myokit/CVODE, casadi and trained emulators as readily as against OpenCOR. The
#: name came from the one backend that existed when they were written.
#:
#: Kept because they are imported by name from outside this repository -- CUFLynx reaches
#: for ``OpencorParamID`` through its ca_import shim -- and a rename that breaks a
#: downstream import on upgrade is a rename that gets reverted. They are aliases, not
#: subclasses, so isinstance and pickling behave identically.
OpencorParamID = ParamID
OpencorMCMC = MCMC


class MCMC_plotter:
    """
    This class contains plotting wrapper for mcmc
    """

    def __init__(self, model_path, model_type, param_id_method, file_name_prefix,
                 params_for_id_path=None, num_calls_to_function=1000,
                 param_id_obs_path=None, sim_time=2.0, pre_time=20.0, 
                 solver_info=None, 
                 dt=0.01, UQ_options=None, mcmc_options=None,
                 param_id_output_dir=None, resources_dir=None,
                 DEBUG=False):

        self.model_path = model_path
        self.model_type = model_type
        self.param_id_method = param_id_method
        self.file_name_prefix = file_name_prefix
        self.params_for_id_path = params_for_id_path
        self.num_calls_to_function = num_calls_to_function
        self.param_id_obs_path = param_id_obs_path
        self.sim_time = sim_time
        self.pre_time = pre_time
        self.solver_info = solver_info
        self.dt = dt
        self.DEBUG =DEBUG
        
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        
        self.param_id_obs_file_prefix = re.sub(r"\.json", "", os.path.split(param_id_obs_path)[1])
        case_type = f'{param_id_method}_{file_name_prefix}_{self.param_id_obs_file_prefix}'
        if self.rank == 0:
            if param_id_output_dir is None:
                self.param_id_output_dir = default_param_id_output_dir()
            else:
                self.param_id_output_dir = param_id_output_dir
            
            if not os.path.exists(self.param_id_output_dir):
                os.mkdir(self.param_id_output_dir)
            self.output_dir = os.path.join(self.param_id_output_dir, f'{case_type}')
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)
            self.plot_dir = os.path.join(self.output_dir, 'plots_param_id')
            if not os.path.exists(self.plot_dir):
                os.mkdir(self.plot_dir)
        
        if resources_dir is None:
            self.resources_dir = default_resources_dir()
        else:
            self.resources_dir = resources_dir


        self.best_param_vals = None
        self.best_param_names = None

        self.UQ_options = _resolve_UQ_options(UQ_options, mcmc_options)

        # thresholds for identifiability TODO optimise these
        self.threshold_param_importance = 0.1
        self.keep_threshold_param_importance = 0.8
        self.threshold_collinearity = 20
        self.threshold_collinearity_pairs = 10
        self.second_deriv_threshold = -1000

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.num_procs = self.comm.Get_size()

    def plot_mcmc_and_predictions(self, mcmc=None):
        if self.rank != 0:
            return
        if mcmc == None:
            print('creating mcmc object')
            if self.rank == 0:
                mcmc = CVS0DParamID(self.model_path, self.model_type, self.param_id_method, True,
                                    self.file_name_prefix,
                                    params_for_id_path=self.params_for_id_path,
                                    param_id_obs_path=self.param_id_obs_path,
                                    sim_time=self.sim_time, pre_time=self.pre_time, dt=self.dt,
                                    param_id_output_dir=self.param_id_output_dir, resources_dir=self.resources_dir,
                                    solver_info=self.solver_info, UQ_options=self.UQ_options,
                                    DEBUG=self.DEBUG, one_rank=True)
                if os.path.exists(os.path.join(mcmc.output_dir, 'param_names_to_remove.csv')):
                    with open(os.path.join(mcmc.output_dir, 'param_names_to_remove.csv'), 'r') as r:
                        param_names_to_remove = []
                        for row in r:
                            name_list = row.split(',')
                            name_list = [name.strip() for name in name_list]
                            param_names_to_remove.append(name_list)
                    mcmc.remove_params_by_name(param_names_to_remove)

        if self.best_param_vals is not None:
            self.best_param_vals = np.load(os.path.join(mcmc.output_dir, 'best_param_vals.npy'))

        mcmc.set_best_param_vals(self.best_param_vals)

        print('Plotting mcmc parameter distributions')
        mcmc.plot_mcmc()
        print('Plotting core predictions distribution to check uncertainty on predictions')
        mcmc.postprocess_predictions()
        print('Plotting complete')

class ProgressBar(object):
    """
    Alternatively: Could call ProgBarLogger like in keras
    """

    def __init__(self, n_calls, n_jobs=1, file=sys.stderr):
        self.n_calls = n_calls
        self.n_jobs = n_jobs
        self.iter_no = 0
        self.file = file
        self._start_time = time.time()

    def _to_precision(self, x, precision=5):
        return ("{0:.%ie} seconds"%(precision - 1)).format(x)

    def progress(self, iter_no, curr_min):
        bar_len = 60
        filled_len = int(round(bar_len*iter_no/float(self.n_calls)))

        percents = round(100.0*iter_no/float(self.n_calls), 1)
        bar = '='*filled_len + '-'*(bar_len - filled_len)
        print(f'[{bar}] {percents}% | Elapsed Time: {time.time() - self._start_time} | Current Minimum: {curr_min}')

    def __call__(self, res):
        curr_y = res.func_vals[-1]
        curr_min = res.fun
        self.iter_no += self.n_jobs
        self.progress(self.iter_no, curr_min)

    def call(self, res):
        self.__call__(res)

