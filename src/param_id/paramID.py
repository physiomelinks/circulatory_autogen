'''
@author: Finbar J. Argus
'''

import numpy as np
import os
import sys
from sys import exit
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../solver_wrappers'))
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
import paperPlotSetup
import stat_distributions
import diagnostics
import utility_funcs
import traceback
from utility_funcs import Normalise_class
paperPlotSetup.Setup_Plot(3)
from solver_wrappers import get_simulation_helper
from protocol_runners.protocol_executor import ProtocolExecutor
from parsers.PrimitiveParsers import scriptFunctionParser
from mpi4py import MPI
import re
from numpy import genfromtxt
from importlib import import_module
# import tqdm # TODO this needs to be installed for corner plot but doesnt need an import here
mcmc_lib = 'emcee' # TODO make this a user variable
if mcmc_lib == 'emcee':
    try:
        import emcee
    except ImportError:
        emcee = None
elif mcmc_lib == 'zeus':
    try:
        import zeus
    except ImportError:
        zeus = None
else:
    print(f'unknown mcmc lib : {mcmc_lib}')
try:
    import corner
except ImportError:
    corner = None
import csv
import shutil
from datetime import date, datetime
# from skopt import gp_minimize, Optimizer
from parsers.PrimitiveParsers import CSVFileParser, ObsAndParamDataParser, PARAM_ID_METHODS
from param_id.optimisers import GeneticAlgorithmOptimiser, BayesianOptimiser, CMAESOptimiser, \
    SciPyMinimizeOptimiser, MultiStartSciPyMinimizeOptimiser
from param_id.differentiable import (
    assert_casadi_differentiable,
    assert_mle_cost_for_bayesian,
    is_circulatory_differentiable,
)
from param_id.plot_outputs import ParamIDPlotOutputs
from param_id import casadi_backend
from param_id import fsa_backend
from param_id import aadc_backend
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
warnings.filterwarnings( "ignore", module = "matplotlib/..*" )
# TODO maybe remove matplotlib warnings as above

# set resource limit to inf to stop seg fault problem #TODO remove this, I don't think it does much
# import resource
# curlimit = resource.getrlimit(resource.RLIMIT_STACK)
# resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY,resource.RLIM_INFINITY))

# This mcmc_object will be an instance of the OpencorParamID class
# it needs to be global so that it can be used in calculate_lnlikelihood()
# without having its attributes pickled. opencor simulation objects
# can't be pickled because they are pyqt.
mcmc_object = None


def ensure_mle_cost_type_for_bayesian_inner(inner, inp_data_dict):
    """
    Set ``obs_info['cost_type']`` on an OpencorParamID / OpencorMCMC instance so every
    observable uses an ``@is_MLE`` cost (required for ``ln L = -cost`` in MCMC / Laplace).

    Chooses the first ``cost_type`` string found in optimiser / mcmc option dicts in
    ``inp_data_dict`` that names an ``@is_MLE`` cost in ``inner.cost_funcs_dict``;
    otherwise ``gaussian_MLE``.
    """
    if inner is None or getattr(inner, "obs_info", None) is None:
        return
    costs = getattr(inner, "cost_funcs_dict", None) or {}
    chosen = None
    option_dicts = []
    if inp_data_dict.get("DEBUG"):
        option_dicts.append(inp_data_dict.get("debug_optimiser_options") or {})
        option_dicts.append(inp_data_dict.get("debug_mcmc_options") or {})
    option_dicts.append(inp_data_dict.get("optimiser_options") or {})
    option_dicts.append(inp_data_dict.get("mcmc_options") or {})
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
    inner.obs_info["cost_type"] = [chosen] * n
    inner.cost_type = inner.obs_info["cost_type"]


# Re-exported for backwards compatibility; the canonical definition is in param_id.aadc_backend.
TAPE_CONSISTENT_AADC_METHODS = aadc_backend.TAPE_CONSISTENT_METHODS

# The CasADi symbolic cost/gradient/observable machinery lives in param_id.casadi_backend; the
# methods below delegate to it. require_casadi/as_casadi_column are re-bound to their previous
# private names because the *generic* cost-assembly layer that stays here (cost_calc,
# get_obs_output_dict) still builds SX expressions directly and calls them.
_require_casadi = casadi_backend.require_casadi
_as_casadi_column = casadi_backend.as_casadi_column


class CVS0DParamID():
    """Parameter identification (calibration) for a 0D CVS model.

    This is the main user-facing entry point for calibration. It wraps an inner
    optimisation engine ([`OpencorParamID`][param_id.paramID.OpencorParamID], or
    [`OpencorMCMC`][param_id.paramID.OpencorMCMC] when ``mcmc_instead=True``) and
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
        model_type: One of ``'cellml_only'``, ``'python'``, ``'casadi_python'``.
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
        mcmc_options: Options dict for MCMC (used when ``mcmc_instead=True``).
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
                 solver_info=None, mcmc_options=None, optimiser_options=None, 
                 do_ad=False, DEBUG=False,
                 param_id_output_dir=None, resources_dir=None, one_rank=False):
        self.model_path = model_path
        self.param_id_method = param_id_method
        self.mcmc_instead = mcmc_instead
        self.model_type = model_type
        self.file_name_prefix = file_name_prefix

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.num_procs = self.comm.Get_size()

        self.mcmc_options = mcmc_options
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
                self.param_id_output_dir = os.path.join(os.path.dirname(__file__), '../../param_id_output')
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
            self.resources_dir = os.path.join(os.path.dirname(__file__), '../../resources')
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
        self.obs_and_param_parser = ObsAndParamDataParser()
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
            # This mcmc_object will be an instance of the OpencorParamID class
            # it needs to be global so that it can be used in calculate_lnlikelihood()
            # without having its attributes pickled. opencor simulation objects
            # can't be pickled because they are pyqt.
            global mcmc_object 
            mcmc_object = OpencorMCMC(self.model_path,
                                           self.obs_info, self.param_id_info,
                                           self.protocol_info, self.prediction_info, self.solver_info, dt=self.dt,
                                           mcmc_options=mcmc_options,
                                           DEBUG=self.DEBUG, model_type=self.model_type)
            self.n_steps = mcmc_object.n_steps
        else:
            if model_type in ['cellml_only', 'python', 'casadi_python', 'aadc_python', 'python_user_defined']:
                self.param_id = OpencorParamID(self.model_path, self.param_id_method,
                                               self.obs_info, self.param_id_info, self.protocol_info,
                                               self.prediction_info, self.solver_info, dt=self.dt,
                                               optimiser_options=self.optimiser_options, 
                                               do_ad=do_ad, DEBUG=self.DEBUG, 
                                               model_type=self.model_type)
                self.n_steps = self.param_id.n_steps
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
            'sim_time', 'pre_time', 'dt', 'solver_info', 'mcmc_options',
            'optimiser_options', 'DEBUG', 'param_id_output_dir', 'resources_dir',
            'one_rank', 'do_ad',
        ]
        kwargs = {key: inp_data_dict[key] for key in arg_options if key in inp_data_dict}

        # Support common naming used elsewhere
        if 'file_name_prefix' not in kwargs and 'file_prefix' in inp_data_dict:
            kwargs['file_name_prefix'] = inp_data_dict['file_prefix']

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

    def run_mcmc(self):
        """Run MCMC sampling (requires the instance was built with ``mcmc_instead=True``)."""
        mcmc_object.run()
    
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

class OpencorParamID():
    """
    Class for doing parameter identification on opencor models
    """
    def __init__(self, model_path, param_id_method,
                 obs_info, param_id_info, protocol_info, prediction_info,
                 solver_info, dt=0.01, 
                 optimiser_options=None, do_ad=False, 
                 DEBUG=False, model_type=None):

        self.model_path = model_path
        self.param_id_method = param_id_method
        self.output_dir = None
        self.model_type = model_type

        self.solver_info = solver_info
        self.obs_info = obs_info
        self.param_id_info = param_id_info
        self.prediction_info = prediction_info # currently not used
        self.optimiser_options = optimiser_options
        if self.param_id_info is not None:
            self.num_params = len(self.param_id_info["param_names"])
            self.param_norm_obj = Normalise_class(self.param_id_info["param_mins"], self.param_id_info["param_maxs"])

        self.protocol_info = protocol_info

        self.sfp = scriptFunctionParser()

        if self.model_type == "casadi_python":
            mode = "casadi"
        elif self.model_type == "aadc_python":
            mode = "numpy"  # AADC uses numpy for passive (non-tape) cost evaluation
        else:
            mode = "numpy"
        self.operation_funcs_dict = self.sfp.get_operation_funcs_dict(mode)
        self.cost_funcs_dict = self.sfp.get_cost_funcs_dict(mode)

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

        self.sim_helper = self.initialise_sim_helper()
        self._protocol_executor = ProtocolExecutor(self.sim_helper)

        if self.sim_time is not None and self.pre_time is not None:
            self.sim_helper.update_times(self.dt, 0.0, self.sim_time, self.pre_time)
            self.n_steps = int(self.sim_time/self.dt)
        else:
            self.n_steps = None

        offline_pre_time = None
        if self.protocol_info is not None:
            offline_pre_time = self.protocol_info.get('offline_pre_time')
        if offline_pre_time is not None and float(offline_pre_time) > 0:
            # offline_pre_time used to be run ONCE here, freezing the resulting state into the
            # solver's default_state_inits for the whole calibration. That state is the steady
            # state of the *initial* parameter guess, and it was never recomputed as the
            # optimiser moved -- yet parameters change the steady state that is reached. Two
            # consequences: every evaluation started from a state correct only for the initial
            # params (a bias that itself varies with the params, so it distorts the cost
            # landscape); and the gradient silently lost the d(steady state)/d(p) term, because
            # FSA/AD treats that frozen state as a constant. AD-vs-FD verification cannot catch
            # the latter -- FD perturbs the same frozen state, so both agree and both are wrong.
            #
            # Fold the duration into each experiment's first-sub warm-up instead. That warm-up
            # IS re-integrated at the current parameter values on every evaluation (see
            # ProtocolExecutor, which passes pre_times[exp_idx] to update_times for sub 0), so
            # the starting state and its sensitivity are both correct. This gives up the
            # speed-up the offline pass existed to provide; reinstating a *correct* offline
            # optimisation is tracked in issue #269.
            offline_pre_time = float(offline_pre_time)
            pre_times = self.protocol_info.get('pre_times')
            if pre_times is not None:
                self.protocol_info = dict(self.protocol_info)
                self.protocol_info['pre_times'] = [
                    float(pt or 0.0) + offline_pre_time for pt in pre_times
                ]
                self.pre_time = self.protocol_info['pre_times'][0]
            else:
                self.pre_time = float(self.pre_time or 0.0) + offline_pre_time
            if self.sim_time is not None and self.pre_time is not None:
                self.sim_helper.update_times(self.dt, 0.0, self.sim_time, self.pre_time)

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
        
        if self.obs_info is not None:
            self.cost_type = self.obs_info["cost_type"]
        else:
            self.cost_type = None
        if mode == "casadi":
            assert_casadi_differentiable(
                self.obs_info, self.cost_type, self.operation_funcs_dict, self.cost_funcs_dict
            )
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
                wd = self.protocol_info["scaled_weight_prob_dist_from_exp_sub"][exp_idx][sub_idx]
                n = int(
                    np.sum(wc != 0)
                    + np.sum(ws != 0)
                    + np.sum(wa != 0)
                    + np.sum(wp != 0)
                    + np.sum(wd != 0)
                )
                row.append(n)
                total += n
            by_exp_sub.append(row)
        self._num_weighted_obs_by_exp_sub = by_exp_sub
        self._lnlikelihood_denorm_factor = float(total) if total > 0 else 1.0

    def initialise_sim_helper(self):
        # Get method from solver_info (check both 'solver' and 'method' for backward compatibility)
        solver = self.solver_info.get('solver')
        helper_cls = get_simulation_helper(solver=solver, model_type=self.model_type,
                                           model_path=self.model_path, dt=self.dt, sim_time=self.sim_time,
                                           solver_info=self.solver_info, pre_time=self.pre_time)
        return helper_cls
    
    def add_user_operation_func(self, func):
        if self.model_type == "casadi_python" and not is_circulatory_differentiable(func):
            raise ValueError(
                f"User operation {func.__name__!r} must be decorated with @differentiable for casadi_python mode."
            )
        self.operation_funcs_dict = self.sfp.add_user_operation_func(self.operation_funcs_dict, func)
    
    def add_user_cost_func(self, func):
        if self.model_type == "casadi_python" and not is_circulatory_differentiable(func):
            raise ValueError(
                f"User cost function {func.__name__!r} must be decorated with @differentiable for casadi_python mode."
            )
        self.cost_funcs_dict = self.sfp.add_user_cost_func(self.cost_funcs_dict, func)
    
    def set_best_param_vals(self, best_param_vals):
        self.best_param_vals = best_param_vals
    
    def set_param_names(self, param_names):
        self.param_id_info["param_names"] = param_names
        self.num_params = len(self.param_id_info["param_names"])
    
    def set_param_id_info(self, param_id_info):
        self.param_id_info = param_id_info
        self.num_params = len(self.param_id_info["param_names"])
        self.param_norm_obj = Normalise_class(self.param_id_info["param_mins"], self.param_id_info["param_maxs"])
    
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
        self._refresh_num_weighted_obs_tables()

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
        self.param_init = self.sim_helper.get_init_param_vals(self.param_id_info["param_names"])

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

        # Run the protocol loop via the shared ProtocolExecutor.
        # reset_after_experiment mirrors the original `reset` flag: when do_ad=True
        # (reset=False) the solver state must be preserved across experiments.
        sim_success, results_by_sub, extra_by_sub, _ = self._protocol_executor.run_protocol(
            self.protocol_info,
            id_param_names=self.param_id_info["param_names"],
            id_param_vals=param_vals,
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
        for exp_idx in exp_idxs_to_run:
            for this_sub_idx in range(num_sub_per_exp[exp_idx]):
                subexp_count = int(np.sum([num_sub for num_sub in
                                           num_sub_per_exp[:exp_idx]]) + this_sub_idx)

                sub_cost = self.get_cost_from_operands(
                    operands_outputs_list[subexp_count],
                    exp_idx=exp_idx, sub_idx=this_sub_idx,
                )
                cost += sub_cost
                if self._num_weighted_obs_by_exp_sub is not None:
                    weighted_obs_denominator += self._num_weighted_obs_by_exp_sub[exp_idx][this_sub_idx]
                else:
                    wc = self.protocol_info["scaled_weight_const_from_exp_sub"][exp_idx][this_sub_idx]
                    ws = self.protocol_info["scaled_weight_series_from_exp_sub"][exp_idx][this_sub_idx]
                    wa = self.protocol_info["scaled_weight_amp_from_exp_sub"][exp_idx][this_sub_idx]
                    wp = self.protocol_info["scaled_weight_phase_from_exp_sub"][exp_idx][this_sub_idx]
                    wd = self.protocol_info["scaled_weight_prob_dist_from_exp_sub"][exp_idx][this_sub_idx]
                    weighted_obs_denominator += int(
                        np.sum(wc != 0)
                        + np.sum(ws != 0)
                        + np.sum(wa != 0)
                        + np.sum(wp != 0)
                        + np.sum(wd != 0)
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
    
    def get_lnprior_from_params(self, param_vals):
        lnprior = 0
        for idx, param_val in enumerate(param_vals):
            if self.param_id_info["param_prior_types"] is not None:
                prior_dist = self.param_id_info["param_prior_types"][idx]
            else:
                prior_dist = None

            if not prior_dist or prior_dist == 'uniform':
                if param_val < self.param_id_info["param_mins"][idx] or param_val > self.param_id_info["param_maxs"][idx]:
                    return -np.inf
                else:
                    #prior += 0
                    pass
            
            elif prior_dist == 'exponential':
                lamb = 1.0 # TODO make this user modifiable
                if param_val < self.param_id_info["param_mins"][idx] or param_val > self.param_id_info["param_maxs"][idx]:
                    return -np.inf
                else:
                    # the normalisation isnt needed here but might be nice to
                    # make sure prior for each param is between 0 and 1
                    lnprior += -lamb*param_val/self.param_id_info["param_maxs"][idx]

            elif prior_dist == 'normal':
                if param_val < self.param_id_info["param_mins"][idx] or param_val > self.param_id_info["param_maxs"][idx]:
                    return -np.inf
                else:
                    # temporarily make the std 1/6 of the user defined range and the mean the centre of the range
                    std = 1/6*(self.param_id_info["param_maxs"][idx] - self.param_id_info["param_mins"][idx])
                    mean = 0.5*(self.param_id_info["param_maxs"][idx] + self.param_id_info["param_mins"][idx])
                    lnprior += -0.5*((param_val - mean)/std)**2


        return lnprior

    def get_lnlikelihood_lnprior_from_params(self, param_vals, reset=True):
        lnprior = self.get_lnprior_from_params(param_vals)

        if not np.isfinite(lnprior):
            return -np.inf

        lnlikelihood = self.get_lnlikelihood_from_params(param_vals)

        return lnprior + lnlikelihood

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

    def get_cost_from_operands(self, operands_outputs, exp_idx = 0, sub_idx = 0):

        if self.model_type == 'casadi_python':
            is_symbolic = True
        else:
            is_symbolic = False

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

    def cost_calc(self, obs_dict, exp_idx=0, sub_idx=0, is_symbolic=False):


        const = obs_dict['const']
        series = obs_dict['series']
        amp = obs_dict['amp']
        phase = obs_dict['phase']
        val_for_prob_dist = obs_dict['val_for_prob_dist']

        # update cost weights for this experiment and subexperiment
        updated_weight_const_vec = self.protocol_info["scaled_weight_const_from_exp_sub"][exp_idx][sub_idx]
        updated_weight_series_vec = self.protocol_info["scaled_weight_series_from_exp_sub"][exp_idx][sub_idx]
        updated_weight_amp_vec = self.protocol_info["scaled_weight_amp_from_exp_sub"][exp_idx][sub_idx]
        updated_weight_phase_vec = self.protocol_info["scaled_weight_phase_from_exp_sub"][exp_idx][sub_idx]
        updated_weight_prob_dist_vec = self.protocol_info["scaled_weight_prob_dist_from_exp_sub"][exp_idx][sub_idx]
        
        # get number of obs that don't have zero weights (cached in __init__ / refresh on obs/protocol change)
        if self._num_weighted_obs_by_exp_sub is not None:
            num_weighted_obs = self._num_weighted_obs_by_exp_sub[exp_idx][sub_idx]
        else:
            num_weighted_obs = int(
                np.sum(updated_weight_const_vec != 0)
                + np.sum(updated_weight_series_vec != 0)
                + np.sum(updated_weight_amp_vec != 0)
                + np.sum(updated_weight_phase_vec != 0)
                + np.sum(updated_weight_prob_dist_vec != 0)
            )
        
        # this subexperiment doesn't have any weighted observables, so no cost
        if num_weighted_obs == 0.0:
            return 0.0
        
        if len(self.obs_info["ground_truth_phase"]) == 0:
            phase = None
        if self.obs_info["ground_truth_phase"].all() == None:
            phase = None

        # TODO: Fix for amp, phase, and val_for_prob_dist
        if is_symbolic:
            _require_casadi()
            cost = ca.SX(0)
            if const is not None:
                for const_idx in range(const.size1()):
                    obs_idx = self.obs_info['const_idx_to_obs_idx'][const_idx]
                    if updated_weight_const_vec[const_idx] != 0:
                        cost += self.cost_funcs_dict[self.cost_type[obs_idx]](const[const_idx], self.obs_info["ground_truth_const"][const_idx],
                                                        self.obs_info["std_const_vec"][const_idx], updated_weight_const_vec[const_idx])

            if series is not None:
                for series_idx in range(len(series)):
                    obs_idx = self.obs_info['series_idx_to_obs_idx'][series_idx]
                    weight_entry = updated_weight_series_vec[series_idx]
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

                    cost += self.cost_funcs_dict[self.cost_type[obs_idx]](
                        series_entry, obs_entry, std_entry, weight_entry)

            # Silently returning a zero cost for observables we can't differentiate would look
            # like a perfectly converged fit, so fail loudly instead.
            if amp is not None or phase is not None or val_for_prob_dist is not None:
                raise NotImplementedError(
                    'automatic differentiation of frequency (amp/phase) and prob_dist '
                    'observables is not implemented. Use constant or series data items, or '
                    'turn off do_ad.')

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
                if updated_weight_const_vec[const_idx] != 0:
                    cost += self.cost_funcs_dict[self.cost_type[obs_idx]](const[const_idx], self.obs_info["ground_truth_const"][const_idx],
                                                    self.obs_info["std_const_vec"][const_idx], updated_weight_const_vec[const_idx])
        
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
                # interpolates the simulated series onto the observation times when
                # dt != obs_dt; shared with the symbolic cost so both agree exactly
                series_entry, obs_entry, std_entry = self._align_series_to_ground_truth(
                    np.asarray(series[series_idx], dtype=float).flatten(), series_idx)

                weight_entry = updated_weight_series_vec[series_idx]

                obs_idx = self.obs_info['series_idx_to_obs_idx'][series_idx]
                if weight_entry != 0:
                    series_cost += self.cost_funcs_dict[self.cost_type[obs_idx]](series_entry, obs_entry, std_entry, weight_entry)


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
                weight_entry = updated_weight_amp_vec[amp_idx]
                std_entry = self.obs_info["std_amp_vec"][amp_idx]
                if hasattr(weight_entry, '__len__'):
                    if not all(val==0 for val in weight_entry):
                        amp_cost += self.cost_funcs_dict[self.cost_type[obs_idx]](amp_entry, obs_entry, std_entry, weight_entry)
                else:
                    if weight_entry != 0:
                        amp_cost += self.cost_funcs_dict[self.cost_type[obs_idx]](amp_entry, obs_entry, std_entry, weight_entry)

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
                weight_entry = updated_weight_phase_vec[phase_idx]
                if hasattr(weight_entry, '__len__'):
                    if not all(val==0 for val in weight_entry):
                        phase_cost += self.cost_funcs_dict[self.cost_type[obs_idx]](phase_entry, obs_entry, std_entry, weight_entry)
                else:
                    if weight_entry != 0:
                        phase_cost += self.cost_funcs_dict[self.cost_type[obs_idx]](phase_entry, obs_entry, std_entry, weight_entry)

        prob_dist_cost = 0
        if val_for_prob_dist is not None:
            for prob_dist_idx in range(len(val_for_prob_dist)):
                obs_idx = self.obs_info['prob_dist_idx_to_obs_idx'][prob_dist_idx]
                if updated_weight_prob_dist_vec[prob_dist_idx] != 0:
                    prob_dist_cost += self.cost_funcs_dict[self.cost_type[obs_idx]](val_for_prob_dist[prob_dist_idx], 
                                                                    self.obs_info["ground_truth_prob_dist_params"][prob_dist_idx],
                                                                    updated_weight_prob_dist_vec[prob_dist_idx])
            

        return cost + series_cost + amp_cost + phase_cost + prob_dist_cost

    def get_obs_output_dict(self, operands_outputs, get_all_series=False, is_symbolic=False):
        #need to added an array to save tmp data, each calibration need to updated/re-initial
        self.temp_results = {}
        
        if operands_outputs == None:
            if get_all_series:
                return None, None
            else:
                return None

        if is_symbolic:
            _require_casadi()
            # TODO: Test series, amp, phase and prob_dist_vec
            obs_const_vec = ca.SX.zeros(len(self.obs_info["ground_truth_const"]), 1)
            obs_series_list_of_arrays = [None]*len(self.obs_info["ground_truth_series"])
            obs_amp_list_of_arrays = [None]*len(self.obs_info["ground_truth_amp"])
            obs_phase_list_of_arrays = [None]*len(self.obs_info["ground_truth_phase"])
            obs_val_for_prob_dist_vec = ca.SX.zeros(len(self.obs_info["ground_truth_prob_dist_params"]), 1)
        else:     
            obs_const_vec = np.zeros((len(self.obs_info["ground_truth_const"]), ))
            obs_series_list_of_arrays = [None]*len(self.obs_info["ground_truth_series"])
            obs_amp_list_of_arrays = [None]*len(self.obs_info["ground_truth_amp"])
            obs_phase_list_of_arrays = [None]*len(self.obs_info["ground_truth_phase"])
            obs_val_for_prob_dist_vec = np.zeros((len(self.obs_info["ground_truth_prob_dist_params"]), ))

        if get_all_series:
            obs_series_array_all = [None]*len(operands_outputs)
        

        const_count = 0
        series_count = 0
        freq_count = 0
        prob_dist_count = 0
        for JJ in range(len(operands_outputs)):
            if self.obs_info["data_types"][JJ] == 'frequency':
                pass
            elif get_all_series:
                if self.obs_info["operations"][JJ] is None:
                    obs_series_array_all[JJ] = operands_outputs[JJ][0]
                elif hasattr(self.operation_funcs_dict[self.obs_info["operations"][JJ]], 'series_to_constant'):
                    raw_kwargs = self.obs_info["operation_kwargs"][JJ]
                    kwargs = raw_kwargs.copy() if isinstance(raw_kwargs, dict) else {}

                    for k, v in list(kwargs.items()):
                        if isinstance(v, str) and v in self.temp_results:
                            #kwargs[k] = self.temp_results[v]
                            if v in self.temp_results:
                                kwargs[k] = self.temp_results[v]
                            else:
                                raise KeyError(f"[ERROR] '{v}' not found in temp_results for key '{k}'")
                    obs_series_array_all[JJ] = self.operation_funcs_dict[self.obs_info["operations"][JJ]](*operands_outputs[JJ],series_output=True,**kwargs)
                else:
                    val_or_array = self.operation_funcs_dict[
                            self.obs_info["operations"][JJ]](*operands_outputs[JJ], **self.obs_info["operation_kwargs"][JJ])
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
            if self.obs_info["operations"][JJ] == None:
                obs = operands_outputs[JJ][0]
            else:
                if self.obs_info["data_types"][JJ] != 'frequency':
                    key_idxt = self.obs_info["names_for_plotting"][JJ]
                    raw_kwargs = self.obs_info["operation_kwargs"][JJ]
                    #every time check it and update to {} when not exist
                    if isinstance(raw_kwargs, dict):
                        kwargs = raw_kwargs.copy()
                    else:
                        kwargs = {}
                    #if exist, extract value, convey it to participate in new cost_function
                    for k, v in list(kwargs.items()):
                        if isinstance(v, str) and v in self.temp_results:
                            if v in self.temp_results:
                                kwargs[k] = self.temp_results[v]
                            else:
                                raise KeyError(f"[ERROR] '{v}' not found in temp_results for key '{k}'")
                    #need to replace below sentence, otherwise will be print error
                    obs = self.operation_funcs_dict[self.obs_info["operations"][JJ]](*operands_outputs[JJ], **kwargs)
                    #each predict result saved into tmp array
                    self.temp_results[key_idxt] = obs
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
                    complex_num = self.operation_funcs_dict[self.obs_info["operations"][JJ]](*complex_operands, **self.obs_info["operation_kwargs"][JJ]) 
                    # TODO check this works for all cases
                    # I am checking the sign of the mean operated on time domain signal to ensure 
                    # the first amplitude is negative if it is a negative signal
                    # sign_signal = np.sign(self.operation_funcs_dict[self.obs_info["operations"][JJ]](* \
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
            elif self.obs_info["data_types"][JJ] == 'prob_dist':
                obs_val_for_prob_dist_vec[prob_dist_count] = obs
                prob_dist_count += 1

        if const_count == 0:
            obs_const_vec = None
        if series_count == 0:
            obs_series_list_of_arrays = None
        if freq_count == 0:
            obs_amp_list_of_arrays = None
            obs_phase_list_of_arrays = None
        if prob_dist_count == 0:
            obs_val_for_prob_dist_vec = None
        obs_dict = {'const': obs_const_vec, 'series': obs_series_list_of_arrays,
                    'amp': obs_amp_list_of_arrays, 'phase': obs_phase_list_of_arrays,
                    'val_for_prob_dist': obs_val_for_prob_dist_vec}

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
                    wd = self.protocol_info["scaled_weight_prob_dist_from_exp_sub"][exp_idx][sub_idx]
                    D += int(np.sum(wc != 0) + np.sum(ws != 0) + np.sum(wa != 0)
                             + np.sum(wp != 0) + np.sum(wd != 0))
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
        """Compute cost J(p), dispatching to CasADi or AADC or numpy."""
        if self.model_type == 'casadi_python':
            return float(self.get_cost_ca(param_vals))
        if self.model_type == 'aadc_python' and self.do_ad:
            # When the gradient comes off the tape the cost has to as well, or the optimiser
            # descends a different function than it evaluates. See get_cost_aadc.
            return float(self.get_cost_aadc(param_vals))
        return float(self.get_cost_from_params(param_vals))

    def get_gradient(self, param_vals):
        """Compute gradient ∇J(p), dispatching to CasADi, AADC, or Myokit FSA."""
        if self.model_type == 'casadi_python':
            return self.get_jac_cost_ca(param_vals)
        elif self.model_type == 'aadc_python':
            return self.get_jac_cost_aadc(param_vals)
        elif self.fsa_gradient_available():
            return self.get_jac_cost_fsa(param_vals)
        else:
            raise ValueError(f"Gradient not available for model_type={self.model_type}")

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

class OpencorMCMC(OpencorParamID): 
    """
    Class for doing mcmc on opencor models
    
    # TODO check the parallelisation for this mcmc
    """

    def __init__(self, model_path,
                 obs_info, param_id_info, protocol_info, prediction_info, solver_info,
                 dt=0.01, mcmc_options=None, DEBUG=False, model_type=None):
        super().__init__(model_path, "MCMC",
                obs_info, param_id_info, protocol_info, prediction_info, solver_info,
                dt=dt, DEBUG=DEBUG, model_type=model_type)

        # mcmc init stuff
        self.sampler = None
        if mcmc_options is not None:
            self.mcmc_options = mcmc_options
            if 'num_steps' not in self.mcmc_options.keys(): 
                self.mcmc_options['num_steps'] = 5000
                print('number of mcmc steps is not set, choosing default of 5000')
            if 'num_walkers' not in self.mcmc_options.keys():
                self.mcmc_options['num_walkers'] = 2*self.num_params
                print('number of mcmc walkers is not set, ',
                    'choosing default of 2*num_params')
        else:
            self.mcmc_options = {}
            self.mcmc_options['num_steps'] = 5000
            self.mcmc_options['num_walkers'] = 2*self.num_params
            print('number of mcmc steps and walkers is not set, ',
                  'choosing defaults of 5000 and 2*num_params')

        self.DEBUG = DEBUG
        assert_mle_cost_for_bayesian(
            self.cost_type, self.cost_funcs_dict, "MCMC (log-likelihood uses -cost)"
        )

    def run(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        num_procs = comm.Get_size()
        if rank == 0:
            print('Running mcmc')


        if num_procs > 1:
            # from pathos import multiprocessing
            # from pathos.multiprocessing import ProcessPool
            from schwimmbad import MPIPool

            if rank == 0:
                if self.best_param_vals is not None:
                    best_param_vals_norm = self.param_norm_obj.normalise(self.best_param_vals)
                    # create initial params in gaussian ball around best_param_vals estimate
                    init_param_vals_norm = (np.ones((self.mcmc_options['num_walkers'], self.num_params))*best_param_vals_norm).T + \
                                       0.1*np.random.randn(self.num_params, self.mcmc_options['num_walkers'])
                    init_param_vals_norm = np.clip(init_param_vals_norm, 0.001, 0.999)
                    init_param_vals = self.param_norm_obj.unnormalise(init_param_vals_norm)
                else:
                    init_param_vals_norm = np.random.rand(self.num_params, self.mcmc_options['num_walkers'])
                    init_param_vals = self.param_norm_obj.unnormalise(init_param_vals_norm)

            try:
                pool = MPIPool() # workers dont get past this line in this try, they wait for work to do
            except:
                return

            if not pool.is_master():
                pool.wait()
                return

            if mcmc_lib == 'emcee':
                self.sampler = emcee.EnsembleSampler(self.mcmc_options['num_walkers'], self.num_params, calculate_lnlikelihood,
                                            pool=pool)
            elif mcmc_lib == 'zeus':
                self.sampler = zeus.EnsembleSampler(self.mcmc_options['num_walkers'], self.num_params, calculate_lnlikelihood,
                                                        pool=pool)

            start_time = time.time()
            self.sampler.run_mcmc(init_param_vals.T, self.mcmc_options['num_steps'], progress=True, tune=True)
            print(f'mcmc time = {time.time() - start_time}')
            pool.close()

        else:
            if self.best_param_vals is not None:
                best_param_vals_norm = self.param_norm_obj.normalise(self.best_param_vals)
                init_param_vals_norm = (np.ones((self.mcmc_options['num_walkers'], self.num_params))*best_param_vals_norm).T + \
                                   0.01*np.random.randn(self.num_params, self.mcmc_options['num_walkers'])
                init_param_vals_norm = np.clip(init_param_vals_norm, 0.001, 0.999)
                init_param_vals = self.param_norm_obj.unnormalise(init_param_vals_norm)
            else:
                init_param_vals_norm = np.random.rand(self.num_params, self.mcmc_options['num_walkers'])
                init_param_vals = self.param_norm_obj.unnormalise(init_param_vals_norm)

            if mcmc_lib == 'emcee':
                self.sampler = emcee.EnsembleSampler(self.mcmc_options['num_walkers'], self.num_params, calculate_lnlikelihood)
            elif mcmc_lib == 'zeus':
                self.sampler = zeus.EnsembleSampler(self.mcmc_options['num_walkers'], self.num_params, calculate_lnlikelihood)

            start_time = time.time()
            self.sampler.run_mcmc(init_param_vals.T, self.mcmc_options['num_steps']) # , progress=True)
            print(f'mcmc time = {time.time()-start_time}')

        if rank == 0:
            # TODO save chains
            if mcmc_lib == 'emcee':
                print(f'acceptance fraction was {self.sampler.acceptance_fraction}')
            samples = self.sampler.get_chain()
            mcmc_chain_path = os.path.join(self.output_dir, 'mcmc_chain.npy')
            np.save(mcmc_chain_path, samples)
            print('mcmc complete')
            print(f'mcmc chain saved in {mcmc_chain_path}')

            # save best param vals and best cost from mcmc mean
            samples = samples[samples.shape[0]//2:, :, :]
            # thin = 10
            # samples = samples[::thin, :, :]
            flat_samples = samples.reshape(-1, self.num_params)
            means = np.zeros((self.num_params))
            medians = np.zeros((self.num_params))
            for param_idx in range(self.num_params):
                means[param_idx] = np.mean(flat_samples[:, param_idx])
                medians[param_idx] = np.median(flat_samples[:, param_idx])

            # rerun with original and mcmc optimal param vals
            mcmc_best_param_vals = medians  # means
            # TODO change the below to get_cost_from_params when inheriting
            mcmc_best_cost, _ = self.get_cost_and_obs_from_params(mcmc_best_param_vals, reset=True)
            if self.best_param_vals is None:
                self.best_param_vals = mcmc_best_param_vals
                self.best_cost = mcmc_best_cost
                print('cost from mcmc median param vals is {}'.format(self.best_cost))
                print('saving best_param_vals and best_cost from mcmc medians')

                np.save(os.path.join(self.output_dir, 'best_cost'), self.best_cost)
                np.save(os.path.join(self.output_dir, 'best_param_vals'), self.best_param_vals)
            else:
                original_best_cost, _ = self.get_cost_and_obs_from_params(self.best_param_vals, reset=True)
                if mcmc_best_cost < original_best_cost:
                    self.best_param_vals = mcmc_best_param_vals
                    self.best_cost = mcmc_best_cost
                    print('cost from mcmc median param vals is {}'.format(self.best_cost))
                    print('resaving best_param_vals and best_cost from mcmc medians')

                    np.save(os.path.join(self.output_dir, 'best_cost'), self.best_cost)
                    np.save(os.path.join(self.output_dir, 'best_param_vals'), self.best_param_vals)
                else:
                    self.best_cost = original_best_cost
                    # leave the original best fit param val as the best fit value, mcmc just gives distributions
                    print('cost from mcmc median param vals is {}'.format(mcmc_best_cost))
                    print('Keeping the genetic algorithm best fit as it is lower, ({})'.format(self.best_cost))

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

class MCMC_plotter:
    """
    This class contains plotting wrapper for mcmc
    """

    def __init__(self, model_path, model_type, param_id_method, file_name_prefix,
                 params_for_id_path=None, num_calls_to_function=1000,
                 param_id_obs_path=None, sim_time=2.0, pre_time=20.0, 
                 solver_info=None, 
                 dt=0.01, mcmc_options=None, 
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
                self.param_id_output_dir = os.path.join(os.path.dirname(__file__), '../../param_id_output')
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
            self.resources_dir = os.path.join(os.path.dirname(__file__), '../../resources')
        else:
            self.resources_dir = resources_dir


        self.best_param_vals = None
        self.best_param_names = None

        self.mcmc_options = mcmc_options

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
                                    solver_info=self.solver_info, mcmc_options=self.mcmc_options,
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

