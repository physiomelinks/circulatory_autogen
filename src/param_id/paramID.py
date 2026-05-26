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
import emcee
import re
from numpy import genfromtxt
from importlib import import_module
# import tqdm # TODO this needs to be installed for corner plot but doesnt need an import here
try:
    import corner
except ImportError:
    corner = None
import csv
from datetime import date
# from skopt import gp_minimize, Optimizer
from parsers.PrimitiveParsers import CSVFileParser, ObsAndParamDataParser
from param_id.optimisers import GeneticAlgorithmOptimiser, BayesianOptimiser, CMAESOptimiser, SciPyMinimizeOptimiser
from param_id.differentiable import (
    assert_casadi_differentiable,
    assert_mle_cost_for_bayesian,
    is_circulatory_differentiable,
)
from param_id.plot_outputs import ParamIDPlotOutputs
import pandas as pd
try:
    import casadi as ca
except ImportError:
    ca = None
import json
import math
import seaborn as sns
import arviz as az
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

import pytensor.tensor as pt
from pytensor.compile.ops import as_op
@as_op(itypes=[pt.dvector], otypes=[pt.dscalar])
def logp_op(theta):
        # 1. Get the original log-likelihood/prior value
        logp_val = mcmc_object.get_lnlikelihood_lnprior_from_params(theta)
                
        logp_val = np.asarray(logp_val)

        if logp_val.shape != ():
            logp_val = np.sum(logp_val)

        return np.array(float(logp_val))

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
        option_dicts.append(inp_data_dict.get("debug_UQ_options") or {})
    option_dicts.append(inp_data_dict.get("optimiser_options") or {})
    option_dicts.append(inp_data_dict.get("UQ_options") or {})
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


def _require_casadi():
    if ca is None:
        raise ImportError(
            "CasADi is required for symbolic or casadi_python workflows but is not installed. "
            "Install the casadi package (for example: pip install casadi)."
        )


class CVS0DParamID():
    """
    Class for doing parameter identification on a 0D cvs model
    """
    def __init__(self, model_path, model_type, param_id_method, mcmc_instead=False, file_name_prefix='no_name',
                 params_for_id_path=None,
                 param_id_obs_path=None, sim_time=2.0, pre_time=20.0, dt=0.01,
                 solver_info=None, UQ_options=None, optimiser_options=None, 
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

        self.UQ_options = UQ_options
        # Set UQ library from UQ_options
        self.UQ_library = self.UQ_options.get('library', 'emcee') if self.UQ_options else 'emcee'
        
        # Import MCMC libraries based on library
        if self.UQ_library == 'zeus':
            try:
                import zeus
                self.zeus = zeus
            except ImportError:
                self.zeus = None
        elif self.UQ_library == 'pymc':
            try:
                import pymc as pm
                import arviz as az
                import pytensor.tensor as pt
                from pytensor.compile.ops import as_op
                self.pm = pm
                self.az = az
                self.pt = pt
                self.as_op = as_op
                @as_op(itypes=[pt.dvector], otypes=[pt.dscalar])
                def logp_op(theta):
                    # 1. Get the original log-likelihood/prior value
                    logp_val = mcmc_object.get_lnlikelihood_lnprior_from_params(theta)
                    
                    logp_val = np.asarray(logp_val)

                    if logp_val.shape != ():
                        logp_val = np.sum(logp_val)

                    return np.array(float(logp_val))
                self.logp_op = logp_op
            except ImportError as e:
                print(f"Failed to import pymc dependencies: {e}")
                self.pm = None
        else:
            print(f'unknown mcmc lib : {self.UQ_library}')
        
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
                sim_time=sim_time
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
                                           UQ_options=UQ_options,
                                           DEBUG=self.DEBUG, model_type=self.model_type)
            self.n_steps = mcmc_object.n_steps
        else:
            if model_type in ['cellml_only', 'python', 'casadi_python']:
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

    @classmethod
    def init_from_dict(cls, inp_data_dict):
        # Only pass kwargs that exist in inp_data_dict
        arg_options = [
            'model_path', 'model_type', 'param_id_method', 'mcmc_instead',
            'file_name_prefix', 'params_for_id_path', 'param_id_obs_path',
            'sim_time', 'pre_time', 'dt', 'solver_info', 'UQ_options',
            'optimiser_options', 'DEBUG', 'param_id_output_dir', 'resources_dir',
            'one_rank',
        ]
        kwargs = {key: inp_data_dict[key] for key in arg_options if key in inp_data_dict}

        # Support common naming used elsewhere
        if 'file_name_prefix' not in kwargs and 'file_prefix' in inp_data_dict:
            kwargs['file_name_prefix'] = inp_data_dict['file_prefix']

        return cls(**kwargs)

    @classmethod
    def init_from_all_dicts(cls, inp_data_dict, obs_data_dict, params_for_id_dict):
        new_object = cls.init_from_dict(inp_data_dict)
        new_object.set_ground_truth_data(obs_data_dict)
        new_object.set_params_for_id(params_for_id_dict)
        return new_object

    def temp_test(self):
        self.param_id.temp_test()
    def temp_test2(self):
        self.param_id.temp_test2()

    def run(self):
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
        if return_series:
            obs_dicts, obs_arrays = self.param_id.simulate_once(reset=reset, only_one_exp=only_one_exp, return_series=return_series)
            self.best_output_calculated = True
            return obs_dicts, obs_arrays
        else:
            obs_dict, _ = self.param_id.simulate_once(reset=reset, only_one_exp=only_one_exp)
            self.best_output_calculated = True
            return obs_dict

    def update_param_range(self, params_to_update_list_of_lists, mins, maxs):
        for params_to_update_list, min, max in zip(params_to_update_list_of_lists, mins, maxs):
            for JJ, param_name_list in enumerate(self.param_id_info["param_names"]):
                if param_name_list == params_to_update_list:
                    self.param_id_info["param_mins"][JJ] = min
                    self.param_id_info["param_maxs"][JJ] = max

    def set_output_dir(self, path):
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
        self.param_id.add_user_operation_func(func)
    
    def add_user_cost_func(self, func):
        self.param_id.add_user_cost_func(func)
    
    def set_param_names(self, param_names):
        if self.mcmc_instead:
            mcmc_object.set_param_names(param_names)
        else:
            self.param_id.set_param_names(param_names)
    
    def set_optimiser_options(self, optimiser_options):
        self.optimiser_options = optimiser_options
        self.param_id.set_optimiser_options(optimiser_options)

    def set_param_id_method(self, param_id_method):
        self.param_id_method = param_id_method
        self.param_id.set_param_id_method(param_id_method)

    def set_ground_truth_data(self, obs_data_dict):
        if self.rank == 0:
            print(f'Setting ground truth data: {obs_data_dict}')
        parsed_data = self.obs_and_param_parser.parse_obs_data_json(
            obs_data_dict=obs_data_dict,
            pre_time=self.pre_time,
            sim_time=self.sim_time
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
        if self.rank == 0:
            print(f'Setting params for id: {params_for_id_dict}')
        self.param_id_info = self.obs_and_param_parser.get_param_id_info_from_entries(params_for_id_dict)
        self.obs_and_param_parser.save_param_names(self.param_id_info, self.output_dir)
        self.param_id.set_param_id_info(self.param_id_info)
        if self.rank == 0:
            print(f'Params for id set: {self.param_id_info["param_names"]}')

    def set_best_param_vals(self, best_param_vals):
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
        if self.rank == 0:
            param_vals = self._resolve_best_param_vals_for_outputs()
            if param_vals is not None and not self.mcmc_instead:
                self.param_id.save_all_outputs_per_experiment(
                    param_vals, suffix="_plot"
                )
        ParamIDPlotOutputs(self).plot_outputs()

    def get_mcmc_samples(self):
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
        burn_in_idx = int(samples.shape[0] * self.UQ_options['settings']['burn_in'])  
        samples = samples[burn_in_idx:, :, :]
        
        flat_samples = samples.reshape(-1, num_params)

        return flat_samples, samples, num_params

    def plot_mcmc(self):

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
                                fontsize=20, hist_kwargs={"density": True}, show_titles=True)
        else:
            fig = corner.corner(flat_samples[:, overwrite_params_to_plot_idxs], bins=20, hist_bin_factor=2, smooth=0.5, quantiles=(0.05, 0.5, 0.95),
                                labels=[label_list[II] for II in overwrite_params_to_plot_idxs],
                                truths=self.param_id.best_param_vals[overwrite_params_to_plot_idxs],
                                fontsize=20, hist_kwargs={"density": True}, show_titles=True)
        
        fig.text(
            0.95, 0.8, 
            "Titles show: Median\nErrors show: 5% & 95% Quantiles", 
            ha='right', va='top', 
            fontsize=8, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
        )

        # overlay analytical prior PDF
        axes = fig.get_axes()  
        num_params = len(overwrite_params_to_plot_idxs)  

        for i in range(num_params):  
            ax = axes[i * num_params + i]  # Diagonal plot
            
            param_idx = overwrite_params_to_plot_idxs[i]  
            param_min = self.param_id_info["param_mins"][param_idx]  
            param_max = self.param_id_info["param_maxs"][param_idx]  
            
            ax.set_xlim(param_min, param_max)

            x_values = np.linspace(param_min, param_max, 200)  
            pdf_values = self.get_prior_pdf(param_idx, x_values)  
            
            current_label = 'Prior' if i == 0 else None
            ax.fill_between(x_values, 0, pdf_values, alpha=0.3, color='C2', 
                            label=current_label, zorder=0)

            ymin, ymax = ax.get_ylim()
            prior_peak = np.max(pdf_values)
            ax.set_ylim(0, max(float(ymax), float(prior_peak)) * 1.1)
        
        fig.legend(loc='upper right', fontsize='small')

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

        for i in range(num_params):  
            for j in range(num_params):  
                if i != j and i > j:  # Off-diagonal plots
                    
                    ax_idx = i * num_params + j  
                    ax = axes[ax_idx]
                    
                    # Set axis ranges to match parameter bounds  
                    x_param_idx = overwrite_params_to_plot_idxs[j]  
                    y_param_idx = overwrite_params_to_plot_idxs[i] 

                    ax.set_xlim(self.param_id_info["param_mins"][x_param_idx],   
                            self.param_id_info["param_maxs"][x_param_idx])  
                    ax.set_ylim(self.param_id_info["param_mins"][y_param_idx],   
                            self.param_id_info["param_maxs"][y_param_idx])

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
                                fontsize=20, hist_kwargs={"density": True}, show_titles=True)
        else:
            fig = corner.corner(flat_samples[:, overwrite_params_to_plot_idxs], bins=20, hist_bin_factor=2, smooth=0.5, quantiles=(0.05, 0.5, 0.95),
                                labels=[label_list[II] for II in overwrite_params_to_plot_idxs],
                                truths=self.param_id.best_param_vals[overwrite_params_to_plot_idxs],
                                fontsize=20, hist_kwargs={"density": True}, show_titles=True)

        fig.text(
            0.95, 0.8, 
            "Titles show: Median\nErrors show: 5% & 95% Quantiles", 
            ha='right', va='top', 
            fontsize=8, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
        )

        # overlay analytical prior PDF
        axes = fig.get_axes()  
        num_params = len(overwrite_params_to_plot_idxs)  

        for i in range(num_params):  
            ax = axes[i * num_params + i]  # Diagonal plot
            
            param_idx = overwrite_params_to_plot_idxs[i]  
            param_min = self.param_id_info["param_mins"][param_idx]  
            param_max = self.param_id_info["param_maxs"][param_idx]  
            
            ax.set_xlim(param_min, param_max)

            x_values = np.linspace(param_min, param_max, 200)  
            pdf_values = self.get_prior_pdf(param_idx, x_values)  
            
            current_label = 'Prior' if i == 0 else None
            ax.fill_between(x_values, 0, pdf_values, alpha=0.3, color='C2', 
                            label=current_label, zorder=0)

            ymin, ymax = ax.get_ylim()
            prior_peak = np.max(pdf_values)
            ax.set_ylim(0, max(float(ymax), float(prior_peak)) * 1.1)
            
        fig.legend(loc='upper right', fontsize='small')

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

        for i in range(num_params):  
            for j in range(num_params):  
                if i != j and i > j:  # Off-diagonal plots
                    
                    ax_idx = i * num_params + j  
                    ax = axes[ax_idx]  

                    # Set axis ranges to match parameter bounds  
                    x_param_idx = overwrite_params_to_plot_idxs[j]  
                    y_param_idx = overwrite_params_to_plot_idxs[i] 
                    
                    ax.set_xlim(self.param_id_info["param_mins"][x_param_idx],   
                            self.param_id_info["param_maxs"][x_param_idx])  
                    ax.set_ylim(self.param_id_info["param_mins"][y_param_idx],   
                            self.param_id_info["param_maxs"][y_param_idx])

        plt.subplots_adjust(hspace=0.12, wspace=0.1)

        plt.savefig(os.path.join(self.plot_dir, f'mcmc_cornerplot_subset_{self.file_name_prefix}_'
                                                f'{self.param_id_obs_file_prefix}.pdf'))
        plt.close()

        # Also check autocorrelation times for mcmc chain
        if self.UQ_library == 'emcee':
            tau = self.calculate_autocorrelation_time(samples)
            print(f"Auto-correlation time: {tau}")

        swapped_samples = np.swapaxes(samples, 0, 1)
        param_names = self.param_id_info["param_names_for_plotting"]
        dataset = az.convert_to_dataset(
                {"params": swapped_samples},
                coords={"param_dim": param_names},
                dims={"params": ["chain", "draw", "param_dim"]}
        )
        print(az.summary(dataset, round_to=3))

        # Add autocorrelation plots  
        self.plot_autocorrelation(samples, num_params)

        # plot chain averages to check for convergence
        self.plot_chain_avg()

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

    def plot_autocorrelation(self, samples, num_params):  
        """Create autocorrelation plots for each parameter"""  
        fig, axes = plt.subplots(num_params, figsize=(10, 2*num_params), sharex=True)  
        all_bounded = True
        for i in range(num_params):  
            if hasattr(axes, '__len__'):  
                ax = axes[i]  
            else:  
                ax = axes  
                
            # Calculate autocorrelation for each walker  
            for walker in range(samples.shape[1]):  
                autocorr = emcee.autocorr.function_1d(samples[:, walker, i])  
                ax.plot(autocorr, alpha=0.3)  

                # Check if autocorrelation exceeds bounds (skip lag=0 which is always 1.0)  
                window_size = int(0.2 * len(autocorr))  # You can adjust this based on your total step count
                if np.any(np.abs(autocorr[-window_size:]) > 0.1):  
                    all_bounded = False
            
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.7)
            ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
            ax.axhline(y=-0.1, color='r', linestyle='--', alpha=0.7, linewidth=1.5)

            ax.set_ylabel(f'${self.param_id_info["param_names_for_plotting"][i]}$')  
            ax.set_xlim(0, len(autocorr))  

        
        ax.set_xlabel("Lag")  
        plt.tight_layout()  
        plt.savefig(os.path.join(self.plot_dir, f'autocorrelation_plot_{self.file_name_prefix}_'  
                                f'{self.param_id_obs_file_prefix}.pdf'))  
        plt.close()

        return all_bounded

    def plot_chain_avg(self):

        """  
        Plot the average value across all MCMC chains for each parameter.  
        This helps visualize the convergence and overall trend of the sampling.  
        """

        if self.rank != 0:  
            return  
        
        flat_samples, samples, num_params = self.get_mcmc_samples()  

        fig, axes = plt.subplots(num_params, figsize=(10, 2*num_params), sharex=True)  

        num_steps, num_chains, num_params = samples.shape
        window_size = 10  # Adjust based on your total step count

        if window_size >= num_steps:
            print(f"Warning: window_size {window_size} is greater than or equal to num_steps {num_steps}. Skipping chain average plot.")
            return

        for i in range(num_params):
            ax = axes[i] if num_params > 1 else axes
            
            for j in range(num_chains):
                chain_data = samples[:, j, i]
                
                # Moving average calculation using convolution
                window = np.ones(window_size) / window_size
                moving_avg = np.convolve(chain_data, window, mode='valid')
                
                # Note: convolve 'valid' mode shrinks the array by window_size - 1
                # We adjust the x-axis to match the end of the window
                x_axis = np.arange(window_size - 1, num_steps)
                ax.plot(x_axis, moving_avg, alpha=0.6)

            overall_mean = np.mean(samples[:, :, i])
            ax.axhline(y=overall_mean, color='r', linestyle='--', label=f'Overall Mean: {overall_mean:.3g}')
            
            param_name = self.param_id_info["param_names_for_plotting"][i]
            ax.set_ylabel(f'${param_name}$')
            ax.legend(loc='upper right', fontsize='small')

        ax.set_xlabel("Step Number")  
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f'mcmc_chain_averages_{self.file_name_prefix}_'  
                            f'{self.param_id_obs_file_prefix}.pdf'))  
        plt.close()  
        
        print(f"Chain averages plot saved to {self.plot_dir}")

    def get_posterior_stats(self, samples):

        swapped_samples = np.swapaxes(samples, 0, 1)
        param_names = self.param_id_info["param_names_for_plotting"]
        
        dataset = az.convert_to_dataset(
                {"params": swapped_samples},
                coords={"param_dim": param_names},
                dims={"params": ["chain", "draw", "param_dim"]}
        )

        stats = az.summary(dataset)

        return stats

    def calc_effective_sample_size(self, samples):

        swapped_samples = np.swapaxes(samples, 0, 1)
        param_names = self.param_id_info["param_names_for_plotting"]
        
        dataset = az.convert_to_dataset(
                {"params": swapped_samples},
                coords={"param_dim": param_names},
                dims={"params": ["chain", "draw", "param_dim"]}
        )

        summary = az.summary(dataset)

        # Extract ess_bulk and ess_tail for each parameter  
        ess_dict = {  
            'ess_bulk': summary['ess_bulk'].to_dict(),  
            'ess_tail': summary['ess_tail'].to_dict()  
        }

        return ess_dict
    
    def calc_rhat(self, samples):

        swapped_samples = np.swapaxes(samples, 0, 1)
        param_names = self.param_id_info["param_names_for_plotting"]
        dataset = az.convert_to_dataset(
                {"params": swapped_samples},
                coords={"param_dim": param_names},
                dims={"params": ["chain", "draw", "param_dim"]}
        )

        summary = az.summary(dataset)

        rhat_dict = summary['r_hat'].to_dict()

        return rhat_dict

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
        self.param_id.set_bayesian_parameters(n_calls, n_initial_points, acq_func, random_state,
                                              acq_func_kwargs=acq_func_kwargs)

    def close_simulation(self):
        if self.mcmc_instead:
            mcmc_object.close_simulation()
        else:
            self.param_id.close_simulation()
    
    def get_best_param_vals(self):
        if self.mcmc_instead:
            return mcmc_object.best_param_vals
        else:
            return self.param_id.best_param_vals

    def get_param_names(self):
        if self.mcmc_instead:
            return mcmc_object.param_id_info["param_names"]
        else:
            return self.param_id.param_id_info["param_names"]

    def get_param_importance(self):
        return self.param_id.param_importance

    def get_collinearity_idx(self):
        return self.param_id.collinearity_idx

    def get_collinearity_idx_pairs(self):
        return self.param_id.collinearity_idx_pairs

    def get_pred_param_importance(self):
        return self.param_id.pred_param_importance

    def get_pred_collinearity_idx_pairs(self):
        return self.param_id.pred_collinearity_idx_pairs

    def remove_params_by_idx(self, param_idxs_to_remove):
        self.__set_and_save_param_names(idxs_to_ignore=param_idxs_to_remove)
        if self.mcmc_instead:
            mcmc_object.remove_params_by_idx(param_idxs_to_remove)
        else:
            self.param_id.remove_params_by_idx(param_idxs_to_remove)

    def remove_params_by_name(self, param_names_to_remove):
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

        print('Creating boxplots for MCMC samples')  
        self.plot_boxplots_for_predictions(flat_samples, n_sims=n_sims)

    def plot_boxplots_for_predictions(self, flat_samples, n_sims=50, show_points=True):
        """
        Consolidated visualization: Violin + Boxplot + Jittered points.
        Fixed for Seaborn 0-based indexing and cleaner data handling.
        """
        if self.rank != 0:
            return

        # 1. Initialize storage using names_for_plotting
        data_item_exp_values = {name: {} for name in self.obs_info["names_for_plotting"]}
        sim_obj = mcmc_object if self.mcmc_instead else self.param_id

        # 2. Run Simulations
        n_actual = min(n_sims, len(flat_samples))
        sample_indices = np.random.choice(len(flat_samples), n_actual, replace=False)

        i=0
        for idx in sample_indices:
            sample_params = flat_samples[idx, :]
            _, obs_list = sim_obj.get_cost_and_obs_from_params(sample_params, reset=True)

            subexp_count = 0
            for exp_idx in range(self.protocol_info['num_experiments']):
                for sub_in_exp_idx in range(self.protocol_info['num_sub_per_exp'][exp_idx]):
                    
                    if subexp_count >= len(obs_list) or obs_list[subexp_count] is None:
                        subexp_count += 1
                        continue

                    obs_proc = sim_obj.get_obs_output_dict(obs_list[subexp_count])
                    subexp_count += 1

                    # Map to relevant features
                    for obs_idx, name in enumerate(self.obs_info["names_for_plotting"]):
                        if (self.obs_info["experiment_idxs"][obs_idx] == exp_idx and 
                            self.obs_info["subexperiment_idxs"][obs_idx] == sub_in_exp_idx):
                            
                            d_type = self.obs_info["data_types"][obs_idx]
                            val = None
                            try:
                                if d_type == "constant":  val = obs_proc['const'][0]
                                elif d_type == "series":  val = np.max(obs_proc['series'][obs_idx])
                                elif d_type == "frequency": val = obs_proc['amp'][obs_idx]
                                elif d_type == "prob_dist": val = obs_proc['val_for_prob_dist'][obs_idx]
                            except (IndexError, KeyError): continue

                            if val is not None:
                                data_item_exp_values[name].setdefault(exp_idx, []).append(val)

            sim_obj.sim_helper.reset_and_clear()
            i += 1
            print(f"Processed {i}/{n_actual} samples for boxplot data.")

        # 3. Add Experimental Ground Truth
        for i, name in enumerate(self.obs_info["names_for_plotting"]):
            d_type = self.obs_info["data_types"][i]
            exp_list = data_item_exp_values[name].setdefault("exp_data", [])

            if d_type == "constant":
                mean, std = self.obs_info["ground_truth_const"][i], self.obs_info["std_const_vec"][i]
                exp_list.extend(np.random.normal(mean, std, 20))
            elif d_type == "prob_dist":
                exp_list.extend(self.obs_info["ground_truth_prob_dist_params"][i]["data_points"])

                # 3.5 Save all outputs to CSV for testing  
        csv_rows = []  
        for feature, exp_dict in data_item_exp_values.items():  
            for key, values in exp_dict.items():  
                if key == "exp_data":  
                    data_type = "experimental"  
                else:  
                    data_type = "simulated"  
                for val in values:  
                    csv_rows.append({  
                        "feature": feature,  
                        "experiment_idx": key,  
                        "value": val,  
                        "data_type": data_type  
                    })  
          
        csv_df = pd.DataFrame(csv_rows)  
        csv_path = os.path.join(self.output_dir, "posterior_predictions.csv")  
        csv_df.to_csv(csv_path, index=False)  
        print(f"Saved posterior predictions to {csv_path}")

        # 4. Plotting Loop
        for feature, exp_dict in data_item_exp_values.items():
            # Prepare lists for plotting, keeping 'exp_data' at the end for consistency
            sorted_keys = sorted(exp_dict.keys(), key=lambda x: str(x))
            values, labels, colors = [], [], []

            for key in sorted_keys:
                if not exp_dict[key]: continue
                values.append(exp_dict[key])
                if key == "exp_data":
                    labels.append("Experimental")
                    colors.append("red")
                else:
                    labels.append(self.protocol_info["experiment_labels"][key] if key < len(self.protocol_info["experiment_labels"]) else f"Exp {key}")
                    colors.append(self.protocol_info["experiment_colors"][key] if key < len(self.protocol_info["experiment_colors"]) else f"C{key}")

            if not values: continue

            fig, ax = plt.subplots(figsize=(6.5, 4.5))

            # Main Violin + Boxplot (sns handles box internally)
            sns.violinplot(data=values, ax=ax, palette=colors, cut=3, inner="box", saturation=0.8, bw_method='scott')

            # Style bodies
            for i, collection in enumerate(ax.collections):
                if i < len(values):
                    collection.set_alpha(0.35)
                    collection.set_edgecolor("none")

            # Stats and Custom markers
            for i, vals in enumerate(values):
                mean_v, std_v = np.mean(vals), np.std(vals)
                
                # Mean Diamond
                ax.scatter(i, mean_v, marker="D", color="white", edgecolor="black", s=30, zorder=4)

                # Stats annotation (using relative offset for y_pos)
                y_range = np.max(vals) - np.min(vals)
                ax.text(i, np.max(vals) + (0.05 * y_range), fr"${mean_v:.2g} \pm {std_v:.2g}$", 
                        ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.5, ec="none"))

                if show_points:
                    x_jitter = np.random.normal(i, 0.04, size=len(vals))
                    ax.scatter(x_jitter, vals, color="black", s=5, alpha=0.2, zorder=2)

            # Labels and Spines
            obs_idx = self.obs_info["names_for_plotting"].index(feature)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=15)
            ax.set_ylabel(f"{feature} ({self.obs_info['units'][obs_idx]})")
            ax.set_title(feature)
            sns.despine()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, f"posterior_{feature.replace(' ', '_')}.png"), dpi=300)
            plt.close()

            self.plot_distribution_grid(data_item_exp_values)

    def plot_distribution_grid(self, data_item_exp_values):
        """
        Creates a summary figure with subplots for every feature.
        Plots Histogram + KDE for combined Model results vs Experimental data.
        """
        features = self.obs_info["names_for_plotting"]
        num_features = len(features)
        
        # Calculate layout: 3 columns, dynamic rows
        cols = 3
        rows = (num_features + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4))
        axes = axes.flatten()

        for i, feature in enumerate(features):
            ax = axes[i]
            exp_dict = data_item_exp_values.get(feature, {})
            
            # 1. Aggregate all model predictions from various experiments
            model_vals = []
            for key, vals in exp_dict.items():
                if key != "exp_data":
                    model_vals.extend(vals)
            
            # 2. Extract experimental ground truth
            exp_vals = exp_dict.get("exp_data", [])

            # Plotting Helper
            def draw_dist(data, label, color):
                if len(data) < 2: return
                # Histogram (density=True is critical for KDE alignment)
                # ax.hist(data, bins=50, density=True, alpha=0.2, color=color)
                
                # KDE calculation
                try:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(data, bw_method=0.1)
                    x_grid = np.linspace(min(data)-0.5*np.std(data), max(data)+0.5*np.std(data), 100)
                    ax.plot(x_grid, kde(x_grid), color=color, lw=2, label=label)
                except np.linalg.LinAlgError:
                    # KDE can fail if data has zero variance; fallback to histogram
                    ax.hist(data, bins=50, density=True, alpha=0.2, color=color, label=label)

            # Plot both sets
            draw_dist(model_vals, "Model Posterior", "#1f77b4") # Muted Blue
            draw_dist(exp_vals, "Experimental", "#d62728")     # Muted Red

            # Formatting
            obs_idx = self.obs_info["names_for_plotting"].index(feature)
            ax.set_title(f"{feature}", fontweight='bold')
            ax.set_xlabel(f"Value ({self.obs_info['units'][obs_idx]})")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8, frameon=False)
            sns.despine(ax=ax)

        # Clean up empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        grid_path = os.path.join(self.plot_dir, "all_features_kde_grid.png")
        plt.savefig(grid_path, dpi=300)
        plt.close()

    def get_prior_pdf(self, param_idx, x_values):  
        """  
        Get the analytical probability density function for a parameter's prior.  
        
        Args:  
            param_idx: Index of the parameter  
            x_values: Array of x values to evaluate PDF at  
            
        Returns:  
            Array of PDF values  
        """  
        prior_dist = self.param_id_info["param_prior_types"][param_idx]  
        param_min = self.param_id_info["param_mins"][param_idx]  
        param_max = self.param_id_info["param_maxs"][param_idx]  
        
        if prior_dist == 'uniform' or prior_dist is None:  
            # Uniform distribution: PDF = 1/(b-a) within bounds  
            pdf = np.zeros_like(x_values)  
            mask = (x_values >= param_min) & (x_values <= param_max)  
            pdf[mask] = 1.0 / (param_max - param_min)  
            return pdf  
            
        elif prior_dist == 'exponential':  
            # Exponential with lambda=1.0, truncated to bounds  
            lamb = 1.0  
            pdf = np.zeros_like(x_values)  
            mask = (x_values >= param_min) & (x_values <= param_max)  
            # PDF = lambda * exp(-lambda * x) / normalization  
            pdf[mask] = lamb * np.exp(-lamb * x_values[mask])  
            # Normalize for truncation  
            norm = np.exp(-lamb * param_min) - np.exp(-lamb * param_max)  
            pdf[mask] /= norm  
            return pdf  
            
        elif prior_dist == 'normal':  
            # Normal with mean=center, std=range/6, truncated to bounds  
            std = 1/6 * (param_max - param_min)  
            mean = 0.5 * (param_max + param_min)  
            pdf = np.zeros_like(x_values)  
            mask = (x_values >= param_min) & (x_values <= param_max)  
            # Normal PDF: (1/(sqrt(2*pi)*sigma)) * exp(-0.5*((x-mu)/sigma)^2)  
            pdf[mask] = (1.0 / (np.sqrt(2 * np.pi) * std)) * \
                        np.exp(-0.5 * ((x_values[mask] - mean) / std) ** 2)  
            return pdf  
        
        return np.zeros_like(x_values)

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

        mode = "casadi" if self.model_type == "casadi_python" else "numpy"
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

        else:
            print(f'param_id_method {self.param_id_method} hasn\'t been implemented')
            exit()

        if rank == 0:
            print('')
            print(f'{self.param_id_method} is complete')
            # print init params and final params
            print('init params     : {}'.format(self.param_init))
            print('best fit params : {}'.format(self.best_param_vals))
            print('best cost       : {}'.format(self.best_cost))

            self.save_all_outputs_per_experiment(self.best_param_vals, suffix="")

            if self.param_id_method == 'sp_minimize':
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

        # TODO: Fix for series, amp, phase, and val_for_prob_dist
        if is_symbolic:
            _require_casadi()
            cost = ca.SX(0)
            if const is not None:
                for const_idx in range(const.size1()):
                    obs_idx = self.obs_info['const_idx_to_obs_idx'][const_idx]
                    if updated_weight_const_vec[const_idx] != 0:
                        cost += self.cost_funcs_dict[self.cost_type[obs_idx]](const[const_idx], self.obs_info["ground_truth_const"][const_idx],
                                                        self.obs_info["std_const_vec"][const_idx], updated_weight_const_vec[const_idx])
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
                if self.obs_info["obs_dt"][series_idx] != self.dt:
                    # interpolate the series to the dt of the ground truth series
                    time_series = np.linspace(0, series[series_idx].shape[0]*self.dt, series[series_idx].shape[0])
                    obs_time_series = np.linspace(0, self.obs_info["ground_truth_series"][series_idx].shape[0]*self.obs_info["obs_dt"][series_idx],
                                                    self.obs_info["ground_truth_series"][series_idx].shape[0])

                    series_entry = np.interp(obs_time_series, time_series, series[series_idx])
                    obs_entry = self.obs_info["ground_truth_series"][series_idx]
                    std_entry = self.obs_info["std_series_vec"][series_idx]
                else:
                    min_len_series = min(self.obs_info["ground_truth_series"][series_idx].shape[0], len(series[series_idx]))
                    series_entry = series[series_idx][:min_len_series]
                    obs_entry = self.obs_info["ground_truth_series"][series_idx][:min_len_series]
                    # TODO make sure the std entries are the same shape as the obs entries
                    std_entry = self.obs_info["std_series_vec"][series_idx][:min_len_series]
                    
                
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
    
    def build_casadi_functions(self, param_names, param_vals = None, get_all_series=False):
        _require_casadi()
        self.sim_helper._create_param_subset(param_names, param_vals)

        self.cost_symb, self.obs_dict_symb = self.get_cost_and_obs_from_params(param_vals, do_ad=True)

        obs_outputs = []
        obs_meta = []

        for i, obs_item in enumerate(self.obs_dict_symb):
            output_dict = self.get_obs_output_dict(obs_item, get_all_series, is_symbolic=True)
            if get_all_series: 
                obs_dict_item, obs_series_array_all = output_dict
                self.obs_series_array_all_vec = ca.vertcat(*obs_series_array_all)
            else:
                obs_dict_item = output_dict

            for key in ['const', 'series', 'amp', 'phase', 'val_for_prob_dist']:
                val = obs_dict_item[key]

                if val is not None:
                    obs_outputs.append(val)
                    obs_meta.append((key, i, val.size1()))

        self.obs_vec = ca.vertcat(*obs_outputs)
        self.obs_meta = obs_meta

        self.jac_cost_symb = ca.gradient(self.cost_symb, self.sim_helper.variables_symb_subset)

        self.cost_func = ca.Function('cost_func', [self.sim_helper.states_symb, self.sim_helper.variables_symb], [self.cost_symb])
        
        if get_all_series:
            self.obs_func = ca.Function('obs_func', [self.sim_helper.states_symb, self.sim_helper.variables_symb], [self.obs_vec, self.obs_series_array_all_vec])
        else:
            self.obs_func = ca.Function('obs_func', [self.sim_helper.states_symb, self.sim_helper.variables_symb], [self.obs_vec])

        self.jac_cost_func = ca.Function('jac_cost_func', [self.sim_helper.states_symb, self.sim_helper.variables_symb], [self.jac_cost_symb])
    
    def get_jac_cost_ca(self, param_vals):
        param_names = self.param_id_info["param_names"]
        self.build_casadi_functions(param_names, param_vals)
        jac_cost = np.array(self.jac_cost_func(self.sim_helper.states, self.sim_helper.variables)).flatten()
        return jac_cost
    
    def get_cost_ca(self, param_vals):
        param_names = self.param_id_info["param_names"]
        self.build_casadi_functions(param_names, param_vals)
        cost= self.cost_func(self.sim_helper.states, self.sim_helper.variables)
        return cost
    
    def get_obs_ca(self, param_vals, get_all_series=False):
        param_names = self.param_id_info["param_names"]
        self.build_casadi_functions(param_names, param_vals, get_all_series)
        obs_val = self.obs_func(self.sim_helper.states, self.sim_helper.variables)

        if get_all_series:
            obs_dict, obs_series_array_all = obs_val
            series_np = np.array(obs_series_array_all)

            obs_series_array_all_formatted = [
                [series_np[i, :] for i in range(series_np.shape[0])]
            ]
        else:
            obs_dict = obs_val
        obs_dict = np.array(obs_dict).flatten()

        obs = []

        num_items = len(self.obs_dict_symb)
        for _ in range(num_items):
            obs.append({
                'const': None,
                'series': None,
                'amp': None,
                'phase': None,
                'val_for_prob_dist': None
            })

        idx = 0
        for key, i, size in self.obs_meta:
            values = obs_dict[idx:idx+size]

            if size == 1:
                values = values[0]

            obs[i][key] = values

            idx += size

        if get_all_series:
            return obs, obs_series_array_all_formatted
        else:
            return obs

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
                 dt=0.01, UQ_options=None, DEBUG=False, model_type=None):
        super().__init__(model_path, "MCMC",
                obs_info, param_id_info, protocol_info, prediction_info, solver_info,
                dt=dt, DEBUG=DEBUG, model_type=model_type)

        # mcmc init stuff
        self.sampler = None
        if UQ_options is not None:

            self._validate_UQ_options(UQ_options)

            self.UQ_options = UQ_options

            # Set library from UQ_options
            self.library = self.UQ_options.get('library', 'emcee')
            self.UQ_settings = self.UQ_options.get('settings', {}) 

            if self.library == 'emcee': 
                if 'num_steps' not in self.UQ_settings: 
                    self.UQ_settings['num_steps'] = 5000
                    print('number of mcmc steps is not set, choosing default of 5000')
                if 'num_walkers' not in self.UQ_settings:
                    self.UQ_settings['num_walkers'] = 2*self.num_params
                    print('number of mcmc walkers is not set, ',
                        'choosing default of 2*num_params')
                if 'burn_in' not in self.UQ_settings:  
                    self.UQ_settings['burn_in'] = 0.5  
                    print(f'burn_in is not set, choosing default of {self.UQ_settings["burn_in"]}')
                if 'method' not in self.UQ_settings:
                    self.UQ_settings['method'] = 'mcmc'
                    print(f'Method should be {self.UQ_settings["method"]} for {self.library}')
            
            elif self.library == 'pymc':
                if 'num_draws' not in self.UQ_settings: 
                    self.UQ_settings['num_draws'] = 1000
                    print('number of mcmc draws is not set, choosing default of 1000 for pymc')
                if 'num_chains' not in self.UQ_settings:
                    self.UQ_settings['num_chains'] = 4
                    print('number of mcmc chains is not set, ',
                        'choosing default of 4 for pymc')
                if 'burn_in' not in self.UQ_settings:  
                    self.UQ_settings['burn_in'] = 0.0  
                    print(f'burn_in is not set, choosing default of {self.UQ_settings["burn_in"]} for pymc')
                if 'method' not in self.UQ_settings:
                    self.UQ_settings['method'] = 'mcmc'
                    print(f'Method should be {self.UQ_settings["method"]} for {self.library}')
                if 'num_tune' not in self.UQ_settings:
                    self.UQ_settings['num_tune'] = 1000
                    print('number of mcmc tuning steps is not set, choosing default of 1000 for pymc')

        else:
            self.UQ_options = {}
            self.library = 'emcee'  # default to emcee
            self.UQ_settings['num_steps'] = 5000
            self.UQ_settings['num_walkers'] = 2*self.num_params
            self.UQ_settings['burn_in'] = 0.0  
            self.UQ_settings['method'] = 'mcmc'
            UQ_options['library'] = self.library
            UQ_options['settings'] = self.UQ_settings
            print(f'number of mcmc steps, walkers, burn_in, and method not set, '  
                f'choosing defaults of 5000, 2*num_params, {self.UQ_options["burn_in"]}, and {self.UQ_options["method"]} for {self.library}')
        
        # Import MCMC libraries based on library
        if self.library == 'zeus':
            try:
                import zeus
                self.zeus = zeus
            except ImportError:
                self.zeus = None
        elif self.library == 'pymc':
            try:
                import pymc as pm
                import pytensor.tensor as pt
                from pytensor.compile.ops import as_op
                self.pm = pm
                self.az = az
                self.pt = pt
                self.as_op = as_op
                @as_op(itypes=[pt.dvector], otypes=[pt.dscalar])
                def logp_op(theta):
                    # 1. Get the original log-likelihood/prior value
                    logp_val = mcmc_object.get_lnlikelihood_lnprior_from_params(theta)
                    
                    logp_val = np.asarray(logp_val)

                    if logp_val.shape != ():
                        logp_val = np.sum(logp_val)

                    return np.array(float(logp_val))
                self.logp_op = logp_op
            except ImportError as e:
                print(f"Failed to import pymc dependencies: {e}")
                self.pm = None
        else:
            print(f'unknown mcmc lib : {self.library}')

        self.DEBUG = DEBUG
        assert_mle_cost_for_bayesian(
            self.cost_type, self.cost_funcs_dict, "MCMC (log-likelihood uses -cost)"
        )

    def _validate_UQ_options(self, UQ_options):  
        
        """Validate UQ_options structure and library-specific settings."""  
        if UQ_options is None:  
            return  
        
        # Check library is specified  
        if 'library' not in UQ_options:  
            raise ValueError("UQ_options must contain 'library' key")  
        
        library = UQ_options['library']  
        valid_libraries = ['emcee', 'zeus', 'pymc']  
        if library not in valid_libraries:  
            raise ValueError(f"Invalid library '{library}'. Must be one of: {valid_libraries}")  
        
        # Check settings exist  
        if 'settings' not in UQ_options:  
            raise ValueError("UQ_options must contain 'settings' key")  
        
        settings = UQ_options['settings']  
        
        # Define valid settings for each library  
        valid_settings = {  
            'emcee': ['num_steps', 'num_walkers', 'burn_in', 'method'],  
            'zeus': ['num_steps', 'num_walkers', 'burn_in', 'method'],  
            'pymc': ['num_draws', 'num_chains', 'burn_in', 'num_tune', 'method']  
        }  
        
        # Check for invalid settings  
        valid_for_library = valid_settings[library]  
        for key in settings.keys():  
            if key not in valid_for_library:  
                raise ValueError(  
                    f"Invalid setting '{key}' for library '{library}'. "  
                    f"Valid settings for {library} are: {valid_for_library}"  
                )  
        
        # Check for required settings  
        required_settings = {  
            'emcee': ['num_steps', 'num_walkers'],  
            'zeus': ['num_steps', 'num_walkers'],  
            'pymc': ['num_draws', 'num_chains']  
        }  
        
        for required in required_settings[library]:  
            if required not in settings:  
                raise ValueError(  
                    f"Missing required setting '{required}' for library '{library}'. "  
                    f"Required settings for {library} are: {required_settings[library]}"  
                )
        
    def cost_calc(self, obs_dict, exp_idx=0, sub_idx=0, is_symbolic=False):
        """  
        Override cost calculation for MCMC to normalize non-zero weights to 1.0  
        """  

        const = obs_dict['const']
        series = obs_dict['series']
        amp = obs_dict['amp']
        phase = obs_dict['phase']
        val_for_prob_dist = obs_dict['val_for_prob_dist']

        # Get the original weights  
        updated_weight_const_vec = self.protocol_info["scaled_weight_const_from_exp_sub"][exp_idx][sub_idx]  
        updated_weight_series_vec = self.protocol_info["scaled_weight_series_from_exp_sub"][exp_idx][sub_idx]  
        updated_weight_amp_vec = self.protocol_info["scaled_weight_amp_from_exp_sub"][exp_idx][sub_idx]  
        updated_weight_phase_vec = self.protocol_info["scaled_weight_phase_from_exp_sub"][exp_idx][sub_idx]  
        updated_weight_prob_dist_vec = self.protocol_info["scaled_weight_prob_dist_from_exp_sub"][exp_idx][sub_idx]  
          
        # Normalize non-zero weights to 1.0 for MCMC  
        updated_weight_const_vec = np.where(updated_weight_const_vec != 0, 1.0, 0.0)  
        updated_weight_series_vec = np.where(updated_weight_series_vec != 0, 1.0, 0.0)  
        updated_weight_amp_vec = np.where(updated_weight_amp_vec != 0, 1.0, 0.0)  
        updated_weight_phase_vec = np.where(updated_weight_phase_vec != 0, 1.0, 0.0)  
        updated_weight_prob_dist_vec = np.where(updated_weight_prob_dist_vec != 0, 1.0, 0.0)  


        # get number of obs that don't have zero weights
        num_weighted_obs = np.sum(updated_weight_const_vec != 0) + \
                            np.sum(updated_weight_series_vec != 0) + \
                            np.sum(updated_weight_amp_vec != 0) + \
                            np.sum(updated_weight_phase_vec != 0) + \
                            np.sum(updated_weight_prob_dist_vec != 0)
        
        # this subexperiment doesn't have any weighted observables, so no cost
        if num_weighted_obs == 0.0:
            return 0.0
        
        if len(self.obs_info["ground_truth_phase"]) == 0:
            phase = None
        if self.obs_info["ground_truth_phase"].all() == None:
            phase = None
        
        if is_symbolic:
            _require_casadi()
            cost = ca.SX(0)
            if const is not None:
                for const_idx in range(const.size1()):
                    obs_idx = self.obs_info['const_idx_to_obs_idx'][const_idx]
                    if updated_weight_const_vec[const_idx] != 0:
                        cost += self.cost_funcs_dict[self.cost_type[obs_idx]](const[const_idx], self.obs_info["ground_truth_const"][const_idx],
                                                        self.obs_info["std_const_vec"][const_idx], updated_weight_const_vec[const_idx])
            return cost
        
        cost = 0.0
        if const is not None:
            for const_idx in range(len(const)):
                obs_idx = self.obs_info['const_idx_to_obs_idx'][const_idx]
                if updated_weight_const_vec[const_idx] != 0:
                    cost += self.cost_funcs_dict[self.cost_type[obs_idx]](const[const_idx], self.obs_info["ground_truth_const"][const_idx],
                                                    self.obs_info["std_const_vec"][const_idx], updated_weight_const_vec[const_idx])
        
        assert isinstance(cost, float), 'cost is not a float'

        series_cost = 0
        if series is not None:
        
            for series_idx in range(len(series)):
                if self.obs_info["obs_dt"][series_idx] != self.dt:
                    # interpolate the series to the dt of the ground truth series
                    time_series = np.linspace(0, series[series_idx].shape[0]*self.dt, series[series_idx].shape[0])
                    obs_time_series = np.linspace(0, self.obs_info["ground_truth_series"][series_idx].shape[0]*self.obs_info["obs_dt"][series_idx],
                                                    self.obs_info["ground_truth_series"][series_idx].shape[0])

                    series_entry = np.interp(obs_time_series, time_series, series[series_idx])
                    obs_entry = self.obs_info["ground_truth_series"][series_idx]
                    std_entry = self.obs_info["std_series_vec"][series_idx]
                else:
                    min_len_series = min(self.obs_info["ground_truth_series"][series_idx].shape[0], len(series[series_idx]))
                    series_entry = series[series_idx][:min_len_series]
                    obs_entry = self.obs_info["ground_truth_series"][series_idx][:min_len_series]
                    # TODO make sure the std entries are the same shape as the obs entries
                    std_entry = self.obs_info["std_series_vec"][series_idx][:min_len_series]
                    
                
                weight_entry = updated_weight_series_vec[series_idx]
                
                obs_idx = self.obs_info['series_idx_to_obs_idx'][series_idx]
                if weight_entry != 0:
                    series_cost += self.cost_funcs_dict[self.cost_type[obs_idx]](series_entry, obs_entry, std_entry, weight_entry)


        amp_cost = 0
        if amp is not None:
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

    def run(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        num_procs = comm.Get_size()
        if rank == 0:
            print(f'Running MCMC with {num_procs} processes using {self.library}')

        if num_procs > 1:
            # from pathos import multiprocessing
            # from pathos.multiprocessing import ProcessPool
            from schwimmbad import MPIPool

            if rank == 0 or (self.library == 'pymc'):
                num_chains_or_walkers = self.UQ_settings.get('num_chains') if self.library == 'pymc' else self.UQ_settings.get('num_walkers')
                if self.library == 'pymc':
                    num_chains_or_walkers = num_chains_or_walkers // num_procs if num_procs > 1 else num_chains_or_walkers
                
                if self.best_param_vals is not None:
                    best_param_vals_norm = self.param_norm_obj.normalise(self.best_param_vals)
                    # create initial params in gaussian ball around best_param_vals estimate
                    init_param_vals_norm = (np.ones((num_chains_or_walkers, self.num_params))*best_param_vals_norm).T + \
                                       0.1*np.random.randn(self.num_params, num_chains_or_walkers   )
                    init_param_vals_norm = np.clip(init_param_vals_norm, 0.001, 0.999)
                    init_param_vals = self.param_norm_obj.unnormalise(init_param_vals_norm)
                else:
                    init_param_vals_norm = np.random.rand(self.num_params, num_chains_or_walkers)
                    init_param_vals = self.param_norm_obj.unnormalise(init_param_vals_norm)

            if self.library == 'pymc':
                # BYPASS MPIPool: Let every rank reach the sampler
                print(f"Rank {rank} entering PyMC sampler logic")
            else:
                try:
                    pool = MPIPool() # workers dont get past this line in this try, they wait for work to do
                except:
                    return

                if not pool.is_master():
                    pool.wait()
                    return

            if self.library == 'emcee':
                robust_moves = [
                    (emcee.moves.StretchMove(), 1.0),      # 100% -
                ]
                # robust_moves = [
                #     # 80% weight: Handles strong parameter correlations using walker differences
                #     (emcee.moves.DEMove(), 0.60),
                    
                #     # 20% weight: Adds a scale-invariant jump to escape local minima/modes
                #     (emcee.moves.DESnookerMove(), 0.40),
                # ]
                self.sampler = emcee.EnsembleSampler(self.UQ_settings.get('num_walkers'), self.num_params, calculate_lnlikelihood,
                                            pool=pool, moves=robust_moves)
            elif self.library == 'zeus':
                self.sampler = self.zeus.EnsembleSampler(self.UQ_settings.get('num_walkers'), self.num_params, calculate_lnlikelihood,
                                                        pool=pool)
            elif self.library == 'pymc':  
                self.sampler = PyMCMPISampler(self.UQ_settings.get('num_chains'), self.num_params, calculate_lnlikelihood,  
                                            pool=True, param_id_info=self.param_id_info, num_tune=self.UQ_settings.get('num_tune'))          

            start_time = time.time()

            if self.UQ_settings['method'] == 'smc' and self.library == 'pymc':
                self.sampler.run_mcmc(init_param_vals.T, self.UQ_settings.get('num_draws'), method='smc') # , progress=True)
            else:
                print(f"Rank {rank} entering standard MCMC sampling with sampler {self.sampler}")
                num_draws_or_steps = self.UQ_settings.get('num_draws') if self.library == 'pymc' else self.UQ_settings.get('num_steps')
                self.sampler.run_mcmc(init_param_vals.T, num_draws_or_steps, progress=True)

            print(f'mcmc time = {time.time() - start_time}')
            
            if self.library != 'pymc':
                pool.close()

        else:
            num_chains_or_walkers = self.UQ_settings.get('num_chains') if self.library == 'pymc' else self.UQ_settings.get('num_walkers')
            if self.best_param_vals is not None:
                best_param_vals_norm = self.param_norm_obj.normalise(self.best_param_vals)
                init_param_vals_norm = (np.ones((num_chains_or_walkers, self.num_params))*best_param_vals_norm).T + \
                                   0.01*np.random.randn(self.num_params, num_chains_or_walkers)
                init_param_vals_norm = np.clip(init_param_vals_norm, 0.001, 0.999)
                init_param_vals = self.param_norm_obj.unnormalise(init_param_vals_norm)
            else:
                init_param_vals_norm = np.random.rand(self.num_params, num_chains_or_walkers)
                init_param_vals = self.param_norm_obj.unnormalise(init_param_vals_norm)

            if self.library == 'emcee':
                robust_moves = [
                    (emcee.moves.StretchMove(), 1.0),      # 100% -
                ]
                self.sampler = emcee.EnsembleSampler(num_chains_or_walkers, self.num_params, calculate_lnlikelihood, moves=robust_moves)
            elif self.library == 'zeus':
                self.sampler = self.zeus.EnsembleSampler(num_chains_or_walkers, self.num_params, calculate_lnlikelihood)
            elif self.library == 'pymc':  
                self.sampler = PyMCMPISampler(num_chains_or_walkers, self.num_params, calculate_lnlikelihood, 
                                              param_id_info=self.param_id_info, num_tune=self.UQ_settings.get('num_tune'))

            start_time = time.time()

            if self.UQ_settings['method'] == 'smc' and self.library == 'pymc':
                self.sampler.run_mcmc(init_param_vals.T, self.UQ_settings.get('num_draws'), method='smc', progress=True)
            else:
                num_draws_or_steps = self.UQ_settings.get('num_draws') if self.library == 'pymc' else self.UQ_settings.get('num_steps')
                self.sampler.run_mcmc(init_param_vals.T, num_draws_or_steps, progress=True)

            print(f'mcmc time = {time.time()-start_time}')

        if rank == 0:
            # TODO save chains
            if self.library == 'emcee':
                print(f'acceptance fraction was {self.sampler.acceptance_fraction}')
            samples = self.sampler.get_chain()
            mcmc_chain_path = os.path.join(self.output_dir, 'mcmc_chain.npy')
            np.save(mcmc_chain_path, samples)
            print('mcmc complete')
            print(f'mcmc chain saved in {mcmc_chain_path}')

            # save best param vals and best cost from mcmc mean
            burn_in_idx = int(samples.shape[0] * self.UQ_settings['burn_in'])  
            samples = samples[burn_in_idx:, :, :]
            
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
                 dt=0.01, UQ_options=None, 
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

        self.UQ_options = UQ_options

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

class PyMCMPISampler:  
    """Custom pyMC sampler that works with MPIPool like emcee/zeus"""  
    
    def __init__(self, num_walkers, num_params, log_prob_fn, pool=None, param_id_info=None, num_tune=1000):  
        self.num_walkers = num_walkers  
        self.num_params = num_params  
        self.log_prob_fn = log_prob_fn  
        self.pool = pool
        self.chain = None  
        self.param_id_info = param_id_info
        self.num_tune = num_tune
        
        import pytensor.tensor as pt
        import pymc as pm
        from pytensor.compile.ops import as_op
        
        self.pm = pm
        self.pt = pt
        self.as_op = as_op
        
        # Import PyMC dependencies
        try:
            self.logp_op = as_op(itypes=[pt.dvector], otypes=[pt.dscalar])(logp_op)
        except ImportError as e:
            raise ImportError(f"PyMC is required for PyMCMPISampler but is not installed: {e}")
    
    def run_mcmc(self, initial_state, num_draws_or_steps, method='mcmc', progress=False):    
        """Main entry point - choose between MCMC and SMC"""  
        if method == 'smc':  
            return self._run_smc(num_draws_or_steps)  
        else:  
            return self._run_mcmc(initial_state, num_draws_or_steps, num_tune=self.num_tune)
    
    def _run_smc(self, num_draws):  
        # """Run SMC with ABC using MPI parallelization"""  
        comm = MPI.COMM_WORLD    
        rank = comm.Get_rank()    
        num_procs = comm.Get_size()  

        def create_pymc_model():  
                with self.pm.Model() as model:  
                    # Create parameters with appropriate priors  
                    params = []  
                    for i in range(self.num_params):  
                        param_min = self.param_id_info["param_mins"][i]  
                        param_max = self.param_id_info["param_maxs"][i]  
                        prior_type = self.param_id_info["param_prior_types"][i]  
                        param_name = self.param_id_info["param_names_for_plotting"][i]

                        if prior_type == 'uniform' or not prior_type:  
                            params.append(self.pm.Uniform(param_name, lower=param_min, upper=param_max))  
                        elif prior_type == 'exponential':  
                            # Use λ=1.0 as in the original implementation  
                            lamb = 1.0  
                            params.append(self.pm.Exponential(param_name, lam=lamb))  
                        elif prior_type == 'normal':  
                            # Calculate mean and std as in the original implementation  
                            std = 1/6 * (param_max - param_min)  
                            mean = 0.5 * (param_max + param_min)  
                            params.append(self.pm.Normal(param_name, mu=mean, sigma=std))
                    
                    stacked_params = self.pm.math.stack(params) 

                    # Use the existing likelihood function  
                    self.pm.Potential('likelihood', logp_op(stacked_params))  
                
                return model
        
        model = create_pymc_model()  
        comm.Barrier()
        n_chains = self.num_walkers // num_procs if num_procs > 1 else self.num_walkers
        with model:  
            trace = self.pm.sample_smc(draws=num_draws,
                chains=n_chains,   
                cores=1,  
                progressbar= rank == 0)  

        print(f'Rank {rank} finished SMC sampling, waiting for others...')
        comm.Barrier()

        local_chain = self._convert_trace_to_emcee_format(trace)
        gathered_data = comm.gather(local_chain, root=0)

        if rank == 0:
            all_chains = np.array(gathered_data) 
            combined = np.concatenate(all_chains, axis=1)  # combine walkers
            self.chain = combined
            return combined
        else:
            return None

    def _run_mcmc(self, initial_state, num_draws, num_tune=1000):

        comm = MPI.COMM_WORLD  
        rank = comm.Get_rank()  
        num_procs = comm.Get_size()  

        with self.pm.Model() as model:
            
            params = []
            self.param_names = []
            for i in range(self.num_params):  
                param_min = self.param_id_info["param_mins"][i]  
                param_max = self.param_id_info["param_maxs"][i]  
                prior_type = self.param_id_info["param_prior_types"][i]  
                param_name = self.param_id_info["param_names_for_plotting"][i]
                self.param_names.append(param_name)

                if prior_type == 'uniform' or not prior_type:  
                    params.append(self.pm.Uniform(param_name, lower=param_min, upper=param_max))  
                elif prior_type == 'exponential':  
                    # Use λ=1.0 as in the original implementation  
                    lamb = 1.0  
                    params.append(self.pm.Exponential(param_name, lam=lamb))  
                elif prior_type == 'normal':  
                    # Calculate mean and std as in the original implementation  
                    std = 1/6 * (param_max - param_min)  
                    mean = 0.5 * (param_max + param_min)  
                    params.append(self.pm.Normal(param_name, mu=mean, sigma=std))

            stacked_params = self.pm.math.stack(params) 

            # Use the existing likelihood function  
            self.pm.Potential('likelihood', logp_op(stacked_params))  

            # comm.Barrier()  # Ensure all ranks have reached this point before sampling
            n_chains = self.num_walkers // num_procs if num_procs > 1 else self.num_walkers
            print(f"Rank {rank} starting MCMC sampling with {n_chains} chains")
            
            initvals_list = []
            for chain_idx in range(n_chains):
                # Create a dictionary for this specific chain
                chain_initvals = {
                    name: initial_state[chain_idx, param_idx] 
                    for param_idx, name in enumerate(self.param_names)
                }
                initvals_list.append(chain_initvals)

            trace = self.pm.sample(
                draws=num_draws,
                tune=num_tune,
                chains=n_chains,
                cores=1,
                step=self.pm.Metropolis(),
                progressbar= rank == 0,
                initvals=initvals_list
            )

        print(f'Rank {rank} finished pyMC MCMC sampling, waiting for others...')
        # Gather traces

        comm.Barrier()  # Ensure all ranks have finished sampling before gathering

        local_chain = self._convert_trace_to_emcee_format(trace)
        all_chains = comm.gather(local_chain, root=0)

        if rank == 0:
            combined = np.concatenate(all_chains, axis=1)  # combine walkers
            self.chain = combined
            return combined
        else:
            return None
    
    def _convert_trace_to_emcee_format(self, trace):  
        """Convert pyMC trace to emcee-compatible format"""  
        try:  
            # Extract samples from pyMC trace  
            if hasattr(trace, 'posterior'):  
                # Get the parameter variables from the trace  
                param_names = self.param_id_info["param_names_for_plotting"]
                
                # Extract samples as numpy array  
                samples_array = []  
                for param_name in param_names:  
                    if param_name in trace.posterior:  
                        param_samples = trace.posterior[param_name].values  
                        samples_array.append(param_samples)  
                    else:  
                        print(f"Warning: {param_name} not found in trace")  
                        return None  
                
                # Stack parameters: shape (chains, draws, params)  
                samples = np.stack(samples_array, axis=-1)  
                
                # Convert to emcee format: shape (steps, walkers, params)  
                # pyMC typically returns (chains, draws), so we transpose  
                samples = samples.transpose(1, 0, 2)  
                
                return samples  
            else:  
                print("Error: Trace object has no posterior attribute")  
                return None  
                
        except Exception as e:  
            print(f"Error converting trace to emcee format: {e}")  
            return None  
      
    def get_chain(self):  
        return self.chain

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

