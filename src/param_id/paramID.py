'''
@author: Finbar J. Argus
'''

import numpy as np
import os
import sys
from sys import exit
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities'))
# Ensure solver_wrappers is importable for SimulationHelper/opencor_helper
sys.path.append(os.path.join(os.path.dirname(__file__), '../solver_wrappers'))
import math as math
import opencor as oc
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
from parsers.PrimitiveParsers import scriptFunctionParser
from mpi4py import MPI
import re
from numpy import genfromtxt
from importlib import import_module
# import tqdm # TODO this needs to be installed for corner plot but doesnt need an import here
mcmc_lib = 'emcee' # TODO make this a user variable
if mcmc_lib == 'emcee':
    import emcee
elif mcmc_lib == 'zeus':
    import zeus
else:
    print(f'unknown mcmc lib : {mcmc_lib}')
import corner
import csv
from datetime import date
# from skopt import gp_minimize, Optimizer
from parsers.PrimitiveParsers import CSVFileParser
from param_id.optimisers import GeneticAlgorithmOptimiser, BayesianOptimiser, CMAESOptimiser
import pandas as pd
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
global mcmc_object
mcmc_object = None


class CVS0DParamID():
    """
    Class for doing parameter identification on a 0D cvs model
    """
    def __init__(self, model_path, model_type, param_id_method, mcmc_instead, file_name_prefix,
                 params_for_id_path=None,
                 param_id_obs_path=None, sim_time=2.0, pre_time=20.0, dt=0.01,
                 solver_info=None, mcmc_options=None, ga_options=None, optimiser_options=None, DEBUG=False,
                 param_id_output_dir=None, resources_dir=None, one_rank=False):
        self.model_path = model_path
        self.param_id_method = param_id_method
        self.mcmc_instead = mcmc_instead
        self.model_type = model_type
        self.file_name_prefix = file_name_prefix

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.num_procs = self.comm.Get_size()

        # For backwards compatibility, merge ga_options into optimiser_options
        self.optimiser_options = optimiser_options or {}
        if ga_options is not None:
            for key, value in ga_options.items():
                if key not in self.optimiser_options:
                    self.optimiser_options[key] = value
        self.mcmc_options = mcmc_options
        self.solver_info = solver_info
        self.dt = dt

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
        self.params_for_id_path = params_for_id_path
        if param_id_obs_path:
            self.__set_obs_names_and_df(param_id_obs_path, sim_time=sim_time, pre_time=pre_time)
        if self.params_for_id_path:
            self.__set_and_save_param_names()

        # ground truth values
        self.__get_ground_truth_values()

        # get prediction variables
        self.__set_prediction_var() # To be made obsolete, as prediction_info gets 
                                    # parsed in __set_obs_names_and_df

        if self.mcmc_instead:
            # This mcmc_object will be an instance of the OpencorParamID class
            # it needs to be global so that it can be used in calculate_lnlikelihood()
            # without having its attributes pickled. opencor simulation objects
            # can't be pickled because they are pyqt.
            global mcmc_object 
            mcmc_object = OpencorMCMC(self.model_path,
                                           self.obs_info, self.param_id_info,
                                           self.protocol_info, self.prediction_info, dt=self.dt,
                                           solver_info=self.solver_info, mcmc_options=mcmc_options,
                                           DEBUG=self.DEBUG, model_type=self.model_type)
            self.n_steps = mcmc_object.n_steps
        else:
            if model_type in ['cellml_only', 'python']:
                optimiser_options = getattr(self, 'optimiser_options', None)
                self.param_id = OpencorParamID(self.model_path, self.param_id_method,
                                               self.obs_info, self.param_id_info, self.protocol_info,
                                               self.prediction_info, dt=self.dt,
                                               solver_info=self.solver_info, ga_options=ga_options,
                                               optimiser_options=optimiser_options, DEBUG=self.DEBUG, 
                                               model_type=self.model_type)
                self.n_steps = self.param_id.n_steps
        if self.rank == 0:
            self.set_output_dir(self.output_dir)
        
        self.best_output_calculated = False
        self.sensitivity_calculated = False

    def temp_test(self):
        self.param_id.temp_test()
    def temp_test2(self):
        self.param_id.temp_test2()

    def run(self):
        self.param_id.run()

    def run_mcmc(self):
        mcmc_object.run()

    def run_single_sensitivity(self,sensitivity_output_path):
        self.param_id.run_single_sensitivity(sensitivity_output_path)

    def simulate_with_best_param_vals(self, reset=True, only_one_exp=-1):
        self.param_id.simulate_once(reset=reset, only_one_exp=only_one_exp)
        self.best_output_calculated = True

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
    
    def set_param_names(self, param_names):
        if self.mcmc_instead:
            mcmc_object.set_param_names(param_names)
        else:
            self.param_id.set_param_names(param_names)
        

        # TODO have to save param names as in __set_and_save_param_names!!

    def set_best_param_vals(self, best_param_vals):
        if self.mcmc_instead:
            mcmc_object.set_best_param_vals(best_param_vals)
        else:
            self.param_id.set_best_param_vals(best_param_vals)

    def plot_outputs(self):
        print('plotting best observables')
        m3_to_cm3 = 1e6
        Pa_to_kPa = 1e-3
        no_conv = 1.0
        if len(self.obs_info["ground_truth_phase"]) == 0:
            phase = False
        elif self.obs_info["ground_truth_phase"].all() == None:
            phase = False
        else: 
            phase = True

        cost, best_fit_operands_list = self.param_id.get_cost_and_obs_from_params(self.param_id.best_param_vals)
        list_of_obs_dicts = []
        list_of_all_series = []
        for obs in best_fit_operands_list:
            obs_dict, all_series = self.param_id.get_obs_output_dict(obs, get_all_series=True)
            list_of_obs_dicts.append(obs_dict)
            list_of_all_series.append(all_series)


        # _________ Plot best comparison _____________
        fig, axs = plt.subplots(squeeze=False)
        axs = axs[0,0]
        if phase:
            fig_phase, axs_phase = plt.subplots(squeeze=False)
            axs_phase = axs_phase[0,0]

        obs_tuples_unique = []
        for idx, obs_name in enumerate(self.obs_info["obs_names"]):
            if (obs_name, self.obs_info['experiment_idxs'][idx]) not in obs_tuples_unique:
                obs_tuples_unique.append((obs_name, self.obs_info['experiment_idxs'][idx]))

        plot_idx = 0
        subexp_count = -1
        tSim_per_sub_count = []
        sim_time_tot_per_exp = []
        n_steps_per_sub_count = []

        for exp_idx in range(self.protocol_info['num_experiments']):
            subexp_count += 1
            sim_time_tot_per_exp.append(np.sum(self.protocol_info['sim_times'][exp_idx]))
            n_steps_tot = int(sim_time_tot_per_exp[exp_idx]/self.dt)
            n_steps_per_sub_count = n_steps_per_sub_count + [int(self.protocol_info["sim_times"][exp_idx][II]/self.dt) for 
                            II in range(self.protocol_info["num_sub_per_exp"][exp_idx])]
            tSim = np.linspace(0.0, np.sum(self.protocol_info["sim_times"][exp_idx]), 
                            n_steps_tot + 1)
            tSim_per_sub_count.append(np.linspace(0.0, self.protocol_info["sim_times"][exp_idx][0], 
                                    n_steps_per_sub_count[subexp_count] + 1))
            start_time_sum = self.protocol_info["sim_times"][exp_idx][0]
            
            for II in range(1, self.protocol_info['num_sub_per_exp'][exp_idx]):
                subexp_count += 1
                tSim_per_sub_count.append(np.linspace(start_time_sum, 
                                            start_time_sum + self.protocol_info["sim_times"][exp_idx][II], 
                                            n_steps_per_sub_count[subexp_count] + 1))
                start_time_sum = start_time_sum + self.protocol_info["sim_times"][exp_idx][II]

        percent_error_vec = np.zeros((self.obs_info["num_obs"],))
        phase_error_vec = np.zeros((self.obs_info["num_obs"],))
        std_error_vec = np.zeros((self.obs_info["num_obs"],))
        series_label_set = False
        for unique_obs_count in range(len(obs_tuples_unique)):
            this_obs_waveform_plotted = False
            const_idx = -1
            series_idx = -1
            freq_idx = -1
            prob_dist_idx = -1
            for II in range(self.obs_info["num_obs"]):
                if self.obs_info["data_types"][II] == "constant":
                    const_idx += 1
                elif self.obs_info["data_types"][II] == "series":
                    series_idx += 1
                elif self.obs_info["data_types"][II] == "frequency":
                    freq_idx += 1
                elif self.obs_info["data_types"][II] == "prob_dist":
                    prob_dist_idx += 1
                
                # if the observable is the same as the one we are plotting
                if (self.obs_info["obs_names"][II], self.obs_info['experiment_idxs'][II]) == \
                        obs_tuples_unique[unique_obs_count]:
                    exp_idx = self.obs_info["experiment_idxs"][II]
                    this_sub_idx = self.obs_info["subexperiment_idxs"][II]
                    subexp_count = int(np.sum([num_sub for num_sub in 
                                               self.protocol_info["num_sub_per_exp"][:exp_idx]]) + this_sub_idx)
                    
                    series_per_sub = list_of_all_series[subexp_count]
                
                    best_fit_obs_const = list_of_obs_dicts[subexp_count]['const']
                    best_fit_obs_series = list_of_obs_dicts[subexp_count]['series']
                    best_fit_obs_amp = list_of_obs_dicts[subexp_count]['amp']
                    best_fit_obs_phase = list_of_obs_dicts[subexp_count]['phase']
                    best_fit_obs_prob_dist = list_of_obs_dicts[subexp_count]['val_for_prob_dist']


                    if len(self.obs_info["ground_truth_series"]) > 0:
                        if self.obs_info["obs_dt"][series_idx] == self.dt:
                            min_len_series = min(self.obs_info["ground_truth_series"][series_idx].shape[0], len(best_fit_obs_series[0]))
                        else:
                            min_len_series = len(best_fit_obs_series[0])
                    obs_name_for_plot = self.obs_info["names_for_plotting"][II]
                    
                    
                    if obs_name_for_plot.count('_') > 1:
                        print(f'obs_data variable "{obs_name_for_plot}" has too many underscores',
                            'for plotting a label. Include a "name_for_plotting" key in the ',
                            'obs_data json file entry')
                        exit()

                    if self.obs_info["units"][II] == 'm3_per_s':
                        conversion = m3_to_cm3
                        unit_label = '[cm^3/s]'
                    elif self.obs_info["units"][II] == 'm_per_s':
                        conversion = no_conv
                        unit_label = '[m/s]'
                    elif self.obs_info["units"][II] == 'm3':
                        conversion = m3_to_cm3
                        unit_label = '[cm^3]'
                    elif self.obs_info["units"][II] == 'J_per_m3':
                        conversion = Pa_to_kPa
                        unit_label = '[kPa]'
                    else:
                        conversion = 1.0
                        unit_label = f'[{self.obs_info["units"][II]}]'

                    if self.obs_info["data_types"][II] == 'series':
                        # if there is series data for this observable, use that label
                        axs.set_ylabel(f'${obs_name_for_plot}$ ${unit_label}$', fontsize=18)
                        series_label_set = True
                
                    if not this_obs_waveform_plotted:
                        axs.set_ylabel(f'${obs_name_for_plot}$ ${unit_label}$', fontsize=18)
                        if not self.obs_info["data_types"][II] == 'frequency':
                            # plot the waveform for all subexperiments
                            for temp_sub_idx in range(self.protocol_info["num_sub_per_exp"][exp_idx]):
                                temp_subexp_count = int(np.sum([num_sub for num_sub in 
                                                            self.protocol_info["num_sub_per_exp"][:exp_idx]]) + \
                                                            temp_sub_idx)
                                
                                temp_series_per_sub = list_of_all_series[temp_subexp_count]
                                if temp_sub_idx == 0:
                                    axs.plot(tSim_per_sub_count[temp_subexp_count], conversion*temp_series_per_sub[II][:], 
                                             color=self.protocol_info["experiment_colors"][exp_idx], label='output')
                                else:
                                    axs.plot(tSim_per_sub_count[temp_subexp_count], conversion*temp_series_per_sub[II][:], 
                                             color=self.protocol_info["experiment_colors"][exp_idx])
                                    
                            
                            axs.set_xlim(0.0, sim_time_tot_per_exp[exp_idx])
                            axs.set_xlabel('Time [$s$]', fontsize=18)
                        else:
                            axs.plot(self.obs_info["freqs"][II], conversion * best_fit_obs_amp[freq_idx],
                                                    color=self.protocol_info["experiment_colors"][exp_idx], marker='v', 
                                                    linestyle='', label='model output')
                            if phase:
                                axs_phase.plot(self.obs_info["freqs"][II], best_fit_obs_phase[freq_idx],
                                                        color=self.protocol_info["experiment_colors"][exp_idx], marker='v', 
                                                        linestyle='', label='model output')
                                axs_phase.set_ylabel(f'${obs_name_for_plot}$ phase', fontsize=18)

                            axs.set_xlim(0.0, self.obs_info["freqs"][II][-1])
                            axs.set_xlabel('frequency [$Hz$]', fontsize=18)
                        this_obs_waveform_plotted = True

 
                    if self.obs_info["data_types"][II] == 'constant':
                        if self.obs_info['plot_type'][II] == 'horizontal':
                            # TODO allow operation_funcs to define how we plot variables
                            # TODO so the user can customise plotting... Maybe not needed.
                            # create a vector equal to the constant value for plotting over the series
                            const_plot_gt = (self.obs_info["ground_truth_const"][const_idx])*\
                                            np.ones((n_steps_per_sub_count[subexp_count] + 1),)
                            const_plot_bf = (best_fit_obs_const[const_idx])*\
                                            np.ones((n_steps_per_sub_count[subexp_count] + 1),)

                            axs.plot(tSim_per_sub_count[subexp_count], conversion*const_plot_gt,
                                                    color=self.obs_info['plot_colors'][II], linestyle='--', 
                                                    label=f'{self.obs_info["operations"][II]} gt')
                            axs.plot(tSim_per_sub_count[subexp_count], conversion*const_plot_bf,
                                                    color=self.obs_info['plot_colors'][II], linestyle='-', 
                                                    label=f'{self.obs_info["operations"][II]} output')
                        elif self.obs_info['plot_type'][II] == 'horizontal_from_min':
                            # create a vector equal to the constant value for plotting over the series
                            min_val = np.min(series_per_sub[II])
                            const_plot_gt = (min_val + self.obs_info["ground_truth_const"][const_idx])*\
                                            np.ones((n_steps_per_sub_count[subexp_count] + 1),)
                            const_plot_bf = (min_val + best_fit_obs_const[const_idx])*\
                                            np.ones((n_steps_per_sub_count[subexp_count] + 1),)

                            axs.plot(tSim_per_sub_count[subexp_count], conversion*const_plot_gt,
                                                    color=self.obs_info['plot_colors'][II], linestyle='--', 
                                                    label=f'{self.obs_info["operations"][II]} gt')
                            axs.plot(tSim_per_sub_count[subexp_count], conversion*const_plot_bf,
                                                    color=self.obs_info['plot_colors'][II], linestyle='-', 
                                                    label=f'{self.obs_info["operations"][II]} output')
                        elif self.obs_info['plot_type'][II] == 'vertical':
                            # plot a vertical line at the t (x) value of the constant
                            axs.axvline(x=self.obs_info["ground_truth_const"][const_idx] - 
                                        self.protocol_info['pre_times'][exp_idx],
                                        color=self.obs_info['plot_colors'][II],
                                        linestyle='--', label=f'{self.obs_info["operations"][II]} desired')
                            axs.axvline(x=best_fit_obs_const[const_idx] - 
                                        self.protocol_info['pre_times'][exp_idx],
                                        color=self.obs_info['plot_colors'][II],
                                        label=f'{self.obs_info["operations"][II]} output')
                        elif self.obs_info['plot_type'][II] == 'vertical_from_subexp_start':
                            # calculate the t value, t values should be set as from the start of the subexperiment
                            t_gt = self.obs_info["ground_truth_const"][const_idx] + \
                                    tSim_per_sub_count[subexp_count][0] 
                            t_bf = best_fit_obs_const[const_idx] + \
                                    tSim_per_sub_count[subexp_count][0]
                            
                            # plot a vertical line at the t (x) value of the constant
                            axs.axvline(x=t_gt, 
                                        color=self.obs_info['plot_colors'][II],
                                        linestyle='--', label=f'{self.obs_info["operations"][II]} desired')
                            axs.axvline(x=t_bf,
                                        color=self.obs_info['plot_colors'][II],
                                        label=f'{self.obs_info["operations"][II]} output')
                        elif self.obs_info['plot_type'][II] == None:
                            pass
                        else:
                            print(f'plot_type for {self.obs_info["obs_names"][II]} ',
                                    f'of {self.obs_info["plot_constant_with_series_type"][II]} is not recognised',
                                    'for constants it must be in [None, horizontal, veritical, horizontal_from_min], exiting')
                            exit()
                    elif self.obs_info["data_types"][II] == 'series':
                        # create time vector for the grount_truth_series
                        start_time = np.sum(self.protocol_info['sim_times'][exp_idx][:this_sub_idx])
                        t_obs = np.linspace(start_time, start_time + self.protocol_info["sim_times"][exp_idx][this_sub_idx], 
                                            len(self.obs_info["ground_truth_series"][series_idx]))
                        axs.plot(t_obs, conversion*self.obs_info["ground_truth_series"][series_idx],
                                                'k--', label='gt')
                    elif self.obs_info["data_types"][II] == 'frequency':
                        axs.plot(self.obs_info["freqs"][II],
                                                conversion*self.obs_info["ground_truth_amp"][freq_idx],
                                                'kx', label='gt')
                        if phase:
                            axs_phase.plot(self.obs_info["freqs"][II],
                                                    self.obs_info["ground_truth_phase"][freq_idx],
                                                    'kx', label='gt')
                    if self.obs_info["data_types"][II] == 'prob_dist':
                        if self.obs_info["plot_type"][II] == 'horizontal':
                            for mean_idx, val_to_plot in enumerate(self.obs_info["ground_truth_prob_dist_params"][prob_dist_idx]["means"]):
                                mean_prob_dist_plot_gt = val_to_plot*\
                                                np.ones((n_steps_per_sub_count[subexp_count] + 1),)
                                axs.plot(tSim_per_sub_count[subexp_count], conversion*mean_prob_dist_plot_gt,
                                                        color=self.obs_info['plot_colors'][II], linestyle='--', 
                                                        label=f'{self.obs_info["operations"][II]} gt mean {mean_idx}')
                            val_for_prob_dist_plot_bf = (best_fit_obs_prob_dist[prob_dist_idx])*\
                                            np.ones((n_steps_per_sub_count[subexp_count] + 1),)

                            axs.plot(tSim_per_sub_count[subexp_count], conversion*val_for_prob_dist_plot_bf,
                                    color=self.obs_info['plot_colors'][II], linestyle='-', 
                                    label=f'{self.obs_info["operations"][II]} output')
                        elif self.obs_info["plot_type"] == None:
                            pass
                        else:
                            # other plot types havent been set up for prob dists yet
                            pass

                    #also calculate the RMS error for each observable
                    if exp_idx == self.obs_info["experiment_idxs"][II] and \
                            this_sub_idx == self.obs_info["subexperiment_idxs"][II]:
                        if self.obs_info["data_types"][II] == "constant":
                            percent_error_vec[II] = 100*(best_fit_obs_const[const_idx] - self.obs_info["ground_truth_const"][const_idx])/ \
                                                            (self.obs_info["ground_truth_const"][const_idx] + 1e-10) # add eps to avoid div by 0
                            std_error_vec[II] = (best_fit_obs_const[const_idx] - self.obs_info["ground_truth_const"][const_idx])/ \
                                                            self.obs_info["std_const_vec"][const_idx]
                        elif self.obs_info["data_types"][II] == "series":

                            if self.obs_info["obs_dt"][series_idx] != self.dt:
                                # interpolate the series to the dt of the ground truth series
                                time_series = np.linspace(0, best_fit_obs_series[series_idx].shape[0]*self.dt, best_fit_obs_series[series_idx].shape[0])
                                obs_time_series = np.linspace(0, self.obs_info["ground_truth_series"][series_idx].shape[0]*self.obs_info["obs_dt"][series_idx],
                                                                self.obs_info["ground_truth_series"][series_idx].shape[0])

                                series_entry = np.interp(obs_time_series, time_series, best_fit_obs_series[series_idx])
                                obs_entry = self.obs_info["ground_truth_series"][series_idx]
                                std_entry = self.obs_info["std_series_vec"][series_idx]
                            else:
                                min_len_series = min(self.obs_info["ground_truth_series"][series_idx].shape[0], len(best_fit_obs_series[series_idx]))
                                series_entry = best_fit_obs_series[series_idx][:min_len_series]
                                obs_entry = self.obs_info["ground_truth_series"][series_idx][:min_len_series]
                                # TODO make sure the std entries are the same shape as the obs entries
                                std_entry = self.obs_info["std_series_vec"][series_idx][:min_len_series]

                            percent_error_vec[II] = 100*np.sum(np.abs((obs_entry -
                                                                    series_entry) /
                                                                    (np.mean(obs_entry))))/len(obs_entry)
                            std_error_vec[II] = np.sum(np.abs((obs_entry -
                                                            series_entry /
                                                            (std_entry)))/len(obs_entry))
                        elif self.obs_info["data_types"][II] == "frequency":
                            std_error_vec[II] = np.sum(np.abs((best_fit_obs_amp[freq_idx] - self.obs_info["ground_truth_amp"][freq_idx]) *
                                                    self.obs_info["weight_amp_vec"][freq_idx] /
                                                    self.obs_info["std_amp_vec"][freq_idx]) / len(best_fit_obs_amp[freq_idx]))
                            percent_error_vec[II] = 100*np.sum(np.abs((best_fit_obs_amp[freq_idx] - self.obs_info["ground_truth_amp"][freq_idx]) /
                                                            np.mean(self.obs_info["ground_truth_amp"][freq_idx]))
                                                            / len(best_fit_obs_amp[freq_idx]))
                            if phase:
                                phase_error_vec[II] = np.sum(np.abs((best_fit_obs_phase[freq_idx] - self.obs_info["ground_truth_phase"][freq_idx])*
                                                                    self.obs_info["weight_phase_vec"][freq_idx]))/len(best_fit_obs_phase[freq_idx])
                        if self.obs_info["data_types"][II] == "prob_dist":
                            print('prob dist error not implemented properly yet error from first mean presented')
                            percent_error_vec[II] = 100*(best_fit_obs_prob_dist[prob_dist_idx] - 
                                                         self.obs_info["ground_truth_prob_dist_params"][prob_dist_idx]["means"][0])/ \
                                                            (self.obs_info["ground_truth_prob_dist_params"][prob_dist_idx]["means"][0]
                                                             + 1e-10) # add eps to avoid div by 0
                            if "stds" in self.obs_info["ground_truth_prob_dist_params"][prob_dist_idx].keys():
                                std_error_vec[II] = (best_fit_obs_prob_dist[prob_dist_idx] - 
                                                            self.obs_info["ground_truth_prob_dist_params"][prob_dist_idx]["means"][0])/ \
                                                                self.obs_info["ground_truth_prob_dist_params"][prob_dist_idx]["stds"][0]
                            else:
                                std_error_vec[II] = 0.0 # TODO this is a placeholder
                                


            # axs.set_ylim(ymin=0.0)
            # axs.set_yticks(np.arange(0, 21, 10))

            plot_saved = False

            axs.legend(fontsize=10)
            if phase:
                axs_phase.legend(loc='upper right', fontsize=10)
            fig.tight_layout()
            if phase:
                fig_phase.tight_layout()
            fig.savefig(os.path.join(self.plot_dir,
                                        f'reconstruct_{self.file_name_prefix}_'
                                        f'{self.param_id_obs_file_prefix}_{plot_idx}.eps'))
            fig.savefig(os.path.join(self.plot_dir,
                                        f'reconstruct_{self.file_name_prefix}_'
                                        f'{self.param_id_obs_file_prefix}_{plot_idx}.pdf'))
            axs.legend().get_frame().set_alpha(0.5)
            fig.savefig(os.path.join(self.plot_dir,
                                        f'reconstruct_{self.file_name_prefix}_'
                                        f'{self.param_id_obs_file_prefix}_{plot_idx}.png'))
            plt.close(fig)
            
            if phase:
                fig_phase.savefig(os.path.join(self.plot_dir,
                                        f'phase_reconstruct_{self.file_name_prefix}_'
                                        f'{self.param_id_obs_file_prefix}_{plot_idx}.eps'))
                fig_phase.savefig(os.path.join(self.plot_dir,
                                        f'phase_reconstruct_{self.file_name_prefix}_'
                                        f'{self.param_id_obs_file_prefix}_{plot_idx}.pdf'))
                axs_phase.legend().get_frame().set_alpha(0.5)
                fig_phase.savefig(os.path.join(self.plot_dir,
                                        f'phase_reconstruct_{self.file_name_prefix}_'
                                        f'{self.param_id_obs_file_prefix}_{plot_idx}.png'))

                plt.close(fig_phase)

            plot_saved = True
            plot_idx += 1
            # create new plot
            if unique_obs_count != len(obs_tuples_unique) - 1:
                fig, axs = plt.subplots(squeeze=False)
                axs = axs[0,0]
                if phase:
                    fig_phase, axs_phase = plt.subplots(squeeze=False)
                    axs_phase = axs_phase[0,0]
                plot_saved = False
        
        # save std error and percentage error
        np.save(os.path.join(self.output_dir, 'percent_error_vec.npy'), percent_error_vec)
        np.save(os.path.join(self.output_dir, 'std_error_vec.npy'), std_error_vec)

        # save final plot if it is not a full set of subplots
        if not plot_saved:
            axs.legend(loc='lower right', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir,
                                    f'reconstruct_{self.file_name_prefix}_'
                                    f'{self.param_id_obs_file_prefix}_{plot_idx}.eps'))
            plt.savefig(os.path.join(self.plot_dir,
                                    f'reconstruct_{self.file_name_prefix}_'
                                    f'{self.param_id_obs_file_prefix}_{plot_idx}.pdf'))
            plt.close()

        # Make a bar plot with all percentage errors.
        obs_names_for_plot_list = []

        for II in range(self.obs_info["num_obs"]):
            name_str = f'${self.obs_info["names_for_plotting"][II]}$'
            obs_names_for_plot_list.append(name_str)
        obs_names_for_plot = np.array(obs_names_for_plot_list)

        num_plots = len(obs_names_for_plot)// 10 + 1

        for plot_idx in range(num_plots):
            fig, axs = plt.subplots()
            if plot_idx == num_plots - 1:
                bar_list = axs.bar(obs_names_for_plot[plot_idx*10:], percent_error_vec[plot_idx*10:], label='% error', width=1.0, color='b', edgecolor='black')
            else:
                bar_list = axs.bar(obs_names_for_plot[plot_idx*10:plot_idx*10+10], percent_error_vec[plot_idx*10:plot_idx*10+10], label='% error', width=1.0, color='b', edgecolor='black')

            # axs.set_ylim(ymin=0.0)
            # axs.set_yticks(np.arange(0, 21, 10))
            axs.axhline(y=0.0,linewidth= 3, color='k', linestyle= 'dotted')

            # bar_list[0].set_facecolor('r')
            # bar_list[1].set_facecolor('r')

            # axs.legend()
            axs.set_ylabel(r'E$_{\%}$')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir,
                                    f'error_bars_{self.file_name_prefix}_'
                                    f'{self.param_id_obs_file_prefix}_{plot_idx}.eps'))
            plt.savefig(os.path.join(self.plot_dir,
                                    f'error_bars_{self.file_name_prefix}_'
                                    f'{self.param_id_obs_file_prefix}_{plot_idx}.pdf'))
            plt.savefig(os.path.join(self.plot_dir,
                                    f'error_bars_{self.file_name_prefix}_'
                                    f'{self.param_id_obs_file_prefix}_{plot_idx}.png'))
            plt.close()

            #plot error as number of standard deviations of
            fig, axs = plt.subplots()
            if plot_idx == num_plots - 1:
                bar_list = axs.bar(obs_names_for_plot[plot_idx*10:], std_error_vec[plot_idx*10:], label='% error', width=1.0, color='b', edgecolor='black')
            else:
                bar_list = axs.bar(obs_names_for_plot[plot_idx*10:plot_idx*10+10], std_error_vec[plot_idx*10:plot_idx*10+10], label='% error', width=1.0, color='b', edgecolor='black')
            axs.axhline(y=0.0,linewidth=3, color='k', linestyle= 'dotted')

            

            # bar_list[0].set_facecolor('r')
            # bar_list[1].set_facecolor('r')

            # axs.legend()
            axs.set_ylabel('E$_{std}$')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir,
                                    f'std_error_bars_{self.file_name_prefix}_'
                                    f'{self.param_id_obs_file_prefix}_{plot_idx}.eps'))
            plt.savefig(os.path.join(self.plot_dir,
                                    f'std_error_bars_{self.file_name_prefix}_'
                                    f'{self.param_id_obs_file_prefix}_{plot_idx}.pdf'))
            plt.savefig(os.path.join(self.plot_dir,
                                    f'std_error_bars_{self.file_name_prefix}_'
                                    f'{self.param_id_obs_file_prefix}_{plot_idx}.png'))
            plt.close()

        print('______observable errors______')
        for obs_idx in range(self.obs_info["num_obs"]):
            if self.gt_df.iloc[obs_idx]["data_type"] == "constant":
                if self.obs_info["operations"][obs_idx] is not None:
                    print(f'{self.obs_info["names_for_plotting"][obs_idx]} {self.obs_info["operations"][obs_idx]} error:')
                else:
                    print(f'{self.obs_info["names_for_plotting"][obs_idx]} {self.obs_info["data_types"][obs_idx]} error:')
                print(f'{percent_error_vec[obs_idx]:.2f} %')
            if self.gt_df.iloc[obs_idx]["data_type"] == "series":
                if self.obs_info["operations"][obs_idx] is not None:
                    print(f'{self.obs_info["names_for_plotting"][obs_idx]} {self.obs_info["operations"][obs_idx]} series error:')
                else:
                    print(f'{self.obs_info["obs_names"][obs_idx]} {self.obs_info["data_types"][obs_idx]} error:')
                print(f'{percent_error_vec[obs_idx]:.2f} %')
            if self.gt_df.iloc[obs_idx]["data_type"] == "frequency":
                print(f'{self.obs_info["names_for_plotting"][obs_idx]} {self.obs_info["data_types"][obs_idx]} error:')
                print(f'{percent_error_vec[obs_idx]:.2f} %')
                if phase:
                    print(f'{self.obs_info["names_for_plotting"][obs_idx]} {self.obs_info["data_types"][obs_idx]} phase error:')
                    print(f'{phase_error_vec[obs_idx]:.2f}')
            # TODO find a way to usefully plot the error for prob dists

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
        samples = samples[samples.shape[0]//2:, :, :]
        # thin = 5
        # samples = samples[::thin, :, :]
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

    def set_genetic_algorithm_parameters(self, n_calls):
        self.param_id.set_genetic_algorithm_parameters(n_calls)

    def set_bayesian_parameters(self, n_calls, n_initial_points, acq_func, random_state, acq_func_kwargs={}):
        self.param_id.set_bayesian_parameters(n_calls, n_initial_points, acq_func, random_state,
                                              acq_func_kwargs=acq_func_kwargs)

    def close_simulation(self):
        if self.mcmc_instead:
            mcmc_object.close_simulation()
        else:
            self.param_id.close_simulation()

    def __set_obs_names_and_df(self, param_id_obs_path, pre_time=None, sim_time=None):
        # TODO this function should be in the parsing section. as it parses the 
        # ground truth data.
        # TODO it should also be cleaned up substantially.
        """_summary_

        Args:
            param_id_obs_path (_type_): _description_
            pre_time (_type_): _description_
            sim_time (_type_): _description_
        """
        with open(param_id_obs_path, encoding='utf-8-sig') as rf:
            json_obj = json.load(rf)
        if type(json_obj) == list:
            self.gt_df = pd.DataFrame(json_obj)
            self.protocol_info = {"pre_times": [pre_time], 
                                    "sim_times": [[sim_time]],
                                    "params_to_change": [[None]]}
            self.prediction_info = {'names': [],
                                    'units': [],
                                    'names_for_plotting': [],
                                    'experiment_idxs': []}
        elif type(json_obj) == dict:
            if 'data_items' in json_obj.keys():
                self.gt_df = pd.DataFrame(json_obj['data_items'])
            elif 'data_item' in json_obj.keys():
                self.gt_df = pd.DataFrame(json_obj['data_item']) # should be data_items but accept this
            else:
                print("data_items not found in json object. ",
                      "Please check that data_items is the key for the list of data items")
            if 'protocol_info' in json_obj.keys():
                self.protocol_info = json_obj['protocol_info']
                if "sim_times" not in self.protocol_info.keys():
                    self.protocol_info["sim_times"] = [[sim_time]]
                if "pre_times" not in self.protocol_info.keys():
                    self.protocol_info["pre_times"] = [pre_time]
            else:
                if pre_time is None or sim_time is None:
                    print("protocol_info not found in json object. ",
                          "If this is the case sim_time and pre_time must be set",
                          "in the user_inputs.yaml file")
                    exit()

                self.protocol_info = {"pre_times": [pre_time], 
                                      "sim_times": [[sim_time]],
                                      "params_to_change": [[None]]}
            if 'prediction_items' in json_obj.keys():
                self.prediction_info = {'names': [],
                                        'units': [],
                                        'names_for_plotting': [],
                                        'experiment_idxs': []}

                for entry in json_obj['prediction_items']:
                    if 'variable' in entry.keys():
                        self.prediction_info['names'].append(entry['variable'])
                    else:
                        print('"variable" not found in prediction item in obs_data.json file, ',
                              'exitiing') 
                        exit()
                    if 'unit' in entry.keys():
                        self.prediction_info['units'].append(entry['unit'])
                    else:
                        print('"unit" not found in prediction item in obs_data.json file, ',
                              'exitiing') 
                        exit()
                    if 'name_for_plotting' in entry.keys():
                        self.prediction_info['names_for_plotting'].append(entry['name_for_plotting'])
                    else:
                        self.prediction_info['names_for_plotting'].append(entry['variable'])
                    if 'experiment_idx' in entry.keys():
                        self.prediction_info['experiment_idxs'].append(entry['experiment_idx'])
                    else:
                        self.prediction_info['experiment_idxs'].append(0)
            else:
                self.prediction_info = None
        else:
            print(f"unknown data type for imported json object of {type(json_obj)}")
        
        self.obs_info = {}
        self.obs_info["obs_names"] = [self.gt_df.iloc[II]["variable"] for II in range(self.gt_df.shape[0])]

        # OBSOLETE self.obs_types = [self.gt_df.iloc[II]["obs_type"] for II in range(self.gt_df.shape[0])]
        self.obs_info["data_types"] = [self.gt_df.iloc[II]["data_type"] for II in range(self.gt_df.shape[0])]
        self.obs_info["units"] = [self.gt_df.iloc[II]["unit"] for II in range(self.gt_df.shape[0])]
        self.obs_info["experiment_idxs"] = [self.gt_df.iloc[II]["experiment_idx"] if "experiment_idx" in 
                                            self.gt_df.iloc[II].keys() else 0 for II in range(self.gt_df.shape[0])]
        self.obs_info["subexperiment_idxs"] = [self.gt_df.iloc[II]["subexperiment_idx"] if "subexperiment_idx" in
                                               self.gt_df.iloc[II].keys() else 0 for II in range(self.gt_df.shape[0])]

        # get plotting color, asign to randomish color if not defined
        # list of all possible colors
        possible_colors = ['b', 'g', 'c', 'm', 'y', 
                           'tab:brown', 'tab:pink', 'tab:olive', 'tab:orange'] # don't include red or black, 
                                                    # because they are used for plotting the series
        self.obs_info["plot_colors"] = [self.gt_df.iloc[II]["plot_color"] if "plot_color" in 
                                        self.gt_df.iloc[II].keys() else possible_colors[II%len(possible_colors)] 
                                        for II in range(self.gt_df.shape[0])]
        self.obs_info["plot_type"] = []

        # get plotting type
        # TODO make the plot_types operation_funcs so the user can defined how they are plotted.
        warning_printed = False
        for II in range(self.gt_df.shape[0]):
            if "plot_type" not in self.gt_df.iloc[II].keys():
                if self.gt_df.iloc[II]["data_type"] == "constant":
                    if not warning_printed:
                        print('constant data types plot type defaults to horizontal lines',
                            'change "plot_type" in obs_data.json to change this')
                        warning_printed = True
                    self.obs_info["plot_type"].append("horizontal")
                elif self.gt_df.iloc[II]["data_type"] == "prob_dist":
                    if not warning_printed:
                        print('prob_dist data types plot type defaults to horizontal lines',
                            'change "plot_type" in obs_data.json to change this')
                        warning_printed = True
                    self.obs_info["plot_type"].append("horizontal")
                elif self.gt_df.iloc[II]["data_type"] == "series":
                    self.obs_info["plot_type"].append("series")
                elif self.gt_df.iloc[II]["data_type"] == "frequency":
                    self.obs_info["plot_type"].append("frequency")
                elif self.gt_df.iloc[II]["data_type"] == "plot_dist":
                    self.obs_info["plot_type"].append("horizontal")
                else:
                    print(f'data type {self.gt_df.iloc[II]["data_type"]} not recognised')
            else:
                self.obs_info["plot_type"].append(self.gt_df.iloc[II]["plot_type"])
                if self.obs_info["plot_type"][II] in ["None", "null", "Null", "none", "NONE"]:
                    self.obs_info["plot_type"][II] = None

        self.obs_info["operations"] = []
        self.obs_info["names_for_plotting"] = []
        self.obs_info["operands"] = []
        self.obs_info["freqs"] = []
        self.obs_info["operation_kwargs"] = []
        # below we remove the need for obs_types, but keep it backwards compatible so 
        # previous specifications of obs_type = mean etc should still work
        for II in range(self.gt_df.shape[0]):
            if "operation" not in self.gt_df.iloc[II].keys() or \
                    self.gt_df.iloc[II]["operation"] in ["Null", "None", "null", "none", "", "nan", np.nan]:
                if "obs_type" in self.gt_df.iloc[II].keys():
                    if self.gt_df.iloc[II]["obs_type"] == "series":
                        self.obs_info["operations"].append(None)
                        if "operands" in self.gt_df.iloc[II].keys():
                            self.obs_info["operands"].append(self.gt_df.iloc[II]["operands"])
                        else:
                            self.obs_info["operands"].append(None)
                    elif self.gt_df.iloc[II]["obs_type"] == "frequency":
                        self.obs_info["operations"].append(None)
                        if "operands" in self.gt_df.iloc[II].keys():
                            self.obs_info["operands"].append(self.gt_df.iloc[II]["operands"])
                        else:
                            self.obs_info["operands"].append(None)
                    # TODO remove these eventually when I get rid of obs_type
                    elif self.gt_df.iloc[II]["obs_type"] == "min":
                        self.obs_info["operations"].append("min")
                        self.obs_info["operands"].append([self.gt_df.iloc[II]["variable"]])
                    elif self.gt_df.iloc[II]["obs_type"] == "max":
                        self.obs_info["operations"].append("max")
                        self.obs_info["operands"].append([self.gt_df.iloc[II]["variable"]])
                    elif self.gt_df.iloc[II]["obs_type"] == "mean":
                        self.obs_info["operations"].append("mean")
                        self.obs_info["operands"].append([self.gt_df.iloc[II]["variable"]])
                else:
                    self.obs_info["operations"].append(None)
                    if "operands" in self.gt_df.iloc[II].keys():
                        self.obs_info["operands"].append(self.gt_df.iloc[II]["operands"])
                    else:
                        self.obs_info["operands"].append(None)
            elif self.gt_df.iloc[II]["operation"] in ["Null", "None", "null", "none", ""]:
                self.obs_info["operations"].append(None)
                self.obs_info["operands"].append(None)
            else:
                self.obs_info["operations"].append(self.gt_df.iloc[II]["operation"])
                self.obs_info["operands"].append(self.gt_df.iloc[II]["operands"])

            if "frequencies" not in self.gt_df.iloc[II].keys():
                self.obs_info["freqs"].append(None)
            else:
                self.obs_info["freqs"].append(self.gt_df.iloc[II]["frequencies"])

            if "name_for_plotting" in self.gt_df.iloc[II].keys():
                self.obs_info['names_for_plotting'].append(self.gt_df.iloc[II]["name_for_plotting"])
            else:
                self.obs_info['names_for_plotting'].append(self.obs_info["obs_names"][II])

            if "operation_kwargs" in self.gt_df.iloc[II].keys() and self.gt_df.iloc[II]["operation_kwargs"] \
                    not in ["Null", "None", "null", "none", "", np.nan]:
                self.obs_info["operation_kwargs"].append(self.gt_df.iloc[II]["operation_kwargs"])
            else:
                self.obs_info["operation_kwargs"].append({})

            if self.gt_df.iloc[II]["data_type"] == "series":
                if 'sample_rate' in self.gt_df.iloc[II].keys():
                    print(f'WARNING sample_rate found in obs_data.json for observable {self.obs_info["obs_names"][II]}, ',
                        'this is deprecated, please use obs_dt instead. Setting obs_dt = (1 / sample_rate)')
                    self.gt_df.iloc[II]["obs_dt"] = 1 / self.gt_df.iloc[II]["sample_rate"]
                if 'dt' in self.gt_df.iloc[II].keys():
                    print(f'WARNING dt found in obs_data.json for observable {self.obs_info["obs_names"][II]}, ',
                        'this is deprecated, please use obs_dt instead. Setting obs_dt = dt')
                    self.gt_df.iloc[II]["obs_dt"] = self.gt_df.iloc[II]["dt"]
            else: 
                if 'obs_dt' in self.gt_df.iloc[II].keys():
                    print(
                        f"obs_dt found in obs_data.json for observable {self.obs_info['obs_names'][II]}, "
                        f"with data type {self.gt_df.iloc[II]['data_type']}. obs_dt should only be used for series data. Exiting"
                    )
                    exit()
                if 'dt' in self.gt_df.iloc[II].keys():
                    print(
                        f"WARNING dt found in obs_data.json for observable {self.obs_info['obs_names'][II]}, "
                        f"with data type {self.gt_df.iloc[II]['data_type']}. dt is deprecated for obs_dt "
                        "and obs_dt should only be used for series data. Exiting"
                    )
                    exit()
                if 'sample_rate' in self.gt_df.iloc[II].keys():
                    print(
                        f"WARNING sample_rate found in obs_data.json for observable {self.obs_info['obs_names'][II]}, "
                        f"with data type {self.gt_df.iloc[II]['data_type']}. sample_rate is deprecated for obs_dt "
                        "and obs_dt should only be used for series data. Exiting"
                    )
                    exit()


        self.obs_info["num_obs"] = len(self.obs_info["obs_names"])

        # how much to weight the different observable errors by
        self.obs_info["weight_const_vec"] = np.array([self.gt_df.iloc[II]["weight"] for II in range(self.gt_df.shape[0])
                                          if self.gt_df.iloc[II]["data_type"] == "constant"])

        self.obs_info["weight_series_vec"] = np.array([self.gt_df.iloc[II]["weight"] for II in range(self.gt_df.shape[0])
                                           if self.gt_df.iloc[II]["data_type"] == "series"])

        self.obs_info["weight_amp_vec"] = np.array([self.gt_df.iloc[II]["weight"] for II in range(self.gt_df.shape[0])
                                           if self.gt_df.iloc[II]["data_type"] == "frequency"])
        
        self.obs_info["weight_prob_dist_vec"] = np.array([self.gt_df.iloc[II]["weight"] for II in range(self.gt_df.shape[0])
                                          if self.gt_df.iloc[II]["data_type"] == "prob_dist"])

        weight_phase_list = [] 
        for II in range(self.gt_df.shape[0]):
            if self.gt_df.iloc[II]["data_type"] == "frequency":
                if "phase_weight" not in self.gt_df.iloc[II].keys():
                    weight_phase_list.append(1)
                else:
                    weight_phase_list.append(self.gt_df.iloc[II]["phase_weight"])
        self.obs_info["weight_phase_vec"] = np.array(weight_phase_list)

        # set the cost type for each observable
        self.obs_info["cost_type"] = []
        for II in range(self.gt_df.shape[0]):
            if "cost_type" in self.gt_df.iloc[II].keys() and self.gt_df.iloc[II]["cost_type"] not in [np.nan, None, "None", ""]:
                self.obs_info["cost_type"].append(self.gt_df.iloc[II]["cost_type"])
            else:
                if self.optimiser_options is not None and len(self.optimiser_options) > 0:
                    if "cost_type" in self.optimiser_options.keys():
                        self.obs_info["cost_type"].append(self.optimiser_options["cost_type"]) # default to cost type in optimiser_options
                    else:
                        self.obs_info["cost_type"].append("MSE") # default to mean squared error
                elif self.mcmc_options is not None:
                    if "cost_type" in self.mcmc_options.keys():
                        self.obs_info["cost_type"].append(self.mcmc_options["cost_type"]) # default to cost type in mcmc_options
                    else:
                        self.obs_info["cost_type"].append("MSE") # default to mean squared error
                else:
                    print("cost_type not found in obs_data.json, ga_options, or mcmc_options, exiting")
                    exit()



        # preprocess information in the protocol_info dataframe
        self.protocol_info['num_experiments'] = len(self.protocol_info["sim_times"])
        self.protocol_info['num_sub_per_exp'] = [len(self.protocol_info["sim_times"][II]) for II in range(self.protocol_info["num_experiments"])]
        self.protocol_info['num_sub_total'] = sum(self.protocol_info['num_sub_per_exp'])

        # calculate total experiment sim times
        self.protocol_info["total_sim_times_per_exp"] = []
        self.protocol_info["tSims_per_exp"] = []
        self.protocol_info["num_steps_total_per_exp"] = []
        for exp_idx in range(self.protocol_info['num_experiments']):
            total_sim_time = np.sum([self.protocol_info["sim_times"][exp_idx][II] for
                            II in range(self.protocol_info["num_sub_per_exp"][exp_idx])])
            num_steps_total = int(total_sim_time/self.dt)
            tSim_per_exp = np.linspace(0.0, total_sim_time, num_steps_total + 1)
            self.protocol_info["total_sim_times_per_exp"].append(total_sim_time)
            self.protocol_info["tSims_per_exp"].append(tSim_per_exp)
            self.protocol_info["num_steps_total_per_exp"].append(num_steps_total)
            

        if "experiment_colors" not in self.protocol_info.keys():
            self.protocol_info["experiment_colors"] = ['r']
            if self.protocol_info['num_experiments'] > 1:
                self.protocol_info["experiment_colors"] = ['r']*self.protocol_info['num_experiments']
        else:
            if len(self.protocol_info["experiment_colors"]) != self.protocol_info['num_experiments']:
                print('experiment_colors in obs_data.json not the same length as num_experiments, exiting')
                exit()

        if "experiment_labels" in self.protocol_info.keys():
            if len(self.protocol_info["experiment_labels"]) != self.protocol_info['num_experiments']:
                print('experiment_labels in obs_data.json not the same length as num_experiments, exiting')
                exit()
        else:
            self.protocol_info["experiment_labels"] = [None]
            if self.protocol_info['num_experiments'] > 1:
                self.protocol_info["experiment_labels"] = [None]*self.protocol_info['num_experiments']
        
        # set experiment and subexperiment idxs to 0 if they are not defined. print warning if multiple subexperiments
        for II in range(self.gt_df.shape[0]):
            if "experiment_idx" not in self.gt_df.iloc[II].keys():
                self.gt_df["experiment_idx"] = 0
                if self.protocol_info['num_sub_total'] > 1:
                    print(f'experiment_idx not found in obs_data.json entry {self.gt_df.iloc[II]["variable"]}, '
                          'but multiple experiments are defined.',
                          'Setting experiment_idx to 0 for all data points')
            if "subexperiment_idx" not in self.gt_df.iloc[II].keys():
                self.gt_df["subexperiment_idx"] = 0
                if self.protocol_info['num_sub_total'] > 1:
                    print(f'subexperiment_idx not found in obs_data.json entry {self.gt_df.iloc[II]["variable"]}, '
                          'but multiple subexperiments are defined.',
                          'Setting subexperiment_idx to 0 for all data points')
        
        # calculate the mapping from sub and experiment idx to the weight of the observable for that subexperiment
        const_map = [[[] for sub_idx in range(self.protocol_info['num_sub_per_exp'][exp_idx])]
                     for exp_idx in range(self.protocol_info['num_experiments'])]
        series_map = [[[] for sub_idx in range(self.protocol_info['num_sub_per_exp'][exp_idx])]
                     for exp_idx in range(self.protocol_info['num_experiments'])]
        amp_map = [[[] for sub_idx in range(self.protocol_info['num_sub_per_exp'][exp_idx])]
                     for exp_idx in range(self.protocol_info['num_experiments'])]
        phase_map = [[[] for sub_idx in range(self.protocol_info['num_sub_per_exp'][exp_idx])]
                     for exp_idx in range(self.protocol_info['num_experiments'])]
        prob_dist_map = [[[] for sub_idx in range(self.protocol_info['num_sub_per_exp'][exp_idx])]
                     for exp_idx in range(self.protocol_info['num_experiments'])]

        for exp_idx in range(self.protocol_info['num_experiments']):
            for this_sub_idx in range(self.protocol_info['num_sub_per_exp'][exp_idx]):

                for II in range(self.gt_df.shape[0]):
                    if self.gt_df.iloc[II]["data_type"] == "constant":
                        if self.gt_df.iloc[II]["experiment_idx"] == exp_idx and \
                            self.gt_df.iloc[II]["subexperiment_idx"] == this_sub_idx:
                            const_map[exp_idx][this_sub_idx].append(self.gt_df.iloc[II]["weight"])
                        else:
                            # if the data point is not in assigned to this experiment/subexperiment, 
                            # set the weight mapping to 0, so it doesn't influence the cost in this 
                            # subexperiment
                            const_map[exp_idx][this_sub_idx].append(0.0)
                    if self.gt_df.iloc[II]["data_type"] == "series":
                        if self.gt_df.iloc[II]["experiment_idx"] == exp_idx and \
                            self.gt_df.iloc[II]["subexperiment_idx"] == this_sub_idx:
                            series_map[exp_idx][this_sub_idx].append(self.gt_df.iloc[II]["weight"])
                        else:
                            series_map[exp_idx][this_sub_idx].append(0.0)

                    if self.gt_df.iloc[II]["data_type"] == "frequency":
                        if self.gt_df.iloc[II]["experiment_idx"] == exp_idx and \
                            self.gt_df.iloc[II]["subexperiment_idx"] == this_sub_idx:
                            amp_map[exp_idx][this_sub_idx].append(self.gt_df.iloc[II]["weight"])
                            if "phase_weight" not in self.gt_df.iloc[II].keys():
                                # if there is no phase weight, weight it the same as the amplitude
                                phase_map[exp_idx][this_sub_idx].append(self.gt_df.iloc[II]["weight"])
                            else:
                                phase_map[exp_idx][this_sub_idx].append(self.gt_df.iloc[II]["phase_weight"])
                        else:
                            amp_map[exp_idx][this_sub_idx].append(0.0)
                            phase_map[exp_idx][this_sub_idx].append(0.0)

                    if self.gt_df.iloc[II]["data_type"] == "prob_dist":
                        if self.gt_df.iloc[II]["experiment_idx"] == exp_idx and \
                            self.gt_df.iloc[II]["subexperiment_idx"] == this_sub_idx:
                            prob_dist_map[exp_idx][this_sub_idx].append(self.gt_df.iloc[II]["weight"])
                        else:
                            prob_dist_map[exp_idx][this_sub_idx].append(0.0)

                # make each weight vector a numpy array
                const_map[exp_idx][this_sub_idx] = np.array(const_map[exp_idx][this_sub_idx])
                series_map[exp_idx][this_sub_idx] = np.array(series_map[exp_idx][this_sub_idx])
                amp_map[exp_idx][this_sub_idx] = np.array(amp_map[exp_idx][this_sub_idx])
                phase_map[exp_idx][this_sub_idx] = np.array(phase_map[exp_idx][this_sub_idx])
                prob_dist_map[exp_idx][this_sub_idx] = np.array(prob_dist_map[exp_idx][this_sub_idx])

        self.protocol_info["scaled_weight_const_from_exp_sub"] = const_map
        self.protocol_info["scaled_weight_series_from_exp_sub"] = series_map
        self.protocol_info["scaled_weight_amp_from_exp_sub"] = amp_map
        self.protocol_info["scaled_weight_phase_from_exp_sub"] = phase_map
        self.protocol_info["scaled_weight_prob_dist_from_exp_sub"] = prob_dist_map
        return

    def __set_and_save_param_names(self, idxs_to_ignore=None):
        # This should also be a function under parsers.

        # Each entry in param_names is a name or list of names that gets modified by one parameter
        self.param_id_info = {}
        if self.params_for_id_path:
            csv_parser = CSVFileParser()
            input_params = csv_parser.get_data_as_dataframe_multistrings(self.params_for_id_path)
            self.param_id_info["param_names"] = []
            param_names_for_gen = []
            for II in range(input_params.shape[0]):
                if idxs_to_ignore is not None:
                    if II in idxs_to_ignore:
                        continue
                self.param_id_info["param_names"].append([input_params["vessel_name"][II][JJ] + '/' +
                                               input_params["param_name"][II]for JJ in
                                               range(len(input_params["vessel_name"][II]))])

                if input_params["vessel_name"][II][0] == 'global':
                    param_names_for_gen.append([input_params["param_name"][II]])

                else:
                    param_names_for_gen.append([input_params["param_name"][II] + '_' +
                                                input_params["vessel_name"][II][JJ] # re.sub('_T$', '', input_params["vessel_name"][II][JJ])
                                                for JJ in range(len(input_params["vessel_name"][II]))])

            # set param ranges from file and strings for plotting parameter names
            if idxs_to_ignore is not None:
                self.param_id_info["param_mins"] = np.array([float(input_params["min"][JJ]) for JJ in range(input_params.shape[0])
                                            if JJ not in idxs_to_ignore])
                self.param_id_info["param_maxs"] = np.array([float(input_params["max"][JJ]) for JJ in range(input_params.shape[0])
                                            if JJ not in idxs_to_ignore])
                if "name_for_plotting" in input_params.columns:
                    self.param_id_info["param_names_for_plotting"] = np.array([input_params["name_for_plotting"][JJ]
                                                            for JJ in range(input_params.shape[0])
                                                            if JJ not in idxs_to_ignore])
                else:
                    self.param_id_info["param_names_for_plotting"] = np.array([self.param_id_info["param_names"][JJ][0]
                                                            for JJ in range(len(self.param_id_info["param_names"]))
                                                            if JJ not in idxs_to_ignore])
            else:
                self.param_id_info["param_mins"] = np.array([float(input_params["min"][JJ]) for JJ in range(input_params.shape[0])])
                self.param_id_info["param_maxs"] = np.array([float(input_params["max"][JJ]) for JJ in range(input_params.shape[0])])
                if "name_for_plotting" in input_params.columns:
                    self.param_id_info["param_names_for_plotting"] = np.array([input_params["name_for_plotting"][JJ]
                                                            for JJ in range(input_params.shape[0])])
                else:
                    self.param_id_info["param_names_for_plotting"] = np.array([param_name[0] for param_name in self.param_id_info["param_names"]])
                # check that max is greater than min
                for JJ in range(len(self.param_id_info["param_mins"])):
                    if self.param_id_info["param_maxs"][JJ] <= self.param_id_info["param_mins"][JJ]:
                        raise ValueError(f"Parameter {self.param_id_info['param_names'][JJ]} has max <= min")
            

            # set param_priors
            if "prior" in input_params.columns:
                self.param_id_info["param_prior_types"] = np.array([input_params["prior"][JJ] for JJ in range(input_params.shape[0])])
            else:
                self.param_id_info["param_prior_types"] = np.array(["uniform" for JJ in range(input_params.shape[0])])


        else:
            print(f'params_for_id_path cannot be None, exiting')

        if self.rank == 0:
            with open(os.path.join(self.output_dir, 'param_names.csv'), 'w') as f:
                wr = csv.writer(f)
                wr.writerows(self.param_id_info["param_names"])
            with open(os.path.join(self.output_dir, 'param_names_for_gen.csv'), 'w') as f:
                wr = csv.writer(f)
                wr.writerows(param_names_for_gen)
            # save param_names_for_plotting to csv
            with open(os.path.join(self.output_dir, 'param_names_for_plotting.csv'), 'w') as f:
                for name in self.param_id_info["param_names_for_plotting"]:
                    f.write(name + '\n')
        return

    def __get_ground_truth_values(self):

        # _______ First we access data for constant values

        # TODO make all of the below lists instead of arrays? So we can have different sized entries.

        ground_truth_const = np.array([self.gt_df.iloc[II]["value"] for II in range(self.gt_df.shape[0])
                                        if self.gt_df.iloc[II]["data_type"] == "constant"])

        # _______ Then for time series
        ground_truth_series = [np.array(self.gt_df.iloc[II]["value"]) for II in range(self.gt_df.shape[0])
                                        if self.gt_df.iloc[II]["data_type"] == "series"]

        # _______ Then for frequency series
        ground_truth_amp = np.array([self.gt_df.iloc[II]["value"] for II in range(self.gt_df.shape[0])
                                        if self.gt_df.iloc[II]["data_type"] == "frequency"])

        # then for ground truth probability distributions
        ground_truth_prob_dist_params = np.array([self.gt_df.iloc[II]["prob_dist_params"] for II in range(self.gt_df.shape[0])
                                            if self.gt_df.iloc[II]["data_type"] == "prob_dist"])


        # _______ and the phase of the freq data
        ground_truth_phase_list = []
        for II in range(self.gt_df.shape[0]):
            if self.gt_df.iloc[II]["data_type"] == "frequency":
                if "phase" not in self.gt_df.iloc[II].keys():
                    ground_truth_phase_list.append(None)
                else:
                    ground_truth_phase_list.append(self.gt_df.iloc[II]["phase"])
        ground_truth_phase = np.array(ground_truth_phase_list)

        # get the dt for the series data
        dt_list = []
        for II in range(self.gt_df.shape[0]):
            if self.gt_df.iloc[II]["data_type"] == "series":
                if "obs_dt" not in self.gt_df.iloc[II].keys():
                    print("dt not found in obs_data.json for series data, exiting")
                    exit()
                dt_list.append(self.gt_df.iloc[II]["obs_dt"])
        
        self.obs_info["obs_dt"] = np.array(dt_list)
        
        if len(self.obs_info["obs_dt"]) > 0:
            if min(self.obs_info["obs_dt"]) < self.dt:
                print("one of the dt in obs_data.json is less than the dt in user_inputs.yaml, the output timestep"
                    "defined in user_inputs.yaml must be less than the smallest dt for your data. Exiting")
                exit()

        # The std for the different observables
        self.obs_info["std_const_vec"] = np.array([self.gt_df.iloc[II]["std"] for II in range(self.gt_df.shape[0])
                                       if self.gt_df.iloc[II]["data_type"] == "constant"])

        self.obs_info["std_series_vec"] = [np.array(self.gt_df.iloc[II]["std"]) for II in range(self.gt_df.shape[0])
                                        if self.gt_df.iloc[II]["data_type"] == "series"]

        self.obs_info["std_amp_vec"] = np.array([self.gt_df.iloc[II]["std"] for II in range(self.gt_df.shape[0])
                                        if self.gt_df.iloc[II]["data_type"] == "frequency"])

        # if len(ground_truth_series) > 0:
            # TODO what if we have ground truths of different size or sample rate?
            # ground_truth_series = np.stack(ground_truth_series)
            # removed because we have data of different sizes

        if len(ground_truth_amp) > 0:
            ground_truth_amp = np.stack(ground_truth_amp)

        if len(ground_truth_phase) > 0:
            ground_truth_phase = np.stack(ground_truth_phase)

        if self.rank == 0:
            np.save(os.path.join(self.output_dir, 'ground_truth_const.npy'), ground_truth_const)
            if len(ground_truth_series) > 0:
                np.save(os.path.join(self.output_dir, 'ground_truth_series.npy'), 
                        np.array(ground_truth_series, dtype=object), allow_pickle=True)
            if len(ground_truth_amp) > 0:
                np.save(os.path.join(self.output_dir, 'ground_truth_amp.npy'), ground_truth_amp)
            if len(ground_truth_phase) > 0:
                np.save(os.path.join(self.output_dir, 'ground_truth_phase.npy'), ground_truth_phase)

        self.obs_info["ground_truth_const"] = ground_truth_const
        self.obs_info["ground_truth_prob_dist_params"] = ground_truth_prob_dist_params
        self.obs_info["ground_truth_series"] = ground_truth_series
        self.obs_info["ground_truth_amp"] = ground_truth_amp
        self.obs_info["ground_truth_phase"] = ground_truth_phase

        # create a mapping between const_idx and the obs_idx
        const_count = 0
        series_count = 0
        freq_count = 0
        prob_dist_count = 0
        self.obs_info["const_idx_to_obs_idx"] = []
        self.obs_info["series_idx_to_obs_idx"] = []
        self.obs_info["freq_idx_to_obs_idx"] = []
        self.obs_info["prob_dist_idx_to_obs_idx"] = []
        for obs_idx in range(self.obs_info["num_obs"]):
            if self.obs_info["data_types"][obs_idx] == "constant":
                self.obs_info["const_idx_to_obs_idx"].append(obs_idx)
                const_count += 1
            elif self.obs_info["data_types"][obs_idx] == "series":
                self.obs_info["series_idx_to_obs_idx"].append(obs_idx)
                series_count += 1
            elif self.obs_info["data_types"][obs_idx] == "frequency":
                self.obs_info["freq_idx_to_obs_idx"].append(obs_idx)
                freq_count += 1
            elif self.obs_info["data_types"][obs_idx] == "prob_dist":
                self.obs_info["prob_dist_idx_to_obs_idx"].append(obs_idx)
                prob_dist_count += 1

        return 
    
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

    def __set_prediction_var(self):
        # TODO make the prediction_variables csv obsolete. Should be in obs_data.json
        # TODO when done this function will also be obsolete

        if self.prediction_info is not None:
            # prediction_info has already been parsed from obs_data.json file
            pass
        else:
            self.prediction_info = {}
            pred_var_path = os.path.join(self.resources_dir, f'{self.file_name_prefix}_prediction_variables.csv')
            if os.path.exists(pred_var_path):
                # TODO change this to loading with parser
                csv_parser = CSVFileParser()
                pred_var_df = csv_parser.get_data_as_dataframe_multistrings(pred_var_path)
                self.prediction_info['names'] = [pred_var_df["name"][II].strip()
                                                for II in range(pred_var_df.shape[0])]
                self.prediction_info['units'] = [pred_var_df["unit"][II].strip()
                                                for II in range(pred_var_df.shape[0])]
                self.prediction_info['names_for_plotting'] = np.array([pred_var_df["name_for_plotting"][II].strip()
                                                for II in range(pred_var_df.shape[0])])
            else:
                self.prediction_info['names'] = None
                self.prediction_info['units'] = None
                self.prediction_info['names_for_plotting'] = None

            self.prediction_info['experiment_idxs'] = [0] # only functionality for first experiment 
                                                          # when using (to be obsolete) csv file

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
                 dt=0.01, solver_info=None, 
                 ga_options=None, optimiser_options=None, DEBUG=False, model_type=None):

        self.model_path = model_path
        self.param_id_method = param_id_method
        self.output_dir = None
        self.model_type = model_type

        self.solver_info = solver_info
        self.obs_info = obs_info
        self.param_id_info = param_id_info
        self.prediction_info = prediction_info # currently not used
        self.num_params = len(self.param_id_info["param_names"])

        self.protocol_info = protocol_info
        self.param_norm_obj = Normalise_class(self.param_id_info["param_mins"], self.param_id_info["param_maxs"])

        sfp = scriptFunctionParser()
        self.operation_funcs_dict = sfp.get_operation_funcs_dict()
        self.cost_funcs_dict = sfp.get_cost_funcs_dict()

        # set up opencor simulation
        self.dt = dt
        if self.protocol_info['sim_times'][0][0] is not None:
            self.sim_time = self.protocol_info['sim_times'][0][0]
        else:
            # set temporary sim time, just to initialise the sim_helper
            self.sim_time = 0.001
        if self.protocol_info['pre_times'][0] is not None:
            self.pre_time = self.protocol_info['pre_times'][0]
        else:
            # set temporary pre time, just to initialise the sim_helper
            self.pre_time = 0.001

        self.sim_helper = self.initialise_sim_helper()

        self.sim_helper.update_times(self.dt, 0.0, self.sim_time, self.pre_time)

        self.n_steps = int(self.sim_time/self.dt)

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

        # Merge ga_options into optimiser_options for backwards compatibility
        self.optimiser_options = optimiser_options or {}
        if ga_options is not None:
            for key, value in ga_options.items():
                if key not in self.optimiser_options:
                    self.optimiser_options[key] = value
        
        # Set default options if not provided
        if 'cost_convergence' not in self.optimiser_options:
            self.optimiser_options['cost_convergence'] = 0.0001
        if 'max_patience' not in self.optimiser_options:
            self.optimiser_options['max_patience'] = 10
        if 'num_calls_to_function' not in self.optimiser_options:
            self.optimiser_options['num_calls_to_function'] = 10000
        
        self.cost_type = self.obs_info["cost_type"]
        self.DEBUG = DEBUG

    def initialise_sim_helper(self):
        helper_cls = get_simulation_helper(self.model_type)
        return helper_cls(self.model_path, self.dt, self.sim_time,
                          solver_info=self.solver_info, pre_time=self.pre_time)
    
    def set_best_param_vals(self, best_param_vals):
        self.best_param_vals = best_param_vals
    
    def set_param_names(self, param_names):
        self.param_id_info["param_names"] = param_names
        self.num_params = len(self.param_id_info["param_names"])

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
                ga_options=None,  # Legacy parameter, kept for backwards compatibility
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
                ga_options=None,  # Legacy parameter, kept for backwards compatibility
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
                ga_options=None,  # Legacy parameter, kept for backwards compatibility
                optimiser_options=self.optimiser_options,
                DEBUG=self.DEBUG
            )
            optimiser.run()
            self.best_param_vals = optimiser.best_param_vals
            self.best_cost = optimiser.best_cost



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

        return

    def get_cost_obs_and_pred_from_params(self, param_vals, reset=True, 
                                          only_one_exp=-1, pred_names=None):

        pred_outputs_list = []
        if self.protocol_info["num_sub_total"] == 1:
            # do normal cost calculation
            # TODO technically this if chunk isn't needed, as the below works for general experiment numbers
            # TODO but I have left it because it is much simpler and easier to understand
            self.sim_helper.set_param_vals(self.param_id_info["param_names"], param_vals)
            self.sim_helper.reset_states() # this needs to be done to make sure states defined by a constant are set
            success = self.sim_helper.run()
            if success:
                operands_outputs = self.sim_helper.get_results(self.obs_info["operands"])

                cost = self.get_cost_from_operands(operands_outputs)

                operands_outputs_list = [operands_outputs]
                if pred_names is not None:
                    pred_outputs = self.sim_helper.get_results(pred_names)
                    pred_outputs_list.append(pred_outputs)
                # reset params
                if reset:
                    self.sim_helper.reset_and_clear()

            else:
                # simulation set cost to large,
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print('simulation failed with params...')
                    print(param_vals)
                return np.inf, [], []
        else:
            # loop through subexperiments
            if only_one_exp == -1:
                # unless the user wants to just to one experiment, reset must be true
                reset = True
                exp_idxs_to_run = range(self.protocol_info["num_experiments"])
            else:
                exp_idxs_to_run = [only_one_exp]
                
            operands_outputs_list = []
            for exp_idx in range(self.protocol_info["num_experiments"]):
                # set param vals for this iteration of param_id
                self.sim_helper.set_param_vals(self.param_id_info["param_names"], param_vals)
                self.sim_helper.reset_states() # this needs to be done to make sure states defined by a constant are set
                current_time = 0
                for this_sub_idx in range(self.protocol_info["num_sub_per_exp"][exp_idx]):
                    if exp_idx not in exp_idxs_to_run:
                        operands_outputs_list.append(None)
                        continue
                    subexp_count = int(np.sum([num_sub for num_sub in 
                                               self.protocol_info["num_sub_per_exp"][:exp_idx]]) + this_sub_idx)
            
                    self.sim_time = self.protocol_info["sim_times"][exp_idx][this_sub_idx]
                    self.pre_time = self.protocol_info["pre_times"][exp_idx]
                    if self.protocol_info["num_sub_total"] > 1:
                        # resize the experiment and change parameters for this subexperiment
                        if this_sub_idx == 0:
                            # we need a presim here 
                            self.sim_helper.update_times(self.dt, 0.0, self.sim_time, self.pre_time)
                            current_time += self.pre_time
                        else:
                            self.sim_helper.update_times(self.dt, current_time, 
                                                        self.sim_time, 0.0)
                    # change parameters
                    self.sim_helper.set_param_vals(list(self.protocol_info["params_to_change"].keys()), 
                                            [self.protocol_info["params_to_change"][param_name][exp_idx][this_sub_idx] for \
                                                param_name in self.protocol_info["params_to_change"].keys()])

                    success = self.sim_helper.run()
                    current_time += self.sim_time
                    if success:
                        # TODO currently we calculate the outputs for all subexperiments, which is inefficient
                        # TODO we could calculate the outputs for each subexperiment only when needed for the cost
                        # TODO Fine for now, simulation time is much greater than cost calculation, so no big issue yet.
                        operands_outputs = self.sim_helper.get_results(self.obs_info["operands"])

                        operands_outputs_list.append(operands_outputs)

                        if pred_names is not None:
                            pred_outputs = self.sim_helper.get_results(pred_names)
                            pred_outputs_list.append(pred_outputs)
                        
                        # reset params
                        if reset and this_sub_idx == self.protocol_info["num_sub_per_exp"][exp_idx] - 1:
                            # reset if we are at the end of this experiment
                            self.sim_helper.reset_and_clear()

                    else:
                        # simulation set cost to large,
                        print('simulation failed with params...')
                        print(param_vals)
                        print('failed on experiment idx = {} subexperiment idx = {}'.format(exp_idx, this_sub_idx))
                        return np.inf, [], []


            cost = 0
            for exp_idx in exp_idxs_to_run:
                if exp_idx not in exp_idxs_to_run:
                    continue
                for this_sub_idx in range(self.protocol_info["num_sub_per_exp"][exp_idx]):
                    subexp_count = int(np.sum([num_sub for num_sub in 
                                               self.protocol_info["num_sub_per_exp"][:exp_idx]]) + this_sub_idx)

                    sub_cost = self.get_cost_from_operands(operands_outputs_list[subexp_count], 
                                                               exp_idx=exp_idx, sub_idx=this_sub_idx)   
                    cost += sub_cost
            
            # average cost over all subexperiments so that it is comparable between diff number of subexperiments
            cost = cost/self.protocol_info["num_sub_total"] 

        return cost, operands_outputs_list, pred_outputs_list

    def get_cost_and_obs_from_params(self, param_vals, reset=True, only_one_exp=-1):
        cost, obs, _ = self.get_cost_obs_and_pred_from_params(param_vals, reset=reset, only_one_exp=only_one_exp)
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
        lnlikelihood = -0.5*cost # TODO check this is correct for all multimodal distributions

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

        obs_dict = self.get_obs_output_dict(operands_outputs)
        # calculate error between the observables of this set of parameters
        # and the ground truth
        
        cost = self.cost_calc(obs_dict, exp_idx=exp_idx, sub_idx=sub_idx)

        return cost

    def cost_calc(self, obs_dict, exp_idx=0, sub_idx=0):
        

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
            

        cost = (cost + series_cost + amp_cost + phase_cost + prob_dist_cost) / num_weighted_obs

        

        return cost

    def get_obs_output_dict(self, operands_outputs, get_all_series=False):
        if operands_outputs == None:
            return None
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
                    obs_series_array_all[JJ] = self.operation_funcs_dict[
                            self.obs_info["operations"][JJ]](*operands_outputs[JJ], series_output=True, **self.obs_info["operation_kwargs"][JJ]) 
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
                    obs = self.operation_funcs_dict[self.obs_info["operations"][JJ]](*operands_outputs[JJ], **self.obs_info["operation_kwargs"][JJ]) 
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

    def simulate_once(self, param_vals=None, reset=True, only_one_exp=-1):
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
        obs_dicts = [self.get_obs_output_dict(obs_item) for obs_item in obs]

        if only_one_exp != -1:
            # only print out results if doing all experiments, otherwise cost will be strange
            return None, None

        best_cost = np.load(os.path.join(self.output_dir, 'best_cost.npy'))
        print(f'cost should be {best_cost}')
        print('cost check after single simulation is {}'.format(cost_check))

        print(f'final obs values :')
        for idx, obs_dict in enumerate(obs_dicts):
            print(f'subexperiment {idx+1}:')
            # TODO make the printing of the obs_dict more informative
            print(obs_dict['const'])

    def set_genetic_algorithm_parameters(self, n_calls):
        if not self.param_id_method == 'genetic_algorithm':
            print('param_id is not set up as a genetic algorithm')
            exit()
        self.optimiser_options['num_calls_to_function'] = n_calls
        # TODO add more of the gen alg constants here so they can be changed by user.

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
                 obs_info, param_id_info, protocol_info, prediction_info,
                 dt=0.01, solver_info=None, mcmc_options=None, DEBUG=False, model_type=None):
        super().__init__(model_path, "MCMC",
                obs_info, param_id_info, protocol_info, prediction_info,
                dt=dt, solver_info=solver_info, DEBUG=DEBUG, model_type=model_type)

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
                 dt=0.01, mcmc_options=None, ga_options=None,
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
        
        self.param_id_obs_file_prefix = re.sub('\.json', '', os.path.split(param_id_obs_path)[1])
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

