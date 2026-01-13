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
import opencor as oc
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import paperPlotSetup
import diagnostics
import utility_funcs
import traceback
from utility_funcs import Normalise_class
paperPlotSetup.Setup_Plot(3)
from opencor_helper import SimulationHelper
from parsers.PrimitiveParsers import scriptFunctionParser
from mpi4py import MPI
import re
from numpy import genfromtxt
from importlib import import_module
import csv
from datetime import date
# from skopt import gp_minimize, Optimizer
from parsers.PrimitiveParsers import CSVFileParser
import pandas as pd
import json
import math
# from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib/..*" )
from sensitivity_analysis.sobolSA import sobol_SA

GREEN = '\033[92m'
CYAN = '\033[36m'
RED = '\033[31m'
# ANSI escape code to reset the color back to the terminal's default
RESET = '\033[0m'

class SensitivityAnalysis():
    """
    Class for doing sensitivity analysis on a 0D model
    """
    def __init__(self, model_path, model_type, file_name_prefix, DEBUG=False,
                 param_id_output_dir=None, resources_dir=None, model_out_names=[], 
                 solver_info={}, dt=0.01, ga_options={}, param_id_obs_path=None, params_for_id_path=None):

        self.model_path = model_path
        self.model_type = model_type
        self.file_name_prefix = file_name_prefix
        self.DEBUG = DEBUG
        self.param_id_output_dir = param_id_output_dir
        self.resources_dir = resources_dir
        self.model_out_names = model_out_names
        self.solver_info = solver_info
        self.dt = dt
        # For backwards compatibility, accept both ga_options and optimiser_options
        # The parser will merge ga_options into optimiser_options, but we keep ga_options for now
        self.ga_options = ga_options
        self.param_id_obs_path = param_id_obs_path
        self.params_for_id_path = params_for_id_path

    def run_sensitivity_analysis(self, sa_options):
        if sa_options['method'] == 'naive':
            self.run_naive_sensitivity()
        elif sa_options['method'] == 'sobol':
            self.run_sobol_sensitivity(sa_options)
        else:
            print('ERROR: sensitivity analysis method not recognised')
            exit()

    def run_sobol_sensitivity(self, sa_options):
        # TODO create param_id object so we can use the cost functions?
        # TODO

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        num_samples = sa_options['num_samples']
        sample_type = sa_options['sample_type']
        output_dir = sa_options['output_dir']

        SA_cfg = {
            "sample_type" : sample_type,
            "num_samples": num_samples,
        }

        if self.param_id_obs_path is None or self.params_for_id_path is None:
            print(f'{RED}ERROR: need to provide param_id_obs_path and params_for_id_path for sobol sensitivity analysis{RESET}')
            exit()

        SA_manager = sobol_SA(self.model_path, self.model_out_names, self.solver_info, SA_cfg, self.dt, 
                            output_dir, param_id_path=self.param_id_obs_path, params_for_id_path=self.params_for_id_path,
                            verbose=False, use_MPI=True, ga_options=self.ga_options)
        S1_all, ST_all, S2_all = SA_manager.run()

        if rank == 0:
            print(f"{GREEN}Sensitivity analysis completed successfully :){RESET}")
            print(f'{CYAN}saving results in {output_dir}{RESET}')
            SA_manager.plot_sobol_first_order_idx(S1_all, ST_all)
            SA_manager.plot_sobol_S2_idx(S2_all)

    
    def run_naive_sensitivity(self, param_id_output_paths):
        # TODO this was the original method, which should be made obsolete.
        # TODO haven't tested this yet.
        # TODO change this function to plot_sensitivity
        if self.rank !=0:
            return

        print('running sensitivity analysis')
        if param_id_output_paths == None:
            sample_path_list = [self.output_dir]
        else:
            sample_path_list = []
            sample_paths = pd.read_csv(param_id_output_paths)
            for i in range(len(sample_paths)):
                sample_path_list.append(sample_paths.iat[i,0])
            if len(sample_path_list) < 1:
                sample_path_list = [self.output_dir]

        do_triples_and_quads = False

        for i in range(len(sample_path_list)):
            self.param_id.run_single_sensitivity(sample_path_list[i], do_triples_and_quads)


        # FA: Eventually we will automate the multiple runs of param_id and store the outputs in idxed
        # FA: directories that can all be accessed automatically by this function without defining input paths.
        m3_to_cm3 = 1e6
        Pa_to_kPa = 1e-3
        number_samples = len(sample_path_list)
        x_values = []
        for i in range(len(self.param_id_info["param_names"])):
            x_values.append(self.param_id_info["param_names"][i][0])

        for i in range(number_samples):
            sample_path = sample_path_list[i]
            normalised_jacobian = np.load(os.path.join(sample_path, 'normalised_jacobian_matrix.npy'))
            parameter_importance = np.load(os.path.join(sample_path, 'parameter_importance.npy'))
            collinearity_idx_i = np.load(os.path.join(sample_path, 'collinearity_idx.npy'))
            if i == 0:
                normalised_jacobian_average = np.zeros(normalised_jacobian.shape)
                parameter_importance_average = np.zeros(parameter_importance.shape)
                collinearity_idx_average = np.zeros(collinearity_idx_i.shape)
            normalised_jacobian_average = normalised_jacobian_average + normalised_jacobian/number_samples
            parameter_importance_average = parameter_importance_average + parameter_importance/number_samples
            collinearity_idx_average = collinearity_idx_average + collinearity_idx_i/number_samples
            self.collinearity_idx = collinearity_idx_average

        collinearity_max = collinearity_idx_average.max()

        if do_triples_and_quads:
            #find maximum average value of collinearity triples
            for i in range(len(x_values)):
                for j in range(number_samples):
                    sample_path = sample_path_list[j]
                    collinearity_idx_triple = np.load(os.path.join(sample_path, 'collinearity_triples' + str(i) + '.npy'))
                    if j==0:
                        collinearity_idx_triple_average = np.zeros(collinearity_idx_triple.shape)
                    collinearity_idx_triple_average = collinearity_idx_triple_average + collinearity_idx_triple/number_samples
                if collinearity_idx_triple_average.max() > collinearity_max:
                    collinearity_max = collinearity_idx_triple_average.max()

            #find maximum value of collinearity quads
            for i in range(len(x_values)):
                for j in range(len(x_values)):
                    for k in range(number_samples):
                        sample_path = sample_path_list[k]
                        collinearity_idx_quad = np.load(
                            os.path.join(sample_path, 'collinearity_quads' + str(i) + '_' + str(j) + '.npy'))
                        if k==0:
                            collinearity_idx_quad_average = np.zeros(collinearity_idx_quad.shape)
                        collinearity_idx_quad_average = collinearity_idx_quad_average + collinearity_idx_quad / number_samples
                    if collinearity_idx_quad_average.max() > collinearity_max:
                        collinearity_max = collinearity_idx_quad_average.max()

        #find maximum average value and average values for collinearity idx
        for i in range(number_samples):
            sample_path = sample_path_list[i]
            collinearity_idx_pairs_i = np.load(
                os.path.join(sample_path, 'collinearity_pairs.npy'))
            if i==0:
                collinearity_idx_pairs_average = np.zeros(collinearity_idx_pairs_i.shape)
            collinearity_idx_pairs_average = collinearity_idx_pairs_average + collinearity_idx_pairs_i/number_samples
            self.collinearity_idx_pairs = collinearity_idx_pairs_average

        if collinearity_idx_pairs_average.max() > collinearity_max:
            collinearity_max = collinearity_idx_pairs_average.max()

        number_Params = len(normalised_jacobian_average)
        #plot jacobian
        subplot_height = 1
        subplot_width = 1
        plt.rc('xtick', labelsize=3)
        fig, axs = plt.subplots(subplot_height, subplot_width)

        x_labels = []
        subset = []
        x_idx = 0
        for obs_idx in range(len(self.obs_info["obs_names"])):
            # if self.obs_info["data_types"][obs_idx] != "series":
            x_labels.append(self.obs_info["obs_names"][obs_idx] + " " + self.obs_info["operations"][obs_idx])
            subset.append(x_idx)
            x_idx = x_idx + 1

        for param_idx in range(len(self.param_id_info["param_names"])):
            y_values = []
            for obs_idx in range(len(x_labels)):
                y_values.append(abs((normalised_jacobian_average[param_idx][subset[obs_idx]])))
            axs.plot(x_labels, y_values, label=f'${self.param_id_info["param_names_for_plotting"][param_idx][0]}$')
        axs.set_yscale('log')
        axs.legend(loc='lower left', fontsize=6)
        plt.savefig(os.path.join(self.plot_dir,
                                     f'reconstruct_{self.file_name_prefix}_'
                                     f'{self.param_id_obs_file_prefix}_sensitivity_average.eps'))
        plt.savefig(os.path.join(self.plot_dir,
                                     f'reconstruct_{self.file_name_prefix}_'
                                     f'{self.param_id_obs_file_prefix}_sensitivity_average.pdf'))
        plt.close()
        #plot parameter importance
        plt.rc('xtick', labelsize=6)
        plt.rc('ytick', labelsize=12)
        figB, axsB = plt.subplots(1, 1)


        axsB.bar(x_values, parameter_importance_average)
        axsB.set_ylabel("Parameter Importance", fontsize=12)
        plt.savefig(os.path.join(self.plot_dir,
                                     f'reconstruct_{self.file_name_prefix}_'
                                     f'{self.param_id_obs_file_prefix}_parameter_importance_average.eps'))
        plt.savefig(os.path.join(self.plot_dir,
                                     f'reconstruct_{self.file_name_prefix}_'
                                     f'{self.param_id_obs_file_prefix}_parameter_importance_average.pdf'))
        plt.close()
        #plot collinearity idx average
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=4)
        figC, axsC = plt.subplots(1, 1)
        x_values_cumulative = []
        x_values_temp = x_values[0]
        for i in range(len(x_values)):
            x_values_cumulative.append(x_values_temp)
            if (i + 1) < len(x_values):
                x_values_temp = x_values[i + 1] + "\n" + x_values_temp
        axsC.barh(x_values_cumulative, collinearity_idx_average)
        plt.savefig(os.path.join(self.plot_dir,
                                     f'reconstruct_{self.file_name_prefix}_'
                                     f'{self.param_id_obs_file_prefix}_collinearity_index_average.eps'))
        plt.savefig(os.path.join(self.plot_dir,
                                     f'reconstruct_{self.file_name_prefix}_'
                                     f'{self.param_id_obs_file_prefix}_collinearity_index_average.pdf'))
        plt.close()

        plt.rc('xtick', labelsize=4)
        plt.rc('ytick', labelsize=4)
        figD, axsD = plt.subplots(1, 1)

        X, Y = np.meshgrid(range(len(x_values)), range(len(x_values)))
        if do_triples_and_quads:
            collinearity_levels = np.linspace(0, collinearity_max, 20)
            co = axsD.contourf(X, Y, collinearity_idx_pairs_average, levels=collinearity_levels)
        else:
            co = axsD.contourf(X, Y, collinearity_idx_pairs_average)
        co = fig.colorbar(co, ax=axsD)
        axsD.set_xticks(range(len(x_values)))
        axsD.set_yticks(range(len(x_values)))
        axsD.set_xticklabels(x_values)
        axsD.set_yticklabels(x_values)

        plt.savefig(os.path.join(self.plot_dir,
                                     f'reconstruct_{self.file_name_prefix}_'
                                     f'{self.param_id_obs_file_prefix}_collinearity_pairs_average.eps'))
        plt.savefig(os.path.join(self.plot_dir,
                                     f'reconstruct_{self.file_name_prefix}_'
                                     f'{self.param_id_obs_file_prefix}_collinearity_pairs_average.pdf'))

        if do_triples_and_quads:
            #plot collinearity triples average
            for i in range(len(x_values)):
                for j in range(number_samples):
                    sample_path = sample_path_list[j]
                    collinearity_idx_triple = np.load(os.path.join(sample_path, 'collinearity_triples' + str(i) + '.npy'))
                    if j == 0:
                        collinearity_idx_triple_average = np.zeros(collinearity_idx_triple.shape)
                    collinearity_idx_triple_average = collinearity_idx_triple_average + collinearity_idx_triple/number_samples

                figE, axsE = plt.subplots(1, 1)
                co = axsE.contourf(X, Y, collinearity_idx_triple_average, levels=collinearity_levels)
                co = fig.colorbar(co, ax=axsE)
                axsE.set_xticks(range(len(x_values)))
                axsE.set_yticks(range(len(x_values)))
                axsE.set_xticklabels(x_values)
                axsE.set_yticklabels(x_values)

                plt.savefig(os.path.join(self.plot_dir,
                                             f'reconstruct_{self.file_name_prefix}_'
                                             f'{self.param_id_obs_file_prefix}_collinearity_triples_average' + str(i) + '.eps'))
                plt.savefig(os.path.join(self.plot_dir,
                                             f'reconstruct_{self.file_name_prefix}_'
                                             f'{self.param_id_obs_file_prefix}_collinearity_triples_average' + str(i) + '.pdf'))
                plt.close()
            #plot collinearity quads average
            for i in range(len(x_values)):
                for j in range(len(x_values)):
                    for k in range(number_samples):
                        sample_path = sample_path_list[k]
                        collinearity_idx_quad = np.load(
                            os.path.join(sample_path, 'collinearity_quads' + str(i) + '_' + str(j) + '.npy'))
                        if k==0:
                            collinearity_idx_quad_average = np.zeros(collinearity_idx_quad.shape)
                        collinearity_idx_quad_average = collinearity_idx_quad_average + collinearity_idx_quad / number_samples
                    figE, axsE = plt.subplots(1, 1)
                    co = axsE.contourf(X, Y, collinearity_idx_quad_average, levels=collinearity_levels)
                    co = fig.colorbar(co, ax=axsE)
                    axsE.set_xticks(range(len(x_values)))
                    axsE.set_yticks(range(len(x_values)))
                    axsE.set_xticklabels(x_values)
                    axsE.set_yticklabels(x_values)

                    plt.savefig(os.path.join(self.plot_dir,
                                                 f'reconstruct_{self.file_name_prefix}_'
                                                 f'{self.param_id_obs_file_prefix}collinearity_quads_average' + str(i) + '_' + str(
                                                     j) + '.eps'))
                    plt.savefig(os.path.join(self.plot_dir,
                                                 f'reconstruct_{self.file_name_prefix}_'
                                                 f'{self.param_id_obs_file_prefix}collinearity_quads_average' + str(i) + '_' + str(
                                                     j) + '.pdf'))
                    plt.close()

        self.sensitivity_calculated = True
        print('sensitivity analysis complete')
