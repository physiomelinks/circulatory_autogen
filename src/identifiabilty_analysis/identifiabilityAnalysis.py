'''
@author: Finbar J. Argus
'''

import numpy as np
import os
import sys
from sys import exit
from matplotlib.ticker import FuncFormatter
import corner
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
from utility_funcs import calculate_hessian
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

class IdentifiabilityAnalysis():
    """
    Class for doing identifiability analysis on a 0D model
    """
    def __init__(self, model_path, model_type, file_name_prefix, DEBUG=False,
                 param_id_output_dir=None, resources_dir=None, param_id=None):

        self.model_path = model_path
        self.model_type = model_type
        self.file_name_prefix = file_name_prefix
        self.DEBUG = DEBUG
        self.param_id_output_dir = param_id_output_dir
        self.resources_dir = resources_dir
        self.best_param_vals = None
        self.covariance_matrix_Laplace = None
        self.mean_Lapalace = None
        self.param_id = param_id
        if self.param_id is None:
            # TODO intialise the param_id_object here
            raise ValueError("param_id object must be provided to IdentifiabilityAnalysis")
        

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

    def set_best_param_vals(self, best_param_vals):
        self.param_id.set_best_param_vals(best_param_vals)
        self.best_param_vals = best_param_vals

    def run(self, ia_options):
        """
        Run the identifiability analysis based on the chosen method
        """
        if ia_options['method'] == 'profile_likelihood':
            self.run_profile_likelihood(ia_options)
        elif ia_options['method'] == 'Laplace':
            if self.rank == 0:
                # currently Laplace is not parallelised, so only run on rank 0
                self.run_laplace_approximation(ia_options)
        return

    def run_profile_likelihood(self, ia_options):
        # TODO
        print("Profile Likelihood method not yet implemented")
        exit()
        pass

    def run_laplace_approximation(self, ia_options):

        # TODO fix hessian calculation now that it uses lnlikelihood + lnprior
        Hessian = calculate_hessian(self.param_id)
        
        # Handle singular or near-singular matrices by using pseudo-inverse
        # and adding small regularization if needed
        try:
            covariance_matrix = np.linalg.inv(Hessian)
        except np.linalg.LinAlgError:
            # If matrix is singular, try pseudo-inverse with regularization
            print("Warning: Hessian is singular or near-singular. Using pseudo-inverse with regularization.")
            # Add small regularization term to diagonal
            regularization = 1e-10 * np.eye(Hessian.shape[0])
            try:
                covariance_matrix = np.linalg.inv(Hessian + regularization)
            except np.linalg.LinAlgError:
                # If still singular, use pseudo-inverse
                print("Warning: Regularized Hessian still singular. Using pseudo-inverse.")
                covariance_matrix = np.linalg.pinv(Hessian)
        
        mean = self.best_param_vals
        print("Laplace Approximation Results:")
        print("Mean (Best Parameter Values):", mean)
        print("Covariance Matrix:\n", covariance_matrix)
        self.covariance_matrix_Laplace = covariance_matrix
        self.mean_Lapalace = mean
        parent_dir = os.path.dirname(self.param_id_output_dir)
        np.save(os.path.join(parent_dir, self.file_name_prefix + '_laplace_mean.npy'), self.mean_Lapalace)
        np.save(os.path.join(parent_dir, self.file_name_prefix + '_laplace_covariance.npy'), self.covariance_matrix_Laplace)

    def plot_laplace_results(self, parameter_names, output_dir):
        """
        Plot the results of the Laplace approximation as corner plots.

        Args:
          parameter_names: List of parameter names corresponding to the best_param_vals.
          output_dir: Directory to save the plots.
        """
          

        if self.covariance_matrix_Laplace is None or self.mean_Lapalace is None:
            try:
                parent_dir = os.path.dirname(self.param_id_output_dir)
                self.mean_Lapalace = np.load(os.path.join(parent_dir, self.file_name_prefix + '_laplace_mean.npy'))
                self.covariance_matrix_Laplace = np.load(os.path.join(parent_dir, self.file_name_prefix + '_laplace_covariance.npy'))
                print("Loaded Laplace approximation results from files.")
            except Exception as e:
                print("Error loading Laplace approximation results:", e)
                print("Please run the Laplace approximation before plotting.")
                return

        samples = np.random.multivariate_normal(self.mean_Lapalace, self.covariance_matrix_Laplace, size=100000)
        print(f'samples shape: {samples.shape}')
        figure = corner.corner(samples, labels=parameter_names, truths=self.mean_Lapalace, bins=20, hist_bin_factor=2, smooth=0.5, quantiles=(0.05, 0.5, 0.95))
        plot_path = os.path.join(output_dir, f"{self.file_name_prefix}_laplace_corner_plot.pdf")
        
        axes = figure.get_axes()
        num_params = len(parameter_names)
        # for idx, ax in enumerate(axes):
        #     if idx >= num_params*(num_params - 1):

        #         ax.tick_params(axis='both', rotation=0)
        #         formatterx = matplotlib.ticker.ScalarFormatter()
        #         ax.xaxis.set_major_formatter(formatterx)
        #         ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        #     if idx%num_params == 0:

        #         ax.tick_params(axis='both', rotation=0)
        #         formattery = matplotlib.ticker.ScalarFormatter()
        #         ax.yaxis.set_major_formatter(formattery)
        #         ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    
        # from matplotlib.ticker import FuncFormatter

        # sci_formatter = FuncFormatter(lambda x, _: f"{x:.2e}")

        # for idx, ax in enumerate(axes):
        #     if idx >= num_params * (num_params - 1):
        #         ax.xaxis.set_major_formatter(sci_formatter)
        #     if idx % num_params == 0:
        #         ax.yaxis.set_major_formatter(sci_formatter)

        


        def make_sci_label_formatter(exponent):
            def formatter(val, pos):
                if val == 0:
                    return "0"
                else:
                    # Divide by 10**exponent and format
                    return f"{val / (10 ** exponent):.2f}"
            return FuncFormatter(formatter)

        # We'll store the exponent per axis
        for idx, ax in enumerate(axes):
            if idx >= num_params * (num_params - 1):  # Bottom row → x-axis
                x_min, x_max = ax.get_xlim()
                if x_max == x_min:
                    continue
                # Use log10 of max abs value to determine exponent
                exponent = int(np.floor(np.log10(np.max(np.abs([x_min, x_max])))))
                
                # Apply formatter that divides by 10^exp
                ax.xaxis.set_major_formatter(make_sci_label_formatter(exponent))
                
                # Add ×10^exp label just outside the plot
                ax.text(1.0, 0, fr'$\times 10^{{{exponent}}}$', 
                        transform=ax.transAxes,
                        va='bottom', ha='right',
                        fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            if idx % num_params == 0:  # Left column → y-axis
                y_min, y_max = ax.get_ylim()
                if y_max == y_min:
                    continue
                exponent = int(np.floor(np.log10(np.max(np.abs([y_min, y_max])))))
                
                ax.yaxis.set_major_formatter(make_sci_label_formatter(exponent))
                
                # Add ×10^exp label above the y-axis
                ax.text(0, 1.0, fr'$\times 10^{{{exponent}}}$', 
                        transform=ax.transAxes,
                        va='top', ha='left',
                        rotation=0,
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))


        plt.subplots_adjust(hspace=0.12, wspace=0.1)
        figure.savefig(plot_path)
        print(f"Laplace approximation corner plot saved to {plot_path}")

