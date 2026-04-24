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
paperPlotSetup.Setup_Plot(3)
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
from parsers.PrimitiveParsers import YamlFileParser
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
    def __init__(self, model_path, model_type, file_name_prefix, sa_options, DEBUG=False,
                 param_id_output_dir=None, resources_dir=None, model_out_names=[], 
                 solver_info={}, dt=0.01, optimiser_options={}, param_id_obs_path=None, params_for_id_path=None):

        self.model_path = model_path
        self.model_type = model_type
        self.file_name_prefix = file_name_prefix
        self.DEBUG = DEBUG
        self.param_id_output_dir = param_id_output_dir
        self.resources_dir = resources_dir
        self.model_out_names = model_out_names
        self.solver_info = solver_info
        self.dt = dt
        self.optimiser_options = optimiser_options
        self.param_id_obs_path = param_id_obs_path
        self.params_for_id_path = params_for_id_path
        self.sa_options = sa_options
        sa_output_dir = sa_options['output_dir']
        
        self.SA_manager = sobol_SA(self.model_path, self.model_out_names, self.solver_info, sa_options, self.dt, 
                            sa_output_dir, param_id_path=self.param_id_obs_path, params_for_id_path=self.params_for_id_path,
                            verbose=False, use_MPI=True)

    @classmethod
    def init_from_dict(cls, inp_data_dict):
        # parse the user inputs dictionary
        yaml_parser = YamlFileParser()
        inp_data_dict = yaml_parser.parse_user_inputs_file(inp_data_dict)
        # Only pass kwargs that exist in inp_data_dict
        arg_options = [
            'model_path', 'model_type', 'file_name_prefix', 'sa_options', 'DEBUG', 'param_id_output_dir',
            'resources_dir', 'model_out_names', 'solver_info',
            'dt', 'optimiser_options', 'param_id_obs_path', 'params_for_id_path'
        ]
        kwargs = {key: inp_data_dict[key] for key in arg_options if key in inp_data_dict}

        # Support common naming used elsewhere
        if 'file_name_prefix' not in kwargs and 'file_prefix' in inp_data_dict:
            kwargs['file_name_prefix'] = inp_data_dict['file_prefix']

        return cls(**kwargs)

    def init_from_all_dicts(cls, inp_data_dict, obs_data_dict, params_for_id_dict, sa_options):
        sa = cls.init_from_dict(inp_data_dict)
        sa.set_ground_truth_data(obs_data_dict)
        sa.set_params_for_id(params_for_id_dict)
        sa.set_sa_options(sa_options)
        return sa

    def add_user_operation_func(self, func):
        self.SA_manager.add_user_operation_func(func)

    def set_sa_options(self, sa_options):
        self.SA_manager.set_sa_options(sa_options)

    def set_ground_truth_data(self, obs_data_dict):
        self.SA_manager.set_ground_truth_data(obs_data_dict)
        
    def set_params_for_id(self, params_for_id_dict):
        self.SA_manager.set_params_for_id(params_for_id_dict)
    
    def set_model_out_names(self, obs_data_dict):
        # TODO fix for arbitrary number of operands
        # mohammad must have done this already.
        self.model_out_names = []
        for item in obs_data_dict["data_items"]:
            if len(item["operands"]) > 1:
                print(f'{RED}ERROR: more than one operand for {item["name_for_plotting"]}, not supported{RESET}')
                exit()
            self.model_out_names.append(item["operands"][0])

    def run_sensitivity_analysis(self, sa_options=None):
        if sa_options is None:
            sa_options = self.sa_options
        else:
            self.set_sa_options(sa_options)

        if sa_options['method'] == 'naive':
            self.run_naive_sensitivity()
        elif sa_options['method'] == 'sobol':
            self.run_sobol_sensitivity(sa_options)
        else:
            print('ERROR: sensitivity analysis method not recognised')
            exit()

    def run_sobol_sensitivity(self, sa_options=None):

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        output_dir = self.SA_manager.output_dir

        self.SA_manager.set_sa_options(sa_options)

        if self.SA_manager.gt_df is None or self.SA_manager.param_id_info is None:
            print(f'{RED}ERROR: need to set ground truth data and params for id before running sobol sensitivity analysis{RESET}')
            exit()

        S1_all, ST_all, S2_all = self.SA_manager.run()

        if rank == 0:
            print(f"{GREEN}Sensitivity analysis completed successfully :){RESET}")
            print(f'{CYAN}saving results in {output_dir}{RESET}')
            self.SA_manager.save_sobol_indices(S1_all, ST_all, S2_all)
            self.SA_manager.plot_sobol_first_order_idx(S1_all, ST_all)
            self.SA_manager.plot_sobol_S2_idx(S2_all)
            self.SA_manager.plot_sobol_heatmap(S1_all, ST_all)

    
    def choose_most_impactful_params_sobol(self, top_n=5, index_type='ST', criterion='max', threshold=0.01, use_threshold=False):
        """
        Ranks and returns parameters based on Sobol indices.
        
        Args:
            top_n (int): Max number of parameters to return.
            index_type (str): 'ST' or 'S1'.
            criterion (str or func): 'max', 'mean', or custom lambda.
            threshold (float): Minimum score required. Only applied if use_threshold=True.
            use_threshold (bool): Whether to reject parameters below the threshold.
        """
        comm = MPI.COMM_WORLD
        if comm.Get_rank() != 0:
            return None

        indices_dict = self.SA_manager.load_sobol_indices()
        if not indices_dict or index_type.upper() not in indices_dict:
            print(f"{RED}ERROR: Index type '{index_type}' not found.{RESET}")
            return []

        data = indices_dict[index_type.upper()]
        
        # Flatten structure: {param_name: [val_out1, val_out2, ...]}
        param_scores_list = {}
        for out_name, params in data.items():
            for p_name, val in params.items():
                if p_name not in param_scores_list:
                    param_scores_list[p_name] = []
                param_scores_list[p_name].append(val)

        # Mapping criterion to calculation
        if criterion == 'max':
            calc_func = max
        elif criterion == 'mean':
            calc_func = lambda x: sum(x) / len(x)
        elif callable(criterion):
            calc_func = criterion
        else:
            print(f"{RED}ERROR: Invalid criterion '{criterion}'{RESET}")
            return []

        # 1. Calculate scores
        processed_scores = {p: calc_func(vals) for p, vals in param_scores_list.items()}

        # 2. Filter only if requested
        if use_threshold:
            filtered_data = {p: s for p, s in processed_scores.items() if s >= threshold}
            status_msg = f"filtered by threshold >= {threshold}"
        else:
            filtered_data = processed_scores
            status_msg = "unfiltered"

        # 3. Sort and select
        sorted_items = sorted(filtered_data.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:top_n]
        top_params = [item[0] for item in top_items]

        # 4. Final output
        if not top_params:
            print(f"{RED}No parameters found for criterion '{criterion}' ({status_msg}).{RESET}")
            return []

        print(f"{GREEN}Selected {len(top_params)} parameters (Criteria: {criterion}, Mode: {status_msg}):{RESET}")
        for i, (p, score) in enumerate(top_items):
            print(f"  {i+1}. {p:<35} | Score: {score:.4f}")
            
        return top_params

