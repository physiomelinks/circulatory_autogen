'''
@author: Mohammad H. Shafieizadegan
@ reference: https://salib.readthedocs.io/en/latest/index.html
'''

import json
import os
import sys
from sys import exit
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities'))
import math as math
import opencor as oc
from opencor_helper import SimulationHelper
from SALib.sample import saltelli
import pandas as pd
from SALib.analyze import sobol
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from parsers.PrimitiveParsers import scriptFunctionParser
from mpi4py import MPI
from parsers.PrimitiveParsers import CSVFileParser
import csv
from tqdm import tqdm  # make sure tqdm is installed

class sobol_SA():

    """
        A class for performing sensitivity analysis
        How to use:
        1. Initialize the class with the model path, output names, solver info, sensitivity analysis configuration, protocol info, time step, and save path.
        2. Call the `run` method with a feature extractor function and any additional arguments needed by that function.
        3. The 'run' method will generate sobol indices
        4. You can plot the results using the `plot_sobol_first_order_idx` and `plot_sobol_S2_idx` methods.
    """

    def _is_rank0(self):
        try:
            return MPI.COMM_WORLD.Get_rank() == 0
        except Exception:
            return True

    def _rank0_print(self, *args, **kwargs):
        if self._is_rank0():
            print(*args, **kwargs)

    def __init__(self, model_path, model_out_names, solver_info, SA_cfg, dt, save_path, 
                 param_id_path = None, params_for_id_path=None, use_MPI = False, verbose=False, ga_options=None,
                 sim_time=2.0, pre_time=20.0):

        """
        Initializes the Sensitivity_analysis class.
        Parameters:
            model_path (str): Path to the model file.
            model_out_names (list): Names of the model outputs to be analyzed.
            solver_info (dict): Solver configuration parameters.
            SA_cfg (dict): Configuration for sensitivity analysis, including sample type, number of samples,
                           parameter names, and their bounds.
            protocol_info (dict): Information about the simulation protocol, including simulation times and pre-times.
            dt (float): Time step for the simulation.
            save_path (str): Directory where results will be saved.
            verbose (bool): If True, prints additional information during execution.
        """

        self.model_path = model_path
        self.output_dir = None
        self.verbose = verbose
        self.save_path = save_path

        self.solver_info = solver_info
        self.sample_type = SA_cfg["sample_type"]
        self.num_params = None
        self.model_output_names = model_out_names
        # For backwards compatibility, accept both ga_options and optimiser_options
        # optimiser_options takes precedence
        self.ga_options = ga_options  # Keep for backwards compatibility
        self.optimiser_options = None  # Will be set if passed
        self.protocol_info = None
        self.dt = dt
        
        # set up observables functions
        sfp = scriptFunctionParser()
        self.operation_funcs_dict = sfp.get_operation_funcs_dict()
        self.__set_obs_names_and_df(param_id_path, sim_time=sim_time, pre_time=pre_time)

        # set up opencor simulation
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

        self.set_output_dir(save_path)

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.num_procs = self.comm.Get_size()
        self.use_mpi = use_MPI

        self.params_for_id_path = params_for_id_path
        if self.params_for_id_path:
            self.__set_and_save_param_names()
        self.SA_cfg = self.create_SA_cfg(self.sample_type, SA_cfg["num_samples"])

    def create_SA_cfg(self, sample_type, num_samples):
        
        # Use param_id_info to build SA_cfg dynamically
        if not hasattr(self, "param_id_info") or not self.param_id_info:
            raise ValueError("param_id_info is not set. Please run __set_and_save_param_names() first.")

        SA_cfg = {
            "sample_type": sample_type,
            "param_names": [name[0] if isinstance(name, list) else name for name in self.param_id_info["param_names"]],
            "num_samples": num_samples,
            "param_mins": list(self.param_id_info["param_mins"]),
            "param_maxs": list(self.param_id_info["param_maxs"])
        }

        # if self.verbose:
        #     print("Sensitivity Analysis Configuration:")
        #     print(json.dumps(SA_cfg, indent=4))

        self.num_params = len(SA_cfg["param_names"])

        return SA_cfg

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

            # set param_priors
            if "prior" in input_params.columns:
                self.param_id_info["param_prior_types"] = np.array([input_params["prior"][JJ] for JJ in range(input_params.shape[0])])
            else:
                self.param_id_info["param_prior_types"] = np.array(["uniform" for JJ in range(input_params.shape[0])])


        else:
            self._rank0_print(f'params_for_id_path cannot be None, exiting')

        if self.rank == 0:
            with open(os.path.join(self.output_dir, 'param_names.csv'), 'w') as f:
                wr = csv.writer(f)
                wr.writerows(self.param_id_info["param_names"])
            with open(os.path.join(self.output_dir, 'param_names_for_gen.csv'), 'w') as f:
                wr = csv.writer(f)
                wr.writerows(param_names_for_gen)
        return

    def initialise_sim_helper(self):
        return SimulationHelper(self.model_path, self.dt, self.sim_time,
                                solver_info=self.solver_info, pre_time=self.pre_time)

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
                self._rank0_print("data_items not found in json object. ",
                      "Please check that data_items is the key for the list of data items")
            if 'protocol_info' in json_obj.keys():
                self.protocol_info = json_obj['protocol_info']
                if "sim_times" not in self.protocol_info.keys():
                    self.protocol_info["sim_times"] = [[sim_time]]
                if "pre_times" not in self.protocol_info.keys():
                    self.protocol_info["pre_times"] = [pre_time]
            else:
                if pre_time is None or sim_time is None:
                    self._rank0_print("protocol_info not found in json object. ",
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
                        self._rank0_print('"variable" not found in prediction item in obs_data.json file, ',
                              'exitiing') 
                        exit()
                    if 'unit' in entry.keys():
                        self.prediction_info['units'].append(entry['unit'])
                    else:
                        self._rank0_print('"unit" not found in prediction item in obs_data.json file, ',
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
            self._rank0_print(f"unknown data type for imported json object of {type(json_obj)}")
        
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
                        self._rank0_print('constant data types plot type defaults to horizontal lines',
                            'change "plot_type" in obs_data.json to change this')
                        warning_printed = True
                    self.obs_info["plot_type"].append("horizontal")
                elif self.gt_df.iloc[II]["data_type"] == "prob_dist":
                    if not warning_printed:
                        self._rank0_print('prob_dist data types plot type defaults to horizontal lines',
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
                    self._rank0_print(f'data type {self.gt_df.iloc[II]["data_type"]} not recognised')
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
                # Check optimiser_options first, then ga_options for backwards compatibility
                options_dict = self.optimiser_options if self.optimiser_options is not None else self.ga_options
                if options_dict is not None:
                    if "cost_type" in options_dict.keys():
                        self.obs_info["cost_type"].append(options_dict["cost_type"]) # default to cost type in options
                    else:
                        self.obs_info["cost_type"].append("MSE") # default to mean squared error
                elif self.mcmc_options is not None:
                    if "cost_type" in self.mcmc_options.keys():
                        self.obs_info["cost_type"].append(self.mcmc_options["cost_type"]) # default to cost type in mcmc_options
                    else:
                        self.obs_info["cost_type"].append("MSE") # default to mean squared error
                else:
                    self._rank0_print("cost_type not found in obs_data.json, ga_options, or mcmc_options, exiting")
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
                self._rank0_print('experiment_colors in obs_data.json not the same length as num_experiments, exiting')
                exit()

        if "experiment_labels" in self.protocol_info.keys():
            if len(self.protocol_info["experiment_labels"]) != self.protocol_info['num_experiments']:
                self._rank0_print('experiment_labels in obs_data.json not the same length as num_experiments, exiting')
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
                    self._rank0_print(f'experiment_idx not found in obs_data.json entry {self.gt_df.iloc[II]["variable"]}, '
                          'but multiple experiments are defined.',
                          'Setting experiment_idx to 0 for all data points')
            if "subexperiment_idx" not in self.gt_df.iloc[II].keys():
                self.gt_df["subexperiment_idx"] = 0
                if self.protocol_info['num_sub_total'] > 1:
                    self._rank0_print(f'subexperiment_idx not found in obs_data.json entry {self.gt_df.iloc[II]["variable"]}, '
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
    
    def set_output_dir(self, path):
        
        self.output_dir = path
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_samples(self):

        problem = {
            'num_vars': self.num_params,
            'names': self.SA_cfg["param_names"],
            'bounds': list(zip(self.SA_cfg["param_mins"], self.SA_cfg["param_maxs"]))
        }
        self.problem = problem

        self.num_samples = self.SA_cfg["num_samples"]

        if self.SA_cfg["sample_type"] == "saltelli":
            samples = saltelli.sample(problem, self.num_samples, calc_second_order=True)  # Enable second-order interactions
        elif self.SA_cfg["sample_type"] == "sobol":
            samples = sobol.sample(problem, self.num_samples, calc_second_order=True)  # Enable second-order interactions
        else:
            raise ValueError(f"Unsupported sample type: {self.SA_cfg['sample_type']}")
        
        return samples
    
    def run_model_and_get_results(self, param_vals):
        self.sim_helper.set_param_vals(self.SA_cfg["param_names"], param_vals)
        self.sim_helper.reset_states()
        self.sim_helper.run()

        operands = self.sim_helper.get_results(self.obs_info["operands"])

        self.sim_helper.reset_and_clear()
        # y = self.sim_helper.get_results(self.model_output_names)
        # t = self.sim_helper.tSim - self.pre_time
        # return y, t
        return operands
    
    def generate_outputs_mpi(self, samples):
        # Split samples across ranks
        n_samples = len(samples)
        samples_per_rank = n_samples // self.num_procs
        remainder = n_samples % self.num_procs

        if self.rank < remainder:
            start = self.rank * (samples_per_rank + 1)
            end = start + samples_per_rank + 1
        else:
            start = self.rank * samples_per_rank + remainder
            end = start + samples_per_rank

        local_samples = samples[start:end]

        self._rank0_print(f"[MPI Rank {self.rank}] Starting samples {start}:{end} (total {len(local_samples)})")

        local_outputs = []

        # Create a single progress bar for rank 0 only to avoid noisy output from all ranks
        with tqdm(total=len(local_samples), desc=f"Rank {self.rank}", position=self.rank, leave=True, disable=self.rank != 0) as pbar:
            for param_vals in local_samples:

                # --- handle single vs multi subexperiment ---
                if self.protocol_info["num_sub_total"] == 1:
                    # simple case (one experiment only)
                    self.sim_helper.set_param_vals(self.param_id_info["param_names"], param_vals)
                    self.sim_helper.reset_states()
                    success = self.sim_helper.run()

                    operands_outputs_dict = {}
                    if success:
                        operands_outputs = self.sim_helper.get_results(self.obs_info["operands"])
                        # For single experiment and subexperiment, use (0, 0) as key
                        operands_outputs_dict[(0, 0)] = operands_outputs
                    else:
                        self._rank0_print(f"[MPI Rank {self.rank}] Simulation failed for params: {param_vals}")
                        local_outputs.append([np.inf])  # fail marker
                        pbar.update(1)
                        continue

                else:
                    # multiple subexperiments
                    current_time = 0
                    operands_outputs_dict = {}
                    for exp_idx in range(self.protocol_info["num_experiments"]):
                        self.sim_helper.set_param_vals(self.param_id_info["param_names"], param_vals)
                        self.sim_helper.reset_states()

                        for this_sub_idx in range(self.protocol_info["num_sub_per_exp"][exp_idx]):
                            subexp_count = int(np.sum(
                                [num_sub for num_sub in self.protocol_info["num_sub_per_exp"][:exp_idx]]
                            ) + this_sub_idx)

                            self.sim_time = self.protocol_info["sim_times"][exp_idx][this_sub_idx]
                            self.pre_time = self.protocol_info["pre_times"][exp_idx]

                            if self.protocol_info["num_sub_total"] > 1:
                                if this_sub_idx == 0:
                                    self.sim_helper.update_times(self.dt, 0.0, self.sim_time, self.pre_time)
                                    current_time += self.pre_time
                                else:
                                    self.sim_helper.update_times(self.dt, current_time, self.sim_time, 0.0)

                            # set subexperiment-specific parameters
                            self.sim_helper.set_param_vals(
                                list(self.protocol_info["params_to_change"].keys()),
                                [
                                    self.protocol_info["params_to_change"][param_name][exp_idx][this_sub_idx]
                                    for param_name in self.protocol_info["params_to_change"].keys()
                                ]
                            )

                            success = self.sim_helper.run()

                            retry_count = 0
                            max_retries = 0
                            original_MaximumStep = self.solver_info.get("MaximumStep", None)
                            original_MaximumNumberOfSteps = self.solver_info.get("MaximumNumberOfSteps", None)

                            while not success and retry_count < max_retries:

                                original_MaximumStep = self.solver_info.get("MaximumStep", None)
                                reduced_MaximumStep = original_MaximumStep / 2 if original_MaximumStep else 0.001
                                increased_MaximumNumberOfSteps = original_MaximumNumberOfSteps * 2 if original_MaximumNumberOfSteps else 1000000
                                retry_count += 1
                                # Reduce max_dt for retry
                                self.solver_info["MaximumStep"] = reduced_MaximumStep
                                self.solver_info["MaximumNumberOfSteps"] = increased_MaximumNumberOfSteps
                                
                                self.sim_helper.set_param_vals(self.param_id_info["param_names"], param_vals)
                                self.sim_helper.reset_states()
                                success = self.sim_helper.run()

                            # Restore original max_dt after retries
                            self.solver_info["MaximumStep"] = original_MaximumStep
                            self.solver_info["MaximumNumberOfSteps"] = original_MaximumNumberOfSteps
                            if success:
                                current_time += self.sim_time
                                operands_outputs = self.sim_helper.get_results(self.obs_info["operands"])
                                operands_outputs_dict[(exp_idx, this_sub_idx)] = operands_outputs

                                # reset at the end of each experiment
                                if this_sub_idx == self.protocol_info["num_sub_per_exp"][exp_idx] - 1:
                                    self.sim_helper.reset_and_clear()
                            else:
                                self._rank0_print(f"[MPI Rank {self.rank}] Simulation failed for params: {param_vals}, subexp={subexp_count} after {retry_count} retries")
                                # Set a flag in operands_outputs_dict to indicate failure
                                operands_outputs_dict[(exp_idx, this_sub_idx)] = {"failed": True}

                                # reset at the end of each experiment
                                if this_sub_idx == self.protocol_info["num_sub_per_exp"][exp_idx] - 1:
                                    self.sim_helper.reset_and_clear()

                features = []
                for j in range(len(self.obs_info["operations"])):
                    func = self.operation_funcs_dict[self.obs_info["operations"][j]]
                    exp_idx = self.obs_info["experiment_idxs"][j]
                    subexp_idx = self.obs_info["subexperiment_idxs"][j]
                    operands_outputs = operands_outputs_dict.get((exp_idx, subexp_idx), None)
                    if operands_outputs is not None and not (isinstance(operands_outputs, dict) and operands_outputs == {"failed": True}):
                        feature = func(*operands_outputs[j], **self.obs_info["operation_kwargs"][j])
                        features.append(feature)
                    else:
                        # WARNING: using mean biases variance estimates (shrinks variance), underestimates sensitivity
                        # TODO: come up with a better way to impute missing features
                        # Append the mean of the current features (ignoring None) -> reduces variance and bias induces toward zero
                        features.append(np.mean(features))

                local_outputs.append(features)
                pbar.update(1)

        self._rank0_print(f"[MPI Rank {self.rank}] Finished processing samples {start}:{end}")

        # Gather results at rank 0
        all_outputs = self.comm.gather(local_outputs, root=0)

        if self.rank == 0:
            outputs = [item for sublist in all_outputs for item in sublist]
            outputs = np.array(outputs)
            self._rank0_print(f"[MPI Rank 0] Gathered and flattened all outputs. Total outputs: {outputs.shape}")
            return outputs
        else:
            return None

    def sobol_index(self, outputs):

        if self.rank !=0:
            return None, None, None
        
        outputs = np.array(outputs)
    
        if outputs.ndim == 1:
            outputs = outputs[:, np.newaxis]  # convert to (n_samples, 1)

        n_outputs = outputs.shape[1]
        S1_all = np.zeros((n_outputs, self.num_params))
        ST_all = np.zeros((n_outputs, self.num_params))
        S2_all = np.zeros((n_outputs, self.num_params, self.num_params))

        for i in range(n_outputs):
            Si = sobol.analyze(self.problem, outputs[:,i], print_to_console=self.verbose)
            S1_all[i, :] = Si['S1']
            ST_all[i, :] = Si['ST']
            S2_all[i, :] = np.array(Si['S2'])

        return S1_all, ST_all, S2_all

    def plot_sobol_first_order_idx(self, S1_all, ST_all):

        if self.rank !=0:
            return
        
        """
        Plot first-order and total-order Sobol indices for multiple outputs.

        Parameters:
            S1_all (np.ndarray): First-order Sobol indices, shape (n_outputs, n_params)
            ST_all (np.ndarray): Total-order Sobol indices, shape (n_outputs, n_params)
        """
        n_outputs = S1_all.shape[0]
        x = np.arange(self.num_params)

        for i in range(n_outputs):
            S1 = S1_all[i]
            ST = ST_all[i]
            output_name = rf"${self.obs_info['names_for_plotting'][i]}$ - experiment{self.obs_info["experiment_idxs"][i]}, subexperiment{self.obs_info["subexperiment_idxs"][i]}"
            # output_name = self.obs_info["names_for_plotting"][i] if hasattr(self, "obs_info") else f"Output_{i}"

            # Set figure width adaptively based on number of parameters (xticks)
            fig_width = max(12, 1.0 * len(self.SA_cfg["param_names"]))
            plt.figure(figsize=(fig_width, 5))
            plt.bar(x - 0.2, S1, width=0.4, label='First-order', color='blue', alpha=0.7)
            plt.bar(x + 0.2, ST, width=0.4, label='Total-order', color='red', alpha=0.7)

            plt.xticks(x, self.SA_cfg["param_names"], rotation=45, fontsize=8)
            plt.ylabel('Sensitivity Index')
            plt.title(rf'Sobol Sensitivity - {output_name}')
            plt.legend()
            plt.tight_layout()

            file_name = f"{output_name}_n{self.num_samples}_First_order_idx.png"
            plt.savefig(os.path.join(self.save_path, file_name))
            plt.clf()
            plt.close()

    def plot_sobol_S2_idx(self, S2_all):
        """
        Plot second-order Sobol interaction indices for multiple outputs.

        Parameters:
            S2_all (np.ndarray): Second-order indices, shape (n_outputs, n_params, n_params)
        """

        if self.rank !=0:
            return
        
        n_outputs = S2_all.shape[0]
        for i in range(n_outputs):
            S2 = S2_all[i]
            output_name = rf"${self.obs_info['names_for_plotting'][i]}$ - experiment{self.obs_info["experiment_idxs"][i]}, subexperiment{self.obs_info["subexperiment_idxs"][i]}"

            # plt.figure(figsize=(6, 5))
            fig_width = max(6, 1.0 * len(self.SA_cfg["param_names"]))
            plt.figure(figsize=(fig_width, fig_width))
            sns.heatmap(S2, annot=True, fmt=".2f", xticklabels=self.SA_cfg["param_names"], yticklabels=self.SA_cfg["param_names"], cmap="coolwarm")
            plt.title(rf"2nd order Sobol Indices - {output_name}")
            plt.tight_layout()

            filename = f"{output_name}_n{self.num_samples}_2nd_order_idx.png"
            plt.savefig(os.path.join(self.save_path, filename))
            plt.clf()
            plt.close()

    def run(self):
        samples = self.generate_samples()
        if self.use_mpi:
            outputs = self.generate_outputs_mpi(samples)
            if self.rank == 0:
                S1_all, ST_all, S2_all = self.sobol_index(outputs)
                # print(f">>>>>>>>>>  {S1_all}, {ST_all}, {S2_all}")
                return S1_all, ST_all, S2_all
            else:
                return None, None, None
        else:
            outputs = self.generate_outputs(samples)
            S1_all, ST_all, S2_all = self.sobol_index(outputs)
            return S1_all, ST_all, S2_all

