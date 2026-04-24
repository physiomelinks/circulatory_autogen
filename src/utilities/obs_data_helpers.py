import numpy as np
import pandas as pd
try:
    import yaml
except ImportError:  # optional dependency
    yaml = None
import json
import os, sys
import re
root_dir = os.path.join(os.path.dirname(__file__), '../..')

class ObsDataCreator:
    def __init__(self):
        self.obs_data_dict = {}
        self.obs_data_dict['protocol_info'] = {}
        self.obs_data_dict['prediction_items'] = []
        self.obs_data_dict['data_items'] = []

    def add_protocol_info(self, pre_times, sim_times, params_to_change, experiment_labels=None):
        """
        Add protocol information to the dictionary.
        pre_times: list of pre-simulation times for each experiment
        sim_times: 2D list of lists of simulation times for each experiment and subexperiment
        params_to_change: dictionary with parameter names as keys and list of lists of values Each parameter should have a value
        entry the same shape as sim_times.
        experiment_labels: list of labels for each experiment
        """
        # check pre_times is list and sim_times 2D list of lists
        if not isinstance(pre_times, list):
            raise ValueError("pre_times should be a list")
        if not isinstance(sim_times, list) or not all(isinstance(sublist, list) for sublist in sim_times):
            raise ValueError("sim_times should be a 2D list of lists")
        # check sizes of lists are correct
        num_exps = len(sim_times)
        if len(pre_times) != num_exps:
            raise ValueError("pre_times should have the same length as the number of experiments (number of rows of sim_times).")
        if not all(isinstance(entry, list) for entry in sim_times):
            raise ValueError("sim_times should be a 2D list with one row for each experiment and a column for each subexperiment.")
        if experiment_labels is not None:
            if len(experiment_labels) != num_exps:
                raise ValueError("experiment_labels should have the same length as the number of experiments (number of rows of sim_times).")
        else:
            # if experiment_labels is not provided, create default labels
            experiment_labels = [f'exp_{i}' for i in range(num_exps)]
        if type(params_to_change) is not dict:
            raise ValueError("params_to_change should be a dictionary with parameter names as keys and lists of values as values.")
        for param, values in params_to_change.items():
            if len(values) != num_exps:
                raise ValueError(f"Parameter {param} should have the same number of values as the number of experiments ({num_exps}).")
            if not all(isinstance(v, list) for v in values):
                raise ValueError(f"Parameter {param} should have a list of values for each subexperiment.")
            if not all(len(v) == len(sim_times[i]) for i, v in enumerate(values)):
                raise ValueError(f"Parameter {param} should have the same number of values as the number of subexperiments for each experiment.")
        
        # now add to dict
        self.obs_data_dict['protocol_info']['pre_times'] = pre_times
        self.obs_data_dict['protocol_info']['sim_times'] = sim_times
        self.obs_data_dict['protocol_info']['params_to_change'] = params_to_change
        self.obs_data_dict['protocol_info']['experiment_labels'] = experiment_labels

    def add_prediction_item(self, variable, unit, experiment_idx):
        """
        Add a prediction item to the dictionary.
        variable: name of the variable to predict
        unit: unit of the variable
        experiment_idx: index of the experiment this prediction item belongs to
        """
        # check that experiment_idx is valid
        if experiment_idx < 0 or experiment_idx >= len(self.obs_data_dict['protocol_info']['sim_times']):
            raise ValueError(f"experiment_idx {experiment_idx} is out of bounds for the number of experiments ({len(self.obs_data_dict['protocol_info']['sim_times'])}).")

        prediction_item = {
            'variable': variable,
            'unit': unit,
            'experiment_idx': experiment_idx
        }
        self.obs_data_dict['prediction_items'].append(prediction_item)

    #TODO Create functions for adding entries to the data_items
    def add_data_item(self, entry):
        """
        Add a data item to the dictionary.
        entry: dictionary containing the data item
        """
        required_keys = ['variable', 'name_for_plotting', 'operands', 
                         'unit', 'value', 'std']
        required_series_keys = ['obs_dt']
        optional_keys = ['name_for_plotting', 'operation', 'weight', 'std', 'experiment_idx', 'subexperiment_idx']
                         
        if 'name_for_plotting' not in entry:
            entry['name_for_plotting'] = entry['variable']
        # check that name_for_plotting only has one _ in it and remove if not
        if entry['name_for_plotting'].count('_') > 1:
            print('Warning: name_for_plotting contains multiple underscores, replacing with \_')
            entry['name_for_plotting'] = re.sub('_', r'\_', entry['name_for_plotting'])
        if 'operation' not in entry:
            entry['operation'] = None # default to None if not provided
        if 'weight' not in entry:
            entry['weight'] = 1.0 # default to 1.0 if not provided

        if 'subexperiment_idx' not in entry:
            entry['subexperiment_idx'] = 0  # default to 0 if not provided
        if 'experiment_idx' not in entry:
            entry['experiment_idx'] = 0  # default to 0 if not provided
        for key in required_keys:
            if key not in entry:
                raise ValueError(f"Entry is missing required key: {key}")
        # check if value is a list or array and asign data_type accordingly
        if 'data_type' not in entry:
            if type(entry['value']) is list or type(entry['value']) is np.ndarray:
                entry['data_type'] = 'series'
                if 'obs_dt' in entry.keys():
                    pass
                elif 'dt' in entry.keys():
                    print("Warning: 'dt' for the time step of series data items is deprecated, ",
                        "please use 'obs_dt' instead. Setting 'obs_dt' to 'dt'.")
                    entry['obs_dt'] = entry['dt']
                    pass
                else:
                    raise ValueError(f"obs_dt is required for series entries")
            else:
                entry['data_type'] = 'constant'


        if self.obs_data_dict['protocol_info'] != {}:
            # check that experiment_idx and subexperiment_idx are valid if there is a protocol_info
            if entry['experiment_idx'] < 0 or entry['experiment_idx'] >= len(self.obs_data_dict['protocol_info']['sim_times']):
                raise ValueError(f"experiment_idx {entry['experiment_idx']} is out of bounds for the number of experiments ({len(self.obs_data_dict['protocol_info']['sim_times'])}).")
            if entry['subexperiment_idx'] < 0 or entry['subexperiment_idx'] >= len(self.obs_data_dict['protocol_info']['sim_times'][entry['experiment_idx']]):
                raise ValueError(f"subexperiment_idx {entry['subexperiment_idx']} is out of bounds for the number of subexperiments in experiment {entry['experiment_idx']} ({len(self.obs_data_dict['protocol_info']['sim_times'][entry['experiment_idx']])}).")

        for key in entry.keys():
            if isinstance(entry[key], np.ndarray):
                entry[key] = entry[key].tolist()
            elif isinstance(entry[key], np.generic):
                entry[key] = entry[key].item()

        self.obs_data_dict['data_items'].append(entry)
    
    def get_obs_data_dict(self):
        """
        Returns the observation data dictionary.
        """
        return self.obs_data_dict

    def dump_to_path(self, output_path):
        """
        Dumps the observation data dictionary to a JSON file.
        """
        with open(output_path, 'w') as f:
            obs_data_dict = dict(self.obs_data_dict)
            if not obs_data_dict.get('prediction_items'):
                obs_data_dict.pop('prediction_items', None)
            json.dump(obs_data_dict, f, indent=2)
        print(f"Observation data dumped to {output_path}")
    
    def load_from_json_file(self, input_path):
        """
        Loads the observation data dictionary from a JSON file.
        input_path: path to the JSON file
        """
        with open(input_path, 'r') as f:
            data = json.load(f)
        self.obs_data_dict = data
        print(f"Observation data loaded from {input_path}")
        return data
