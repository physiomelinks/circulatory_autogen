'''
Created on 29/10/2021

@author: Finbar Argus, Gonzalo D. Maso Talou
'''

import pandas as pd
import numpy as np
import os
import sys
import csv
import json
import copy
import yaml
import re

root_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir, 'src'))
user_inputs_dir = os.path.join(root_dir, 'user_run_files')
src_dir = os.path.join(os.path.dirname(__file__), '..')
param_id_dir = os.path.join(src_dir, 'param_id')
base_dir = os.path.join(src_dir, '..')
operation_funcs_user_dir = os.path.join(base_dir, 'funcs_user')

class scriptFunctionParser(object):
    '''
    Parses scripts with functions into objects (dicts) which holds the functions
    '''

    def __init__(self):
        sys.path.append(param_id_dir)
        sys.path.append(operation_funcs_user_dir)
        '''
        Constructor
        '''
    
    def get_operation_funcs_dict(self):
        import operation_funcs
        import operation_funcs_user
        operation_funcs_dict = {}
        funcs = [item for item in dir(operation_funcs) if callable(getattr(operation_funcs, item))]
        funcs_user = [item for item in dir(operation_funcs_user) if callable(getattr(operation_funcs_user, item))]

        # create dict with keys of string of function names
        for func in funcs:
            operation_funcs_dict[func] = getattr(operation_funcs, func)
        for func in funcs_user:
            operation_funcs_dict[func] = getattr(operation_funcs_user, func)
        
        return operation_funcs_dict

    def get_cost_funcs_dict(self):
        # import cost_funcs # currently all costs are in cost_funcs_user
        import cost_funcs_user
        cost_funcs_dict = {}
        # funcs = [item for item in dir(cost_funcs) if callable(getattr(cost_funcs, item))]
        funcs_user = [item for item in dir(cost_funcs_user) if callable(getattr(cost_funcs_user, item))]

        # create dict with keys of string of function names
        # for func in funcs:
        #     cost_funcs_dict[func] = getattr(cost_funcs, func)
        for func in funcs_user:
            cost_funcs_dict[func] = getattr(cost_funcs_user, func)
        
        return cost_funcs_dict

class YamlFileParser(object):
    '''
    Parses Yaml files 
    '''
    def __init__(self):
        '''
        Constructor
        '''
    
    def parse_user_inputs_file(self, inp_data_dict, obs_path_needed=False, do_generation_with_fit_parameters=False):
        
        if inp_data_dict is None:
            with open(os.path.join(user_inputs_dir, 'user_inputs.yaml'), 'r') as file:
                inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)
            if "user_inputs_path_override" in inp_data_dict.keys() and inp_data_dict["user_inputs_path_override"]:
                if os.path.exists(inp_data_dict["user_inputs_path_override"]):
                    user_files_dir = os.path.dirname(inp_data_dict["user_inputs_path_override"])
                    with open(inp_data_dict["user_inputs_path_override"], 'r') as file:
                        inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)
                else:
                    print(f"User inputs file not found at {inp_data_dict['user_inputs_path_override']}")
                    print("Check the user_inputs_path_override key in user_inputs.yaml and set it to False if "
                            "you want to use the default user_inputs.yaml location")
                    exit()
            else:
                user_files_dir = ''
        else:
            user_files_dir = ''
    
        file_prefix = inp_data_dict['file_prefix']

        # overwrite dir paths if set in user_inputs.yaml
        if "resources_dir" in inp_data_dict.keys():
            inp_data_dict['resources_dir'] = os.path.join(user_files_dir, inp_data_dict['resources_dir'])
        else:
            inp_data_dict['resources_dir'] = os.path.join(root_dir, 'resources')
        if "param_id_output_dir" in inp_data_dict.keys():
            inp_data_dict['param_id_output_dir'] = os.path.join(user_files_dir, inp_data_dict['param_id_output_dir'])
        else:
            inp_data_dict['param_id_output_dir'] = os.path.join(root_dir, 'param_id_output')
        if "generated_models_dir" in inp_data_dict.keys():
            inp_data_dict['generated_models_dir'] = os.path.join(user_files_dir, inp_data_dict['generated_models_dir'])
        else:
            inp_data_dict['generated_models_dir'] = os.path.join(root_dir, 'generated_models')
        
        if obs_path_needed:
            if 'param_id_obs_path' in inp_data_dict.keys():
                inp_data_dict['param_id_obs_path'] = os.path.join(user_files_dir, inp_data_dict['param_id_obs_path'])
                if not os.path.exists(inp_data_dict['param_id_obs_path']):
                    print(f'param_id_obs_path={inp_data_dict["param_id_obs_path"]} does not exist')
                    exit()
            else:
                print(f'param_id_obs_path needs to be defined in user_inputs.yaml')
                exit()

            if 'params_for_id_file' in inp_data_dict.keys():
                inp_data_dict['params_for_id_path'] = os.path.join(inp_data_dict['resources_dir'], inp_data_dict['params_for_id_file'])
            else:
                inp_data_dict['params_for_id_path'] = os.path.join(inp_data_dict['resources_dir'], f'{file_prefix}_params_for_id.csv')

            if not os.path.exists(inp_data_dict['params_for_id_path']):
                print(f'params_for_id path of {inp_data_dict["params_for_id_path"]} doesn\'t exist, user must create this file')
                exit()

        if do_generation_with_fit_parameters:
            data_str_addon = re.sub('.json', '', os.path.split(inp_data_dict['param_id_obs_path'])[1])
            inp_data_dict['param_id_output_dir_abs_path'] = os.path.join(inp_data_dict['param_id_output_dir'], 
                                                                         inp_data_dict['param_id_method'] + f'_{file_prefix}_{data_str_addon}')
            inp_data_dict['generated_models_subdir'] = os.path.join(inp_data_dict['generated_models_dir'], 
                                                                    file_prefix + '_' + data_str_addon)
        else:
            inp_data_dict['generated_models_subdir'] = os.path.join(inp_data_dict['generated_models_dir'], file_prefix)
        
        if not os.path.exists(inp_data_dict['generated_models_dir']):
            os.mkdir(inp_data_dict['generated_models_dir'])

        if not os.path.exists(inp_data_dict['generated_models_subdir']):
            os.mkdir(inp_data_dict['generated_models_subdir'])
            
        if inp_data_dict.get('model_type') == 'python':
            model_ext = '.py'
        else:
            model_ext = '.cellml'

        inp_data_dict['model_path'] = os.path.join(inp_data_dict['generated_models_subdir'], f'{file_prefix}{model_ext}')

        if do_generation_with_fit_parameters:
            inp_data_dict['uncalibrated_model_path'] = os.path.join(inp_data_dict["generated_models_dir"], file_prefix,
                                               file_prefix + model_ext)
        else:
            inp_data_dict['uncalibrated_model_path'] = inp_data_dict['model_path']


        if 'pre_time' in inp_data_dict.keys():
            inp_data_dict['pre_time'] = inp_data_dict['pre_time']
        else:
            inp_data_dict['pre_time'] = None
        if 'sim_time' in inp_data_dict.keys():
            inp_data_dict['sim_time'] = inp_data_dict['sim_time']
        else:
            inp_data_dict['sim_time'] = None

        if inp_data_dict['solver_info'] is None:
            print('solver_info must be defined in user_inputs.yaml',
                'MaximumStep is now an entry of solver_info in the user_inputs.yaml file')
            exit()

        # For the Python solver path, use the 'solver' entry to select the solve_ivp method.
        # Fall back to legacy 'method' or the top-level solver setting for backwards compatibility.
        solver_method = inp_data_dict['solver']
        if solver_method is None and 'method' in inp_data_dict['solver_info']:
            solver_method = inp_data_dict['solver_info']['method']
            print(f'solver_info["method"] = {solver_method} is now deprecated, use solver = {solver_method} instead')
            print('This will be removed in a future version')
            inp_data_dict['solver'] = solver_method
        if solver_method is None:
            print('solver must be an entry in the user_inputs.yaml file')
            exit()
        # also store it as method.
        inp_data_dict['solver_info']['method'] = solver_method

        if 'DEBUG' in inp_data_dict.keys(): 
            if inp_data_dict['DEBUG']:
                # For backwards compatibility, still set ga_options if debug_ga_options exists
                if 'debug_ga_options' in inp_data_dict.keys():
                    inp_data_dict['ga_options'] = inp_data_dict['debug_ga_options']
                if 'debug_mcmc_options' in inp_data_dict.keys():
                    inp_data_dict['mcmc_options'] = inp_data_dict['debug_mcmc_options']
            else:
                pass
        else:
            inp_data_dict['DEBUG'] = False

        if not 'external_modules_dir' in inp_data_dict.keys():
            inp_data_dict['external_modules_dir'] = None
        else:
            # check if it is an absolute path
            if not os.path.isabs(inp_data_dict['external_modules_dir']):
                inp_data_dict['external_modules_dir'] = os.path.join(user_files_dir, inp_data_dict['external_modules_dir'])
            else:
                inp_data_dict['external_modules_dir'] = inp_data_dict['external_modules_dir']
            # check if external_modules_dir is a valid directory
            if not os.path.exists(inp_data_dict['external_modules_dir']):
                print(f'external_modules_dir={inp_data_dict["external_modules_dir"]} does not exist')
                exit()
        
        # for sensitivity analysis and parameter identification
        if not 'sa_options' in inp_data_dict.keys():
            inp_data_dict['sa_options'] = None

        if inp_data_dict['sa_options'] is None:
            inp_data_dict['sa_options'] = {
                'method': 'sobol',
                'num_samples': 32,
                'sample_type': 'saltelli',
                'output_dir': os.path.join(root_dir, 'sensitivity_outputs', file_prefix + '_SA_results')
            }
        else:
            if 'output_dir' not in inp_data_dict['sa_options'].keys():
                inp_data_dict['sa_options']['output_dir'] = os.path.join(root_dir, 'sensitivity_outputs', file_prefix + '_SA_results')  
            else:
                if not os.path.isabs(inp_data_dict['sa_options']['output_dir']):
                    inp_data_dict['sa_options']['output_dir'] = os.path.join(root_dir, 'sensitivity_outputs', inp_data_dict['sa_options']['output_dir']) 
            
            if not os.path.exists(inp_data_dict['sa_options']['output_dir']):
                os.makedirs(inp_data_dict['sa_options']['output_dir'])
            
            if 'method' not in inp_data_dict['sa_options'].keys():
                print('No method specified for sensitivity analysis, setting to sobol by default')
                inp_data_dict['sa_options']['method'] = 'sobol'
            if 'num_samples' not in inp_data_dict['sa_options'].keys():
                print('No num_samples specified for sensitivity analysis, setting to 32 by default')
                inp_data_dict['sa_options']['num_samples'] = 32
            if 'sample_type' not in inp_data_dict['sa_options'].keys():
                print('No sample_type specified for sensitivity analysis, setting to saltelli by default')
                inp_data_dict['sa_options']['sample_type'] = 'saltelli'
            
        if 'do_ia' not in inp_data_dict.keys():
            inp_data_dict['do_ia'] = False
        
        if 'ia_options' not in inp_data_dict.keys():
            inp_data_dict['ia_options'] = {
                'method': 'Laplace'
            }
        else:
            if 'method' not in inp_data_dict['ia_options'].keys():
                print('No method specified for identifiability analysis, setting to Laplace by default')
                inp_data_dict['ia_options']['method'] = 'Laplace'
        
        # Parse optimiser_options - this is the new unified way to specify options
        # Handle backwards compatibility: if ga_options or debug_ga_options is specified, merge into optimiser_options
        if 'optimiser_options' not in inp_data_dict.keys():
            inp_data_dict['optimiser_options'] = {}
        else:
            # Ensure optimiser_options is a dictionary
            if inp_data_dict['optimiser_options'] is None:
                inp_data_dict['optimiser_options'] = {}
        
        # Backwards compatibility: convert ga_options to optimiser_options
        # Only copy entries that don't already exist in optimiser_options to avoid duplicates
        # Note: If DEBUG is True, ga_options may have been set to debug_ga_options above,
        # but we still want to merge the original ga_options first (if it exists and DEBUG is False)
        # or merge debug_ga_options with higher precedence when DEBUG is True
        if not inp_data_dict['DEBUG']:
            # When DEBUG is False, merge ga_options normally
            if 'ga_options' in inp_data_dict.keys() and inp_data_dict['ga_options'] is not None:
                ga_opts = inp_data_dict['ga_options']
                if isinstance(ga_opts, dict):
                    for key, value in ga_opts.items():
                        # Only add if not already in optimiser_options
                        if key not in inp_data_dict['optimiser_options']:
                            inp_data_dict['optimiser_options'][key] = value
                        # Warn if there's a conflict (same key, different value)
                        elif inp_data_dict['optimiser_options'][key] != value:
                            print(f'Warning: ga_options["{key}"] conflicts with optimiser_options["{key}"]. '
                                  f'Using optimiser_options value: {inp_data_dict["optimiser_options"][key]}')
        
        # Handle debug_ga_options for backwards compatibility
        # When DEBUG is True, debug_ga_options should override optimiser_options
        if inp_data_dict['DEBUG']:
            if 'debug_ga_options' in inp_data_dict.keys() and inp_data_dict['debug_ga_options'] is not None:
                debug_ga_opts = inp_data_dict['debug_ga_options']
                if isinstance(debug_ga_opts, dict):
                    for key, value in debug_ga_opts.items():
                        # Debug options override optimiser_options (they take precedence)
                        if key in inp_data_dict['optimiser_options']:
                            if inp_data_dict['optimiser_options'][key] != value:
                                print(f'Note: debug_ga_options["{key}"] overriding optimiser_options["{key}"] '
                                      f'({inp_data_dict["optimiser_options"][key]} -> {value})')
                        inp_data_dict['optimiser_options'][key] = value
        
        # Handle debug_optimiser_options (new preferred way)
        # When DEBUG is True, debug_optimiser_options should override everything
        if inp_data_dict['DEBUG']:
            if 'debug_optimiser_options' in inp_data_dict.keys() and inp_data_dict['debug_optimiser_options'] is not None:
                debug_opts = inp_data_dict['debug_optimiser_options']
                if isinstance(debug_opts, dict):
                    for key, value in debug_opts.items():
                        # Debug optimiser options override everything (highest precedence)
                        if key in inp_data_dict['optimiser_options']:
                            if inp_data_dict['optimiser_options'][key] != value:
                                print(f'Note: debug_optimiser_options["{key}"] overriding optimiser_options["{key}"] '
                                      f'({inp_data_dict["optimiser_options"][key]} -> {value})')
                        inp_data_dict['optimiser_options'][key] = value

        # for generation only
    
        inp_data_dict['vessels_csv_abs_path'] = os.path.join(inp_data_dict['resources_dir'], file_prefix + '_vessel_array.csv')
        inp_data_dict['parameters_csv_abs_path'] = os.path.join(inp_data_dict['resources_dir'], inp_data_dict['input_param_file'])

        return inp_data_dict

class CSVFileParser(object):
    '''
    Parses CSV files
    '''

    def __init__(self):
        '''
        Constructor
        '''
        
    def get_data_as_dataframe_multistrings(self, filename, has_header=True):
        '''
        Returns the data in the CSV file as a Pandas dataframe where entries in the data array that have two
        entries are put in a list in the entry for the dataframe
        :param filename: filename of CSV file
        :param has_header: If CSV file has a header
        '''
        if( has_header ):
            csv_dataframe = pd.read_csv(filename, dtype=str, na_filter=False)
        else:
            csv_dataframe = pd.read_csv(filename, dtype=str, header=None, na_filter=False)

        csv_dataframe = csv_dataframe.rename(columns=lambda x: x.strip())
        for II in range(csv_dataframe.shape[0]):
            for column_name in csv_dataframe.columns:
                entry = csv_dataframe[column_name][II]
                if type(entry) is not str:
                    sub_entries = []
                else:
                    sub_entries = entry.split()

                if column_name in ['vessel_name', 'inp_vessels', 'out_vessels']:
                    if sub_entries == []:
                        new_entry = []
                        pass
                    else:
                        new_entry = [sub_entry.strip() for sub_entry in sub_entries]
                else:
                    if sub_entries == []:
                        new_entry = []
                    else:
                        new_entry = sub_entries[0].strip()

                csv_dataframe.loc[II, column_name] = new_entry

        # for column_name in csv_dataframe.columns:
        #     if column_name == 'vessel_name':
        #         continue
        #     csv_dataframe[column_name] = csv_dataframe[column_name].str.strip()
    
        return csv_dataframe

    def get_data_as_dataframe(self, filename, has_header=True):
        '''
        Returns the data in the CSV file as a Pandas dataframe
        :param filename: filename of CSV file
        :param has_header: If CSV file has a header
        '''
        if (has_header):
            csv_dataframe = pd.read_csv(filename, dtype=str)
        else:
            csv_dataframe = pd.read_csv(filename, dtype=str, header=None)

        for column_name in csv_dataframe.columns:
            csv_dataframe[column_name] = csv_dataframe[column_name].str.strip()

        return csv_dataframe

    def get_data_as_nparray(self,filename,has_header=True):
        '''
        Returns the data in the CSV file as a numpy array
        :param filename: filename of CSV file
        :param has_header: If CSV file has a header
        '''

        csv_dataframe = self.get_data_as_dataframe(filename, has_header)
    
        csv_np_array = csv_dataframe.to_numpy()
        dtypes = []
        for column in list(csv_dataframe.columns):
            dtypes.append((column,'<U80'))
            
        csv_np_array = np.array(list(zip(*csv_np_array.T)), dtype=dtypes)
    
        return csv_np_array

    def get_data_as_dictionary(self,filename):
        '''
        Returns the data in the CSV file as a Python dictionary
        :param filename: filename of CSV file
        '''

        csv_dataframe = self.get_data_as_dataframe(filename).T
        csv_dictionary = csv_dataframe.to_dict()
        
        return list(csv_dictionary.values())


    def get_param_id_params_as_lists_of_tuples(self, param_id_dir):

        param_names = []

        with open(os.path.join(os.path.join(param_id_dir, 'param_names_for_gen.csv')), 'r') as f:
            rd = csv.reader(f)
            for row in rd:
                param_names.append(row)


        # get date identifier of the parameter id
        date_id = np.load(os.path.join(os.path.join(param_id_dir, 'date.npy'))).item()

        param_vals = np.load(os.path.join(param_id_dir, 'best_param_vals.npy'))
        param_name_and_val = []
        
        for name_or_list, val in zip(param_names, param_vals):
            if isinstance(name_or_list, list):
                for name in name_or_list:
                    param_name_and_val.append((name, val))
            else:
                param_name_and_val.append((name, val))

        return param_name_and_val, date_id


class JSONFileParser(object):
    '''
    Parses json files
    '''

    def __init__(self):
        '''
        Constructor
        '''

    def json_to_dataframe(self, json_path):
        with open(json_path, encoding='utf-8-sig') as rf:
            json_obj = json.load(rf)
        df = pd.DataFrame(json_obj)
        return df

    def json_to_dataframe_with_user_dir(self, json_dir, json_user_dir, external_modules_dir):
        dfs = [self.json_to_dataframe(os.path.join(json_dir, file)) \
                for file in os.listdir(json_dir) if file.endswith('.json')]
        user_module_dfs = [self.json_to_dataframe(os.path.join(json_user_dir, file)) \
                for file in os.listdir(json_user_dir) if file.endswith('.json')]
        if external_modules_dir is not None:
            external_module_dfs = [self.json_to_dataframe(os.path.join(external_modules_dir, file)) \
                    for file in os.listdir(external_modules_dir) if file.endswith('.json')]
        else:
            external_module_dfs = []
            
        df = None
        for json_df in dfs:
            if df is None:
                df = json_df
            else:
                # concatenate dataframes, ignore index to reset the index
                # so that it is not duplicated
                df = pd.concat([df, json_df], ignore_index=True)

        for user_module_df in user_module_dfs:
            df = pd.concat([df, user_module_df], ignore_index=True)
        for external_module_df in external_module_dfs:
            df = pd.concat([df, external_module_df], ignore_index=True)
        return df

    def append_module_config_info_to_vessel_df(self, vessel_df, module_df):
        # add columns to vessel_df

        add_on_lists = {column:[] for column in module_df.columns[2:]}
        for vessel_tup in vessel_df.itertuples():
            vessel_type = vessel_tup.vessel_type
            BC_type = vessel_tup.BC_type
            if len(BC_type) <1 or len(vessel_type) <1:
                print('You have an empty entry in your vessel array, exiting')
                exit()
            this_vessel_module_df = module_df.loc[((module_df["vessel_type"] == vessel_type)
                                                   & (module_df["BC_type"] == BC_type))].squeeze()
            if this_vessel_module_df.empty:
                print(f'combination of vessel_type = {vessel_type} and BC_type = {BC_type} doesn\'t exist, check module_config.json',
                        'for this combination')
                exit()
            for column in add_on_lists:
                # deepcopy to make sure that the lists for different vessel same module are not linked
                val = this_vessel_module_df[column]
                is_na = False
                try:
                    mask = pd.isna(val)
                    if isinstance(mask, (np.bool_, bool)):
                        is_na = bool(mask)
                    else:
                        # array-like: consider NaN only if all entries are NaN
                        is_na = bool(np.all(mask))
                except Exception:
                    is_na = False

                if is_na:
                    add_on_lists[column].append("None")
                else:
                    add_on_lists[column].append(copy.deepcopy(val))

        for column in add_on_lists:
            vessel_df[column] = add_on_lists[column]





