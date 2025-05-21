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
            
        inp_data_dict['model_path'] = os.path.join(inp_data_dict['generated_models_subdir'], f'{file_prefix}.cellml')


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

        if 'DEBUG' in inp_data_dict.keys(): 
            if inp_data_dict['DEBUG']:
                inp_data_dict['ga_options'] = inp_data_dict['debug_ga_options']
                inp_data_dict['mcmc_options'] = inp_data_dict['debug_mcmc_options']
            else:
                pass
        else:
            inp_data_dict['DEBUG'] = False

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

    def json_to_dataframe_with_user_dir(self, json_path, json_dir):
        df = self.json_to_dataframe(json_path)
        user_module_dfs = [self.json_to_dataframe(os.path.join(json_dir, file)) \
                for file in os.listdir(json_dir) if file.endswith('.json')]
        for user_module_df in user_module_dfs:
            df = pd.concat([df, user_module_df], ignore_index=True)
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
            elif this_vessel_module_df.ndim > 1:
                print(f'combination of vessel_type = {vessel_type} and BC_type = {BC_type} is not unique, Make sure you',
                      'are not trying to redefine a module with a defined type and BC that is already in module_config.json',
                      'or in one of the other user defined module_config.json files in module_config_user/')
                exit()
            for column in add_on_lists:
                # deepcopy to make sure that the lists for different vessel same module are not linked
                try:
                    if np.isnan(this_vessel_module_df[column]):
                        add_on_lists[column].append("None")
                    else:
                        add_on_lists[column].append(copy.deepcopy(this_vessel_module_df[column]))
                except:
                    add_on_lists[column].append(copy.deepcopy(this_vessel_module_df[column]))

        for column in add_on_lists:
            vessel_df[column] = add_on_lists[column]





