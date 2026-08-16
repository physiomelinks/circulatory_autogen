'''
Created on 29/10/2021

@author: Finbar J. Argus
'''

import sys
import os
import re
import pandas as pd
from libcuflynx.utilities.paths import default_resources_dir, default_user_inputs_dir

user_inputs_dir = default_user_inputs_dir()

from libcuflynx.parsers.PrimitiveParsers import JSONFileParser, CSVFileParser
import traceback
import yaml

def insert_parameters(parameters_path, parameters_to_add_path):
    
    cp = CSVFileParser()
    jp = JSONFileParser()

    if parameters_path.endswith('csv'):
        params_df = cp.get_data_as_dataframe(parameters_path)
    elif parameters_path.endswith('json'):
        params_df = jp.json_to_dataframe(parameters_path)

    print('before')
    print(params_df)
    
    if parameters_to_add_path.endswith('csv'):
        new_params_df = cp.get_data_as_dataframe(parameters_to_add_path)
    elif parameters_to_add_path.endswith('json') :
        new_params_df = jp.json_to_dataframe(parameters_to_add_path)

    # params_df = pd.concat([params_df, new_params_df], ignore_index=True)
    params_df = new_params_df.set_index('variable_name').combine_first(params_df.set_index('variable_name')).reset_index()
    params_df.to_csv(parameters_path, index=None, header=True, columns=['variable_name', 'units', 'value', 'data_reference'])
    print('after')
    print(params_df)

if __name__ == '__main__':
    
    try:
        if len(sys.argv) == 2:
            with open(os.path.join(user_inputs_dir, 'user_inputs.yaml'), 'r') as file:
                inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)
            if inp_data_dict["user_inputs_path_override"]:
                if os.path.exists(inp_data_dict["user_inputs_path_override"]):
                    with open(inp_data_dict["user_inputs_path_override"], 'r') as file:
                        inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)
                else:
                    print(f"User inputs file not found at {inp_data_dict['user_inputs_path_override']}")
                    print("Check the user_inputs_path_override key in user_inputs.yaml and set it to False if "
                          "you want to use the default user_inputs.yaml location")
                    exit()

            resources_dir = default_resources_dir()
            # overwrite dir paths if set in user_inputs.yaml
            if "resources_dir" in inp_data_dict.keys():
                resources_dir = inp_data_dict['resources_dir']
                
            parameters_csv_abs_path = os.path.join(resources_dir, inp_data_dict['input_param_file'])
            insert_parameters(parameters_csv_abs_path, sys.argv[1])
        elif len(sys.argv) == 3:
            insert_parameters(sys.argv[1], sys.argv[2])

        
    except:
        print(traceback.format_exc())
        print("expected usage: read_and_insert_parameters.py parameters_to_add.json")
        exit()
