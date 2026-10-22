'''
Created on 29/10/2021

@author: Finbar J. Argus
'''

import sys
import os
import re
from libcuflynx.utilities.paths import default_resources_dir, default_user_inputs_dir

user_inputs_dir = default_user_inputs_dir()

from libcuflynx.utilities.parameters_file import (get_leaf, read_parameters_file, set_leaf,
                                                  name_path, parameter_rows, write_parameters_file,
                                                  leaf_from_row)
import traceback
import yaml

def insert_parameters(parameters_path, parameters_to_add_path):
    '''Set every parameter of the parameters file ``parameters_to_add_path`` in the one at
    ``parameters_path`` (both nested JSON, utilities/parameters_file.py), in place.'''
    doc = read_parameters_file(parameters_path)
    for row in parameter_rows(read_parameters_file(parameters_to_add_path)):
        path = name_path(row['variable_name'])
        print(f'{row["variable_name"]}: {(get_leaf(doc, path) or {}).get("value")} -> '
              f'{row["value"]}')
        set_leaf(doc, path, leaf_from_row(row))
    write_parameters_file(parameters_path, doc)

if __name__ == '__main__':
    
    try:
        if len(sys.argv) == 2:
            with open(os.path.join(user_inputs_dir, 'user_inputs.yaml'), 'r') as file:
                inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)
            # .get(): the key is commented out in the shipped user_inputs.yaml, so absent
            # means "no override", not a KeyError.
            if inp_data_dict.get("user_inputs_path_override"):
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
