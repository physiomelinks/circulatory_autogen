import pandas as pd
import json
import numpy as np
from scipy import signal as sig

# This function saves a list of dictionaries to a json file
def save_list_of_dictionaries_to_json_file(list_of_dicts, output_json_file_name):

    json_df = pd.DataFrame(list_of_dicts)

    result = json_df.to_json(orient='records')
    parsed = json.loads(result)
    with open(output_json_file_name, 'w') as wf:
        json.dump(parsed, wf, indent=2)

output_json_file_name = "/home/farg967/Documents/random/example_obs_data.json"

dt = 0.01
T = 1.0
nSteps = int(T / dt)

no_conv = 1.0
mmHg_to_Pa = 133.332
ml_per_s_to_m3_per_s = 1e-6
ml_to_m3 = 1e-6

# list of dictionaries to save as json file
variable_list = ['v_ao', 'u_LV', 'u_LV']
data_type = ['series', 'constant', 'constant']
unit = ['m3_per_s', 'J_per_m3', 'J_per_m3']
name_for_plotting = ['$v_{ao}$', '$u_{LV}$', '$u_{LV}$']
weight = [1.0, 1.0, 1.0]
obs_type = ['series', 'min', 'max']

value_pre_conv = [[1.0 for II in range(nSteps)], 8000.0, 12000.0] # these values are just examples
                                                                  # actual values can be accessed from csv or other
# include conversions if you need to convert original units to the units specified in unit
conversions = [mmHg_to_Pa, ml_per_s_to_m3_per_s, ml_per_s_to_m3_per_s]
# conversions = [no_conv, no_conv, no_conv]

# create std and value list that has been converted to correct units
# the below assumes that the CV is 0.1 for all variables
std = []
value = []
for val, conv in zip(value_pre_conv, conversions):
    if type(val) is list:
        std.append([0.1*val_*conv for val_ in val])
        value.append([val_*conv for val_ in val])
    else:
        std.append(0.1*val)
        value.append(val*conv)

sample_rate = [1/dt, 'null', 'null'] # sample rate in Hz, null if not a series

num_entries = len(variable_list)
list_of_dicts = []
for II in range(num_entries):
    entry = {'variable': variable_list[II],
             'data_type': data_type[II],
             'unit': unit[II],
             'name_for_plotting': name_for_plotting[II],
             'weight': weight[II],
             'obs_type': obs_type[II],
             'std': std[II],
             'value': value[II],
             'sample_rate': sample_rate[II]}
    list_of_dicts.append(entry)

save_list_of_dictionaries_to_json_file(list_of_dicts, output_json_file_name)



