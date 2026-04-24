#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 09:45:17 2024

@author: bghi639
"""

import sys
import os
import re
import traceback
import yaml
import pandas as pd
from collections import Counter
from pathlib import Path
import json

def generate_param_array(inp_data_dict=None):

    # 1. Setup Path (Only when function runs)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(root_dir)
    root_dir = os.path.dirname(root_dir)

    src_path = os.path.join(root_dir, 'src')
    if src_path not in sys.path:
        sys.path.append(src_path)

    # 2. Import Classes (Lazy Load)
    from parsers.ModelParsers import CSV0DModelParser
    from generators.CVSCellMLGenerator import CVS0DCellMLGenerator
    from generators.CVSCppGenerator import CVS0DCppGenerator
    from parsers.PrimitiveParsers import YamlFileParser
    from parsers.PrimitiveParsers import CSVFileParser
    # --- LAZY LOADING END ---

    user_inputs_dir = os.path.join(root_dir, 'user_run_files')

    yaml_parser = YamlFileParser()
    inp_data_dict = yaml_parser.parse_user_inputs_file(inp_data_dict)

    file_prefix = inp_data_dict['file_prefix']
    resources_dir = inp_data_dict['resources_dir']
    vessels_csv_abs_path = inp_data_dict['vessels_csv_abs_path']
    parameters_csv_abs_path = inp_data_dict['parameters_csv_abs_path']

    parser = CSV0DModelParser(vessels_csv_abs_path, parameters_csv_abs_path)
    # print(file_prefix)
    # print(resources_dir)
    # print(vessels_csv_abs_path)
    # print(parameters_csv_abs_path)
    # print(parser.vessel_filename)
    # print(parser.parameter_filename)

    # csv_parser = CSVFileParser()
    # vessels_df = csv_parser.get_data_as_dataframe_multistrings(parser.vessel_filename, True)
    vessels_df = pd.read_csv(parser.vessel_filename, header=0, dtype=str) #, na_filter=False)
    vessels_df = vessels_df.fillna('')
    # print(vessels_df.head())
    # print(vessels_df.keys())

    nVess = vessels_df.shape[0]

    column_types = {
        'variable_name': 'str',
        'units': 'str',
        'value': 'float',
        'data_reference': 'str'
    }
    params_df = pd.DataFrame(columns=column_types.keys()).astype(column_types)
    # print(params_df.keys())


    # module_config = '../generators/resources/module_config.json'
    # with open(module_config, "r") as file:
    #     modules = json.load(file)
    module_config_fold1 = root_dir_src + '/generators/resources/'
    module_config_fold2 = root_dir + '/module_config_user/'
    modules = []
    for filename in os.listdir(module_config_fold1):
        if filename.endswith(".json"):
            with open(os.path.join(module_config_fold1, filename), "r") as file:
                temp_data = json.load(file)
                # modules.append(temp_data)
                if isinstance(temp_data, list):
                    modules.extend(temp_data)
                else:
                    modules.append(temp_data)
    for filename in os.listdir(module_config_fold2):
        if filename.endswith(".json"):
            with open(os.path.join(module_config_fold2, filename), "r") as file:
                temp_data = json.load(file)
                # modules.append(temp_data)
                if isinstance(temp_data, list):
                    modules.extend(temp_data)
                else:
                    modules.append(temp_data)


    # this list might not be complete
    default_global_const = [
                        ['T', 'second', 1.0, 'user_defined'],
                        ['rho',	'Js2_per_m5', '1040.0',	'known'],
                        ['mu', 'Js_per_m3', 0.004, 'known'],
                        ['nu', 'dimensionless', 0.5, 'known'],
                        ['SMvolfrac_art', 'dimensionless', 0.1, 'Toro_2021'],
                        ['SMvolfrac_ven', 'dimensionless', 0.08, 'Toro_2021'],
                        ['beta_g', 'dimensionless', 0, 'user_defined'],
                        ['g', 'm_per_s2', 9.81, 'known'],
                        ['a_vessel', 'dimensionless', 0.2802,' Avolio_1980'],
                        ['b_vessel', 'per_m', -505.3, 'Avolio_1980'],
                        ['c_vessel', 'dimensionless', 0.1324, 'Avolio_1980'],
                        ['d_vessel', 'per_m', -11.14, 'Avolio_1980'],
                        ['R_flag', 'dimensionless', 1, 'user_defined'],
                        ['pressure_venous_dist', 'J_per_m3', 0.0, 'user_defined'], # 666.6119
                        ['I_T_global', 'Js2_per_m6', 1.0e-06, 'user_defined'],
                        ["q_ra_us", "m3", 0.000004, 'Korakianitis_2006_Table_1'],
                        ["q_rv_us", "m3", 0.00001, 'Korakianitis_2006_Table_1'],
                        ["q_la_us", "m3", 0.000004, 'Korakianitis_2006_Table_1'],
                        ["q_lv_us", "m3", 0.000005, 'Korakianitis_2006_Table_1'],
                        ["q_ra_init", "m3", 0.000004, 'Korakianitis_2006_Table_1'],
                        ["q_rv_init", "m3", 0.00001, 'Korakianitis_2006_Table_1'],
                        ["q_la_init", "m3", 0.000004, 'Korakianitis_2006_Table_1'],
                        ["q_lv_init", "m3", 0.001, 'Korakianitis_2006_Table_1_TO_BE_IDENTIFIED'],
                        ["T_ac", "second", 0.17, 'Liang_2009_Table_2'],
                        ["T_ar", "second", 0.17, 'Liang_2009_Table_2'],
                        ["t_astart", "second", 0.8, 'Liang_2009_Table_2'],
                        ["T_vc", "second", 0.34, 'Liang_2009_Table_2'],
                        ["T_vr", "second", 0.15, 'Liang_2009_Table_2'],
                        ["t_vstart", "second", 0, 'Liang_2009_Table_2'],
                        ["E_ra_A", "J_per_m6", 7998000,'Liang_2009_Table_2'],
                        ["E_ra_B", "J_per_m6", 9331000,'Liang_2009_Table_2'],
                        ["E_rv_A", "J_per_m6", 73315000, 'Liang_2009_Table_2_TO_BE_IDENTIFIED'],
                        ["E_rv_B", "J_per_m6", 6665000, 'Liang_2009_Table_2'],
                        ["E_la_A", "J_per_m6", 9331000, 'Liang_2009_Table_2'],
                        ["E_la_B", "J_per_m6", 11997000, 'Liang_2009_Table_2'],
                        ["E_lv_A", "J_per_m6", 366575000, 'Liang_2009_Table_2_TO_BE_IDENTIFIED'],
                        ["E_lv_B", "J_per_m6", 10664000, 'Liang_2009_Table_2'],
                        ["K_vo_trv", "m3_per_Js", 0.3, 'Mynard_2012'],
                        ["K_vo_puv", "m3_per_Js", 0.2, 'Mynard_2012'],
                        ["K_vo_miv", "m3_per_Js", 0.3, 'Mynard_2012'],
                        ["K_vo_aov", "m3_per_Js", 0.12, 'Mynard_2012'],
                        ["K_vc_trv", "m3_per_Js", 0.4, 'Mynard_2012'],
                        ["K_vc_puv", "m3_per_Js", 0.2, 'Mynard_2012'],
                        ["K_vc_miv", "m3_per_Js", 0.4, 'Mynard_2012'],
                        ["K_vc_aov", "m3_per_Js", 0.12, 'Mynard_2012'],
                        ["M_rg_trv", "dimensionless", 0, 'Mynard_2012'],
                        ["M_rg_puv", "dimensionless", 0, 'Mynard_2012'],
                        ["M_rg_miv", "dimensionless", 0, 'Mynard_2012'],
                        ["M_rg_aov", "dimensionless", 0, 'Mynard_2012'],
                        ["M_st_trv", "dimensionless", 1, 'Mynard_2012'],
                        ["M_st_puv", "dimensionless", 1, 'Mynard_2012'],
                        ["M_st_miv", "dimensionless", 1, 'Mynard_2012'],
                        ["M_st_aov", "dimensionless", 1, 'Mynard_2012'],
                        ["l_eff", "metre", 0.01, 'TO_BE_IDENTIFIED'],
                        ["A_nn_trv", "m2", 0.0008, 'Mynard_2012'],
                        ["A_nn_puv", "m2", 0.00071, 'Mynard_2012'],
                        ["A_nn_miv", "m2", 0.00077, 'Mynard_2012'],
                        ["A_nn_aov", "m2", 0.00068, 'Mynard_2012'],
                    ]


    global_const = []

    for i in range(nVess):

        nameV = vessels_df.at[i,'name']
        typeBC = vessels_df.at[i,'BC_type']
        typeV = vessels_df.at[i,'vessel_type']
        # print(nameV, typeBC, typeV)

        matches = [entry for entry in modules if entry.get('BC_type')==typeBC and entry.get('vessel_type')==typeV]
        if len(matches)>1:
            sys.exit('ERROR :: multiple modules found for this vessel_type and BC_type combination : '+typeV+' '+typeBC+' :: Check your module_config.json file.')

        mod = matches[0]
        modType = mod['module_type']
        vars = mod['variables_and_units']
        nVars = len(vars)
        for j in range(nVars):
            var = vars[j]
            if var[-1]=='variable':
                pass
            elif var[-1]=='boundary_condition':
                pass
            elif var[-1]=='global_constant':
                var_new = [var[0], var[1]]
                if var_new not in global_const:
                    global_const.append(var_new)
            elif var[-1]=='constant':
                var_name = var[0]+'_'+nameV
                new_row = {'variable_name': var_name,
                            'units': var[1],
                            'value': -1.0,
                            'data_reference': 'TO_DO'}
                params_df = pd.concat([params_df, pd.DataFrame([new_row])], ignore_index=True)

            
    for i in range(len(global_const)):
        const_name = global_const[i][0]

        found = -1
        for j in range(len(default_global_const)):
            if const_name==default_global_const[j][0]:
                new_row = {'variable_name': const_name,
                                    'units': default_global_const[j][1],
                                    'value': default_global_const[j][2],
                                    'data_reference': default_global_const[j][3]}
                params_df = pd.concat([params_df, pd.DataFrame([new_row])], ignore_index=True)
                found = 1
                break

        if found==-1:
            new_row = {'variable_name': const_name,
                                    'units': global_const[i][1],
                                    'value': -1.0,
                                    'data_reference': 'TO_DO'}
            params_df = pd.concat([params_df, pd.DataFrame([new_row])], ignore_index=True)

    
    # # params_df.to_csv(parameters_csv_abs_path, sep=',', index=False, header=True)
    params_df.to_csv(parameters_csv_abs_path, index=False, header=True)
                
    print('DONE :: Parameters array file for model '+file_prefix+' generated and saved.')


if __name__ == '__main__':
    try:
        generate_param_array()

    except Exception as e:
        print(f"Failed to run python script generate_param_array.py: {e}", file=sys.stderr)
        exit()
