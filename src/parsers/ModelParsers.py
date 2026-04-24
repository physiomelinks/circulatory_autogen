'''
Created on 29/10/2021

@author: Finbar Argus, Gonzalo D. Maso Talou
'''


from parsers.PrimitiveParsers import CSVFileParser, JSONFileParser
from models.LumpedModels import CVS0DModel
from checks.LumpedModelChecks import LumpedCompositeCheck, LumpedBCVesselCheck, LumpedIDParamsCheck, LumpedPortVariableCheck
import pandas as pd
import numpy as np
import json
import re
import os

generator_resources_dir_path = os.path.join(os.path.dirname(__file__), '../generators/resources')
base_dir = os.path.join(os.path.dirname(__file__), '../..')


class CSV0DModelParser(object):
    '''
    Creates a 0D model representation from a vessel and a parameter CSV files.
    '''
    def __init__(self, inp_data_dict, parameter_id_dir=None):
        
        self.vessel_filename = inp_data_dict['vessels_csv_abs_path']
        self.vessel_filename_0d = None
        self.vessel_filename_1d = None
        if (inp_data_dict['model_type'] == 'cpp' and inp_data_dict['couple_to_1d']):
            self.vessel_filename_0d = inp_data_dict['vessels_0d_csv_abs_path']
            self.vessel_filename_1d = inp_data_dict['vessels_1d_csv_abs_path']
            
        self.parameter_filename = inp_data_dict['parameters_csv_abs_path']
        self.external_modules_dir = inp_data_dict['external_modules_dir']
        self.parameter_id_dir = parameter_id_dir
        self.module_config_dir = generator_resources_dir_path
        self.module_config_user_dir = os.path.join(base_dir, 'module_config_user')
        self.csv_parser = CSVFileParser()
        self.json_parser = JSONFileParser()

        self.conn_1d_0d_info = None

    def split_0d_1d_vessel_array(self):
        vessels_df = pd.read_csv(self.vessel_filename, header=0, dtype=str, skipinitialspace=True) #, na_filter=False)
        vessels_df = vessels_df.fillna('')
        
        vessels_rows_0d = []
        vessels_rows_1d = []
        for _, row in vessels_df.iterrows():
            if row["vessel_type"]=="FV1D_vessel":
                vessels_rows_1d.append(row)
            else:
                vessels_rows_0d.append(row)

        vessels_df_0d = pd.DataFrame(vessels_rows_0d, columns=vessels_df.columns)
        vessels_df_1d = pd.DataFrame(vessels_rows_1d, columns=vessels_df.columns)

        vessels_df_0d = vessels_df_0d.reset_index(drop=True)
        vessels_df_1d = vessels_df_1d.reset_index(drop=True)
        #XXX Bea: I think I need the code below for compatibility with Finbar's original code
        # where he was assuming that the names of 1d vessels connected to 0d modules were always of format "FV1D_#", or "FV1D_##", or "FV1D_###", etc
        # but still, I dont think this would always work
        # I think my code should now be more general, as it doesnt require 1D FV vessel names to be of that specific format
        # i.e., no need of a numeric index at the end of the vessel name
        if vessels_df_1d.at[0,'name'].startswith('FV1D_') and len(re.findall(r'\d+$', vessels_df_1d.at[0,'name'])):    
            # extract numeric suffix and convert to int
            vessels_df_1d['order'] = vessels_df_1d['name'].str.extract(r'_(\d+)$').astype(int)
            # sort by numeric suffix
            vessels_df_1d = vessels_df_1d.sort_values('order').drop(columns='order').reset_index(drop=True)
        
        
        #XXX Deal with volume_sum module (if present)
        # idxBVsum_list = vessels_df_0d.index[vessels_df_0d["vessel_type"]=='volume_sum']
        idxBVsum_list = vessels_df_0d.index[vessels_df_0d["vessel_type"].str.startswith(('volume_sum','sum_blood_vol'))]
        if len(idxBVsum_list)==0:
            print("NO volume_sum module found.")
            pass
        
        elif len(idxBVsum_list)>1:
            idxBVsumTot = -1
            for k in range(len(idxBVsum_list)):
                idxBVsum = idxBVsum_list[k]
                inp_vess_BVsum =  vessels_df_0d.at[idxBVsum,"inp_vessels"].split()
                out_vess_BVsum =  vessels_df_0d.at[idxBVsum,"out_vessels"].split()
                if len(inp_vess_BVsum)>0 and len(out_vess_BVsum)==0:
                    idxBVsumTot = idxBVsum
                    break
            print(f"Multiple volume_sum modules found. Total volume_sum module found with input modules : {vessels_df_0d.at[idxBVsumTot,'inp_vessels']}")
            N1d = vessels_df_1d.shape[0]
            if N1d>0:
                name_BVsumTot =  vessels_df_0d.at[idxBVsumTot,"name"]
                vessels_df_0d.loc[len(vessels_df_0d)] = ['volume_sum_1D', 'nn', 'FV1D_volume_sum', '', name_BVsumTot]

                for k in range(len(idxBVsum_list)):
                    idxBVsum = idxBVsum_list[k]
                    name_BVsum =  vessels_df_0d.at[idxBVsum,"name"]

                    inp_vess_BVsum =  vessels_df_0d.at[idxBVsum,"inp_vessels"].split()
                    inp_vess_BVsum_new = []

                    for i in range(len(inp_vess_BVsum)):
                        inp_vess = inp_vess_BVsum[i]
                        inp_vess_is_0d = -1
                        for j in range(vessels_df_0d.shape[0]):
                            if vessels_df_0d.at[j,"name"]==inp_vess:
                                inp_vess_is_0d = 1
                                break
                        if inp_vess_is_0d==1:
                            inp_vess_BVsum_new.append(inp_vess)
                        else:
                            for j in range(vessels_df_1d.shape[0]):
                                if vessels_df_1d.at[j,"name"]==inp_vess:
                                    out_vess_1d =  vessels_df_1d.at[j,"out_vessels"].split()
                                    if name_BVsum in out_vess_1d:
                                        out_vess_1d = [out_vess for out_vess in out_vess_1d if out_vess != name_BVsum]
                                        vessels_df_1d.at[j,"out_vessels"] = ' '.join(out_vess_1d)
                                        break

                    if idxBVsum == idxBVsumTot:
                        inp_vess_BVsum_new.append('volume_sum_1D')
                    vessels_df_0d.at[idxBVsum,"inp_vessels"] = ' '.join(inp_vess_BVsum_new)

                idxBVsum_drop = []
                for k in range(len(idxBVsum_list)):
                    idxBVsum = idxBVsum_list[k]
                    name_BVsum =  vessels_df_0d.at[idxBVsum,"name"]
                    inp_vess_BVsum =  vessels_df_0d.at[idxBVsum,"inp_vessels"].split()

                    if len(inp_vess_BVsum)==0:
                        inp_vess_BVsumTot =  vessels_df_0d.at[idxBVsumTot,"inp_vessels"].split()
                        inp_vess_BVsumTot = [inp_vess for inp_vess in inp_vess_BVsumTot if inp_vess != name_BVsum]
                        vessels_df_0d.at[idxBVsumTot,"inp_vessels"] = ' '.join(inp_vess_BVsumTot)
                        idxBVsum_drop. append(idxBVsum)

                if len(idxBVsum_drop)>0:
                    vessels_df_0d = vessels_df_0d.drop(index=idxBVsum_drop).reset_index(drop=True)

        elif len(idxBVsum_list)==1:
            idxBVsum = idxBVsum_list[0]
            print(f"volume_sum module found with input modules : {vessels_df_0d.at[idxBVsum,'inp_vessels']}")
            N1d = vessels_df_1d.shape[0]
            if N1d>0:
                name_BVsum =  vessels_df_0d.at[idxBVsum,"name"]
                vessels_df_0d.loc[len(vessels_df_0d)] = ['volume_sum_1D', 'nn', 'FV1D_volume_sum', '', name_BVsum]

                inp_vess_BVsum =  vessels_df_0d.at[idxBVsum,"inp_vessels"].split()
                inp_vess_BVsum_new = []

                for i in range(len(inp_vess_BVsum)):
                    inp_vess = inp_vess_BVsum[i]
                    inp_vess_is_0d = -1
                    for j in range(vessels_df_0d.shape[0]):
                        if vessels_df_0d.at[j,"name"]==inp_vess:
                            inp_vess_is_0d = 1
                            break
                    if inp_vess_is_0d==1:
                        inp_vess_BVsum_new.append(inp_vess)
                    else:
                        for j in range(vessels_df_1d.shape[0]):
                            if vessels_df_1d.at[j,"name"]==inp_vess:
                                out_vess_1d =  vessels_df_1d.at[j,"out_vessels"].split()
                                if name_BVsum in out_vess_1d:
                                    out_vess_1d = [out_vess for out_vess in out_vess_1d if out_vess != name_BVsum]
                                    vessels_df_1d.at[j,"out_vessels"] = ' '.join(out_vess_1d)
                                    break

                inp_vess_BVsum_new.append('volume_sum_1D')
                vessels_df_0d.at[idxBVsum,"inp_vessels"] = ' '.join(inp_vess_BVsum_new)


        #XXX TODO deal with K_tube modules (if present)
        for i in range(vessels_df_1d.shape[0]):
            vess1d = vessels_df_1d.at[i,"name"]
            # inp_vessels = vessels_df_1d.at[i,"inp_vessels"].split()
            out_vessels = vessels_df_1d.at[i,"out_vessels"].split()
            for out_vess in out_vessels:
                if out_vess.startswith("K_tube_"):
                    out_vessels.remove(out_vess)
                    vessels_df_1d.at[i,"out_vessels"] = ' '.join(out_vessels)
                    found_K_mod = False
                    for j in range(vessels_df_0d.shape[0]):
                        if vessels_df_0d.at[j,"name"]==out_vess:
                            if (vessels_df_0d.at[j,"BC_type"]=="nn" and vessels_df_0d.at[j,"inp_vessels"].split()[0]==vess1d):
                                print(f"WARNING: found {out_vess} module coupled to FV1D_vessel {vess1d}. Removing it for now as tube law of 1D vessels is completely managed in 1D model solver.")
                                vessels_df_0d.drop(index=j, inplace=True)
                                vessels_df_0d = vessels_df_0d.reset_index(drop=True)
                                found_K_mod = True
                                break
                            else:
                                print(f"ERROR: wrong BC_type for {out_vess} module OR {out_vess} module incorrectly connected to FV1D_vessel {vess1d}. Check your vessel array file.")
                                exit()
                    if not found_K_mod:
                        print(f"ERROR: entry for {out_vess} module coupled to FV1D_vessel {vess1d} NOT found. Check your vessel array file.")
                        exit()


        self.conn_1d_0d_info = {}
        N1d0d = 0
        
        #XXX For now, only one-1d-to-one-0d connections are considered and implemented
        #XXX TODO Make this more general
        for i in range(vessels_df_1d.shape[0]):
            vess1d = vessels_df_1d.at[i,"name"]
            inp_vessels = vessels_df_1d.at[i,"inp_vessels"].split()
            out_vessels = vessels_df_1d.at[i,"out_vessels"].split()

            if len(inp_vessels)>0:
                for inp_vess in inp_vessels:
                    idx1d = vessels_df_1d.index[vessels_df_1d["name"]==inp_vess]
                    if len(idx1d)==1:
                        pass
                    elif len(idx1d)==0:
                        idx0d = vessels_df_0d.index[vessels_df_0d["name"]==inp_vess]
                        if len(idx0d)==1:
                            idx0d_int = idx0d[0]
                            if (vessels_df_0d.at[idx0d_int,"vessel_type"].startswith(("inlet_flow","inlet_pressure"))
                                    and vessels_df_0d.at[idx0d_int,"BC_type"].startswith("nn")):

                                    if vessels_df_0d.at[idx0d_int,"vessel_type"].startswith("inlet_flow"):
                                        vessels_df_1d.at[i,"inp_vessels"] = 'input_flow_BC'
                                    elif vessels_df_0d.at[idx0d_int,"vessel_type"].startswith("inlet_pressure"):
                                        vessels_df_1d.at[i,"inp_vessels"] = 'input_pressure_BC'

                                    vessels_df_0d.drop(index=idx0d_int, inplace=True)
                                    vessels_df_0d = vessels_df_0d.reset_index(drop=True)

                            else:
                                # inp_vessels_0d = vessels_df_0d.at[idx0d[0],"inp_vessels"].split()
                                out_vessels_0d = vessels_df_0d.at[idx0d_int,"out_vessels"].split()
                                BC_type_0d = vessels_df_0d.at[idx0d_int,"BC_type"]
                                BC_type_1d = vessels_df_1d.at[i,"BC_type"]
                                if vess1d in out_vessels_0d:
                                    port_0d = -1 
                                    if BC_type_0d.startswith(("pp","vp")):
                                        bc_0d = 1 # pressure bc
                                        port_0d = 0 # flow port
                                        # if BC_type_1d=="nn":
                                        #     BC_type_1d = "vv"
                                    elif BC_type_0d.startswith(("pv","vv")):
                                        bc_0d = 0 # flow bc
                                        port_0d = 1 # pressure port
                                        # if BC_type_1d=="nn":
                                        #     BC_type_1d = "pv" 
                                    elif BC_type_0d.startswith("nn"):
                                        pass
                                    self.conn_1d_0d_info[str(N1d0d+1)] = {"vess1d_idx": i,
                                                                        "vess1d_bc_in0_or_out1": 0,
                                                                        "vess0d_idx": int(idx0d_int),
                                                                        "cellml_idx": -1,
                                                                        "cellml_bc_in0_or_out1": 1,
                                                                        "cellml_bc_flow0_or_press1": bc_0d,
                                                                        "port_idx": -1,
                                                                        "port_flow0_or_press1": port_0d,
                                                                        "port_state0_or_var1": -1,
                                                                        "R_T_variable_idx": -1}
                                        
                                    found_idx1d = -1
                                    for j in range(vessels_df_0d.shape[0]):
                                        if vessels_df_0d.at[j,"name"]==vess1d:
                                            found_idx1d = j
                                            break
                                    if found_idx1d == -1:
                                        vessels_df_0d.loc[len(vessels_df_0d)] = [vess1d,
                                                                                BC_type_1d,
                                                                                vessels_df_1d.at[i,"vessel_type"],
                                                                                vessels_df_1d.at[i,"inp_vessels"],
                                                                                '']
                                    else:
                                        if vessels_df_0d.at[found_idx1d,"inp_vessels"]=='':
                                            vessels_df_0d.at[found_idx1d,"inp_vessels"] = vessels_df_1d.at[i,"inp_vessels"]

                                    N1d0d += 1
                                else:
                                    print(f"ERROR :: {inp_vess} found in inp_vessels of 1D vessel {vess1d}, but {vess1d} not found in out_vessels_0d of 0D vessel {inp_vess}")
                                    exit()
                        elif len(idx0d)==0:
                            print(f"ERROR :: no index found for {inp_vess} in 1D or 0D vessel array")
                            exit()
                        elif len(idx0d)>1:
                            print(f"ERROR :: more than one index found for {inp_vess} in 0D vessel array")
                            exit()
                    else: # TODO
                        print(f"ERROR :: more than one index found for {inp_vess} in 1D vessel array")
                        exit()

            if len(out_vessels)>0:
                for out_vess in out_vessels:
                    idx1d = vessels_df_1d.index[vessels_df_1d["name"]==out_vess]
                    if len(idx1d)==1:
                        pass
                    elif len(idx1d)==0:
                        idx0d = vessels_df_0d.index[vessels_df_0d["name"]==out_vess]
                        if len(idx0d)==1:
                            idx0d_int = idx0d[0]
                            if (vessels_df_0d.at[idx0d_int,"vessel_type"].startswith(("outlet_flow","outlet_pressure"))
                                    and vessels_df_0d.at[idx0d_int,"BC_type"].startswith("nn")):

                                    if vessels_df_0d.at[idx0d_int,"vessel_type"].startswith("outlet_flow"):
                                        vessels_df_1d.at[i,"out_vessels"] = 'output_flow_BC'
                                    elif vessels_df_0d.at[idx0d_int,"vessel_type"].startswith("outlet_pressure"):
                                        vessels_df_1d.at[i,"out_vessels"] = 'output_pressure_BC'

                                    vessels_df_0d.drop(index=idx0d_int, inplace=True)
                                    vessels_df_0d = vessels_df_0d.reset_index(drop=True)
                            
                            else:
                                inp_vessels_0d = vessels_df_0d.at[idx0d_int,"inp_vessels"].split()
                                # out_vessels_0d = vessels_df_0d.at[idx0d[0],"out_vessels"].split()
                                BC_type_0d = vessels_df_0d.at[idx0d_int,"BC_type"]
                                BC_type_1d = vessels_df_1d.at[i,"BC_type"]
                                if vess1d in inp_vessels_0d:
                                    port_0d = -1
                                    if BC_type_0d.startswith(("pp","pv")):
                                        bc_0d = 1 # pressure bc
                                        port_0d = 0 # flow port
                                        # if BC_type_1d=="nn":
                                        #     BC_type_1d = "vv"
                                    elif BC_type_0d.startswith(("vp","vv")):
                                        bc_0d = 0 # flow bc
                                        port_0d = 1 # pressure port 
                                        # if BC_type_1d=="nn":
                                        #     BC_type_1d = "vp"
                                    elif BC_type_0d.startswith("nn"):
                                        pass
                                    self.conn_1d_0d_info[str(N1d0d+1)] = {"vess1d_idx": i,
                                                                        "vess1d_bc_in0_or_out1": 1,
                                                                        "vess0d_idx": int(idx0d_int),
                                                                        "cellml_idx": -1,
                                                                        "cellml_bc_in0_or_out1": 0,
                                                                        "cellml_bc_flow0_or_press1": bc_0d,
                                                                        "port_idx": -1,
                                                                        "port_flow0_or_press1": port_0d,
                                                                        "port_state0_or_var1": -1,
                                                                        "R_T_variable_idx": -1}
                                    
                                    found_idx1d = -1
                                    for j in range(vessels_df_0d.shape[0]):
                                        if vessels_df_0d.at[j,"name"]==vess1d:
                                            found_idx1d = j
                                            break
                                    if found_idx1d == -1:
                                        vessels_df_0d.loc[len(vessels_df_0d)] = [vessels_df_1d.at[i,"name"],
                                                                                BC_type_1d,
                                                                                vessels_df_1d.at[i,"vessel_type"],
                                                                                '',
                                                                                vessels_df_1d.at[i,"out_vessels"]]
                                    else:
                                        if vessels_df_0d.at[found_idx1d,"out_vessels"]=='':
                                            vessels_df_0d.at[found_idx1d,"out_vessels"] = vessels_df_1d.at[i,"out_vessels"]
                                    
                                    N1d0d += 1
                                else: 
                                    print(f"ERROR :: {out_vess} found in out_vessels of 1D vessel {vess1d}, but {vess1d} not found in inp_vessels_0d of 0D vessel {out_vess}")
                                    exit()
                        elif len(idx0d)==0:
                            print(f"ERROR :: no index found for {out_vess} in 1D or 0D vessel array")
                            exit()
                        elif len(idx0d)>1:
                            print(f"ERROR :: more than one index found for {out_vess} in 0D vessel array")
                            exit()
                    else:
                        print(f"ERROR :: more than one index found for {out_vess} in 1D vessel array")
                        exit()

        print(f"Number of one-to-one 1D-0D connections: {N1d0d}")
        
        # Then, add 'global' connections, such as the one for volume sum
        # here we could add other 'special' global connections
        # we are assuming that all these special connections are stored at the of the self.conn_1d_0d_info structure and corresponding file
        for i in range(vessels_df_0d.shape[0]):
            if vessels_df_0d.at[i,"vessel_type"]=='FV1D_volume_sum':
                # sumBV0d = vessels_df_0d.at[i,"out_vessels"].split()[0]
                # idx0d = vessels_df_0d.index[vessels_df_0d["name"]==sumBV0d]
                self.conn_1d_0d_info[str(N1d0d+1)] = {"vess1d_idx": -1,
                                                    "vess1d_bc_in0_or_out1": -1,
                                                    "vess0d_idx": i,
                                                    "cellml_idx": -1,
                                                    "cellml_bc_in0_or_out1": -1,
                                                    "cellml_bc_flow0_or_press1": -1,
                                                    "port_idx": -1,
                                                    "port_flow0_or_press1": -1,
                                                    "port_state0_or_var1": -1,
                                                    "R_T_variable_idx": -1,
                                                    "port_volume_sum": 1}
                N1d0d += 1
        
        print(f"Number of global 1D-0D connections: {N1d0d}")
        # if N1d0d>0:
        #     print(self.conn_1d_0d_info)
        
        idx_last = self.vessel_filename.rfind("vessel_array")
        json_filename = self.vessel_filename[:idx_last] + "coupler1d0d_unfinished.json"
        with open(json_filename, "w") as f:
            json.dump(self.conn_1d_0d_info, f, indent=4)

        # print("0D vessel array", vessels_df_0d.head())
        # print("1D vessel array", vessels_df_1d.head())
        # print("\n")
        
        vessels_df_0d.to_csv(self.vessel_filename_0d, index=None, header=True) # or index=False
        vessels_df_1d.to_csv(self.vessel_filename_1d, index=None, header=True)



    def load_model(self):
        # TODO if file ending is csv. elif file ending is json
        # TODO create a json_parser
        if self.vessel_filename_0d is None:
            vessels_df = self.csv_parser.get_data_as_dataframe_multistrings(self.vessel_filename, True)
        else:
            self.split_0d_1d_vessel_array()
            vessels_df = self.csv_parser.get_data_as_dataframe_multistrings(self.vessel_filename_0d, True)
        

        # TODO remove the below:
        #  Temporarily we add a pulmonary system if there isnt one defined, this should be defined by
        #   the user but we include this to improve backwards compatitibility.

        # TODO This should check if the vessel_type is heart, not the name
        #  we should be able to call the heart module whatever we want
        if len(vessels_df.loc[vessels_df["name"] == 'heart']) == 1:
            if len(vessels_df.loc[vessels_df["name"] == 'heart'].out_vessels.values[0]) < 2:
                # if the heart only has one output we assume it doesn't have an output to a pulmonary artery
                # add pulmonary vein and artery to df
                vessels_df.loc[vessels_df.index.max()+1] = ['par', 'vp', 'arterial_simple', ['heart'], ['pvn']]
                vessels_df.loc[vessels_df.index.max()+1] = ['pvn', 'vp', 'arterial_simple', ['par'], ['heart']]
                # add pulmonary artery (par) to output of heart and pvn to input
                vessels_df.loc[vessels_df["name"] == 'heart'].out_vessels.values[0].append('par')
                vessels_df.loc[vessels_df["name"] == 'heart'].inp_vessels.values[0].append('pvn')
        elif len(vessels_df.loc[vessels_df["name"] == 'heart']) == 0:
            pass
        else:
            print('cannot have 2 hearts or more, we dont model octopii')
            exit()


        module_df = self.json_parser.json_to_dataframe_with_user_dir(self.module_config_dir, self.module_config_user_dir, self.external_modules_dir)
        
        # Check for repeated entries of vessel_type and BC_type in module_df
        duplicates = module_df[module_df.duplicated(subset=["vessel_type", "BC_type"], keep=False)]
        if not duplicates.empty:
            print("ERROR: Repeated entries of vessel_type and BC_type found in module_config.json:")
            print(duplicates)
            exit()
         
        # add module info to each row of vessel array
        self.json_parser.append_module_config_info_to_vessel_df(vessels_df, module_df)

        ports_columns = ["entrance_ports", "exit_ports", "general_ports"]
        for col in ports_columns:
            vessels_df[col] = vessels_df[col].apply(lambda x: [] if x == "None" else x)

        # TODO change to using a pandas dataframe
        parameters_array_orig = self.csv_parser.get_data_as_nparray(self.parameter_filename, True)
        # Reduce parameters_array so that it only includes the required parameters for
        # this vessel_array.
        # This will output True if all the required parameters have been defined and
        # False if they have not.
        # TODO change reduce_parameters_array to be automatic with respect to the modules
        parameters_array = self.__reduce_parameters_array(parameters_array_orig, vessels_df)
        # this vessel_array
        if self.parameter_id_dir:
            param_id_name_and_vals, param_id_date = self.csv_parser.get_param_id_params_as_lists_of_tuples(
                self.parameter_id_dir)
        else:
            param_id_name_and_vals = None
            param_id_date= None

        model_0D = CVS0DModel(vessels_df,parameters_array,
                              param_id_name_and_vals=param_id_name_and_vals,
                              param_id_date=param_id_date)

        # get the allowable types from the modules_config.json file
        model_0D.possible_vessel_BC_types = list(set(list(zip(module_df["vessel_type"].to_list(), module_df["BC_type"].to_list()))))
        
        if self.parameter_id_dir:
            check_list = [LumpedBCVesselCheck(), LumpedPortVariableCheck(), LumpedIDParamsCheck()]
        else:
            check_list = [LumpedBCVesselCheck(), LumpedPortVariableCheck()]

        checker = LumpedCompositeCheck(check_list=check_list)
        checker.execute(model_0D)

        return model_0D

    def __reduce_parameters_array(self, parameters_array_orig, vessels_df):
        # TODO get the required params from the BG modules
        #  maybe have a config file for each module that specifies the cellml model and the required constants
        required_params = []
        num_params = 0
        # Add pulmonary parameters # TODO put this into the for loop when pulmonary vessels are modules
        # TODO include units and model_environment in the appended item so they can be included
        for vessel_tup in vessels_df.itertuples():
            # TODO check that removing this doesn't break anything
            # elif vessel_tup.vessel_type == 'terminal' or vessel_tup.vessel_type == 'terminal2':
            #     str_addon = re.sub('_T$', '', f'_{vessel_tup.name}')
            #     module = 'systemic'
            str_addon = f'_{vessel_tup.name}'
            # add str_addon to param name from module_config if constant
            if (vessel_tup.variables_and_units is None 
                or vessel_tup.variables_and_units=='None' 
                or len(vessel_tup.variables_and_units) == 0):
                continue

            required_params += [(vessel_tup.variables_and_units[i][0] + str_addon,
                                 vessel_tup.variables_and_units[i][1],vessel_tup.variables_and_units[i][3])  for
                                   i in range(len(vessel_tup.variables_and_units)) if
                                   vessel_tup.variables_and_units[i][3] in ['constant']]
            
            # add parameter if it is set as boundary_condition
            required_params += [(vessel_tup.variables_and_units[i][0] + str_addon,
                                 vessel_tup.variables_and_units[i][1],vessel_tup.variables_and_units[i][3])  for
                                   i in range(len(vessel_tup.variables_and_units)) if
                                   vessel_tup.variables_and_units[i][3] in ['boundary_condition']]

            # dont add str_addon if global_constant
            required_params += [(vessel_tup.variables_and_units[i][0],
                                 vessel_tup.variables_and_units[i][1],vessel_tup.variables_and_units[i][3])  for
                                i in range(len(vessel_tup.variables_and_units)) if
                                vessel_tup.variables_and_units[i][3] in ['global_constant']]
            # new_global_params = [(vessel_tup.variables_and_units[i][0],
            #                      vessel_tup.variables_and_units[i][1], module)  for
            #                      i in range(len(vessel_tup.variables_and_units)) if
            #                      vessel_tup.variables_and_units[i][3] == 'global_constant' and
            #                      vessel_tup.variables_and_units[i][0] not in required_params]
            # required_params += new_global_params
        # The below params are no longer in the params files.
        # # append global parameters
        # required_params.append([f'beta_g', 'dimensionless', 'systemic'])
        # required_params.append([f'gain_int', 'dimensionless', 'systemic'])
        # num_params += 2

        required_params_unique = []
        for entry in required_params:
            if entry not in required_params_unique:
                required_params_unique.append(entry)
        required_params = np.array(required_params_unique)

        parameters_list = []

        for idx, param_tuple in enumerate(required_params):
            actually_exit = False
            try:
                new_entry = parameters_array_orig[np.where(parameters_array_orig["variable_name"] ==
                                                                       param_tuple[0])][0]
                new_entry = [item for item in new_entry]
                if new_entry[1] != param_tuple[1]:
                    print('')
                    print(f'ERROR: units of {new_entry[1]} in parameters.csv file does not \n'
                          f'match with units of {param_tuple[1]} in module_config.json file \n'
                          f'for param {new_entry[0]}, exiting \n')
                    print('')
                    actually_exit = True
                    exit()

                new_entry.insert(2, param_tuple[2])
                parameters_list.append(new_entry)
                # overwrite 2 index with local or global, it doesn't matter where the
            except:
                # the other entries apart from name in this row are left empty
                new_entry = ([param_tuple[0], param_tuple[1], param_tuple[2],
                                     'EMPTY_MUST_BE_FILLED', 'EMPTY_MUST_BE_FILLED'])
                parameters_list.append(new_entry)
                if actually_exit:
                    exit()
        if len(parameters_list) == 0:
            return np.empty((len(parameters_list)),
                                    dtype=[('variable_name', 'U80'), ('units', 'U80'),('const_type', 'U80'),
                                           ('value', 'U80'), ('data_reference', 'U80')])
        elif len(set([len(a) for a in parameters_list])) != 1:
            print('parameters rows are of non consistent length, exiting')
            exit()
        parameters_array = np.empty((len(parameters_list)),
                                    dtype=[('variable_name', 'U80'), ('units', 'U80'),('const_type', 'U80'),
                                           ('value', 'U80'), ('data_reference', 'U80')])
        parameters_array['variable_name'] = np.array(parameters_list)[:,0]
        parameters_array['units'] = np.array(parameters_list)[:,1]
        parameters_array['const_type'] = np.array(parameters_list)[:,2]
        parameters_array['value'] = np.array(parameters_list)[:,3]
        parameters_array['data_reference'] = np.array(parameters_list)[:,4]
        return parameters_array





