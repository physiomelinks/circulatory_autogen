'''
Created on 29/10/2021

@author: Finbar Argus, Gonzalo D. Maso Talou
'''


from libcuflynx.parsers.PrimitiveParsers import CSVFileParser, JSONFileParser
from libcuflynx.utilities.config_schemas import (load_expanded_vessel_records, load_module_array,
                                                 vessel_records_to_string_frame)
from libcuflynx.models.LumpedModels import CVS0DModel
from libcuflynx.utilities.parameter_names import host_model_names
from libcuflynx.utilities.parameters_file import parameters_array as rows_to_parameters_array, read_parameter_rows
from libcuflynx.checks.LumpedModelChecks import LumpedCompositeCheck, LumpedBCVesselCheck, LumpedIDParamsCheck, LumpedPortVariableCheck
import pandas as pd
import numpy as np
import json
import warnings
import re
import os

from libcuflynx.utilities.module_library import ModuleSources
from libcuflynx.utilities.parameterisations import reissue_warnings
from libcuflynx.utilities.vessel_bc import is_vessel_bc

# The keys every parameter row has. They are looked up by name, so a parameter may carry other
# keys (FTU_wCVS_parameters.json has a 'comp_env', for instance) without shifting the ones that
# matter -- see issue #159. 'const_type' is deliberately absent: it is supplied by the module
# config, not by the parameters file.
_REQUIRED_PARAMETER_COLUMNS = ('variable_name', 'units', 'value', 'data_reference')


def _vessel_variables(vessels_df):
    '''``{vessel name: {variable: kind}}`` from a module dataframe with module info.'''
    out = {}
    for row in vessels_df.itertuples():
        variables = row.variables_and_units
        if variables is None or isinstance(variables, str):
            variables = []
        out[row.name] = {v[0]: v[3] for v in variables if len(v) > 3}
    return out


def _with_model_names(parameters_array, vessels_df, source):
    '''``parameters_array`` with its ``<vessel>/<variable>`` names turned into model names
    (``parameter_names.host_model_names``, which refuses the older forms). Global names are
    checked only in a model without 1D vessels: the 1D generator reads its own rows (``rho``,
    ``mu``, vessel geometry) from the parameters file by name.'''
    fields = parameters_array.dtype.names or ()
    if 'variable_name' not in fields or not len(parameters_array):
        return parameters_array
    names = parameters_array['variable_name'].tolist()
    has_1d = 'module_type' in vessels_df and (vessels_df['module_type'] == 'FV1D_vessel').any()
    model_names = host_model_names(names, _vessel_variables(vessels_df), source,
                                   check_globals=not has_1d)
    width = max(len(n) for n in model_names)
    dtype = [(f, f'<U{max(width, parameters_array.dtype[f].itemsize // 4)}')
             if f == 'variable_name' else (f, parameters_array.dtype[f]) for f in fields]
    out = parameters_array.astype(dtype)
    out['variable_name'] = model_names
    return out


def merge_default_parameters(parameters_array, extra_param_rows):
    '''
    ``parameters_array`` (the structured array of a parameters CSV) with the rows of
    ``extra_param_rows`` (dicts keyed by column name) whose variable_name it does not already
    have appended -- so the host file's values win; of several extra rows with one name, the
    first is used. Columns a row lacks are left empty.
    '''
    if not extra_param_rows:
        return parameters_array
    fields = parameters_array.dtype.names or ()
    existing = set(parameters_array['variable_name'].tolist()) if 'variable_name' in fields else set()
    new_rows = []
    for row in extra_param_rows:
        if row['variable_name'] not in existing:
            existing.add(row['variable_name'])
            new_rows.append(row)
    if not new_rows:
        return parameters_array
    added = np.array([tuple(str(row.get(field, '')) for field in fields) for row in new_rows],
                     dtype=parameters_array.dtype)
    return np.concatenate([parameters_array, added])



def _append_vessel_row(df, name, module_subtype, module_type, inp_instances, out_instances, blank=''):
    """Append a vessel row by column name. Any other column (a module array's extra keys, e.g.
    ``parameterisation``) gets ``blank``: ``''`` in a string frame, ``[]`` in the list-form one. Appending
    a fixed 5-element list failed with "cannot set a row with mismatched columns" as soon as the
    array had one."""
    values = {'name': name, 'module_subtype': module_subtype, 'module_type': module_type,
              'inp_instances': inp_instances, 'out_instances': out_instances}
    row = [values.get(column, blank) for column in df.columns]
    df.loc[df.index.max() + 1 if len(df) else 0] = row

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
        # Built-in, module_config_user, external_modules_dir and module_library_dirs configs;
        # the CellML generator resolves its module files from the same ModuleSources.
        self.module_sources = ModuleSources(inp_data_dict)
        self.csv_parser = CSVFileParser()
        self.json_parser = JSONFileParser()

        self.conn_1d_0d_info = None


    def split_0d_1d_module_array(self, supermodule_registry=None, component_registry=None):
        '''
        Writes the 0D and 1D parts of the module array (JSON or CSV, either layout, with its
        supermodule instances expanded) to vessel_filename_0d and vessel_filename_1d, as CSV.
        Returns the default parameter rows of the supermodules and parameterisations (see
        config_schemas.load_module_array).
        '''
        records, extra_param_rows = load_expanded_vessel_records(self.vessel_filename,
                                                                 supermodule_registry,
                                                                 component_registry,
                                                                 self.parameter_filename)
        # strings, with the inp/out lists space-separated, as the code below expects
        vessels_df = vessel_records_to_string_frame(records)
        
        vessels_rows_0d = []
        vessels_rows_1d = []
        for _, row in vessels_df.iterrows():
            if row["module_type"]=="FV1D_vessel":
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
        # idxBVsum_list = vessels_df_0d.index[vessels_df_0d["module_type"]=='volume_sum']
        idxBVsum_list = vessels_df_0d.index[vessels_df_0d["module_type"].str.startswith(('volume_sum','sum_blood_vol'))]
        if len(idxBVsum_list)==0:
            print("NO volume_sum module found.")
            pass
        
        elif len(idxBVsum_list)>1:
            idxBVsumTot = -1
            for k in range(len(idxBVsum_list)):
                idxBVsum = idxBVsum_list[k]
                inp_vess_BVsum =  vessels_df_0d.at[idxBVsum,"inp_instances"].split()
                out_vess_BVsum =  vessels_df_0d.at[idxBVsum,"out_instances"].split()
                if len(inp_vess_BVsum)>0 and len(out_vess_BVsum)==0:
                    idxBVsumTot = idxBVsum
                    break
            print(f"Multiple volume_sum modules found. Total volume_sum module found with input modules : {vessels_df_0d.at[idxBVsumTot,'inp_instances']}")
            N1d = vessels_df_1d.shape[0]
            if N1d>0:
                name_BVsumTot =  vessels_df_0d.at[idxBVsumTot,"name"]
                _append_vessel_row(vessels_df_0d, 'volume_sum_1D', 'nn', 'FV1D_volume_sum', '', name_BVsumTot)

                for k in range(len(idxBVsum_list)):
                    idxBVsum = idxBVsum_list[k]
                    name_BVsum =  vessels_df_0d.at[idxBVsum,"name"]

                    inp_vess_BVsum =  vessels_df_0d.at[idxBVsum,"inp_instances"].split()
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
                                    out_vess_1d =  vessels_df_1d.at[j,"out_instances"].split()
                                    if name_BVsum in out_vess_1d:
                                        out_vess_1d = [out_vess for out_vess in out_vess_1d if out_vess != name_BVsum]
                                        vessels_df_1d.at[j,"out_instances"] = ' '.join(out_vess_1d)
                                        break

                    if idxBVsum == idxBVsumTot:
                        inp_vess_BVsum_new.append('volume_sum_1D')
                    vessels_df_0d.at[idxBVsum,"inp_instances"] = ' '.join(inp_vess_BVsum_new)

                idxBVsum_drop = []
                for k in range(len(idxBVsum_list)):
                    idxBVsum = idxBVsum_list[k]
                    name_BVsum =  vessels_df_0d.at[idxBVsum,"name"]
                    inp_vess_BVsum =  vessels_df_0d.at[idxBVsum,"inp_instances"].split()

                    if len(inp_vess_BVsum)==0:
                        inp_vess_BVsumTot =  vessels_df_0d.at[idxBVsumTot,"inp_instances"].split()
                        inp_vess_BVsumTot = [inp_vess for inp_vess in inp_vess_BVsumTot if inp_vess != name_BVsum]
                        vessels_df_0d.at[idxBVsumTot,"inp_instances"] = ' '.join(inp_vess_BVsumTot)
                        idxBVsum_drop. append(idxBVsum)

                if len(idxBVsum_drop)>0:
                    vessels_df_0d = vessels_df_0d.drop(index=idxBVsum_drop).reset_index(drop=True)

        elif len(idxBVsum_list)==1:
            idxBVsum = idxBVsum_list[0]
            print(f"volume_sum module found with input modules : {vessels_df_0d.at[idxBVsum,'inp_instances']}")
            N1d = vessels_df_1d.shape[0]
            if N1d>0:
                name_BVsum =  vessels_df_0d.at[idxBVsum,"name"]
                _append_vessel_row(vessels_df_0d, 'volume_sum_1D', 'nn', 'FV1D_volume_sum', '', name_BVsum)

                inp_vess_BVsum =  vessels_df_0d.at[idxBVsum,"inp_instances"].split()
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
                                out_vess_1d =  vessels_df_1d.at[j,"out_instances"].split()
                                if name_BVsum in out_vess_1d:
                                    out_vess_1d = [out_vess for out_vess in out_vess_1d if out_vess != name_BVsum]
                                    vessels_df_1d.at[j,"out_instances"] = ' '.join(out_vess_1d)
                                    break

                inp_vess_BVsum_new.append('volume_sum_1D')
                vessels_df_0d.at[idxBVsum,"inp_instances"] = ' '.join(inp_vess_BVsum_new)


        #XXX TODO deal with K_tube modules (if present)
        for i in range(vessels_df_1d.shape[0]):
            vess1d = vessels_df_1d.at[i,"name"]
            # inp_instances = vessels_df_1d.at[i,"inp_instances"].split()
            out_instances = vessels_df_1d.at[i,"out_instances"].split()
            for out_vess in out_instances:
                if out_vess.startswith("K_tube_"):
                    out_instances.remove(out_vess)
                    vessels_df_1d.at[i,"out_instances"] = ' '.join(out_instances)
                    found_K_mod = False
                    for j in range(vessels_df_0d.shape[0]):
                        if vessels_df_0d.at[j,"name"]==out_vess:
                            if (not is_vessel_bc(vessels_df_0d.at[j,"module_subtype"]) and vessels_df_0d.at[j,"inp_instances"].split()[0]==vess1d):
                                print(f"WARNING: found {out_vess} module coupled to FV1D_vessel {vess1d}. Removing it for now as tube law of 1D vessels is completely managed in 1D model solver.")
                                vessels_df_0d.drop(index=j, inplace=True)
                                vessels_df_0d = vessels_df_0d.reset_index(drop=True)
                                found_K_mod = True
                                break
                            else:
                                print(f"ERROR: wrong module_subtype for {out_vess} module OR {out_vess} module incorrectly connected to FV1D_vessel {vess1d}. Check your module array file.")
                                exit()
                    if not found_K_mod:
                        print(f"ERROR: entry for {out_vess} module coupled to FV1D_vessel {vess1d} NOT found. Check your module array file.")
                        exit()


        self.conn_1d_0d_info = {}
        N1d0d = 0
        
        #XXX For now, only one-1d-to-one-0d connections are considered and implemented
        #XXX TODO Make this more general
        for i in range(vessels_df_1d.shape[0]):
            vess1d = vessels_df_1d.at[i,"name"]
            inp_instances = vessels_df_1d.at[i,"inp_instances"].split()
            out_instances = vessels_df_1d.at[i,"out_instances"].split()

            if len(inp_instances)>0:
                for inp_vess in inp_instances:
                    idx1d = vessels_df_1d.index[vessels_df_1d["name"]==inp_vess]
                    if len(idx1d)==1:
                        pass
                    elif len(idx1d)==0:
                        idx0d = vessels_df_0d.index[vessels_df_0d["name"]==inp_vess]
                        if len(idx0d)==1:
                            idx0d_int = idx0d[0]
                            if (vessels_df_0d.at[idx0d_int,"module_type"].startswith(("inlet_flow","inlet_pressure"))
                                    and not is_vessel_bc(vessels_df_0d.at[idx0d_int,"module_subtype"])):

                                    if vessels_df_0d.at[idx0d_int,"module_type"].startswith("inlet_flow"):
                                        vessels_df_1d.at[i,"inp_instances"] = 'input_flow_BC'
                                    elif vessels_df_0d.at[idx0d_int,"module_type"].startswith("inlet_pressure"):
                                        vessels_df_1d.at[i,"inp_instances"] = 'input_pressure_BC'

                                    vessels_df_0d.drop(index=idx0d_int, inplace=True)
                                    vessels_df_0d = vessels_df_0d.reset_index(drop=True)

                            else:
                                # inp_instances_0d = vessels_df_0d.at[idx0d[0],"inp_instances"].split()
                                out_instances_0d = vessels_df_0d.at[idx0d_int,"out_instances"].split()
                                module_subtype_0d = vessels_df_0d.at[idx0d_int,"module_subtype"]
                                module_subtype_1d = vessels_df_1d.at[i,"module_subtype"]
                                if vess1d in out_instances_0d:
                                    port_0d = -1 
                                    if module_subtype_0d.startswith(("pp","vp")):
                                        bc_0d = 1 # pressure bc
                                        port_0d = 0 # flow port
                                        # if module_subtype_1d=="nn":
                                        #     module_subtype_1d = "vv"
                                    elif module_subtype_0d.startswith(("pv","vv")):
                                        bc_0d = 0 # flow bc
                                        port_0d = 1 # pressure port
                                        # if module_subtype_1d=="nn":
                                        #     module_subtype_1d = "pv" 
                                    elif module_subtype_0d.startswith("nn"):
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
                                        _append_vessel_row(vessels_df_0d, vess1d, module_subtype_1d,
                                                           vessels_df_1d.at[i,"module_type"],
                                                           vessels_df_1d.at[i,"inp_instances"], '')
                                    else:
                                        if vessels_df_0d.at[found_idx1d,"inp_instances"]=='':
                                            vessels_df_0d.at[found_idx1d,"inp_instances"] = vessels_df_1d.at[i,"inp_instances"]

                                    N1d0d += 1
                                else:
                                    print(f"ERROR :: {inp_vess} found in inp_instances of 1D vessel {vess1d}, but {vess1d} not found in out_instances_0d of 0D vessel {inp_vess}")
                                    exit()
                        elif len(idx0d)==0:
                            print(f"ERROR :: no index found for {inp_vess} in 1D or 0D module array")
                            exit()
                        elif len(idx0d)>1:
                            print(f"ERROR :: more than one index found for {inp_vess} in 0D module array")
                            exit()
                    else: # TODO
                        print(f"ERROR :: more than one index found for {inp_vess} in 1D module array")
                        exit()

            if len(out_instances)>0:
                for out_vess in out_instances:
                    idx1d = vessels_df_1d.index[vessels_df_1d["name"]==out_vess]
                    if len(idx1d)==1:
                        pass
                    elif len(idx1d)==0:
                        idx0d = vessels_df_0d.index[vessels_df_0d["name"]==out_vess]
                        if len(idx0d)==1:
                            idx0d_int = idx0d[0]
                            if (vessels_df_0d.at[idx0d_int,"module_type"].startswith(("outlet_flow","outlet_pressure"))
                                    and not is_vessel_bc(vessels_df_0d.at[idx0d_int,"module_subtype"])):

                                    if vessels_df_0d.at[idx0d_int,"module_type"].startswith("outlet_flow"):
                                        vessels_df_1d.at[i,"out_instances"] = 'output_flow_BC'
                                    elif vessels_df_0d.at[idx0d_int,"module_type"].startswith("outlet_pressure"):
                                        vessels_df_1d.at[i,"out_instances"] = 'output_pressure_BC'

                                    vessels_df_0d.drop(index=idx0d_int, inplace=True)
                                    vessels_df_0d = vessels_df_0d.reset_index(drop=True)
                            
                            else:
                                inp_instances_0d = vessels_df_0d.at[idx0d_int,"inp_instances"].split()
                                # out_instances_0d = vessels_df_0d.at[idx0d[0],"out_instances"].split()
                                module_subtype_0d = vessels_df_0d.at[idx0d_int,"module_subtype"]
                                module_subtype_1d = vessels_df_1d.at[i,"module_subtype"]
                                if vess1d in inp_instances_0d:
                                    port_0d = -1
                                    if module_subtype_0d.startswith(("pp","pv")):
                                        bc_0d = 1 # pressure bc
                                        port_0d = 0 # flow port
                                        # if module_subtype_1d=="nn":
                                        #     module_subtype_1d = "vv"
                                    elif module_subtype_0d.startswith(("vp","vv")):
                                        bc_0d = 0 # flow bc
                                        port_0d = 1 # pressure port 
                                        # if module_subtype_1d=="nn":
                                        #     module_subtype_1d = "vp"
                                    elif module_subtype_0d.startswith("nn"):
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
                                        _append_vessel_row(vessels_df_0d, vessels_df_1d.at[i,"name"], module_subtype_1d,
                                                           vessels_df_1d.at[i,"module_type"], '',
                                                           vessels_df_1d.at[i,"out_instances"])
                                    else:
                                        if vessels_df_0d.at[found_idx1d,"out_instances"]=='':
                                            vessels_df_0d.at[found_idx1d,"out_instances"] = vessels_df_1d.at[i,"out_instances"]
                                    
                                    N1d0d += 1
                                else: 
                                    print(f"ERROR :: {out_vess} found in out_instances of 1D vessel {vess1d}, but {vess1d} not found in inp_instances_0d of 0D vessel {out_vess}")
                                    exit()
                        elif len(idx0d)==0:
                            print(f"ERROR :: no index found for {out_vess} in 1D or 0D module array")
                            exit()
                        elif len(idx0d)>1:
                            print(f"ERROR :: more than one index found for {out_vess} in 0D module array")
                            exit()
                    else:
                        print(f"ERROR :: more than one index found for {out_vess} in 1D module array")
                        exit()

        print(f"Number of one-to-one 1D-0D connections: {N1d0d}")
        
        # Then, add 'global' connections, such as the one for volume sum
        # here we could add other 'special' global connections
        # we are assuming that all these special connections are stored at the of the self.conn_1d_0d_info structure and corresponding file
        for i in range(vessels_df_0d.shape[0]):
            if vessels_df_0d.at[i,"module_type"]=='FV1D_volume_sum':
                # sumBV0d = vessels_df_0d.at[i,"out_instances"].split()[0]
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
        
        idx_last = max(self.vessel_filename.rfind("module_array"), self.vessel_filename.rfind("vessel_array"))
        json_filename = self.vessel_filename[:idx_last] + "coupler1d0d_unfinished.json"
        with open(json_filename, "w") as f:
            json.dump(self.conn_1d_0d_info, f, indent=4)

        # print("0D module array", vessels_df_0d.head())
        # print("1D module array", vessels_df_1d.head())
        # print("\n")
        
        vessels_df_0d.to_csv(self.vessel_filename_0d, index=None, header=True) # or index=False
        vessels_df_1d.to_csv(self.vessel_filename_1d, index=None, header=True)
        return extra_param_rows



    def load_model(self):
        # Supermodule entries of the module configs. They never join the module dataframe:
        # their instances are expanded into prefixed submodules as the module array is read,
        # before anything below (the module-config join) sees it.
        supermodule_registry = self.module_sources.supermodule_registry()
        # Component entries with the config file each came from: a record's parameterisation
        # ("parameterisation", or the entry's default) is read from
        # parameterisations/ (or the older instances/) next to it
        # (utilities/parameterisations.py).
        component_registry = self.module_sources.component_registry()
        # The module array is JSON records or a CSV converted to the same records
        # (utilities/config_schemas.py). extra_param_rows are the default parameters under
        # the expanded names -- supermodule parameterisations and default_parameters, then
        # module parameterisations -- and are merged into the parameters below.
        # Parameterisations that set one global to different values warn; those warnings are held
        # until the host parameters are read, since a host value settles the conflict.
        with warnings.catch_warnings(record=True) as held:
            warnings.simplefilter('always')
            if self.vessel_filename_0d is None:
                vessels_df, extra_param_rows = load_module_array(self.vessel_filename,
                                                                 supermodule_registry,
                                                                 component_registry,
                                                                 self.parameter_filename)
            else:
                extra_param_rows = self.split_0d_1d_module_array(supermodule_registry,
                                                                 component_registry)
                vessels_df, _ = load_module_array(self.vessel_filename_0d)


        # a type several module sources define comes from the most specific one
        # (utilities/module_library.py), so each (module_type, module_subtype) is here once
        module_df = self.json_parser.json_files_to_dataframe(
            self.module_sources.config_files, exclude=self.module_sources.excluded_entries)
        duplicates = module_df[module_df.duplicated(subset=["module_type", "module_subtype"], keep=False)]
        if not duplicates.empty:
            raise ValueError('Repeated (module_type, module_subtype) entries in the module configs:\n'
                             + duplicates[['module_type', 'module_subtype']].to_string())
        component_keys = set(zip(module_df["module_type"], module_df["module_subtype"]))
        shadowed = sorted(key for key in supermodule_registry if key in component_keys)
        if shadowed:
            raise ValueError(f'{shadowed} are defined both as supermodules and as component '
                             f'modules in the module configs; a (module_type, module_subtype) can be '
                             f'only one of them.')
         
        # add module info to each row of module array
        self.json_parser.append_module_config_info_to_vessel_df(vessels_df, module_df)

        ports_columns = ["entrance_ports", "exit_ports", "general_ports"]
        for col in ports_columns:
            vessels_df[col] = vessels_df[col].apply(lambda x: [] if x == "None" else x)

        # the model's parameters file, nested JSON, as rows named <vessel>/<variable> (globals
        # by their plain names; utilities/parameters_file.py)
        parameters_array_orig = rows_to_parameters_array(read_parameter_rows(self.parameter_filename))
        # A parameter may be written <vessel>/<variable> (utilities/parameter_names.py); from
        # here on every name is the model's {variable}_{vessel}.
        parameters_array_orig = _with_model_names(parameters_array_orig, vessels_df,
                                                  self.parameter_filename)
        # a host value settles a conflict between parameterisations (written global/T or T)
        reissue_warnings(held, settled=set(parameters_array_orig['variable_name'].tolist()))
        # Supermodule and parameterisation parameters fill in whatever the parameters file does not set,
        # before the reduction, so everything downstream (generation, parameter id) sees them.
        parameters_array_orig = merge_default_parameters(parameters_array_orig, extra_param_rows)
        # Reduce parameters_array so that it only includes the required parameters for
        # this module_array.
        # This will output True if all the required parameters have been defined and
        # False if they have not.
        # TODO change reduce_parameters_array to be automatic with respect to the modules
        parameters_array = self.__reduce_parameters_array(parameters_array_orig, vessels_df)
        # this module_array
        if self.parameter_id_dir:
            param_id_name_and_vals, param_id_date = self.csv_parser.get_param_id_params_as_lists_of_tuples(
                self.parameter_id_dir)
        else:
            param_id_name_and_vals = None
            param_id_date= None

        model_0D = CVS0DModel(vessels_df,parameters_array,
                              param_id_name_and_vals=param_id_name_and_vals,
                              param_id_date=param_id_date)
        # every parameter, supermodule defaults included, before the reduction to this 0D model:
        # the 1D generator reads its vessels' parameters from here, not from the file
        model_0D.all_parameters_array = parameters_array_orig
        # {model name: the name the model's parameters file gives it}
        model_0D.parameter_file_names = self.parameter_file_names

        # get the allowable types from the modules_config.json file
        model_0D.possible_module_type_subtypes = list(set(list(zip(module_df["module_type"].to_list(), module_df["module_subtype"].to_list()))))
        
        if self.parameter_id_dir:
            check_list = [LumpedBCVesselCheck(), LumpedPortVariableCheck(), LumpedIDParamsCheck()]
        else:
            check_list = [LumpedBCVesselCheck(), LumpedPortVariableCheck()]

        checker = LumpedCompositeCheck(check_list=check_list)
        checker.execute(model_0D)

        return model_0D

    def __reduce_parameters_array(self, parameters_array_orig, vessels_df):
        self.parameter_file_names = {}
        # TODO get the required params from the BG modules
        #  maybe have a config file for each module that specifies the cellml model and the required constants
        required_params = []
        num_params = 0
        # Add pulmonary parameters # TODO put this into the for loop when pulmonary vessels are modules
        # TODO include units and model_environment in the appended item so they can be included
        for vessel_tup in vessels_df.itertuples():
            # TODO check that removing this doesn't break anything
            # elif vessel_tup.module_type == 'terminal' or vessel_tup.module_type == 'terminal2':
            #     str_addon = re.sub('_T$', '', f'_{vessel_tup.name}')
            #     module = 'systemic'
            str_addon = f'_{vessel_tup.name}'
            # add str_addon to param name from module_config if constant
            if (vessel_tup.variables_and_units is None 
                or vessel_tup.variables_and_units=='None' 
                or len(vessel_tup.variables_and_units) == 0):
                continue

            # how the model's parameters file names each one (<vessel>/<variable>, a global plain)
            for row in vessel_tup.variables_and_units:
                if row[3] in ('constant', 'boundary_condition'):
                    self.parameter_file_names[row[0] + str_addon] = f'{vessel_tup.name}/{row[0]}'
                elif row[3] == 'global_constant':
                    self.parameter_file_names[row[0]] = row[0]
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

        # Read each row by field NAME, not position (issue #159): a parameter may carry other
        # keys (resources/FTU_wCVS_parameters.json has 'comp_env'), which become fields of the
        # structured array too.
        missing_columns = [name for name in _REQUIRED_PARAMETER_COLUMNS
                           if name not in (parameters_array_orig.dtype.names or ())]
        if missing_columns:
            print('')
            print(f'ERROR: {self.parameter_filename} is missing the required column(s) '
                  f'{missing_columns}. \n'
                  f'Required columns are {list(_REQUIRED_PARAMETER_COLUMNS)}; the file has '
                  f'{list(parameters_array_orig.dtype.names or ())}. \n'
                  f'Columns are matched by header name, so any additional columns are ignored '
                  f'and their order does not matter, exiting \n')
            print('')
            exit()

        parameters_list = []

        for idx, param_tuple in enumerate(required_params):
            matches = parameters_array_orig[
                np.where(parameters_array_orig["variable_name"] == param_tuple[0])]
            if len(matches) == 0:
                # the other entries apart from name in this row are left empty
                parameters_list.append([param_tuple[0], param_tuple[1], param_tuple[2],
                                        'EMPTY_MUST_BE_FILLED', 'EMPTY_MUST_BE_FILLED'])
                continue

            row = matches[0]
            if row['units'] != param_tuple[1]:
                print('')
                print(f'ERROR: units of {row["units"]} in the parameters file does not \n'
                      f'match with units of {param_tuple[1]} in module_config.json file \n'
                      f'for param {row["variable_name"]}, exiting \n')
                print('')
                exit()

            # const_type comes from the module config (local or global), not from the CSV.
            parameters_list.append([row['variable_name'], row['units'], param_tuple[2],
                                    row['value'], row['data_reference']])
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





