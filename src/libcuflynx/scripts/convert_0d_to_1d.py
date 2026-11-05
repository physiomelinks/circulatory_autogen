#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 2026

@author: bghi639
"""

import numpy as np
import pandas as pd
import os
import sys
import csv
from pathlib import Path
from libcuflynx.utilities.config_schemas import (dump_vessel_records, read_module_array_records,
                                                 module_array_path, vessel_records_to_string_frame)
from libcuflynx.utilities.parameters_file import (get_leaf, leaf_from_row, name_path,
                                                  parameters_file_name, read_parameters_file,
                                                  set_leaf, write_parameters_file)
from libcuflynx.utilities.module_library import ModuleSources
from libcuflynx.generators.port_order import neighbour_port_index


def _write_json_records(source_path, source_records, df_vess, out_path):
    """The converted frame as JSON records: each source record (every key kept) with the
    frame's edited columns."""
    by_name = {r['name']: r for r in source_records}
    records = []
    for _, row in df_vess.iterrows():
        record = dict(by_name.get(row['name'], {}))
        for column in df_vess.columns:
            value = row[column]
            if column in ('inp_instances', 'out_instances'):
                record[column] = str(value).split()
            elif column in record or value not in ('', None):
                record[column] = value
        records.append(record)
    with open(out_path, 'w') as f:
        f.write(dump_vessel_records(records))


def _coupling_variable(df_vess, registry, vess, vess1d, side):
    """The variable of 0D module ``vess`` that a 1D vessel ``vess1d`` sets through the port
    joining them: the port is found from ``vess``'s module config (generators/port_order.py),
    and of its (flow, pressure) variables the one ``vess``'s boundary condition at that end
    takes. ``side`` is "exit" when ``vess`` feeds ``vess1d``, "entrance" when it is fed by it."""
    row = df_vess.loc[df_vess['name'] == vess].iloc[0]
    key = (row['module_type'], row['module_subtype'])
    if key not in registry:
        raise ValueError(f'No module config for {key} (module "{vess}"): the 1D coupling '
                         f'variable is read from its ports.')
    ports = registry[key][f'{side}_ports']
    neighbours = row['out_instances' if side == 'exit' else 'inp_instances'].split()
    facing = ('entrance_ports', 'general_ports') if side == 'exit' else ('exit_ports', 'general_ports')

    def port_types(name):
        rows = df_vess.loc[df_vess['name'] == name]
        if rows.empty:
            return []
        entry = registry.get((rows.iloc[0]['module_type'], rows.iloc[0]['module_subtype']), {})
        return [p['port_type'] for k in facing for p in entry.get(k, [])]

    idx = neighbour_port_index(ports, neighbours, vess1d, 'vessel_port', port_types)
    if idx is None:
        idx = next(i for i, p in enumerate(ports) if p['port_type'] == 'vessel_port')
    flow, pressure = ports[idx]['variables'][:2]
    bc_letter = row['module_subtype'][1 if side == 'exit' else 0]
    return (pressure, 'J_per_m3') if bc_letter == 'p' else (flow, 'm3_per_s')


def convert_0d_to_1d(model, folder_0d, param_file_0d, folder_hyb=None, vess_1d_list=[],
                     module_sources_inputs=None):
    """``module_sources_inputs``: the module source settings (``module_library_dirs``,
    ``use_builtin_modules``, ``external_modules_dir``) whose module configs give each 0D
    neighbour's coupling port; default the built-in modules."""

    if len(vess_1d_list) == 0:
        print(f"Warning :: No vessels to convert from 0D to 1D for model {model}. Use fully 0D model input files.")
        return

    folder_0d = Path(folder_0d)

    if folder_hyb is None:
        folder_hyb = folder_0d
    
    folder_hyb = Path(folder_hyb)
    if not os.path.exists(folder_hyb):
        os.makedirs(folder_hyb)
    
    # <model>_0d_module_array.json or .csv (or a PhLynx module array), as strings with the
    # inp/out lists space-separated; <model>_module_array.* when the 0D files carry no suffix
    source_path = module_array_path(str(folder_0d), f"{model}_0d")
    if not os.path.exists(source_path):
        source_path = module_array_path(str(folder_0d), model)
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Neither {model}_0d_module_array nor {model}_module_array (.json/.csv) "
                                f"found in {folder_0d}.")
    source_records = read_module_array_records(source_path)
    df_vess = vessel_records_to_string_frame(source_records)
    # the 0D model's parameters file (nested JSON); the rows below add the coupling variables
    parameters = read_parameters_file(folder_0d / param_file_0d)
    df_params = pd.DataFrame(columns=['variable_name', 'units', 'value', 'data_reference'])

    registry = ModuleSources(module_sources_inputs or {}).component_registry()

    n1d = len(vess_1d_list)
    nV = df_vess.shape[0]

    idx_to_drop = []
    for i in range(n1d):
        vess = vess_1d_list[i]
        idxs = df_vess.index[df_vess['name'] == vess].tolist()
        if len(idxs) == 0:
            sys.exit(f"Error :: Vessel {vess} not found in 0D module array for model {model}.")

        idxV = idxs[0]
        df_vess.at[idxV, 'module_subtype'] = 'nn'
        df_vess.at[idxV, 'module_type'] = 'FV1D_vessel'

        out_vess = df_vess.at[idxV, 'out_instances'].split()
        if 'K_tube_'+vess in out_vess:
            out_vess.remove('K_tube_'+vess)

            for j in range(nV):
                if df_vess.at[j, 'name'] == 'K_tube_'+vess:
                    idx_to_drop.append(j)
                    break
        df_vess.at[idxV, 'out_instances'] = ' '.join(out_vess)

    # The 1D solver reads a 1D vessel's values (and its K_tube_ module's) by flat name, l_parent,
    # E_K_tube_parent: they leave their vessel and become globals of the hybrid model.
    flat_vessels = list(vess_1d_list) + [df_vess.at[j, 'name'] for j in idx_to_drop]
    for vess in flat_vessels:
        for variable, leaf in (parameters.pop(vess, None) or {}).items():
            parameters.setdefault('global', {})[f'{variable}_{vess}'] = leaf

    df_vess = df_vess.drop(index=idx_to_drop)
    df_vess.reset_index(drop=True, inplace=True)

    nV = df_vess.shape[0]
    for j in range(nV):
        if df_vess.at[j, 'module_type']=='FV1D_vessel':
            vess1d = df_vess.at[j, 'name']
            inp_vess = df_vess.at[j, 'inp_instances'].split()
            out_vess = df_vess.at[j, 'out_instances'].split()
            
            for vess in inp_vess:
                idxs = df_vess.index[df_vess['name'] == vess].tolist()
                if len(idxs) == 0:
                    sys.exit(f"Error :: Input vessel {vess} not found for 1D vessel {vess1d}.")

                idxV = idxs[0]
                if df_vess.at[idxV, 'module_type'] != 'FV1D_vessel' and \
                        df_vess.at[idxV, 'module_subtype'].startswith(('vp', 'pp', 'pv', 'vv')):
                    variable, units = _coupling_variable(df_vess, registry, vess, vess1d, 'exit')
                    df_params.loc[df_params.shape[0]] = [vess+'/'+variable, units, 0.0, 'WONT_BE_USED']

            for vess in out_vess:
                idxs = df_vess.index[df_vess['name'] == vess].tolist()
                if len(idxs) == 0:
                    sys.exit(f"Error :: Output vessel {vess} not found for 1D vessel {vess1d}.")

                idxV = idxs[0]
                if df_vess.at[idxV, 'module_type'] != 'FV1D_vessel' and \
                        df_vess.at[idxV, 'module_subtype'].startswith(('vp', 'pp', 'pv', 'vv')):
                    variable, units = _coupling_variable(df_vess, registry, vess, vess1d, 'entrance')
                    df_params.loc[df_params.shape[0]] = [vess+'/'+variable, units, 0.0, 'WONT_BE_USED']

    # The FV 1D solver the 1D vessels run in: a row of its own (module FV1D_solver), from whose
    # module config the C++ generator writes the coupler's configuration.
    if (df_vess['module_type'] == 'FV1D_vessel').any() and not (df_vess['module_type'] == 'FV1D_solver').any():
        df_vess.loc[len(df_vess)] = {c: '' for c in df_vess.columns} | {
            'name': 'FV1D_solver', 'module_subtype': 'nn', 'module_type': 'FV1D_solver'}

    if str(source_path).endswith('.json'):
        # JSON in, JSON out: a CSV cannot hold a supermodule instance's per_submodule_* links
        # (or any other object-valued key), which the frame above leaves out
        _write_json_records(source_path, source_records, df_vess,
                            folder_hyb / f"{model}_hybrid_module_array.json")
    else:
        df_vess.to_csv(folder_hyb / f"{model}_hybrid_module_array.csv", index=False, header=True)
    # a coupling parameter added for two 1D vessels, or one the file has, is written once
    for row in df_params.to_dict('records'):
        path = name_path(row['variable_name'])
        if get_leaf(parameters, path) is None:
            set_leaf(parameters, path, leaf_from_row({k: str(v) for k, v in row.items()}))
    write_parameters_file(folder_hyb / parameters_file_name(f"{model}_hybrid"), parameters)

    print(f"Converted {n1d} vessels from 0D to 1D for model {model}.")
    print(f"Input hybrid model files saved successfully at {folder_hyb}")

    return


if __name__ == "__main__":
    
    model = "cvs_model"
    # model = "cvs_model_with_arm"
    
    folder_0d = "/hpc/bghi639/Software/VITAL_TrainingSchool2_Tutorials/"+model+"_0d/resources/"

    param_file_0d = model+"_0d_parameters.json"

    folder_hyb = "/hpc/bghi639/Software/VITAL_TrainingSchool2_Tutorials/"+model+"_hybrid/resources/"

    vess_1d_list = ['A_aorta_ascending_1', 'A_aorta_ascending_2', 'A_aorta_ascending_3', 'A_aorta_ascending_4',
                    'A_aortic_arch_1', 'A_aortic_arch_2', 'A_aortic_arch_3', 'A_aortic_arch_4',
                    'A_brachiocephalic_trunk',
                    'A_common_carotid_L', 'A_common_carotid_R',
                    'A_subclavian_L', 'A_subclavian_R',
                    'A_aorta_thoracic_1', 'A_aorta_thoracic_2', 'A_aorta_thoracic_3', 'A_aorta_thoracic_4', 'A_aorta_thoracic_5',
                    'A_aorta_abdominal_1', 'A_aorta_abdominal_2', 'A_aorta_abdominal_3', 'A_aorta_abdominal_4', 'A_aorta_abdominal_5', 'A_aorta_abdominal_6',
                    'A_common_iliac_L', 'A_common_iliac_R']
    
    if model.endswith("_with_arm"):
        vess_1d_list.extend(['A_axillary_L', 'A_brachial_L', 'A_radial_L', 'A_ulnar_L',
                             'A_superficial_palmar_arch_L_1', 'A_superficial_palmar_arch_L_2',
                             'A_comm_palmar_digital_L_1', 'A_comm_palmar_digital_L_2', 'A_comm_palmar_digital_L_3'])
        
    print(f"Converting vessels from 0D to 1D for model {model}:")
    print(vess_1d_list)

    convert_0d_to_1d(model, folder_0d, param_file_0d, folder_hyb, vess_1d_list)

    print("\n")
    print("Done!")