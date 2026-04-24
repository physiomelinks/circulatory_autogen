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
import json
import csv
from pathlib import Path


def convert_0d_to_1d(model, folder_0d, param_file_0d, folder_hyb=None, vess_1d_list=[]):

    if len(vess_1d_list) == 0:
        print(f"Warning :: No vessels to convert from 0D to 1D for model {model}. Use fully 0D model input files.")
        return

    folder_0d = Path(folder_0d)

    if folder_hyb is None:
        folder_hyb = folder_0d
    
    folder_hyb = Path(folder_hyb)
    if not os.path.exists(folder_hyb):
        os.makedirs(folder_hyb)
    
    df_vess = pd.read_csv(folder_0d / f"{model}_0d_vessel_array.csv")
    df_params = pd.read_csv(folder_0d / param_file_0d)

    n1d = len(vess_1d_list)
    nV = df_vess.shape[0]

    idx_to_drop = []
    for i in range(n1d):
        vess = vess_1d_list[i]
        idxs = df_vess.index[df_vess['name'] == vess].tolist()
        if len(idxs) == 0:
            sys.exit(f"Error :: Vessel {vess} not found in 0D vessel array for model {model}.")

        idxV = idxs[0]
        df_vess.at[idxV, 'BC_type'] = 'nn'
        df_vess.at[idxV, 'vessel_type'] = 'FV1D_vessel'

        out_vess = df_vess.at[idxV, 'out_vessels'].split()
        if 'K_tube_'+vess in out_vess:
            out_vess.remove('K_tube_'+vess)

            for j in range(nV):
                if df_vess.at[j, 'name'] == 'K_tube_'+vess:
                    idx_to_drop.append(j)
                    break
        df_vess.at[idxV, 'out_vessels'] = ' '.join(out_vess)

    df_vess = df_vess.drop(index=idx_to_drop)
    df_vess.reset_index(drop=True, inplace=True)

    nV = df_vess.shape[0]
    for j in range(nV):
        if df_vess.at[j, 'vessel_type']=='FV1D_vessel':
            vess1d = df_vess.at[j, 'name']
            inp_vess = df_vess.at[j, 'inp_vessels'].split()
            out_vess = df_vess.at[j, 'out_vessels'].split()
            
            for vess in inp_vess:
                idxs = df_vess.index[df_vess['name'] == vess].tolist()
                if len(idxs) == 0:
                    sys.exit(f"Error :: Input vessel {vess} not found for 1D vessel {vess1d}.")

                idxV = idxs[0]
                if df_vess.at[idxV, 'vessel_type'] != 'FV1D_vessel':
                    if df_vess.at[idxV, 'BC_type'].startswith(('vp', 'pp')):
                        if vess=='heart':
                            if vess1d=='par':
                                df_params.loc[df_params.shape[0]] = ['u_par_'+vess, 'J_per_m3', 0.0, 'WONT_BE_USED']
                            else: # aortic root
                                df_params.loc[df_params.shape[0]] = ['u_root_'+vess, 'J_per_m3', 0.0, 'WONT_BE_USED']
                        else:
                            df_params.loc[df_params.shape[0]] = ['u_out_'+vess, 'J_per_m3', 0.0, 'WONT_BE_USED']
                    elif df_vess.at[idxV, 'BC_type'].startswith(('pv', 'vv')):
                        df_params.loc[df_params.shape[0]] = ['v_out_'+vess, 'm3_per_s', 0.0, 'WONT_BE_USED']

            for vess in out_vess:
                idxs = df_vess.index[df_vess['name'] == vess].tolist()
                if len(idxs) == 0:
                    sys.exit(f"Error :: Output vessel {vess} not found for 1D vessel {vess1d}.")

                idxV = idxs[0]
                if df_vess.at[idxV, 'vessel_type'] != 'FV1D_vessel':
                    if df_vess.at[idxV, 'BC_type'].startswith(('pv', 'pp')):
                        df_params.loc[df_params.shape[0]] = ['u_in_'+vess, 'J_per_m3', 0.0, 'WONT_BE_USED']
                    elif df_vess.at[idxV, 'BC_type'].startswith(('vp', 'vv')):
                        if vess=='heart':
                            if vess1d=='pvn':
                                df_params.loc[df_params.shape[0]] = ['v_pvn_'+vess, 'm3_per_s', 0.0, 'WONT_BE_USED']
                            elif 'ivc' in vess1d:
                                df_params.loc[df_params.shape[0]] = ['v_ivc_'+vess, 'm3_per_s', 0.0, 'WONT_BE_USED']
                            elif 'svc' in vess1d:
                                df_params.loc[df_params.shape[0]] = ['v_svc_'+vess, 'm3_per_s', 0.0, 'WONT_BE_USED']
                        else:
                            df_params.loc[df_params.shape[0]] = ['v_in_'+vess, 'm3_per_s', 0.0, 'WONT_BE_USED']

    df_vess.to_csv(folder_hyb / f"{model}_hybrid_vessel_array.csv", index=False, header=True)
    df_params.to_csv(folder_hyb / f"{model}_hybrid_parameters.csv", index=False, header=True)

    print(f"Converted {n1d} vessels from 0D to 1D for model {model}.")
    print(f"Input hybrid model files saved successfully at {folder_hyb}")

    return


if __name__ == "__main__":
    
    model = "cvs_model"
    # model = "cvs_model_with_arm"
    
    folder_0d = "/hpc/bghi639/Software/VITAL_TrainingSchool2_Tutorials/"+model+"_0d/resources/"

    param_file_0d = model+"_0d_parameters.csv"

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