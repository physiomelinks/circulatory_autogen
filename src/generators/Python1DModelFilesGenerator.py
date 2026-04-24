#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 16:38:24 2024

@author: bghi639
"""

import numpy as np
import json
import csv
import pandas as pd
import os
import sys
import copy
from configparser import ConfigParser


def generate1DPythonModelFiles(df_vess, df_params, vess_file, nodes_file, names_file, conn_1d_0d_info):

    # units conversion factors between 0D solver (Kg,m,s) and 1D solver (g,cm,s)
    convLen = 1.0e+2 # length units conversion from [m] to [cm]
    convQ = 1e+06 # flow units conversion from [m3/s] to [ml/s]=[cm3/s]
    convP = 1.0e+1 # pressure units conversion from [J/m3]=[Kg/m/s2] to [dyne/cm2]
    convR = convP/convQ # 1.0e-05 # resistance units conversion from [J s/m6] to [dyne s/cm5]
    convC = convQ/convP # 1.0e+05 # compliance units conversion from [m6/J] to [cm5/dyne]
    # convI = 1e-05 # inductance units conversion from [J s2/m6] to [dyne s2/cm5]

    # mmHgDyncm2 = 1333.2238 # pressure units conversion from [mmHg] to [dyne/cm2] for convenience

    EeRef = 4.0e5 # [J/m3]


    nRow = df_vess.shape[0]
    nCol = df_vess.shape[1]

    print("1D vessel array file")
    print(type(df_vess))
    print('****')
    print(df_vess.dtypes)
    print('****')
    print(df_vess.head())
    print('****')
    print(df_vess.keys())
    print('****')
    print(nRow, nCol)
    print('***********************************************************************')

    nV = nRow
    print('Number of 1D vessels :', nV)
    print('****')

    vess = {}
    terms = []
    idxsV = []
    namesV = []

    for i in range(nV):
        nameV = df_vess.at[i,'name']
        idxsV.append(i)
        namesV.append(nameV)
        
        vess[i] = {'id' : i, 'name' : nameV,
                    'n1' : -1, 'n2' : -1,
                    'r_prox' : -1.0, 'r_dist' : -1.0, 'len' : -1.0, 
                    'Ee' : -1.0, 'Km' : -1.0, 'P0' : -1.0, 'Pext' : -1.0,
                    'type' : -1}
                
    # for i in range(nV):
    #     print(i, vess[i])
    # print('***********************************************************************')

    print('Writing vessel names file')
    df_names = pd.DataFrame(data={'vess_ID': idxsV, 'vess_name': namesV})
    df_names.to_csv(names_file, sep=',', index=False, header=True)  
    print('DONE!')
    print('****') 
    

    for i in range(nV):
        nameV = vess[i]['name']

        #XXX TODO improve this code below to differentiate between arterial and venous vessels, 
        # assuming that we do not have a 'vessel_type' in the CA vessel array file that can tell us this information
        if nameV.startswith(("A_","a_")) or any(sub in nameV for sub in ("art", "Art", "aort", "Aort")):
            vess[i]['type'] = 1 # artery
        elif nameV.startswith(("V_","v_")) or any(sub in nameV for sub in ("ven", "Ven", "vein", "Vein")):
            vess[i]['type'] = 0 # vein
        else:
            print(f"WARNING: art/ven type couldn't be determined for vessel {nameV}. Add it manually.")
            

        nameP = "l_"+nameV
        idxParam = df_params.index[df_params["variable_name"] == nameP].tolist()
        if len(idxParam)==1:
            vess[i]['len'] = float(df_params.at[idxParam[0],"value"])*convLen
        elif len(idxParam)==0:
            sys.exit(f"Parameter {nameP} not found in parameters dataframe.")
        elif len(idxParam)>1:
            sys.exit(f"Multiple matches for parameter {nameP} found in parameters dataframe.")

        nameP = "rProx_0_"+nameV
        idxParam = df_params.index[df_params["variable_name"] == nameP].tolist()
        nameP2 = "rDist_0_"+nameV
        idxParam2 = df_params.index[df_params["variable_name"] == nameP2].tolist()
        if len(idxParam)==1:
            if len(idxParam2)==1:
                vess[i]['r_prox'] = float(df_params.at[idxParam[0],"value"])*convLen
                vess[i]['r_dist'] = float(df_params.at[idxParam2[0],"value"])*convLen
            elif len(idxParam2)==0:
                sys.exit(f"Parameter {nameP} found in parameters dataframe, but not parameter {nameP2}. They should go in pair.")
            elif len(idxParam2)>1:
                sys.exit(f"Multiple matches for parameter {nameP2} found in parameters dataframe.")
        elif len(idxParam)>1:
            sys.exit(f"Multiple matches for parameter {nameP} found in parameters dataframe.")
        elif len(idxParam)==0:
            if len(idxParam2)==1:
                sys.exit(f"Parameter {nameP2} found in parameters dataframe, but not parameter {nameP}. They should go in pair.")
            elif len(idxParam2)>1:
                sys.exit(f"Parameter {nameP2} found in parameters dataframe (with multiple matches), but not parameter {nameP}. They should go in pair and be unique.")
            elif len(idxParam2)==0:
                nameP0 = "r_0_"+nameV
                print(f"Parameters {nameP} and {nameP2} not found in parameters dataframe. Searching for {nameP0}.")
                idxParam0 = df_params.index[df_params["variable_name"] == nameP0].tolist()
                if len(idxParam0)==1:
                    vess[i]['r_prox'] = float(df_params.at[idxParam0[0],"value"])*convLen
                    vess[i]['r_dist'] = float(df_params.at[idxParam0[0],"value"])*convLen
                elif len(idxParam0)==0:
                    sys.exit(f"Parameter {nameP0} not found in parameters dataframe.")
                elif len(idxParam0)>1:
                    sys.exit(f"Multiple matches for parameter {nameP0} found in parameters dataframe.")

        nameP = "E_K_tube_"+nameV
        idxParam = df_params.index[df_params["variable_name"] == nameP].tolist()
        nameP2 = "E_"+nameV
        idxParam2 = df_params.index[df_params["variable_name"] == nameP2].tolist()
        if len(idxParam)==1:
            vess[i]['Ee'] = float(df_params.at[idxParam[0],"value"])*convP
        elif len(idxParam)==0:
            if len(idxParam2)==1:
                print(f"Parameter {nameP} not found in parameters dataframe, parameter {nameP2} found instead.")
                vess[i]['Ee'] = float(df_params.at[idxParam2[0],"value"])*convP
            elif len(idxParam2)==0:
                print(f"Parameters {nameP} and {nameP2} both not found in parameters dataframe. Assigning default value.")
                vess[i]['Ee'] = EeRef*convP
            elif len(idxParam2)>1:
                sys.exit(f"Parameter {nameP} not found in parameters dataframe. Parameter {nameP2} found instead but with multiple matches.")
        elif len(idxParam)>1:
            sys.exit(f"Multiple matches for parameter {nameP} found in parameters dataframe.")

        nameP = "K_m_K_tube_"+nameV
        idxParam = df_params.index[df_params["variable_name"] == nameP].tolist()
        nameP2 = "K_m_"+nameV
        idxParam2 = df_params.index[df_params["variable_name"] == nameP2].tolist()
        if len(idxParam)==1:
            vess[i]['Km'] = float(df_params.at[idxParam[0],"value"])*convP
        elif len(idxParam)==0:
            if len(idxParam2)==1:
                print(f"Parameter {nameP} not found in parameters dataframe, parameter {nameP2} found instead.")
                vess[i]['Km'] = float(df_params.at[idxParam2[0],"value"])*convP
            elif len(idxParam2)==0:
                print(f"Parameters {nameP} and {nameP2} both not found in parameters dataframe. Assigning default value.")
                vess[i]['Km'] = 0.0
            elif len(idxParam2)>1:
                sys.exit(f"Parameter {nameP} not found in parameters dataframe. Parameter {nameP2} found instead but with multiple matches.")
        elif len(idxParam)>1:
            sys.exit(f"Multiple matches for parameter {nameP} found in parameters dataframe.")

        nameP = "u_0_"+nameV
        idxParam = df_params.index[df_params["variable_name"] == nameP].tolist()
        if len(idxParam)==1:
            vess[i]['P0'] = float(df_params.at[idxParam[0],"value"])*convP
        elif len(idxParam)==0:
            print(f"Parameter {nameP} not found in parameters dataframe. Assigning default value.")
            vess[i]['P0'] = 0.0
        elif len(idxParam)>1:
            sys.exit(f"Multiple matches for parameter {nameP} found in parameters dataframe.")

        nameP = "u_ext_"+nameV
        idxParam = df_params.index[df_params["variable_name"] == nameP].tolist()
        if len(idxParam)==1:
            vess[i]['Pext'] = float(df_params.at[idxParam[0],"value"])*convP
        elif len(idxParam)==0:
            print(f"Parameter {nameP} not found in parameters dataframe. Assigning default value.")
            vess[i]['Pext'] = 0.0
        elif len(idxParam)>1:
            sys.exit(f"Multiple matches for parameter {nameP} found in parameters dataframe.")


    # for i in range(nV):
    #     print(i, vess[i])
    # print('***********************************************************************')


    nodes = {}
    iN = 0
    for i in range(nV):
        iV = vess[i]['id']
        nameV = vess[i]['name']
        n1 = vess[i]['n1']
        n2 = vess[i]['n2']

        # print(iV, nameV)
        
        if n1==-1:
            vess[i]['n1'] = iN

            conn1d0d_found = False
            for j in range(len(conn_1d_0d_info)):
                if (conn_1d_0d_info[str(j+1)]['vess1d_idx']==iV and conn_1d_0d_info[str(j+1)]['vess1d_bc_in0_or_out1']==0):
                    nodes[iN] = {'id' : iN, 'type' : 999, 'vess_list' : [iV]} # code 999: hybrid junction / 1d-0d connection
                    conn1d0d_found = True
                    break
            
            inp_vess = df_vess.at[i,'inp_vessels']
            out_vess = df_vess.at[i,'out_vessels']

            if conn1d0d_found:
                for j in range(len(inp_vess)):
                    nameV2 = inp_vess[j]

                    for k in range(nV):
                        if vess[k]['name']==nameV2:
                            print(f"Another 1D vessel found in hybrid connection: {nameV2}")
                            if nameV in df_vess.at[k,'out_vessels']:
                                vess[k]['n2'] = iN
                                iV2 = vess[k]['id']
                                nodes[iN]['vess_list'].append(iV2)
                iN +=1
                    
            else:
                if (len(inp_vess)==1 and inp_vess[0]=='input_flow_BC'):
                    nodes[iN] = {'id' : iN, 'type' : 0, 'vess_list' : [iV]} # code 0: inflow BC
                    iN +=1
                elif (len(inp_vess)==1 and inp_vess[0]=='input_pressure_BC'):
                    nodes[iN] = {'id' : iN, 'type' : 1, 'vess_list' : [iV]} # code 1: input pressure BC
                    iN +=1

                # elif (len(inp_vess)==1 and inp_vess[0]=='output_flow_BC'):
                #     nodes[iN] = {'id' : iN, 'type' : 0, 'vess_list' : [iV]}
                # elif (len(inp_vess)==1 and inp_vess[0]=='output_pressure_BC'):
                #     nodes[iN] = {'id' : iN, 'type' : 1, 'vess_list' : [iV]}
                
                else:
                    idVJ_list = [iV]
                    nameVJ_list = [nameV]
                    sideVJ_list = [-1]

                    junc1d = True
                    for j in range(len(inp_vess)):
                        nameV2 = inp_vess[j]
                        
                        # iV2 = [kk for kk in range(nV) if vess[kk]['name']==nameV2][0]
                        idxV2 = [kk for kk in range(nV) if vess[kk]['name']==nameV2]

                        if len(idxV2)==0:
                            junc1d = False
                            print(f"WARNING: input vessel {nameV2} to 1D vessel {nameV} is 0D. Skipping this connection here as this is not a fully 1D junction.")
                            continue
                        iV2 = idxV2[0]
                        nameVJ_list.append(nameV2)
                        idVJ_list.append(iV2)

                    if junc1d:   
                        for j in range(len(inp_vess)):
                            search_vess = inp_vess[j]
                            for k in range(nV):
                                nameV2 = df_vess.at[k]['name'] # vess[k]['name']
                                inp_vess2 = df_vess.at[k,'inp_vessels']
                                out_vess2 = df_vess.at[k,'out_vessels']
                                if nameV2 == search_vess: 
                                    for h in range(len(nameVJ_list)):
                                        if (nameVJ_list[h]!=nameV2 and nameVJ_list[h] in inp_vess2):
                                            vess[k]['n1'] = iN

                                            for p in range(len(inp_vess2)):
                                                nameV3 = inp_vess2[p]
                                                iV3 = [kk for kk in range(nV) if vess[kk]['name']==nameV3][0]
                                                if nameV3 not in nameVJ_list:
                                                    inp_vess3 = df_vess.at[iV3,'inp_vessels']
                                                    out_vess3 = df_vess.at[iV3,'out_vessels']
                                                    if any(x in nameVJ_list for x in inp_vess3):
                                                        vess[iV3]['n1'] = iN
                                                    elif any(x in nameVJ_list for x in out_vess3):
                                                        vess[iV3]['n2'] = iN
                                                    nameVJ_list.append(nameV3)
                                                    idVJ_list.append(iV3)  
                                            
                                            sideVJ_list.append(-1)
                                            break
                                        
                                        elif (nameVJ_list[h]!=nameV2 and nameVJ_list[h] in out_vess2):
                                            vess[k]['n2'] = iN
                                            
                                            for p in range(len(out_vess2)):
                                                nameV3 = out_vess2[p]
                                                iV3 = [kk for kk in range(nV) if vess[kk]['name']==nameV3][0]
                                                if nameV3 not in nameVJ_list:
                                                    inp_vess3 = df_vess.at[iV3,'inp_vessels']
                                                    out_vess3 = df_vess.at[iV3,'out_vessels']
                                                    if any(x in nameVJ_list for x in inp_vess3):
                                                        vess[iV3]['n1'] = iN
                                                    elif any(x in nameVJ_list for x in out_vess3):
                                                        vess[iV3]['n2'] = iN
                                                    nameVJ_list.append(nameV3)
                                                    idVJ_list.append(iV3)
                                            
                                            sideVJ_list.append(1)
                                            break
                                    break
                        
                        nVJ = len(idVJ_list)
                        idVJ_list.sort()
                        nodes[iN] = {'id' : iN, 'type' : nVJ, 'vess_list' : idVJ_list} # code N>1: 1D junction with N vessels
                        if nVJ==2:
                            if sideVJ_list[0]==sideVJ_list[1]:
                                print(idVJ_list,sideVJ_list)
                        elif nVJ==3:
                            if sideVJ_list[0]==sideVJ_list[1] and sideVJ_list[0]==sideVJ_list[2]:
                                print(idVJ_list,sideVJ_list)
                        # print("fully 1d junction node:",iN,nVJ,idVJ_list,sideVJ_list)
                        iN +=1
            
        if n2==-1:
            vess[i]['n2'] = iN

            conn1d0d_found = False
            for j in range(len(conn_1d_0d_info)):
                if (conn_1d_0d_info[str(j+1)]['vess1d_idx']==iV and conn_1d_0d_info[str(j+1)]['vess1d_bc_in0_or_out1']==1):
                    nodes[iN] = {'id' : iN, 'type' : 999, 'vess_list' : [iV]} # code 999: hybrid junction / 1d-0d connection
                    conn1d0d_found = True
                    break

            inp_vess = df_vess.at[i,'inp_vessels']
            out_vess = df_vess.at[i,'out_vessels']

            if conn1d0d_found:
                for j in range(len(out_vess)):
                    nameV2 = out_vess[j]

                    for k in range(nV):
                        if vess[k]['name']==nameV2:
                            print(f"Another 1D vessel found in hybrid connection: {nameV2}")
                            if nameV in df_vess.at[k,'inp_vessels']:
                                vess[k]['n1'] = iN
                                iV2 = vess[k]['id']
                                nodes[iN]['vess_list'].append(iV2)
                iN +=1

            else:
                if (len(out_vess)==1 and out_vess[0]=='output_flow_BC'):
                    nodes[iN] = {'id' : iN, 'type' : -3, 'vess_list' : [iV]} # code -3: ouflow BC; code -1 (and sometimes -2) reserved to RCR / single-resistance BC
                    iN +=1
                elif (len(out_vess)==1 and out_vess[0]=='output_pressure_BC'):
                    nodes[iN] = {'id' : iN, 'type' : -4, 'vess_list' : [iV]} # code -4: output pressure BC; code -1 (and sometimes -2) reserved to RCR / single-resistance BC
                    iN +=1

                # elif (len(out_vess)==1 and out_vess[0]=='input_flow_BC'):
                #     nodes[iN] = {'id' : iN, 'type' : 0, 'vess_list' : [iV]}
                # elif (len(out_vess)==1 and out_vess[0]=='input_pressure_BC'):
                #     nodes[iN] = {'id' : iN, 'type' : 1, 'vess_list' : [iV]}

                else:
                    idVJ_list = [iV]
                    nameVJ_list = [nameV]
                    sideVJ_list = [1]

                    junc1d = True
                    for j in range(len(out_vess)):
                        nameV2 = out_vess[j]
                        
                        # iV2 = [kk for kk in range(nV) if vess[kk]['name']==nameV2][0]
                        idxV2 = [kk for kk in range(nV) if vess[kk]['name']==nameV2]

                        if len(idxV2)==0:
                            junc1d = False
                            print(f"WARNING: output vessel {nameV2} to 1D vessel {nameV} is 0D. Skipping this connection here as this is not a fully 1D junction.")
                            continue
                        iV2 = idxV2[0]
                        nameVJ_list.append(nameV2)
                        idVJ_list.append(iV2)

                    if junc1d:  
                        for j in range(len(out_vess)):
                            search_vess = out_vess[j]
                            for k in range(nV):
                                nameV2 = vess[k]['name']
                                inp_vess2 = df_vess.at[k,'inp_vessels']
                                out_vess2 = df_vess.at[k,'out_vessels']
                                if nameV2 == search_vess: 
                                    for h in range(len(nameVJ_list)):
                                        if (nameVJ_list[h]!=nameV2 and nameVJ_list[h] in inp_vess2):
                                            vess[k]['n1'] = iN
                                            
                                            for p in range(len(inp_vess2)):
                                                nameV3 = inp_vess2[p]
                                                iV3 = [kk for kk in range(nV) if vess[kk]['name']==nameV3][0]
                                                if nameV3 not in nameVJ_list:
                                                    inp_vess3 = df_vess.at[iV3,'inp_vessels']
                                                    out_vess3 = df_vess.at[iV3,'out_vessels']
                                                    if any(x in nameVJ_list for x in inp_vess3):
                                                        vess[iV3]['n1'] = iN
                                                    elif any(x in nameVJ_list for x in out_vess3):
                                                        vess[iV3]['n2'] = iN
                                                    nameVJ_list.append(nameV3)
                                                    idVJ_list.append(iV3)
                                            
                                            sideVJ_list.append(-1)
                                            break
                                        
                                        elif (nameVJ_list[h]!=nameV2 and nameVJ_list[h] in out_vess2):
                                            vess[k]['n2'] = iN

                                            for p in range(len(out_vess2)):
                                                nameV3 = out_vess2[p]
                                                iV3 = [kk for kk in range(nV) if vess[kk]['name']==nameV3][0]
                                                if nameV3 not in nameVJ_list:
                                                    inp_vess3 = df_vess.at[iV3,'inp_vessels']
                                                    out_vess3 = df_vess.at[iV3,'out_vessels']
                                                    if any(x in nameVJ_list for x in inp_vess3):
                                                        vess[iV3]['n1'] = iN
                                                    elif any(x in nameVJ_list for x in out_vess3):
                                                        vess[iV3]['n2'] = iN
                                                    nameVJ_list.append(nameV3)
                                                    idVJ_list.append(iV3)
                                            
                                            sideVJ_list.append(1)
                                            break
                                    break
                        
                        nVJ = len(idVJ_list)
                        idVJ_list.sort()
                        nodes[iN] = {'id' : iN, 'type' : nVJ, 'vess_list' : idVJ_list} # code N>1: 1D junction with N vessels
                        if nVJ==2:
                            if sideVJ_list[0]==sideVJ_list[1]:
                                print(idVJ_list,sideVJ_list)
                        elif nVJ==3:
                            if sideVJ_list[0]==sideVJ_list[1] and sideVJ_list[0]==sideVJ_list[2]:
                                print(idVJ_list,sideVJ_list)
                        # print("fully 1d junction node:",iN,nVJ,idVJ_list,sideVJ_list)
                        iN +=1

            
    nN = len(nodes)
    print('Number of nodes :', nN)
    print('****')

    print("1D vessels:")
    for iV in range(nV):
        print(iV, vess[iV])
    print('***********************************************************************') 
    print("1D network nodes:")
    for iN in range(nN):
        print(iN, nodes[iN])
    print('***********************************************************************')


    print('Writing vessels file')
    f = open(vess_file, "w")

    # f.write("# 0: vess ID; 1: first node; 2: second node; 3: length [cm]; 4: inlet radius [cm]; 5: outlet radius [cm]; 6: tot term resistance [dyn s/cm5]; 7: term compliance [cm5/dyn]\n")
    # f.write("# 0: vess_ID; 1: first_node; 2: second_node; 3: length [cm]; 4: inlet_radius [cm]; 5: outlet_radius [cm]; 6: tot_term_resistance [dyn s/cm5]; 7: term_compliance [cm5/dyn]; 8: wall_thickness [cm]; 9: elastic_mod [dyn/cm2]\n")
    f.write("# 0: vess_ID; 1: first_node; 2: second_node; "
            "3: length [cm]; 4: inlet_radius [cm]; 5: outlet_radius [cm]; "
            "6: tot_term_resistance [dyn s/cm5]; 7: term_compliance [cm5/dyn]; "
            "8: wall_thickness [cm]; 9: elastic_mod [dyn/cm2]; 10: visco_mod [dyn s/cm2]; "
            "11: p_0 [dyn/cm2]; 12: p_ext [dyn/cm2]; 13: art_ven_type "
            "\n"
        )


    for i in range(nV):
        iV = int(vess[i]['id'])
        nameV = vess[i]['name']
        
        f.write("%.i " % (vess[i]['id']))
        f.write("%.i " % (vess[i]['n1']))
        f.write("%.i " % (vess[i]['n2']))
        
        f.write("%.18e " % (vess[i]['len']))
        
        f.write("%.18e " % (vess[i]['r_prox']))
        f.write("%.18e " % (vess[i]['r_dist']))
        
        f.write("%.18e " % (-1.0)) # total terminal resistance
        f.write("%.18e " % (-1.0)) # terminal compliance
        
        f.write("%.18e " % (-1.0)) # wall thickness - not provided if directly computed by the 1D BFM solver

        f.write("%.18e " % (vess[i]['Ee'])) # elastic/Young modulus - usually not provided in the vessel input file for the 1D BFM solver
        f.write("%.18e " % (vess[i]['Km'])) # viscous coefficient - usually not provided in the vessel input file for the 1D BFM solver

        f.write("%.18e " % (vess[i]['P0'])) # reference pressure - usually not provided in the vessel input file for the 1D BFM solver
        f.write("%.18e " % (vess[i]['Pext'])) # external pressure - usually not provided in the vessel input file for the 1D BFM solver

        f.write("%.i " % (vess[i]['type']))

        f.write("\n")

    f.close()
    print('DONE!')
    print('****') 


    print('Writing nodes file')
    f = open(nodes_file, "w")

    f.write("# 0: node_ID; 1: node_type; 2: x_coord [cm]; 3: y_coord [cm]; 4: z_coord [cm] \n")

    for i in range(nN):
        f.write("%.i " % (nodes[i]['id']))
        f.write("%.i " % (nodes[i]['type']))
        f.write("%.18e " % (0.0))
        f.write("%.18e " % (0.0))
        f.write("%.18e " % (0.0))
        f.write("\n")

    f.close()
    print('DONE!')
    print('****')

    return vess, nodes


def generate1DPythonSimInitFile(df_params, vess, nodes, sim_init_file, filename_prefix, run_folder, ODEsolver, dtSample, computeTotBV):

    config = ConfigParser(allow_no_value=True, delimiters=(':',)) # delimiters=('=', ':')
    config.optionxform = str  # keep keys case-sensitive
    
    # print(filename_prefix)
    # print(run_folder)
    
    params = 'constant'
    # params = 'linear'
    nVar1D = 2
    if params=='linear':
        nVar1D = 5

    # 'wallthickness' = 1, if vessel wall thickness to be computed; 'wallthickness' = 0, if provided in the vess file
    wallthickness = 1
    
    config['network'] = {
        'networkName': filename_prefix[:-3],
        'geomFold': 'NA',
        'testFold': run_folder,
        'inputFoldName': 'input_files',
        'vessFile': 'vess_'+filename_prefix[:-3]+'.txt',
        'nodeFile': 'nodes_'+filename_prefix[:-3]+'.txt',
        'nameFile': 'names_'+filename_prefix[:-3]+'.csv',
        'inflowFile_prefix': 'NA',
        'outflowFile_prefix': 'NA',
        'nVess': len(vess),
        'nNode': len(nodes),
        'nVar1D': nVar1D,
        'nVarJ': 2,
        'params': params,
        'wallthickness': wallthickness,
        'couple_to_0d': True,
        'compute_tot_volume': computeTotBV
    }

    T0 = 1.0
    for i in range(df_params.shape[0]):
        if (df_params.at[i,'variable_name']=='T' and df_params.at[i,'units']=='second'):
            T0 = float(df_params.at[i,'value'])
            break
    tEnd = 10*T0
    # tEnd = nCC*T0
    
    config['discretization'] = {
        'nCellsMin': 2,
        'nCells': -1,
        'dxMax': 1.0, #0.1,
        'NMAX': 10000000000000,
        'T0': T0,
        'tIni': 0.0,
        'tEnd': tEnd,
        'dtSample': dtSample, 
        'tolTime': 1e-11 # 1e-10
    }
    # T0 and tEnd will be overridden by the coupler using T0 and nCC instead


    rho = 1.040
    for i in range(df_params.shape[0]):
        if (df_params.at[i,'variable_name']=='rho' and df_params.at[i,'units']=='Js2_per_m5'):
            rho = float(df_params.at[i,'value'])*1e-3
            break
    mu = 0.040
    for i in range(df_params.shape[0]):
        if (df_params.at[i,'variable_name']=='mu' and df_params.at[i,'units']=='Js_per_m3'):
            mu = float(df_params.at[i,'value'])*1e+1
            break
    nu = 0.5
    for i in range(df_params.shape[0]):
        if (df_params.at[i,'variable_name']=='nu' and df_params.at[i,'units']=='dimensionless'):
            nu = float(df_params.at[i,'value'])
            break
    EeRefA = 4.0e6 # [dyn/cm2]
    if 'Ee' in vess[0]:
        if vess[0]['Ee']>=0.:
            EeRefA = float(vess[0]['Ee'])
    P0A = 1.0e+5 # [dyn/cm2] =~ 75. mmHg # 5.0e+4 [dyn/cm2] =~ 37.5 mmHg
    if 'P0' in vess[0]:
        if vess[0]['P0']>=0.:
            P0A = float(vess[0]['P0'])

    config['model'] = {
        'mmHgDyncm2': 1333.2238,
        'rho': rho,
        'mu': mu,
        'nu': nu,
        'velProf': 2.0,
        'coriolis': 1.0,
        'EeA': EeRefA,
        'EeV': -1.0,
        'mA': 0.5,
        'nA': 0.0,
        'mV': 10.0,
        'nV': -1.5,
        'pries': 0,
        'nonlinear': 1,
        'couplePtot': 0, # 1,
        'P0A': P0A,
        'P0V': 0.0,
        'PiniA': P0A,
        'PiniV': 0.0,
        'Pe': 0.0,
        'Pv': 6666.0
    }


    config['ICs'] = {
        'ICtype': 1,
        'ICstate': -1
    }
    # 'ICtype' = 0 if Pinit=0; 1 if Pinit=PiniA/PiniV; 2 if Pinit=P0A/P0V


    nIn = 0
    idxs_nIn = []
    nOut = 0
    idxs_nOut = []
    for i in range(len(nodes)):
        if (nodes[i]['type']==0 or nodes[i]['type']==1):
            nIn +=1
            idxs_nIn.append(nodes[i]['id'])
        elif (nodes[i]['type']==-1 or nodes[i]['type']==-3 or nodes[i]['type']==-4):
            nOut +=1
            idxs_nOut.append(nodes[i]['id'])

    idxs_vIn = []
    if nIn>0:
        for i in range(nIn):
            nID = idxs_nIn[i]
            for j in range(len(vess)):
                if (vess[j]['n1']==nID or vess[j]['n2']==nID):
                    idxs_vIn.append( str(vess[j]['id']) )
                    break

    idxs_vOut = []
    if nOut>0:
        for i in range(nOut):
            nID = idxs_nOut[i]
            for j in range(len(vess)):
                if (vess[j]['n1']==nID or vess[j]['n2']==nID):
                    idxs_vOut.append( str(vess[j]['id']) )
                    break

    config['BCs'] = {
        'nIn': nIn,
        'inflowIdxs': ' '.join(idxs_vIn),
        'nOut': nOut,
        'outflowIdxs': ' '.join(idxs_vOut),
        'termType': 'RCR',
        'PoutFile': 'NA'
    }


    if ODEsolver=='explEul':
        FVord = 1
        ODEord = 1
    else:
        FVord = 2
        if (ODEsolver=='Heun' or ODEsolver=='midpoint'):
            ODEord = 2
        elif ODEsolver=='RK4':
            ODEord = 4
        else: # ODEsolver = 'CVODE'
            ODEord = -1

    config['numerics'] = {
        'CFL': 0.9,
        'FVord': FVord,
        'numFlux': 'HLLArteries',
        'WBscheme': 1,
        'slopeType': 'ENO',
        'ODEord': ODEord,
        'ODEsolver': ODEsolver
    }


    # tSaveIni = np.maximum(0.0, (nCC-2)*T0)
    config['results'] = {
        'outputFoldName': 'res',
        'tSaveIni': -1.0, # 0.0,
        'tSaveEnd': -1.0, # 1000.0,
        'saveResSpace': 0,
        'tSaveIniSpace': -1.0, # 0.0,
        'tSaveEndSpace': -1.0 # 1000.0
    }


    # MAP = 85.*1333.2238 # [dyne/cm2] 
    MAP = 90.*1333.2238 # [dyne/cm2] 
    config['calibration'] = {
        'MAP': MAP,
        'CsysA': 1.7,
        'CsysA_units': 'ml/mmHg',
        'CsysV': 146.0,
        'CsysV_units': 'ml/mmHg',
        'CO': 100.0,
        'useImpedance': 1
    }
    # MAP (mean arterial pressure) = 113324.023 [dyne/cm2] =~ 85 [mmHg]; provided, but not really used
    # CsysA (total systemic arterial vascular compliance) = 1.7 [mL/mmHg]; provided, but not really used
    # CO (cardiac output) = 100.0 [ml/s]; provided, but not really used
    # useImpedance = 1 to use vessel characteristic impedance to split the total terminal resistance of RCR model, 0 otherwise (some R1/R2 to Rtot ratio is used instead)


    # config['new_section'] = {
    #     'new_key' : 'new_value'
    # }


    print(f'Writing 1D simulation initialisation file {sim_init_file}')
    with open(sim_init_file, 'w') as configfile:
        config.write(configfile)
    print('DONE!')
    print('****')

