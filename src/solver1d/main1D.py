#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:01:04 2020

@author: beatriceghitti
"""

import math
import numpy as np
import pandas as pd
import os
import sys
sys.stdout.flush()
import argparse
# import configparser
from configparser import ConfigParser
# import warnings
import time as timing
# import random
# from scipy.interpolate import interp1d
import json

import vessel as vessel
import model as model
import numerics as numerics
import bcs as bcs
import windkessel as wdkl
import auxCouplingFunctions as coupling
import auxPipeFunctions as pipe

PETSC_SOLVER = "CN"     # Crank-Nicolson
# PETSC_SOLVER = "BDF1"   # BDF o1 (Backward Euler)
# PETSC_SOLVER = "BDF2"   # BDF o2



def get_column_mapping(filepath):
    with open(filepath, 'r') as f:
        header = f.readline()
    
    # Split by semicolon and remove the '#' and whitespace
    columns = header.strip().replace('#', '').split(';')
    
    mapping = {}
    for i, col in enumerate(columns):
        # Remove the index prefix (e.g., " 1: ") and units (e.g., "[m3]")
        # Example: " 1: heart/q_ra[m3]" -> "heart/q_ra"
        clean_name = col.split(':')[-1].split('[')[0].strip()
        
        if clean_name: # Avoid empty strings from trailing semicolons
            mapping[clean_name] = i
            
    return mapping



def run1DBFmod(AICstate=[], QICstate=[]):
    
    is_coupled = len(sys.argv) > 1 and sys.argv[1] == "1"
    print('1d solver :: is_coupled : ', is_coupled)

    print("1d solver :: input arguments")
    for i, arg in enumerate(sys.argv):
        print(f"arg[{i}]: {repr(arg)}")

    init_file = None
    if is_coupled:
        init_file = sys.argv[5]
    else:
        if len(sys.argv) > 1:
            init_file = sys.argv[1]
    
    if init_file is None:
        print(f"1d solver :: Error: path to 1d simulation initialisation file is {init_file}. Exiting.")
        exit()

    if not os.path.exists(init_file):
        print(f"1d solver :: Error: 1d simulation initialisation file {init_file} does not exist. Exiting.")
        exit()

    config = ConfigParser(allow_no_value=True, delimiters=(':',)) # delimiters=('=', ':')
    config.optionxform = str  # keep keys case-sensitive
    config.read(init_file)

    networkName = config.get('network', 'networkName', fallback='aortic_bif_hybrid')

    ODEsolver = None
    T0 = 0.
    nCC = 0
    pipePath = None
    initStatePath = "None"
    initStatesFile = "None"
    initVarsFile = "None"
    if is_coupled:
        ODEsolver = sys.argv[2]
        T0 = float(sys.argv[3])
        nCC = int(sys.argv[4])
        pipePath = sys.argv[6]
        
        initStatePath = sys.argv[7]
        initStatesFile = initStatePath + "sol0D_states.txt"
        initVarsFile = initStatePath + "sol0D_variables.txt"
    else:
        ODEsolver = config.get('numerics', 'ODEsolver', fallback='RK4')
        T0 = config.getfloat('discretization', 'T0', fallback=1.0) # [s] cardiac cycle duration
        nCC = config.getint('discretization', 'nCC', fallback=0) # number of cardiac cycles

    testFold = config.get('network', 'testFold', fallback=os.path.dirname(init_file))
    inputFoldName = config.get('network', 'inputFoldName', fallback='input_files')
    inputFold = testFold+'/'+inputFoldName+'/'

    #XXX
    write_pipe_dt = None
    read_pipe_dt = None
    write_pipe = None
    read_pipe = None
    write_pipe_vol = None

    couple_volume_sum = config.getboolean('network', 'compute_tot_volume', fallback=False)
   
    N1d0dTot = 0
    N1d0d = 0
    pipe_1d_0d_info = {}
    FLOAT_SIZE = 8  # size of float64 in bytes
    DATA_LENGTH = 0 
    if is_coupled:
        with open(inputFold+networkName+'_coupler1d0d.json', "r") as file:
            pipe_1d_0d_info = json.load(file)

        N1d0dTot = len(pipe_1d_0d_info)

        for i in range(N1d0dTot):
            if "port_volume_sum" in pipe_1d_0d_info[str(i+1)]:
                if pipe_1d_0d_info[str(i+1)]["port_volume_sum"]==1:
                    pass
                else:
                    N1d0d +=1
            else:
                N1d0d +=1
        
        DATA_LENGTH = 2 # number of elements in the array
        # no need to transfer dt for each 1d-0d coupling pair
        # as we have introduced a dedicated pipe to transfer time level and time step

        write_pipe_dt, read_pipe_dt, write_pipe, read_pipe, write_pipe_vol = pipe.openPipes(pipePath, N1d0d, couple_volume_sum)
        if write_pipe_dt==None:
            return 1


    # [units conversion factors]
    convLen = 1.0e-2 # length units conversion from [cm] to [m]
    convQ = 1e-06 # flow units conversion from [cm3/s] to [m3/s]
    convP = 1e-01 # pressure units conversion from [dyne/cm2] to [J/m3]
    convR = convP/convQ # 1e+05 # resistance units conversion from [dyne s/cm5] to [J s/m6]
    convC = convQ/convP # 1.0e-05 # compliance units conversion from [cm5/dyne] to [m6/J] 
    mmHgDyncm2 = config.getfloat('model', 'mmHgDyncm2', fallback=1333.2238) # pressure units conversion from [mmHg] to [dyne/cm2]
    
    # [network]
    nVess = config.getint('network', 'nVess', fallback=3)
    nNode = config.getint('network', 'nNode', fallback=4)
    nVar1D = config.getint('network', 'nVar1D', fallback=2)
    nVarJ = config.getint('network', 'nVarJ', fallback=2)
    params = config.get('network', 'params', fallback='constant')
    wallthickness = config.getint('network', 'wallthickness', fallback=1)

    nameFile = config.get('network', 'nameFile', fallback='NA')
    vessFile = config.get('network', 'vessFile', fallback='vess_'+networkName+'.txt')
    nodeFile = config.get('network', 'nodeFile', fallback='nodes_'+networkName+'.txt')

    vessData = np.genfromtxt(inputFold+vessFile)   
    nodeData = np.genfromtxt(inputFold+nodeFile)

    if nVess>1:
        nColV = vessData.shape[1]
        nColN = nodeData.shape[1]
    else:
        nColV = len(vessData)
        nColN = len(nodeData)

    df_names = pd.DataFrame()
    if nameFile!='NA':
        df_names = pd.read_csv(inputFold+nameFile, header=0) #, sep=',', index_col=None)
    
    print('[network]')
    print('network : ', networkName)
    print('nVess : ', nVess)
    print('nNode : ', nNode)
    print('nVar1D : ', nVar1D)
    print('params : ', params)
    print('compute wall thickness : ', wallthickness)
    
    # [discretization]
    nCellsMin = config.getint('discretization', 'nCellsMin', fallback=2)
    nCells = config.getint('discretization', 'nCells', fallback=-1)
    dxMax = config.getfloat('discretization', 'dxMax', fallback=0.1)  # 1 [mm] = 0.1 [cm] 
    
    NMAX = config.getint('discretization', 'NMAX', fallback=10000000000000) # maximum number of iteration
    tIni = config.getfloat('discretization', 'tIni', fallback=0.0)
    # tolTime = 1e-11
    tolTime = config.getfloat('discretization', 'tolTime', fallback=1e-11) # 1e-10 # 1e-11 # 1e-13
    if nCC>0:
        tEndGlob = nCC*T0
    else:
        tEndGlob = config.getfloat('discretization', 'tEnd', fallback=10.0)
        nCC = int(tEndGlob/T0)
    tEndGlob = round(tEndGlob/tolTime)*tolTime
    dtSample = config.getfloat('discretization', 'dtSample', fallback=0.001)
    tEnd = dtSample
    

    print('[discretization]')
    print('dxMax : ', dxMax, ' [cm]')
    print('nCellsMin : ', nCellsMin)
    print('nCells : ', nCells)
    print('NMAX : ', NMAX)
    print('T0 : ', T0)
    print('tIni : ', tIni)
    print('tEndGlob : ', tEndGlob)
    print('dtSample : ', dtSample)

    # [results]
    saveResTime = 0
    tSaveIni = config.getfloat('results', 'tSaveIni', fallback=0.)
    tSaveEnd = config.getfloat('results', 'tSaveEnd', fallback=1000.)
    if tSaveIni<0.:
        tSaveIni = np.maximum(0.0, (nCC-2)*T0)
    if tSaveEnd<0.:
        tSaveEnd = tEndGlob
    outputFoldName = config.get('results', 'outputFoldName', fallback='res')
    outFolder = testFold+'/'+outputFoldName+'/'
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
    
    saveResSpace = config.getint('results', 'saveResSpace', fallback=0)    
    tSaveIniSpace = config.getfloat('results', 'tSaveIniSpace', fallback=1000.)
    tSaveEndSpace = config.getfloat('results', 'tSaveEndSpace', fallback=1000.)
    if saveResSpace:
        if tSaveIniSpace<0.:
            tSaveIniSpace = tSaveIni
        if tSaveEndSpace<0.:
            tSaveEndSpace = tSaveEnd
    outFolderSpace = outFolder+"space/"
    if not os.path.exists(outFolderSpace):
        os.makedirs(outFolderSpace)
        
    print('[results]')
    print('tSaveIni : ', tSaveIni)
    print('tSaveEnd : ', tSaveEnd)
    if saveResSpace:
        print('tSaveIniSpace : ', tSaveIniSpace)
        print('tSaveEndSpace : ', tSaveEndSpace)
    
    # [model]
    rho = config.getfloat('model', 'rho', fallback=1.040)  # [g/cm^3] blood density
    muRef = config.getfloat('model', 'mu', fallback=0.040)  # [g/cm/s] blood viscosity
    velProf = config.getfloat('model', 'velProf', fallback=2.)  # velocity profile order
    xi = 2.*(velProf + 2.) # fallback=8.
    coriolis = config.getfloat('model', 'coriolis', fallback=1.)
    # arteries
    EeRefA = config.getfloat('model', 'EeA', fallback=4.0e+6)  # [dyne/cm^2] elastic/Young's modulus # 2.25e+6
    mA = config.getfloat('model', 'mA', fallback=0.5)
    nA = config.getfloat('model', 'nA', fallback=0.)
    # veins
    EeRefV = config.getfloat('model', 'EeV', fallback=1.0e+5)  # [dyne/cm^2] elastic/Young's modulus # 2.25e+6
    mV = config.getfloat('model', 'mV', fallback=10.)
    nV = config.getfloat('model', 'nV', fallback=-1.5)
    pries = config.getint('model', 'pries', fallback=0)
    couplePtot = config.getint('model', 'couplePtot', fallback=1)

    # arteries
    P0A = config.getfloat('model', 'P0A', fallback=1.0e+5) # [dyne/cm^2] reference pressure
    PiniA = config.getfloat('model', 'PiniA', fallback=1.0e+5) # [dyne/cm^2] initial pressure
    # veins
    P0V = config.getfloat('model', 'P0V', fallback=6666.) # [dyne/cm^2] reference pressure
    PiniV = config.getfloat('model', 'PiniV', fallback=6666.) # [dyne/cm^2] initial pressure
    Pe = config.getfloat('model', 'Pe', fallback=0.) # [dyne/cm^2] external pressure
    Pv = config.getfloat('model', 'Pv', fallback=6666.) # [dyne/cm^2] outflow/outlet pressure (approx. 5. [mmHg] for venous pressure)
    #XXX TODO P0 and Pe can be vessel-specific and given in the vessels file

    print('[model]')
    print('rho : ', rho, ' [g/cm^3]')
    print('muRef : ', muRef, ' [g/cm/s]')
    print('use pries : ', pries)
    print('velProf : ', velProf)
    print('xi : ', xi)
    print('coriolis : ', coriolis)
    print('[arteries]')
    print('EeRefA : ', EeRefA, ' [dyne/cm^2]')
    print('mA : ', mA)
    print('nA : ', nA)
    print('P0A : ', P0A, ' [dyne/cm2] =', round(P0A/mmHgDyncm2,3), ' [mmHg]')
    print('PiniA : ', PiniA, ' [dyne/cm2] =', round(PiniA/mmHgDyncm2,3), ' [mmHg]')
    print('[veins]')
    print('EeRefV : ', EeRefV, ' [dyne/cm^2]')
    print('mV : ', mV)
    print('nV : ', nV)
    print('P0V : ', P0V, ' [dyne/cm2] =', round(P0V/mmHgDyncm2,3), ' [mmHg]')
    print('PiniV : ', PiniV, ' [dyne/cm2] =', round(PiniV/mmHgDyncm2,3), ' [mmHg]')
    print('Pe : ', Pe, ' [dyne/cm2] =', round(Pe/mmHgDyncm2,3), ' [mmHg]')
    print('Pv : ', Pv, ' [dyne/cm2] =', round(Pv/mmHgDyncm2,3), ' [mmHg]')
    
    # [numerics]
    CFL = config.getfloat('numerics', 'CFL', fallback=0.9)
    FVord = config.getint('numerics', 'FVord', fallback=2)
    numFlux = config.get('numerics', 'numFlux', fallback='HLLArteries')  # numFlux = 'HLLArteries'; numFlux = 'GodunovExactArteries'
    if FVord==1:
        slopeType = 'none'
    elif FVord==2:
        slopeType = config.get('numerics', 'slopeType', fallback='ENO')
    
    if FVord==1:
        if ODEsolver=='explEul':
            ODEord = 1
        # elif ODEsolver=='CVODE': #XXX use FVord=2 only with CVODE
        #     ODEord = -1
        elif ODEsolver=='PETSC':
            # ODEord = -1
            if PETSC_SOLVER=="BDF1":
                ODEord = 1
            else:
                sys.exit(f'ERROR :: FV and ODE solver orders NOT matching: FVord={FVord} || {ODEsolver} with {PETSC_SOLVER}')
        else:
            sys.exit(f'ERROR :: FV and ODE solver orders NOT matching: FVord={FVord} || {ODEsolver}')
    elif FVord==2:
        if ODEsolver=='Heun' or ODEsolver=='midpoint':
            ODEord = 2
        elif ODEsolver=='RK4':
            ODEord = 4
        elif ODEsolver=='CVODE':
            ODEord = -1
        elif ODEsolver=='PETSC':
            # ODEord = -1
            if PETSC_SOLVER=="CN" or PETSC_SOLVER=="BDF2":
                ODEord = 2
            else:
                sys.exit(f'ERROR :: FV and ODE solver orders NOT matching: FVord={FVord} || {ODEsolver} with {PETSC_SOLVER}')
        else:
            sys.exit(f'ERROR :: FV and ODE solver orders NOT matching: FVord={FVord} || {ODEsolver}')

    if ODEsolver=='explEul':
        dtFrac = np.array([0.])
        weights = np.array([1.])
    elif ODEsolver=='Heun':
        dtFrac = np.array([0.,1.0])
        weights = np.array([0.5,0.5])
    elif ODEsolver=='midpoint':
        dtFrac = np.array([0.,0.5])
        weights = np.array([0.5,0.5])
    elif ODEsolver=='RK4':
        dtFrac = np.array([0.,0.5,0.5,1.0])
        weights = np.array([1./6.,1./3.,1./3.,1./6.])
    elif ODEsolver=='CVODE':
        dtFrac = []
        weights = []
    elif ODEsolver=='PETSC':
        if PETSC_SOLVER=="BDF1":
            dtFrac = np.array([1.])
            weights = np.array([1.])
        elif PETSC_SOLVER=="CN":
            dtFrac = np.array([0., 1.])
            weights = np.array([0.5, 0.5])
        elif PETSC_SOLVER=="BDF2":
            dtFrac = []
            weights = []
    

    # initialize mathematical model
    if FVord==1:
        mod = model.model(rho, xi, Pe, nVar1D)
    elif FVord==2:
        mod = model.model(rho, xi, Pe, nVar1D, slopeTy=slopeType)
    # initialize numerics
    num = numerics.numericalMethod(CFL, mod)
    
    print('[numerics]')
    print('CFL : ', CFL)
    print('FVord : ', FVord)
    print('numFlux : ', numFlux)
    print('slopeType : ', slopeType)
    # print('WBscheme : ', WBscheme)
    # print('nGP : ', nGP)
    print('ODEord : ', ODEord)
    print('ODEsolver : ', ODEsolver)
    
    # [ICs]
    ICtype = config.getint('ICs', 'ICtype', fallback=1) # ICtype=0 if Pinit=0; 1 if Pinit=Pini; 2 if Pinit=P0
    ICstate = -1
    # ICstate = config.getint('ICs', 'ICstate', fallback=-1)
    if initStatePath != "None":
        ICstate = 1  # overwrite ICstate if initial states from file are provided
    
    print('[ICs]')
    if ICstate==1:
        print('ICstate : ', ICstate, '==> initial states from files')
        print('initStatesFile : ', initStatesFile)
        print('initVarsFile : ', initVarsFile)
    else:
        print('ICstate : ', ICstate, '==> NO initial states provided')
        print('initStatesFile : ', initStatesFile)
        print('initVarsFile : ', initVarsFile)
        if ICtype==0:
            print('ICtype : ', ICtype, '==> Pinit = 0')
        elif ICtype==1:
            print('ICtype : ', ICtype, '==> Pinit = PiniA')
        elif ICtype==2:
            print('ICtype : ', ICtype, '==> Pinit = P0')
    # print('ICstate : ', ICstate)

    # [BCs]
    inflowFile_prefix = config.get('network', 'inflowFile_prefix', fallback='NA')
    outflowFile_prefix = config.get('network', 'outflowFile_prefix', fallback='NA')
    nIn = config.getint('BCs', 'nIn', fallback=1)  # number of inlet vessels/nodes
    inflowIdxs = None
    if nIn>0:
        inflowIdxs = config.get('BCs', 'inflowIdxs', fallback='0') # indexes of inlet/inflow vessels
    nOut = config.getint('BCs', 'nOut', fallback=2) # number of outlet vessels/nodes
    outflowIdxs = None
    if nOut>0:
        outflowIdxs = config.get('BCs', 'outflowIdxs', fallback='1 2') # indexes of outlet/terminal vessels
    PoutFileName = config.get('BCs', 'PoutFile', fallback='NA') # [dyne/cm^2] name of file with time-varying outflow/outlet pressure curve in veins/capillaries
    termType = config.get('BCs', 'termType', fallback='RCR') # termType = 'singleR' / 'RCR' / 'pOutflow'
    useImpedance = config.getint('calibration', 'useImpedance', fallback=1) 
    
    inVess = []
    if nIn>0:
        if (inflowIdxs is not None and inflowIdxs!='NA'):
            strToList = inflowIdxs.split(' ')
            for i in range(nIn):
                inVess.append(int(strToList[i]))
    
    outVess = []
    if nOut>0:
        if (outflowIdxs is not None and outflowIdxs!='NA'):
            strToList = outflowIdxs.split(' ')
            for i in range(nOut):
                outVess.append(int(strToList[i]))

    hybConnVess = []
    if is_coupled:
        if N1d0dTot>0:
            for i in range(N1d0dTot):
                if "port_volume_sum" in pipe_1d_0d_info[str(i+1)]:
                    if pipe_1d_0d_info[str(i+1)]["port_volume_sum"]==1:
                        pass
                    else:
                        vess1d_idx = pipe_1d_0d_info[str(i+1)]["vess1d_idx"]
                        hybConnVess.append(vess1d_idx)
                else:
                    vess1d_idx = pipe_1d_0d_info[str(i+1)]["vess1d_idx"]
                    hybConnVess.append(vess1d_idx)
    #XXX if a vessel index is appearing twice (or more) in the list hybConnVess, it means that it has multiple connections with the 0D sub-model, 
    # for instance, both its inlet and outlet are connected to 0D vessels/compartments

    # initialize boundary conditions
    bcFunc = bcs.BCs(T0)
    
    print('[BCs]')
    print('Number of inlet vessels : ', nIn)
    print('Indexes of inlet vessels : ', inVess)
    print('Number of outlet vessels : ', nOut)
    print('Indexes of outlet vessels : ', outVess)
    print('Total number of 1d-0d connections : ', N1d0dTot)
    print('Number of one-to-one 1d-0d connections : ', N1d0d)
    print('Indexes of 1d vessels with hybrid 1d-0d connections : ', hybConnVess)
    print('termType : ', termType)
    # print('PoutFileName : ', PoutFileName)
    if termType=='RCR':
        print('useImpedance : ', useImpedance) 
    
    print('Initializing vessels and nodes, terminal elements and junctions...')
    
    # initialize vessel data

    # 0: vess_ID; 
    # 1: first_node; 
    # 2: second_node; 
    # 3: length [cm]; 
    # 4: inlet_radius [cm]; 
    # 5: outlet_radius [cm]; 
    if nVess==1:
        length = np.array([vessData[3]]) # [cm] length of segments
        Rin = np.array([vessData[4]]) # [cm] initial/inlet/proximal radius of segments
        Rout = np.array([vessData[5]]) # [cm] final/outlet/distal radius of segments
    else:
        length = vessData[:,3] # [cm] length of segments
        Rin = vessData[:,4] # [cm] initial/inlet/proximal radius of segments
        Rout = vessData[:,5] # [cm] final/outlet/distal radius of segments
    R0 = np.zeros(nVess) # [cm] constant reference radius of segments
    A0 = np.zeros_like(R0) # [cm^2] reference cross-sectional area of segments
    # V0 = np.zeros_like(R0) # [cm^3] reference volume of segments
    
    # 6: tot_term_resistance [dyn s/cm5]; 
    # 7: term_compliance [cm5/dyn]; 
    # 8: wall_thickness [cm]; 
    # 9: elastic_mod [dyn/cm2]; 
    # 10: visco_mod [dyn s/cm2]; 
    # 11: p_0 [dyn/cm2]; 
    # 12: p_ext [dyn/cm2] 
    # 13: art_ven_type (1 for artery, 0 for vein)
    if nVess==1:
        RwkTot = np.array([vessData[6]]) # [dyn s/cm5]
        Cwk = np.array([vessData[7]]) # [cm5/dyne]
        h0 = np.array([vessData[8]]) # [cm] wall thickness
        Ee = np.array([vessData[9]]) # [dyn/cm2] Young
    else:
        RwkTot = vessData[:,6] # [dyn s/cm5]
        Cwk = vessData[:,7] # [cm5/dyne]
        h0 = vessData[:,8] # [cm] wall thickness
        Ee = vessData[:,9] # [dyn/cm2] Young's modulus
    
    Rwk1 = np.zeros_like(R0)
    Rwk2 = np.zeros_like(R0)
    K = np.zeros_like(R0) # wall stiffness
    mu = np.zeros_like(R0) # blood viscosity
    # Km = vessData[:,10] # [dyn s/cm2] viscoelastic parameter
    if nColV>11:
        if nVess==1:
            vessP0 = np.array([vessData[11]]) # [dyn/cm2] vessel-specific reference pressure
        else:
            vessP0 = vessData[:,11] # [dyn/cm2] vessel-specific reference pressure
    else:
        vessP0 = P0A*np.ones(nVess)
    if nColV>12:
        if nVess==1:
            vessPe = np.array([vessData[12]]) # [dyn/cm2] vessel-specific external pressure
        else:
            vessPe = vessData[:,12] # [dyn/cm2] vessel-specific external pressure
    else:
        vessPe = Pe*np.ones(nVess)
    if nColV>13: #XXX TODO parameter to distinguish between artery (1) and vein (0) vessel type
        if nVess==1:
            vessTy = np.array([vessData[13]]) # 1 for artery, 0 for vein
        else:
            vessTy = vessData[:,13]
    else:
        vessTy = np.ones(nVess)

    mTL = np.zeros_like(R0)
    nTL = np.zeros_like(R0)

    for i in range(nVess):
        if Ee[i]<0.:
            if vessTy[i]==1:
                Ee[i] = EeRefA
            else:
                Ee[i] = EeRefV

        if vessTy[i]==-1:
            sys.exit(f"1d solver :: Error: undetermined art/ven type for vessel {i}")

    for i in range(nVess):
        if vessP0[i]<0.:
            if vessTy[i]==1:
                vessP0[i] = P0A
            else:
                vessP0[i] = P0V
        if vessPe[i]<0.:
            vessPe[i] = Pe

        if vessTy[i]==1:
            mTL[i] = mA
            nTL[i] = nA
        else:
            mTL[i] = mV
            nTL[i] = nV
    
    for i in range(nVess):
        R0[i] = 0.5*(Rin[i] + Rout[i])
        A0[i] = math.pi*R0[i]**2.
        # V0[i] = length[i]*A0[i]

        if wallthickness==1:
            if vessTy[i]==1:
                h0[i] = mod.wallThicknessArtADAN(A0[i])
            else:
                # wall thickness expressed in relation to the vessel radius at equilibrium 
                # for veins, h0=5%r0
                h0[i] = mod.wallThicknessVen(R0[i], 0.05)
        
        if Ee[i]>0.:
            K[i] = mod.stiffnessParam(A0[i], Ee[i], mTL[i], nTL[i], h0[i])
        else:
            K[i] = mod.computeKfromRefWaveSpeed(A0[i], mTL[i], nTL[i])
              
        if pries==0:
            mu[i] = muRef
        elif pries==1:
            mu[i] = mod.viscosityPries(A0[i])

    if nOut>0:
        if termType=='singleR': # terminal vessels coupled to single-resistance terminal models 
            Rwk1[:] = np.copy(RwkTot) # [dyn s/cm5]
        elif termType=='RCR': # terminal vessels coupled to RCR terminal models
            if networkName.startswith(('aorticbif', 'aortic_bif')):
                for i in range(nVess):
                    if RwkTot[i]>0.:
                        Rwk1[i] = 0.021493817644410357*RwkTot[i] # proximal terminal resistance
                        Rwk2[i] = (1-0.021493817644410357)*RwkTot[i] # distal terminal resistance
            else:
                if useImpedance==1:
                    for i in range(nVess):
                        if RwkTot[i]>0.:
                            Z0 = mod.characteristicImpedance(A0[i], K[i], mTL[i], nTL[i])
                            Rwk1[i] = Z0
                            Rwk2[i] = RwkTot[i]-Z0
                else:
                    for i in range(nVess):
                        if RwkTot[i]>0.:
                            Rwk1[i] = 0.2*RwkTot[i] # 0.15
                            Rwk2[i] = 0.8*RwkTot[i] # 0.85
    # elif termType=='juncHyb':
    #     pass
    
    
    # CLASSIFICATION OF NODES
    Segments = []
    # Segments[i] = [original segment number (starting from 1) or label/name,
    #                   new segment number (starting from 0), 
    #                   first node number, first node type, 
    #                   second node number, second node type]

    # Node types:
    #   0 -> inflow (inflow/inflow_o2)
    #   -1 -> terminal (RCR/singleR/pOutflow)
    #   999 -> hybrid junction (juncHyb)
    #   >=2 (<999) -> fully 1Djunction (junc1D)
    for i in range(nVess):
        Segments.append([])
        if nVess==1:
            idxV = int(vessData[0])
            n1 = int(vessData[1])
            n2 = int(vessData[2])
        else:
            idxV = int(vessData[i,0])
            n1 = int(vessData[i,1])
            n2 = int(vessData[i,2])

        if df_names.empty:
            Segments[i].append( str(idxV+1) )
        else:
            for j in range(df_names.shape[0]):
                if int(df_names.at[j,'vess_ID']) == idxV:
                    Segments[i].append( df_names.at[j,'vess_name'] )
                    break
        Segments[i].append(idxV)
        
        Segments[i].append(n1)
        if int(nodeData[n1,1]) == 0:
            if FVord == 1:
                Segments[i].append('inflow')
            elif FVord == 2:
                Segments[i].append('inflow_o2')
        elif int(nodeData[n1,1]) == -1:
            Segments[i].append(termType)
        elif int(nodeData[n1,1]) > 1:
            if int(nodeData[n1,1]) == 999:
                Segments[i].append('juncHyb')
            else:
                Segments[i].append('junc1D')
        else:
            sys.exit(f"Node type not defined for node {n1} : {nodeData[n1,1]}")
                
        Segments[i].append(n2)
        if int(nodeData[n2,1]) == 0:
            if FVord == 1:
                Segments[i].append('inflow')
            elif FVord == 2:
                Segments[i].append('inflow_o2')
        elif int(nodeData[n2,1]) == -1:
            Segments[i].append(termType)
        elif int(nodeData[n2,1]) > 1:
            if int(nodeData[n2,1]) == 999:
                Segments[i].append('juncHyb')
            else:
                Segments[i].append('junc1D')
        else:
            sys.exit(f"Node type not defined for node {n2} : {nodeData[n2,1]}")
    
    print('1d solver :: Segments : ', Segments)

    Nodes = []
    # Nodes[i] = [node number, node type, 
    #               segment number sharing the node, 
    #               -1 IF the segment shares its first node / +1 IF the segment shares its second node,
    #               IF junction, total number of vessels converging to it]
    
    IdxsJ1D = []
    # IdxsJ0D = []
    # IdxsJ1D0D = []
    IdxsJhyb = []
    IdxsTerm = []
    for i in range(nNode):
        Nodes.append([])
        idxN = int(nodeData[i,0])
        Nodes[i].append(idxN)
        
        if int(nodeData[i,1]) == 0:
            if FVord == 1:
                Nodes[i].append('inflow')
            elif FVord == 2:
                Nodes[i].append('inflow_o2')
            for j in range(nVess):
                if Segments[j][2] == idxN:
                    Nodes[i].append(int(Segments[j][1]))
                    Nodes[i].append(-1)
                    break
                if Segments[j][4] == idxN:
                    Nodes[i].append(int(Segments[j][1]))
                    Nodes[i].append(1)
                    # break
                    sys.exit(f"1d solver :: WARNING: configuration of inflow BC still to be implemented for node {idxN}")
        elif int(nodeData[i,1]) == -1:
            Nodes[i].append(termType)
            IdxsTerm.append(idxN)
            for j in range(nVess):
                if Segments[j][2] == idxN:
                    Nodes[i].append(int(Segments[j][1]))
                    Nodes[i].append(-1)
                    # break
                    sys.exit(f"1d solver :: WARNING: configuration of terminal BC {termType} still to be implemented for node {idxN}")
                if Segments[j][4] == idxN:
                    Nodes[i].append(int(Segments[j][1]))
                    Nodes[i].append(1)
                    break
        elif int(nodeData[i,1]) > 1:
            if int(nodeData[i,1]) == 999:
                Nodes[i].append('juncHyb')
                IdxsJhyb.append(idxN)
                nVessJunc = 0
                for j in range(nVess):
                    if Segments[j][2] == idxN:
                        Nodes[i].append(int(Segments[j][1]))
                        Nodes[i].append(-1)
                        nVessJunc += 1
                    elif Segments[j][4] == idxN:
                        Nodes[i].append(int(Segments[j][1]))
                        Nodes[i].append(1)
                        nVessJunc += 1
                # if nVessJunc == 1:
                #     pass
                # else:
                #     sys.exit(f"1d solver :: Error: found 1d-0d junction connection with more than one 1d vessel: {nVessJunc}. Configuration still to be implemented.")
                Nodes[i].append(nVessJunc)
                print(f"1d solver :: found 1d-0d junction connection with {nVessJunc} 1d vessels.")
            else:
                Nodes[i].append('junc1D')
                IdxsJ1D.append(idxN)
                nVessJunc = 0
                for j in range(nVess):
                    if Segments[j][2] == idxN:
                        Nodes[i].append(int(Segments[j][1]))
                        Nodes[i].append(-1)
                        nVessJunc += 1
                    elif Segments[j][4] == idxN:
                        Nodes[i].append(int(Segments[j][1]))
                        Nodes[i].append(1)
                        nVessJunc += 1
                if nVessJunc == int(nodeData[i,1]):
                    Nodes[i].append(nVessJunc)
                else:
                    sys.exit(f"1d solver :: Error: failed to find all vessels converging to the 1D junction {idxN}: {nVessJunc} instead of {int(nodeData[i,1])}")    
        else:
            sys.exit(f"Node type not defined for node {idxN} : {nodeData[i,1]}")
    
    print('1d solver :: Nodes : ', Nodes)

    nTerm = len(IdxsTerm)
    nJhyb = len(IdxsJhyb)
    if nJhyb!=N1d0d:
        sys.exit(f"1d solver :: ERROR: number of hybrid junctions found different from number of expected 1d-0d connections: {nJhyb} & {N1d0d}")


    # initialize class for auxiliary coupling functions and time evolution
    couplingFunc = coupling.auxCouplingFunctions(Segments, Nodes, mod, num, ODEsolver, couplePtot)

    
    # initialize 1D vessels of the network
    v = []
    vessType = []
    meshSize = []
    Idxs1D = []        
    # Idxs0D = []
    M = np.zeros(nVess, dtype=int)
    
    for i in range(nVess):
        nameV = Segments[i][0]
        idxV = Segments[i][1]
        leftBCtype = Segments[i][3]
        rightBCtype = Segments[i][5]
        
        M[i] = int( num.vessCellNumber(length[i], dxMax, nCellsMin) )
        
        vessType.append('1D')
        Idxs1D.append(i)
        
        inflowDataFile = None
        if (leftBCtype == 'inflow' or leftBCtype == 'inflow_o2'):
            if inflowFile_prefix != 'NA':
                # inflowDataFile = inputFold + inflowFile_prefix + str(Segments[i][1]) + '.txt'
                inflowDataFile = inputFold+inflowFile_prefix+'.txt'
            # inflowDataFile = inputFold+"v_aov_cvs_model_0d.txt"
            # # inflowDataFile = inputFold+"inflow_adan.txt"

        v.append(vessel.vessel(length[i], M[i], nVar1D, leftBCtype, rightBCtype, 
                               vessType[i], K[i], A0[i], mTL[i], nTL[i], vessP0[i], mu[i], 
                               mod, inflowDataFile, bcFunc,
                               outFolder, saveResSpace, idxV, nameV))
        
        meshSize.append( v[i].dx )
    
    meshFile = open(outFolderSpace+'mesh.txt','w')
    meshFile.write("# 0: vess ID; 1: nCells; 2: dx [cm]; 3,...,nCells+3: xLoc [cm]\n")
    Mmax = np.max(M)
    for i in range(nVess):
        meshFile.write("%.i " % (Segments[i][1]))
        meshFile.write("%.i " % (M[i]))
        meshFile.write("%.18e " % (v[i].dx))
        for j in range(Mmax):
            if j<M[i]:
                meshFile.write("%.18e " % (v[i].x[j]))
            else:
                meshFile.write("%.18e " % (0.0))
        meshFile.write("\n")
    meshFile.close()
    
    # initialize RCR Windkessel terminal elements   
    T = [] 
    if nTerm>0:
        for i in range(nNode):
            if Nodes[i][1]=='RCR':
                vessTypeT = 'RCR'
                iV = Nodes[i][2]
                T.append( wdkl.windkessel(0., 'NA', 'NA', vessTypeT, 0., 0., 0., mTL[iV], nTL[iV], vessP0[iV], mod, -1., outFolder, iV) ) 
            elif Nodes[i][1]=='singleR':
                vessTypeT = 'singleR'
                iV = Nodes[i][2]
                T.append( wdkl.windkessel(0., 'NA', 'NA', vessTypeT, 0., 0., 0., mTL[iV], nTL[iV], vessP0[iV], mod, -1., outFolder, -1) )
            elif Nodes[i][1]=='pOutflow':
                T.append( 'NA' )
            else:
                T.append( 'NA' )
            
    
    # INITIAL CONDITIONS
    if initStatePath != "None" and ICstate==1:
        
        map_states = get_column_mapping(initStatesFile)
        map_vars = get_column_mapping(initVarsFile)

        states_data = np.genfromtxt(initStatesFile)
        vars_data = np.genfromtxt(initVarsFile)

        idx_time = -1 # 0
        for i in range(nVess): 
            nameV = Segments[i][0]
            idxV = Segments[i][1]

            flow_var = f"{nameV}/v"
            press_var = f"{nameV}/u"

            flow_is_state = True
            press_is_state = True
            idx_flow = -1
            idx_press = -1

            idx_flow = map_states.get(flow_var, -1)
            if idx_flow==-1:
                idx_flow = map_vars.get(flow_var, -1)
                if idx_flow==-1:
                    sys.exit(f"1d solver :: Error: flow variable not found for vessel {nameV} in the initial states/variables files: {idx_flow}.")
                    # Qinit = 0.
                flow_is_state = False

            idx_press = map_states.get(press_var, -1)
            if idx_press==-1:
                idx_press = map_vars.get(press_var, -1)
                if idx_press==-1:
                    sys.exit(f"1d solver :: Error: pressure variable not found for vessel {nameV} in the initial states/variables files: {idx_press}.")
                    # Pinit = 90.*mmHgDyncm2
                press_is_state = False

            if idx_flow!=-1:
                if flow_is_state:
                    Qinit = states_data[idx_time,idx_flow]/convQ
                else:
                    Qinit = vars_data[idx_time,idx_flow]/convQ
            if idx_press!=-1:
                if press_is_state:
                    Pinit = states_data[idx_time,idx_press]/convP
                else:
                    Pinit = vars_data[idx_time,idx_press]/convP
            Ainit = mod.aFp(Pinit, K[i], A0[i], mTL[i], nTL[i], vessP0[i])
            # print(idxV, nameV, idx_flow, Qinit, idx_press, Pinit/mmHgDyncm2, Ainit, K[i], A0[i])

            AICvess = np.zeros_like(v[i].Q[:,0])
            AICvess[:] = Ainit
            QICvess = np.zeros_like(v[i].Q[:,1])
            QICvess[:] = Qinit
            v[i].setInitialConditions(ICtype, Pinit, Qinit, AICvess, QICvess, ICstate)
    
    else:
        Qinit = 0. # [ml/s] zero initial flow rate
        Pinit = np.zeros_like(vessP0)
        if ICtype==0:
            pass
        elif ICtype==1:
            for i in range(nVess): 
                if vessTy[i]==1:
                    Pinit[i] = PiniA
                else:
                    Pinit[i] = PiniV
        elif ICtype==2:
            Pinit = np.copy(vessP0) # [dyne/cm^2] initial pressure
        else:
            sys.exit(f"1d solver :: Error: wrong IC type assigned {ICtype}")
        
        # set initial data for vessels 
        for i in range(nVess): 
            # v[i].Q[:,0] = Ainit[i]
            # v[i].Q[:,1] = Qinit
            if ICstate==-1:
                AICvess = np.zeros_like(v[i].Q[:,0])
                QICvess = np.zeros_like(v[i].Q[:,1])
                v[i].setInitialConditions(ICtype, Pinit[i], Qinit, AICvess, QICvess)
            elif ICstate==1:
                AICvess = np.copy(AICstate[i])
                QICvess = np.copy(QICstate[i])
                v[i].setInitialConditions(ICtype, Pinit[i], Qinit, AICvess, QICvess, ICstate)

            
    # set initial pressure for terminal elements
    if (len(T)>0 and termType=='RCR'):      
        for i in IdxsTerm:
            iV = Nodes[i][2]
            if ICstate==-1:
                T[i].Qsol[0] = Pinit[iV]
                T[i].x = np.copy(T[i].Qsol)
            elif ICstate==1:
                if Nodes[i][3]==1: # outlet
                    AStar = v[iV].Q[-1,0]
                    qStar = v[iV].Q[-1,1]
                elif Nodes[i][3]==-1: # inlet
                    AStar = v[iV].Q[0,0]
                    qStar = v[iV].Q[0,1]
                    pStar = mod.pFa(AStar, K[iV], A0[iV], mTL[iV], nTL[iV], vessP0[iV])
                if Nodes[i][3]==1: # outlet 
                    T[i].Qsol[0] = pStar-Rwk1[iV]*qStar
                elif Nodes[i][3]==-1: # inlet
                    T[i].Qsol[0] = pStar+Rwk1[iV]*qStar
                T[i].x = np.copy(T[i].Qsol)


    if is_coupled and couple_volume_sum:
        # compute and send initial 1d total volume to 0d solver via the coupler 
        # to compute total (0d+1d) volume in the system
        VsumTot = 0.
        for i in range(nVess): 
            num.computeVesselVolume(v[i])
            VsumTot += v[i].V

        one_data_vol = np.zeros(DATA_LENGTH, dtype=np.float64)
        one_data_vol[0] = VsumTot
        write_pipe_vol.write(one_data_vol.tobytes())
        write_pipe_vol.flush()

   
    # store the numerical solution at initial time
    for i in range(nVess):
        if np.abs(tIni-tSaveIni)<=tolTime:
            v[i].outputResults(tIni)        
            if saveResSpace:
                if np.abs(tIni-tSaveIniSpace)<=tolTime:
                    v[i].outputResultsSpace(tIni)
     
    if (len(T)>0 and termType=='RCR'):
        for i in IdxsTerm:
            if np.abs(tIni-tSaveIni)<=tolTime:
                T[i].outputResults(tIni, T[i].Qsol)
    

    # ITERATION
    run_one = True
    time = tIni
    it = 0
    dt = 0.
    dtEvol = 0.
    dtEvol_old = 0.
    print(f'1d solver :: Initialization complete and successful with initial time : {time}')
    print(' ')

    # Time loop
    start = timing.time()

    while run_one==True: # for it in range(NMAX):
        
        try:
            # Compute time step
            dtLoc = np.zeros(nVess)
            for i in range(nVess):
                dtLoc[i] = num.timeStep(v[i].Q, v[i].dx, K[i], A0[i], mTL[i], nTL[i])        
            dt = np.min(dtLoc)
            if dt<=tolTime:
                sys.exit(f"1d solver :: Error: 1D time step {dt} smaller than tolerance {tolTime}")
            if time+dt>tEnd:
                dt = tEnd - time  
            # if time>=tEndGlob:
            #     break
            
            print("1d solver :: computed dt : %.8e" % dt)
            
            if not is_coupled:
                # if time+dt>tEnd:
                #     dt = tEnd - time   
                print("1d solver :: IT : %i" % it, " || time : %.8e" % time, " || dt : %.8e" % dt)

                for i in range(nVess):
                    v[i].FnumQBCL[:] = 0.
                    v[i].FnumQBCR[:] = 0.
                
                couplingFunc.solve_time_step_1D(time, dt, T0, v, T, Idxs1D, IdxsJhyb, IdxsTerm, nVess, nNode, nVar1D, FVord, numFlux, slopeType, termType, K, A0, mu, mTL, nTL, vessP0, Rwk1, Cwk, Rwk2, Pv)
            
            else:
                dt0D = -1.0
                # matching global time step between 1d and 0d
                # wait for and read 0d time step data from coupler
                parent_data_bytes = read_pipe_dt.read(FLOAT_SIZE * DATA_LENGTH)
                if len(parent_data_bytes) != FLOAT_SIZE * DATA_LENGTH:
                    sys.exit(f"Error: Expected {FLOAT_SIZE * DATA_LENGTH} bytes but got {len(parent_data_bytes)} for time step read pipe", file=sys.stderr)
                
                parent_data_dt = np.frombuffer(parent_data_bytes, dtype=np.float64)
                dt0D = parent_data_dt[1]
                
                dt = np.minimum(dt,dt0D)
                # if time+dt>tEnd:
                #     dt = tEnd - time 

                one_data_dt = np.zeros(DATA_LENGTH, dtype=np.float64)
                one_data_dt[0] = time
                one_data_dt[1] = dt
                # send back time step to 0d solver
                write_pipe_dt.write(one_data_dt.tobytes())
                write_pipe_dt.flush()

                print("1d solver :: IT : %i" % it, " || time : %.8e" % time, " || dt : %.8e" % dt)
                
                timeLoc = time
                dtLoc = 0.0
                if ODEsolver=='CVODE':
                    dtFrac = []
                    weights = []

                for i in range(nVess):
                    if ODEsolver=="PETSC":
                        v[i].FnumQBCL_old[:] = np.copy( v[i].FnumQBCL_new[:] )
                        v[i].FnumQBCR_old[:] = np.copy( v[i].FnumQBCR_new[:] )
                    v[i].FnumQBCL[:] = 0.
                    v[i].FnumQBCR[:] = 0.
                
                count = 0
                while True:
                    # wait for and read 0d solver internal time step data from coupler
                    parent_data_bytes = read_pipe_dt.read(FLOAT_SIZE * DATA_LENGTH)
                    if len(parent_data_bytes) != FLOAT_SIZE * DATA_LENGTH:
                        sys.exit(f"Error: Expected {FLOAT_SIZE * DATA_LENGTH} bytes but got {len(parent_data_bytes)} for time step read pipe", file=sys.stderr)
                    
                    parent_data_dt = np.frombuffer(parent_data_bytes, dtype=np.float64)
                    timeLoc = parent_data_dt[0]
                    dtLoc = parent_data_dt[1]

                    if dtLoc<0.:
                        # print(f"1d solver :: Received negative 0d internal time step dtLoc {dtLoc}")
                        write_pipe_dt.write(parent_data_dt.tobytes())
                        write_pipe_dt.flush()
                        break
                    
                    
                    if ODEsolver=="CVODE":
                        if dtLoc<tolTime:
                            weights.append( 0. )
                        else:
                            weights.append( dtLoc/dt )

                        if round(np.sum(weights),8)>1.0:
                            print("1d solver :: ERROR: Sum of integration weights > 1 : ", np.sum(weights), round(np.sum(weights),8), weights)
                            sys.exit()

                    # print("1d solver :: sub-step ", count, " || ", time, dt, " || ", timeLoc, dtLoc, " || ",  (time+dt), timeLoc)

                    parent_data = []
                    one_data = []

                    for i in range(N1d0d):
                        pipeID = str(i+1)
                        if "port_volume_sum" in pipe_1d_0d_info[pipeID]:
                            if pipe_1d_0d_info[pipeID]["port_volume_sum"] == 1:
                                print("ERROR :: this is a volume sum port, use its dedicated pipes for this connection. Exiting.")
                                sys.exit()
                        
                        # wait for and read data from coupler for each 1d-0d connection
                        # Read data as numpy array
                        parent_data_bytes = read_pipe[i].read(FLOAT_SIZE * DATA_LENGTH)
                        if len(parent_data_bytes) != FLOAT_SIZE * DATA_LENGTH:
                            sys.exit(f"Error: Expected {FLOAT_SIZE * DATA_LENGTH} bytes but got {len(parent_data_bytes)} for pipe "+pipeID, file=sys.stderr)
                        
                        parent_data.append( np.frombuffer(parent_data_bytes, dtype=np.float64) )
                        
                    if FVord==2:
                        if ODEsolver=="CVODE":
                            # # dtEvol = (timeLoc+dtLoc)-time
                            # dtEvol = timeLoc-time
                            # # if i==0:
                            # #     print(ODEsolver+" with dtEvol="+str(dtEvol))
                            if dtLoc<tolTime:
                                dtEvol = dtEvol_old #XXX because CVODE did not take a step
                            else:
                                dtEvol = timeLoc-time #XXX because timeLoc is already the advanced time attempted by CVODE 
                        else:
                            # dtEvol = dtFrac[count]*dt
                            dtEvol = dtLoc
                        dtEvol_old = dtEvol
                        

                    for i in range(N1d0d):
                        pipeID = str(i+1)
                        # print(pipeID)
                        idxV = pipe_1d_0d_info[pipeID]["vess1d_idx"]
                        idxBC = pipe_1d_0d_info[pipeID]["vess1d_bc_in0_or_out1"]
                        if idxBC==0: # inlet 
                            g1D = -1.0
                            idxJ = Segments[idxV][2]
                            typeJ = Segments[idxV][3]
                        elif idxBC==1: # outlet
                            g1D = 1.0
                            idxJ = Segments[idxV][4]
                            typeJ = Segments[idxV][5]
                        type_0Dport = pipe_1d_0d_info[pipeID]["port_flow0_or_press1"]
                        type_0Dbc = pipe_1d_0d_info[pipeID]["cellml_bc_flow0_or_press1"]
                        idx_0Dbc = pipe_1d_0d_info[pipeID]["cellml_bc_in0_or_out1"]

                        if (typeJ=='juncHyb' and Nodes[idxJ][1]=='juncHyb'):
                            nVJ = Nodes[idxJ][-1]
                        else:
                            sys.exit(f"1d solver :: Error: junction type mismatch for 1d-0d coupling at node {idxJ} : {typeJ} & {Nodes[idxJ][1]}")
                            
                        if nVJ==1:
                            if type_0Dbc==0 and type_0Dport==1: # 0d flow bc --> 0d pressure port --> receiving pressure and resistance
                                P0D = parent_data[i][0]/convP
                                Rwk = parent_data[i][1]/convR
                                if FVord==1:
                                    if idxBC==0: # inlet
                                        QStar1D, Q0D = couplingFunc.coupler1D_0DflowBC_o1(P0D, Rwk, g1D, v[idxV].dx, v[idxV].Q[0,:], K[idxV], A0[idxV], mTL[idxV], nTL[idxV], vessP0[idxV], 0)
                                    elif idxBC==1: # outlet
                                        QStar1D, Q0D = couplingFunc.coupler1D_0DflowBC_o1(P0D, Rwk, g1D, v[idxV].dx, v[idxV].Q[-1,:], K[idxV], A0[idxV], mTL[idxV], nTL[idxV], vessP0[idxV], 0)
                                elif FVord==2:
                                    if idxBC==0: # inlet
                                        QStar1D, Q0D = couplingFunc.coupler1D_0DflowBC_o2(P0D, Rwk, g1D, v[idxV].dx, dtEvol, [v[idxV].Q[0,:],v[idxV].Q[1,:]], K[idxV], A0[idxV], mTL[idxV], nTL[idxV], vessP0[idxV], mu[idxV])
                                    elif idxBC==1: # outlet
                                        QStar1D, Q0D = couplingFunc.coupler1D_0DflowBC_o2(P0D, Rwk, g1D, v[idxV].dx, dtEvol, [v[idxV].Q[-2,:],v[idxV].Q[-1,:]], K[idxV], A0[idxV], mTL[idxV], nTL[idxV], vessP0[idxV], mu[idxV])
                        
                            elif type_0Dbc==1 and type_0Dport==0: # 0d pressure bc --> 0d flow port --> receiving flow
                                Q0D = parent_data[i][0]/convQ
                                if FVord==1:
                                    if idxBC==0: # inlet
                                        QStar1D, P0D = couplingFunc.coupler1D_0DpressureBC_o1(Q0D, g1D, v[idxV].dx, v[idxV].Q[0,:], K[idxV], A0[idxV], mTL[idxV], nTL[idxV], vessP0[idxV], 0)
                                    elif idxBC==1: # outlet
                                        QStar1D, P0D = couplingFunc.coupler1D_0DpressureBC_o1(Q0D, g1D, v[idxV].dx, v[idxV].Q[-1,:], K[idxV], A0[idxV], mTL[idxV], nTL[idxV], vessP0[idxV], 0)
                                elif FVord==2:
                                    if idxBC==0: # inlet
                                        QStar1D, P0D = couplingFunc.coupler1D_0DpressureBC_o2(Q0D, g1D, v[idxV].dx, dtEvol, [v[idxV].Q[0,:],v[idxV].Q[1,:]], K[idxV], A0[idxV], mTL[idxV], nTL[idxV], vessP0[idxV], mu[idxV])
                                    elif idxBC==1: # outlet
                                        QStar1D, P0D = couplingFunc.coupler1D_0DpressureBC_o2(Q0D, g1D, v[idxV].dx, dtEvol, [v[idxV].Q[-2,:],v[idxV].Q[-1,:]], K[idxV], A0[idxV], mTL[idxV], nTL[idxV], vessP0[idxV], mu[idxV])

                        else:
                            if type_0Dbc==0 and type_0Dport==1: # 0d flow bc --> 0d pressure port --> receiving pressure and resistance
                                sys.exit(f"1d solver :: Error: hybrid junction with nVJ={nVJ} 1D vessels and 0D pressure port (flow BC) still to be implemented. Exiting.") 
                            
                            elif type_0Dbc==1 and type_0Dport==0: # 0d pressure bc --> 0d flow port --> receiving flow
                                Q0D = parent_data[i][0]/convQ
                                vessJ = []
                                gJ = []
                                for j in range(nVJ):
                                    vessJ.append( Nodes[idxJ][2+j*2] )
                                    gJ.append( Nodes[idxJ][2+j*2+1] )

                                if FVord==1:
                                    Q1DJ = []
                                    KJ = []
                                    A0J = []
                                    mTLJ = []
                                    nTLJ = []
                                    P0J = []
                                    for j in range(nVJ):
                                        iV = vessJ[j]
                                        KJ.append(K[iV])
                                        A0J.append(A0[iV])
                                        mTLJ.append(mTL[iV])
                                        nTLJ.append(nTL[iV])
                                        P0J.append(vessP0[iV])
                                        if gJ[j]==-1:
                                            Q1DJ.append( v[iV].Q[0,:] )
                                        elif gJ[j]==1:
                                            Q1DJ.append( v[iV].Q[-1,:] )
                                    
                                    if idx_0Dbc==0: # inlet
                                        g0D = -1
                                    elif idx_0Dbc==1: # outlet
                                        g0D = 1
                                    
                                    QStar1D_multi, P0D = couplingFunc.coupler1Dmulti_0D_o1(nVJ, Q1DJ, gJ, Q0D, g0D, KJ, A0J, mTLJ, nTLJ, P0J)
                                
                                elif FVord==2:
                                    dxJ = []
                                    Q1DJ = []
                                    KJ = []
                                    A0J = []
                                    muJ = []
                                    mTLJ = []
                                    nTLJ = []
                                    P0J = []
                                    for j in range(nVJ):
                                        iV = vessJ[j]
                                        dxJ.append(v[iV].dx)
                                        KJ.append(K[iV])
                                        A0J.append(A0[iV])
                                        muJ.append(mu[iV])
                                        mTLJ.append(mTL[iV])
                                        nTLJ.append(nTL[iV])
                                        P0J.append(vessP0[iV])
                                        if gJ[j]==-1:
                                            Q1DJ.append( [v[iV].Q[0,:], v[iV].Q[1,:]] )
                                        elif gJ[j]==1:
                                            Q1DJ.append( [v[iV].Q[-2,:], v[iV].Q[-1,:]] )
                                    
                                    if idx_0Dbc==0: # inlet
                                        g0D = -1
                                    elif idx_0Dbc==1: # outlet
                                        g0D = 1
                                        
                                    QStar1D_multi, P0D = couplingFunc.coupler1Dmulti_0D_o2(dtEvol, nVJ, Q1DJ, gJ, dxJ, Q0D, g0D, KJ, A0J, mTLJ, nTLJ, P0J, muJ)
 
                        one_data.append( np.zeros(DATA_LENGTH, dtype=np.float64) )
                        if type_0Dport==1: # 0d flow bc / pressure port 
                            one_data[i][0] = Q0D*convQ # sending back computed flow rate
                        elif type_0Dport==0: # 0d pressure bc / flow port
                            one_data[i][0] = P0D*convP # sending back computed pressure 
                    
                        # write and send to coupler data with computed 0d coupling variables for each 1d-0d connection
                        # Write the numpy array as binary
                        write_pipe[i].write(one_data[i].tobytes())
                        write_pipe[i].flush()

                        if FVord==1:
                            if (ODEsolver=="explEul" and count==0):
                                # print(ODEsolver+" with count="+str(count)+" || dtLoc="+str(dtLoc)+" || weight="+str(weights[count]))
                                if nVJ==1:
                                    if idxBC==0:
                                        v[idxV].FnumQBCL[:] = weights[count]*mod.pF(QStar1D, K[idxV], A0[idxV], mTL[idxV], nTL[idxV])
                                    elif idxBC==1:
                                        v[idxV].FnumQBCR[:] = weights[count]*mod.pF(QStar1D, K[idxV], A0[idxV], mTL[idxV], nTL[idxV])
                                else:
                                    for j in range(nVJ):
                                        iV = vessJ[j]
                                        if gJ[j]==-1:
                                            v[iV].FnumQBCL[:] = weights[count]*mod.pF(QStar1D_multi[j,:], K[iV], A0[iV], mTL[iV], nTL[iV])
                                        elif gJ[j]==1:
                                            v[iV].FnumQBCR[:] = weights[count]*mod.pF(QStar1D_multi[j,:], K[iV], A0[iV], mTL[iV], nTL[iV])

                            elif ODEsolver=="PETSC" and PETSC_SOLVER=="BDF1":
                                if nVJ==1:
                                    if idxBC==0:
                                        v[idxV].FnumQBCL[:] = mod.pF(QStar1D, K[idxV], A0[idxV], mTL[idxV], nTL[idxV])
                                    elif idxBC==1:
                                        v[idxV].FnumQBCR[:] = mod.pF(QStar1D, K[idxV], A0[idxV], mTL[idxV], nTL[idxV])
                                else:
                                    for j in range(nVJ):
                                        iV = vessJ[j]
                                        if gJ[j]==-1:
                                            v[iV].FnumQBCL[:] = mod.pF(QStar1D_multi[j,:], K[iV], A0[iV], mTL[iV], nTL[iV])
                                        elif gJ[j]==1:
                                            v[iV].FnumQBCR[:] = mod.pF(QStar1D_multi[j,:], K[iV], A0[iV], mTL[iV], nTL[iV])

                            else:
                                sys.exit("FVord=1 + ODEsolver="+str(ODEsolver)+" with count "+str(count))

                        elif FVord==2:
                            if (ODEsolver=="Heun" or ODEsolver=="midpoint"):
                                if (count==0 or count==1):
                                    if nVJ==1:
                                        if idxBC==0:
                                            v[idxV].FnumQBCL[:] = v[idxV].FnumQBCL[:] + weights[count]*mod.pF(QStar1D, K[idxV], A0[idxV], mTL[idxV], nTL[idxV])
                                        elif idxBC==1:
                                            v[idxV].FnumQBCR[:] = v[idxV].FnumQBCR[:] + weights[count]*mod.pF(QStar1D, K[idxV], A0[idxV], mTL[idxV], nTL[idxV])
                                    else:
                                        for j in range(nVJ):
                                            iV = vessJ[j]
                                            if gJ[j]==-1:
                                                v[iV].FnumQBCL[:] = v[iV].FnumQBCL[:] + weights[count]*mod.pF(QStar1D_multi[j,:], K[iV], A0[iV], mTL[iV], nTL[iV])
                                            elif gJ[j]==1:
                                                v[iV].FnumQBCR[:] = v[iV].FnumQBCR[:] + weights[count]*mod.pF(QStar1D_multi[j,:], K[iV], A0[iV], mTL[iV], nTL[iV])
                                else:
                                    sys.exit("FVord=2 + ODEsolver="+str(ODEsolver)+" with count "+str(count))

                            elif ODEsolver=="RK4":
                                if count<4:
                                    if nVJ==1:
                                        if idxBC==0:
                                            v[idxV].FnumQBCL[:] = v[idxV].FnumQBCL[:] + weights[count]*mod.pF(QStar1D, K[idxV], A0[idxV], mTL[idxV], nTL[idxV])
                                        elif idxBC==1:
                                            v[idxV].FnumQBCR[:] = v[idxV].FnumQBCR[:] + weights[count]*mod.pF(QStar1D, K[idxV], A0[idxV], mTL[idxV], nTL[idxV])
                                    else:
                                        for j in range(nVJ):
                                            iV = vessJ[j]
                                            if gJ[j]==-1:
                                                v[iV].FnumQBCL[:] = v[iV].FnumQBCL[:] + weights[count]*mod.pF(QStar1D_multi[j,:], K[iV], A0[iV], mTL[iV], nTL[iV])
                                            elif gJ[j]==1:
                                                v[iV].FnumQBCR[:] = v[iV].FnumQBCR[:] + weights[count]*mod.pF(QStar1D_multi[j,:], K[iV], A0[iV], mTL[iV], nTL[iV])
                                else:
                                    sys.exit("FVord=2 + ODEsolver="+str(ODEsolver)+" with count "+str(count))

                            elif ODEsolver=="CVODE":
                                if nVJ==1:
                                    if idxBC==0:
                                        v[idxV].FnumQBCL[:] = v[idxV].FnumQBCL[:] + weights[count]*mod.pF(QStar1D, K[idxV], A0[idxV], mTL[idxV], nTL[idxV])
                                    elif idxBC==1:
                                        v[idxV].FnumQBCR[:] = v[idxV].FnumQBCR[:] + weights[count]*mod.pF(QStar1D, K[idxV], A0[idxV], mTL[idxV], nTL[idxV])
                                else:
                                    for j in range(nVJ):
                                        iV = vessJ[j]
                                        if gJ[j]==-1:
                                            v[iV].FnumQBCL[:] = v[iV].FnumQBCL[:] + weights[count]*mod.pF(QStar1D_multi[j,:], K[iV], A0[iV], mTL[iV], nTL[iV])
                                        elif gJ[j]==1:
                                            v[iV].FnumQBCR[:] = v[iV].FnumQBCR[:] + weights[count]*mod.pF(QStar1D_multi[j,:], K[iV], A0[iV], mTL[iV], nTL[iV])

                            elif ODEsolver=="PETSC":
                                if PETSC_SOLVER=="CN":
                                    if nVJ==1:
                                        if idxBC==0:
                                            v[idxV].FnumQBCL_new[:] = mod.pF(QStar1D, K[idxV], A0[idxV], mTL[idxV], nTL[idxV])
                                            v[idxV].FnumQBCL[:] = weights[0]*v[idxV].FnumQBCL_old[:] + weights[1]*v[idxV].FnumQBCL_new[:]
                                        elif idxBC==1:
                                            v[idxV].FnumQBCR_new[:] = mod.pF(QStar1D, K[idxV], A0[idxV], mTL[idxV], nTL[idxV])
                                            v[idxV].FnumQBCR[:] = weights[0]*v[idxV].FnumQBCR_old[:] + weights[1]*v[idxV].FnumQBCR_new[:]
                                    else:
                                        for j in range(nVJ):
                                            iV = vessJ[j]
                                            if gJ[j]==-1:
                                                v[iV].FnumQBCL_new[:] = mod.pF(QStar1D_multi[j,:], K[iV], A0[iV], mTL[iV], nTL[iV])
                                                v[iV].FnumQBCL[:] = weights[0]*v[iV].FnumQBCL_old[:] + weights[1]*v[iV].FnumQBCL_new[:]
                                            elif gJ[j]==1:
                                                v[iV].FnumQBCR_new[:] = mod.pF(QStar1D_multi[j,:], K[iV], A0[iV], mTL[iV], nTL[iV])
                                                v[iV].FnumQBCR[:] = weights[0]*v[iV].FnumQBCR_old[:] + weights[1]*v[iV].FnumQBCR_new[:]

                                elif PETSC_SOLVER=="BDF2": #XXX TODO
                                    sys.exit(f"Coupling between FVord=2 and ODEsolver={ODEsolver} with {PETSC_SOLVER} not yet implemented. Exiting.")

                            else:
                                sys.exit("FVord=2 + ODEsolver="+str(ODEsolver)+" with count "+str(count))

                    count +=1

                couplingFunc.solve_time_step_1D(time, dt, T0, v, T, Idxs1D, IdxsJhyb, IdxsTerm, nVess, nNode, nVar1D, FVord, numFlux, slopeType, termType, K, A0, mu, mTL, nTL, vessP0, Rwk1, Cwk, Rwk2, Pv, is_coupled)

                if couple_volume_sum:
                    # compute and send initial 1d total volume to 0d solver via the coupler 
                    # to compute total (0d+1d) volume in the system
                    VsumTot = 0.
                    for i in range(nVess): 
                        num.computeVesselVolume(v[i])
                        VsumTot += v[i].V

                    one_data_vol = np.zeros(DATA_LENGTH, dtype=np.float64)
                    one_data_vol[0] = VsumTot
                    write_pipe_vol.write(one_data_vol.tobytes())
                    write_pipe_vol.flush()

            # UPDATE TIME
            it +=1
            time = time + dt
            time = round(time/tolTime)*tolTime
            
            saveResTime = 0
            # if np.abs(time-tEnd)<=tolTime:
            if time>=tEnd-tolTime:
                tEnd = tEnd + dtSample
                tEnd = round(tEnd/tolTime)*tolTime
                saveResTime = 1
        
            # STORE SOLUTION IN TIME & SPACE
            if saveResTime:
                # if (time>=tSaveIni and time<=tSaveEnd):
                if time>=tSaveIni:
                    for i in range(nVess):
                        v[i].outputResults(time)
                        if saveResSpace:
                            if (time>=tSaveIniSpace and time<=tSaveEndSpace):
                                v[i].outputResultsSpace(time)
                    if (len(T)>0 and termType=='RCR'):
                        for j in IdxsTerm:
                            T[j].outputResults(time, T[j].Qsol)

            
            if (time>=tEndGlob-1e-8):
                print("### 1d solver :: Stop execution! ###")
                run_one = False
        
        except Exception as e:
            print(f"1d solver :: Error: {e}", file=sys.stderr)
            break


    if is_coupled:
        pipe.closePipes(write_pipe_dt, 1)
        pipe.closePipes(read_pipe_dt, 1)
        pipe.closePipes(write_pipe, N1d0d)
        pipe.closePipes(read_pipe, N1d0d)
        pipe.closePipes(write_pipe_vol, 1)
        print("### 1d solver :: All pipes closed. ###")
    
    for i in range(nVess):
        v[i].outFile1.close()
        if saveResSpace:
            v[i].outFileQ.close()
            v[i].outFileP.close()
            v[i].outFileA.close()
    if termType=='RCR':
        for i in IdxsTerm:
            T[i].outFile1.close()     
        
    end = timing.time()
    CPUtime = end - start
    print(f"1d solver :: 1D final time : {round(time,5)} s || Number of cardiac cycles : {int(round(time,5)/T0)}")
    print(f"1d solver :: Number of 1D iterations : {it}")
    print(f"1d solver :: 1D CPU time : {CPUtime} s") # in seconds (since the epoch)    
    # print('#######################################################################')


if __name__=='__main__':
    # HOW TO EXECUTE: python main1D.py "<path_to_file>/input.ini"
    run1DBFmod()
  
