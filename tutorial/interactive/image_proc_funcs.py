#!/usr/bin/env python

#####################
### // Imports // ###
#####################

import sys
import os

# --- CRITICAL FIX: PREVENT DEADLOCKS ---
# Force all math libraries to use 1 thread to stop freezing
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
# Force Matplotlib to not use a window
os.environ['MPLBACKEND'] = 'Agg' 
# ---------------------------------------

# --- DEBUG: Start ---
print("--- [DEBUG] Starting Imports in image_proc_funcs.py ---", file=sys.stderr)

try:
    print("--- [DEBUG] Importing numpy...", file=sys.stderr)
    import numpy as np
    
    print("--- [DEBUG] Importing skimage...", file=sys.stderr)
    import skimage
    from skimage.transform import rescale, resize
    
    # print("--- [DEBUG] Importing vedo...", file=sys.stderr)
    # import vedo
    # # CRITICAL FIX: Prevent vedo from looking for a window manager
    # vedo.settings.default_backend = 'k3d' 
    # vedo.settings.screeshot_large_image = True
    
    print("--- [DEBUG] Importing tqdm...", file=sys.stderr)
    from tqdm import tqdm
    
    print("--- [DEBUG] Importing h5py...", file=sys.stderr)
    import h5py
    
    # print("--- [DEBUG] Importing tifffile...", file=sys.stderr)
    # import tifffile
    
    print("--- [DEBUG] Importing scipy...", file=sys.stderr)
    from scipy import ndimage
    from scipy.spatial import distance_matrix
    
    print("--- [DEBUG] Importing pandas...", file=sys.stderr)
    import pandas as pd
    
    print("--- [DEBUG] Importing ast...", file=sys.stderr)
    from ast import literal_eval
    
    print("--- [DEBUG] Importing standard libs (copy, random, json, re, subprocess, glob)...", file=sys.stderr)
    from copy import deepcopy
    import random
    import json
    import re
    import subprocess
    from glob import glob
    
    # print("--- [DEBUG] Importing networkx...", file=sys.stderr)
    # import networkx as nx
    
    print("--- [DEBUG] Importing pathlib...", file=sys.stderr)
    from pathlib import Path

    # --- FIX: GLOBAL PATH SETUP (Hoisted from generate_parameter_array) ---
    # We set this up ONCE at the start to prevent hanging inside functions.
    root_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(root_dir) # Go up one level
    root_dir = os.path.dirname(root_dir) # Go up another level (to reach project root)
    
    src_path = os.path.join(root_dir, 'src')
    if src_path not in sys.path:
        sys.path.append(src_path)

    # NOTE: If you have imports from other local files, add them here too
    # print("--- [DEBUG] Importing local modules...", file=sys.stderr)
    # from generate_param_array import VesselNetwork

    print("--- [DEBUG] ALL IMPORTS SUCCESSFUL ---", file=sys.stderr)

except Exception as e:
    print(f"\n!!! [CRITICAL ERROR] Import crashed: {e} !!!\n", file=sys.stderr)
    raise e

# # from generate_vessel_array import *
# from generate_param_array import *

###############################
### // Class Definitions // ###
###############################

class VesselNetwork():
    
    def __init__(self, C_vessel, vessel_names, vessel_mods, vessel_centroids=None):

            self.C_vessel = C_vessel
            self.vessel_names = vessel_names
            self.vessel_mods = vessel_mods
            self.vessel_centroids = vessel_centroids
            self.vessel_df = None
            self.parameter_df = None

    def generate_vessel_array(self):

        # --- LAZY LOAD ---
        import pandas as pd
        import networkx as nx
        from copy import copy
        import time
        import random

        random.seed(42)
        np.random.seed(42)
        # -----------------

        print('Generating Vessel Array...')

        #####################################
        ### // Define helper functions // ###
        #####################################

        def is_compatible(BC_n, BC_in, BC_out):

            BC_n = list(BC_n)
            BC_in = list(BC_in)
            BC_out = list(BC_out)

            if BC_n == 'nn':
                return True
            elif (BC_in == 'nn') and (BC_n[1] != BC_out[0]):
                return True
            elif (BC_in[1] != BC_n[0]) and (BC_out == 'nn'):
                return True
            else:
                return (BC_in[1] != BC_n[0]) and (BC_n[1] != BC_out[0])
            
        ##################################
        ### // Define main function // ###
        ##################################

        # ============================================================
        # Sanitize Adjacency Matrix: Remove Bidirectional Connections
        # ============================================================
        # This prevents the "Short Circuit" error where two vessels 
        # are inputs and outputs to each other.
        
        # Find all pairs (i, j) where both i->j and j->i exist
        rows, cols = np.where((self.C_vessel == 1) & (self.C_vessel.T == 1))
        
        for r, c in zip(rows, cols):
            # Only process each pair once (when row < col)
            if r < c: 
                print(f"  [WARNING] Found bidirectional loop between Vessel {r} ({self.vessel_names[r]}) "
                      f"and Vessel {c} ({self.vessel_names[c]}).")
                print(f"  -> Fixing by removing connection {c} -> {r} (Enforcing {r} -> {c})")
                
                # Arbitrarily cut the 'return' path to enforce DAG (Directed Acyclic Graph)
                # You could add logic here to keep the direction based on Z-height if needed
                self.C_vessel[c, r] = 0 

        # Also remove self-loops (vessel connected to itself)
        np.fill_diagonal(self.C_vessel, 0)
        # ============================================================

        # # ============================================================
        # # NEW FIX: Remove Closed Loops (Cycles)
        # # ============================================================
        # # Docker math differences can create loops (A->B->C->A) that don't exist locally.
        # # These loops confuse the solver. We must find and break them.
        
        # print("Sanitizing Graph Topology (Cycle Removal)...")
        # # Create a temporary graph to find cycles
        # G_temp = nx.DiGraph(self.C_vessel)
        
        # # Find cycles (this is fast for small loops)
        # try:
        #     cycles = list(nx.simple_cycles(G_temp))
        #     if len(cycles) > 0:
        #         print(f"  [WARNING] Found {len(cycles)} closed loops (cycles) in the network.")
        #         print("  -> Breaking cycles to prevent solver stalemate...")
                
        #         for cycle in cycles:
        #             # cycle is a list of nodes, e.g., [A, B, C] meaning A->B->C->A
        #             if len(cycle) > 1:
        #                 # Arbitrarily cut the connection between the first two nodes
        #                 u, v = cycle[0], cycle[1]
        #                 self.C_vessel[u, v] = 0
        #                 print(f"     * Cut connection {u} -> {v}")
                        
        #         print("  Cycle removal complete.")
        #     else:
        #         print("  No cycles found. Topology is clean.")
                
        # except Exception as e:
        #     print(f"  [Error] Cycle detection failed: {e}")
        # # ============================================================

        ### // Initialise Variables // ###

        # TODO FIX THIS!!!
        # temporarily remove self-loops 
        # remove any inputs to the first node (index 0)
        first_node_idx = 0
        first_node_inp = np.where(self.C_vessel[:, first_node_idx] != 0)[0]
        if first_node_inp.size > 0:
            self.C_vessel[first_node_inp, first_node_idx] = 0

        n_vessel = self.C_vessel.shape[0]
        n_vessel_idx = np.arange(0, n_vessel)

        vessel_dict = {'name': self.vessel_names,
                       'BC_type': np.repeat(None, n_vessel),
                       'vessel_type': self.vessel_mods,
                       'inp_vessels': np.repeat(None, n_vessel),
                       'out_vessels': np.repeat(None, n_vessel),
                       'x': self.vessel_centroids[:, 0] if self.vessel_centroids is not None else np.nan, # [NEW]
                       'y': self.vessel_centroids[:, 1] if self.vessel_centroids is not None else np.nan, # [NEW]
                       'z': self.vessel_centroids[:, 2] if self.vessel_centroids is not None else np.nan, # [NEW]
                       'junc_type': np.repeat(None, n_vessel)} ### Debugging purposes only
        
        self.vessel_df = pd.DataFrame(vessel_dict)

        n_vessel_in_out_idx = np.array([]).astype(int)
        # n_vessel_in_only_idx = np.array([]).astype(int)
        # n_vessel_out_only_idx = np.array([]).astype(int)
        n_vessel_Min_idx = np.array([]).astype(int)
        n_vessel_Nout_idx = np.array([]).astype(int)
        n_vessel_MinNout_idx = np.array([]).astype(int)

        N_ITER = 0
        MAX_ITER = 100

        ### // BC_type initial assignment // ###

        for i in n_vessel_idx:

            inp_vessel_i = self.C_vessel[:, i]
            out_vessel_i = self.C_vessel[i, :]

            inp_vessel_locs = np.where(inp_vessel_i == 1)[0]
            out_vessel_locs = np.where(out_vessel_i == 1)[0]

            if (len(inp_vessel_locs) == 0) & (len(out_vessel_locs) >= 1):
                self.vessel_df.at[i, 'BC_type'] = 'pv'
            elif (len(inp_vessel_locs) >= 1) & (len(out_vessel_locs) == 0):
                self.vessel_df.at[i, 'BC_type'] = 'vp'
            elif (len(inp_vessel_locs) == 1) and (len(out_vessel_locs) > 1):
                self.vessel_df.at[i, 'BC_type'] = 'pv'
            elif (len(inp_vessel_locs) > 1) and (len(out_vessel_locs) == 1):
                self.vessel_df.at[i, 'BC_type'] = 'vp'
            elif (len(inp_vessel_locs) == 1) and (len(out_vessel_locs) == 1):                    
                self.vessel_df.at[i, 'BC_type'] = 'vv'
            else:
                self.vessel_df.at[i, 'BC_type'] = 'vv'

        ### // BC-type re-assignment and correction // ###

        print('Assigning Network Boundary Conditions...')

        while N_ITER <= MAX_ITER:

            ### // (Re)-Assign and get idxs of in-out, and multi-in/out vessels respectively, and (re)-assign junc_types // ###

            n_vessel_in_out_idx = np.array([]).astype(int)
            # n_vessel_in_only_idx = np.array([]).astype(int)
            # n_vessel_out_only_idx = np.array([]).astype(int)
            n_vessel_Min_idx = np.array([]).astype(int)
            n_vessel_Nout_idx = np.array([]).astype(int)
            n_vessel_MinNout_idx = np.array([]).astype(int)


            if len(n_vessel_idx) != 0:
            
                for idx in n_vessel_idx:

                    inp_vessel_i = self.C_vessel[:, idx]
                    out_vessel_i = self.C_vessel[idx, :]

                    inp_vessel_locs = np.where(inp_vessel_i == 1)[0]
                    out_vessel_locs = np.where(out_vessel_i == 1)[0]

                    if (len(inp_vessel_locs) == 1) and (len(out_vessel_locs) == 0):
                        n_vessel_in_out_idx = np.append(n_vessel_in_out_idx, idx)
                        self.vessel_df.at[idx, 'junc_type'] = 'outlet'
                    elif (len(inp_vessel_locs) == 0) and (len(out_vessel_locs) == 1):
                        n_vessel_in_out_idx = np.append(n_vessel_in_out_idx, idx)
                        self.vessel_df.at[idx, 'junc_type'] = 'inlet'
                    elif (len(inp_vessel_locs) == 1) and (len(out_vessel_locs) == 1):
                        n_vessel_in_out_idx = np.append(n_vessel_in_out_idx, idx)
                        self.vessel_df.at[idx, 'junc_type'] = None
                    elif (len(inp_vessel_locs) > 1) and (len(out_vessel_locs) == 0):
                        n_vessel_Min_idx = np.append(n_vessel_Min_idx, idx)
                        self.vessel_df.at[idx, 'junc_type'] = 'Noutlet'
                    elif (len(inp_vessel_locs) == 0) and (len(out_vessel_locs) > 1):
                        n_vessel_Nout_idx = np.append(n_vessel_Nout_idx, idx)
                        self.vessel_df.at[idx, 'junc_type'] = 'Minlet'
                    elif (len(inp_vessel_locs) > 1) and (len(out_vessel_locs) == 1):
                        n_vessel_Min_idx = np.append(n_vessel_Min_idx, idx)
                        self.vessel_df.at[idx, 'junc_type'] = 'Min'
                    elif (len(inp_vessel_locs) == 1) and (len(out_vessel_locs) > 1):
                        n_vessel_Nout_idx = np.append(n_vessel_Nout_idx, idx)
                        self.vessel_df.at[idx, 'junc_type'] = 'Nout'
                    else:
                        n_vessel_MinNout_idx = np.append(n_vessel_MinNout_idx, idx)
                        self.vessel_df.at[idx, 'junc_type'] = 'MinNout'
                
                    inp_vessel_names = list(self.vessel_df.loc[inp_vessel_locs, 'name'])
                    out_vessel_names = list(self.vessel_df.loc[out_vessel_locs, 'name'])

                    self.vessel_df.at[idx, 'inp_vessels'] = ' '.join(inp_vessel_names) if len(inp_vessel_names) > 0 else None
                    self.vessel_df.at[idx, 'out_vessels'] = ' '.join(out_vessel_names) if len(out_vessel_names) > 0 else None

            ### // BC_type correction for in-out vessels // ###

            if len(n_vessel_in_out_idx) != 0:

                for idx in n_vessel_in_out_idx:

                    inp_vessel_i = self.C_vessel[:, idx] ### Get all input vessels of the current vessel
                    out_vessel_i = self.C_vessel[idx, :] ### Get all output vessels of the current vessel

                    inp_vessel_locs = np.where(inp_vessel_i == 1)[0] ### Get all idx's of all input vessels to the current vessel
                    out_vessel_locs = np.where(out_vessel_i == 1)[0] ### Get all idx's of all output vessels to the current vessel

                    n_vessel_BC_type = copy(self.vessel_df.at[idx, 'BC_type']) ### Get current vessel BC_type
                    n_vessel_BC_type_nn = list(n_vessel_BC_type) ### Split current vessel BC_type into left and right part

                    if len(inp_vessel_locs) == 0:
                        inp_vessel_BC_type = 'nn'
                    else:
                        inp_vessel_BC_type = copy(self.vessel_df.at[inp_vessel_locs[0], 'BC_type']) ### Get the current input vessel's BC_type
                    
                    if len(out_vessel_locs) == 0:
                        out_vessel_BC_type = 'nn'
                    else:
                        out_vessel_BC_type = copy(self.vessel_df.at[out_vessel_locs[0], 'BC_type']) ### Get the current output vessel's BC_type

                    ### // BC_type correction comparing to input vessel // ###
                    
                    if list(inp_vessel_BC_type)[1] == 'p' and n_vessel_BC_type_nn[0] == 'v':
                        pass
                    elif list(inp_vessel_BC_type)[1] == 'v' and n_vessel_BC_type_nn[0] == 'p':
                        pass
                    elif list(inp_vessel_BC_type)[1] == 'p' and n_vessel_BC_type_nn[0] == 'p':
                        n_vessel_BC_type_nn[0] = 'v'
                        n_vessel_BC_type = ''.join(n_vessel_BC_type_nn)
                    elif list(inp_vessel_BC_type)[1] == 'v' and n_vessel_BC_type_nn[0] == 'v':
                        n_vessel_BC_type_nn[0] = 'p'
                        n_vessel_BC_type = ''.join(n_vessel_BC_type_nn)

                    ### // BC_type correction comparing to output vessel // ###

                    if n_vessel_BC_type_nn[1] == 'p' and list(out_vessel_BC_type)[0] == 'v':
                        pass
                    elif n_vessel_BC_type_nn[1] == 'v' and list(out_vessel_BC_type)[0] == 'p':
                        pass
                    elif n_vessel_BC_type_nn[1] == 'p' and list(out_vessel_BC_type)[0] == 'p':
                        n_vessel_BC_type_nn[1] = 'v'
                        n_vessel_BC_type = ''.join(n_vessel_BC_type_nn)
                    elif n_vessel_BC_type_nn[1] == 'v' and list(out_vessel_BC_type)[0] == 'v':
                        n_vessel_BC_type_nn[1] = 'p'
                        n_vessel_BC_type = ''.join(n_vessel_BC_type_nn)

                    ### // Assign corrected BC_type to vessel_df // ###

                    self.vessel_df.at[idx, 'BC_type'] = n_vessel_BC_type

            ### // BC_type correction for Nout vessels // ###

            if len(n_vessel_Nout_idx) != 0:

                for idx in n_vessel_Nout_idx:

                    inp_vessel_i = self.C_vessel[:, idx] ### Get all input vessels of the current vessel
                    out_vessel_i = self.C_vessel[idx, :] ### Get all output vessels of the current vessel

                    inp_vessel_locs = np.where(inp_vessel_i == 1)[0] ### Get all idx's of all input vessels to the current vessel
                    out_vessel_locs = np.where(out_vessel_i == 1)[0] ### Get all idx's of all output vessels to the current vessel

                    n_vessel_BC_type = copy(self.vessel_df.at[idx, 'BC_type']) ### Get current vessel BC_type

                    if len(inp_vessel_locs) == 0:
                        inp_vessel_BC_type = 'nn'
                    else:
                        inp_vessel_BC_type = copy(self.vessel_df.at[inp_vessel_locs[0], 'BC_type']) ### Get the current input vessel's BC_type

                    for i in range(len(out_vessel_locs)): ### For every output vessel of the current vessel...

                        n_vessel_out = len(np.where(self.C_vessel[idx, :] == 1)[0]) ### Get the number of outputs of the current vessel

                        if n_vessel_out == 1: ### If the current vessel now has 1 output 
                            break

                        out_vessel_BC_type = copy(self.vessel_df.at[out_vessel_locs[i], 'BC_type']) ### Get the current output vessel's BC_type
                        
                        if is_compatible(n_vessel_BC_type, inp_vessel_BC_type, out_vessel_BC_type) == False: ### If surrounding BCs of the curret vessel are not compatible...
                            
                            if is_compatible('pv', inp_vessel_BC_type, out_vessel_BC_type): ### Then if 'vv' is a compatible BC given the surrounding BCs...
                                n_vessel_BC_type = 'pv'
                                self.vessel_df.at[idx, 'BC_type'] = 'pv' ### Force the current vessel BC to be 'pv'

                            elif is_compatible('vv', inp_vessel_BC_type, out_vessel_BC_type): ### Then if 'vv' is a compatible BC given the surrounding BCs...
                                n_vessel_BC_type = 'vv'
                                self.vessel_df.at[idx, 'BC_type'] = 'vv' ### Force the current vessel BC to be 'vv'

                            else: ### Otherwise...

                                for j in range(len(out_vessel_locs)): ### For every output vessel of the current vessel...

                                    n_vessel_out = len(np.where(self.C_vessel[idx, :] == 1)[0]) ### Get the number of outputs of the current vessel
                                    n_out_vessel_inp = len(np.where(self.C_vessel[:, out_vessel_locs[j]] == 1)[0]) ### Get the number of inputs to the output vessel

                                    if n_out_vessel_inp > 1: ### If there is more than one input vessel to the current vessel's output vessel...
                                        
                                        self.C_vessel[idx, :][out_vessel_locs[j]] = 0 ### Decouple the current vessel as an input vessel to the output vessel 
                                        self.C_vessel[:, out_vessel_locs[j]][idx] = 0 ### Decouple the output vessel as an output vessel to the current vessel                                

                                    n_vessel_out = len(np.where(self.C_vessel[idx, :] == 1)[0])

                                    if n_vessel_out == 1: ### If the current vessel now has 1 output 
                                        break

            ### // BC_type correction for Min vessels // ###

            if len(n_vessel_Min_idx) != 0:

                for idx in n_vessel_Min_idx:

                    inp_vessel_i = self.C_vessel[:, idx]
                    out_vessel_i = self.C_vessel[idx, :]

                    inp_vessel_locs = np.where(inp_vessel_i == 1)[0]
                    out_vessel_locs = np.where(out_vessel_i == 1)[0]

                    n_vessel_BC_type = copy(self.vessel_df.at[idx, 'BC_type'])
                    
                    if len(out_vessel_locs) == 0:
                        out_vessel_BC_type = 'nn'
                    else:
                        out_vessel_BC_type = copy(self.vessel_df.at[out_vessel_locs[0], 'BC_type']) ### Get the current output vessel's BC_type

                    for i in range(len(inp_vessel_locs)):

                        n_vessel_inp = len(np.where(self.C_vessel[:, idx] == 1)[0]) ### Get the number of inputs of the current vessel

                        if n_vessel_inp == 1:
                            break

                        inp_vessel_BC_type = copy(self.vessel_df.at[inp_vessel_locs[i], 'BC_type'])
                        
                        if is_compatible(n_vessel_BC_type, inp_vessel_BC_type, out_vessel_BC_type) == False:

                            if is_compatible('vp', inp_vessel_BC_type, out_vessel_BC_type): ### Then if 'vv' is a compatible BC given the surrounding BCs...
                                n_vessel_BC_type = 'vp'
                                self.vessel_df.at[idx, 'BC_type'] = 'vp' ### Force the current vessel BC to be 'vv'
                            
                            elif is_compatible('vv', inp_vessel_BC_type, out_vessel_BC_type):
                                n_vessel_BC_type = 'vv'
                                self.vessel_df.at[idx, 'BC_type'] = 'vv'

                            else:
                                
                                for j in range(len(inp_vessel_locs)): ### For every input vessel of the current vessel...

                                    n_vessel_inp = len(np.where(self.C_vessel[idx, :] == 1)[0]) ### Get the number of outputs of the current vessel
                                    n_inp_vessel_out = len(np.where(self.C_vessel[inp_vessel_locs[j], :] == 1)[0]) ### Get the number of outputs of the input vessel

                                    if n_inp_vessel_out > 1: ### If there is more than one output vessel to the current vessel's input vessel...
                                        
                                        self.C_vessel[:, idx][inp_vessel_locs[j]] = 0 ### Decouple the current vessel as an output vessel to the input vessel 
                                        self.C_vessel[inp_vessel_locs[j], :][idx] = 0 ### Decouple the input vessel as an input vessel to the current vessel                                

                                    n_vessel_inp = len(np.where(self.C_vessel[:, idx] == 1)[0]) ### Get the number of inputs of the current vessel

                                    if n_vessel_inp == 1:
                                        break

            ### // BC_type correction for MinNout vessels // ###

            if len(n_vessel_MinNout_idx) != 0:

                for idx in n_vessel_MinNout_idx:

                    inp_vessel_i = self.C_vessel[:, idx]
                    out_vessel_i = self.C_vessel[idx, :]

                    inp_vessel_locs = np.where(inp_vessel_i == 1)[0]
                    out_vessel_locs = np.where(out_vessel_i == 1)[0]

                    n_vessel_BC_type = copy(self.vessel_df.at[idx, 'BC_type'])
                    
                    if len(inp_vessel_locs) == 0:
                        inp_vessel_BC_type = 'nn'

                    if len(out_vessel_locs) == 0:
                        out_vessel_BC_type = 'nn'

                    for i in range(len(inp_vessel_locs)):

                        inp_vessel_BC_type = copy(self.vessel_df.at[inp_vessel_locs[i], 'BC_type'])

                        for j in range(len(out_vessel_locs)): ### For every output vessel of the current vessel...

                            n_vessel_out = len(np.where(self.C_vessel[idx, :] == 1)[0]) ### Get the number of outputs of the current vessel

                            if n_vessel_out == 1: ### If the current vessel now has 1 output 
                                break

                            out_vessel_BC_type = copy(self.vessel_df.at[out_vessel_locs[j], 'BC_type']) ### Get the current output vessel's BC_type
                            
                            if is_compatible(n_vessel_BC_type, inp_vessel_BC_type, out_vessel_BC_type) == False: ### If surrounding BCs of the curret vessel are not compatible...

                                if is_compatible('vv', inp_vessel_BC_type, out_vessel_BC_type): ### Then if 'vv' is a compatible BC given the surrounding BCs...
                                    n_vessel_BC_type = 'vv'
                                    self.vessel_df.at[idx, 'BC_type'] = 'vv' ### Force the current vessel BC to be 'vv'

                                else: ### Otherwise...

                                    for k in range(len(out_vessel_locs)): ### For every output vessel of the current vessel...

                                        n_vessel_out = len(np.where(self.C_vessel[idx, :] == 1)[0]) ### Get the number of outputs of the current vessel
                                        n_out_vessel_inp = len(np.where(self.C_vessel[:, out_vessel_locs[k]] == 1)[0]) ### Get the number of inputs to the output vessel

                                        if n_out_vessel_inp > 1: ### If there is more than one input vessel to the current vessel's output vessel...
                                            
                                            self.C_vessel[idx, :][out_vessel_locs[k]] = 0 ### Decouple the current vessel as an input vessel to the output vessel 
                                            self.C_vessel[:, out_vessel_locs[k]][idx] = 0 ### Decouple the output vessel as an output vessel to the current vessel                                

                                        n_vessel_out = len(np.where(self.C_vessel[idx, :] == 1)[0])

                                        if n_vessel_out == 1: ### If the current vessel now has 1 output 
                                            break
            
            ### // BC_type compatibility verfication across entire network // ###

            n_vessel_incompatible_idx = np.array([])
            n_vessel_incompatible_dict = {"idx": [],
                                        "inp_vessel": [],
                                        "n_vessel_idx": [],
                                        "out_vessel": []}
            n_vessel_incompatible_df = pd.DataFrame(copy(n_vessel_incompatible_dict))

            for idx in n_vessel_idx:

                inp_vessel_i = self.C_vessel[:, idx] ### Get all input vessels of the current vessel
                out_vessel_i = self.C_vessel[idx, :] ### Get all output vessels of the current vessel

                inp_vessel_locs = np.where(inp_vessel_i == 1)[0] ### Get all idx's of all input vessels to the current vessel
                out_vessel_locs = np.where(out_vessel_i == 1)[0] ### Get all idx's of all output vessels to the current vessel

                n_vessel_BC_type = copy(self.vessel_df.at[idx, 'BC_type']) ### Get current vessel BC_type

                if (len(inp_vessel_locs) == 0) and (len(out_vessel_locs) >= 1):

                    inp_vessel_BC_type = 'nn'

                    for out_vessel_idx in out_vessel_locs:

                        out_vessel_BC_type = self.vessel_df.at[out_vessel_idx, 'BC_type']

                        if is_compatible(n_vessel_BC_type, inp_vessel_BC_type, out_vessel_BC_type) == False:
                            
                            n_vessel_incompatible_idx = np.append(n_vessel_incompatible_idx, idx)
                            
                            n_vessel_name = self.vessel_df.at[idx, 'name']
                            out_vessel_name = self.vessel_df.at[out_vessel_idx, 'name']

                            n_vessel_junc_type = self.vessel_df.at[idx, 'junc_type']
                            out_vessel_junc_type = self.vessel_df.at[out_vessel_idx, 'junc_type']


                            n_vessel_incompatible_dict = {"idx": idx,
                                                        "inp_vessel": str(None) + ' (' + 'nn' + ' || ' + str(None) + ')',
                                                        "n_vessel_idx": n_vessel_name + ' (' + n_vessel_BC_type + ' || ' + str(n_vessel_junc_type) + ')',
                                                        "out_vessel": out_vessel_name + ' (' + out_vessel_BC_type + ' || ' + str(out_vessel_junc_type) + ')'}

                            n_vessel_incompatible_df.loc[len(n_vessel_incompatible_df)] = n_vessel_incompatible_dict
                
                elif (len(inp_vessel_locs) >= 1) and (len(out_vessel_locs) == 0):

                    out_vessel_BC_type = 'nn'

                    for inp_vessel_idx in inp_vessel_locs:

                        inp_vessel_BC_type = self.vessel_df.at[inp_vessel_idx, 'BC_type']

                        if is_compatible(n_vessel_BC_type, inp_vessel_BC_type, out_vessel_BC_type) == False:
                            
                            n_vessel_incompatible_idx = np.append(n_vessel_incompatible_idx, idx)
                            
                            n_vessel_name = self.vessel_df.at[idx, 'name']
                            inp_vessel_name = self.vessel_df.at[inp_vessel_idx, 'name']

                            n_vessel_junc_type = self.vessel_df.at[idx, 'junc_type']
                            inp_vessel_junc_type = self.vessel_df.at[inp_vessel_idx, 'junc_type']


                            n_vessel_incompatible_dict = {"idx": idx,
                                                        "inp_vessel": inp_vessel_name + ' (' + inp_vessel_BC_type + ' || ' + str(inp_vessel_junc_type) + ')',
                                                        "n_vessel_idx": n_vessel_name + ' (' + n_vessel_BC_type + ' || ' + str(n_vessel_junc_type) + ')',
                                                        "out_vessel": str(None) + ' (' + 'nn' + ' || ' + str(None) + ')'}

                            n_vessel_incompatible_df.loc[len(n_vessel_incompatible_df)] = n_vessel_incompatible_dict
                
                else:
                
                    for inp_vessel_idx in inp_vessel_locs:
                        
                        inp_vessel_BC_type = self.vessel_df.at[inp_vessel_idx, 'BC_type']

                        for out_vessel_idx in out_vessel_locs:

                            out_vessel_BC_type = self.vessel_df.at[out_vessel_idx, 'BC_type']

                            if is_compatible(n_vessel_BC_type, inp_vessel_BC_type, out_vessel_BC_type) == False:
                                
                                n_vessel_incompatible_idx = np.append(n_vessel_incompatible_idx, idx)
                                
                                n_vessel_name = self.vessel_df.at[idx, 'name']
                                inp_vessel_name = self.vessel_df.at[inp_vessel_idx, 'name']
                                out_vessel_name = self.vessel_df.at[out_vessel_idx, 'name']

                                n_vessel_junc_type = self.vessel_df.at[idx, 'junc_type']
                                inp_vessel_junc_type = self.vessel_df.at[inp_vessel_idx, 'junc_type']
                                out_vessel_junc_type = self.vessel_df.at[out_vessel_idx, 'junc_type']


                                n_vessel_incompatible_dict = {"idx": idx,
                                                            "inp_vessel": inp_vessel_name + ' (' + inp_vessel_BC_type + ' || ' + str(inp_vessel_junc_type) + ')',
                                                            "n_vessel_idx": n_vessel_name + ' (' + n_vessel_BC_type + ' || ' + str(n_vessel_junc_type) + ')',
                                                            "out_vessel": out_vessel_name + ' (' + out_vessel_BC_type + ' || ' + str(out_vessel_junc_type) + ')'}

                                n_vessel_incompatible_df.loc[len(n_vessel_incompatible_df)] = n_vessel_incompatible_dict

            # Remove duplicates to prevent exponential slowdown
            n_vessel_incompatible_idx = np.unique(n_vessel_incompatible_idx)
            
            if len(n_vessel_incompatible_idx) > 0:
                
                # print('Vessels:\n', self.vessel_df.loc[n_vessel_incompatible_idx, 'name'], '\nhave incompatible BCs')
                # print()

                print(len(n_vessel_incompatible_idx))
                N_ITER += 1

            else:

                n_vessel_in_out_idx = np.array([]).astype(int)
                # n_vessel_in_only_idx = np.array([]).astype(int)
                # n_vessel_out_only_idx = np.array([]).astype(int)
                n_vessel_Min_idx = np.array([]).astype(int)
                n_vessel_Nout_idx = np.array([]).astype(int)
                n_vessel_MinNout_idx = np.array([]).astype(int)

                for i in range(n_vessel):
                    
                    inp_vessel_i = self.C_vessel[:, i]
                    out_vessel_i = self.C_vessel[i, :]

                    inp_vessel_locs = np.where(inp_vessel_i == 1)[0]
                    out_vessel_locs = np.where(out_vessel_i == 1)[0]

                    if (len(inp_vessel_locs) == 1) and (len(out_vessel_locs) == 0):
                        n_vessel_in_out_idx = np.append(n_vessel_in_out_idx, i)
                        self.vessel_df.at[i, 'junc_type'] = 'outlet'
                    elif (len(inp_vessel_locs) == 0) and (len(out_vessel_locs) == 1):
                        n_vessel_in_out_idx = np.append(n_vessel_in_out_idx, i)
                        self.vessel_df.at[i, 'junc_type'] = 'inlet'
                    elif (len(inp_vessel_locs) == 1) and (len(out_vessel_locs) == 1):
                        n_vessel_in_out_idx = np.append(n_vessel_in_out_idx, i)
                        self.vessel_df.at[i, 'junc_type'] = None
                    elif (len(inp_vessel_locs) > 1) and (len(out_vessel_locs) == 0):
                        n_vessel_Min_idx = np.append(n_vessel_Min_idx, i)
                        self.vessel_df.at[i, 'junc_type'] = 'Noutlet'
                    elif (len(inp_vessel_locs) == 0) and (len(out_vessel_locs) > 1):
                        n_vessel_Nout_idx = np.append(n_vessel_Nout_idx, i)
                        self.vessel_df.at[i, 'junc_type'] = 'Minlet'
                    elif (len(inp_vessel_locs) > 1) and (len(out_vessel_locs) == 1):
                        n_vessel_Min_idx = np.append(n_vessel_Min_idx, i)
                        self.vessel_df.at[i, 'junc_type'] = 'Min'
                    elif (len(inp_vessel_locs) == 1) and (len(out_vessel_locs) > 1):
                        n_vessel_Nout_idx = np.append(n_vessel_Nout_idx, i)
                        self.vessel_df.at[i, 'junc_type'] = 'Nout'
                    else:
                        n_vessel_MinNout_idx = np.append(n_vessel_MinNout_idx, idx)
                        self.vessel_df.at[i, 'junc_type'] = 'MinNout'

                    inp_vessel_names = list(self.vessel_df.loc[inp_vessel_locs, 'name'])
                    out_vessel_names = list(self.vessel_df.loc[out_vessel_locs, 'name'])

                    self.vessel_df.at[i, 'inp_vessels'] = ' '.join(inp_vessel_names) if len(inp_vessel_names) > 0 else None
                    self.vessel_df.at[i, 'out_vessels'] = ' '.join(out_vessel_names) if len(out_vessel_names) > 0 else None

                    if self.vessel_df.at[i, 'junc_type'] != None:
                        self.vessel_df.loc[i, 'vessel_type'] += ('_' + self.vessel_df.at[i, 'junc_type'])

                    N_ITER = np.inf

        self.vessel_df = self.vessel_df.drop('junc_type', axis=1)
        self.vessel_df = self.vessel_df.dropna(how='all')

        ### // Add inlet and outlet modules to vessel_df // ###

        inlet_vessels_all = np.array(self.vessel_df['inp_vessels'])
        outlet_vessels_all = np.array(self.vessel_df['out_vessels'])

        inlet_vessels_locs = np.where(inlet_vessels_all == None)[0]
        outlet_vessels_locs = np.where(outlet_vessels_all == None)[0]

        inlet_vessels_names = np.array(self.vessel_df.loc[inlet_vessels_locs, 'name'])
        outlet_vessels_names = np.array(self.vessel_df.loc[outlet_vessels_locs, 'name'])

        self.vessel_df.loc[inlet_vessels_locs, 'inp_vessels'] = inlet_vessels_names + '_PI'
        self.vessel_df.loc[outlet_vessels_locs, 'out_vessels'] = outlet_vessels_names + '_PO'

        # [MODIFIED] Added x, y, z extraction for Inlets
        inlet_vessels_dict = {'name': inlet_vessels_names + '_PI',
                              'BC_type': np.repeat('nn', len(inlet_vessels_names)),
                              'vessel_type': 'P_inlet',
                              'inp_vessels': np.repeat(None, len(inlet_vessels_names)),
                              'out_vessels': np.array(self.vessel_df.loc[inlet_vessels_locs, 'name']),
                              'x': np.array(self.vessel_df.loc[inlet_vessels_locs, 'x']), # Copy parent x
                              'y': np.array(self.vessel_df.loc[inlet_vessels_locs, 'y']), # Copy parent y
                              'z': np.array(self.vessel_df.loc[inlet_vessels_locs, 'z'])  # Copy parent z
                              }

        # [MODIFIED] Added x, y, z extraction for Outlets
        outlet_vessels_dict = {'name': outlet_vessels_names + '_PO',
                              'BC_type': np.repeat('nn', len(outlet_vessels_names)),
                              'vessel_type': 'P_outlet',
                              'inp_vessels': np.array(self.vessel_df.loc[outlet_vessels_locs, 'name']), 
                              'out_vessels': np.repeat(None, len(outlet_vessels_names)),
                              'x': np.array(self.vessel_df.loc[outlet_vessels_locs, 'x']), # Copy parent x
                              'y': np.array(self.vessel_df.loc[outlet_vessels_locs, 'y']), # Copy parent y
                              'z': np.array(self.vessel_df.loc[outlet_vessels_locs, 'z'])  # Copy parent z
                              }
        
        inlet_vessels_df = pd.DataFrame(inlet_vessels_dict)
        outlet_vessels_df = pd.DataFrame(outlet_vessels_dict)
        
        inlet_outlet_vessels_df = pd.concat([inlet_vessels_df, outlet_vessels_df], ignore_index=True)
        self.vessel_df = pd.concat([self.vessel_df, inlet_outlet_vessels_df], ignore_index=True)
        self.vessel_df['BC_type'] = self.vessel_df['BC_type'].astype(str) + '_micro'

        print('Network Solved!')

    def generate_parameter_array(self, inp_data_dict=None):

        # --- LAZY LOAD & BYPASS ---
        # import pandas as pd
        
        # 1. Setup Path to Source (Bypassing generate_param_array.py)
        root_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(root_dir)
        root_dir = os.path.dirname(root_dir)

        # 2. Add to sys.path (Only if missing, prevents duplicates)
        src_path = os.path.join(root_dir, 'src')
        if src_path not in sys.path:
            sys.path.append(src_path)

        # 3. Import Parsers
        try:
            from parsers.PrimitiveParsers import YamlFileParser
            from parsers.ModelParsers import CSV0DModelParser
        except ImportError as e:
            # If this fails, it usually means the 'src' path calculation is wrong for your folder structure
            print(f"CRITICAL ERROR: Could not import parsers. Checked path: {src_path}")
            raise e
        # -------------------------------

        print('Generating Parameter Array...')

        user_inputs_dir = os.path.join(root_dir, 'user_run_files')

        yaml_parser = YamlFileParser()
        inp_data_dict = yaml_parser.parse_user_inputs_file(inp_data_dict)

        file_prefix = inp_data_dict['file_prefix']
        resources_dir = inp_data_dict['resources_dir']
        vessels_csv_abs_path = inp_data_dict['vessels_csv_abs_path']
        parameters_csv_abs_path = inp_data_dict['parameters_csv_abs_path']

        parser = CSV0DModelParser(inp_data_dict)

        vessels_df = vessels_df = pd.read_csv(parser.vessel_filename, header=0, dtype=str, skipinitialspace=True)
        vessels_df = vessels_df.fillna('')

        nVess = vessels_df.shape[0]

        column_types = {
            'variable_name': 'str',
            'units': 'str',
            'value': 'float',
            'data_reference': 'str'
        }
        self.parameter_df = pd.DataFrame(columns=column_types.keys()).astype(column_types)

        # module_config_fold1 = root_dir_src + '/generators/resources/'
        module_config_fold2 = root_dir + '/module_config_user/'
        modules = []
        # for filename in os.listdir(module_config_fold1):
        #     if filename.endswith(".json"):
        #         with open(os.path.join(module_config_fold1, filename), "r") as file:
        #             temp_data = json.load(file)
        #             if isinstance(temp_data, list):
        #                 modules.extend(temp_data)
        #             else:
        #                 modules.append(temp_data)
        for filename in os.listdir(module_config_fold2):
            if filename.endswith(".json"):
                with open(os.path.join(module_config_fold2, filename), "r") as file:
                    temp_data = json.load(file)
                    if isinstance(temp_data, list):
                        modules.extend(temp_data)
                    else:
                        modules.append(temp_data)

        default_global_const = [['T', 'second', 1.0, 'user_defined'],
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
                                ["A_nn_aov", "m2", 0.00068, 'Mynard_2012'],]

        global_const = []

        for i in range(nVess):

            nameV = vessels_df.at[i,'name']
            typeBC = vessels_df.at[i,'BC_type']
            typeV = vessels_df.at[i,'vessel_type']

            matches = [entry for entry in modules if entry.get('BC_type')==typeBC and entry.get('vessel_type')==typeV]
            if len(matches)>1:
                sys.exit('ERROR :: multiple modules found for this vessel_type and BC_type combination : '+typeV+' '+typeBC+' :: Check your module_config.json file.')

            if len(matches)==0:
                sys.exit('ERROR :: no modules found for this vessel_type and BC_type combination : '+typeV+' '+typeBC+' :: Check your module_config.json file.')

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
                    self.parameter_df = pd.concat([self.parameter_df, pd.DataFrame([new_row])], ignore_index=True)

                
        for i in range(len(global_const)):
            const_name = global_const[i][0]

            found = -1
            for j in range(len(default_global_const)):
                if const_name==default_global_const[j][0]:
                    new_row = {'variable_name': const_name,
                                        'units': default_global_const[j][1],
                                        'value': default_global_const[j][2],
                                        'data_reference': default_global_const[j][3]}
                    self.parameter_df = pd.concat([self.parameter_df, pd.DataFrame([new_row])], ignore_index=True)
                    found = 1
                    break

            if found==-1:
                new_row = {'variable_name': const_name,
                                        'units': global_const[i][1],
                                        'value': -1.0,
                                        'data_reference': 'TO_DO'}
                self.parameter_df = pd.concat([self.parameter_df, pd.DataFrame([new_row])], ignore_index=True)

        # parameters_csv_abs_path_temp = '/home/dsas627/PycharmProjects/me_bioeng_cb_vessel_network/resources/image_to_model_parameters.csv'
        self.parameter_df.to_csv(parameters_csv_abs_path, index=False, header=True)
                    
        print('DONE :: Parameters array file for model '+file_prefix+' generated and saved.')

    def populate_parameter_array(self):

        import pandas as pd
        import networkx as nx
        from copy import copy
        print('Populating Parameter Array...')

        ### // Extract all variable types from parameter_df // ###

        vessel_name_series = copy(self.vessel_df['name']) ### Initialise a copy of the series of vessel names
        parameter_name_series = copy(self.parameter_df['variable_name']) ### Initialise a copy of the series of parameter names 
        parameter_types = {} ### Initialise a dict to store unique parameter types

        for vessel_name in vessel_name_series:

            if (vessel_name.endswith('_PI') == True) | (vessel_name.endswith('_PO') == True): ### If the current 'vessel' is a pressure BC module...
                continue ### Ignore current 'vessel'
            
            variable_name_locs = self.parameter_df['variable_name'].str.contains(vessel_name, na=False) ### Get the boolean mask of parameter names with the current vessel name
            variable_names_with_vessel_name = self.parameter_df.loc[variable_name_locs, 'variable_name'] ### Get all parameter names with the current vessel_names
            extracted_parameter_types = variable_names_with_vessel_name.str.split('_' + vessel_name, n=1).str[0] ### Get the parameter name w/o the vessel name
            
            vessel_param_default_entries = np.repeat('to_fill', len(vessel_name_series))

            ### For each parameter name, create an array of 'to_fill'. This initialises the values to be filled per parameter for each vessel.
            extracted_parameter_types_dict = {key_name: np.repeat('to_fill', len(vessel_name_series)) for key_name in extracted_parameter_types}
            parameter_types.update(extracted_parameter_types_dict) ### Update parameter name dict

        vessel_param_df = pd.DataFrame(parameter_types) ### Create a vessel parameter dataframe from the parameter_type dict
        vessel_param_df['vessel_name'] = list(vessel_name_series) ### Add a vessel_name column to the vessel parameter dataframe
        vessel_param_df.insert(0, 'vessel_name', vessel_param_df.pop('vessel_name')) ### Bring the vessel parameter name column to the LHS of the dataframe

        parameter_name_series_set = set(parameter_name_series)

        def check_vessel_parameter_exists(row, valid_vessel_parameters):

            vessel = row['vessel_name']
    
            # We iterate over the row's index (which are the column names)
            for col_name in row.index:
                # Skip the 'vessel_name' column itself
                if col_name == 'vessel_name':
                    continue
                    
                # Check if the cell value is 'to_fill'
                if row[col_name] == 'to_fill':
                    # Create the combo string, e.g., "A_0_u_ext"
                    combo_string = f"{col_name}_{vessel}"
                    
                    # If the combo string does NOT exist in our set
                    if combo_string not in valid_vessel_parameters:
                        # Replace the value
                        row[col_name] = np.nan
                        
            # Return the modified row
            return row

        vessel_param_df = vessel_param_df.apply(check_vessel_parameter_exists, axis=1, args=(parameter_name_series_set,))


        def create_vessel_parameter_df(vessel_df):
            """
            Calculates steady-state pressure in a vascular network based on vessel geometry.

            This function performs the following steps:
            1.  Samples radius (r_0) and length (l) for each vessel from a normal distribution
                based on its 'vessel_type'.
            2.  Calculates the hydraulic resistance of each vessel using the Hagen-Poiseuille equation.
            3.  Constructs a graph of the vascular network.
            4.  Identifies pressure boundary conditions: vessels with 'PI' (pressure inlet) in their
                name are high-pressure, and 'PO' (pressure outlet) are low-pressure.
            5.  Models the network as a system of linear equations weighted by conductance (1/resistance).
            6.  Solves the system to find the pressure at each internal vessel.
            7.  Returns a DataFrame with vessel name, calculated pressure, radius, and length.

            Args:
                vessel_df (pd.DataFrame): A DataFrame with columns: 'name', 'vessel_type',
                                        'inp_vessels', and 'out_vessels'.

            Returns:
                pd.DataFrame: A DataFrame with columns: 'name', 'u_0' (pressure),
                            'r_0' (radius), and 'l' (length).
            """
            # --- 1. Data Preparation ---
            # Clean up whitespace from vessel names, which can cause matching issues
            vessel_df['name'] = vessel_df['name'].str.strip()

            if 'out_vessels' in vessel_df.columns and vessel_df['out_vessels'].dtype == 'O':
                vessel_df['out_vessels'] = vessel_df['out_vessels'].apply(
                    lambda s: [v.strip() for v in s.split()] if isinstance(s, str) else []
                )
            if 'inp_vessels' in vessel_df.columns and vessel_df['inp_vessels'].dtype == 'O':
                vessel_df['inp_vessels'] = vessel_df['inp_vessels'].apply(
                    lambda s: [v.strip() for v in s.split()] if isinstance(s, str) else []
                )

            network_vessels_df = vessel_df.copy()

            # --- 2. Sample Vessel Geometry and Calculate Resistance ---
            # Dictionary of mean and std for radius [m] and length [m] by vessel type
            vessel_properties = {
                # Type: (radius_mean, radius_std, length_mean, length_std, stiffness_mean, stiffness_std)
                'artery': (2.8e-3, 1.1e-3, 2.5e-1, 0.75e-1, 160e3, 40e3),
                'arteriole': (2.8e-5, 1.1e-5, 2.8e-3, 1.1e-3, 6.5e3, 0),
                'capillary': (3.8e-6, 0.6e-6, 6.5e-4, 1.8e-4, 6.5e3, 0),
                'venule': (2.7e-5, 1.2e-5, 2.3e-3, 0.9e-3, 6.5e3, 0),
                'vein': (3.0e-3, 1.0e-3, 2.5e-1, 0.75e-1, 210e3, 220e3),
            }

            # def get_vessel_info(vessel_type):
            #     """Samples radius and length from normal distribution based on vessel type."""
            #     # Find the base type (e.g., 'capillary_1' -> 'capillary')
            #     base_type = next((key for key in vessel_properties if key in vessel_type.lower()), None)
            #     if base_type:
            #         r_mean, r_std, l_mean, l_std, E_mean, E_std = vessel_properties[base_type]
            #         # Ensure non-negative values by taking the absolute
            #         radius = abs(np.random.normal(r_mean, r_std))
            #         length = abs(np.random.normal(l_mean, l_std))
            #         stiffness = abs(np.random.normal(E_mean, E_std))
            #         volume = abs(0)
            #         # radius = abs(2e-6)
            #         # length = abs(2e-3)
            #         # stiffness = abs(6.5e6)
            #         # volume = abs(0)
            #         return radius, length, stiffness, volume
            #     return np.nan, np.nan, np.nan, np.nan

            CB_geom_params_df = pd.read_csv('label_2_edge_list_for_matrix.csv')
            CB_geom_params_dict = CB_geom_params_df.set_index('Edge_ID_in_Matrix')[['EstimatedRadius', 'WorldLength']].to_dict('index')
            
            def get_vessel_info_row(row, vessel_props_dict, ref_data_dict):
                """
                Designed for df.apply().
                row: The current row of the main dataframe.
                vessel_props_dict: The global dictionary of base type stats.
                ref_data_dict: The dictionary created from the reference CSV.
                """
                vessel_type = row['vessel_type'] # Adjust column name if different
                vessel_name = row['name'] # Adjust column name if different
                
                # --- Stiffness Sampling ---
                base_type = next((key for key in vessel_props_dict if key in vessel_type.lower()), None)
                
                # Default values
                radius, length, stiffness, volume = np.nan, np.nan, np.nan, np.nan
                
                if base_type:
                    _, _, _, _, E_mean, E_std = vessel_props_dict[base_type]
                    stiffness = abs(np.random.normal(E_mean, E_std))
                    # volume = abs(0)

                    # --- Radius/Length Lookup ---
                    # Extract ID (e.g., 'L1_E45' -> 45)
                    match = re.search(r'E(\d+)', vessel_name)
                    
                    if match:
                        edge_id = int(match.group(1))
                        
                        # Fast dictionary lookup
                        if edge_id in ref_data_dict.index:
                            data = ref_data_dict.loc[edge_id]
                            radius = data['EstimatedRadius'] * 1e-6 ### microns
                            length = data['WorldLength'] * 1e-6 ### microns
                            volume = np.pi * radius ** 2 * length 
                
                # Return as a Series so they become new columns automatically
                return pd.Series([radius, length, stiffness, volume], 
                                index=['radius', 'length', 'stiffness', 'volume'])

            # Apply the function to create r_0 and l columns
            # geometries = network_vessels_df['vessel_type'].apply(get_vessel_info)
            # network_vessels_df[['r_0', 'l', 'E', 'q_C_init']] = pd.DataFrame(geometries.tolist(), index=network_vessels_df.index)

            geometries = network_vessels_df.apply(get_vessel_info_row,
                                                  axis=1,
                                                  args=(vessel_properties, CB_geom_params_df))
            
            network_vessels_df[['r_0','l','E','q_C_init']] = pd.DataFrame(geometries, 
                                                                          index=network_vessels_df.index)


            # TODO radii are not being set correctly, they are tiny. 
            # Temporarily increase the radii by a factor of 4
            network_vessels_df['r_0'] = network_vessels_df['r_0'] * 4

            vessel_param_df[['r_0', 'l']] = network_vessels_df[['r_0', 'l']] ### Set the r_0 and l values in the vessel parameter dataframe to be the same as the calculated values in the vessel network dataframe
            vessel_param_df['r'] = network_vessels_df['r_0'] ### Set r in the vessel parameter dataframe to be the same as the calculated radii values from the vessel network dataframe
            vessel_param_df['E'] = network_vessels_df['E'] ### Set E in the vessel parameter dataframe to be the same as the calculated E values from the vessel network dataframe
            vessel_param_df['q_C_init'] = copy(network_vessels_df['q_C_init'])
            vessel_param_df['q_0'] = copy(network_vessels_df['q_C_init']) ### Set q_C_init in the vessel parameter dataframe to be the same as the calculated volume values from the vessel network dataframe

            # Calculate hydraulic resistance using Hagen-Poiseuille equation
            MU = 3.5e-3  # Dynamic viscosity of blood in Pa·s
            # R = (8 * mu * l) / (pi * r^4)
            network_vessels_df['resistance'] = (8 * MU * network_vessels_df['l']) / (np.pi * network_vessels_df['r_0']**4)
            # Create a lookup map for resistance
            resistance_map = network_vessels_df.set_index('name')['resistance'].to_dict()

            # --- 3. Build Network Graph ---
            valid_vessel_names = set(network_vessels_df['name'])
            G = nx.Graph()
            for _, row in network_vessels_df.iterrows():
                source_vessel = row['name']
                G.add_node(source_vessel)
                # Add edges from both input and output vessel lists for robustness
                for target_vessel in row['out_vessels'] + row['inp_vessels']:
                    if target_vessel in valid_vessel_names:
                        G.add_edge(source_vessel, target_vessel)

            # --- 4. Identify Boundary and Internal Nodes ---
            boundary_pressures = {}
            internal_nodes = []
            ARTERIAL_PRESSURE = 6666.12 ### 50 mmHg
            VENOUS_PRESSURE = 666.612 ### 5 mmHg

            for node in G.nodes():
                if 'PI' in node:
                    boundary_pressures[node] = ARTERIAL_PRESSURE
                elif 'PO' in node:
                    boundary_pressures[node] = VENOUS_PRESSURE
                # Only include nodes with a valid, finite resistance in the calculation
                elif node in resistance_map and pd.notna(resistance_map[node]) and np.isfinite(resistance_map[node]):
                    internal_nodes.append(node)

            if not internal_nodes:
                print("Warning: No internal nodes to solve for.")
                return network_vessels_df[['name', 'r_0', 'l']].merge(
                    pd.DataFrame(list(boundary_pressures.items()), columns=['name', 'u_0']), on='name', how='left'
                )

            # --- 5. Set up the Linear System Ax = b (weighted by conductance) ---
            n = len(internal_nodes)
            node_to_idx = {node: i for i, node in enumerate(internal_nodes)}
            A = np.zeros((n, n))
            b = np.zeros(n)

            for i, node in enumerate(internal_nodes):
                total_conductance = 0
                for neighbor in G.neighbors(node):
                    R_node = resistance_map.get(node, float('inf'))
                    
                    # --- MODIFIED LOGIC FOR CONNECTION RESISTANCE ---
                    if neighbor in boundary_pressures:
                        # For a connection to a boundary, resistance is that of the internal vessel itself.
                        R_connection = R_node
                    else:
                        # For a connection to another internal vessel, average the resistances.
                        R_neighbor = resistance_map.get(neighbor, float('inf'))
                        R_connection = (R_node + R_neighbor) / 2.0
                    
                    if R_connection == 0 or np.isinf(R_connection) or np.isnan(R_connection):
                        continue
                        
                    conductance = 1.0 / R_connection
                    total_conductance += conductance
                    
                    if neighbor in internal_nodes:
                        j = node_to_idx[neighbor]
                        A[i, j] = -conductance
                    elif neighbor in boundary_pressures:
                        b[i] += conductance * boundary_pressures[neighbor]
                
                A[i, i] = total_conductance + 1e-12 ### Adds a small leak (regularisation) to prevent floating nodes
            
            # --- 6. Solve for Unknown Pressures ---
            try:
                solved_pressures_vec, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                solved_pressures = {node: solved_pressures_vec[i] for i, node in enumerate(internal_nodes)}
            except np.linalg.LinAlgError:
                raise np.linalg.LinAlgError("The system of equations is singular and cannot be solved.")

            # --- 7. Format and Return the Final DataFrame ---
            all_pressures = boundary_pressures.copy()
            all_pressures.update(solved_pressures)
            
            result_df = pd.DataFrame(list(all_pressures.items()), columns=['vessel_name', 'u_0'])
            result_lookup_map = result_df.set_index('vessel_name')['u_0']

            vessel_param_df['u_0'] = vessel_param_df['vessel_name'].map(result_lookup_map)

            for vessel_name in vessel_name_series:

                if ('_PI' in vessel_name) | ('_PO' in vessel_name):

                    P = copy(vessel_param_df.loc[vessel_param_df['vessel_name'] == vessel_name, 'u_0'].iloc[0])
                    vessel_param_df.loc[vessel_param_df['vessel_name'] == vessel_name, 'u_0'] = np.nan
                    vessel_param_df.loc[vessel_param_df['vessel_name'] == vessel_name, 'P'] = str(P)

            vessel_param_df['u_ext'] = vessel_param_df['u_ext'].replace('to_fill', 0.)
            vessel_param_df['theta'] = vessel_param_df['u_ext'].replace('to_fill', 0.)

            return vessel_param_df
                
        vessel_parameters_df = create_vessel_parameter_df(self.vessel_df)
        valid_variable_names = set(self.parameter_df['variable_name'])

        def populate_param_array(row, valid_vessel_parameters):

            vessel = row['vessel_name']
    
            # We iterate over the row's index (which are the column names)
            for col_name in row.index:
                # Skip the 'vessel_name' column itself
                if col_name == 'vessel_name':
                    continue
                    
                # Check if the cell value is not NaN
                if row[col_name] != np.nan:
                    # Create the combo string, e.g., "A_0_u_ext"
                    combo_string = f"{col_name}_{vessel}"
                    
                    # If the combo string does NOT exist in our set
                    if combo_string in valid_variable_names:
                        # Replace the value
                        self.parameter_df.loc[self.parameter_df['variable_name'] == combo_string, 'value'] = row[col_name]
                        
            # Return the modified row
            return row
        
        vessel_param_df = vessel_param_df.apply(populate_param_array, axis=1, args=(valid_variable_names,))

        return None

class IlastikClassifier():
    def __init__(self, ilastik_binary_path, project_file_path):
        """
        Initialize the Ilastik classifier wrapper.
        
        :param ilastik_binary_path: Path to the ilastik executable.
        :param project_file_path: Path to the trained .ilp project file.
        """

        self.binary = ilastik_binary_path
        self.project = project_file_path
        
        # Validate paths immediately to catch errors early
        if not os.path.exists(self.binary):
            raise FileNotFoundError(f"Ilastik binary not found at: {self.binary}")
        if not os.path.exists(self.project):
            raise FileNotFoundError(f"Project file not found at: {self.project}")

    def segment_images(self, input_dir, output_dir, input_ext="*.tif", export_source="Simple Segmentation"):
        """
        Runs the segmentation on a batch of images.
        
        :param input_dir: Folder containing images to process.
        :param output_dir: Folder to save results.
        :param input_ext: File pattern (e.g., "*.tif", "*.png", "*.h5").
        :param export_source: "Simple Segmentation" (integers) or "Probabilities" (floats).
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Find images
        # We use Path for cleaner OS-agnostic handling
        input_path = Path(input_dir)
        images = list(input_path.glob(input_ext))
        
        if not images:
            raise FileNotFoundError(f"No images found in {input_dir} matching {input_ext}")

        print(f"Found {len(images)} images. Starting Ilastik engine...")

        # Construct the internal command (Hidden from you during usage)
        # We assume 0-255 renormalization is OFF for integer masks, ON for probabilities usually
        # but here we stick to defaults.
        cmd = [
            str(self.binary),
            "--headless",
            f"--project={self.project}",
            "--output_format=hdf5",
            f"--export_source={export_source}",
            f"--output_filename_format={output_dir}/{{nickname}}_seg.hdf5"
        ]
        
        # Add all images to the argument list
        cmd.extend([str(img) for img in images])
        
        # Run internally
        try:
            # check=True will raise an error if Ilastik fails
            process = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("------------------------------------------------")
            print("Ilastik Output Log (Success):")
            print(process.stdout) # Prints the internal Ilastik logs to your IDE console
            print("------------------------------------------------")
            print(f"Successfully processed {len(images)} images.")
            print(f"Results saved in: {output_dir}")
            
        except subprocess.CalledProcessError as e:
            print("!!! Error occurred during processing !!!")
            raise e

##################################
### // Function Definitions // ###
##################################

def load_segmentation_data(filepath, hdf5_dataset_name=None):
    import tifffile

    if not filepath:
        raise ValueError("Input file path is not set.")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: File not found at {filepath}")
    _, extension = os.path.splitext(str(filepath).lower())
    try:
        if extension in ['.h5', '.hdf5']:
            if not hdf5_dataset_name:
                raise ValueError("Error: HDF5 dataset name is required.")
            with h5py.File(filepath, 'r') as f:
                if hdf5_dataset_name not in f:
                    raise ValueError(f"Error: Dataset '{hdf5_dataset_name}' not found.")
                data = np.array(f[hdf5_dataset_name])
        elif extension in ['.tif', '.tiff']:
            data = tifffile.imread(filepath)
        else:
            raise ValueError(f"Error: Unsupported file extension '{extension}'.")

        if data.ndim == 4 and data.shape[-1] == 1:
            data = np.squeeze(data, axis=-1)
        elif data.ndim == 4:
            if data.shape[0] == 1:
                data = data[0]
            elif data.shape[-1] > 1:
                data = data[..., 0]

        if data.ndim != 3:
            raise ValueError(f"Error: Loaded data is not 3D (shape: {data.shape}).")
        print(f"Successfully loaded data: {data.shape}, {data.dtype}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise e

def discretize_network_with_grid(graph, grid_planes_x, grid_planes_y, grid_planes_z, vedo_spacing):
    """
    Refines a network graph by adding new nodes where segments intersect with a 3D grid.
    This version correctly propagates and splits the 'voxel_path' attribute.
    """
    print("         Discretizing network with grid (and propagating voxel paths)...")
    new_graph = graph.copy()
    
    next_node_id = max(graph.nodes) + 1 if graph.nodes else 0
    original_edges = list(graph.edges())

    for u, v in original_edges:
        p1_data = graph.nodes[u]
        p2_data = graph.nodes[v]
        p1_world = np.array(p1_data['pos_xyz_world'])
        p2_world = np.array(p2_data['pos_xyz_world'])
        
        # --- NEW: Retrieve the voxel path for this edge ---
        edge_data = graph.edges[u, v]
        if 'voxel_path' not in edge_data:
            continue # Skip edges that don't have a path (should not happen)
        
        original_voxel_path = edge_data['voxel_path']
        if original_voxel_path.shape[0] < 2:
            continue

        # --- NEW: Order the voxel path from start to end ---
        ordered_voxel_path = _order_voxel_path(original_voxel_path, p1_data['pos_zyx_image'])

        # Calculate intersections using the simple straight-line model
        intersections_on_segment = []
        plane_definitions = [
            (grid_planes_x, np.array([1, 0, 0])),
            (grid_planes_y, np.array([0, 1, 0])),
            (grid_planes_z, np.array([0, 0, 1]))
        ]

        for planes, normal in plane_definitions:
            for plane_coord in planes:
                plane_point = normal * plane_coord
                line_dir = p2_world - p1_world

                dot_product = np.dot(line_dir, normal)
                if abs(dot_product) > 1e-6:
                    w = p1_world - plane_point
                    t = -np.dot(w, normal) / dot_product

                    if 0 < t < 1:
                        intersect_point_world = p1_world + t * line_dir
                        # --- NEW: Store the intersection point AND its proportional distance 't' ---
                        intersections_on_segment.append({'pos': intersect_point_world, 't': t})

        if intersections_on_segment:
            # Sort intersections by their distance from the start point p1
            intersections_on_segment.sort(key=lambda item: item['t'])

            if new_graph.has_edge(u, v):
                new_graph.remove_edge(u, v)

            last_node = u
            last_split_idx = 0
            
            for item in intersections_on_segment:
                point_world = item['pos']
                proportional_dist = item['t']
                
                # --- NEW: Determine where to split the ordered voxel path ---
                split_idx = int(proportional_dist * len(ordered_voxel_path))
                
                # Convert world coords back to image coords for the new node
                z_img, y_img, x_img = (
                    point_world[2] / vedo_spacing[2],
                    point_world[1] / vedo_spacing[1],
                    point_world[0] / vedo_spacing[0]
                )
                point_image_zyx = (z_img, y_img, x_img)

                new_node_id = next_node_id
                new_graph.add_node(new_node_id,
                                   pos_zyx_image=point_image_zyx,
                                   pos_xyz_world=tuple(point_world),
                                   type='intersection')
                next_node_id += 1
                
                # --- NEW: Add edge with the correct sub-path of voxels ---
                sub_path = ordered_voxel_path[last_split_idx:split_idx]
                new_graph.add_edge(last_node, new_node_id, voxel_path=sub_path)
                
                last_node = new_node_id
                last_split_idx = split_idx

            # Add the final edge from the last intersection to the original end point
            final_sub_path = ordered_voxel_path[last_split_idx:]
            new_graph.add_edge(last_node, v, voxel_path=final_sub_path)
            
    return new_graph

def get_grid_block_id(point_xyz, grid_origin_xyz, cell_size_xyz):
    """Calculates the (i, j, k) index of the grid cell for a given point."""
    if np.any(cell_size_xyz <= 0): return None
    relative_pos = np.array(point_xyz) - np.array(grid_origin_xyz)
    indices = relative_pos / np.array(cell_size_xyz)
    return tuple(indices.astype(int))

def get_grid_block_id_format(indices_tuple, grid_resolution_xyz):

    indices_int = np.array(indices_tuple)
    n_x, n_y, n_z = grid_resolution_xyz
    tissue_block_id = (indices_int[0] * n_y * n_z) + (indices_int[1] * n_z) + indices_int[2]
    indices_formatted = 'tissue_block_' + str(tissue_block_id)

    return indices_formatted

def create_mask_for_line(p1_img, p2_img, shape):
    """Creates a boolean mask for a line segment between two points in a 3D volume."""
    from skimage.draw import line_nd
    line_mask = np.zeros(shape, dtype=bool)
    # Ensure points are integers for indexing
    p1_int = np.round(p1_img).astype(int)
    p2_int = np.round(p2_img).astype(int)

    rr, cc, zz = line_nd(p1_int, p2_int)
    # Ensure indices are within bounds
    valid_indices = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1]) & (zz >= 0) & (zz < shape[2])
    line_mask[rr[valid_indices], cc[valid_indices], zz[valid_indices]] = True
    return line_mask

def _order_voxel_path(path_voxels_zyx, start_node_pos_zyx):
    
    # --- LAZY LOAD ---
    from scipy.spatial import KDTree  # <--- PASTE THIS HERE
    # -----------------

    """
    Orders a set of unordered voxel coordinates into a continuous path.

    Args:
        path_voxels_zyx (np.array): A (N, 3) array of voxel ZYX coordinates.
        start_node_pos_zyx (tuple): The ZYX coordinate of the node to start the path from.

    Returns:
        np.array: The (N, 3) array of voxel coordinates, now ordered.
    """
    if path_voxels_zyx.shape[0] < 2:
        return path_voxels_zyx

    # Use a KDTree for efficient nearest neighbor searches
    tree = KDTree(path_voxels_zyx)
    
    # Find the voxel in the path closest to the official start node's position
    _, start_idx = tree.query(start_node_pos_zyx)
    
    # Efficiently create a set of indices for fast removal
    remaining_indices = set(range(path_voxels_zyx.shape[0]))
    
    ordered_path = np.zeros_like(path_voxels_zyx)
    ordered_path[0] = path_voxels_zyx[start_idx]
    remaining_indices.remove(start_idx)
    
    current_idx = start_idx
    for i in range(1, len(path_voxels_zyx)):
        # Find the 2 nearest neighbors (1 will be the point itself)
        # We limit the search radius to 2 voxels to ensure connectivity.
        distances, indices = tree.query(path_voxels_zyx[current_idx], k=4, distance_upper_bound=2.0)
        
        found_next = False
        for neighbor_idx in indices:
            if neighbor_idx in remaining_indices:
                current_idx = neighbor_idx
                ordered_path[i] = path_voxels_zyx[current_idx]
                remaining_indices.remove(current_idx)
                found_next = True
                break
        
        if not found_next:
            # If no neighbor is found within the radius, path is broken.
            # Return the path found so far.
            return ordered_path[:i]
            
    return ordered_path

#############################
### // Main (Function) // ###
#############################

def run_image_to_model(target_image_path, resources_path, ilastik_path, model_path, 
                       input_batch_processing_path, output_batch_processing_path, sub_volume, run_ilastik_batch_processing):

    # --- LAZY LOAD IMPORTS ---
    import pandas as pd
    import networkx as nx
    import vedo
    # -------------------------

    # Prevent crashing by looking for a screen that doesn't exist
    vedo.settings.default_backend = 'k3d'
    
    ############################
    ### // Ilastik Config // ###
    ############################

    run_ilastik_batch_processing = run_ilastik_batch_processing

    # ilastik_path = "/home/dsas627/Desktop/ilastik-1.4.1rc2-gpu-Linux/run_ilastik.sh"
    # model_path = "/home/dsas627/Desktop/Ilastik Image Segmentations/C2-Zstack1_Animal2_NG2_dsRed_CD31_647_GLUT_15042025_vessels_processed.ilp"
    # input_batch_processing_path = "/home/dsas627/Desktop/UCL_confocal/batch_process_input_folder/"
    # output_batch_processing_path = "/home/dsas627/Desktop/UCL_confocal/batch_process_output_folder/"

    ilastik_path = ilastik_path
    model_path = model_path
    input_batch_processing_path = input_batch_processing_path
    output_batch_processing_path = output_batch_processing_path
    
    #######################################
    ### // User Output Folder Config // ###
    #######################################

    output_dir = Path.cwd() / "resources/user_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    #############################
    ### // Image(s) Config // ###
    #############################

    input_file_path = target_image_path
    # input_file_path = "/home/dsas627/PycharmProjects/me_bioeng_cb_vessel_network/Segmentation (Label 1)_skeletal_muscle_pc_no_raw_data.h5"
    labels_to_render_str = "2"
    hdf5_dataset_name_if_applicable = "exported_data"
    # voxel_spacing_str = "1.8660,1.8660,1.8639"
    voxel_spacing_str = "0.6251700,0.6251700,0.9030483"

    ##########################################
    ### // Network Visualisation Config // ###
    ##########################################

    surface_opacity = 0.25
    background_color_str = "white"
    output_image_path_or_none = None
    interactive_mode = True
    centerline_color_str = "blue"
    centerline_point_radius = 3
    render_original_surface_with_centerline = True
    render_nodes_in_3d = True
    junction_node_color_str = "red"
    junction_node_radius = 6
    endpoint_node_color_str = "lime"
    endpoint_node_radius = 4
    inlet_node_color_str = "cyan"
    inlet_node_radius = 7
    outlet_node_color_str = "magenta"
    outlet_node_radius = 7

    #################################
    ### // File I/O CSV Config // ###
    #################################

    output_adjacency_matrix_path = "adjacency_matrix.csv"
    output_node_list_path = "node_list_coordinates.csv"
    output_edge_adjacency_matrix_path = "edge_adjacency_matrix.csv"
    output_edge_list_path = "edge_list_for_matrix.csv"

    ##########################################
    ### // CB Network Generation Config // ###
    ##########################################

    bypass_network_gen_and_just_plot_binary_volume = False

    ### Median Filter Coefficient: 
    ### Decrease for denser less uniform networks, increase for less dense, more uniform networks
    ### Acts like a vessel density resolution slider
    ### e.g. CB Vessels: 1 || Skeletal Muscle Vessels: 10
    smoothing_size = 1

    ### Pyramidal Downsampling Factor for Skeletoniztion: 
    ### Increase to decrease skeletonization time by a factor of n**3 at the cost of losing geometric resolution
    downsample_factor = 1

    compute_connectivity_matrix = True
    process_sub_volume = True
    sub_volume_percentage = sub_volume
    sub_volume_center_offset_x_percent = 0.0
    sub_volume_center_offset_y_percent = 0.0
    sub_volume_center_offset_z_percent = 0.0
    define_inlets_outlets_heuristically = True
    inlet_heuristic_faces = ["Z_min"]
    outlet_heuristic_faces = ["Z_max"]
    boundary_proximity_threshold_percent = 0.05
    pressure_inlet_value = 6666.12
    pressure_outlet_value = 666.612
    # New configuration for segment connectivity DataFrame
    generate_segment_connectivity_file = True
    output_segment_connectivity_filename = "image_to_model_vessel_xyz.csv"  # Base name
    # New configuration for the vessel parameters DataFrame
    generate_vessel_parameters_file = True
    output_vessel_parameters_filename = "image_to_model_parameters.csv"
    # --- NEW: Glomus Cell Superimposition ---
    render_with_glomus_cells = False
    glomus_file_path = "/home/dsas627/PycharmProjects/me_bioeng_cb_vessel_network/Segmentation (Label 1)_glomus_cells.h5"  # <-- IMPORTANT: SET THIS FILE PATH
    glomus_hdf5_dataset_name = "exported_data"
    glomus_color_str = "green"
    glomus_surface_opacity = 0.8
    # --- NEW: Grid Discretization ---
    discretize_with_grid = True
    grid_resolution_xyz = (5, 5, 5)
    intersection_node_color_str = "purple"
    intersection_node_radius = 5
    # --- NEW: Tissue Block Centroid Calculation ---
    generate_tissue_block_centroid_file = True
    output_tissue_block_centroid_filename = "tissue_block_centroids.csv"
    # --- NEW: Tissue Block Face Centroid Calculation ---
    generate_tissue_block_face_file = True
    output_tissue_block_face_filename = "tissue_block_faces.csv"
    # --- NEW: Vessels by Tissue Block Calculation ---
    generate_vessels_by_block_file = False
    output_vessels_by_block_filename = "vessels_by_block.csv"
    # --- NEW: Vessel and Volume Element Array ---
    generate_vessel_and_volume_array_file = True
    output_vessel_and_volume_array_filename = "image_to_model_vessel_and_volume_element_array.csv"
    # --- NEW: Processing for Circulatory Autogen ---
    run_processing_for_circ_autogen = True

    ########################################
    ### // Circulatory Autogen Config // ###
    ########################################

    run_circ_autogen = False

    #################################
    ### // Ilastik Shenanigans // ###
    #################################

    if run_ilastik_batch_processing:

        classifier = IlastikClassifier(ilastik_path, model_path)

        classifier.segment_images(input_dir=input_batch_processing_path,
                                output_dir=output_batch_processing_path,
                                input_ext="*.tif")

    ####################################################
    ### // Process Image Segmentation Shenanigans // ###
    ####################################################

    ### ================================================================================================

    ### // v DEBUG: Load from batch processing output folder v // ###

    # output_folder_path = Path(output_batch_processing_path)
    # segmentation_files = [str (p) for p in output_folder_path.glob('*.h5')]

    # segmentation_data = load_segmentation_data(segmentation_files[1], hdf5_dataset_name_if_applicable)
    # if segmentation_data is None:
    #     return

    ### ================================================================================================

    ### ================================================================================================

    ### // v DEBUG: Load from hard-coded input file path // v ###

    segmentation_data = load_segmentation_data(input_file_path, hdf5_dataset_name_if_applicable)
    if segmentation_data is None:
        return

    ### ================================================================================================

    # --- INSERT THIS DEBUG BLOCK ---
    print("DEBUG: Unique values in loaded data:", np.unique(segmentation_data))
    # -------------------------------

    # --- Load Glomus Data (if enabled) ---
    glomus_data = None
    if render_with_glomus_cells:
        print("\nLoading glomus cell data...")
        glomus_data = load_segmentation_data(glomus_file_path, glomus_hdf5_dataset_name)
        if glomus_data is None:
            print("Warning: Could not load glomus cell data. Continuing without it.")
            render_with_glomus_cells = False  # Disable feature if loading fails

    original_shape_zyx = segmentation_data.shape

    if process_sub_volume:
        print(f"\nProcessing sub-volume. Original shape: {original_shape_zyx}")
        if not (0 < sub_volume_percentage <= 1.0):
            print("Error: sub_volume_percentage must be between 0 (exclusive) and 1.0 (inclusive). Using full volume.")
        else:
            orig_z_dim, orig_y_dim, orig_x_dim = original_shape_zyx
            target_z_dim = max(1, int(orig_z_dim * sub_volume_percentage))
            target_y_dim = max(1, int(orig_y_dim * sub_volume_percentage))
            target_x_dim = max(1, int(orig_x_dim * sub_volume_percentage))
            orig_center_z = orig_z_dim / 2
            orig_center_y = orig_y_dim / 2
            orig_center_x = orig_x_dim / 2
            offset_z = int(orig_z_dim * sub_volume_center_offset_z_percent)
            offset_y = int(orig_y_dim * sub_volume_center_offset_y_percent)
            offset_x = int(orig_x_dim * sub_volume_center_offset_x_percent)
            sub_vol_center_z = orig_center_z + offset_z
            sub_vol_center_y = orig_center_y + offset_y
            sub_vol_center_x = orig_center_x + offset_x
            z_start = max(0, int(sub_vol_center_z - target_z_dim / 2))
            # z_start = 0
            z_end = min(orig_z_dim, z_start + target_z_dim)
            # z_end = orig_z_dim
            y_start = max(0, int(sub_vol_center_y - target_y_dim / 2))
            y_end = min(orig_y_dim, y_start + target_y_dim)
            x_start = max(0, int(sub_vol_center_x - target_x_dim / 2))
            # x_start = 0
            x_end = min(orig_x_dim, x_start + target_x_dim)
            # x_end = orig_x_dim
            if z_end > orig_z_dim: z_start = max(0, orig_z_dim - target_z_dim)
            if y_end > orig_y_dim: y_start = max(0, orig_y_dim - target_y_dim)
            if x_end > orig_x_dim: x_start = max(0, orig_x_dim - target_x_dim)
            z_end = min(orig_z_dim, z_start + target_z_dim)
            y_end = min(orig_y_dim, y_start + target_y_dim)
            x_end = min(orig_x_dim, x_start + target_x_dim)

            segmentation_data = segmentation_data[z_start:z_end, y_start:y_end, x_start:x_end]
            print(f"  Cropped segmentation to sub-volume. New shape: {segmentation_data.shape}")

            # --- Crop Glomus Data Consistently ---
            if glomus_data is not None:
                if glomus_data.shape == original_shape_zyx:
                    glomus_data = glomus_data[z_start:z_end, y_start:y_end, x_start:x_end]
                    print(f"  Cropped glomus cell data to sub-volume. New shape: {glomus_data.shape}")
                else:
                    print(
                        "Warning: Glomus data has different original dimensions. Cannot crop consistently. Disabling feature.")
                    render_with_glomus_cells = False

            print(f"  Sub-volume ZYX voxel start offset from original: ({z_start}, {y_start}, {x_start})")
            if segmentation_data.size == 0:
                raise ValueError("Sub-volume cropping resulted in an empty volume. Please check parameters.")

    try:
        user_spacing_x, user_spacing_y, user_spacing_z = map(float, voxel_spacing_str.split(','))
        vedo_spacing = (user_spacing_x, user_spacing_y, user_spacing_z)
        print(f"Using voxel spacing (X,Y,Z): {vedo_spacing}")

        # ==========================================================
        # /// BYPASS MODE: PLOT RAW BINARY VOLUME ONLY ///
        # ==========================================================

        if bypass_network_gen_and_just_plot_binary_volume:  # Set to False to disable this bypass and run full pipeline
            print("\n--- BYPASSING NETWORK GENERATION: PLOTTING RAW VOLUME ---")
            
            # 1. Pick your label (1 = Vessel, 2 = Background, usually)
            target_label_id = 2
            
            # 2. Create the binary mask
            print(f"Isolating Label {target_label_id}...")
            raw_binary_volume = (segmentation_data == target_label_id).astype(np.uint8)
            
            if not np.any(raw_binary_volume):
                print(f"ERROR: No voxels found for Label {target_label_id}!")
                return

            # 3. Generate the surface using vedo
            print("Generating 3D surface (this may take a moment)...")
            vol_raw = vedo.Volume(raw_binary_volume, spacing=vedo_spacing)
            
            # value=0.5 draws the boundary between 0 and 1
            surf_raw = vol_raw.isosurface(value=0.5).color("yellow").alpha(0.5)
            
            # 4. Show and Exit
            print("Displaying plot...")
            vedo.show(surf_raw, 
                    axes=1, 
                    bg='white', 
                    title=f"Raw Binary Volume: Label {target_label_id}", 
                    viewup='z')
            
            print("Exiting script early.")
            return
        
        # ==========================================================

    except ValueError:
        raise ValueError(f"Error: Invalid spacing format: '{voxel_spacing_str}'.")

    try:
        bg_col = eval(background_color_str) if '(' in background_color_str else background_color_str
    except Exception as e:
        print(f"Warning: Could not parse background color. Defaulting to white. Error: {e}")
        bg_col = 'white'

    plotter_combined_axes_type = 0

    # # --- Create Glomus Actor (if enabled) ---
    glomus_actor = None
    # if render_with_glomus_cells and glomus_data is not None and np.any(glomus_data):
    #     print("\nGenerating 3D surface for glomus cells...")
    #     glomus_vol = vedo.Volume(glomus_data, spacing=vedo_spacing)
    #     glomus_actor = glomus_vol.isosurface(value=0.5)
    #     glomus_actor.color(glomus_color_str)
    #     glomus_actor.opacity(glomus_surface_opacity)
    #     glomus_actor.name = "Glomus_Cells"
    #     print("  Glomus cell actor created.")

    # MODIFIED: Initialize only the necessary actor lists
    surface_only_actors = []
    # This list will temporarily hold unfiltered actors needed for grid bounds calculation
    temp_actors_for_bounds = []

    if labels_to_render_str.lower() == "all":
        print("Rendering all distinct non-zero labels using legosurface...")
        if not np.issubdtype(segmentation_data.dtype, np.integer):
            segmentation_data = segmentation_data.astype(np.uint16)
        vol = vedo.Volume(segmentation_data, spacing=vedo_spacing)
        lego = vol.legosurface(vmin=1)
        if lego.npoints > 0:
            lego.cmap('viridis').opacity(surface_opacity)
            temp_actors_for_bounds.append(lego.clone())
        else:
            print("No surfaces generated for 'all' labels mode.")
    else:
        try:
            label_ids = [int(label.strip()) for label in labels_to_render_str.split(',')]
            if not label_ids:
                raise ValueError("No labels provided.")

            available_colors = ['tomato', 'mediumseagreen', 'cornflowerblue', 'gold', 'orchid', 'darkturquoise',
                                'sandybrown', 'lightpink', 'olivedrab', 'slateblue', 'darkorange', 'skyblue', 'crimson',
                                'forestgreen', 'royalblue', 'yellow', 'darkviolet', 'deepskyblue', 'chocolate',
                                'hotpink']

            for i, label_id in enumerate(label_ids):
                print(f"\n  Processing label {label_id}...")
                binary_volume = (segmentation_data == label_id)
                if not np.any(binary_volume):
                    print(f"  Warning: Label {label_id} not found in (sub)volume. Skipping.")
                    continue

                volume_shape_for_masking = binary_volume.shape
                vol_label = vedo.Volume(binary_volume.astype(np.uint8), spacing=vedo_spacing)
                isosurface = vol_label.isosurface(value=0.5)
                if isosurface.npoints == 0:
                    print(f"    No surface points for label {label_id}. Skipping.")
                    continue

                color_name = available_colors[i % len(available_colors)]
                isosurface_seg_only = isosurface.clone()
                isosurface_seg_only.color(color_name)
                isosurface_seg_only.name = f"Surface_Label_{label_id}_SegOnly"
                isosurface_seg_only.opacity(surface_opacity)
                surface_only_actors.append(isosurface_seg_only)
                print(f"    Label {label_id} surface for seg-only plot prepared.")

                num_skeleton_voxels = 0
                skeleton_volume = None
                all_nodes_image_zyx = np.empty((0, 3))
                node_types_list = []
                radius_map = None

                try:
                    from skimage.morphology import skeletonize
                    print(f"    Generating 3D skeleton for label {label_id}...")

                    # --- NEW: SMOOTHING STEP TO FIX DENSE NETWORKS ---
                    
                    if smoothing_size > 1:
                        print(f"      [Fix] Applying Fast Binary Smoothing (size={smoothing_size}) to clean large vessels...")
                        
                        # Convert to float for accurate mean calculation
                        float_vol = binary_volume.astype(np.float32)
                        
                        # Apply uniform filter (mean)
                        blurred_vol = ndimage.uniform_filter(float_vol, size=smoothing_size)
                        
                        # Threshold back to binary
                        binary_volume = (blurred_vol > 0.5)
                        
                        # Optional: If you have "swiss cheese" holes inside the vessel, fill them
                        binary_volume = ndimage.binary_closing(binary_volume, iterations=2)
                    else:
                        print(f"      [Fix] Smoothing size is {smoothing_size}, skipping smoothing step.")

                    
                    print(f"      Skeletonizing...")

                    # --- OPTIMISATION: PYRAMIDAL SKELETONIZATION ---

                    if downsample_factor > 1:
                        print(f"        [Optimisation] Downsampling by {downsample_factor}x for faster skeletonization...")
                        
                        # 1. Downscale the binary volume
                        # anti_aliasing=False and order=0 preserves the binary nature (0 or 1)
                        print(f"          Rescaling...")
                        small_vol = rescale(binary_volume, 1.0/downsample_factor, 
                                            order=0, preserve_range=True, anti_aliasing=False).astype(bool)
                        
                        # 2. Skeletonize the small volume (The heavy lifting happens here, but it's fast now)
                        print(f"          Skeletonizing Rescaled binary_volume...")
                        small_skel = skeletonize(small_vol)
                        
                        # 3. Upscale back to original size
                        # We use resize to ensure the shape matches the original 'binary_volume' exactly
                        print(f"          Upsizing Skeletonized Rescaled binary_volume...")
                        thick_skel = resize(small_skel, binary_volume.shape, 
                                            order=0, preserve_range=True, anti_aliasing=False).astype(bool)
                        
                        # 4. Final thinning pass
                        # The upscaled skeleton is now "thick" (e.g., 2x2 voxels wide).
                        # We run skeletonize one last time to thin it back to 1 pixel. 
                        # This is very fast because the volume is already mostly empty.
                        print(f"          Skeletonizing Final Processed binary_volume...")
                        skeleton_volume = skeletonize(thick_skel)
                    else:
                        # Standard slow processing
                        skeleton_volume = skeletonize(binary_volume)
                    # ------------------------------------------------

                    num_skeleton_voxels = np.sum(skeleton_volume)
                    print(f"    Skeleton: {num_skeleton_voxels} voxels found.")
                    if num_skeleton_voxels > 0:
                        s_indices = np.argwhere(skeleton_volume)
                        s_pts_vedo = np.zeros_like(s_indices, dtype=float)
                        s_pts_vedo[:, 0] = s_indices[:, 2] * vedo_spacing[0]
                        s_pts_vedo[:, 1] = s_indices[:, 1] * vedo_spacing[1]
                        s_pts_vedo[:, 2] = s_indices[:, 0] * vedo_spacing[2]
                        skeleton_actor = vedo.Points(s_pts_vedo, r=centerline_point_radius, c=centerline_color_str)
                        skeleton_actor.name = f"Centerline_Label_{label_id}"
                        # MODIFIED: Append to temporary list for bounds calculation
                        temp_actors_for_bounds.append(skeleton_actor)
                        print(f"    Added centerline for label {label_id}.")

                        if binary_volume is not None:
                            try:
                                print(
                                    f"        Calculating distance transform for radius estimation for label {label_id}...")
                                spacing_zyx_for_dt = (vedo_spacing[2], vedo_spacing[1], vedo_spacing[0])
                                radius_map = ndimage.distance_transform_edt(binary_volume, sampling=spacing_zyx_for_dt)
                                print(f"        Distance transform calculated.")
                            except Exception as e_dt:
                                print(f"        Error calculating distance transform: {e_dt}")
                                radius_map = None
                except Exception as e_skel:
                    print(f"    Error skeletonizing: {e_skel}")

                G_undirected = nx.Graph()

                if (
                        render_nodes_in_3d or compute_connectivity_matrix) and skeleton_volume is not None and num_skeleton_voxels > 0:
                    print(f"      Detecting nodes (junctions/endpoints) using ndimage for label {label_id}...")
                    kernel = np.ones((3, 3, 3), dtype=np.uint8)
                    kernel[1, 1, 1] = 0
                    neighbor_map = ndimage.convolve(skeleton_volume.astype(np.uint8), kernel, mode='constant', cval=0)

                    refined_junction_centroids_image_current = np.empty((0, 3))
                    raw_j = skeleton_volume & (neighbor_map > 2)
                    if np.any(raw_j):
                        lbl_j, n_j = ndimage.label(raw_j, structure=np.ones((3, 3, 3), dtype=bool))
                        if n_j > 0:
                            com_j = ndimage.center_of_mass(raw_j.astype(float), lbl_j, np.arange(1, n_j + 1))
                            v_j = [c for c in com_j if isinstance(c, tuple) and not np.any(np.isnan(c))] if isinstance(
                                com_j, list) else (
                                [com_j] if isinstance(com_j, tuple) and not np.any(np.isnan(com_j)) else [])
                            if v_j:
                                refined_junction_centroids_image_current = np.array(v_j)
                    print(f"        Found {refined_junction_centroids_image_current.shape[0]} raw junction centroids.")

                    refined_endpoint_centroids_image_current = np.empty((0, 3))
                    raw_ep = skeleton_volume & (neighbor_map == 1)
                    if np.any(raw_ep):
                        lbl_ep, n_ep = ndimage.label(raw_ep, structure=np.ones((3, 3, 3), dtype=bool))
                        if n_ep > 0:
                            com_ep = ndimage.center_of_mass(raw_ep.astype(float), lbl_ep, np.arange(1, n_ep + 1))
                            v_ep = [c for c in com_ep if
                                    isinstance(c, tuple) and not np.any(np.isnan(c))] if isinstance(com_ep, list) else (
                                [com_ep] if isinstance(com_ep, tuple) and not np.any(np.isnan(com_ep)) else [])
                            if v_ep:
                                refined_endpoint_centroids_image_current = np.array(v_ep)
                    print(f"        Found {refined_endpoint_centroids_image_current.shape[0]} raw endpoint centroids.")

                    current_label_nodes_zyx_list = []
                    current_node_types_list = []
                    if refined_junction_centroids_image_current.size > 0:
                        current_label_nodes_zyx_list.extend(list(refined_junction_centroids_image_current))
                        current_node_types_list.extend(['junction'] * refined_junction_centroids_image_current.shape[0])
                    if refined_endpoint_centroids_image_current.size > 0:
                        current_label_nodes_zyx_list.extend(list(refined_endpoint_centroids_image_current))
                        current_node_types_list.extend(['endpoint'] * refined_endpoint_centroids_image_current.shape[0])

                    if current_label_nodes_zyx_list:
                        all_nodes_image_zyx = np.array(current_label_nodes_zyx_list)
                        node_types_list = list(current_node_types_list)

                        if compute_connectivity_matrix and define_inlets_outlets_heuristically:
                            print("        Applying heuristics to define inlets/outlets...")
                            vol_dims_zyx = binary_volume.shape
                            z_thresh = vol_dims_zyx[0] * boundary_proximity_threshold_percent
                            y_thresh = vol_dims_zyx[1] * boundary_proximity_threshold_percent
                            x_thresh = vol_dims_zyx[2] * boundary_proximity_threshold_percent

                            for node_idx, node_coord_zyx in enumerate(all_nodes_image_zyx):
                                if node_types_list[node_idx] == 'endpoint':
                                    z_c, y_c, x_c = node_coord_zyx
                                    is_inlet, is_outlet = False, False
                                    if "Z_min" in inlet_heuristic_faces and z_c < z_thresh: is_inlet = True
                                    if "Z_max" in inlet_heuristic_faces and z_c > (
                                            vol_dims_zyx[0] - 1 - z_thresh): is_inlet = True
                                    if "Y_min" in inlet_heuristic_faces and y_c < y_thresh: is_inlet = True
                                    if "Y_max" in inlet_heuristic_faces and y_c > (
                                            vol_dims_zyx[1] - 1 - y_thresh): is_inlet = True
                                    if "X_min" in inlet_heuristic_faces and x_c < x_thresh: is_inlet = True
                                    if "X_max" in inlet_heuristic_faces and x_c > (
                                            vol_dims_zyx[2] - 1 - x_thresh): is_inlet = True

                                    if "Z_min" in outlet_heuristic_faces and z_c < z_thresh: is_outlet = True
                                    if "Z_max" in outlet_heuristic_faces and z_c > (
                                            vol_dims_zyx[0] - 1 - z_thresh): is_outlet = True
                                    if "Y_min" in outlet_heuristic_faces and y_c < y_thresh: is_outlet = True
                                    if "Y_max" in outlet_heuristic_faces and y_c > (
                                            vol_dims_zyx[1] - 1 - y_thresh): is_outlet = True
                                    if "X_min" in outlet_heuristic_faces and x_c < x_thresh: is_outlet = True
                                    if "X_max" in outlet_heuristic_faces and x_c > (
                                            vol_dims_zyx[2] - 1 - x_thresh): is_outlet = True

                                    if is_inlet and is_outlet:
                                        node_types_list[node_idx] = 'inlet_outlet'
                                    elif is_inlet:
                                        node_types_list[node_idx] = 'inlet'
                                    elif is_outlet:
                                        node_types_list[node_idx] = 'outlet'
                            print(f"        Node types updated with inlet/outlet heuristics.")

                if render_nodes_in_3d and all_nodes_image_zyx.shape[0] > 0:
                    # MODIFIED: This block now only creates actors for the temporary bounds calculation list
                    coords_by_type = {'junction': [], 'endpoint': [], 'inlet': [], 'outlet': [], 'inlet_outlet': []}
                    for idx, node_type_from_list in enumerate(node_types_list):
                        coords_by_type.setdefault(node_type_from_list, []).append(all_nodes_image_zyx[idx])

                    node_render_configs = {
                        'junction': (junction_node_color_str, junction_node_radius, "Junctions"),
                        'endpoint': (endpoint_node_color_str, endpoint_node_radius, "Endpoints"),
                        'inlet': (inlet_node_color_str, inlet_node_radius, "Inlets"),
                        'outlet': (outlet_node_color_str, outlet_node_radius, "Outlets"),
                        'inlet_outlet': ('purple', junction_node_radius, "Inlet_Outlets")
                    }

                    for n_type_key, (color, radius, name_prefix) in node_render_configs.items():
                        if coords_by_type.get(n_type_key):
                            centroids_zyx_nodes = np.array(coords_by_type[n_type_key])
                            if centroids_zyx_nodes.shape[0] > 0:
                                pts_vedo_nodes = np.zeros_like(centroids_zyx_nodes, dtype=float)
                                pts_vedo_nodes[:, 0] = centroids_zyx_nodes[:, 2] * vedo_spacing[0]
                                pts_vedo_nodes[:, 1] = centroids_zyx_nodes[:, 1] * vedo_spacing[1]
                                pts_vedo_nodes[:, 2] = centroids_zyx_nodes[:, 0] * vedo_spacing[2]

                                node_actor = vedo.Points(pts_vedo_nodes, r=radius, c=color)
                                node_actor.name = f"{name_prefix}_Label_{label_id}"
                                temp_actors_for_bounds.append(node_actor)
                                print(
                                    f"        Added {centroids_zyx_nodes.shape[0]} '{n_type_key}' nodes for bounds calculation.")

                if compute_connectivity_matrix and G_undirected is not None and skeleton_volume is not None and num_skeleton_voxels > 0 and all_nodes_image_zyx.size > 0:
                    print(f"      Populating initial undirected graph G_undirected for label {label_id}...")
                    all_nodes_world_xyz = np.zeros_like(all_nodes_image_zyx, dtype=float)
                    all_nodes_world_xyz[:, 0] = all_nodes_image_zyx[:, 2] * vedo_spacing[0]
                    all_nodes_world_xyz[:, 1] = all_nodes_image_zyx[:, 1] * vedo_spacing[1]
                    all_nodes_world_xyz[:, 2] = all_nodes_image_zyx[:, 0] * vedo_spacing[2]
                    for node_idx, img_coords_zyx_centroid in enumerate(all_nodes_image_zyx):
                        G_undirected.add_node(node_idx, pos_zyx_image=tuple(img_coords_zyx_centroid),
                                                pos_xyz_world=tuple(all_nodes_world_xyz[node_idx]),
                                                type=node_types_list[node_idx])
                    print(f"        Added {G_undirected.number_of_nodes()} nodes to undirected graph G_undirected.")

                    if 'neighbor_map' not in locals() and skeleton_volume is not None:
                        kernel = np.ones((3, 3, 3), dtype=np.uint8)
                        kernel[1, 1, 1] = 0
                        neighbor_map = ndimage.convolve(skeleton_volume.astype(np.uint8), kernel, mode='constant',
                                                        cval=0)

                    raw_j_mask = skeleton_volume & (neighbor_map > 2)
                    raw_ep_mask = skeleton_volume & (neighbor_map == 1)
                    all_raw_node_vox_mask = raw_j_mask | raw_ep_mask
                    skel_segments_only = skeleton_volume.copy()
                    skel_segments_only[all_raw_node_vox_mask] = False
                    labeled_segments, num_segments = ndimage.label(skel_segments_only,
                                                                    structure=np.ones((3, 3, 3), dtype=bool))
                    print(f"        Found {num_segments} potential segments.")

                    # --- SEGMENT PROCESSING LOOP FOR GRAPH & FEEDING STATUS ---
                    if num_segments > 0:
                        if all_nodes_image_zyx.shape[0] > 0:
                            from scipy.spatial import KDTree
                            node_kdtree = KDTree(all_nodes_image_zyx)
                        
                        print(f"        Processing {num_segments} segments for graph connections...")

                        # 1. Get tight bounding boxes
                        segment_slices = ndimage.find_objects(labeled_segments)
                        vol_shape = labeled_segments.shape

                        for seg_id in tqdm(range(1, num_segments + 1), desc="      Segments", unit="segment",
                                            leave=False):
                            
                            # 2. Get the tight slice
                            sl = segment_slices[seg_id - 1]
                            if sl is None: continue

                            # 3. EXPAND THE SLICE (Padding)
                            # We must pad the slice to capture the junction nodes which are just *outside* the segment
                            padding = 3 
                            
                            sl_padded = tuple(
                                slice(max(0, s.start - padding), min(max_dim, s.stop + padding))
                                for s, max_dim in zip(sl, vol_shape)
                            )

                            # 4. Crop using the PADDED slice
                            seg_crop = labeled_segments[sl_padded]
                            current_seg_mask_local = (seg_crop == seg_id)

                            if not np.any(current_seg_mask_local): continue

                            if all_nodes_image_zyx.shape[0] > 0:
                                seg_vox_count = np.sum(current_seg_mask_local)
                                if seg_vox_count < 2: continue
                                
                                # Perform dilation on the padded local crop
                                dilated_seg_local = ndimage.binary_dilation(current_seg_mask_local, structure=np.ones((3, 3, 3)))
                                
                                # Crop the global node mask to the PADDED slice
                                node_mask_crop = all_raw_node_vox_mask[sl_padded]
                                
                                # Now this intersection will work because we included the neighbors
                                contact_zone_local = dilated_seg_local & node_mask_crop
                                contact_indices_local = np.argwhere(contact_zone_local)

                                conn_refined_node_ids = set()
                                if contact_indices_local.shape[0] > 0:
                                    # Calculate offset based on PADDED slice start
                                    offset = np.array([s.start for s in sl_padded])
                                    contact_node_raw_vox_zyx = contact_indices_local + offset

                                    dists, closest_node_idxs = node_kdtree.query(contact_node_raw_vox_zyx, k=1)
                                    if not isinstance(closest_node_idxs, (np.ndarray, list)): closest_node_idxs = [
                                        closest_node_idxs]
                                    if not isinstance(dists, (np.ndarray, list)): dists = [dists]

                                    for idx_kdt, dist_v in zip(closest_node_idxs, dists):
                                        if dist_v < 7.0:
                                            conn_refined_node_ids.add(idx_kdt)

                                if len(conn_refined_node_ids) == 2:
                                    n1_id, n2_id = list(conn_refined_node_ids)
                                    if n1_id == n2_id: continue

                                    p1w = G_undirected.nodes[n1_id]['pos_xyz_world']
                                    p2w = G_undirected.nodes[n2_id]['pos_xyz_world']
                                    seg_len_w = np.linalg.norm(np.array(p1w) - np.array(p2w))
                                    if seg_len_w == 0: seg_len_w = np.mean(vedo_spacing)

                                    # Get voxel coords (Local + Padded Offset -> Global)
                                    local_voxels = np.argwhere(current_seg_mask_local)
                                    offset = np.array([s.start for s in sl_padded])
                                    voxel_coords_zyx = local_voxels + offset

                                    G_undirected.add_edge(n1_id, n2_id, voxel_path=voxel_coords_zyx)

                    # --- NEW PIPELINE STEP 1: Filter for the largest connected component ---
                    print("      Filtering for the largest connected component...")
                    connected_components = sorted(nx.connected_components(G_undirected), key=len, reverse=True)
                    if connected_components:
                        giant_component_nodes = connected_components[0]
                        num_removed = G_undirected.number_of_nodes() - len(giant_component_nodes)
                        print(f"        Identified giant component with {len(giant_component_nodes)} nodes. Removing {num_removed} nodes from small, disconnected fragments.")
                        G_clean_undirected = G_undirected.subgraph(giant_component_nodes).copy()
                    else:
                        print("        Warning: No connected components found. Using original graph.")
                        G_clean_undirected = G_undirected.copy()

                    # --- NEW PIPELINE STEP 2: Discretize the CLEAN graph ---
                    if discretize_with_grid:
                        full_network_assembly = vedo.Assembly(temp_actors_for_bounds)
                        bounds = full_network_assembly.bounds()
                        grid_planes_x = np.linspace(bounds[0], bounds[1], grid_resolution_xyz[0] + 1)
                        grid_planes_y = np.linspace(bounds[2], bounds[3], grid_resolution_xyz[1] + 1)
                        grid_planes_z = np.linspace(bounds[4], bounds[5], grid_resolution_xyz[2] + 1)

                        # --- ADDED: Define grid origin and cell size for later use ---
                        grid_origin = np.array([bounds[0], bounds[2], bounds[4]])
                        grid_dims = np.array([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
                        cell_size = grid_dims / np.array(grid_resolution_xyz)

                        G_discretized = discretize_network_with_grid(
                                G_clean_undirected, grid_planes_x, grid_planes_y, grid_planes_z, vedo_spacing)
                    else:
                        G_discretized = G_clean_undirected.copy()

                    # --- NEW PIPELINE STEP 3: Calculate Flow on the CLEAN, Discretized Graph ---
                    print(f"        Calculating flow direction for the filtered and discretized graph...")
                    inlet_nodes = {n for n, data in G_discretized.nodes(data=True) if data['type'] == 'inlet'}
                    outlet_nodes = {n for n, data in G_discretized.nodes(data=True) if data['type'] == 'outlet'}

                    boundary_pressure_nodes = {}
                    for n, data in G_discretized.nodes(data=True):
                        if data['type'] == 'inlet':
                            boundary_pressure_nodes[n] = pressure_inlet_value
                        elif data['type'] == 'outlet':
                            boundary_pressure_nodes[n] = pressure_outlet_value
                        elif data['type'] == 'inlet_outlet':
                            boundary_pressure_nodes[n] = (pressure_inlet_value + pressure_outlet_value) / 2.0

                    candidate_internal_nodes = {n for n in G_discretized.nodes() if n not in boundary_pressure_nodes}
                    solvable_internal_nodes_set = set()

                    if candidate_internal_nodes:
                        for component_node_set in nx.connected_components(G_discretized):
                            internal_nodes_in_comp = candidate_internal_nodes.intersection(component_node_set)
                            if not internal_nodes_in_comp: continue
                            boundary_nodes_in_comp = boundary_pressure_nodes.keys() & component_node_set
                            if bool(boundary_nodes_in_comp):
                                solvable_internal_nodes_set.update(internal_nodes_in_comp)

                    internal_nodes_for_solve = sorted(list(solvable_internal_nodes_set))
                    node_pressures = boundary_pressure_nodes.copy()
                    for n in internal_nodes_for_solve: node_pressures[n] = np.nan
                    
                    G_final_processed = G_discretized.to_directed() if not G_discretized.is_directed() else G_discretized.copy()

                    if internal_nodes_for_solve and (inlet_nodes or outlet_nodes or len(boundary_pressure_nodes) >= 2):
                        edge_conductances = {}
                        for u_edge, v_edge in G_discretized.edges():
                            p1w = G_discretized.nodes[u_edge]['pos_xyz_world']
                            p2w = G_discretized.nodes[v_edge]['pos_xyz_world']
                            length = np.linalg.norm(np.array(p1w) - np.array(p2w))
                            
                            # Estimate radius by averaging endpoint radii from distance transform if available
                            r1 = radius_map[tuple(np.round(G_discretized.nodes[u_edge]['pos_zyx_image']).astype(int))] if radius_map is not None else np.mean(vedo_spacing) / 2
                            r2 = radius_map[tuple(np.round(G_discretized.nodes[v_edge]['pos_zyx_image']).astype(int))] if radius_map is not None else np.mean(vedo_spacing) / 2
                            radius = (r1 + r2) / 2.0
                            
                            if length <= 1e-9: length = 1e-9
                            if radius <= 1e-9: radius = 1e-9
                            
                            # --- FIX: Convert units for accurate Conductance/Flow in the CSV ---
                            radius_m = radius * 1e-6   # Convert microns to meters
                            length_m = length * 1e-6   # Convert microns to meters
                            viscosity_blood = 3.5e-3   # Pa*s (approximate for blood)
                            
                            # Calculate Conductance in m^3 / (Pa*s)
                            K_uv = (np.pi * (radius_m ** 4)) / (8 * viscosity_blood * length_m)
                            
                            edge_conductances[(u_edge, v_edge)] = K_uv
                            edge_conductances[(v_edge, u_edge)] = K_uv

                        internal_node_map = {node_id: i for i, node_id in enumerate(internal_nodes_for_solve)}
                        num_internal = len(internal_nodes_for_solve)
                        M_coeff = np.zeros((num_internal, num_internal))
                        B_const = np.zeros(num_internal)

                        for i_row, node_i_id in enumerate(internal_nodes_for_solve):
                            sum_K_ij_diag = 0.0
                            sum_K_ij_Pj_boundary = 0.0
                            for neighbor_j_id in G_discretized.neighbors(node_i_id):
                                K_ij = edge_conductances.get((node_i_id, neighbor_j_id), 1e-12)
                                sum_K_ij_diag += K_ij
                                if neighbor_j_id in internal_node_map:
                                    j_idx = internal_node_map[neighbor_j_id]
                                    M_coeff[i_row, j_idx] = -K_ij
                                else:
                                    sum_K_ij_Pj_boundary += K_ij * node_pressures.get(neighbor_j_id, 0)
                            M_coeff[i_row, i_row] = sum_K_ij_diag
                            B_const[i_row] = sum_K_ij_Pj_boundary

                        try:
                            if num_internal > 0:
                                solved_P_internal = np.linalg.solve(M_coeff, B_const)
                                for i_map_idx, node_i_id_internal in enumerate(internal_nodes_for_solve):
                                    node_pressures[node_i_id_internal] = solved_P_internal[i_map_idx]
                                
                                G_final_processed = nx.DiGraph()
                                for n_id, data_dict in G_discretized.nodes(data=True):
                                    G_final_processed.add_node(n_id, **data_dict, pressure=node_pressures.get(n_id, np.nan))
                                for u_node, v_node in G_discretized.edges():
                                    P_u, P_v = node_pressures.get(u_node, np.nan), node_pressures.get(v_node, np.nan)
                                    K_uv_edge = edge_conductances.get((u_node, v_node), 1e-12)
                                    if not (np.isnan(P_u) or np.isnan(P_v)):
                                        flow = (P_u - P_v) * K_uv_edge
                                        if P_u > P_v:
                                            G_final_processed.add_edge(u_node, v_node, flow_rate=flow, pressure_drop=P_u - P_v, conductance=K_uv_edge)
                                        elif P_v > P_u:
                                            G_final_processed.add_edge(v_node, u_node, flow_rate=-flow, pressure_drop=P_v - P_u, conductance=K_uv_edge)
                        except np.linalg.LinAlgError:
                            print("          Error: Singular matrix during flow calculation.")

                    # --- PIPELINE STEP 4: Classify segments, perform final filter, and save all files ---
                    print("\n      Classifying and saving final network data...")
                    # --- NEW: Add final attributes and PURGE invalid geometry ---
                    edges_to_purge = []
                    
                    # Iterate over a list copy so we can modify the graph safely
                    for u, v, data in list(G_final_processed.edges(data=True)):
                        p1_world = G_final_processed.nodes[u]['pos_xyz_world']
                        p2_world = G_final_processed.nodes[v]['pos_xyz_world']
                        p1_img = G_final_processed.nodes[u]['pos_zyx_image']
                        p2_img = G_final_processed.nodes[v]['pos_zyx_image']
                        
                        # 1. Calculate Raw Geometry
                        raw_length = np.linalg.norm(np.array(p1_world) - np.array(p2_world))
                        
                        segment_mask_for_attrs = create_mask_for_line(p1_img, p2_img, volume_shape_for_masking)
                        if radius_map is not None:
                            local_rads = radius_map[segment_mask_for_attrs]
                            raw_radius = np.mean(local_rads) if local_rads.size > 0 else np.mean(vedo_spacing) / 2
                        else:
                            raw_radius = np.mean(vedo_spacing) / 2

                        # 2. Check for Invalid Geometry (Pruning Condition)
                        # Thresholds: Length < 1 micron, Radius < 0.1 micron (prevents Division by Zero/NaN)
                        if raw_length < 1e-3 or raw_radius < 1e-7:
                            edges_to_purge.append((u, v))
                            continue # Skip this edge, it will be deleted
                        
                        # 3. Assign Attributes if valid
                        data['length'] = raw_length
                        data['radius'] = raw_radius
                        
                        if discretize_with_grid:
                            centroid = (np.array(p1_world) + np.array(p2_world)) / 2
                            grid_id_raw = get_grid_block_id(centroid, grid_origin, cell_size)
                            if grid_id_raw:
                                n_x, n_y, n_z = grid_resolution_xyz
                                clamped_i = max(0, min(grid_id_raw[0], n_x - 1))
                                clamped_j = max(0, min(grid_id_raw[1], n_y - 1))
                                clamped_k = max(0, min(grid_id_raw[2], n_z - 1))
                                data['grid_block_id'] = get_grid_block_id_format((clamped_i, clamped_j, clamped_k), grid_resolution_xyz)
                        
                        if render_with_glomus_cells and glomus_data is not None:
                            data['feeding_status'] = 'feeding' if np.any(segment_mask_for_attrs & glomus_data) else 'non-feeding'

                    # --- EXECUTE GEOMETRY PURGE ---
                    if edges_to_purge:
                        print(f"      Purging {len(edges_to_purge)} edges with near-zero length or radius...")
                        G_final_processed.remove_edges_from(edges_to_purge)

                    # --- Final Filtering and File Saving ---
                    if G_final_processed.number_of_edges() > 0:
                        G_ln = nx.line_graph(G_final_processed)
                        initial_edge_list = sorted(list(G_ln.nodes()))

                        if initial_edge_list:
                            edge_adj_mtx = nx.to_numpy_array(G_ln, nodelist=initial_edge_list)
                            
                            # --- RE-INTRODUCED CRUCIAL FILTERING LOGIC (Strong Version) ---
                            print("         Performing strong topological pruning (removing floating islands)...")

                            # 1. Build a temporary graph of the edge connections
                            G_temp_connectivity = nx.from_numpy_array(edge_adj_mtx)

                            # 2. Find the largest connected component (The Main Network)
                            #    This keeps the main network and discards ALL disconnected clusters (islands)
                            if G_temp_connectivity.number_of_nodes() > 0:
                                largest_cc_indices = max(nx.connected_components(G_temp_connectivity), key=len)
                                keep_edge_indices = sorted(list(largest_cc_indices))
                            else:
                                keep_edge_indices = []

                            num_original_edges = len(initial_edge_list)
                            num_processed_edges = len(keep_edge_indices)
                            
                            if num_processed_edges < num_original_edges:
                                print(f"          Pruned {num_original_edges - num_processed_edges} disconnected edges (floating islands).")
                                # Slice the matrix and list to keep only valid edges
                                processed_edge_adj_mtx = edge_adj_mtx[np.ix_(keep_edge_indices, keep_edge_indices)]
                                processed_edge_list = [initial_edge_list[i] for i in keep_edge_indices]
                            else:
                                print("          Network is fully connected. No pruning needed.")
                                processed_edge_adj_mtx = edge_adj_mtx
                                processed_edge_list = initial_edge_list

                            # --- VERIFICATION STEP: Check the filtered matrix for any remaining isolated segments ---
                            print("         Verifying the filtered adjacency matrix...")
                            if num_processed_edges > 0:
                                # Recalculate sums on the new, potentially smaller, processed matrix
                                final_row_sums = processed_edge_adj_mtx.sum(axis=1)
                                final_col_sums = processed_edge_adj_mtx.sum(axis=0)
                                
                                # Find if any indices exist where BOTH the row and its associated column are all zeros
                                remaining_isolated_indices = np.where((final_row_sums == 0) & (final_col_sums == 0))[0]
                                
                                if remaining_isolated_indices.size == 0:
                                    print("           Verification successful: No all-zero rows and columns found. ✅")
                                else:
                                    print(f"          Verification FAILED: Found {remaining_isolated_indices.size} remaining isolated segments. ❌")
                            else:
                                # Handle case where the final matrix is empty
                                print("           Skipping verification as the matrix is empty.")
                            
                            np.savetxt(output_dir / f"label_{label_id}_{output_edge_adjacency_matrix_path}", processed_edge_adj_mtx, delimiter=',', fmt='%d')
                            print(f"        Edge adjacency matrix saved.")
                            
                            # All subsequent file generation now uses the correctly filtered `processed_edge_list`
                            with open(f"label_{label_id}_{output_edge_list_path}", 'w') as f_e:
                                f_e.write("Edge_ID_in_Matrix,Source_Node_in_G,Target_Node_in_G,WorldLength,EstimatedRadius,Conductance,FlowRate,PressureDrop,FeedingStatus,GridBlockID\n")
                                for m_idx, e_tpl in enumerate(processed_edge_list):
                                    attr = G_final_processed.edges[e_tpl]
                                    f_e.write(f"{m_idx},{e_tpl[0]},{e_tpl[1]},{attr.get('length', 0.0):.4f},{attr.get('radius', 0.0):.4f},{attr.get('conductance', 0.0):.4e},{attr.get('flow_rate', 0.0):.4e},{attr.get('pressure_drop', 0.0):.4e},{attr.get('feeding_status', 'N/A')},{attr.get('grid_block_id', 'N/A')}\n")
                            print(f"        Edge list saved.")
                            
                            # All Beatrice-related file generation starts here
                            # It correctly uses the final processed_edge_list
                            if generate_vessels_by_block_file or generate_segment_connectivity_file:
                                vessels_by_block = {get_grid_block_id_format((i_grid, j_grid, k_grid), grid_resolution_xyz): [] for i_grid in range(grid_resolution_xyz[0]) for j_grid in range(grid_resolution_xyz[1]) for k_grid in range(grid_resolution_xyz[2])}
                                for m_idx, e_tpl in enumerate(processed_edge_list):
                                    edge_attr = G_final_processed.edges[e_tpl]
                                    grid_id = edge_attr.get('grid_block_id', 'N/A')
                                    if grid_id in vessels_by_block:
                                        vessels_by_block[grid_id].append(f"L{label_id}_E{m_idx}")
                                
                                if generate_vessels_by_block_file:
                                    block_data_list = [{'GridBlockID': bid, 'Vessels': ' '.join(vlist)} for bid, vlist in vessels_by_block.items()]
                                    if block_data_list:
                                        df_vessels = pd.DataFrame(block_data_list)
                                        df_vessels.to_csv(f"label_{label_id}_{output_vessels_by_block_filename}", index=False)
                                        print(f"          Vessels by block data saved.")

                                if generate_segment_connectivity_file:
                                    list_for_segment_connectivity_df = []
                                    edge_tuple_to_name_map = {edge_tpl_map: f"L{label_id}_E{m_idx}" for m_idx, edge_tpl_map in enumerate(processed_edge_list)}
                                    
                                    for current_edge_tuple in processed_edge_list:
                                        u_seg, v_seg = current_edge_tuple
                                        current_segment_name = edge_tuple_to_name_map[current_edge_tuple]
                                        inp_vessels_names = [edge_tuple_to_name_map[pred_edge] for pred_edge in G_final_processed.in_edges(u_seg) if pred_edge in edge_tuple_to_name_map]
                                        out_vessels_names = [edge_tuple_to_name_map[succ_edge] for succ_edge in G_final_processed.out_edges(v_seg) if succ_edge in edge_tuple_to_name_map]
                                        
                                        # [NEW] Calculate Centroid
                                        p1_world = np.array(G_final_processed.nodes[u_seg]['pos_xyz_world'])
                                        p2_world = np.array(G_final_processed.nodes[v_seg]['pos_xyz_world'])
                                        centroid = (p1_world + p2_world) / 2.0

                                        # [MODIFIED] Append x, y, z to the dictionary
                                        list_for_segment_connectivity_df.append({
                                            'name': current_segment_name, 
                                            'inp_vessels': inp_vessels_names, 
                                            'out_vessels': out_vessels_names,
                                            'x': centroid[0], 
                                            'y': centroid[1], 
                                            'z': centroid[2]
                                        })
                                    
                                    if list_for_segment_connectivity_df:
                                        connectivity_df = pd.DataFrame(list_for_segment_connectivity_df)
                                        connectivity_df['inp_vessels'] = connectivity_df['inp_vessels'].apply(lambda x: ' '.join(map(str, x)))
                                        connectivity_df['out_vessels'] = connectivity_df['out_vessels'].apply(lambda x: ' '.join(map(str, x)))
                                        connectivity_df.to_csv(output_dir / f"label_{label_id}_{output_segment_connectivity_filename}", index=False)
                                        print(f"          Segment connectivity data saved.")
        
        except Exception as e_s:
            print(f"        Error saving output files for Label {label_id}: {e_s}")
            import traceback
            traceback.print_exc()

            if render_original_surface_with_centerline:
                isosurface.color(color_name)
                isosurface.name = f"Surface_Label_{label_id}"
                opac = surface_opacity
                if num_skeleton_voxels > 0 and opac > 0.3: opac = max(0.1, opac * 0.4)
                isosurface.opacity(opac)
                # filtered_actors_for_combined_plot.append(isosurface)
                print(f"    Label {label_id} surface (opacity {opac:.2f}) added to combined plot.")
            elif not render_original_surface_with_centerline and num_skeleton_voxels == 0:
                print(f"    Not rendering surface for label {label_id} (config/no skeleton).")

        except ValueError as e:
            print(f"Error processing labels: {e}")
        except Exception as e:
            print(
                f"Unexpected error during label processing loop for label {label_id if 'label_id' in locals() else 'unknown'}: {e}")
            import traceback
            traceback.print_exc()

    # --- NEW (ROBUST): Generate Actors by mapping voxel paths to filtered graph nodes ---
    print("\n      Generating new visual actors from filtered network data (Voxel-First Method)...")

    filtered_skeleton_actors = []
    filtered_node_actors = {}
    filtered_actors_for_feeding_plot = []

    if 'processed_edge_list' in locals() and 'labeled_segments' in locals() and processed_edge_list:
        # 1. Get the definitive set of nodes that are part of the connected network
        connected_node_ids = set()
        for u_node, v_node in processed_edge_list:
            connected_node_ids.add(u_node)
            connected_node_ids.add(v_node)

        # 2. Build a KDTree of the final, connected nodes for fast geometric lookup
        valid_connected_node_ids = [nid for nid in connected_node_ids if nid in G_final_processed.nodes]
        if not valid_connected_node_ids:
                print("        Warning: No valid nodes found in the filtered list. Cannot generate plot actors.")
        else:
            connected_node_coords = np.array([G_final_processed.nodes[nid]['pos_zyx_image'] for nid in valid_connected_node_ids])
            node_id_list = list(valid_connected_node_ids)
            from scipy.spatial import KDTree
            node_kdtree_final = KDTree(connected_node_coords)

            # 3. Iterate through every single voxel segment identified earlier
            num_segments_total = np.max(labeled_segments)
            print(f"        Mapping {num_segments_total} original voxel segments to the filtered graph...")
            
            skeleton_points_coords = []
            feeding_points_coords = []
            non_feeding_points_coords = []

            for seg_id in tqdm(range(1, num_segments_total + 1), desc="      Voxel Segments", unit="segment", leave=False):
                # Retrieve the tight bounding box slice
                sl = segment_slices[seg_id - 1]
                if sl is None: continue

                # EXPAND THE SLICE (Padding)
                # Padding ensures the convolution for endpoint detection sees the full neighborhood
                padding = 3
                
                sl_padded = tuple(
                    slice(max(0, s.start - padding), min(max_dim, s.stop + padding))
                    for s, max_dim in zip(sl, vol_shape)
                )

                # Crop using the PADDED slice
                seg_crop = labeled_segments[sl_padded]
                current_seg_mask_local = (seg_crop == seg_id)
                
                # Get local voxels
                local_voxels = np.argwhere(current_seg_mask_local)
                if local_voxels.shape[0] < 2: continue
                
                # Calculate global offset based on PADDED slice start
                offset = np.array([s.start for s in sl_padded])
                segment_voxels_zyx = local_voxels + offset

                # Perform endpoint detection on the small local mask
                from scipy.ndimage import convolve
                kernel = np.ones((3,3,3))
                neighbor_counts_local = convolve(current_seg_mask_local.astype(np.uint8), kernel, mode='constant', cval=0) * current_seg_mask_local
                endpoint_mask_local = (neighbor_counts_local == 2)
                
                endpoints_local = np.argwhere(endpoint_mask_local)
                
                # Map local endpoints to global coordinates
                if endpoints_local.shape[0] != 2:
                    # Fallback to first and last point of the global list if convolution is ambiguous
                    endpoints_zyx = np.array([segment_voxels_zyx[0], segment_voxels_zyx[-1]])
                else:
                    endpoints_zyx = endpoints_local + offset

                if endpoints_zyx.shape[0] != 2: continue

                dist1, idx1 = node_kdtree_final.query(endpoints_zyx[0])
                dist2, idx2 = node_kdtree_final.query(endpoints_zyx[1])
                
                if dist1 < 15 and dist2 < 15:
                    node_u = node_id_list[idx1]
                    node_v = node_id_list[idx2]

                    if node_u in connected_node_ids and node_v in connected_node_ids and node_u != node_v:
                        s_pts_vedo = np.zeros_like(segment_voxels_zyx, dtype=float)
                        s_pts_vedo[:, 0] = segment_voxels_zyx[:, 2] * vedo_spacing[0]
                        s_pts_vedo[:, 1] = segment_voxels_zyx[:, 1] * vedo_spacing[1]
                        s_pts_vedo[:, 2] = segment_voxels_zyx[:, 0] * vedo_spacing[2]

                        skeleton_points_coords.append(s_pts_vedo)
                        
                        edge_data = G_final_processed.edges.get((node_u, node_v)) or G_final_processed.edges.get((node_v, node_u), {})
                        if edge_data.get('feeding_status') == 'feeding':
                            feeding_points_coords.append(s_pts_vedo)
                        else:
                            non_feeding_points_coords.append(s_pts_vedo)

            # 4. Generate the node actors
            node_render_configs = {
                'junction': (junction_node_color_str, junction_node_radius, "Junctions"),
                'endpoint': (endpoint_node_color_str, endpoint_node_radius, "Endpoints"),
                'inlet': (inlet_node_color_str, inlet_node_radius, "Inlets"),
                'outlet': (outlet_node_color_str, outlet_node_radius, "Outlets"),
                'inlet_outlet': ('purple', junction_node_radius, "Inlet_Outlets"),
                'intersection': (intersection_node_color_str, intersection_node_radius, "Intersection_Nodes")
            }
            coords_by_type_filtered = {key: [] for key in node_render_configs}
            for node_id in connected_node_ids:
                if node_id in G_final_processed.nodes:
                    node_data = G_final_processed.nodes[node_id]
                    node_type = node_data['type']
                    if node_type in coords_by_type_filtered:
                        coords_by_type_filtered[node_type].append(node_data['pos_xyz_world'])
            
            for n_type, (color, radius, name) in node_render_configs.items():
                if coords_by_type_filtered[n_type]:
                    points = np.array(coords_by_type_filtered[n_type])
                    node_actor = vedo.Points(points, r=radius, c=color)
                    node_actor.name = f"Filtered_{name}"
                    filtered_node_actors[n_type] = node_actor

            # 5. Create the final vedo.Points actors for segments
            if skeleton_points_coords:
                all_skeleton_points = np.vstack(skeleton_points_coords)
                skeleton_actor = vedo.Points(all_skeleton_points, r=centerline_point_radius, c=centerline_color_str)
                filtered_skeleton_actors.append(skeleton_actor)

            if feeding_points_coords:
                all_feeding_points = np.vstack(feeding_points_coords)
                feeding_actor = vedo.Points(all_feeding_points, r=centerline_point_radius, c='orange')
                filtered_actors_for_feeding_plot.append(feeding_actor)
                
            if non_feeding_points_coords:
                all_non_feeding_points = np.vstack(non_feeding_points_coords)
                non_feeding_actor = vedo.Points(all_non_feeding_points, r=centerline_point_radius, c='black')
                filtered_actors_for_feeding_plot.append(non_feeding_actor)

            # --- NEW: VISUAL SNAPPING OF INTERSECTION NODES ---
            # This block takes the floating intersection nodes and snaps them to the
            # nearest point on the actual skeleton for a clean visualization.
            
            snapped_intersection_actor = None
            if 'intersection' in filtered_node_actors and skeleton_volume is not None:
                print("         Snapping intersection nodes to skeleton for visual accuracy...")
                
                # 1. Get the entire skeleton as a point cloud in world coordinates
                skel_indices_zyx = np.argwhere(skeleton_volume)
                if skel_indices_zyx.size > 0:
                    spacing_zyx = np.array([vedo_spacing[2], vedo_spacing[1], vedo_spacing[0]])
                    skeleton_points_world_zyx = skel_indices_zyx * spacing_zyx
                    skeleton_points_world_xyz = skeleton_points_world_zyx[:, ::-1] # Reverse to XYZ
                    
                    # 2. Build a KDTree for ultra-fast nearest neighbor searching
                    skeleton_kdtree = KDTree(skeleton_points_world_xyz)
                    
                    # 3. Get the "floating" positions of the intersection nodes
                    floating_intersection_points = filtered_node_actors['intersection'].points
                    
                    # 4. Query the tree to find the closest skeleton point for each floating node
                    distances, closest_indices = skeleton_kdtree.query(floating_intersection_points)
                    
                    # 5. Get the coordinates of these closest points. These are the "snapped" positions.
                    snapped_positions = skeleton_points_world_xyz[closest_indices]
                    
                    # 6. Create a new vedo actor for the snapped nodes
                    snapped_intersection_actor = vedo.Points(snapped_positions, 
                                                            r=intersection_node_radius, 
                                                            c=intersection_node_color_str)
                    snapped_intersection_actor.name = "Snapped_Intersection_Nodes"

    # 6. Assemble the final actor lists for each plot, using the snapped nodes

    # Create a list of all node actors EXCEPT the original, floating intersection nodes
    nodes_to_plot = [actor for n_type, actor in filtered_node_actors.items() if n_type != 'intersection']

    # Add our new, snapped intersection node actor to the list
    if snapped_intersection_actor is not None:
        nodes_to_plot.append(snapped_intersection_actor)

    all_filtered_nodes = nodes_to_plot # This list now contains the snapped nodes

    filtered_skeleton_junction_actors = filtered_skeleton_actors + all_filtered_nodes
    filtered_actors_for_combined_plot = surface_only_actors + filtered_skeleton_junction_actors
    filtered_actors_for_feeding_plot += all_filtered_nodes
    # --- END of new actor generation ---

    # --- Plotting (using filtered data) ---
    # Note: Surface plots (like Plot 1 and 4) will show the COMPLETE original surface, 
    # as filtering a 3D mesh is non-trivial. The skeleton/nodes overlaid will be filtered.

    plot_pls = False
    if plot_pls:

        if 'filtered_skeleton_junction_actors' in locals():
            if labels_to_render_str.lower() != "all" and surface_only_actors:
                print("\nShowing Plot 1: Segmentation Surfaces Only...")
                plotter_segmentation_only = vedo.Plotter(axes=0, bg=bg_col, title="Segmentation Surfaces Only")
                plotter_segmentation_only.show(*surface_only_actors, interactive=interactive_mode, viewup='z').close()

            if labels_to_render_str.lower() != "all" and filtered_skeleton_junction_actors:
                print("\nShowing Plot 2: Filtered Skeletons and Nodes Only...")
                plotter_skel_nodes = vedo.Plotter(axes=plotter_combined_axes_type, bg=bg_col, title="Filtered Skeletons and Nodes Only")
                plotter_skel_nodes.show(*filtered_skeleton_junction_actors, interactive=interactive_mode, viewup='z').close()

            if filtered_actors_for_combined_plot:
                print("\nShowing Plot 3: Combined Segmentation with Filtered Network...")
                plotter_combined = vedo.Plotter(axes=plotter_combined_axes_type, bg=bg_col, title="Combined Segmentation with Filtered Network")
                plotter_combined.show(*filtered_actors_for_combined_plot, screenshot=output_image_path_or_none, interactive=interactive_mode, viewup='z').close()

            if glomus_actor is not None and surface_only_actors:
                print("\nShowing Plot 4: Superimposed Vessel Surfaces on Glomus Cells...")
                plotter_superimposed = vedo.Plotter(axes=plotter_combined_axes_type, bg=bg_col, title="Vessel Surfaces on Glomus Cells")
                plotter_superimposed.add(glomus_actor).add(surface_only_actors)
                plotter_superimposed.show(interactive=interactive_mode, viewup='z').close()

            if 'filtered_actors_for_feeding_plot' in locals() and filtered_actors_for_feeding_plot:
                print("\nShowing Plot 5: Filtered Vessel Feeding Status...")
                plotter_feeding = vedo.Plotter(axes=plotter_combined_axes_type, bg=bg_col, title="Filtered Vessel Feeding Status (Feeding=Orange)")
                plotter_feeding.add(filtered_actors_for_feeding_plot)
                plotter_feeding.show(interactive=interactive_mode, viewup='z').close()

            if discretize_with_grid and filtered_actors_for_combined_plot:
                print("\nShowing Plot 6: Discretization Grid with Filtered Network...")
                full_network_assembly = vedo.Assembly(filtered_actors_for_combined_plot) # Use original assembly for bounds
                bounds = full_network_assembly.bounds()
                grid_lines = []
                x_coords = np.linspace(bounds[0], bounds[1], grid_resolution_xyz[0] + 1)
                y_coords = np.linspace(bounds[2], bounds[3], grid_resolution_xyz[1] + 1)
                z_coords = np.linspace(bounds[4], bounds[5], grid_resolution_xyz[2] + 1)
                # (Grid line generation remains the same)
                for y in y_coords:
                    for z in z_coords: grid_lines.append(vedo.Line((bounds[0], y, z), (bounds[1], y, z)).c('gray').alpha(0.5))
                for x in x_coords:
                    for z in z_coords: grid_lines.append(vedo.Line((x, bounds[2], z), (x, bounds[3], z)).c('gray').alpha(0.5))
                for x in x_coords:
                    for y in y_coords: grid_lines.append(vedo.Line((x, y, bounds[4]), (x, y, bounds[5])).c('gray').alpha(0.5))
                
                grid_assembly = vedo.Assembly(grid_lines)
                plotter_grid = vedo.Plotter(axes=plotter_combined_axes_type, bg=bg_col, title="Discretization Grid with Filtered Network")
                plotter_grid.add(grid_assembly).add(filtered_actors_for_feeding_plot)
                plotter_grid.show(interactive=interactive_mode, viewup='z').close()

        else:
            print("\nWarning: Filtered actors for plotting were not generated. No plots will be shown.")

        # --- NEW: Plot 7 - Abstract Graph Representation (Simple Lines) ---
        if 'G_final_processed' in locals() and 'filtered_node_actors' in locals() and G_final_processed.number_of_edges() > 0:
            print("\nShowing Plot 7: Abstract Graph Network (Simple Lines)...")
            
            # 1. Create a list to hold the straight-line edge actors
            straight_line_edge_actors = []
            
            # 2. Iterate through every edge in the final, processed graph
            for u, v, data in G_final_processed.edges(data=True):
                # Get the 3D world coordinates of the start and end nodes for the edge
                p1 = G_final_processed.nodes[u]['pos_xyz_world']
                p2 = G_final_processed.nodes[v]['pos_xyz_world']
                
                # Create a Line actor with a fixed color and line width (lw)
                edge_actor = vedo.Line(p1, p2, c='silver', lw=2)
                straight_line_edge_actors.append(edge_actor)
                
            # 3. Collect all the previously generated node actors for display
            all_node_actors = list(filtered_node_actors.values())
            
            # 4. Set up the plotter and add all the actors
            plotter_abstract = vedo.Plotter(axes=plotter_combined_axes_type, bg=bg_col, title="Abstract Graph Network (Simple Lines)")
            plotter_abstract.add(straight_line_edge_actors)
            plotter_abstract.add(all_node_actors)
            
            # 5. Show the interactive plot
            plotter_abstract.show(interactive=interactive_mode, viewup='z').close()

    #########################################
    ### // Vessel Network Construction // ###
    #########################################

    C_vessel_filepath = output_dir / f'label_{label_id}_edge_adjacency_matrix.csv'
    C_vessel = np.genfromtxt(C_vessel_filepath, delimiter=',')

    network_info_filepath = output_dir / f'label_{label_id}_image_to_model_vessel_xyz.csv'
    network_info_df = pd.read_csv(network_info_filepath, engine='python')
    vessel_names = np.array(network_info_df['name'])

    # [NEW] Extract Centroids
    if {'x', 'y', 'z'}.issubset(network_info_df.columns):
        vessel_centroids = network_info_df[['x', 'y', 'z']].values
    else:
        vessel_centroids = None

    vessel_mod_types = np.array(['artery', 'artery', 'artery', 'artery', 'artery'])
    vessel_mods = np.array([random.choice(vessel_mod_types) for i in range(len(vessel_names))])

    vessel_network = VesselNetwork(C_vessel=C_vessel,
                                    vessel_names=vessel_names,
                                    vessel_mods=vessel_mods,
                                    vessel_centroids=vessel_centroids)

    vessel_network.generate_vessel_array()

    vessel_array_csv_filepath_resources = resources_path / 'image_to_model_vessel_array.csv'
    vessel_array_csv_filepath_user_output = output_dir / 'image_to_model_vessel_array.csv'
    vessel_network.vessel_df.to_csv(vessel_array_csv_filepath_resources, index=False)
    vessel_network.vessel_df.to_csv(vessel_array_csv_filepath_user_output, index=False)

    vessel_network.generate_parameter_array()
    vessel_network.populate_parameter_array()

    parameters_csv_abs_path_temp_resources = resources_path / 'image_to_model_parameters.csv'
    parameters_csv_abs_path_temp_user_output = output_dir / 'image_to_model_parameters.csv'
    vessel_network.parameter_df.to_csv(parameters_csv_abs_path_temp_resources, index=False, header=True)
    vessel_network.parameter_df.to_csv(parameters_csv_abs_path_temp_user_output, index=False, header=True)

    #####################################
    ### // Run Circulatory Autogen // ###
    #####################################

    if run_circ_autogen:

        script_path = Path.cwd().parent.parent / "src/scripts/script_generate_with_new_architecture.py"
        script_dir = os.path.dirname(script_path)

        print("Starting script...")

        # No capture_output=True here. 
        # The output will stream directly to your console.
        subprocess.run(
            [sys.executable, "-u", script_path, "False"],  # -u is important for real-time printing!
            cwd=script_dir
        )
