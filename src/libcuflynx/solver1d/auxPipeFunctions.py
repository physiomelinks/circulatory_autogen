#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 2 10:21:04 2025

@author: bghi639
"""

import math
import numpy as np
import os
import sys


def openPipes(pipePath, N1d0d, couple_volume_sum):

    write_pipe_dt = None
    read_pipe_dt = None
    write_pipe = None
    read_pipe = None
    write_pipe_vol = None
    
    try:
        # Create pipe if don't exist and open pipe with explicit buffering    
        # Open write pipe
        if not os.path.exists(pipePath+"one_to_parent_dt"):
            os.mkfifo(pipePath+"one_to_parent_dt")
        # Open write pipe
        # print("1d solver :: Opening time step write pipe...")
        write_pipe_dt = open(pipePath+"one_to_parent_dt", "wb", buffering=0)
        # print("1d solver :: Time step write pipe opened successfully")

        write_pipe = []
        for i in range(N1d0d):
            pipeID = str(i+1)
            if not os.path.exists(pipePath+"one_to_parent_"+pipeID):
                os.mkfifo(pipePath+"one_to_parent_"+pipeID) 
            # Open write pipe
            # print("1d solver :: Opening write pipe "+pipeID+"...")
            write_pipe.append( open(pipePath+"one_to_parent_"+pipeID, "wb", buffering=0) )
            # print("1d solver :: Write pipe "+pipeID+" opened successfully")
        
        if couple_volume_sum:
            # Open write pipe
            if not os.path.exists(pipePath+"one_to_parent_vol"):
                os.mkfifo(pipePath+"one_to_parent_vol")
            # Open write pipe
            write_pipe_vol = open(pipePath+"one_to_parent_vol", "wb", buffering=0)

        # Open read pipe
        if not os.path.exists(pipePath+"parent_to_one_dt"):
            os.mkfifo(pipePath+"parent_to_one_dt")
        # Open read pipe
        # print("1d solver :: Opening time step read pipe...")
        read_pipe_dt = open(pipePath+"parent_to_one_dt", "rb", buffering=0)
        # print("1d solver :: Time step read pipe opened successfully")

        read_pipe = []
        for i in range(N1d0d):
            pipeID = str(i+1)
            if not os.path.exists(pipePath+"parent_to_one_"+pipeID):
                os.mkfifo(pipePath+"parent_to_one_"+pipeID)  
            # Open read pipe
            # print("1d solver :: Opening read pipe "+pipeID+"...")
            read_pipe.append( open(pipePath+"parent_to_one_"+pipeID, "rb", buffering=0) )
            # print("1d solver :: Read pipe "+pipeID+" opened successfully")

        return write_pipe_dt, read_pipe_dt, write_pipe, read_pipe, write_pipe_vol
    

    except Exception as e:
        print(f"Failed to open pipes: {e}", file=sys.stderr)
        if write_pipe_dt:
            write_pipe_dt.close()
        if read_pipe_dt:
            read_pipe_dt.close()
        [pipe.close() for pipe in write_pipe if pipe]
        [pipe.close() for pipe in read_pipe if pipe]
        if write_pipe_vol:
            write_pipe_vol.close()
        
        return None, None, None, None, None

def closePipes(pp, nP):

    if nP==1:
        if pp:
            pp.close()
    else:
        [pipe.close() for pipe in pp if pipe]

    return


   