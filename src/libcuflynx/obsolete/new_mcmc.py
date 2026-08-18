'''
Created on 15/05/2022

@author: Finbar J. Argus
'''

import os


import numpy as np
from libcuflynx.param_id.paramID import CVS0DParamID
from mpi4py import MPI
from argparse import ArgumentParser
import csv

def run_opencor_mcmc(*args, **kwargs):

    model_path, model_type, param_id_method, file_name_prefix = args

    input_params_path=kwargs['input_params_path']
    sensitivity_params_path=kwargs['sensitivity_params_path']
    num_calls_to_function=kwargs['num_calls_to_function']
    param_id_obs_path=kwargs['param_id_obs_path'] 
    sim_time=kwargs['sim_time']
    pre_time=kwargs['pre_time']
    maximumStep=kwargs['maximumStep']
    dt=kwargs['dt']
    load_params=kwargs['load_params']
    DEBUG=kwargs['DEBUG']
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()

    mcmc = CVS0DParamID(model_path, model_type, param_id_method, True,
                             file_name_prefix,
                             input_params_path=input_params_path,
                             sensitivity_params_path=sensitivity_params_path,
                             param_id_obs_path=param_id_obs_path,
                             sim_time=sim_time, pre_time=pre_time, maximumStep=maximumStep,
                             DEBUG=DEBUG)

    if load_params:
        if rank == 0:
            output_dir = mcmc.output_dir
            param_names = []
            with open(os.path.join(output_dir, 'param_names.csv'), 'r') as rf:
                rd = csv.reader(rf)
                for row in rd:
                    param_names.append(row)
                
            param_names = np.array(param_names, dtype='S32')
            bytes_array = param_names.tobytes()
            param_vals = np.load(os.path.join(output_dir, 'best_param_vals.npy'))
            num_params_buf = np.array([len(param_vals), len(bytes_array)])
        else:
            num_params_buf = np.empty(2, dtype=int)

        comm.Bcast(num_params_buf, root=0)
        num_params = num_params_buf[0]
        num_bytes = num_params_buf[1]
        if rank != 0 :
            param_vals = np.empty(num_params, dtype=float)
            bytes_array = bytearray(num_bytes)

        comm.Bcast(param_vals, root=0)
        comm.Bcast(bytes_array, root=0)
        param_names = np.frombuffer(bytes_array, dtype='S32', count=num_params).astype(str)

    mcmc.set_best_param_vals(param_vals)
    mcmc.set_param_names(param_names)
    mcmc.run_mcmc()
    mcmc.plot_mcmc()
    
    mcmc.calculate_mcmc_identifiability()

def parse_input():
    ap = ArgumentParser()
    ap.add_argument('model_path', type=str, default=None)
    ap.add_argument('model_type', type=str, default=None)
    ap.add_argument('param_id_method', type=str, default=None)
    ap.add_argument('file_name_prefix', type=str, default=None)
    ap.add_argument('--input_params_path', type=str, default=None)
    ap.add_argument('--sensitivity_params_path', type=str, default=None)
    ap.add_argument('--param_id_obs_path', type=str, default=None)
    ap.add_argument('--num_calls_to_function', type=int, default=None)
    ap.add_argument('--sim_time', type=float, default=2.0)
    ap.add_argument('--pre_time', type=float, default=20.0)
    ap.add_argument('--maximumStep', type=float, default=0.00003)
    ap.add_argument('--dt', type=float, default=0.01)
    ap.add_argument('--load_params', type=bool, default=False)
    ap.add_argument('--DEBUG', type=bool, default=False)
    
    return ap.parse_args()

if __name__ == '__main__':
    # use argparser to process sys.args into args list and kwargs dict
    # import pdb; pdb.set_trace()

    ns = parse_input()

    kwargs = vars(ns)
    args = [kwargs.pop(a) for a in ['model_path', 'model_type', 
                       'param_id_method', 'file_name_prefix']] 

    run_opencor_mcmc(*args, **kwargs) 
