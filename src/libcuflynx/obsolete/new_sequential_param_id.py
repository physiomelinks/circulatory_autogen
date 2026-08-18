'''
Created on 15/05/2022

@author: Finbar J. Argus
'''

import os


import numpy as np
from libcuflynx.param_id.paramID import CVS0DParamID
from mpi4py import MPI
from argparse import ArgumentParser

def run_opencor_param_id_and_sensitivity(*args, **kwargs):
    
    model_path, model_type, param_id_method, file_name_prefix = args

    input_params_path=kwargs['input_params_path']
    sensitivity_params_path=kwargs['sensitivity_params_path']
    num_calls_to_function=kwargs['num_calls_to_function']
    param_id_obs_path=kwargs['param_id_obs_path'] 
    sim_time=kwargs['sim_time']
    pre_time=kwargs['pre_time']
    maximumStep=kwargs['maximumStep']
    dt=kwargs['dt']
    DEBUG=kwargs['DEBUG']

    param_id = CVS0DParamID(model_path, model_type, param_id_method, 
                            False, file_name_prefix,
                            input_params_path=input_params_path,
                            sensitivity_params_path=sensitivity_params_path,
                            param_id_obs_path=param_id_obs_path,
                            sim_time=sim_time, pre_time=pre_time, 
                            maximumStep=maximumStep, DEBUG=DEBUG)


    param_id.set_genetic_algorithm_parameters(num_calls_to_function)
    best_param_vals = None
    best_param_names = None

    # thresholds for identifiability TODO optimise these
    threshold_param_importance = 0.01
    keep_threshold_param_importance = 0.8
    threshold_collinearity = 20
    threshold_collinearity_pairs = 10

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()

    buff = np.array([False])
    identifiable = buff[0]
    while not identifiable:

        # param_id.temp_test()
        # param_id.temp_test2()

        param_id.run()
        if rank == 0:
            param_id.run_single_sensitivity(None)

            best_param_vals = param_id.get_best_param_vals().copy()
            param_names = param_id.get_param_names().copy()

            param_importance = param_id.get_param_importance().copy()
            collinearity_index = param_id.get_collinearity_index().copy()
            collinearity_index_pairs = param_id.get_collinearity_index_pairs().copy()

            if min(param_importance) > threshold_param_importance and \
                        max(collinearity_index) < threshold_collinearity:
                print(f'The model is identifiable with {len(param_names)} parameters:')
                print(param_names)
                identifiable = True
            else:
                # remove parameters that aren't identifiable
                # and update param_id object
                print(f'The model is NOT identifiable with {len(param_names)} parameters')
                print(f'determining which parameters to remove from identifying set')
                param_idxs_to_remove = []
                for II in range(len(param_names)):
                    param_name = param_names[II]
                    if param_importance[II] < threshold_param_importance:
                        param_idxs_to_remove.append(II)

                    for JJ in range(len(param_names)):
                        if collinearity_index_pairs[II, JJ] > threshold_collinearity_pairs:
                            if param_importance[II] < param_importance[JJ]:
                                param_idxs_to_remove.append(II)

                if len(param_idxs_to_remove) > 1:
                    # make sure we aren't removing important parameters
                    # it is better to remove too few than too many
                    for idx in param_idxs_to_remove:
                        # TODO this doesn't allow us to remove linearly related params if they are both important
                        #  Fix this!
                        if param_importance[idx] > keep_threshold_param_importance:
                            param_idxs_to_remove.remove(idx)

                # TODO future work: if we are reformulating the equations we will need to create and run a
                #  CVS0DCellMLGenerator object.
        # TODO remove this break
        break
        buff[0] = identifiable
        comm.Bcast(buff, root=0)
        identifiable = buff[0]

    best_param_vals = param_id.get_best_param_vals()
    # TODO save best param vals 

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
    ap.add_argument('--DEBUG', type=bool, default=False)
    
    return ap.parse_args()

if __name__ == '__main__':
    # use argparser to process sys.args into args list and kwargs dict
    # import pdb; pdb.set_trace()

    ns = parse_input()

    kwargs = vars(ns)
    args = [kwargs.pop(a) for a in ['model_path', 'model_type', 
                       'param_id_method', 'file_name_prefix']] 

    run_opencor_param_id_and_sensitivity(*args, **kwargs) 

