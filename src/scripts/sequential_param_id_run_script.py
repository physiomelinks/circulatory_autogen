'''
Created on 26/04/2022

@author: Finbar J. Argus
'''

import sys
import os
from mpi4py import MPI
import time
import numpy as np
from distutils import util, dir_util

root_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir, 'src'))

user_inputs_dir = os.path.join(root_dir, 'user_run_files')

from param_id.sequential_paramID import SequentialParamID
import traceback
import yaml
from parsers.PrimitiveParsers import YamlFileParser

if __name__ == '__main__':

    # TODO This needs to be tested for the updated user_inputs parser

    try:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        num_procs = comm.Get_size()
        if rank == 0:
            print(f'Starting sequential parameter ID with {num_procs} MPI rank(s)')
        
        start_time = time.time()
        yaml_parser = YamlFileParser()
        inp_data_dict = yaml_parser.parse_user_inputs_file(None, obs_path_needed=True, do_generation_with_fit_parameters=True)

        DEBUG = inp_data_dict['DEBUG']
        model_path = inp_data_dict['model_path']
        model_type = inp_data_dict['model_type']
        param_id_method = inp_data_dict['param_id_method']
        file_prefix = inp_data_dict['file_prefix']
        params_for_id_path = inp_data_dict['params_for_id_path']
        param_id_obs_path = inp_data_dict['param_id_obs_path']
        sim_time = inp_data_dict['sim_time']
        pre_time = inp_data_dict['pre_time']
        solver_info = inp_data_dict['solver_info']
        dt = inp_data_dict['dt']
        ga_options = inp_data_dict['ga_options']
        resources_dir = inp_data_dict['resources_dir']
        param_id_output_dir = inp_data_dict['param_id_output_dir']
        plot_predictions = inp_data_dict['plot_predictions']
        do_sensitivity = inp_data_dict['do_sensitivity']
        do_mcmc = inp_data_dict['do_mcmc']
        input_params_path = inp_data_dict['input_params_path']
        num_calls_to_function = inp_data_dict['num_calls_to_function']
        mcmc_options = inp_data_dict['mcmc_options']


        seq_param_id = SequentialParamID(model_path, model_type, param_id_method, file_prefix,
                                input_params_path=input_params_path,
                                param_id_obs_path=param_id_obs_path,
                                num_calls_to_function=num_calls_to_function,
                                solver_info=solver_info, dt=dt, mcmc_options=mcmc_options, ga_options=ga_options,
                                DEBUG=DEBUG, 
                                param_id_output_dir=param_id_output_dir, resources_dir=resources_dir)

        seq_param_id.run()

        best_param_vals = seq_param_id.param_id.get_best_param_vals()
        best_param_names = seq_param_id.get_best_param_names()

        if rank == 0:
            wall_time = time.time() - start_time
            print(f'wall time = {wall_time}')
            np.save(os.path.join(seq_param_id.param_id.output_dir, 'wall_time.npy'), wall_time)

    except:
        print(traceback.format_exc())
        comm.Abort()
        exit
