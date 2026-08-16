'''
Created on 26/04/2022

@author: Finbar J. Argus
'''

import sys
import os
from distutils import util, dir_util
import traceback
import subprocess as sp

root_dir_path = os.path.join(os.path.dirname(__file__), '../../..')

resources_dir_path = os.path.join(root_dir_path, 'resources')
param_id_dir_path = os.path.join(root_dir_path, 'src/libcuflynx/param_id')
generated_models_dir_path = os.path.join(root_dir_path, 'generated_models')

if __name__ == '__main__':

    try:
        DEBUG = True

        print('starting script')

        opencor_python_path = sys.argv[1]
        num_procs = int(sys.argv[2])

        if len(sys.argv) != 7:
            print('incorrect number of inputs to sequential_param_id_run_script.py')
            exit()

        param_id_method = sys.argv[3]
        file_name_prefix = sys.argv[4]
        model_path = os.path.join(generated_models_dir_path, f'{file_name_prefix}.cellml')
        model_type = 'CVS0D'  # TODO make this an input variable eventually

        input_params_path = os.path.join(resources_dir_path, f'{file_name_prefix}_params_for_id.csv')
        if not os.path.exists(input_params_path):
            print(f'input_params_path of {input_params_path} doesn\'t exist, user must create this file')
            exit()
        sensitivity_params_path = os.path.join(resources_dir_path, f'{file_name_prefix}_params_for_sensitivity.csv')
        if not os.path.exists(sensitivity_params_path):
            sensitivity_params_path = input_params_path

        param_id_obs_path = sys.argv[6]
        if not os.path.exists(param_id_obs_path):
            print(f'param_id_obs_path={param_id_obs_path} does not exist')
            exit()

        # set the simulation time where the cost is calculated (sim_time) and
        # the amount of simulation time it takes to get to an oscilating steady
        # state before that (pre_time)
        if file_name_prefix == '3compartment' or 'FTU_wCVS':
            pre_time = 20.0
        else:
            pre_time = 20.0
        sim_time = 2.0

        num_calls_to_function = int(sys.argv[5])
        
        maximumStep= 0.00003
        run_param_id_script_path = os.path.join(param_id_dir_path, 'new_sequential_param_id.py')
        run_mcmc_script_path = os.path.join(param_id_dir_path, 'new_mcmc.py')

        python_commands = ['mpiexec', '-n', str(num_procs), opencor_python_path]
            
        # run_param_id_script_path
        args = [model_path, model_type, 
                     param_id_method, file_name_prefix]
        if input_params_path:
            args.extend(['--input_params_path', input_params_path])
        if sensitivity_params_path:
            args.extend(['--sensitivity_params_path', sensitivity_params_path])
        if param_id_obs_path:
            args.extend(['--param_id_obs_path', param_id_obs_path])
        if num_calls_to_function:
            args.extend(['--num_calls_to_function', str(num_calls_to_function)])
        if sim_time:
            args.extend(['--sim_time', str(sim_time)])
        if pre_time:
            args.extend(['--pre_time', str(pre_time)])
        if maximumStep:
            args.extend(['--maximumStep', str(maximumStep)])
        args.extend(['--DEBUG', str(DEBUG)])
        # repeat for each keyword argument
        # repeat for each keyword argument
        run_items = python_commands + [run_param_id_script_path] + args

        # run param_id and sensitivity
        # p = sp.Popen(run_items)#, stdout=sp.PIPE, stderr=sp.PIPE)
        # p.wait()

        # Add new arguments for mcmc
        args.extend(['--load_params', 'False'])# ensures the best found params from
                                            # above are loaded
        run_items = python_commands + [run_mcmc_script_path] + args

        # now run mcmc for practical identifiability
        import faulthandler; faulthandler.enable()
        p = sp.Popen(run_items)#, stdout=sp.PIPE, stderr=sp.PIPE)
        p.wait()

        print('All Finished')
        

    except:
        print(traceback.format_exc())
        print("Usage: parameter_id_method file_name_prefix num_calls_to_function "
              "path_to_obs_file.json")
        print("e.g. genetic_algorithm simple_physiological 10 "
              "path/to/circulatory_autogen/resources/simple_physiological_obs_data.json")
        exit
