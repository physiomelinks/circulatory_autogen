'''
Created on 29/10/2021

@author: Finbar J. Argus
'''

import sys
import os
from mpi4py import MPI
from distutils import util, dir_util

root_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir, 'src'))

from param_id.paramID import CVS0DParamID
from scripts.read_and_insert_parameters import insert_parameters
from scripts.script_generate_with_new_architecture import generate_with_new_architecture
import traceback
import yaml

if __name__ == '__main__':

    ## TODO This script should be updated to use the user_inputs parser and tested

    try:
        user_inputs_dir= os.path.join(root_dir, 'user_run_files')

        with open(os.path.join(user_inputs_dir, 'user_inputs.yaml'), 'r') as file:
            inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)
        if inp_data_dict["user_inputs_path_override"]:
            if os.path.exists(inp_data_dict["user_inputs_path_override"]):
                user_files_dir = os.path.dirname(inp_data_dict["user_inputs_path_override"])
                with open(inp_data_dict["user_inputs_path_override"], 'r') as file:
                    inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)
            else:
                print(f"User inputs file not found at {inp_data_dict['user_inputs_path_override']}")
                print("Check the user_inputs_path_override key in user_inputs.yaml and set it to False if "
                        "you want to use the default user_inputs.yaml location")
                exit()
        else:
            user_files_dir = ''

        DEBUG = inp_data_dict['DEBUG']


        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        num_procs = comm.Get_size()
        if rank == 0:
            if DEBUG:
                print('WARNING: DEBUG IS ON, TURN THIS OFF IF YOU WANT TO DO ANYTHING QUICKLY')
            print(f'Starting multi-run parameter ID with {num_procs} MPI rank(s)')


        resources_dir= os.path.join(root_dir, 'resources')
        param_id_dir = os.path.join(root_dir, 'src/param_id')
        generated_models_dir = os.path.join(root_dir, 'generated_models')

        param_id_method = inp_data_dict['param_id_method']
        file_prefix = inp_data_dict['file_prefix']

        # overwrite dir paths if set in user_inputs.yaml
        if "resources_dir" in inp_data_dict.keys():
            resources_dir = os.path.join(user_files_dir, inp_data_dict['resources_dir'])
        if "param_id_output_dir" in inp_data_dict.keys():
            param_id_output_dir = os.path.join(user_files_dir, inp_data_dict['param_id_output_dir'])
        if "generated_models_dir" in inp_data_dict.keys():
            generated_models_dir = os.path.join(user_files_dir, inp_data_dict['generated_models_dir'])

        generated_models_subdir = os.path.join(generated_models_dir, file_prefix)
        model_path = os.path.join(generated_models_subdir, f'{file_prefix}.cellml')

        model_type = inp_data_dict['model_type']

        input_params_path = os.path.join(resources_dir, f'{file_prefix}_params_for_id.csv')
        if not os.path.exists(input_params_path):
            print(f'input_params_path of {input_params_path} doesn\'t exist, user must create this file')
            exit()
        
        if 'pre_time' in inp_data_dict.keys():
            pre_time = inp_data_dict['pre_time']
        else:
            pre_time = None
        if 'sim_time' in inp_data_dict.keys():
            sim_time = inp_data_dict['sim_time']
        else:
            sim_time = None

        solver_info = inp_data_dict['solver_info']
        dt = inp_data_dict['dt']
        # Prefer optimiser_options; fall back to legacy ga_options for compatibility
        optimiser_options = inp_data_dict.get('optimiser_options', None)
        if optimiser_options is None:
            optimiser_options = inp_data_dict.get('ga_options', {})

        obs_dir = '/eresearch/heart/farg967/Sandboxes/Finbar/combined'
        # obs_dir = '/home/farg967/Documents/data/heart_projects/Argus_2022'
        # obs_files = ['lv_estimation_observables_BB044.json',
        #         'lv_estimation_observables_BB047.json',
        #         'lv_estimation_observables_BB055.json',
        #         'lv_estimation_observables_BB104.json',
        #         'lv_estimation_observables_BB105.json',
        #         'lv_estimation_observables_BB133.json']
        obs_files = ['lv_estimation_observables_BB044_wout_lv.json',
                'lv_estimation_observables_BB047_wout_lv.json',
                'lv_estimation_observables_BB055_wout_lv.json',
                'lv_estimation_observables_BB104_wout_lv.json',
                'lv_estimation_observables_BB105_wout_lv.json',
                'lv_estimation_observables_BB133_wout_lv.json']
        const_files = ['lv_estimation_constants_BB044.json',
                'lv_estimation_constants_BB047.json',
                'lv_estimation_constants_BB055.json',
                'lv_estimation_constants_BB104.json',
                'lv_estimation_constants_BB105.json',
                'lv_estimation_constants_BB133.json']
        param_id_obs_path_list = [os.path.join(obs_dir, f) for f in obs_files]
        param_id_consts_path_list = [os.path.join(obs_dir, f) for f in const_files]
        for param_id_obs_path, consts_path in zip(param_id_obs_path_list, 
                                                  param_id_consts_path_list):
            if not os.path.exists(param_id_obs_path):
                print(f'param_id_obs_path={param_id_obs_path} does not exist')
                exit()
            if not os.path.exists(consts_path):
                print(f'consts_path={consts_path} does not exist')
                exit()

            parameters_csv_abs_path = os.path.join(resources_dir, inp_data_dict['input_param_file'])

            inp_data_dict['param_id_obs_path'] = param_id_obs_path
            if rank == 0:
                insert_parameters(parameters_csv_abs_path, consts_path)
                generate_with_new_architecture(False, inp_data_dict)
            comm.Barrier()


            param_id = CVS0DParamID(model_path, model_type, param_id_method, False, file_prefix,
                                    input_params_path=input_params_path,
                                    param_id_obs_path=param_id_obs_path,
                                    sim_time=sim_time, pre_time=pre_time,
                                    solver_info=solver_info, dt=dt, optimiser_options=optimiser_options, DEBUG=DEBUG,
                                    param_id_output_dir=param_id_output_dir, resources_dir=resources_dir)

            if rank == 0:
                if os.path.exists(os.path.join(param_id.output_dir, 'param_names_to_remove.csv')):
                    os.remove(os.path.join(param_id.output_dir, 'param_names_to_remove.csv'))

            num_calls_to_function = optimiser_options.get('num_calls_to_function', 10000)

            if param_id_method == 'genetic_algorithm':
                param_id.set_genetic_algorithm_parameters(num_calls_to_function)

            param_id.run()
            if rank == 0:
                param_id.simulate_with_best_param_vals()
                param_id.plot_outputs()
            
            comm.Barrier()

            # best_param_vals = param_id.get_best_param_vals()
            param_id.close_simulation()
            # do_mcmc = inp_data_dict['do_mcmc']
            #
            # if DEBUG:
            #     mcmc_options = inp_data_dict['debug_mcmc_options']
            # else:
            #     mcmc_options = inp_data_dict['mcmc_options']
            #
            # if do_mcmc:
            #     mcmc = CVS0DParamID(model_path, model_type, param_id_method, True, file_prefix,
            #                             input_params_path=input_params_path,
            #                             param_id_obs_path=param_id_obs_path,
            #                             sim_time=sim_time, pre_time=pre_time,
            #                             solver_info=solver_info, dt=dt, mcmc_options=mcmc_options, DEBUG=DEBUG)
            #     mcmc.set_best_param_vals(best_param_vals)
            #     # mcmc.set_mcmc_parameters() TODO
            #     mcmc.run_mcmc()
            

    except:
        print(traceback.format_exc())
        comm.Abort()
        exit
