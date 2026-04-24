'''
Created on 29/10/2021

@author: Finbar J. Argus
'''

import sys
import os
from mpi4py import MPI
from distutils import util
import re

root_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir, 'src'))

user_inputs_dir = os.path.join(root_dir, 'user_run_files')

from param_id.paramID import CVS0DParamID, MCMC_plotter
from identifiabilty_analysis.identifiabilityAnalysis import IdentifiabilityAnalysis
from scripts.script_generate_with_new_architecture import generate_with_new_architecture
from utilities.utility_funcs import obj_to_string, change_parameter_values_and_save
import traceback
from distutils import util
import yaml
from parsers.PrimitiveParsers import YamlFileParser
import numpy as np


def plot_param_id(inp_data_dict=None, generate=True):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()
    
    if rank != 0:
        return

    yaml_parser = YamlFileParser()
    inp_data_dict = yaml_parser.parse_user_inputs_file(inp_data_dict, obs_path_needed=True, do_generation_with_fit_parameters=True)

    DEBUG = inp_data_dict['DEBUG']
    model_path = inp_data_dict['model_path']
    uncalibrated_model_path = inp_data_dict['uncalibrated_model_path']
    model_type = inp_data_dict['model_type']
    param_id_method = inp_data_dict['param_id_method']
    file_prefix = inp_data_dict['file_prefix']
    params_for_id_path = inp_data_dict['params_for_id_path']
    param_id_obs_path = inp_data_dict['param_id_obs_path']
    sim_time = inp_data_dict['sim_time']
    pre_time = inp_data_dict['pre_time']
    solver_info = inp_data_dict['solver_info']
    dt = inp_data_dict['dt']
    optimiser_options = inp_data_dict['optimiser_options']
    mcmc_options = inp_data_dict['mcmc_options']
    resources_dir = inp_data_dict['resources_dir']
    param_id_output_dir = inp_data_dict['param_id_output_dir']
    param_id_output_dir_abs_path = inp_data_dict['param_id_output_dir_abs_path']
    plot_predictions = inp_data_dict['plot_predictions']
    do_sensitivity = inp_data_dict['do_sensitivity']
    do_mcmc = inp_data_dict['do_mcmc']
    generated_models_subdir = inp_data_dict['generated_models_subdir']
    do_identify = inp_data_dict['do_ia']
    ia_option = inp_data_dict['ia_options']
    
    # run the generation script with new param values
    if generate:
        generate_with_new_architecture(True, inp_data_dict=inp_data_dict)
    else:
        # if we arent doing the autogeneration then we need to create a model with the best fit parameters anyway
        # load the best parameter values from the param_id_output_dir
       
        best_param_vals = np.load(os.path.join(param_id_output_dir_abs_path, 'best_param_vals.npy'))
        param_names = np.loadtxt(os.path.join(param_id_output_dir_abs_path, 'param_names.csv'), dtype=str)
        
        # change the parameter values in the model file
        change_parameter_values_and_save(uncalibrated_model_path, param_names, best_param_vals, 
                                         model_path)
        # set the model path to the new model file
        print('Best fit model saved to: ' + model_path)
        
        # set the model_path to the uncalibrated model path for now so we can run it. 
        # this will be changed when we can run the new cellml2.0 model from python.
        model_path = uncalibrated_model_path


    
    param_id = CVS0DParamID(model_path, model_type, param_id_method, False, file_prefix,
                            params_for_id_path=params_for_id_path,
                            param_id_obs_path=param_id_obs_path,
                            sim_time=sim_time, pre_time=pre_time,
                            solver_info=solver_info, optimiser_options=optimiser_options, dt=dt,
                            param_id_output_dir=param_id_output_dir, resources_dir=resources_dir, one_rank=True)

    if os.path.exists(os.path.join(param_id.output_dir, 'param_names_to_remove.csv')):
        with open(os.path.join(param_id.output_dir, 'param_names_to_remove.csv'), 'r') as r:
            param_names_to_remove = []
            for row in r:
                name_list = row.split(',')
                name_list = [name.strip() for name in name_list]
                param_names_to_remove.append(name_list)
        param_id.remove_params_by_name(param_names_to_remove)
        

    # simulate with best values first to check cost
    param_id.simulate_with_best_param_vals()
    param_id.plot_outputs()
    if do_mcmc:
        if os.path.exists(os.path.join(param_id.output_dir, 'mcmc_chain.npy')):
            if not plot_predictions:
                param_id.plot_mcmc()
        else:
            print('no mcmc chain has been created at ' + 
                    os.path.join(param_id.output_dir, 'mcmc_chain.npy'))
    param_id.save_prediction_data()
    if do_sensitivity:
        param_id.run_sensitivity(None)
    # param_id.close_simulation() # remove this for identifiability analysis

    print(f'doing ia for rank {rank}')
    
    if do_identify:
        id_analysis = IdentifiabilityAnalysis(model_path, model_type, file_prefix, param_id_output_dir=param_id_output_dir,
                                              resources_dir=resources_dir, param_id=param_id.param_id)  # pass in param_id object so we can use its cost functions
        id_analysis.set_best_param_vals(param_id.get_best_param_vals())       
        # id_analysis.run_identifiability_analysis(ia_option) # this should already be done
        label_list =[f'${param_id.param_id_info["param_names_for_plotting"][II]}$' for II in range(len(param_id.param_id_info["param_names_for_plotting"]))]
        print(label_list)
        id_analysis.plot_laplace_results(label_list, param_id.plot_dir)
        print('Identifiability analysis plotting complete')
    
    param_id.close_simulation()


    if plot_predictions:
        mcmc_plotter = MCMC_plotter(model_path, model_type, param_id_method, file_prefix,
                                            params_for_id_path=params_for_id_path,
                                            param_id_obs_path=param_id_obs_path,
                                            num_calls_to_function=1,
                                            sim_time=sim_time, pre_time=pre_time,
                                            solver_info=solver_info, dt=dt, 
                                            mcmc_options=mcmc_options, DEBUG=DEBUG,
                                            param_id_output_dir=param_id_output_dir, resources_dir=resources_dir)
    
        if do_mcmc:
            mcmc_plotter.plot_mcmc_and_predictions()
    
    return

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    # get argument of whether to run the autogeneration, defaults to True
    if len(sys.argv) == 2:
        generate = util.strtobool(sys.argv[1])
    else:
        generate = True
    try:
        plot_param_id(generate=generate)
    except:
        print(traceback.format_exc())
        comm.Abort()
        exit()