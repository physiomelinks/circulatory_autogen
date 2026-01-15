'''
Created on 29/10/2021

@author: Finbar J. Argus
'''

import sys
import os
from mpi4py import MPI
root_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir, 'src'))
from param_id.paramID import CVS0DParamID
import traceback
import yaml
import numpy as np
from parsers.PrimitiveParsers import YamlFileParser
from identifiabilty_analysis.identifiabilityAnalysis import IdentifiabilityAnalysis

def run_param_id(inp_data_dict=None):

    yaml_parser = YamlFileParser()
    inp_data_dict = yaml_parser.parse_user_inputs_file(inp_data_dict, obs_path_needed=True, do_generation_with_fit_parameters=False)

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
    # Get optimiser_options (parser already merged any legacy ga_options/debug_ga_options)
    optimiser_options = inp_data_dict.get('optimiser_options', None)
    resources_dir = inp_data_dict['resources_dir']
    param_id_output_dir = inp_data_dict['param_id_output_dir']
    


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()
    if rank == 0:
        if DEBUG:
            print('WARNING: DEBUG IS ON, TURN THIS OFF IF YOU WANT TO DO ANYTHING QUICKLY')
        print(f'Starting parameter identification with {num_procs} MPI rank(s)')

    param_id = CVS0DParamID(model_path, model_type, param_id_method, False, file_prefix,
                            params_for_id_path=params_for_id_path,
                            param_id_obs_path=param_id_obs_path,
                            sim_time=sim_time, pre_time=pre_time,
                            solver_info=solver_info, dt=dt,
                            optimiser_options=optimiser_options, DEBUG=DEBUG,
                            param_id_output_dir=param_id_output_dir, resources_dir=resources_dir)

    if rank == 0:
        if os.path.exists(os.path.join(param_id.output_dir, 'param_names_to_remove.csv')):
            os.remove(os.path.join(param_id.output_dir, 'param_names_to_remove.csv'))


    if param_id_method == 'bayesian':
        acq_func = 'PI'  # 'gp_hedge'
        n_initial_points = 5
        random_seed = 1234
        acq_func_kwargs = {'xi': 0.01, 'kappa': 0.1} # these parameters favour exploitation if they are low
                                                            # and exploration if high, see scikit-optimize docs.
                                                            # xi is used when acq_func is “EI” or “PI”,
                                                            # kappa is used when acq_func is "LCB"
                                                            # gp_hedge, chooses the best from "EI", "PI", and "LCB
                                                            # so it needs both xi and kappa
        # TODO this needs to be defined better if we want to keep bayesian optimiser functionality
        # Use optimiser_options (already merged with any legacy options by parser)
        num_calls_to_function = optimiser_options.get('num_calls_to_function')
        if num_calls_to_function is None:
            num_calls_to_function = 10000  # fallback default
        param_id.set_bayesian_parameters(num_calls_to_function, n_initial_points, acq_func,  random_seed,
                                            acq_func_kwargs=acq_func_kwargs)
    param_id.run()
    # param_id.param_id.set_best_param_vals(np.asarray([0.59779409, 0.32321317, 0.05664833, 0.35665839]))
    best_param_vals = param_id.get_best_param_vals()
    
    if rank == 0:
        print('param id complete with method:', param_id_method)

    # param_id.close_simulation() comment for identifiability analysis run otherwise the model will be closed before analysis
    do_mcmc = inp_data_dict['do_mcmc']

    if DEBUG:
        mcmc_options = inp_data_dict['debug_mcmc_options']
    else:
        mcmc_options = inp_data_dict['mcmc_options']

    if do_mcmc:

        if rank == 0:
            print('running mcmc')

        mcmc = CVS0DParamID(model_path, model_type, param_id_method, True, file_prefix,
                                params_for_id_path=params_for_id_path,
                                param_id_obs_path=param_id_obs_path,
                                sim_time=sim_time, pre_time=pre_time,
                                solver_info=solver_info, dt=dt, mcmc_options=mcmc_options, DEBUG=DEBUG,
                                param_id_output_dir=param_id_output_dir, resources_dir=resources_dir)
        mcmc.set_best_param_vals(best_param_vals)
        # mcmc.set_mcmc_parameters() TODO
        mcmc.run_mcmc()

        if rank == 0:
            print('mcmc complete')
    

    if inp_data_dict['do_ia']:
        # id_analysis = IdentifiabilityAnalysis(model_path, model_type, param_id_method, False, file_prefix,
        #                                      params_for_id_path=params_for_id_path,
        #                                      param_id_obs_path=param_id_obs_path,
        #                                      sim_time=sim_time, pre_time=pre_time,
        #                                      solver_info=solver_info, dt=dt, DEBUG=DEBUG,
        #                                      param_id_output_dir=param_id_output_dir, resources_dir=resources_dir,
        #                                      param_id=param_id.param_id) # pass in param_id object so we can use its cost functions
        id_analysis = IdentifiabilityAnalysis(model_path, model_type, file_prefix, param_id_output_dir=param_id_output_dir,
                                            resources_dir=resources_dir, param_id=param_id.param_id)  # pass in param_id object so we can use its cost functions

        id_analysis.set_best_param_vals(best_param_vals)    
        if rank == 0:
            print('Running identifiability analysis with method:', inp_data_dict['ia_options']['method'])
        #id_analysis.run_identifiability_analysis(inp_data_dict['identifiability_analysis_options'])
        id_analysis.run(inp_data_dict['ia_options'])

        if rank == 0:
            print('Identifiability analysis complete')
        

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    try:
        run_param_id()
        MPI.Finalize()
    except:
        print(traceback.format_exc())
        comm.Abort()
        MPI.Finalize()
        exit()
