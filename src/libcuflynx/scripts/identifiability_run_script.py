'''
Created on 29/10/2021

@author: Finbar J. Argus
'''

import sys
import os
# Not `from mpi4py import MPI`: that import initialises MPI and registers an
# atexit MPI_Finalize, and with no launcher present that finalise is what aborts
# on macOS when a NIC goes away (#396). Under mpiexec get_MPI hands back the real
# mpi4py.MPI, so a multi-rank run is unchanged.
from libcuflynx.utilities.mpi_utils import get_MPI as _get_MPI

MPI = _get_MPI()
from libcuflynx.param_id.paramID import CVS0DParamID, ensure_mle_cost_type_for_bayesian_inner
import yaml
import numpy as np
from libcuflynx.parsers.PrimitiveParsers import YamlFileParser
from libcuflynx.identifiabilty_analysis.identifiabilityAnalysis import IdentifiabilityAnalysis
from libcuflynx.parsers.PrimitiveParsers import CSVFileParser, JSONFileParser
from libcuflynx.scripts import _cli

def run_identifiability_analysis(inp_data_dict=None):

    yaml_parser = YamlFileParser()
    inp_data_dict = yaml_parser.parse_user_inputs_file(inp_data_dict, obs_path_needed=True, do_generation_with_fit_parameters=True)

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
    optimiser_options = inp_data_dict['optimiser_options']
    resources_dir = inp_data_dict['resources_dir']
    param_id_output_dir = inp_data_dict['param_id_output_dir']
    


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()
    if rank == 0:
        if DEBUG:
            print('WARNING: DEBUG IS ON, TURN THIS OFF IF YOU WANT TO DO ANYTHING QUICKLY')
        print(f'Starting identifiability analysis with {num_procs} MPI rank(s)')

    param_id = CVS0DParamID(model_path, model_type, param_id_method, False, file_prefix,
                            params_for_id_path=params_for_id_path,
                            param_id_obs_path=param_id_obs_path,
                            sim_time=sim_time, pre_time=pre_time,
                            solver_info=solver_info, dt=dt, optimiser_options=optimiser_options, DEBUG=DEBUG,
                            param_id_output_dir=param_id_output_dir, resources_dir=resources_dir)


    # id_analysis = IdentifiabilityAnalysis(model_path, model_type, param_id_method, False, file_prefix,
    #                                      params_for_id_path=params_for_id_path,
    #                                      param_id_obs_path=param_id_obs_path,
    #                                      sim_time=sim_time, pre_time=pre_time,
    #                                      solver_info=solver_info, dt=dt, DEBUG=DEBUG,
    #                                      param_id_output_dir=param_id_output_dir, resources_dir=resources_dir,
    #                                      param_id=param_id.param_id) # pass in param_id object so we can use its cost functions
    id_analysis = IdentifiabilityAnalysis(model_path, model_type, file_prefix, param_id_output_dir=param_id_output_dir,
                                            resources_dir=resources_dir, param_id=param_id.param_id)  # pass in param_id object so we can use its cost functions

    csv_parser = CSVFileParser()
    param_id_name_and_vals, param_id_date = csv_parser.get_param_id_params_as_lists_of_tuples(inp_data_dict['param_id_output_dir_abs_path'])
    best_param_vals = np.array([val for name, val in param_id_name_and_vals])
    
    id_analysis.set_best_param_vals(best_param_vals)
    if inp_data_dict.get("ia_options", {}).get("method") == "Laplace":
        ensure_mle_cost_type_for_bayesian_inner(param_id.param_id, inp_data_dict)
    #id_analysis.run_identifiability_analysis(inp_data_dict['identifiability_analysis_options'])
    id_analysis.run(inp_data_dict['ia_options'])
    
    if rank == 0:
        print('Identifiability analysis complete')
        

def main(argv=None):
    """Entry point for the ``cuflynx-identifiability`` command."""
    parser = _cli.build_parser(
        'Run identifiability analysis (Laplace or profile likelihood) around the parameter '
        'values a previous calibration wrote to the param_id output directory.')
    args = parser.parse_args(argv)
    inp_data_dict = _cli.load_user_inputs(args)
    return _cli.run_stage(
        lambda: run_identifiability_analysis(inp_data_dict), MPI)


if __name__ == '__main__':
    sys.exit(main())
