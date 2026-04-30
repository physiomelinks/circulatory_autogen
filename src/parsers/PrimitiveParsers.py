'''
Created on 29/10/2021

@author: Finbar Argus, Gonzalo D. Maso Talou
'''

import pandas as pd
import numpy as np
import os
import sys
import csv
import json
import copy
import yaml
try:
    from ruamel.yaml.scalarfloat import ScalarFloat
except Exception:
    ScalarFloat = None
import re

try:
    from mpi4py import MPI
    mpi_available = True
    rank = MPI.COMM_WORLD.Get_rank()
except:
    mpi_available = False
    rank=0

root_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir, 'src'))
user_inputs_dir = os.path.join(root_dir, 'user_run_files')
src_dir = os.path.join(os.path.dirname(__file__), '..')
param_id_dir = os.path.join(src_dir, 'param_id')
base_dir = os.path.join(src_dir, '..')
operation_funcs_user_dir = os.path.join(base_dir, 'funcs_user')

class scriptFunctionParser(object):
    '''
    Parses scripts with functions into objects (dicts) which holds the functions
    '''

    def __init__(self):
        sys.path.append(param_id_dir)
        sys.path.append(operation_funcs_user_dir)
        '''
        Constructor
        '''

    def get_operation_funcs_dict(self, mode="numpy"):
        import operation_funcs
        import operation_funcs_user

        return operation_funcs.OPERATION_FUNCS[mode]
    
    def add_user_operation_func(self, operation_funcs_dict, func):
        operation_funcs_dict[func.__name__] = func
        return operation_funcs_dict
    
    def add_user_cost_func(self, cost_funcs_dict, func):
        cost_funcs_dict[func.__name__] = func
        return cost_funcs_dict
    
    def get_cost_funcs_dict(self, mode="numpy"):
        import cost_funcs_user

        return cost_funcs_user.COST_FUNCS[mode]

class YamlFileParser(object):
    '''
    Parses Yaml files 
    '''
    def __init__(self):
        '''
        Constructor
        '''
    
    def parse_user_inputs_file(self, inp_data_dict, obs_path_needed=False, do_generation_with_fit_parameters=False):
        
        if inp_data_dict is None:
            with open(os.path.join(user_inputs_dir, 'user_inputs.yaml'), 'r') as file:
                inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)
            if "user_inputs_path_override" in inp_data_dict.keys() and inp_data_dict["user_inputs_path_override"]:
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
        else:
            user_files_dir = ''

        if inp_data_dict is None:
            print('no inp_data_dict provided and user_inputs.yaml not found, exiting')
            exit()
            
        if 'file_prefix' not in inp_data_dict.keys():
            print('file_prefix not found in inp_data_dict, exiting')
            exit()
        else:
            file_prefix = inp_data_dict['file_prefix']

        if 'couple_to_1d' not in inp_data_dict.keys():
            inp_data_dict['couple_to_1d'] = False
        
        if 'param_id_method' not in inp_data_dict.keys():
            inp_data_dict['param_id_method'] = 'genetic_algorithm'

        if inp_data_dict.get('param_id_method') == 'sp_minimize' and inp_data_dict.get('model_type') != 'casadi_python':
            print(f'Parameter identification with sp_minimize requires model_type to be "casadi_python"')
            exit()

        # overwrite dir paths if set in user_inputs.yaml
        if "resources_dir" in inp_data_dict.keys():
            inp_data_dict['resources_dir'] = os.path.join(user_files_dir, inp_data_dict['resources_dir'])
        else:
            inp_data_dict['resources_dir'] = os.path.join(root_dir, 'resources')
        if "param_id_output_dir" in inp_data_dict.keys():
            inp_data_dict['param_id_output_dir'] = os.path.join(user_files_dir, inp_data_dict['param_id_output_dir'])
        else:
            inp_data_dict['param_id_output_dir'] = os.path.join(root_dir, 'param_id_output')
        if "generated_models_dir" in inp_data_dict.keys():
            inp_data_dict['generated_models_dir'] = os.path.join(user_files_dir, inp_data_dict['generated_models_dir'])
        else:
            inp_data_dict['generated_models_dir'] = os.path.join(root_dir, 'generated_models')
        
        if obs_path_needed:
            if 'param_id_obs_path' in inp_data_dict.keys():
                inp_data_dict['param_id_obs_path'] = os.path.join(user_files_dir, inp_data_dict['param_id_obs_path'])
                if not os.path.exists(inp_data_dict['param_id_obs_path']):
                    print(f'param_id_obs_path={inp_data_dict["param_id_obs_path"]} does not exist')
                    exit()
            else:
                print(f'param_id_obs_path not defined in user_inputs.yaml')
                print(f'Must run param_id.create_param_id_obs to create the param_id_observables')
                inp_data_dict['param_id_obs_path'] = None

            if 'params_for_id_file' in inp_data_dict.keys():
                inp_data_dict['params_for_id_path'] = os.path.join(inp_data_dict['resources_dir'], inp_data_dict['params_for_id_file'])
            else:
                inp_data_dict['params_for_id_path'] = os.path.join(inp_data_dict['resources_dir'], f'{file_prefix}_params_for_id.csv')
                if not os.path.exists(inp_data_dict['params_for_id_path']):
                    print(f'params_for_id_path={inp_data_dict["params_for_id_path"]} does not exist')
                    print(f'Therefore, you must run param_id.create_params_for_id to define the parameters for identification')
                    inp_data_dict['params_for_id_path'] = None

        if do_generation_with_fit_parameters:
            data_str_addon = re.sub('.json', '', os.path.split(inp_data_dict['param_id_obs_path'])[1])
            inp_data_dict['param_id_output_dir_abs_path'] = os.path.join(inp_data_dict['param_id_output_dir'], 
                                                                         inp_data_dict['param_id_method'] + f'_{file_prefix}_{data_str_addon}')
            inp_data_dict['generated_models_subdir'] = os.path.join(inp_data_dict['generated_models_dir'], 
                                                                    file_prefix + '_' + data_str_addon)
        else:
            inp_data_dict['generated_models_subdir'] = os.path.join(inp_data_dict['generated_models_dir'], file_prefix)
        
        if not os.path.exists(inp_data_dict['generated_models_dir']):
            os.mkdir(inp_data_dict['generated_models_dir'])

        if not os.path.exists(inp_data_dict['generated_models_subdir']):
            os.mkdir(inp_data_dict['generated_models_subdir'])
            
        if 'model_type' not in inp_data_dict.keys():
            inp_data_dict['model_type'] = 'cellml_only'
            
        if inp_data_dict.get('model_type') in ['python', 'casadi_python']:
            model_ext = '.py'
        elif inp_data_dict.get('model_type') == 'cellml_only':
            model_ext = '.cellml'
        elif inp_data_dict.get('model_type') == 'cpp':
            model_ext = '.cpp'
        else:
            print(f'Invalid model type: {inp_data_dict.get("model_type")}')
            exit()

        inp_data_dict['model_path'] = os.path.join(inp_data_dict['generated_models_subdir'], f'{file_prefix}{model_ext}')

        if do_generation_with_fit_parameters:
            inp_data_dict['uncalibrated_model_path'] = os.path.join(inp_data_dict["generated_models_dir"], file_prefix,
                                               file_prefix + model_ext)
        else:
            inp_data_dict['uncalibrated_model_path'] = inp_data_dict['model_path']


        if 'dt' not in inp_data_dict.keys():
            inp_data_dict['dt'] = 0.01
        else:
            if ScalarFloat is not None and isinstance(inp_data_dict['dt'], ScalarFloat):
                inp_data_dict['dt'] = float(inp_data_dict['dt'])
            if type(inp_data_dict['dt']) != float:
                print(f'dt must be a float, but is {type(inp_data_dict["dt"])}')
                exit()
            
        if 'pre_time' in inp_data_dict.keys():
            inp_data_dict['pre_time'] = inp_data_dict['pre_time']
        else:
            inp_data_dict['pre_time'] = 0.0

        if 'sim_time' in inp_data_dict.keys():
            inp_data_dict['sim_time'] = inp_data_dict['sim_time']
        else:
            print(f'sim_time not found in inp_data_dict, setting to None so it can be set in protocol_info')
            inp_data_dict['sim_time'] = None

        # Parse and validate the solver parameter
        # Supported solvers: CVODE_opencor (OpenCOR), CVODE_myokit (Myokit), solve_ivp (Python), or casadi_integrator (CasADi)
        
        valid_cellml_solvers = ['CVODE_opencor', 'CVODE_myokit']
        valid_cellml_methods = ['CVODE']
        valid_cpp_solvers = ['CVODE', 'RK4', 'PETSC'] # TODO should this be different to methods?
        valid_cpp_methods = ['CVODE', 'RK4', 'PETSC']
        # Common solve_ivp methods (add more as needed)
        valid_python_solvers = ['solve_ivp']
        valid_solve_ivp_methods = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA', 'forward_euler']
        # CasADi integrator available plugins
        valid_casadi_solvers = ['casadi_integrator']
        valid_casadi_solver_plugins = ['cvodes', 'idas', 'collocation', 'rk']

        solver_name = inp_data_dict.get('solver_info', {}).get('solver')
        if solver_name is None:
            solver_name = inp_data_dict.get('solver')
        
        # Backward compatibility: 
        if solver_name == 'CVODE':
            solver_name = 'CVODE_myokit' # default to CVODE_myokit for cellml models

        if solver_name is None:
            if inp_data_dict.get('model_type') == 'cellml_only':
                solver_name = 'CVODE_opencor'
            elif inp_data_dict.get('model_type') == 'python':
                solver_name = 'solve_ivp'
            elif inp_data_dict.get('model_type') == 'cpp':
                solver_name = 'CVODE'
            elif inp_data_dict.get('model_type') == 'casadi_python':
                solver_name = 'cvodes'
            else:
                print(f'Invalid model type: {inp_data_dict.get("model_type")}')
                exit()
        else:
            if (solver_name not in valid_cellml_solvers and
                solver_name not in valid_cpp_solvers and
                solver_name not in valid_python_solvers and
                solver_name not in valid_casadi_solvers):
                print(f'Invalid solver: {solver_name}')
                exit()
        

        if 'solver_info' not in inp_data_dict.keys(): 
            inp_data_dict['solver_info'] = get_solver_info_default(inp_data_dict['model_type'])
        else:
            if 'MaximumNumberOfSteps' not in inp_data_dict['solver_info'].keys():
                inp_data_dict['solver_info']['MaximumNumberOfSteps'] = get_solver_info_default(inp_data_dict['model_type'])['MaximumNumberOfSteps']
        if 'solver' not in inp_data_dict['solver_info'].keys():
            inp_data_dict['solver_info']['solver'] = solver_name
        elif inp_data_dict.get('model_type') == 'cpp':
            if solver_name.startswith('RK4'):
                inp_data_dict['solver_info']['solver'] = 'RK4'
                solver_name = 'RK4'
            elif solver_name.startswith('CVODE'):
                inp_data_dict['solver_info']['solver'] = 'CVODE'
                solver_name = 'CVODE'
            elif solver_name.startswith('PETSC'):
                inp_data_dict['solver_info']['solver'] = 'PETSC'
                solver_name = 'PETSC'

        solver_info = get_solver_info_default(inp_data_dict['model_type'])
        # overwrite with user-provided solver_info
        for key, value in inp_data_dict['solver_info'].items():
            solver_info[key] = value
        
        dt_solver = solver_info.get('dt_solver')
        if dt_solver is None:
            dt_solver = solver_info.get('MaximumStep')
        if dt_solver is not None:
            solver_info['dt_solver'] = dt_solver
        if solver_info.get('solver', '').startswith('CVODE') and dt_solver is not None:
            solver_info['MaximumStep'] = dt_solver

        inp_data_dict['solver_info'] = solver_info

        if 'solver' in inp_data_dict:
            del inp_data_dict['solver']

        if 'method' in inp_data_dict.get('solver_info', {}):
            solver_method = inp_data_dict['solver_info']['method']
        else:
            if inp_data_dict.get('model_type') == 'cpp':
                if solver_name.startswith('CVODE'):
                    solver_method = 'CVODE'
                    inp_data_dict['solver_info']['method'] = solver_method
                elif solver_name.startswith('RK4'):
                    solver_method = 'RK4'
                    inp_data_dict['solver_info']['method'] = solver_method
                elif solver_name.startswith('PETSC'):
                    solver_method = 'PETSC'
                    inp_data_dict['solver_info']['method'] = solver_method # TODO Bea: add specific solver to be used within PETSC (CN / BDF1 / BDF2 / ...)
                else:
                    print(f'solver set {solver_name} not compatible with model_type cpp : change this in the user_inputs.yaml file')
            else:
                if solver_name.startswith('CVODE'):
                    solver_method = 'CVODE'
                    inp_data_dict['solver_info']['method'] = solver_method
                elif solver_name.startswith('casadi'):
                    print('Method not set in solver_options, which should be set for solver casadi_integrator, '
                          'using default method cvodes')
                    solver_method = 'cvodes'
                    inp_data_dict['solver_info']['method'] = solver_method
                else:
                    print('Method not set in solver_options, which should be set for solver solve_ivp,'
                        'using default method RK45')
                    solver_method = 'RK45'
                    inp_data_dict['solver_info']['method'] = solver_method

        if (solver_name not in valid_cellml_solvers 
            and solver_name not in valid_python_solvers
            and solver_name not in valid_cpp_solvers
            and solver_name not in valid_casadi_solvers):
            print(f'Invalid solver: {solver_name}')
            print(f'Valid CellML solvers: {valid_cellml_solvers}')
            print(f'Valid Python solvers: {valid_python_solvers}')
            print(f'Valid Cpp solvers: {valid_cpp_solvers}')
            print(f'Valid CasAdi solvers: {valid_casadi_solvers}')
            exit()
        
        
        # Validate solver-model compatibility
        # CellML and CasADi solvers cannot be used with Python models
        if inp_data_dict.get('model_type') == 'python' and solver_name not in valid_python_solvers:
                print(f'Solver {solver_method} cannot be used with Python models (model_type="python")')
                print(f'Use {valid_python_solvers} for Python models')
                exit()

        # solve_ivp methods can only be used with Python models
        if solver_method in valid_solve_ivp_methods:
            if inp_data_dict.get('model_type') not in ['python', None]:
                print(f'solve_ivp method {solver_method} requires model_type to be "python"')
                print('Use CVODE_opencor (or legacy CVODE) or CVODE_myokit for CellML models')
                print('Use CVODE or RK4 or PETSC for Cpp models')
                exit()

        if solver_name in valid_cellml_solvers and solver_method not in valid_cellml_methods:
            print(f'solver method {solver_method} not compatible with solver {solver_name}, use {valid_cellml_methods} for CellML models')
            exit()
        
        if solver_name in valid_cpp_solvers and solver_method not in valid_cpp_methods:
            print(f'solver method {solver_method} not compatible with solver {solver_name}, use {valid_cpp_methods} for Cpp models')
            exit()
        if solver_name in valid_python_solvers and solver_method not in valid_solve_ivp_methods:
            print(f'solver method {solver_method} not compatible with solver {solver_name}, use {valid_solve_ivp_methods} for Python models')
            exit()

        # CellML and Python solvers cannot be used with CasADi Python models
        if inp_data_dict.get('model_type') == 'casadi_python' and solver_name not in valid_casadi_solvers:
                print(f'Solver {solver_method} cannot be used with CasADi Python models (model_type="casadi_python")')
                print(f'Use {valid_casadi_solvers} for CasADi Python models')
                exit()

        # CasADi solvers can only be used with CasADi Python models
        if solver_method in valid_casadi_solver_plugins:
            if inp_data_dict.get('model_type') not in ['casadi_python', None]:
                print(f'CasADi solver {solver_method} requires model_type to be "casadi_python"')
                print('Use CVODE_opencor (or legacy CVODE) or CVODE_myokit for CellML models')
                print('Use CVODE or RK4 or PETSC for Cpp models')
                exit()

        if 'DEBUG' in inp_data_dict.keys(): 
            if inp_data_dict['DEBUG']:
                # For backwards compatibility, still set ga_options if debug_ga_options exists
                if 'debug_ga_options' in inp_data_dict.keys():
                    inp_data_dict['ga_options'] = inp_data_dict['debug_ga_options']
                if 'debug_mcmc_options' in inp_data_dict.keys():
                    inp_data_dict['mcmc_options'] = inp_data_dict['debug_mcmc_options']
            else:
                pass
        else:
            inp_data_dict['DEBUG'] = False

        if 'external_modules_dir' not in inp_data_dict.keys() or inp_data_dict['external_modules_dir'] is None:
            inp_data_dict['external_modules_dir'] = None
        else:
            # check if it is an absolute path
            if not os.path.isabs(inp_data_dict['external_modules_dir']):
                inp_data_dict['external_modules_dir'] = os.path.join(user_files_dir, inp_data_dict['external_modules_dir'])
            else:
                inp_data_dict['external_modules_dir'] = inp_data_dict['external_modules_dir']
            # check if external_modules_dir is a valid directory
            if not os.path.exists(inp_data_dict['external_modules_dir']):
                print(f'external_modules_dir={inp_data_dict["external_modules_dir"]} does not exist')
                exit()
        
        # for sensitivity analysis and parameter identification
        if not 'sa_options' in inp_data_dict.keys():
            inp_data_dict['sa_options'] = None

        if inp_data_dict['sa_options'] is None:
            inp_data_dict['sa_options'] = {
                'method': 'sobol',
                'num_samples': 32,
                'sample_type': 'saltelli',
                'output_dir': os.path.join(root_dir, 'sensitivity_outputs', file_prefix + '_SA_results')
            }
        else:
            if 'output_dir' not in inp_data_dict['sa_options'].keys():
                inp_data_dict['sa_options']['output_dir'] = os.path.join(root_dir, 'sensitivity_outputs', file_prefix + '_SA_results')  
            else:
                if not os.path.isabs(inp_data_dict['sa_options']['output_dir']):
                    inp_data_dict['sa_options']['output_dir'] = os.path.join(root_dir, 'sensitivity_outputs', inp_data_dict['sa_options']['output_dir']) 
            
            if not os.path.exists(inp_data_dict['sa_options']['output_dir']):
                os.makedirs(inp_data_dict['sa_options']['output_dir'], exist_ok=True)
            
            if 'method' not in inp_data_dict['sa_options'].keys():
                print('No method specified for sensitivity analysis, setting to sobol by default')
                inp_data_dict['sa_options']['method'] = 'sobol'
            if 'num_samples' not in inp_data_dict['sa_options'].keys():
                print('No num_samples specified for sensitivity analysis, setting to 32 by default')
                inp_data_dict['sa_options']['num_samples'] = 32
            if 'sample_type' not in inp_data_dict['sa_options'].keys():
                print('No sample_type specified for sensitivity analysis, setting to saltelli by default')
                inp_data_dict['sa_options']['sample_type'] = 'saltelli'
            
        if 'do_ia' not in inp_data_dict.keys():
            inp_data_dict['do_ia'] = False
        
        if 'ia_options' not in inp_data_dict.keys():
            inp_data_dict['ia_options'] = {
                'method': 'Laplace'
            }
        else:
            if 'method' not in inp_data_dict['ia_options'].keys():
                print('No method specified for identifiability analysis, setting to Laplace by default')
                inp_data_dict['ia_options']['method'] = 'Laplace'
        
        # Parse optimiser_options - this is the new unified way to specify options
        # Handle backwards compatibility: if ga_options or debug_ga_options is specified, merge into optimiser_options
        if 'optimiser_options' not in inp_data_dict.keys():
            inp_data_dict['optimiser_options'] = {}
        else:
            # Ensure optimiser_options is a dictionary
            if inp_data_dict['optimiser_options'] is None:
                inp_data_dict['optimiser_options'] = {}
            
        # Merge ga_options into optimiser_options for backwards compatibility
        
        # Backwards compatibility: convert ga_options to optimiser_options
        # Only copy entries that don't already exist in optimiser_options to avoid duplicates
        # Note: If DEBUG is True, ga_options may have been set to debug_ga_options above,
        # but we still want to merge the original ga_options first (if it exists and DEBUG is False)
        # or merge debug_ga_options with higher precedence when DEBUG is True
        if not inp_data_dict['DEBUG']:
            # When DEBUG is False, merge ga_options normally
            if 'ga_options' in inp_data_dict.keys() and inp_data_dict['ga_options'] is not None:
                ga_opts = inp_data_dict['ga_options']
                if isinstance(ga_opts, dict):
                    for key, value in ga_opts.items():
                        # Only add if not already in optimiser_options
                        if key not in inp_data_dict['optimiser_options']:
                            inp_data_dict['optimiser_options'][key] = value
                        # Warn if there's a conflict (same key, different value)
                        elif inp_data_dict['optimiser_options'][key] != value:
                            print(f'Warning: ga_options["{key}"] conflicts with optimiser_options["{key}"]. '
                                  f'Using optimiser_options value: {inp_data_dict["optimiser_options"][key]}')
        
        # Handle debug_ga_options for backwards compatibility
        # When DEBUG is True, debug_ga_options should override optimiser_options
        if inp_data_dict['DEBUG']:
            if 'debug_ga_options' in inp_data_dict.keys() and inp_data_dict['debug_ga_options'] is not None:
                debug_ga_opts = inp_data_dict['debug_ga_options']
                if isinstance(debug_ga_opts, dict):
                    for key, value in debug_ga_opts.items():
                        # Debug options override optimiser_options (they take precedence)
                        if key in inp_data_dict['optimiser_options']:
                            if inp_data_dict['optimiser_options'][key] != value:
                                print(f'Note: debug_ga_options["{key}"] overriding optimiser_options["{key}"] '
                                      f'({inp_data_dict["optimiser_options"][key]} -> {value})')
                        inp_data_dict['optimiser_options'][key] = value
        
        # Handle debug_optimiser_options (new preferred way)
        # Only apply when DEBUG is True to avoid overriding production runs
        if inp_data_dict['DEBUG']:
            if 'debug_optimiser_options' in inp_data_dict.keys() and inp_data_dict['debug_optimiser_options'] is not None:
                debug_opts = inp_data_dict['debug_optimiser_options']
                if isinstance(debug_opts, dict):
                    for key, value in debug_opts.items():
                        if key in inp_data_dict['optimiser_options']:
                            if inp_data_dict['optimiser_options'][key] != value:
                                print(f'Note: debug_optimiser_options["{key}"] overriding optimiser_options["{key}"] '
                                      f'({inp_data_dict["optimiser_options"][key]} -> {value})')
                        inp_data_dict['optimiser_options'][key] = value

        # for generation only
    
        inp_data_dict['vessels_csv_abs_path'] = os.path.join(inp_data_dict['resources_dir'], file_prefix + '_vessel_array.csv')
        inp_data_dict['parameters_csv_abs_path'] = os.path.join(inp_data_dict['resources_dir'], inp_data_dict['input_param_file'])

        if inp_data_dict.get('model_type') == 'cpp' and inp_data_dict.get('couple_to_1d'):
            file_prefix_0d = file_prefix + '_0d'
            file_prefix_1d = file_prefix + '_1d'

            vessels_csv_abs_path = inp_data_dict['vessels_csv_abs_path']
            idx_last = vessels_csv_abs_path.rfind(file_prefix)
            vessel_filename_0d = vessels_csv_abs_path[:idx_last] + file_prefix_0d + vessels_csv_abs_path[idx_last+len(file_prefix):]
            vessel_filename_1d = vessels_csv_abs_path[:idx_last] + file_prefix_1d + vessels_csv_abs_path[idx_last+len(file_prefix):]

            inp_data_dict['file_prefix_0d'] = file_prefix_0d
            inp_data_dict['file_prefix_1d'] = file_prefix_1d
            inp_data_dict['vessels_0d_csv_abs_path'] = vessel_filename_0d
            inp_data_dict['vessels_1d_csv_abs_path'] = vessel_filename_1d

        return inp_data_dict

def get_solver_info_default(model_type):
    if model_type == 'cellml_only':
        return {
            'solver': 'CVODE_opencor',
            'MaximumStep': 0.001,
            'MaximumNumberOfSteps': 5000,
            'rtol': 1e-8,
            'atol': 1e-8
        }
    if model_type == 'python':
        return {
            'solver': 'solve_ivp',
            'MaximumStep': 0.001,
            'MaximumNumberOfSteps': 5000,
            'rtol': 1e-8,
            'atol': 1e-8
        }
    if model_type == 'cpp':
        return {
            'solver': 'CVODE',
            'MaximumStep': 0.001,
            'rtol': 1e-8,
            'atol': 1e-8
        }
    if model_type == 'casadi_python':
        return {
            'solver': 'cvodes',
            'MaximumStep': 0.001,
            'rtol': 1e-8,
            'atol': 1e-8
        }
    raise ValueError(f'Invalid model type: {model_type}')

class CSVFileParser(object):
    '''
    Parses CSV files
    '''

    def __init__(self):
        '''
        Constructor
        '''
        
    def get_data_as_dataframe_multistrings(self, filename, has_header=True):
        '''
        Returns the data in the CSV file as a Pandas dataframe where entries in the data array that have two
        entries are put in a list in the entry for the dataframe
        :param filename: filename of CSV file
        :param has_header: If CSV file has a header
        '''
        if( has_header ):
            csv_dataframe = pd.read_csv(filename, dtype=str, na_filter=False)
        else:
            csv_dataframe = pd.read_csv(filename, dtype=str, header=None, na_filter=False)

        csv_dataframe = csv_dataframe.rename(columns=lambda x: x.strip())
        # Ensure object dtype so list-like assignments are allowed (pandas >=2.0 uses StringArray)
        csv_dataframe = csv_dataframe.astype(object)
        for II in range(csv_dataframe.shape[0]):
            for column_index, column_name in enumerate(csv_dataframe.columns):
                entry = csv_dataframe.iat[II, column_index]
                if type(entry) is not str:
                    sub_entries = []
                else:
                    sub_entries = entry.split()

                if column_name in ['vessel_name', 'inp_vessels', 'out_vessels']:
                    if sub_entries == []:
                        new_entry = []
                        pass
                    else:
                        new_entry = [sub_entry.strip() for sub_entry in sub_entries]
                else:
                    if sub_entries == []:
                        new_entry = []
                    else:
                        new_entry = sub_entries[0].strip()

                # Use iat to avoid pandas trying to broadcast list-like values.
                csv_dataframe.iat[II, column_index] = new_entry

        # for column_name in csv_dataframe.columns:
        #     if column_name == 'vessel_name':
        #         continue
        #     csv_dataframe[column_name] = csv_dataframe[column_name].str.strip()
    
        return csv_dataframe

    def get_data_as_dataframe(self, filename, has_header=True):
        '''
        Returns the data in the CSV file as a Pandas dataframe
        :param filename: filename of CSV file
        :param has_header: If CSV file has a header
        '''
        if (has_header):
            csv_dataframe = pd.read_csv(filename, dtype=str)
        else:
            csv_dataframe = pd.read_csv(filename, dtype=str, header=None)

        for column_name in csv_dataframe.columns:
            csv_dataframe[column_name] = csv_dataframe[column_name].str.strip()

        return csv_dataframe

    def get_data_as_nparray(self,filename,has_header=True):
        '''
        Returns the data in the CSV file as a numpy array
        :param filename: filename of CSV file
        :param has_header: If CSV file has a header
        '''

        csv_dataframe = self.get_data_as_dataframe(filename, has_header)
    
        csv_np_array = csv_dataframe.to_numpy()
        dtypes = []
        for column in list(csv_dataframe.columns):
            dtypes.append((column,'<U80'))
            
        csv_np_array = np.array(list(zip(*csv_np_array.T)), dtype=dtypes)
    
        return csv_np_array

    def get_data_as_dictionary(self,filename):
        '''
        Returns the data in the CSV file as a Python dictionary
        :param filename: filename of CSV file
        '''

        csv_dataframe = self.get_data_as_dataframe(filename).T
        csv_dictionary = csv_dataframe.to_dict()
        
        return list(csv_dictionary.values())


    def get_param_id_params_as_lists_of_tuples(self, param_id_dir):

        param_names = []

        with open(os.path.join(os.path.join(param_id_dir, 'param_names_for_gen.csv')), 'r') as f:
            rd = csv.reader(f)
            for row in rd:
                param_names.append(row)


        # get date identifier of the parameter id
        date_id = np.load(os.path.join(os.path.join(param_id_dir, 'date.npy'))).item()

        param_vals = np.load(os.path.join(param_id_dir, 'best_param_vals.npy'))
        param_name_and_val = []
        
        for name_or_list, val in zip(param_names, param_vals):
            if isinstance(name_or_list, list):
                for name in name_or_list:
                    param_name_and_val.append((name, val))
            else:
                param_name_and_val.append((name, val))

        return param_name_and_val, date_id


class JSONFileParser(object):
    '''
    Parses json files
    '''

    def __init__(self):
        '''
        Constructor
        '''

    def json_to_dataframe(self, json_path):
        with open(json_path, encoding='utf-8-sig') as rf:
            json_obj = json.load(rf)
        df = pd.DataFrame(json_obj)
        return df

    def json_to_dataframe_with_user_dir(self, json_dir, json_user_dir, external_modules_dir):
        dfs = [self.json_to_dataframe(os.path.join(json_dir, file)) \
                for file in os.listdir(json_dir) if file.endswith('.json')]
        user_module_dfs = [self.json_to_dataframe(os.path.join(json_user_dir, file)) \
                for file in os.listdir(json_user_dir) if file.endswith('.json')]
        if external_modules_dir is not None:
            external_module_dfs = [self.json_to_dataframe(os.path.join(external_modules_dir, file)) \
                    for file in os.listdir(external_modules_dir) if file.endswith('.json')]
        else:
            external_module_dfs = []
            
        df = None
        for json_df in dfs:
            if df is None:
                df = json_df
            else:
                # concatenate dataframes, ignore index to reset the index
                # so that it is not duplicated
                df = pd.concat([df, json_df], ignore_index=True)

        for user_module_df in user_module_dfs:
            df = pd.concat([df, user_module_df], ignore_index=True)
        for external_module_df in external_module_dfs:
            df = pd.concat([df, external_module_df], ignore_index=True)
        return df
    
    def get_data_as_dataframe_multistrings(self, filename, has_header=True):
        '''
        Returns the data in the CSV file as a Pandas dataframe where entries in the data array that have two
        entries are put in a list in the entry for the dataframe
        :param filename: filename of CSV file
        '''
        with open(filename, 'r') as f:
            json_obj = json.load(f)
        df = pd.DataFrame(json_obj)
        return df

    def append_module_config_info_to_vessel_df(self, vessel_df, module_df):
        # add columns to vessel_df

        add_on_lists = {column:[] for column in module_df.columns[2:]}
        for vessel_tup in vessel_df.itertuples():
            vessel_type = vessel_tup.vessel_type
            BC_type = vessel_tup.BC_type
            if len(BC_type) <1 or len(vessel_type) <1:
                print('You have an empty entry in your vessel array, exiting')
                exit()
            this_vessel_module_df = module_df.loc[((module_df["vessel_type"] == vessel_type)
                                                   & (module_df["BC_type"] == BC_type))].squeeze()
            if this_vessel_module_df.empty:
                print(f'combination of vessel_type = {vessel_type} and BC_type = {BC_type} doesn\'t exist, check module_config.json',
                        'for this combination')
                exit()
            for column in add_on_lists:
                # deepcopy to make sure that the lists for different vessel same module are not linked
                val = this_vessel_module_df[column]
                is_na = False
                try:
                    mask = pd.isna(val)
                    if isinstance(mask, (np.bool_, bool)):
                        is_na = bool(mask)
                    else:
                        # array-like: consider NaN only if all entries are NaN
                        is_na = bool(np.all(mask))
                except Exception:
                    is_na = False

                if is_na:
                    add_on_lists[column].append("None")
                else:
                    add_on_lists[column].append(copy.deepcopy(val))

        for column in add_on_lists:
            vessel_df[column] = add_on_lists[column]

class ObsAndParamDataParser(object):
    def __init__(self):
        pass

    def parse_obs_data_json(self, param_id_obs_path=None, obs_data_dict=None, pre_time=None, sim_time=None):
        """
        Loads the ground truth observation data from the JSON file and returns 
        the core data structures: gt_df, protocol_info, and prediction_info.
        """
        
        if param_id_obs_path is not None:
            with open(param_id_obs_path, encoding='utf-8-sig') as rf:
                json_obj = json.load(rf)
        elif obs_data_dict is not None:
            json_obj = obs_data_dict
        else:
            print("No obs data path or obs data dict provided, exiting")
            return None

        gt_df, protocol_info, prediction_info = None, None, None
        REQUIRED = "REQUIRED"

        def _is_missing_scalar(val):
            if val is None:
                return True
            try:
                is_na = pd.isna(val)
            except Exception:
                return False
            return isinstance(is_na, (bool, np.bool_)) and bool(is_na)

        # --- Case 1: Simple list of data items ---
        if type(json_obj) == list:
            gt_df = pd.DataFrame(json_obj)
            protocol_info = {"pre_times": [pre_time], 
                             "sim_times": [[sim_time]],
                             "params_to_change": {}}
            prediction_info = {'names': [], 'units': [], 'names_for_plotting': [], 'experiment_idxs': []}
            

        # --- Case 2: Dictionary structure ---
        elif type(json_obj) == dict:
            # Load Data Items (gt_df)
            if 'data_items' in json_obj.keys() or 'data_item' in json_obj.keys():
                data_items = json_obj.get('data_items', json_obj.get('data_item', []))
                gt_df = pd.DataFrame(data_items)
            else:
                print("data_items not found in json object. ",
                      "Please check that data_items is the key for the list of data items")

            # Load Protocol Info
            if 'protocol_info' in json_obj.keys():
                protocol_info = copy.deepcopy(json_obj['protocol_info'])
            else:
                if pre_time is None or sim_time is None:
                    print("protocol_info not found in json object. ",
                          "If this is the case sim_time and pre_time must be set",
                          "in the user_inputs.yaml file")
                    exit()
                protocol_info = {"pre_times": [pre_time], "sim_times": [[sim_time]], "params_to_change": {}}

            protocol_schema = {
                "pre_times": {"types": (list, tuple, np.ndarray), "default": [pre_time] if pre_time is not None else REQUIRED},
                "sim_times": {"types": (list, tuple, np.ndarray), "default": [[sim_time]] if sim_time is not None else REQUIRED},
                "params_to_change": {"types": (dict,), "default": {}},
                "experiment_labels": {"types": (list, tuple, np.ndarray), "default": None},
                "experiment_colors": {"types": (list, tuple, np.ndarray), "default": None},
                "comment": {"types": (str,), "default": None},
            }

            unknown_protocol_keys = sorted(set(protocol_info.keys()) - set(protocol_schema.keys()))
            if len(unknown_protocol_keys) > 0:
                raise ValueError(
                    f"Unknown protocol_info keys not in schema: {unknown_protocol_keys}"
                )

            missing_protocol_required = []
            protocol_type_errors = []
            for key, rules in protocol_schema.items():
                allowed = rules["types"]
                default = rules["default"]

                if key not in protocol_info or _is_missing_scalar(protocol_info[key]):
                    if default == REQUIRED:
                        missing_protocol_required.append(key)
                    else:
                        protocol_info[key] = copy.deepcopy(default)
                        continue

                if not isinstance(protocol_info[key], allowed):
                    protocol_type_errors.append(
                        f"protocol_info['{key}']: expected {allowed}, got {type(protocol_info[key])}"
                    )

            if len(missing_protocol_required) > 0:
                raise ValueError(
                    f"Missing required protocol_info keys: {sorted(missing_protocol_required)}"
                )
            if len(protocol_type_errors) > 0:
                raise ValueError(
                    "Invalid protocol_info value types:\n" + "\n".join(protocol_type_errors)
                )

            # Load Prediction Info
            if 'prediction_items' in json_obj.keys():
                prediction_items = json_obj['prediction_items']
                if not isinstance(prediction_items, (list, tuple)):
                    raise ValueError(
                        f"prediction_items must be a list of dict entries, got {type(prediction_items)}"
                    )

                prediction_entry_schema = {
                    "variable": {"types": (str,), "default": REQUIRED},
                    "unit": {"types": (str,), "default": REQUIRED},
                    "name_for_plotting": {"types": (str,), "default": lambda entry: entry["variable"]},
                    "experiment_idx": {"types": (int, np.integer), "default": 0},
                }

                prediction_info = {'names': [], 'units': [], 'names_for_plotting': [], 'experiment_idxs': []}
                for entry_idx, raw_entry in enumerate(prediction_items):
                    if not isinstance(raw_entry, dict):
                        raise ValueError(
                            f"prediction_items[{entry_idx}] must be a dict, got {type(raw_entry)}"
                        )
                    entry = copy.deepcopy(raw_entry)

                    unknown_pred_keys = sorted(set(entry.keys()) - set(prediction_entry_schema.keys()))
                    if len(unknown_pred_keys) > 0:
                        raise ValueError(
                            f"Unknown keys in prediction_items[{entry_idx}] not in schema: {unknown_pred_keys}"
                        )

                    missing_pred_required = []
                    pred_type_errors = []
                    for key, rules in prediction_entry_schema.items():
                        allowed = rules["types"]
                        default = rules["default"]

                        if key not in entry or _is_missing_scalar(entry[key]):
                            if default == REQUIRED:
                                missing_pred_required.append(key)
                                continue
                            entry[key] = default(entry) if callable(default) else copy.deepcopy(default)

                        if not isinstance(entry[key], allowed):
                            pred_type_errors.append(
                                f"prediction_items[{entry_idx}]['{key}']: expected {allowed}, got {type(entry[key])}"
                            )

                    if len(missing_pred_required) > 0:
                        raise ValueError(
                            f"Missing required keys in prediction_items[{entry_idx}]: {sorted(missing_pred_required)}"
                        )
                    if len(pred_type_errors) > 0:
                        raise ValueError(
                            "Invalid prediction_items value types:\n" + "\n".join(pred_type_errors)
                        )

                    prediction_info['names'].append(entry['variable'])
                    prediction_info['units'].append(entry['unit'])
                    prediction_info['names_for_plotting'].append(entry['name_for_plotting'])
                    prediction_info['experiment_idxs'].append(entry['experiment_idx'])
            else:
                prediction_info = {'names': [], 'units': [], 'names_for_plotting': [], 'experiment_idxs': []}
            
        else:
            print(f"Error: unknown data type for imported json object of {type(json_obj)}")
            return None

        # Fill common optional fields so downstream processing can rely on defaults.
        if gt_df is not None:
            schema = {
                "variable": {"types": (str,), "default": REQUIRED},
                "name_for_plotting": {"types": (str,), "default": lambda df: df["variable"]},
                "data_type": {"types": (str,), "default": REQUIRED},
                "unit": {"types": (str,), "default": REQUIRED},
                "weight": {"types": (int, float, np.integer, np.floating, list, np.ndarray), "default": 1.0},
                "operands": {"types": (list, tuple, np.ndarray), "default": REQUIRED},
                "operation": {"types": (str,), "default": None},
                "operation_kwargs": {"types": (dict,), "default": lambda df: [{} for _ in range(len(df))]},
                "value": {"types": (int, float, np.integer, np.floating, list, np.ndarray), "default": REQUIRED},
                "std": {"types": (int, float, np.integer, np.floating, list, np.ndarray), "default": REQUIRED},
                "experiment_idx": {"types": (int, np.integer), "default": 0},
                "subexperiment_idx": {"types": (int, np.integer), "default": 0},
                "plot_type": {"types": (str,), "default": None},
                "plot_color": {"types": (str,), "default": None},
                "comment": {"types": (str,), "default": None},
                "cost_type": {"types": (str,), "default": "MSE"},
                "obs_type": {"types": (str,), "default": None},
                "frequencies": {"types": (list, tuple, np.ndarray, int, float, np.integer, np.floating), "default": None},
                # If omitted, phase weighting should follow the same weighting as amplitude.
                "phase_weight": {"types": (int, float, np.integer, np.floating, list, np.ndarray), "default": lambda df: df["weight"]},
                "phase": {"types": (list, tuple, np.ndarray, int, float, np.integer, np.floating), "default": None},
                "prob_dist_params": {"types": (dict,), "default": None},
                "obs_dt": {"types": (int, float, np.integer, np.floating), "default": None},
                "dt": {"types": (int, float, np.integer, np.floating), "default": None},
                "sample_rate": {"types": (int, float, np.integer, np.floating), "default": None},
            }

            unknown_cols = sorted(set(gt_df.columns) - set(schema.keys()))
            if len(unknown_cols) > 0:
                raise ValueError(
                    f"Unknown data_item keys not in schema: {unknown_cols}"
                )

            type_errors = []
            missing_required_cols = []
            for col, rules in schema.items():
                allowed = rules["types"]
                default = rules["default"]

                if col not in gt_df.columns:
                    if default == REQUIRED:
                        missing_required_cols.append(col)
                        continue
                    if callable(default):
                        gt_df[col] = pd.Series(default(gt_df), index=gt_df.index)
                    else:
                        gt_df[col] = default
                else:
                    if default != REQUIRED:
                        missing_mask = gt_df[col].apply(_is_missing_scalar)
                        if missing_mask.any():
                            if callable(default):
                                default_series = pd.Series(default(gt_df), index=gt_df.index)
                                gt_df[col] = gt_df[col].where(~missing_mask, default_series)
                            else:
                                gt_df[col] = gt_df[col].where(~missing_mask, default)

                for row_idx, val in gt_df[col].items():
                    if _is_missing_scalar(val):
                        continue
                    if not isinstance(val, allowed):
                        type_errors.append(
                            f"row {row_idx}, column '{col}': expected {allowed}, got {type(val)}"
                        )

            if len(missing_required_cols) > 0:
                raise ValueError(
                    f"Missing required data_item keys: {sorted(missing_required_cols)}"
                )

            if len(type_errors) > 0:
                raise ValueError(
                    "Invalid data_item value types:\n" + "\n".join(type_errors)
                )
        
        return {
            "gt_df": gt_df, 
            "protocol_info": protocol_info, 
            "prediction_info": prediction_info
        }

    def process_obs_info(self, gt_df, output_dir, dt):
        """
        Generates the detailed obs_info dictionary, including names, units, 
        plotting defaults, operations, and kwargs from the ground truth dataframe.
        """
        obs_info = {}
        
        # --- Simple Array Generation ---
        N = gt_df.shape[0]
        
        obs_info["obs_names"] = gt_df["variable"].tolist()
        obs_info["data_types"] = gt_df["data_type"].tolist()
        obs_info["units"] = gt_df["unit"].tolist()
        obs_info["experiment_idxs"] = [gt_df.iloc[II].get("experiment_idx", 0) for II in range(N)]
        obs_info["subexperiment_idxs"] = [gt_df.iloc[II].get("subexperiment_idx", 0) for II in range(N)]

        # --- Plotting Colors ---
        possible_colors = ['b', 'g', 'c', 'm', 'y', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:orange']
        obs_info["plot_colors"] = [gt_df.iloc[II].get("plot_color", possible_colors[II % len(possible_colors)]) 
                                        for II in range(N)]
        
        # --- Plotting Type Defaults (Logic preserved) ---
        obs_info["plot_type"] = []
        warning_printed = False
        for II in range(N):
            if "plot_type" not in gt_df.iloc[II].keys():
                if gt_df.iloc[II]["data_type"] == "constant":
                    if not warning_printed:
                        print('constant data types plot type defaults to horizontal lines',
                            'change "plot_type" in obs_data.json to change this')
                        warning_printed = True
                    obs_info["plot_type"].append("horizontal")
                elif gt_df.iloc[II]["data_type"] == "prob_dist":
                    if not warning_printed:
                        print('prob_dist data types plot type defaults to horizontal lines',
                            'change "plot_type" in obs_data.json to change this')
                        warning_printed = True
                    obs_info["plot_type"].append("horizontal")
                elif gt_df.iloc[II]["data_type"] == "series":
                    obs_info["plot_type"].append("series")
                elif gt_df.iloc[II]["data_type"] == "frequency":
                    obs_info["plot_type"].append("frequency")
                elif gt_df.iloc[II]["data_type"] == "plot_dist":
                    obs_info["plot_type"].append("horizontal")
                else:
                    print(f'data type {gt_df.iloc[II]["data_type"]} not recognised')
            else:
                obs_info["plot_type"].append(gt_df.iloc[II]["plot_type"])
                if obs_info["plot_type"][II] in ["None", "null", "Null", "none", "NONE"]:
                    obs_info["plot_type"][II] = None

        # --- Operations (Mapping obs_type to operation) ---
        obs_info["operations"] = []
        obs_info["operands"] = []
        obs_info["operation_kwargs"] = [gt_df.iloc[II].get("operation_kwargs", {}) for II in range(N)]
        obs_info["freqs"] = [gt_df.iloc[II].get("frequencies") for II in range(N)]
        obs_info["names_for_plotting"] = [gt_df.iloc[II].get("name_for_plotting", obs_info["obs_names"][II]) for II in range(N)]

        for II in range(N):
            op = gt_df.iloc[II].get("operation")
            obs_type = gt_df.iloc[II].get("obs_type")
            operands = gt_df.iloc[II].get("operands")

            if op in ["Null", "None", "null", "none", "", "nan", np.nan, None]:
                if obs_type in ["series", "frequency"]:
                    obs_info["operations"].append(None)
                    obs_info["operands"].append(operands)
                elif obs_type in ["min", "max", "mean"]: 
                    obs_info["operations"].append(obs_type)
                    obs_info["operands"].append([gt_df.iloc[II]["variable"]])
                else:
                    obs_info["operations"].append(None)
                    obs_info["operands"].append(operands)
            else:
                obs_info["operations"].append(op)
                obs_info["operands"].append(operands)

        # --- Weights and Cost Types ---
        weights = gt_df["weight"].to_numpy()
        data_types = np.array(obs_info["data_types"])
        
        obs_info["num_obs"] = N
        obs_info["weight_const_vec"] = weights[data_types == "constant"]
        obs_info["weight_series_vec"] = weights[data_types == "series"]
        obs_info["weight_amp_vec"] = weights[data_types == "frequency"]
        obs_info["weight_prob_dist_vec"] = weights[data_types == "prob_dist"]

        phase_weights = gt_df.apply(
            lambda row: row["phase_weight"] if row.get("phase_weight") is not None else row["weight"],
            axis=1,
        )
        obs_info["weight_phase_vec"] = phase_weights[data_types == "frequency"].to_numpy()

        obs_info["cost_type"] = [gt_df.iloc[II].get("cost_type", "MSE") for II in range(N)]

        obs_info = self.get_ground_truth_values(gt_df, obs_info, output_dir, dt)
        
        return obs_info
    
    def get_ground_truth_values(self, gt_df, obs_info, output_dir, dt):

        # _______ First we access data for constant values

        # TODO make all of the below lists instead of arrays? So we can have different sized entries.

        ground_truth_const = np.array([gt_df.iloc[II]["value"] for II in range(gt_df.shape[0])
                                        if gt_df.iloc[II]["data_type"] == "constant"])

        # _______ Then for time series
        ground_truth_series = [np.array(gt_df.iloc[II]["value"]) for II in range(gt_df.shape[0])
                                        if gt_df.iloc[II]["data_type"] == "series"]

        # _______ Then for frequency series
        ground_truth_amp = np.array([gt_df.iloc[II]["value"] for II in range(gt_df.shape[0])
                                        if gt_df.iloc[II]["data_type"] == "frequency"])

        # then for ground truth probability distributions
        ground_truth_prob_dist_params = np.array([gt_df.iloc[II]["prob_dist_params"] for II in range(gt_df.shape[0])
                                            if gt_df.iloc[II]["data_type"] == "prob_dist"])


        # _______ and the phase of the freq data
        ground_truth_phase_list = []
        for II in range(gt_df.shape[0]):
            if gt_df.iloc[II]["data_type"] == "frequency":
                if "phase" not in gt_df.iloc[II].keys():
                    ground_truth_phase_list.append(None)
                else:
                    ground_truth_phase_list.append(gt_df.iloc[II]["phase"])
        ground_truth_phase = np.array(ground_truth_phase_list)

        # get the dt for the series data
        dt_list = []
        for II in range(gt_df.shape[0]):
            if gt_df.iloc[II]["data_type"] == "series":
                if "obs_dt" not in gt_df.iloc[II].keys():
                    print("dt not found in obs_data.json for series data, exiting")
                    exit()
                dt_list.append(gt_df.iloc[II]["obs_dt"])
        
        obs_info["obs_dt"] = np.array(dt_list)
        
        if len(obs_info["obs_dt"]) > 0:
            if min(obs_info["obs_dt"]) < dt:
                print("one of the dt in obs_data.json is less than the dt in user_inputs.yaml, the output timestep"
                    "defined in user_inputs.yaml must be less than the smallest dt for your data. Exiting")
                exit()

        # The std for the different observables
        obs_info["std_const_vec"] = np.array([gt_df.iloc[II]["std"] for II in range(gt_df.shape[0])
                                       if gt_df.iloc[II]["data_type"] == "constant"])

        obs_info["std_series_vec"] = [np.array(gt_df.iloc[II]["std"]) for II in range(gt_df.shape[0])
                                        if gt_df.iloc[II]["data_type"] == "series"]

        obs_info["std_amp_vec"] = np.array([gt_df.iloc[II]["std"] for II in range(gt_df.shape[0])
                                        if gt_df.iloc[II]["data_type"] == "frequency"])

        # if len(ground_truth_series) > 0:
            # TODO what if we have ground truths of different size or sample rate?
            # ground_truth_series = np.stack(ground_truth_series)
            # removed because we have data of different sizes

        if len(ground_truth_amp) > 0:
            ground_truth_amp = np.stack(ground_truth_amp)

        if len(ground_truth_phase) > 0:
            ground_truth_phase = np.stack(ground_truth_phase)

        if rank == 0:
            np.save(os.path.join(output_dir, 'ground_truth_const.npy'), ground_truth_const)
            if len(ground_truth_series) > 0:
                np.save(os.path.join(output_dir, 'ground_truth_series.npy'), 
                        np.array(ground_truth_series, dtype=object), allow_pickle=True)
            if len(ground_truth_amp) > 0:
                np.save(os.path.join(output_dir, 'ground_truth_amp.npy'), ground_truth_amp)
            if len(ground_truth_phase) > 0:
                np.save(os.path.join(output_dir, 'ground_truth_phase.npy'), ground_truth_phase)

        obs_info["ground_truth_const"] = ground_truth_const
        obs_info["ground_truth_prob_dist_params"] = ground_truth_prob_dist_params
        obs_info["ground_truth_series"] = ground_truth_series
        obs_info["ground_truth_amp"] = ground_truth_amp
        obs_info["ground_truth_phase"] = ground_truth_phase

        # create a mapping between const_idx and the obs_idx
        const_count = 0
        series_count = 0
        freq_count = 0
        prob_dist_count = 0
        obs_info["const_idx_to_obs_idx"] = []
        obs_info["series_idx_to_obs_idx"] = []
        obs_info["freq_idx_to_obs_idx"] = []
        obs_info["prob_dist_idx_to_obs_idx"] = []
        for obs_idx in range(obs_info["num_obs"]):
            if obs_info["data_types"][obs_idx] == "constant":
                obs_info["const_idx_to_obs_idx"].append(obs_idx)
                const_count += 1
            elif obs_info["data_types"][obs_idx] == "series":
                obs_info["series_idx_to_obs_idx"].append(obs_idx)
                series_count += 1
            elif obs_info["data_types"][obs_idx] == "frequency":
                obs_info["freq_idx_to_obs_idx"].append(obs_idx)
                freq_count += 1
            elif obs_info["data_types"][obs_idx] == "prob_dist":
                obs_info["prob_dist_idx_to_obs_idx"].append(obs_idx)
                prob_dist_count += 1

        return obs_info

    def process_protocol_and_weights(self, gt_df, protocol_info, dt):
        """
        Calculates time totals, validates protocol labels/colors, and generates 
        the scaled weight maps for experiment/subexperiment cost calculation.
        """
        protocol = protocol_info
        df = gt_df
        
        # --- Protocol Info Preprocessing ---
        protocol['num_experiments'] = len(protocol["sim_times"])
        protocol['num_sub_per_exp'] = [len(protocol["sim_times"][exp_idx]) for exp_idx in range(protocol["num_experiments"])]
        protocol['num_sub_total'] = sum(protocol['num_sub_per_exp'])
        
        protocol["total_sim_times_per_exp"] = []
        protocol["tSims_per_exp"] = []
        protocol["num_steps_total_per_exp"] = []

        for exp_idx in range(protocol['num_experiments']):
            total_sim_time = np.sum(protocol["sim_times"][exp_idx])
            num_steps_total = int(total_sim_time / dt)
            tSim_per_exp = np.linspace(0.0, total_sim_time, num_steps_total + 1)
            
            protocol["total_sim_times_per_exp"].append(total_sim_time)
            protocol["tSims_per_exp"].append(tSim_per_exp)
            protocol["num_steps_total_per_exp"].append(num_steps_total)
            
        # --- Protocol Info Validation ---
        N_exp = protocol['num_experiments']
        
        if "experiment_colors" not in protocol or protocol["experiment_colors"] is None:
            protocol["experiment_colors"] = ['r'] * N_exp
        elif len(protocol["experiment_colors"]) != N_exp:
            print('Error: experiment_colors length does not match num_experiments, exiting')
            exit()
            
        if "experiment_labels" not in protocol or protocol["experiment_labels"] is None:
            protocol["experiment_labels"] = [None] * N_exp
        elif len(protocol["experiment_labels"]) != N_exp:
            print('Error: experiment_labels length does not match num_experiments, exiting')
            exit()

        # --- Weight Mapping Initialization ---
        
        # Ensure experiment_idx and subexperiment_idx exist in the DataFrame
        # IMPORTANT: These columns must be added safely if they don't exist
        df["experiment_idx"] = df.apply(lambda row: row.get("experiment_idx", 0), axis=1)
        df["subexperiment_idx"] = df.apply(lambda row: row.get("subexperiment_idx", 0), axis=1)

        # Initialize nested lists for weight maps (one list per data type)
        const_map = [[[] for _ in range(protocol['num_sub_per_exp'][exp_idx])] for exp_idx in range(N_exp)]
        series_map = [[[] for _ in range(protocol['num_sub_per_exp'][exp_idx])] for exp_idx in range(N_exp)]
        amp_map = [[[] for _ in range(protocol['num_sub_per_exp'][exp_idx])] for exp_idx in range(N_exp)]
        phase_map = [[[] for _ in range(protocol['num_sub_per_exp'][exp_idx])] for exp_idx in range(N_exp)]
        prob_dist_map = [[[] for _ in range(protocol['num_sub_per_exp'][exp_idx])] for exp_idx in range(N_exp)]

        
        # --- Calculate Scaled Weight Maps ---
        
        for exp_idx in range(N_exp):
            for this_sub_idx in range(protocol['num_sub_per_exp'][exp_idx]):
                
                # Mask to find observations belonging to the current experiment/subexperiment
                mask = (df["experiment_idx"] == exp_idx) & (df["subexperiment_idx"] == this_sub_idx)
                
                # Iterate over all possible data types
                for data_type, weight_map in [
                    ("constant", const_map), ("series", series_map), ("frequency", amp_map), ("prob_dist", prob_dist_map)
                ]:
                    # Create the full weight vector (Weight if matched, 0.0 otherwise)
                    full_weights = np.where(mask & (df["data_type"] == data_type), df["weight"], 0.0)
                    weight_map[exp_idx][this_sub_idx] = full_weights
                
                # Handle phase map separately
                freq_mask = mask & (df["data_type"] == "frequency")
                # Use "phase_weight" if present, otherwise use "weight", or 0.0
                phase_weights = np.where(
                    freq_mask,
                    df.apply(
                        lambda row: row["phase_weight"] if row.get("phase_weight") is not None else row["weight"],
                        axis=1,
                    ),
                    0.0,
                )
                phase_map[exp_idx][this_sub_idx] = phase_weights

        # --- Store Final Maps in protocol_info ---
        protocol["scaled_weight_const_from_exp_sub"] = const_map
        protocol["scaled_weight_series_from_exp_sub"] = series_map
        protocol["scaled_weight_amp_from_exp_sub"] = amp_map
        protocol["scaled_weight_phase_from_exp_sub"] = phase_map
        protocol["scaled_weight_prob_dist_from_exp_sub"] = prob_dist_map
        
        return protocol
    
    def get_param_id_info(self, params_for_id_path, idxs_to_ignore= None):
    
        if not params_for_id_path:
            print(f'params_for_id_path cannot be None, exiting')
            return None

        csv_parser = CSVFileParser()
        input_params = csv_parser.get_data_as_dataframe_multistrings(params_for_id_path)
        return self._build_param_id_info_from_df(input_params, idxs_to_ignore=idxs_to_ignore)

    def get_param_id_info_from_entries(self, params_for_id_entries, idxs_to_ignore=None):
        """
        Build param_id_info from a list/dict of parameter entries.
        Each entry should include: vessel_name, param_name, min, max.
        """
        if params_for_id_entries is None:
            print('params_for_id_entries cannot be None, exiting')
            return None

        # Allow callers to pass a dict wrapper
        if isinstance(params_for_id_entries, dict):
            if "params_for_id_path" in params_for_id_entries:
                return self.get_param_id_info(params_for_id_entries["params_for_id_path"],
                                              idxs_to_ignore=idxs_to_ignore)
            if "params" in params_for_id_entries:
                params_for_id_entries = params_for_id_entries["params"]

        if not isinstance(params_for_id_entries, list):
            raise ValueError("params_for_id_entries must be a list of dicts or include params_for_id_path")

        input_params = pd.DataFrame(params_for_id_entries)
        return self._build_param_id_info_from_df(input_params, idxs_to_ignore=idxs_to_ignore)

    def _build_param_id_info_from_df(self, input_params, idxs_to_ignore=None):
        if input_params is None or input_params.empty:
            raise ValueError("No parameter entries provided")

        required_cols = {"vessel_name", "param_name", "min", "max"}
        missing = required_cols - set(input_params.columns)
        if missing:
            raise ValueError(f"params_for_id is missing required columns: {sorted(list(missing))}")

        input_params = input_params.copy()

        def _to_list(val):
            if isinstance(val, list):
                return val
            if val is None:
                return []
            if isinstance(val, float) and np.isnan(val):
                return []
            val_str = str(val).strip()
            if val_str == "":
                return []
            return [entry.strip() for entry in val_str.split()]

        input_params["vessel_name"] = input_params["vessel_name"].apply(_to_list)

        # --- 1. Filter the DataFrame first ---
        # Create a mask for indices to KEEP (not ignore)
        if idxs_to_ignore is not None:
            all_indices = set(range(input_params.shape[0]))
            valid_indices = sorted(list(all_indices - set(idxs_to_ignore)))
            filtered_params = input_params.iloc[valid_indices].reset_index(drop=True)
        else:
            filtered_params = input_params.reset_index(drop=True)

        N_params = filtered_params.shape[0]

        param_id_info = {}
        param_names_for_gen = []
        param_id_info["param_names"] = []

        # --- 2. Iterate ONLY over the filtered data ---
        for II in range(N_params):
            # Current row data from the filtered DataFrame
            row = filtered_params.iloc[II]

            # A. Build the full, complex names (e.g., 'vessel_name/param_name')
            param_full_names = [
                row["vessel_name"][JJ] + '/' + row["param_name"]
                for JJ in range(len(row["vessel_name"]))
            ]
            param_id_info["param_names"].append(param_full_names)

            # B. Build the simplified names for generator/code
            if row["vessel_name"][0] == 'global':
                param_names_for_gen.append([row["param_name"]])
            else:
                param_gen_names = [
                    row["param_name"] + '_' + row["vessel_name"][JJ]
                    for JJ in range(len(row["vessel_name"]))
                ]
                param_names_for_gen.append(param_gen_names)

        # --- 3. Set Arrays using the filtered DataFrame ---
        param_id_info["param_mins"] = filtered_params["min"].to_numpy(dtype=float)
        param_id_info["param_maxs"] = filtered_params["max"].to_numpy(dtype=float)

        if "name_for_plotting" in filtered_params.columns:
            param_id_info["param_names_for_plotting"] = filtered_params["name_for_plotting"].to_numpy()
        else:
            param_id_info["param_names_for_plotting"] = np.array([p_names[0]
                                                                  for p_names in param_id_info["param_names"]])

        if "prior" in filtered_params.columns:
            param_id_info["param_prior_types"] = filtered_params["prior"].to_numpy()
        else:
            param_id_info["param_prior_types"] = np.array(["uniform"] * N_params)

        param_id_info["param_names_for_gen"] = param_names_for_gen

        return param_id_info

    def save_param_names(self, param_id_info, output_dir):
        """
        Saves the generated parameter names and generator names to CSV files.
        Requires the dictionary returned by _process_param_info.
        """
        if rank == 0:
            # 1. Save param_names (vessel_name/param_name format)
            param_names_path = os.path.join(output_dir, 'param_names.csv')
            with open(param_names_path, 'w', newline='') as f:
                wr = csv.writer(f)
                wr.writerows(param_id_info["param_names"])
            
            # 2. Save param_names_for_gen (simplified format)
            param_gen_path = os.path.join(output_dir, 'param_names_for_gen.csv')
            with open(param_gen_path, 'w', newline='') as f:
                wr = csv.writer(f)
                wr.writerows(param_id_info["param_names_for_gen"])
        return




