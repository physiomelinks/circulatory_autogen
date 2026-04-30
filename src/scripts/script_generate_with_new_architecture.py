'''
Created on 29/10/2021

@author: Gonzalo D. Maso Talou, Finbar J. Argus
'''

import sys
import os
import re
import traceback
import yaml
import numpy as np
root_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir, 'src'))

user_inputs_dir= os.path.join(root_dir, 'user_run_files')

from parsers.ModelParsers import CSV0DModelParser
from generators.CVSCellMLGenerator import CVS0DCellMLGenerator
from generators.CVSCppGenerator import CVS0DCppGenerator, CVS1DPythonGenerator
from generators.PythonGenerator import PythonGenerator
from parsers.PrimitiveParsers import YamlFileParser
from utilities.utility_funcs import change_parameter_values_and_save

try:
    import libcellml as lc
    LIBCELLML_AVAILABLE = True
except ImportError:
    LIBCELLML_AVAILABLE = False


def _is_cellml2_model_with_libcellml(model_path):
    """Return True when strict libCellML parsing/validation succeeds."""
    if not LIBCELLML_AVAILABLE:
        return False

    try:
        with open(model_path, "r", encoding="utf-8") as rf:
            model_text = rf.read()

        parser = lc.Parser(True)
        model = parser.parseModel(model_text)
        if model is None or parser.issueCount() > 0:
            return False

        validator = lc.Validator()
        validator.validateModel(model)
        return validator.issueCount() == 0
    except Exception:
        return False


def generate_with_new_architecture(do_generation_with_fit_parameters=False,
                                   inp_data_dict=None):

    yaml_parser = YamlFileParser()
    inp_data_dict = yaml_parser.parse_user_inputs_file(inp_data_dict, obs_path_needed=False, 
                                                       do_generation_with_fit_parameters=do_generation_with_fit_parameters)

    DEBUG = inp_data_dict['DEBUG']
    file_prefix = inp_data_dict['file_prefix']
    resources_dir = inp_data_dict['resources_dir']
    # resources_dir = inp_data_dict['resources_dir']
    generated_models_dir = inp_data_dict['generated_models_dir']
    generated_models_subdir = inp_data_dict['generated_models_subdir']
    vessels_csv_abs_path = inp_data_dict['vessels_csv_abs_path']
    parameters_csv_abs_path = inp_data_dict['parameters_csv_abs_path']
    file_prefix_0d = inp_data_dict.get('file_prefix_0d')
    file_prefix_1d = inp_data_dict.get('file_prefix_1d')

    solver_info = inp_data_dict['solver_info']
    # Get solver from solver_info (check both 'solver' and 'method' for backward compatibility)
    solver = solver_info.get('solver') or solver_info.get('method')

    if do_generation_with_fit_parameters:
        param_id_output_dir_abs_path = inp_data_dict['param_id_output_dir_abs_path']
        # check if uncalibrated model is cellml2.0
        if inp_data_dict['model_type'] == 'cellml_only':
            uncalibrated_model_path = inp_data_dict['uncalibrated_model_path']
            if _is_cellml2_model_with_libcellml(uncalibrated_model_path):
                best_param_vals_path = os.path.join(param_id_output_dir_abs_path, 'best_param_vals.npy')
                param_names_path = os.path.join(param_id_output_dir_abs_path, 'param_names.csv')
                best_param_vals = np.load(best_param_vals_path)
                param_names = np.loadtxt(param_names_path, dtype=str)

                calibrated_cellml_path = os.path.join(
                    generated_models_subdir, f'{file_prefix}.cellml'
                )
                change_parameter_values_and_save(
                    uncalibrated_model_path,
                    param_names,
                    best_param_vals,
                    calibrated_cellml_path
                )
                inp_data_dict['model_path'] = calibrated_cellml_path
                return
        
        parser = CSV0DModelParser(inp_data_dict, parameter_id_dir=param_id_output_dir_abs_path)
    else:
        parser = CSV0DModelParser(inp_data_dict)

    model = parser.load_model()

    if DEBUG:
        print("Check point 0")
        print("\n")

    if inp_data_dict['model_type'] == 'cellml_only':
        code_generator = CVS0DCellMLGenerator(model, inp_data_dict)
        success = code_generator.generate_files()
    elif inp_data_dict['model_type'] in ['python', 'casadi_python']:
        # First generate the CellML model, then emit a Python module in the same directory.
        cellml_generator = CVS0DCellMLGenerator(model, inp_data_dict)
        success = cellml_generator.generate_files()
        if success:
            cellml_path = os.path.join(generated_models_subdir, f'{file_prefix}.cellml')
            py_gen = PythonGenerator(cellml_path, output_dir=generated_models_subdir, module_name=file_prefix)
            py_gen.generate()
            success = True
    elif inp_data_dict['model_type'] == 'cpp':
        solver_info = inp_data_dict['solver_info']
        solver_cpp = solver_info.get('solver', 'CVODE')
        print(f"Using Cpp solver: {solver_cpp}")
        dtSample = inp_data_dict['dt']
        dtSolver = solver_info['dt_solver']
        nMaxSteps = solver_info['MaximumNumberOfSteps']

        if inp_data_dict['couple_to_1d']:
            # object from class CVS0DCppGenerator
            if 'create_main_0d' in inp_data_dict:
                create_main_0d = inp_data_dict['create_main_0d']
            else:
                create_main_0d = False
            if 'model_1d_config_path' in inp_data_dict:
                model_1d_config_path = inp_data_dict['model_1d_config_path']
            else:
                if 'cpp_1d_model_config_path' in inp_data_dict:
                    model_1d_config_path = inp_data_dict['cpp_1d_model_config_path']
                else:
                    print("No model_1d_config_path or cpp_1d_model_config_path found in inp_data_dict")
                    exit()
            if 'cpp_generated_models_dir' in inp_data_dict:
                cpp_generated_models_dir = inp_data_dict['cpp_generated_models_dir']
            else:
                cpp_generated_models_dir = None

            if 'generate_1d' in inp_data_dict:
                generate_1d = inp_data_dict['generate_1d']
            else:
                generate_1d = False
            if not generate_1d:
                print("WARNING: 1D model input files and solver will not be generated. Check they already exist and they run properly.")
                print("     Otherwise, modify your choice in the user input yaml file.")
               
            if DEBUG:
                print("Check point 1A")

            code_generator = CVS0DCppGenerator(model, generated_models_subdir, file_prefix, #XXX
                                            resources_dir=resources_dir, solver=solver_cpp, 
                                            dtSample=dtSample, dtSolver=dtSolver, nMaxSteps=nMaxSteps,
                                            couple_to_1d=inp_data_dict['couple_to_1d'],
                                            cpp_generated_models_dir=cpp_generated_models_dir,
                                            model_1d_config_path=model_1d_config_path,
                                            create_main_0d=create_main_0d,
                                            conn_1d_0d_info=parser.conn_1d_0d_info,
                                            DEBUG=DEBUG)
            
            if DEBUG:
                print("Check point 2A")
            
            code1d_generator = None
            if generate_1d:
                vessels1d_csv_abs_path = inp_data_dict['vessels_csv_abs_path'] = os.path.join(inp_data_dict['resources_dir'], file_prefix_1d + '_vessel_array.csv')
                if 'solver_1d_type' in inp_data_dict:
                    if inp_data_dict['solver_1d_type'].startswith('py'):
                        code1d_generator = CVS1DPythonGenerator(model, file_prefix_1d, vessels1d_csv_abs_path, parameters_csv_abs_path,
                                                        model_1d_config_path, generated_models_subdir, 
                                                        cpp_generated_models_dir=cpp_generated_models_dir,
                                                        solver=solver_cpp, dtSample=dtSample, dtSolver=dtSolver, 
                                                        conn_1d_0d_info=parser.conn_1d_0d_info)
                    elif inp_data_dict['solver_1d_type'].startswith('cpp'):
                        #XXX This would use the class CVS1DCppGenerator() that Finbar was working on.
                        sys.exit('ERROR :: Class for generating Cpp files for 1D model not yet implemented. Please change your 1D solver type.')
                else:
                    sys.exit('ERROR :: solver_1d_type not found in user input yaml file. Please specify your 1D solver type.')

            if DEBUG:
                print("Check point 3A")
    
        else:
            # object from class CVS0DCppGenerator
            if DEBUG:
                print("Check point 1B")
            code_generator = CVS0DCppGenerator(model, generated_models_subdir, file_prefix,
                                            resources_dir=resources_dir, solver=solver_cpp,
                                            dtSample=dtSample, dtSolver=dtSolver, nMaxSteps=nMaxSteps)
            if DEBUG:
                print("Check point 2B")


        code_generator.generate_cellml(inp_data_dict)
        if DEBUG:
            print("Check point 3")
        code_generator.annotate_cellml()
        if DEBUG:
            print("Check point 4")
        success0d = code_generator.generate_cpp()
        if DEBUG:
            print(f"Check point with success (0d) {success0d}")
        
        success1d = True
        if inp_data_dict['couple_to_1d']:
            if generate_1d:
                success1d = code1d_generator.generate_files()
        if DEBUG:
            print(f"Check point with success (1d) {success1d}")

        if success0d and success1d:
            success = True
        else:
            success = False

    else: 
        print('model_type must be either cellml_only or cpp, not ' + inp_data_dict['model_type'])
        success = False
    
    return success


if __name__ == '__main__':
    if len(sys.argv) > 1:
        do_generation_with_fit_parameters = sys.argv[1] in ['true', 'True']
    else:
        do_generation_with_fit_parameters = False
    
    generate_with_new_architecture(do_generation_with_fit_parameters)
