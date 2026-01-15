'''
Created on 29/10/2021

@author: Finbar Argus, Gonzalo D. Maso Talou
'''

import numpy as np
import re
import pandas as pd
import os
from sys import exit
generators_dir = os.path.dirname(__file__)
base_dir = os.path.join(os.path.dirname(__file__), '../..')
LIBCELLML_available = True
try:
    from libcellml import Annotator, Analyser, AnalyserModel, AnalyserExternalVariable, Generator, GeneratorProfile        
    import utilities.libcellml_helper_funcs as cellml
    import utilities.libcellml_utilities as libcellml_utils
except ImportError as e:
    print("Error -> ", e)
    print('continuing without LibCellML, Warning code checks will not be available.'
          'You will need to open generated models in OpenCOR to check for errors.')
    LIBCELLML_available = False



class CVS0DCellMLGenerator(object):
    '''
    Generates CellML files for the 0D model represented in @
    '''

    def __init__(self, model, inp_data_dict):
        '''
        Constructor
        '''
        self._units_line_re = re.compile('[ \t]*<units .*/>')

        self.model = model
        self.output_dir = inp_data_dict['generated_models_subdir']
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.file_prefix = inp_data_dict['file_prefix']
        self.inp_data_dict = inp_data_dict

        if inp_data_dict['resources_dir'] is None:
            self.resources_dir = os.path.join(generators_dir, '../../resources')
        else:
            self.resources_dir = inp_data_dict['resources_dir']

        self.base_script = os.path.join(generators_dir, 'resources/base_script.cellml')


        self.module_scripts = [os.path.join(generators_dir, 'resources', filename) for filename in
                               os.listdir(os.path.join(generators_dir, 'resources'))
                               if filename.endswith('modules.cellml')]
        self.module_scripts += [os.path.join(base_dir, 'module_config_user', filename) for filename in
                               os.listdir(os.path.join(base_dir, 'module_config_user'))
                               if filename.endswith('modules.cellml')]
        if inp_data_dict['external_modules_dir'] is not None:
            self.module_scripts += [os.path.join(self.inp_data_dict['external_modules_dir'], filename) for filename in
                                os.listdir(os.path.join(self.inp_data_dict['external_modules_dir']))
                                if filename.endswith('modules.cellml')]
        self.units_scripts = [os.path.join(generators_dir, 'resources/units.cellml'),
                              os.path.join(base_dir, 'module_config_user/user_units.cellml')]
        self.all_parameters_defined = False
        self.BC_set = {}
        self.all_units = []

        # this is a temporary hack to include zero flow ivc if only one input to heart TODO make more robust
        self.ivc_connection_done = 0

    def generate_files(self):
        if type(self.model).__name__ != "CVS0DModel":
            print("Error: The model should be a CVS0DModel representation")
            return

        print("Generating model files at {}".format(self.output_dir))

        #    Code to generate model files
        self.__generate_units_file()
        self.__generate_CellML_file()
        if self.model.param_id_name_and_vals:
            self.__modify_parameters_array_from_param_id()
        self.__generate_parameters_csv()
        self.__generate_parameters_file()
        self.__generate_modules_file()

        # TODO check that model generation is successful, possibly by calling to opencor
        print('Model generation complete.')
        print('Checking Status of Model')

        if LIBCELLML_available:
            # parse the model in non-strict mode to allow non CellML 2.0 models
            model = cellml.parse_model(os.path.join(self.output_dir, self.file_prefix + '.cellml'), False)
            # resolve imports, in non-strict mode
            importer = cellml.resolve_imports(model, self.output_dir, False)
            # need a flattened model for analysing
            flat_model = cellml.flatten_model(model, importer)
            model_string = cellml.print_model(flat_model)
            
            # this if we want to create the flat model, for debugging
            with open(os.path.join(self.output_dir, self.file_prefix + '_flat.cellml'), 'w') as f:
                f.write(model_string)

            # analyse the model
            a = Analyser()

            a.analyseModel(flat_model)
            analysed_model = a.model()

            libcellml_utils.print_issues(a)
            print(analysed_model.type())
            # Debug: show how libCellML classifies variables (constant vs variable vs state, etc.)
            try:
                # Always dump for init-states reproducer, otherwise only in DEBUG.
                print_consts_etc = False
                if print_consts_etc or self.file_prefix == "test_init_states":
                    from libcellml import AnalyserVariable

                    type_names = {
                        AnalyserVariable.Type.STATE: "STATE",
                        AnalyserVariable.Type.CONSTANT: "CONSTANT",
                        AnalyserVariable.Type.COMPUTED_CONSTANT: "COMPUTED_CONSTANT",
                        AnalyserVariable.Type.ALGEBRAIC: "ALGEBRAIC",
                        AnalyserVariable.Type.EXTERNAL: "EXTERNAL",
                    }

                    buckets = {}
                    vcount = analysed_model.variableCount() if analysed_model else 0
                    for i in range(vcount):
                        av = analysed_model.variable(i)
                        if av is None:
                            continue
                        v = av.variable()
                        if v is None:
                            continue
                        try:
                            comp = v.parent().name()
                        except Exception:
                            comp = ""
                        key = type_names.get(av.type(), str(av.type()))
                        buckets.setdefault(key, []).append(f"{comp}.{v.name()}" if comp else v.name())

                    print("libCellML variable type summary:")
                    print(f"  analysed_model.variableCount(): {vcount}")
                    for key in sorted(buckets.keys()):
                        items = sorted(buckets[key])
                        print(f"  {key}: {len(items)}")
                        # Print a small sample to keep logs readable
                        for name in items[:20]:
                            print(f"    - {name}")
                        if len(items) > 20:
                            print(f"    ... (+{len(items) - 20} more)")
                    if vcount == 0 or not buckets:
                        # Fallback: dump variables declared in the flattened model so we still see names.
                        try:
                            fallback = []
                            for ci in range(flat_model.componentCount()):
                                comp = flat_model.component(ci)
                                cname = comp.name() if comp else ""
                                if not comp:
                                    continue
                                for vi in range(comp.variableCount()):
                                    var = comp.variable(vi)
                                    if not var:
                                        continue
                                    vname = var.name()
                                    fallback.append(f"{cname}.{vname}" if cname else vname)
                            fallback = sorted(set(fallback))
                            print("  (fallback) flattened model variables (sample):")
                            for name in fallback[:40]:
                                print(f"    - {name}")
                            if len(fallback) > 40:
                                print(f"    ... (+{len(fallback) - 40} more)")
                        except Exception as e:
                            print(f"  (fallback) could not list flattened variables: {e}")
            except Exception as e:
                print(f"Warning: could not dump libCellML variable types: {e}")
            if analysed_model.type() != AnalyserModel.Type.ODE:
                print("WARNING model is has some issues, see above. "
                    "The model might still run with some of the above issues"
                    "but it is recommended to fix them")
        
        print('Testing to see if model opens in OpenCOR')
        opencor_available = True
        try:
            import opencor as oc
        except:
            opencor_available = False
            pass
        if opencor_available:
            sim = oc.open_simulation(os.path.join(self.output_dir, f'{self.file_prefix}.cellml'))
            if sim.valid():
                print('Model generation has been successful.')
                return True
            else:
                if self.all_parameters_defined:
                    print('The OpenCOR model is not yet working, The reason for this is unknown. \n'
                          'Open the model in OpenCor and check the error in the simulation environment \n'
                          'for further error info. \n')
                    return False
                else:
                    print('The OpenCOR model is not yet working because all parameters have not been given values, \n'
                          f'Enter the values in '
                          f'{os.path.join(self.resources_dir, f"{self.file_prefix}_parameters_unfinished.csv")}')
                    return False

        else:
            print('Model generation is complete but OpenCOR could not be opened to test the model. \n'
                  ' We generate the model in python from cellml and check it can be run for a small \n'
                  'amount of time.')
            from generators.PythonGenerator import PythonGenerator
            from solver_wrappers.python_solver_helper import SimulationHelper as PythonSimulationHelper
            gen = PythonGenerator(os.path.join(self.output_dir, f'{self.file_prefix}.cellml'), output_dir=self.output_dir)
            gen.generate()
            sim_helper = PythonSimulationHelper(os.path.join(self.output_dir, f'{self.file_prefix}.py'), dt=0.00001, sim_time=0.00001)
            sim_helper.set_solve_ivp_method('BDF')
            success = sim_helper.run()

            if success:
                print('Model generation has been successful.')
                return True
            else:
                print('Model generation has failed. Or the simulation fails when trying to simulate'
                      'in Python')
                return False

    def __adjust_units_import_line(self, line):
        if 'import xlink:href="units.cellml"' in line:
            line = re.sub('units', f'{self.file_prefix}_units', line)
        return line

    def __generate_CellML_file(self):
        print("Generating CellML file {}.cellml".format(self.file_prefix))
        with open(self.base_script, 'r') as rf:
            with open(os.path.join(self.output_dir, f'{self.file_prefix}.cellml'), 'w') as wf:
                for line in rf:
                    line = self.__adjust_units_import_line(line)
                    if 'import xlink:href="parameters_autogen.cellml"' in line:
                        line = re.sub('parameters_autogen', f'{self.file_prefix}_parameters', line)

                    # copy the start of the basescript until line that says #STARTGENBELOW
                    wf.write(line)
                    if '#STARTGENBELOW' in line:
                        break

                # write units mapping
                print('writing units mapping')
                self.__write_section_break(wf, 'units')
                self.__write_units(wf)

                # import vessels
                print('writing imports')
                self.__write_section_break(wf, 'imports')
                self.__write_imports(wf, self.model.vessels_df)

                # define mapping between modules
                print('writing vessel mappings')
                self.__write_section_break(wf, 'vessel mappings')
                self.__write_module_mappings(wf, self.model.vessels_df)

                # TODO get the units from the indicidual variables that 
                # are being coupled. This is known in the module_config files.
                vol_units = 'm3'
                flow_units = 'm3_per_s'
                for param, unit in zip(self.model.parameters_array["variable_name"], self.model.parameters_array["units"]):
                    if param.startswith('q_'):
                        if unit=='litre':
                            vol_units = 'litre'
                            flow_units = 'L_per_s'
                            break
                        #XXX ADD MORE CASES IF NEEDED
                # print("Volume units: ",vol_units)
                # print("Flow units: ",flow_units)

                # create computation environment to sum flows from terminals
                # to have a total flow input into each first venous component.
                print('writing environment to sum venous input flows')
                self.__write_section_break(wf, 'terminal venous connection')
                self.__write_terminal_venous_connection_comp(wf, self.model.vessels_df, flow_units)

                # create the computation environment to sum flows from junctions 
                # to have the total flow input into the flow-port of each junction connection.
                print('writing environment to sum generic junctions input flows')
                self.__write_section_break(wf, 'generic junction connection')
                self.__write_generic_junction_connection_comp(wf, self.model.vessels_df, flow_units)

                # create computation environment to compute the total blood volume in the whole system 
                # or in specific portions of it.
                print('writing environment to apply operations for multiports')
                self.__write_section_break(wf, 'applying multiport operations for ports')
                self.__write_blood_volume_sum_comp(wf, self.model.vessels_df, vol_units)

                # define variables so they can be accessed
                print('writing variable access')
                self.__write_section_break(wf, 'access_variables')
                self.__write_access_variables(wf, self.model.vessels_df)
                
                # define global parameters so they can be accessed
                print('writing global params variable access')
                self.__write_section_break(wf, 'global_parameters_access')
                self.__write_global_parameters_access_variables(wf, self.model.parameters_array)

                # map between computational environment and module so they can be accessed
                print('writing mappings between computational environment and modules')
                self.__write_section_break(wf, 'own vessel mappings')
                self.__write_comp_to_module_mappings(wf, self.model.vessels_df)
                
                print('writing mappings between computational environment and modules for global parameters')
                self.__write_section_break(wf, 'own global parameters mapping')
                self.__write_global_parameters_comp_to_module_mappings(wf, self.model.parameters_array)

                # map constants to different modules
                print('writing mappings between constant params')
                self.__write_section_break(wf, 'parameters mapping to modules')
                self.__write_param_mappings(wf, self.model.vessels_df, params_array=self.model.parameters_array)

                # map environment time to module times
                print('writing writing time mappings between environment and modules')
                self.__write_section_break(wf, 'time mapping')
                self.__write_time_mappings(wf, self.model.vessels_df)

                # Finalise the file
                wf.write('</model>\n')

    def __generate_parameters_file(self):
        print("Generating CellML file {}_parameters.cellml".format(self.file_prefix))
        """
        Takes in a data frame of the params and generates the parameter_cellml file
        """

        with open(os.path.join(self.output_dir, f'{self.file_prefix}_parameters.cellml'), 'w') as wf:
            wf.write('<?xml version=\'1.0\' encoding=\'UTF-8\'?>\n')
            wf.write('<model name="Parameters" xmlns="http://www.cellml.org/cellml/1.1#'
                     '" xmlns:cellml="http://www.cellml.org/cellml/1.1#" xmlns:xlink="http://www.w3.org/1999/xlink">\n')

            self.__write_section_break(wf, 'parameter_units')
            self.__write_units(wf)

            global_params_array = self.model.parameters_array[np.where(self.model.parameters_array["const_type"] ==
                                                                       'global_constant')]
            module_params_array = self.model.parameters_array[np.where((self.model.parameters_array["const_type"] ==
                                                                        'constant') | (self.model.parameters_array["const_type"] ==
                                                                                       'boundary_condition'))]
            wf.write('<component name="parameters_global">\n')
            self.__write_constant_declarations(wf, global_params_array["variable_name"],
                                               global_params_array["units"],
                                               global_params_array["value"])
            wf.write('</component>\n')
            wf.write('<component name="parameters">\n')
            self.__write_constant_declarations(wf, module_params_array["variable_name"],
                                               module_params_array["units"],
                                               module_params_array["value"])
            wf.write('</component>\n')
            wf.write('</model>\n')

    def __modify_parameters_array_from_param_id(self):
        # first modify param_const names easily by modifying them in the array
        print('modifying constants to values identified from parameter id')
        for const_name, val in self.model.param_id_name_and_vals:
            self.model.parameters_array[np.where(self.model.parameters_array['variable_name'] ==
                                           const_name)[0][0]]['value'] = f'{val:.10e}'
            self.model.parameters_array[np.where(self.model.parameters_array['variable_name'] ==
                                                 const_name)[0][0]]['data_reference'] = \
                f'{self.model.param_id_date}_identified'

    def __generate_parameters_csv(self):
        # check if all the required parameters have been defined, if not we make an "unfinished"
        # csv file which makes it easy for the user to include the required parameters
        if self.all_parameters_defined:
            file_to_create = os.path.join(self.output_dir, f'{self.file_prefix}_parameters.csv')
        else:
            file_to_create = os.path.join(self.resources_dir,
                                          f'{self.file_prefix}_parameters_unfinished.csv')
            print(f'\n WARNING \nRequired parameters are missing. \nCreating a file {file_to_create},\n'
                  f'which has EMPTY_MUST_BE_FILLED tags where parameters\n'
                  f'need to be included. The user should include these parameters then remove \n'
                  f'the "_unfinished" ending of the file name, then rerun the model generation \n'
                  f'with the new parameters file as input.\n')
        df = pd.DataFrame(self.model.parameters_array)
        df = df.drop(columns=["const_type"])
        df.to_csv(file_to_create, index=None, header=True)

    def __generate_units_file(self):
        # TODO allow a specific units file to be generated
        #  This function simply copies the units file
        print(f'Generating CellML file {self.file_prefix}_units.cellml')
        with open(os.path.join(self.output_dir, f'{self.file_prefix}_units.cellml'), 'w') as wf:
            for file_idx, units_script in enumerate(self.units_scripts):
                with open(units_script, 'r') as rf:
                    for line_idx, line in enumerate(rf):
                        # if not first file then skip first two lines and last line
                        if file_idx == 0:
                            if line.startswith('</model>'):
                                # don't write the last line
                                continue
                        elif file_idx == len(self.units_scripts) - 1:
                            if line_idx == 0 or line_idx == 1:
                                # don't write the first two lines
                                continue
                        else:
                            if line_idx == 0 or line_idx == 1:
                                # don't write the first two lines
                                continue
                            if line.startswith('</model>'):
                                # don't write the last line
                                continue

                        wf.write(line)
                        if "units name" in line:
                            self.all_units.append(re.search('units name="(.*?)"', line).group(1))

    def __is_units_line(self, line):
        result = self._units_line_re.search(line)
        if result:
            return result.group(0)

        return None

    def __generate_modules_file(self):
        print(f'Generating modules file {self.file_prefix}_modules.cellml')
        with open(os.path.join(self.output_dir, f'{self.file_prefix}_modules.cellml'), 'w') as wf:
            # write first two lines
            wf.write("<?xml version='1.0' encoding='UTF-8'?>\n")
            wf.write(
                "<model name=\"modules\" xmlns=\"http://www.cellml.org/cellml/1.1#\" xmlns:cellml=\"http://www.cellml.org/cellml/1.1#\" "
                "xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n")
            # also write units in modules file
            self.__write_section_break(wf, 'module_units')
            self.__write_units(wf)

            for II, module_file_path in enumerate(self.module_scripts):
                with open(module_file_path, 'r') as rf:
                    # skip first lines that are either intro lines written above or comments.
                    lines = rf.readlines()
                    if not lines:
                        continue

                    count = 0
                    while True:
                        if "component" in lines[count]:
                            break
                        else:
                            count += 1
                            if count > 50:
                                print(f'Error in {module_file_path}, no component found')
                                exit()

                    # skip last line so enddef; doesn't get written till the end.
                    # skip first non-component lines
                    lines = lines[count:-1]

                    for line in lines:
                        if "<component name" in line:
                            # check the name of the module we are in
                            module_type = re.search('name="(.*?)"', line).group(1)
                        if module_type in self.model.vessels_df.module_type.values or module_type == 'zero_flow':
                            wf.write(line)
            wf.write('</model>\n')
                

    def __write_section_break(self, wf, text):
        wf.write('<!--&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;' +
                 text + '&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;//-->\n')

    def __write_units(self, wf):

        wf.writelines(f'<import xlink:href="{self.file_prefix}_units.cellml">\n')
        for unit in self.all_units:
            wf.writelines(f'    <units name="{unit}" units_ref="{unit}"/>\n')
        # units_list = []

        # for vessel_tup in vessel_df.itertuples():
        #     for variable in vessel_tup.variables_and_units:
        #         if variable[1] not in units_list:
        #             wf.writelines(f'    <units name="{variable[1]}" units_ref="{variable[1]}"/>\n')
        #             units_list.append(variable[1])

        # for param_tup in parameters_array:
        #     if param_tup[1] not in units_list:
        #         wf.writelines(f'    <units name="{param_tup[1]}" units_ref="{param_tup[1]}"/>\n')
        #         units_list.append(param_tup[1])

        wf.writelines('</import>\n')

    def __write_imports(self, wf, vessel_df):
        for vessel_tup in vessel_df.itertuples():
            if vessel_tup.module_format != 'cellml':
                # if not cellml then don't do anything for this vessel/module
                continue
            self.__write_import(wf, vessel_tup)

        # TODO change the below to vessel_type, not "name"
        if len(vessel_df.loc[vessel_df["name"] == 'heart']) == 1:
            # add a zero mapping to heart ivc or svc flow input if only one input is specified
            if "venous_ivc" not in vessel_df.loc[vessel_df["name"] == 'heart'].inp_vessels.values[0] or \
                    "venous_svc" not in vessel_df.loc[vessel_df["name"] == 'heart'].inp_vessels.values[0]:
                wf.writelines([f'<import xlink:href="{self.file_prefix}_modules.cellml">\n',
                               f'    <component component_ref="zero_flow" name="zero_flow_module"/>\n',
                               '</import>\n'])
            if "venous_ivc" not in vessel_df.loc[vessel_df["name"] == 'heart'].inp_vessels.values[0] and \
                    "venous_svc" not in vessel_df.loc[vessel_df["name"] == 'heart'].inp_vessels.values[0]:
                print('either venous_ivc, or venous_svc, or both must be inputs to the heart, exiting')
                exit()
        elif len(vessel_df.loc[vessel_df["name"] == 'heart']) < 1:
            pass
        elif len(vessel_df.loc[vessel_df["name"] == 'heart']) > 1:
            print('you have declared more that one heart module, exiting')
            exit()

    def __write_module_mappings(self, wf, module_df):
        """This function maps between ports of the modules in a module dataframe."""
        # set connected to false for all entrance and general ports TODO this might be a better way to check connected for exit ports too
        entrance_general_ports_connected = {}
        for module_row_idx in range(len(module_df)):
            if module_df.iloc[module_row_idx]["module_format"] != 'cellml':
                # if not cellml then don't do anything for this vessel/module
                continue
            entrance_general_ports_connected[module_df.iloc[module_row_idx]["name"]] = []
            for II in range(len(module_df.iloc[module_row_idx]["entrance_ports"])):
                # module_df.iloc[module_row_idx]["entrance_ports"][II]["connected"] = False
                entrance_general_ports_connected[module_df.iloc[module_row_idx]["name"]].append(False)
            for II in range(len(module_df.iloc[module_row_idx]["general_ports"])):
                entrance_general_ports_connected[module_df.iloc[module_row_idx]["name"]].append(False)

        # set BC_set to false for all boundary_condition variables
        for module_row_idx in range(len(module_df)):
            if module_df.iloc[module_row_idx]["module_format"] != 'cellml':
                # if not cellml then don't do anything for this vessel/module
                continue
            self.BC_set[module_df.iloc[module_row_idx]["name"]] = {}
            for II in range(len(module_df.iloc[module_row_idx]["variables_and_units"])):
                if module_df.iloc[module_row_idx]["variables_and_units"][II][3] == 'boundary_condition':
                    self.BC_set[module_df.iloc[module_row_idx]["name"]] \
                        [module_df.iloc[module_row_idx]["variables_and_units"][II][0]] = False

        # module_df.apply(self.__write_module_mapping_for_row, args=(module_df, wf), axis=1)
        # The above line is much faster but I'm worried about memory access of entrance_ports_connected
        for module_row_idx in range(len(module_df)):
            if module_df.iloc[module_row_idx]["module_format"] != 'cellml':
                # if not cellml then don't do anything for this vessel/module
                continue
            self.__write_module_mapping_for_row(module_df.iloc[module_row_idx], module_df,
                                                entrance_general_ports_connected, wf)  # entrance_ports_connected is a modified
            # in this function

        # check whether the BC variables have been set with a matched module, if so, remove them from 
        # parameters array and if not, set them to constant
        for module_row_idx in range(len(module_df)):
            if module_df.iloc[module_row_idx]["module_format"] != 'cellml':
                # if not cellml then don't do anything for this vessel/module
                continue
            indexes_to_remove = []
            for II in range(len(module_df.iloc[module_row_idx]["variables_and_units"])):
                if module_df.iloc[module_row_idx]["variables_and_units"][II][3] == 'boundary_condition':
                    full_variable_name = module_df.iloc[module_row_idx]["variables_and_units"][II][0] + \
                                         '_' + module_df.iloc[module_row_idx]["name"]
                    if not self.BC_set[module_df.iloc[module_row_idx]["name"]] \
                            [module_df.iloc[module_row_idx]["variables_and_units"][II][0]]:
                        module_df.iloc[module_row_idx]["variables_and_units"][II][3] = 'constant'
                        # self.model.parameters_array[np.where(self.model.parameters_array["variable_name"] == 
                        #                                      full_variable_name)][0][2] = 'constant'
                    else:
                        self.model.parameters_array = np.delete(self.model.parameters_array, np.where(self.model.parameters_array["variable_name"] ==
                                                                                                      full_variable_name))
                        indexes_to_remove.append(II)

            # remove the BC variables from the variables_and_units list
            # these variables will be accesible from the neighbouring
            # modules where they are calculated.
            for index in sorted(indexes_to_remove, reverse=True):
                del module_df.iloc[module_row_idx]["variables_and_units"][index]

        if np.any(self.model.parameters_array["value"] == 'EMPTY_MUST_BE_FILLED'):
            self.all_parameters_defined = False
        else:
            self.all_parameters_defined = True

    def __write_module_mapping_for_row(self, module_row, module_df, entrance_general_ports_connected, wf):
        """This function maps between ports of the modules in a one row of a module dataframe."""

        # input and output modules
        main_module = module_row["name"]
        main_module_BC_type = module_row["BC_type"]
        main_module_type = module_row["module_type"]

        # create a list of dicts that stores info for the exit ports of this main module
        # entrance_port_types is the corresponding connection check for entrance ports and is stored in 
        # this object so it can be obtained for each main module that connects to that out module
        port_types = []
        for out_port_idx in range(len(module_row["exit_ports"])):
            port_type = module_row["exit_ports"][out_port_idx]["port_type"]
            if port_type in [port_types[II]["port_type"] for II in range(len(port_types))]:
                port_type_idx = [port_types[II]["port_type"] for II
                                 in range(len(port_types))].index(port_type)
                port_types[port_type_idx]["out_port_idxs"].append(out_port_idx)
                port_types[port_type_idx]["port_type_count"] += 1
                port_types[port_type_idx]["connected"].append(False)

            else:
                port_types.append({"port_type": module_row["exit_ports"][out_port_idx]["port_type"],
                                   "out_port_idxs": [out_port_idx],
                                   "port_type_count": 1,
                                   "connected": [False]})  # whether this exit port has been connected
                
        for out_port_idx in range(len(module_row["general_ports"])):
            port_type = module_row["general_ports"][out_port_idx]["port_type"]
            if port_type in [port_types[II]["port_type"] for II in range(len(port_types))]:
                port_type_idx = [port_types[II]["port_type"] for II
                                 in range(len(port_types))].index(port_type)
                port_types[port_type_idx]["out_port_idxs"].append(out_port_idx+len(module_row["exit_ports"]))
                port_types[port_type_idx]["port_type_count"] += 1
                port_types[port_type_idx]["connected"].append(False)

            else:
                port_types.append({"port_type": module_row["general_ports"][out_port_idx]["port_type"],
                                   "out_port_idxs": [out_port_idx+len(module_row["exit_ports"])],
                                   "port_type_count": 1,
                                   "connected": [False]})
                
        # check inp vessels are all connected # TODO this should be done in the model parsing stage
        for inp_module_idx, inp_module in enumerate(module_row["inp_vessels"]):
            if not (module_df["name"] == inp_module).any():
                print(f'the input module of {inp_module} is not defined')
                exit()
            inp_module_row = module_df.loc[module_df["name"] == inp_module].squeeze()
            inp_module_BC_type = inp_module_row["BC_type"]
            inp_module_type = inp_module_row["module_type"]

            self.__check_input_output_modules(module_df, inp_module, main_module,
                                              inp_module_BC_type, main_module_BC_type,
                                              inp_module_type, main_module_type)

        for out_module_idx, out_module in enumerate(module_row["out_vessels"]):
            if not (module_df["name"] == out_module).any():
                print(f'the output module of {out_module} is not defined')
                exit()
            out_module_row = module_df.loc[module_df["name"] == out_module].squeeze()
            out_module_BC_type = out_module_row["BC_type"]
            out_module_type = out_module_row["module_type"]

            # check that input and output modules are defined as connection variables
            # for that module and they have corresponding BCs # TODO this should be done in the model parsing stage
            self.__check_input_output_modules(module_df, main_module, out_module,
                                              main_module_BC_type, out_module_BC_type,
                                              main_module_type, out_module_type)
            self.__check_input_output_ports(module_row["exit_ports"], module_row["general_ports"], out_module_row["entrance_ports"],
                                            out_module_row["general_ports"], main_module, out_module)

            # create a list of dicts that stores info for the entrance ports of this output module
            entrance_port_types = []
            for entrance_port_idx in range(len(out_module_row["entrance_ports"])):
                port_type = out_module_row["entrance_ports"][entrance_port_idx]["port_type"]
                if port_type in [entrance_port_types[II]["port_type"] for II in range(len(entrance_port_types))]:
                    port_type_idx = [entrance_port_types[II]["port_type"] for II
                                     in range(len(entrance_port_types))].index(port_type)
                    entrance_port_types[port_type_idx]["entrance_port_idxs"].append(entrance_port_idx)
                    entrance_port_types[port_type_idx]["port_type_count"] += 1
                else:
                    entrance_port_types.append({"port_type": out_module_row["entrance_ports"][entrance_port_idx]["port_type"],
                                                "entrance_port_idxs": [entrance_port_idx],
                                                "port_type_count": 1})
                    
            for entrance_port_idx in range(len(out_module_row["general_ports"])):
                port_type = out_module_row["general_ports"][entrance_port_idx]["port_type"]
                if port_type in [entrance_port_types[II]["port_type"] for II in range(len(entrance_port_types))]:
                    port_type_idx = [entrance_port_types[II]["port_type"] for II
                                     in range(len(entrance_port_types))].index(port_type)
                    entrance_port_types[port_type_idx]["entrance_port_idxs"].append(entrance_port_idx+len(out_module_row["entrance_ports"]))
                    entrance_port_types[port_type_idx]["port_type_count"] += 1
                else:
                    entrance_port_types.append({"port_type": out_module_row["general_ports"][entrance_port_idx]["port_type"],
                                                "entrance_port_idxs": [entrance_port_idx+len(out_module_row["entrance_ports"])],
                                                "port_type_count": 1})

            for port_type_idx in range(len(port_types)):
                entrance_port_type_idx = -1
                entrance_port_idx = -1
                for II in range(len(entrance_port_types)):
                    if entrance_port_types[II]["port_type"] == port_types[port_type_idx]["port_type"]:
                        entrance_port_type_idx = II
                        for JJ in entrance_port_types[II]["entrance_port_idxs"]:
                            if not entrance_general_ports_connected[out_module][JJ]:
                                entrance_port_idx = JJ
                                break
                        break
                if entrance_port_type_idx == -1:
                    # this port type isn't used, continue to next one
                    continue
                if entrance_port_idx == -1:
                    print(f"module {out_module} has a single port {port_types[port_type_idx]['port_type']}, connected",
                          f"to multiple ports. Should it have a multiport:True entry in the",
                          f"module_config.json file??")
                    exit()

                out_port_idx = None
                this_type_port_idx = None
                if port_types[port_type_idx]["port_type"] == "vessel_port":
                    for this_type_port_idx in range(port_types[port_type_idx]["port_type_count"]):
                        if not port_types[port_type_idx]["connected"][this_type_port_idx]:
                            out_port_idx = port_types[port_type_idx]["out_port_idxs"][this_type_port_idx]
                            break

                    # I dont think this block is needed
                    # # get the entrance port idx for the output module
                    # if entrance_port_types[entrance_port_type_idx]["port_type_count"] == 1:
                    #     entrance_port_idx = 0
                    # else:
                    #     entrance_port_idx = -1
                    #     for II in range(len(out_module_row["inp_vessels"])):
                    #         # check that the entrance port corresponds to the main_vessel
                    #         if out_module_row["inp_vessels"][II] == main_module:
                    #             entrance_port_idx = entrance_port_types[entrance_port_type_idx]["entrance_port_idxs"][II]
                    #             break

                    # TODO this part is kind of hacky, but it works, there is definitely a better way to do the mapping with the
                    #  heart module!
                    if out_module_type.startswith('heart'):
                        if len(out_module_row["inp_vessels"]) == 2 and self.ivc_connection_done == 0:
                            # this is the case if there is only one vc and one pulmonary
                            # We map the ivc to a zero flow mapping
                            self.__write_mapping(wf, 'zero_flow_module', 'heart_module', ['v_zero'], ['v_ivc'])
                            self.ivc_connection_done = 1
                            self.BC_set[out_module]['v_ivc'] = True
                            # TODO the above isnt robust

                        for heart_inp_idx in range(3):
                            # there are three vessel_port entrances to the heart, ivc, svc, and pulmonary
                            # in the vessel_array file, they must be ordered ivc, svc, pulmonary
                            if main_module == out_module_row["inp_vessels"][heart_inp_idx]:
                                # if the ivc connection was done artificially, then we need to skip it
                                entrance_port_idx = heart_inp_idx + self.ivc_connection_done
                                break

                    module_exit_general_ports = module_row["exit_ports"] + module_row["general_ports"]
                    out_module_entrance_general_ports = out_module_row["entrance_ports"] + out_module_row["general_ports"]


                    variables_1 = module_exit_general_ports[out_port_idx]['variables']
                    variables_2 = out_module_entrance_general_ports[entrance_port_idx]['variables']

                    if module_row["vessel_type"].endswith('terminal'):
                        # the terminal connections are done through the terminal_venous_connection
                        # TODO after this if loop I set that this connection is done. It is actually done later on
                        # when doing the venous_terminal_connection. Fine for now.
                        pass

                    elif module_row["vessel_type"].startswith(('Nout_', 'MinNout_')):
                        # the generic junction connections are done through the generic_junction_connection
                        pass

                    elif out_module_row["vessel_type"].startswith(('Min_', 'MinNout_')):
                        # the generic junction connections are done through the generic_junction_connection
                        pass

                    elif (any(module_df.loc[module_df["name"] == vess, "vessel_type"].iloc[0].startswith(("Min_", "MinNout_"))
                            for vess in module_row["out_vessels"])):
                        # the generic junction connections are done through the generic_junction_connection
                        pass

                    elif (any(
                            any(module_df.loc[module_df["name"] == temp_inp_vess, "vessel_type"].iloc[0].startswith(("Nout_", "MinNout_"))
                                for temp_inp_vess in module_df.loc[module_df["name"] == temp_out_vess, "inp_vessels"].values[0])
                                    for temp_out_vess in module_row["out_vessels"]
                                        if not module_df.loc[module_df["name"] == temp_out_vess, "BC_type"].iloc[0].startswith("nn"))):
                        # the generic junction connections are done through the generic_junction_connection
                        pass

                    else:
                        main_module_module = main_module + '_module'
                        out_module_module = out_module + '_module'

                        if module_row['module_format'] == 'cellml' and out_module_row['module_format'] == 'cellml':
                            self.__write_mapping(wf, main_module_module, out_module_module, variables_1, variables_2)

                    # only assign connected if the port doesnt have a multi_ports flag
                    if 'multi_port' in module_exit_general_ports[out_port_idx].keys():
                        if module_exit_general_ports[out_port_idx]['multi_port']:
                            pass
                        else:
                            port_types[port_type_idx]["connected"][this_type_port_idx] = True
                    else:
                        pass_port = False

                        #XXX THIS IS NEEDED!!!
                        for iV, temp_out_vess in enumerate(module_row["out_vessels"]):
                            temp_out_vess_BC_type = module_df.loc[module_df["name"] == temp_out_vess, "vessel_type"].iloc[0]
                            if temp_out_vess_BC_type.startswith("Min_") or temp_out_vess_BC_type.startswith("MinNout_"):
                                temp_out_module_row = module_df.loc[module_df["name"] == temp_out_vess].squeeze()
                                multi_port_found = False
                                for iP in range(len(temp_out_module_row["entrance_ports"])):
                                    if 'multi_port' in temp_out_module_row["entrance_ports"][iP].keys():
                                        if (temp_out_module_row["entrance_ports"][iP]["port_type"]=='vessel_port' 
                                            and temp_out_module_row["entrance_ports"][iP]['multi_port'] in ['True', True]):
                                            multi_port_found = True
                                            pass_port = True
                                            break
                                if multi_port_found:
                                    break
                        
                        #XXX TECHNICALLY, NOT NEEDED
                        # for iV, temp_inp_vess in enumerate(out_module_row["inp_vessels"]):
                        #     temp_inp_vess_BC_type = module_df.loc[module_df["name"] == temp_inp_vess, "vessel_type"].iloc[0]
                        #     if temp_inp_vess_BC_type.startswith("Nout_") or temp_inp_vess_BC_type.startswith("MinNout_"):
                        #         temp_inp_module_row = module_df.loc[module_df["name"] == temp_inp_vess].squeeze()
                        #         multi_port_found = False
                        #         for iP in range(len(temp_inp_module_row["exit_ports"])):
                        #             if 'multi_port' in temp_inp_module_row["exit_ports"][iP].keys():
                        #                 if (temp_inp_module_row["exit_ports"][iP]["port_type"]=='vessel_port' 
                        #                     and temp_inp_module_row["exit_ports"][iP]['multi_port'] in ['True', True]):
                        #                     multi_port_found = True
                        #                     pass_port = True
                        #                     break
                        #         if multi_port_found:
                        #             break 

                        if pass_port:
                            pass
                        else:
                            port_types[port_type_idx]["connected"][this_type_port_idx] = True


                    if 'multi_port' in out_module_row["entrance_ports"][entrance_port_idx].keys():
                        if out_module_entrance_general_ports[entrance_port_idx]['multi_port'] in ['True', True]:
                            pass
                        else:
                            entrance_general_ports_connected[out_module][entrance_port_idx] = True                  
                    else:
                        pass_port = False

                        #XXX TECHNICALLY, NOT NEEDED
                        # for iV, temp_out_vess in enumerate(module_row["out_vessels"]):
                        #     temp_out_vess_BC_type = module_df.loc[module_df["name"] == temp_out_vess, "vessel_type"].iloc[0]
                        #     if temp_out_vess_BC_type.startswith("Min_") or temp_out_vess_BC_type.startswith("MinNout_"):
                        #         temp_out_module_row = module_df.loc[module_df["name"] == temp_out_vess].squeeze()
                        #         multi_port_found = False
                        #         for iP in range(len(temp_out_module_row["entrance_ports"])):
                        #             if 'multi_port' in temp_out_module_row["entrance_ports"][iP].keys():
                        #                 if (temp_out_module_row["entrance_ports"][iP]["port_type"]=='vessel_port' 
                        #                     and temp_out_module_row["entrance_ports"][iP]['multi_port'] in ['True', True]):
                        #                     multi_port_found = True
                        #                     pass_port = True
                        #                     break
                        #         if multi_port_found:
                        #             break

                        #XXX THIS IS NEEDED!!!
                        for iV, temp_inp_vess in enumerate(out_module_row["inp_vessels"]):
                            temp_inp_vess_BC_type = module_df.loc[module_df["name"] == temp_inp_vess, "vessel_type"].iloc[0]
                            if temp_inp_vess_BC_type.startswith("Nout_") or temp_inp_vess_BC_type.startswith("MinNout_"):
                                temp_inp_module_row = module_df.loc[module_df["name"] == temp_inp_vess].squeeze()
                                multi_port_found = False
                                for iP in range(len(temp_inp_module_row["exit_ports"])):
                                    if 'multi_port' in temp_inp_module_row["exit_ports"][iP].keys():
                                        if (temp_inp_module_row["exit_ports"][iP]["port_type"]=='vessel_port' 
                                            and temp_inp_module_row["exit_ports"][iP]['multi_port']):
                                            multi_port_found = True
                                            pass_port = True
                                            break
                                if multi_port_found:
                                    break 

                        if pass_port:
                            pass
                        else:
                            entrance_general_ports_connected[out_module][entrance_port_idx] = True
          
                            
                    for II in range(len(variables_1)):
                        if variables_1[II] in self.BC_set[main_module].keys():
                            self.BC_set[main_module][variables_1[II]] = True
                        if variables_2[II] in self.BC_set[out_module].keys():
                            self.BC_set[out_module][variables_2[II]] = True
                
                else:
                    for this_type_port_idx in range(port_types[port_type_idx]["port_type_count"]):
                        if not port_types[port_type_idx]["connected"][this_type_port_idx]:
                            out_port_idx = port_types[port_type_idx]["out_port_idxs"][this_type_port_idx]
                            # get the entrance port idx for the output module

                            # TODO the below if isnt general. Figure out how to generalise this.
                            # if out_module_type == 'flow_sum_2_type':
                            #     for II in range(len(out_module_row["inp_vessels"])):
                            #         # check that the entrance port corresponds to the main_vessel
                            #         if out_module_row["inp_vessels"][II] == main_module:
                            #             entrance_port_idx = \
                            #                 entrance_port_types[entrance_port_type_idx]["entrance_port_idxs"][II]
                            #             break
                            # else:

                            module_exit_general_ports = module_row["exit_ports"] + module_row["general_ports"]
                            out_module_entrance_general_ports = out_module_row["entrance_ports"] + out_module_row["general_ports"]

                            variables_1 = module_exit_general_ports[out_port_idx]['variables']
                            variables_2 = out_module_entrance_general_ports[entrance_port_idx]['variables']

                            # TODO make this more general. Have a entry in module_config.json entrance and exit 
                            # types that specify a certain operation, like averaging wrt flow, like this gas port.
                            if main_module_type == 'tissue_GE_simple_type' and out_module_type == "gas_transport_simple_type":
                                # this connection is done through the terminal_venous_connection
                                # TODO after this if loop I set that this connection is done. It is actually done later on
                                # when doing the venous_terminal_connection. Fine for now.
                                pass

                            # Check if any port in out_module has multi_port='sum'
                            elif 'multi_port' in out_module_entrance_general_ports[entrance_port_idx].keys() and \
                                    out_module_entrance_general_ports[entrance_port_idx]['multi_port']=='sum':
                                # the blood volume sum connections are done through the blood_volume_sum
                                pass

                            else:
                                main_module_module = main_module + '_module'
                                out_module_module = out_module + '_module'

                                if module_row['module_format'] == 'cellml' and out_module_row['module_format'] == 'cellml':
                                    self.__write_mapping(wf, main_module_module, out_module_module, variables_1, variables_2)

                            for II in range(len(variables_1)):
                                if variables_1[II] in self.BC_set[main_module].keys():
                                    self.BC_set[main_module][variables_1[II]] = True
                                if variables_2[II] in self.BC_set[out_module].keys():
                                    self.BC_set[out_module][variables_2[II]] = True

                            # only assign connected if the port doesnt have a multi_ports flag
                            if 'multi_port' in module_exit_general_ports[out_port_idx].keys():
                                if module_exit_general_ports[out_port_idx]['multi_port']:
                                    pass
                                else:
                                    port_types[port_type_idx]["connected"][this_type_port_idx] = True
                            else:
                                port_types[port_type_idx]["connected"][this_type_port_idx] = True

                            if 'multi_port' in out_module_entrance_general_ports[entrance_port_idx].keys():
                                if out_module_entrance_general_ports[entrance_port_idx]['multi_port']:
                                    pass
                                else:
                                    entrance_general_ports_connected[out_module][entrance_port_idx] = True
                            else:
                                entrance_general_ports_connected[out_module][entrance_port_idx] = True

                            # if only_one_port is set to true, then break from this for loop, as this is the only port that should be connected between 
                            # these modules
                            if 'only_one_port' in out_module_entrance_general_ports[entrance_port_idx].keys():
                                if out_module_entrance_general_ports[entrance_port_idx]['only_one_port'] in ['True', True]:
                                    break_out = True
                                    break
                            break_out = False
                            
                    if break_out:
                        break_out = False
                        break     

        return entrance_general_ports_connected

    def __write_terminal_venous_connection_comp(self, wf, vessel_df, flow_units='m3_per_s'):
        first_venous_names = []  # stores name of venous compartments that take flow from terminals
        tissue_GE_names = []  # stores name of venous compartments that take flow from terminals
        for vessel_tup in vessel_df.itertuples():
            if vessel_tup.module_format != 'cellml':
                # if not cellml then don't do anything for this vessel/module
                continue
            if vessel_tup.vessel_type.endswith('terminal'):
                vessel_name = vessel_tup.name

                out_vessel_names = []
                for out_vessel_name in vessel_tup.out_vessels:
                    if vessel_df.loc[vessel_df["name"] == out_vessel_name].squeeze()["vessel_type"].startswith('venous'):
                        # This finds the venous module connection
                        out_vessel_names.append(out_vessel_name)
                if len(out_vessel_names) == 0:
                    # there is no venous compartment connected to this terminal but still create a terminal venous section
                    pass
                else:
                    # map pressure between terminal and first venous compartments
                    for out_vessel_idx in range(len(out_vessel_names)):
                        u_1 = 'u_out'
                        u_2 = 'u'
                        self.__write_mapping(wf, vessel_name +'_module', out_vessel_names[out_vessel_idx] + '_module',
                                            [u_1], [u_2])

                # then map variables between connection and the venous sections
                v_1 = 'v_T'
                v_2 = f'v_{vessel_name}'

                self.__write_mapping(wf, vessel_name + '_module', 'terminal_venous_connection',
                                     [v_1], [v_2])

            # This only allowed venous types to be called venous.
            # # check that vessel type is venous but and that it is the first venous thats connected to a terminal
            # if vessel_tup.vessel_type == 'venous' and \
            #         vessel_df.loc[vessel_df['name'].isin(vessel_tup.inp_vessels)
            #         ]['vessel_type'].str.contains('terminal').any():

            # check if the vessel has a terminal as an input and has a flow input
            if vessel_df.loc[vessel_df['name'].isin(vessel_tup.inp_vessels)
            ]['vessel_type'].str.contains('terminal').any() and vessel_tup.BC_type.startswith('v'):
                vessel_name = vessel_tup.name
                first_venous_names.append(vessel_name)
                v_1 = [f'v_{vessel_name}']
                v_2 = ['v_in']

                self.__write_mapping(wf, 'terminal_venous_connection', vessel_name + '_module',
                                     v_1, v_2)

            if vessel_tup.vessel_type == 'gas_transport_simple' and \
                    vessel_df.loc[vessel_df['name'].isin(vessel_tup.inp_vessels)
                    ]['vessel_type'].str.contains('tissue_GE_simple').any():
                vessel_name = vessel_tup.name

                vars_1 = ['v_venous_total', f'C_O2_p_venous_ave', f'C_CO2_p_venous_ave']
                vars_2 = ['v', 'C_O2_in', 'C_CO2_in']
                self.__write_mapping(wf, 'terminal_venous_connection', vessel_name + '_module',
                                     vars_1, vars_2)

            if vessel_tup.vessel_type == 'tissue_GE_simple':
                vessel_name = vessel_tup.name
                tissue_GE_names.append(vessel_name)

                C_1 = [f'C_O2_p_{vessel_name}', f'C_CO2_p_{vessel_name}']
                C_2 = ['C_O2_p', 'C_CO2_p']

                self.__write_mapping(wf, 'terminal_venous_connection', vessel_name + '_module',
                                     C_1, C_2)

        # loop through vessels to get the terminals to add up for each first venous section
        terminal_names_for_first_venous = [[] for i in range(len(first_venous_names))]
        terminal_names_with_GE = []
        terminal_names = []
        for vessel_tup in vessel_df.itertuples():
            if vessel_tup.vessel_type.endswith('terminal'):
                vessel_name = vessel_tup.name
                for idx, venous_name in enumerate(first_venous_names):
                    for II in range(len(vessel_tup.out_vessels)):
                        if vessel_tup.out_vessels[II] == venous_name:
                            if vessel_name not in vessel_df.loc[vessel_df["name"] == venous_name].squeeze()["inp_vessels"]:
                                print(f'venous input of {venous_name} does not include the terminal vessel '
                                      f'{vessel_name} as an inp_vessel in {self.file_prefix}_vessel_array. '
                                      f'not including terminal names as input has been deprecated')
                                exit()
                            terminal_names_for_first_venous[idx].append(vessel_name)
                for idx, tissue_GE_name in enumerate(tissue_GE_names):
                    for II in range(len(vessel_tup.out_vessels)):
                        if vessel_tup.out_vessels[II] == tissue_GE_name:
                            terminal_names_with_GE.append(vessel_name)
                terminal_names.append(vessel_name)

        # create computation environment for connection and write the
        # variable definition and calculation of flow to each first venous module.
        wf.write(f'<component name="terminal_venous_connection">\n')
        variables = []
        units = []
        in_outs = []

        for terminal_name in terminal_names:
            variables.append(f'v_{terminal_name}')
            units.append(flow_units) # ('m3_per_s')
            in_outs.append('in')

        for idx, venous_name in enumerate(first_venous_names):
            variables.append(f'v_{venous_name}')
            units.append(flow_units) # ('m3_per_s')
            # TODO check this for full CVS case
            # in_outs.append('priv_in_pub_out')
            in_outs.append('out')

        for GE_name in tissue_GE_names:
            variables.append(f'C_O2_p_{GE_name}')
            variables.append(f'C_CO2_p_{GE_name}')
            units.append('dimensionless')
            units.append('dimensionless')
            in_outs.append('in')
            in_outs.append('in')

        if len(tissue_GE_names) > 0:
            variables.append(f'C_O2_p_venous_ave')
            variables.append(f'C_CO2_p_venous_ave')
            units.append('dimensionless')
            units.append('dimensionless')
            in_outs.append('out')
            in_outs.append('out')

            variables.append(f'v_venous_total')
            units.append(flow_units) # ('m3_per_s')
            in_outs.append('out')

        self.__write_variable_declarations(wf, variables, units, in_outs)
        for idx, venous_name in enumerate(first_venous_names):
            rhs_variables = []
            lhs_variable = f'v_{venous_name}'
            for terminal_name in terminal_names_for_first_venous[idx]:
                rhs_variables.append(f'v_{terminal_name}')

            self.__write_variable_sum(wf, lhs_variable, rhs_variables)

        if len(tissue_GE_names) > 0:

            rhs_variables_to_average = []
            rhs_variables_weightings = []
            rhs_variables_to_average_CO2 = []
            # sum all venous components to get a total flow to use for gas transport
            lhs_variable_flow = 'v_venous_total'
            lhs_variable = f'C_O2_p_venous_ave'
            lhs_variable_CO2 = f'C_CO2_p_venous_ave'
            for terminal_name, GE_name in zip(terminal_names_with_GE, tissue_GE_names):
                rhs_variables_to_average.append(f'C_O2_p_{GE_name}')
                rhs_variables_to_average_CO2.append(f'C_CO2_p_{GE_name}')
                rhs_variables_weightings.append(f'v_{terminal_name}')

            self.__write_variable_sum(wf, lhs_variable_flow, rhs_variables_weightings)
            self.__write_variable_average(wf, lhs_variable, rhs_variables_to_average, rhs_variables_weightings)
            self.__write_variable_average(wf, lhs_variable_CO2, rhs_variables_to_average_CO2, rhs_variables_weightings)

        wf.write('</component>\n')


    def __write_generic_junction_connection_comp(self, wf, vessel_df, flow_units='m3_per_s'):
        # this function creates the computation environment to sum flows from junctions 
        # to have the total flow input into the flow-port of each junction connection
        # as done for terminal venous connections
        
        flow_vess_names = []
        flow_vess_types = []
        vess_names_per_junc = []
        vess_signs_per_junc = []
        vess_bcs_per_junc = []

        for vessel_tup in vessel_df.itertuples():
            if vessel_tup.module_format != 'cellml':
                # if not cellml then don't do anything for this vessel/module
                continue

            # if vessel_tup.vessel_type.endswith('Min_junction'):
            if 'Min_junction' in vessel_tup.vessel_type:
                # print("Min_junction vessel found")
                vess_name = vessel_tup.name
                if vessel_tup.BC_type.startswith('vv'):
                    vess_BC = 'vv'
                elif vessel_tup.BC_type.startswith('vp'):
                    vess_BC = 'vp'
                else:
                    print(f'ERROR :: Min_junction {vess_name} has wrong BC_type {vessel_tup.BC_type}. '
                          f'Exiting')
                    exit()

                # print(vess_name, vess_BC)

                in_vessel_names = []
                in_vessel_BCs = []
                in_vessel_signs = []
                for in_vess_name in vessel_tup.inp_vessels:
                    # This finds the vessels connected to the same junction
                    in_vess_BC = vessel_df.loc[vessel_df["name"] == in_vess_name].squeeze()["BC_type"][:2]
                    if in_vess_BC!='nn': # otherwise K_tube modules or other non-vessel modules will also be included
                        in_vessel_names.append(in_vess_name)
                        in_vessel_BCs.append(in_vess_BC)
                        for vessel_tup2 in vessel_df.itertuples():
                            if vessel_tup2.name == in_vess_name:
                                if vess_name in vessel_tup2.inp_vessels:
                                    in_vess_sign = -1.
                                    in_vessel_signs.append(in_vess_sign)
                                    break
                                elif vess_name in vessel_tup2.out_vessels:
                                    in_vess_sign = 1.
                                    in_vessel_signs.append(in_vess_sign)
                                    break
  
                for vessel_tup2 in vessel_df.itertuples():
                    vess_name2 = vessel_tup2.name
                    if vess_name2 in in_vessel_names and vess_name in vessel_tup2.inp_vessels:
                        for in_vess_name in vessel_tup2.inp_vessels:
                            if in_vess_name!=vess_name and in_vess_name not in in_vessel_names:
                                in_vess_BC = vessel_df.loc[vessel_df["name"] == in_vess_name].squeeze()["BC_type"][:2]
                                if in_vess_BC!='nn': # otherwise K_tube modules or other non-vessel modules will also be included
                                    in_vessel_names.append(in_vess_name)
                                    in_vessel_BCs.append(in_vess_BC)
                                    in_vess_sign = -1.
                                    in_vessel_signs.append(in_vess_sign)
                                    # break   
                    elif vess_name2 in in_vessel_names and vess_name in vessel_tup2.out_vessels:
                        for in_vess_name in vessel_tup2.out_vessels:
                            if in_vess_name!=vess_name and in_vess_name not in in_vessel_names:
                                in_vess_BC = vessel_df.loc[vessel_df["name"] == in_vess_name].squeeze()["BC_type"][:2]
                                if in_vess_BC!='nn': # otherwise K_tube modules or other non-vessel modules will also be included
                                    in_vessel_names.append(in_vess_name)
                                    in_vessel_BCs.append(in_vess_BC)
                                    in_vess_sign = -1.
                                    in_vessel_signs.append(in_vess_sign)
                                    # break

                # print(in_vessel_names)
                # print(in_vessel_BCs)
                # print(in_vessel_signs)

                vess_names_per_junc.append(in_vessel_names)
                vess_signs_per_junc.append(in_vessel_signs)
                vess_bcs_per_junc.append(in_vessel_BCs)

                if len(in_vessel_names) == 0:
                    print(f'ERROR :: Min_junction {vess_name} has NO other vessels connected to its inlet node, '
                          f'even if it is a junction node. Exiting')
                    exit()
                else:
                    for k in range(len(in_vessel_names)):
                        # map pressure between current Min_junction vessel and each other vessel converging to the junction 
                        # to ensure continuity of pressure
                        # and then map flow between each vessel and the junction connection (to make the sum of flows later on)
                        # to ensure conservation of mass
                        if in_vessel_BCs[k] == 'pv': 
                            u_2 = 'u_in'
                            v_1 = 'v'
                        elif in_vessel_BCs[k] == 'vp': 
                            u_2 = 'u_out'
                            v_1 = 'v'
                        elif in_vessel_BCs[k] == 'pp': 
                            if in_vessel_signs[k] == -1.:
                                u_2 = 'u_in'
                                v_1 = 'v'
                            elif in_vessel_signs[k] == 1.:
                                u_2 = 'u_out'
                                v_1 = 'v_d'
                        
                        u_1 = 'u'
                        
                        if v_1=='v_d':
                            v_2 = f'v_d_{in_vessel_names[k]}_Min'
                        else:
                            v_2 = f'v_{in_vessel_names[k]}_Min'
                        
                        self.__write_mapping(wf, vess_name+'_module', in_vessel_names[k]+'_module', [u_1], [u_2])

                        self.__write_mapping(wf, in_vessel_names[k]+'_module', 'generic_junction_connection', [v_1], [v_2])

                        if v_1=='v_d':
                            temp_in_vess_name = in_vessel_names[k]
                            vess_names_per_junc[-1][k] = 'd_'+temp_in_vess_name
                    
            # elif vessel_tup.vessel_type.endswith('Nout_junction')
            elif 'Nout_junction' in vessel_tup.vessel_type:
                
                # if vessel_tup.vessel_type.endswith('MinNout_junction'):
                if 'MinNout_junction' in vessel_tup.vessel_type:
                    # print("MinNout_junction vessel found")
                    vess_name = vessel_tup.name
                    if vessel_tup.BC_type.startswith('vv'):
                        vess_BC = 'vv'
                    else:
                        print(f'ERROR :: MinNout_junction {vess_name} has wrong BC_type {vessel_tup.BC_type}. '
                            f'Exiting')
                        exit()

                    # print(vess_name, vess_BC)

                    in_vessel_names = []
                    in_vessel_BCs = []
                    in_vessel_signs = []
                    out_vessel_names = []
                    out_vessel_BCs = []
                    out_vessel_signs = []

                    for in_vess_name in vessel_tup.inp_vessels:
                        # This finds the vessels connected to the same junction
                        in_vess_BC = vessel_df.loc[vessel_df["name"] == in_vess_name].squeeze()["BC_type"][:2]
                        if in_vess_BC!='nn': # otherwise K_tube modules or other non-vessel modules will also be included
                            in_vessel_names.append(in_vess_name)
                            in_vessel_BCs.append(in_vess_BC)
                            for vessel_tup2 in vessel_df.itertuples():
                                if vessel_tup2.name == in_vess_name:
                                    if vess_name in vessel_tup2.inp_vessels:
                                        in_vess_sign = -1.
                                        in_vessel_signs.append(in_vess_sign)
                                        break
                                    elif vess_name in vessel_tup2.out_vessels:
                                        in_vess_sign = 1.
                                        in_vessel_signs.append(in_vess_sign)
                                        break

                    for vessel_tup2 in vessel_df.itertuples():
                        vess_name2 = vessel_tup2.name
                        if vess_name2 in in_vessel_names and vess_name in vessel_tup2.inp_vessels:
                            for in_vess_name in vessel_tup2.inp_vessels:
                                if in_vess_name!=vess_name and in_vess_name not in in_vessel_names:
                                    in_vess_BC = vessel_df.loc[vessel_df["name"] == in_vess_name].squeeze()["BC_type"][:2]
                                    if in_vess_BC!='nn': # otherwise K_tube modules or other non-vessel modules will also be included
                                        in_vessel_names.append(in_vess_name)
                                        in_vessel_BCs.append(in_vess_BC)
                                        in_vess_sign = -1.
                                        in_vessel_signs.append(in_vess_sign)
                                        # break   
                        elif vess_name2 in in_vessel_names and vess_name in vessel_tup2.out_vessels:
                            for in_vess_name in vessel_tup2.out_vessels:
                                if in_vess_name!=vess_name and in_vess_name not in in_vessel_names:
                                    in_vess_BC = vessel_df.loc[vessel_df["name"] == in_vess_name].squeeze()["BC_type"][:2]
                                    if in_vess_BC!='nn': # otherwise K_tube modules or other non-vessel modules will also be included
                                        in_vessel_names.append(in_vess_name)
                                        in_vessel_BCs.append(in_vess_BC)
                                        in_vess_sign = -1.
                                        in_vessel_signs.append(in_vess_sign)
                                        # break

                    # print(in_vessel_names)
                    # print(in_vessel_BCs)
                    # print(in_vessel_signs)
                    
                    vess_names_per_junc.append(in_vessel_names)
                    vess_signs_per_junc.append(in_vessel_signs)
                    vess_bcs_per_junc.append(in_vessel_BCs)
                    
                    if len(in_vessel_names) == 0:
                        print(f'ERROR :: MinNout_junction {vess_name} has NO other vessels connected to it to its inlet node, '
                            f'even if it is a junction node. Exiting')
                        exit()
                    else:
                        for k in range(len(in_vessel_names)):
                            # map pressure between current Min_junction vessel and each other vessel converging to the junction 
                            # to ensure continuity of pressure
                            # and then map flow between each vessel and the junction connection (to make the sum of flows later on)
                            # to ensure conservation of mass
                            if in_vessel_BCs[k] == 'pv': 
                                u_2 = 'u_in'
                                v_1 = 'v'
                            elif in_vessel_BCs[k] == 'vp': 
                                u_2 = 'u_out'
                                v_1 = 'v'
                            elif in_vessel_BCs[k] == 'pp': 
                                if in_vessel_signs[k] == -1.:
                                    u_2 = 'u_in'
                                    v_1 = 'v'
                                elif in_vessel_signs[k] == 1.:
                                    u_2 = 'u_out'
                                    v_1 = 'v_d'
                            
                            u_1 = 'u'

                            if v_1=='v_d':
                                v_2 = f'v_d_{in_vessel_names[k]}_Min'
                            else:
                                v_2 = f'v_{in_vessel_names[k]}_Min'
                            
                            self.__write_mapping(wf, vess_name+'_module', in_vessel_names[k]+'_module', [u_1], [u_2])

                            self.__write_mapping(wf, in_vessel_names[k]+'_module', 'generic_junction_connection', [v_1], [v_2])

                            if v_1=='v_d':
                                temp_in_vess_name = in_vessel_names[k]
                                vess_names_per_junc[-1][k] = 'd_'+temp_in_vess_name

                    for out_vess_name in vessel_tup.out_vessels:
                        # This finds the vessels connected to the same junction
                        out_vess_BC = vessel_df.loc[vessel_df["name"] == out_vess_name].squeeze()["BC_type"][:2]
                        if out_vess_BC!='nn': # otherwise K_tube modules or other non-vessel modules will also be included
                            out_vessel_names.append(out_vess_name)
                            out_vessel_BCs.append(out_vess_BC)
                            for vessel_tup2 in vessel_df.itertuples():
                                if vessel_tup2.name == out_vess_name:
                                    if vess_name in vessel_tup2.inp_vessels:
                                        out_vess_sign = -1.
                                        out_vessel_signs.append(out_vess_sign)
                                        break
                                    elif vess_name in vessel_tup2.out_vessels:
                                        out_vess_sign = 1.
                                        out_vessel_signs.append(out_vess_sign)
                                        break

                    for vessel_tup2 in vessel_df.itertuples():
                        vess_name2 = vessel_tup2.name
                        if vess_name2 in out_vessel_names and vess_name in vessel_tup2.inp_vessels:
                            for out_vess_name in vessel_tup2.inp_vessels:
                                if out_vess_name!=vess_name and out_vess_name not in out_vessel_names:
                                    out_vess_BC = vessel_df.loc[vessel_df["name"] == out_vess_name].squeeze()["BC_type"][:2]
                                    if out_vess_BC!='nn': # otherwise K_tube modules or other non-vessel modules will also be included
                                        out_vessel_names.append(out_vess_name)
                                        out_vessel_BCs.append(out_vess_BC)
                                        out_vess_sign = 1.
                                        out_vessel_signs.append(out_vess_sign)
                                        # break   
                        elif vess_name2 in out_vessel_names and vess_name in vessel_tup2.out_vessels:
                            for out_vess_name in vessel_tup2.out_vessels:
                                if out_vess_name!=vess_name and out_vess_name not in out_vessel_names:
                                    out_vess_BC = vessel_df.loc[vessel_df["name"] == out_vess_name].squeeze()["BC_type"][:2]
                                    if out_vess_BC!='nn': # otherwise K_tube modules or other non-vessel modules will also be included
                                        out_vessel_names.append(out_vess_name)
                                        out_vessel_BCs.append(out_vess_BC)
                                        out_vess_sign = 1.
                                        out_vessel_signs.append(out_vess_sign)
                                        # break

                    # print(out_vessel_names)
                    # print(out_vessel_BCs)
                    # print(out_vessel_signs)
                    
                    vess_names_per_junc.append(out_vessel_names)
                    vess_signs_per_junc.append(out_vessel_signs)
                    vess_bcs_per_junc.append(out_vessel_BCs)

                    if len(out_vessel_names) == 0:
                        print(f'ERROR :: MinNout_junction {vess_name} has NO other vessels connected to its outlet node, '
                            f'even if it is a junction node. Exiting')
                        exit()
                    else:
                        for k in range(len(out_vessel_names)):
                            # map pressure between current Min_junction vessel and each other vessel converging to the junction 
                            # to ensure continuity of pressure
                            # and then map flow between each vessel and the junction connection (to make the sum of flows later on)
                            # to ensure conservation of mass
                            if out_vessel_BCs[k] == 'pv': 
                                u_2 = 'u_in'
                                v_1 = 'v'
                            elif out_vessel_BCs[k] == 'vp': 
                                u_2 = 'u_out'
                                v_1 = 'v'
                            elif out_vessel_BCs[k] == 'pp': 
                                if out_vessel_signs[k] == -1.:
                                    u_2 = 'u_in'
                                    v_1 = 'v'
                                elif out_vessel_signs[k] == 1.:
                                    u_2 = 'u_out'
                                    v_1 = 'v_d'
                            
                            u_1 = 'u_d'

                            if v_1=='v_d':
                                v_2 = f'v_d_{out_vessel_names[k]}_Nout'
                            else:
                                v_2 = f'v_{out_vessel_names[k]}_Nout'
                            
                            self.__write_mapping(wf, vess_name+'_module', out_vessel_names[k]+'_module', [u_1], [u_2])

                            self.__write_mapping(wf, out_vessel_names[k]+'_module', 'generic_junction_connection', [v_1], [v_2])

                            if v_1=='v_d':
                                temp_out_vess_name = out_vessel_names[k]
                                vess_names_per_junc[-1][k] = 'd_'+temp_out_vess_name
                
                else:
                    # print("Nout_junction vessel found")
                    vess_name = vessel_tup.name
                    if vessel_tup.BC_type.startswith('vv'):
                        vess_BC = 'vv'
                    elif vessel_tup.BC_type.startswith('pv'):
                        vess_BC = 'pv'
                    else:
                        print(f'ERROR :: Nout_junction {vess_name} has wrong BC_type {vessel_tup.BC_type}. '
                            f'Exiting')
                        exit()

                    # print(vess_name, vess_BC)

                    out_vessel_names = []
                    out_vessel_BCs = []
                    out_vessel_signs = []
                    for out_vess_name in vessel_tup.out_vessels:
                        # This finds the vessels connected to the same junction
                        out_vess_BC = vessel_df.loc[vessel_df["name"] == out_vess_name].squeeze()["BC_type"][:2]
                        if out_vess_BC!='nn': # otherwise K_tube modules or other non-vessel modules will also be included
                            out_vessel_names.append(out_vess_name)
                            out_vessel_BCs.append(out_vess_BC)
                            for vessel_tup2 in vessel_df.itertuples():
                                if vessel_tup2.name == out_vess_name:
                                    if vess_name in vessel_tup2.inp_vessels:
                                        out_vess_sign = -1.
                                        out_vessel_signs.append(out_vess_sign)
                                        break
                                    elif vess_name in vessel_tup2.out_vessels:
                                        out_vess_sign = 1.
                                        out_vessel_signs.append(out_vess_sign)
                                        break

                    for vessel_tup2 in vessel_df.itertuples():
                        vess_name2 = vessel_tup2.name
                        if vess_name2 in out_vessel_names and vess_name in vessel_tup2.inp_vessels:
                            for out_vess_name in vessel_tup2.inp_vessels:
                                if out_vess_name!=vess_name and out_vess_name not in out_vessel_names:
                                    out_vess_BC = vessel_df.loc[vessel_df["name"] == out_vess_name].squeeze()["BC_type"][:2]
                                    if out_vess_BC!='nn': # otherwise K_tube modules or other non-vessel modules will also be included
                                        out_vessel_names.append(out_vess_name)
                                        out_vessel_BCs.append(out_vess_BC)
                                        out_vess_sign = 1.
                                        out_vessel_signs.append(out_vess_sign)
                                        # break   
                        elif vess_name2 in out_vessel_names and vess_name in vessel_tup2.out_vessels:
                            for out_vess_name in vessel_tup2.out_vessels:
                                if out_vess_name!=vess_name and out_vess_name not in out_vessel_names:
                                    out_vess_BC = vessel_df.loc[vessel_df["name"] == out_vess_name].squeeze()["BC_type"][:2]
                                    if out_vess_BC!='nn': # otherwise K_tube modules or other non-vessel modules will also be included
                                        out_vessel_names.append(out_vess_name)
                                        out_vessel_BCs.append(out_vess_BC)
                                        out_vess_sign = 1.
                                        out_vessel_signs.append(out_vess_sign)
                                        # break

                    # print(out_vessel_names)
                    # print(out_vessel_BCs)
                    # print(out_vessel_signs)

                    vess_names_per_junc.append(out_vessel_names)
                    vess_signs_per_junc.append(out_vessel_signs)
                    vess_bcs_per_junc.append(out_vessel_BCs)

                    if len(out_vessel_names) == 0:
                        print(f'ERROR :: Nout_junction {vess_name} has NO other vessels connected to its outlet node, '
                            f'even if it is a junction node. Exiting')
                        exit()
                    else:
                        for k in range(len(out_vessel_names)):
                            # map pressure between current Min_junction vessel and each other vessel converging to the junction 
                            # to ensure continuity of pressure
                            # and then map flow between each vessel and the junction connection (to make the sum of flows later on)
                            # to ensure conservation of mass
                            if out_vessel_BCs[k] == 'pv': 
                                u_2 = 'u_in'
                                v_1 = 'v'
                            elif out_vessel_BCs[k] == 'vp': 
                                u_2 = 'u_out'
                                v_1 = 'v'
                            elif out_vessel_BCs[k] == 'pp': 
                                if out_vessel_signs[k] == -1.:
                                    u_2 = 'u_in'
                                    v_1 = 'v'
                                elif out_vessel_signs[k] == 1.:
                                    u_2 = 'u_out'
                                    v_1 = 'v_d'
                            
                            if vess_BC == 'pv':
                                u_1 = 'u'
                            elif vess_BC == 'vv':
                                u_1 = 'u_d'
                            
                            if v_1=='v_d':
                                v_2 = f'v_d_{out_vessel_names[k]}_Nout'
                            else:
                                v_2 = f'v_{out_vessel_names[k]}_Nout'
                            
                            self.__write_mapping(wf, vess_name+'_module', out_vessel_names[k]+'_module', [u_1], [u_2])

                            self.__write_mapping(wf, out_vessel_names[k]+'_module', 'generic_junction_connection', [v_1], [v_2])

                            if v_1=='v_d':
                                temp_out_vess_name = out_vessel_names[k]
                                vess_names_per_junc[-1][k] = 'd_'+temp_out_vess_name

            # if vessel_tup.vessel_type.endswith('Min_junction'):
            if 'Min_junction' in vessel_tup.vessel_type:
                # print("Min_junction vessel found AGAIN")
                vess_name = vessel_tup.name
                flow_vess_names.append(vess_name)
                flow_vess_types.append('Min')
                
                v_1 = f'v_{vess_name}_sum_Min' 
                v_2 = 'v_in_sum'
                self.__write_mapping(wf, 'generic_junction_connection', vess_name+'_module', [v_1], [v_2])
            
            # elif vessel_tup.vessel_type.endswith('Nout_junction'):
            elif 'Nout_junction' in vessel_tup.vessel_type:
                
                # if vessel_tup.vessel_type.endswith('MinNout_junction'):
                if 'MinNout_junction' in vessel_tup.vessel_type:
                    # print("MinNout_junction vessel found AGAIN")
                    vess_name = vessel_tup.name
                    flow_vess_names.append(vess_name)
                    flow_vess_names.append(vess_name) # same vessels repeated twice as it has junctions at both inlet and outlet nodes
                    flow_vess_types.append('MinNout')
                    flow_vess_types.append('MinNout')

                    v_1 = f'v_{vess_name}_sum_Min'
                    v_2 = 'v_in_sum'
                    self.__write_mapping(wf, 'generic_junction_connection', vess_name+'_module', [v_1], [v_2])

                    v_11 = f'v_{vess_name}_sum_Nout'
                    v_22 = 'v_out_sum'
                    self.__write_mapping(wf, 'generic_junction_connection', vess_name+'_module', [v_11], [v_22])

                else:
                    # print("Nout_junction vessel found AGAIN")
                    vess_name = vessel_tup.name
                    flow_vess_names.append(vess_name)
                    flow_vess_types.append('Nout')

                    v_1 = f'v_{vess_name}_sum_Nout'
                    v_2 = 'v_out_sum'
                    self.__write_mapping(wf, 'generic_junction_connection', vess_name+'_module', [v_1], [v_2])

        # print(flow_vess_names)
        # print(flow_vess_types)
        # print(" ")
        # print(vess_names_per_junc)
        # print(vess_signs_per_junc)
        # print(vess_bcs_per_junc)


        # create the computation environment for junction connections, and 
        # write the variable definition and calculation of flow to each junction flow port
        wf.write(f'<component name="generic_junction_connection">\n')
        variables = []
        units = []
        in_outs = []

        nJ = len(flow_vess_names)

        for idxV, flow_vess_name in enumerate(flow_vess_names):
            if flow_vess_types[idxV]=='Min':
                for press_vess_name in vess_names_per_junc[idxV]:
                    variables.append(f'v_{press_vess_name}_Min')
                    units.append(flow_units) # ('m3_per_s')
                    in_outs.append('in')

                variables.append(f'v_{flow_vess_name}_sum_Min')
                units.append(flow_units) # ('m3_per_s')
                in_outs.append('out')
            elif flow_vess_types[idxV]=='Nout':
                for press_vess_name in vess_names_per_junc[idxV]:
                    variables.append(f'v_{press_vess_name}_Nout')
                    units.append(flow_units) # ('m3_per_s')
                    in_outs.append('in')

                variables.append(f'v_{flow_vess_name}_sum_Nout')
                units.append(flow_units) # ('m3_per_s')
                in_outs.append('out')
            elif flow_vess_types[idxV]=='MinNout':
                if idxV==0:
                    pass
                else:
                    if flow_vess_name == flow_vess_names[idxV-1]:
                        for press_vess_name in vess_names_per_junc[idxV]:
                            variables.append(f'v_{press_vess_name}_Nout')
                            units.append(flow_units) # ('m3_per_s')
                            in_outs.append('in')
                        
                        variables.append(f'v_{flow_vess_name}_sum_Nout')
                        units.append(flow_units) # ('m3_per_s')
                        in_outs.append('out')

                        for press_vess_name in vess_names_per_junc[idxV-1]:
                            variables.append(f'v_{press_vess_name}_Min')
                            units.append(flow_units) # ('m3_per_s')
                            in_outs.append('in')

                        variables.append(f'v_{flow_vess_name}_sum_Min')
                        units.append(flow_units) # ('m3_per_s')
                        in_outs.append('out')
                    else:
                        pass

        self.__write_variable_declarations(wf, variables, units, in_outs)

        
        for idxV, flow_vess_name in enumerate(flow_vess_names):
            sum_case = -1
            if nJ==1:
                sum_case = 0
            else:
                count_vess = flow_vess_names.count(flow_vess_name)
                if count_vess==1:
                    sum_case = 0
                else:
                    if idxV==0:
                        if flow_vess_name==flow_vess_names[idxV+1]:
                            sum_case = 1
                        # else:
                        #     exit("ERROR")
                    elif idxV==nJ-1:
                        if flow_vess_name==flow_vess_names[idxV-1]:
                            sum_case = 2
                        # else:
                        #     exit("ERROR")
                    else:
                        if flow_vess_name==flow_vess_names[idxV+1]:
                            sum_case = 1
                        elif flow_vess_name==flow_vess_names[idxV-1]:
                            sum_case = 2
                    
            # print("junction", idxV, flow_vess_name, " || sum_case", sum_case)
            
            if sum_case==-1:
                print("ERROR :: sum_case is -1 for "+str(idxV)+" "+flow_vess_name+" . Exiting")
                exit()
            else:
                if sum_case==0:
                    rhs_variables = []
                    rhs_signs = []
                    if flow_vess_types[idxV]=='Min':
                        lhs_variable = f'v_{flow_vess_name}_sum_Min'
                        for press_vess_name in vess_names_per_junc[idxV]:
                            rhs_variables.append(f'v_{press_vess_name}_Min')
                        for press_vess_sign in vess_signs_per_junc[idxV]:
                            rhs_signs.append(press_vess_sign)
                    
                    elif flow_vess_types[idxV]=='Nout':
                        lhs_variable = f'v_{flow_vess_name}_sum_Nout'
                        for press_vess_name in vess_names_per_junc[idxV]:
                            rhs_variables.append(f'v_{press_vess_name}_Nout')
                        for press_vess_sign in vess_signs_per_junc[idxV]:
                            rhs_signs.append(press_vess_sign)
                    
                    self.__write_variable_sum_junc(wf, lhs_variable, rhs_variables, rhs_signs)
                    
                elif sum_case==1:
                    rhs_variables = []
                    rhs_signs = []
                    lhs_variable = f'v_{flow_vess_name}_sum_Min'
                    for press_vess_name in vess_names_per_junc[idxV]:
                        rhs_variables.append(f'v_{press_vess_name}_Min')
                    for press_vess_sign in vess_signs_per_junc[idxV]:
                        rhs_signs.append(press_vess_sign)
                    
                    self.__write_variable_sum_junc(wf, lhs_variable, rhs_variables, rhs_signs)

                elif sum_case==2:
                    rhs_variables = []
                    rhs_signs = []
                    lhs_variable = f'v_{flow_vess_name}_sum_Nout'
                    for press_vess_name in vess_names_per_junc[idxV]:
                        rhs_variables.append(f'v_{press_vess_name}_Nout')
                    for press_vess_sign in vess_signs_per_junc[idxV]:
                        rhs_signs.append(press_vess_sign)
                    
                    self.__write_variable_sum_junc(wf, lhs_variable, rhs_variables, rhs_signs)

        wf.write('</component>\n')

        print("writing environment to sum generic junctions input flows :: SUCCESSFUL")

    def __write_blood_volume_sum_comp(self, wf, vessel_df, vol_units='m3'):
        
        sum_vess_names = []
        vess_to_sum_names = []
        
        for vessel_tup in vessel_df.itertuples():
            if vessel_tup.module_format != 'cellml':
                # if not cellml then don't do anything for this vessel/module
                continue
            
            # Get the module row from the dataframe to check ports
            module_row = vessel_df.loc[vessel_df['name'] == vessel_tup.name].iloc[0]
            
            # Check if any port has multi_port='sum' or if using legacy vessel_type
            # TODO same thing for general and exit ports
            for idx, port in enumerate(module_row['entrance_ports'] + module_row['general_ports'] + \
                    module_row['exit_ports']):
                
                if 'multi_port' in port and port['multi_port'] == 'sum':
                    if idx >= len(module_row['entrance_ports']) + len(module_row['general_ports']):
                        is_exit_port = True
                    else:
                        is_exit_port = False
                    port_type = port['port_type']
                    sum_vess_name = vessel_tup.name
                    sum_vess_variable = port['variables'][0] # assumes only one variable per port
                    if sum_vess_name not in sum_vess_names:
                        sum_vess_names.append(sum_vess_name)
                    
                        inp_vessel_names = []
                        inp_variable_names = []
                        for inp_vessel_name in vessel_tup.inp_vessels:
                            if is_exit_port:
                                coupling_ports = vessel_df.loc[vessel_df["name"] == inp_vessel_name].squeeze()["entrance_ports"] + \
                                    vessel_df.loc[vessel_df["name"] == inp_vessel_name].squeeze()["general_ports"]
                            else:
                                coupling_ports = vessel_df.loc[vessel_df["name"] == inp_vessel_name].squeeze()["exit_ports"] + \
                                    vessel_df.loc[vessel_df["name"] == inp_vessel_name].squeeze()["general_ports"]
                            for couple_port in coupling_ports:
                                if couple_port['port_type'] == port_type:
                                    inp_variable_name = couple_port['variables'][0]
                                    if inp_vessel_name not in inp_vessel_names:
                                        inp_vessel_names.append(inp_vessel_name)
                                        inp_variable_names.append(inp_variable_name)
                        vess_to_sum_names.append(inp_vessel_names)

                        if len(inp_vessel_names) == 0:
                            pass
                        else:
                            # map volume
                            for inp_vessel_idx in range(len(inp_vessel_names)):
                                q_1 = inp_variable_names[inp_vessel_idx]
                                inp_vessel_name = inp_vessel_names[inp_vessel_idx]
                                q_2 = f'q_{inp_vessel_name}'
                                self.__write_mapping(wf, inp_vessel_name+'_module', 'sum_blood_volume', [q_1], [q_2])

                            # then map volume
                            # q_1 = f'q_{sum_vess_name}'
                            # q_2 = sum_vess_variable add _sum to change the name
                            q_1 = f'q_{sum_vess_name}_sum'
                            q_2 = sum_vess_variable
                            self.__write_mapping(wf, 'sum_blood_volume', sum_vess_name+'_module', [q_1], [q_2])

        # create computation environment for connection and write the variable definition 
        # and calculation of total blood volume in the whole system or in specific portions of it
        wf.write(f'<component name="sum_blood_volume">\n')
        variables = []
        units = []
        in_outs = []

        for idx_sum, sum_vess_name in enumerate(sum_vess_names):
            variables.append(f'q_{sum_vess_name}_sum')
            units.append(vol_units)
            in_outs.append('out') 
            for inp_vess_name in vess_to_sum_names[idx_sum]:
                variables.append(f'q_{inp_vess_name}')
                units.append(vol_units) 
                in_outs.append('in') 

        self.__write_variable_declarations(wf, variables, units, in_outs)

        for idx_sum, sum_vess_name in enumerate(sum_vess_names):
            rhs_variables = []
            lhs_variable = f'q_{sum_vess_name}_sum'
            for inp_vess_name in vess_to_sum_names[idx_sum]:
                rhs_variables.append(f'q_{inp_vess_name}')

            self.__write_variable_sum(wf, lhs_variable, rhs_variables)

        wf.write('</component>\n')



    def __write_variable_sum_junc(self, wf, lhs_variable, rhs_variables, rhs_signs):
        # Add-on for writing the sum of flow variables for a generic junction connection

        if lhs_variable.endswith('_Min'): #lhs_variable.endswith('_Jin'):
            invert = False
        elif lhs_variable.endswith('_Nout'): #lhs_variable.endswith('_Jout')
            invert = True

        wf.writelines('<math xmlns="http://www.w3.org/1998/Math/MathML">\n'
                      '   <apply>\n'
                      '       <eq/>\n'
                      f'       <ci>{lhs_variable}</ci>\n')

        if invert:
            if len(rhs_variables) > 1:
                # print("writing sum of multiple variables")
                wf.write('       <apply>\n')
                wf.write('           <plus/>\n')  # Start the sum block

                for idx, var in enumerate(rhs_variables):
                    sign = rhs_signs[idx]
                    if sign < 0:
                        wf.write(f'         <ci>{var}</ci>\n')  # Positive variables don't need an explicit <plus/>
                    elif sign > 0:
                        wf.write('        <apply>\n')
                        wf.write('          <minus/>\n')
                        wf.write(f'          <ci>{var}</ci>\n')
                        wf.write('        </apply>\n')
                
                wf.write('       </apply>\n')

            else:
                # print("writing sum of single variable")
                if rhs_signs[0] < 0.:
                    wf.write(f'            <ci>{rhs_variables[0]}</ci>\n')
                elif rhs_signs[0] > 0.:
                    wf.write('            <apply>\n')
                    wf.write('                <minus/>\n')
                    wf.write(f'                <ci>{rhs_variables[0]}</ci>\n')
                    wf.write('            </apply>\n')
        else:
            if len(rhs_variables) > 1:
                # print("writing sum of multiple variables")
                wf.write('       <apply>\n')
                wf.write('           <plus/>\n')  # Start the sum block

                for idx, var in enumerate(rhs_variables):
                    sign = rhs_signs[idx]
                    if sign > 0:
                        wf.write(f'         <ci>{var}</ci>\n')  # Positive variables don't need an explicit <plus/>
                    elif sign < 0:
                        wf.write('        <apply>\n')
                        wf.write('          <minus/>\n')
                        wf.write(f'          <ci>{var}</ci>\n')
                        wf.write('        </apply>\n')
                
                wf.write('       </apply>\n')

            else:
                # print("writing sum of single variable")
                if rhs_signs[0] > 0.:
                    wf.write(f'            <ci>{rhs_variables[0]}</ci>\n')
                elif rhs_signs[0] < 0.:
                    wf.write('            <apply>\n')
                    wf.write('                <minus/>\n')
                    wf.write(f'                <ci>{rhs_variables[0]}</ci>\n')
                    wf.write('            </apply>\n')

        wf.write('   </apply>\n')
        wf.write('</math>\n')


    def __write_access_variables(self, wf, vessel_df):
        vessel_df.apply(self.__write_access_variables_for_row, args=(wf,), axis=1)

    def __write_access_variables_for_row(self, vessel_row, wf):
        if vessel_row["module_format"] != 'cellml':
            # if not cellml then don't do anything for this vessel/module
            return
        wf.write(f'<component name="{vessel_row["name"]}">\n')
        lines_to_write = []
        for variable, unit, access_str, variable_type in vessel_row["variables_and_units"]:
            if access_str == 'access':
                lines_to_write.append(f'   <variable name="{variable}" public_interface="in" units="{unit}"/>\n')
            elif access_str == 'no_access':
                pass
            else:
                print('________________ERROR_________________')
                print(f'Error: Access string of variable {variable} should be either "access" or "no_access".')
                print('______________________________________')
        wf.writelines(lines_to_write)
        wf.write('</component>\n')
    
    def __write_global_parameters_access_variables(self, wf, parameters_array):
        
        wf.write(f'<component name="global">\n')
        lines_to_write = []
        for parameter, unit, const_type in zip(parameters_array["variable_name"], parameters_array["units"], parameters_array["const_type"]):
            if const_type == 'global_constant':
                lines_to_write.append(f'   <variable name="{parameter}" public_interface="in" units="{unit}"/>\n')
        wf.writelines(lines_to_write)
        wf.write('</component>\n')

    def __write_comp_to_module_mappings(self, wf, vessel_df):
        vessel_df.apply(self.__write_comp_to_module_mappings_for_row, args=(wf,), axis=1)

    def __write_comp_to_module_mappings_for_row(self, vessel_row, wf):
        if vessel_row["module_format"] != 'cellml':
            # if not cellml then don't do anything for this vessel/module
            return
        vessel_name = vessel_row["name"]
        inp_vars = [vessel_row["variables_and_units"][i][0] for i in
                    range(len(vessel_row["variables_and_units"])) if
                    vessel_row["variables_and_units"][i][2] == 'access']
        out_vars = inp_vars

        self.__write_mapping(wf, vessel_name, vessel_name + '_module', inp_vars, out_vars)
    
    def __write_global_parameters_comp_to_module_mappings(self, wf, parameters_array):
        inp_vars = [parameters_array["variable_name"][i] for i in
                    range(len(parameters_array["variable_name"])) if
                    parameters_array["const_type"][i] == 'global_constant']
        out_vars = inp_vars

        self.__write_mapping(wf, 'global', 'parameters_global' ,inp_vars, out_vars)

    def __write_param_mappings(self, wf, vessel_df, params_array=None):
        vessel_df.apply(self.__write_param_mappings_for_row, args=(wf,), params_array=params_array, axis=1)

    def __write_param_mappings_for_row(self, vessel_row, wf, params_array=None):
        if vessel_row["module_format"] != 'cellml':
            # if not cellml then don't do anything for this vessel/module
            return

        vessel_name = vessel_row["name"]
        module_addon = '_module'

        global_variable_addon = f'_{vessel_name}'
        # TODO check this doesnt break anything
        # if vessel_row["vessel_type"] == 'terminal' or vessel_row["vessel_type"] == 'terminal2':
        #     global_variable_addon = re.sub('_T$', '', global_variable_addon)
        params_with_addon_heading = 'parameters'
        params_without_addon_heading = 'parameters_global'

        vars = [vessel_row["variables_and_units"][i][0] for
                i in range(len(vessel_row["variables_and_units"])) if
                vessel_row["variables_and_units"][i][3] == 'constant']

        module_vars = [vars[i] + global_variable_addon for i in range(len(vars))]

        global_vars = [vessel_row["variables_and_units"][i][0] for
                       i in range(len(vessel_row["variables_and_units"])) if
                       vessel_row["variables_and_units"][i][3] == 'global_constant']

        # check that the variables are in the parameter array
        if params_array is not None:
            for variable_name in module_vars:
                if variable_name not in params_array["variable_name"]:
                    print(f'________________ERROR_________________')
                    print(f'variable {variable_name} is not in the parameter '
                          f'dataframe/csv file. It needs to be added!!!')
                    print(f'______________________________________')
                    exit()
        self.__write_mapping(wf, params_with_addon_heading, vessel_name + module_addon,
                             module_vars, vars)

        self.__write_mapping(wf, params_without_addon_heading, vessel_name + module_addon,
                             global_vars, global_vars)

    def __write_time_mappings(self, wf, vessel_df):
        vessel_df.apply(self.__write_time_mappings_for_row, args=(wf,), axis=1)

    def __write_time_mappings_for_row(self, vessel_row, wf):
        vessel_name = vessel_row["name"]
        module_addon = '_module'

        if vessel_row["module_format"] != 'cellml':
            # if not cellml then don't do anything for this vessel/module
            return

        self.__write_mapping(wf, 'environment', vessel_name + module_addon,
                             ['time'], ['t'])

    def __write_time_mappings_OLD(self, wf, vessel_df):
        for vessel_tup in vessel_df.itertuples():
            # input and output vessels
            vessel_name = vessel_tup.name
            module_addon = '_module'
            self.__write_mapping(wf, 'environment', vessel_name + module_addon,
                                 ['time'], ['t'])

    def __write_import(self, wf, vessel_tup):
        module_type = vessel_tup.module_type

        str_addon = '_module'

        wf.writelines([f'<import xlink:href="{self.file_prefix}_modules.cellml">\n',
                       f'    <component component_ref="{module_type}" name="{vessel_tup.name + str_addon}"/>\n',
                       '</import>\n'])

    def __check_input_output_modules(self, vessel_df, main_vessel, out_vessel,
                                     main_vessel_BC_type, out_vessel_BC_type,
                                     main_vessel_type, out_vessel_type):
        if not out_vessel:
            print(f'connection modules incorrectly defined for {main_vessel}')
            exit()
        if main_vessel_type.endswith('terminal'):
            pass
        elif main_vessel not in vessel_df.loc[vessel_df["name"] == out_vessel].inp_vessels.values[0]:
            print(f'"{main_vessel}" and "{out_vessel}" are incorrectly connected, '
                  f'check that they have eachother as output/input')
            exit()
        elif out_vessel not in vessel_df.loc[vessel_df["name"] == main_vessel].out_vessels.values[0]:
            print(f'"{main_vessel}" and "{out_vessel}" are incorrectly connected, '
                  f'check that they have eachother as output/input')
            exit()
        if out_vessel_BC_type.startswith('nn'):
            return
        if main_vessel_BC_type.startswith('nn'):
            return


        if len(main_vessel_BC_type) > 2:
            temp_main_vessel_BC_type = main_vessel_BC_type[:2]
        else:
            temp_main_vessel_BC_type = main_vessel_BC_type

        if temp_main_vessel_BC_type.endswith('v'):
            if not out_vessel_BC_type.startswith('p'):
                print(f'"{main_vessel}" output BC is v, the input BC of "{out_vessel}"',
                    ' should be p')
                exit()
        if temp_main_vessel_BC_type.endswith('p'):
            main_out_vessels = vessel_df.loc[vessel_df["name"] == main_vessel].out_vessels.values[0]
            if not out_vessel_BC_type.startswith('v'):

                pass_inp_out_check = False

                if (any(vessel_df.loc[vessel_df["name"] == vess, "vessel_type"].iloc[0].startswith("Min_")
                            for vess in main_out_vessels)
                    or any(vessel_df.loc[vessel_df["name"] == vess, "vessel_type"].iloc[0].startswith("MinNout_")
                            for vess in main_out_vessels)): 
                    pass_inp_out_check = True
                
                for temp_out_vess in main_out_vessels:
                    inp_vessels = vessel_df.loc[vessel_df["name"] == temp_out_vess].inp_vessels.values[0]
                    for temp_inp_vess in inp_vessels:
                        temp_inp_vess_BC_type = vessel_df.loc[vessel_df["name"] == temp_inp_vess, "vessel_type"].iloc[0]
                        if temp_inp_vess_BC_type.startswith("Nout_") or temp_inp_vess_BC_type.startswith("MinNout_"):
                            pass_inp_out_check = True
                            break

                if not pass_inp_out_check:
                    print(f'"{main_vessel}" output BC is p, the input BC of "{out_vessel}"',
                    ' should be v')
                    exit()

    def __check_input_output_ports(self, exit_ports, output_general_ports, entrance_ports, input_general_ports, upstream_module, downstream_module):
        # check that input and output modules have a matching port
        shared_exit_port = False
        shared_general_port = False
        for exit_port in exit_ports:
            if exit_port["port_type"] in [entrance_port["port_type"] for entrance_port in entrance_ports] or exit_port["port_type"] in [general_port["port_type"] for general_port in input_general_ports]:
                shared_exit_port = True
                break
        for general_port in output_general_ports:
            if general_port["port_type"] in [entrance_port["port_type"] for entrance_port in entrance_ports] or general_port["port_type"] in [general_port["port_type"] for general_port in input_general_ports]:
                shared_general_port = True
                break
        if len(exit_ports) == 0:
            shared_exit_port = True
        if len(output_general_ports) == 0:
            shared_general_port = True
        if shared_exit_port == False and shared_general_port == False:
            print(f'upstream module {upstream_module} and downstream module {downstream_module} do not have a matching port,'
                  f'check the module configuration file')
            print(f'upstream module exit ports: {[exit_port["port_type"] for exit_port in exit_ports]}')
            print(f'upstream module general ports: {[general_port["port_type"] for general_port in output_general_ports]}')
            print(f'downstream module entrance ports: {[entrance_port["port_type"] for entrance_port in entrance_ports]}')
            print(f'downstream module general ports: {[general_port["port_type"] for general_port in input_general_ports]}')
            exit()

    def __write_mapping(self, wf, inp_name, out_name, inp_vars_list, out_vars_list):
        mapping = ['<connection>\n', f'   <map_components component_1="{inp_name}" component_2="{out_name}"/>\n']
        for inp_var, out_var in zip(inp_vars_list, out_vars_list):
            if inp_var and out_var:
                mapping.append(f'   <map_variables variable_1="{inp_var}" variable_2="{out_var}"/>\n')

        mapping.append('</connection>\n')
        if len(mapping) > 3:
            wf.writelines(mapping)

    def __write_variable_declarations(self, wf, variables, units, in_outs):
        for variable, unit, in_out in zip(variables, units, in_outs):
            if in_out == 'priv_in_pub_out':
                wf.write(f'<variable name="{variable}" private_interface="in" public_interface="out" units="{unit}"/>\n')
            else:
                wf.write(f'<variable name="{variable}" public_interface="{in_out}" units="{unit}"/>\n')

    def __write_constant_declarations(self, wf, variable_names, units, values):
        for variable, unit, value in zip(variable_names, units, values):
            wf.write(f'<variable initial_value="{value}" name="{variable}" '
                     f'public_interface="out" units="{unit}"/>\n')

    def __write_variable_sum(self, wf, lhs_variable, rhs_variables):
        wf.writelines('<math xmlns="http://www.w3.org/1998/Math/MathML">\n'
                      '   <apply>\n'
                      '       <eq/>\n'
                      f'       <ci>{lhs_variable}</ci>\n')
        if len(rhs_variables) > 1:
            wf.write('       <apply>\n')
            wf.write('           <plus/>\n')
            for variable in rhs_variables:
                wf.write(f'            <ci>{variable}</ci>\n')
            wf.write('       </apply>\n')
        else:
            wf.write(f'            <ci>{rhs_variables[0]}</ci>\n')

        wf.write('   </apply>\n')
        wf.write('</math>\n')

    def __write_variable_average(self, wf, lhs_variable, rhs_variables_to_average, rhs_variables_weighting):
        """ writes the cellml code for averaging variables with weighting. Designed for getting an equivalent
        concentration

        C = (C_1*v_1 + C_2*v_2)/(v_1 + v_2)
        """
        if len(rhs_variables_to_average) != len(rhs_variables_weighting):
            print('rhs_variables_to_average and rhs_variables_weighting must be the same length, exiting')
            exit()

        wf.writelines('<math xmlns="http://www.w3.org/1998/Math/MathML">\n'
                      '   <apply>\n'
                      '       <eq/>\n'
                      f'       <ci>{lhs_variable}</ci>\n')
        if len(rhs_variables_to_average) > 1:
            wf.write('      <apply>\n')
            wf.write('          <divide/>\n')
            wf.write('          <apply>\n')
            wf.write('              <plus/>\n')
            for variable, weight in zip(rhs_variables_to_average, rhs_variables_weighting):
                wf.write('          <apply>\n')
                wf.write('              <times/>\n')
                wf.write(f'             <ci>{variable}</ci>\n')
                wf.write(f'             <ci>{weight}</ci>\n')
                wf.write('          </apply>\n')
            wf.write('          </apply>\n')
            wf.write('          <apply>\n')
            wf.write('              <plus/>\n')
            wf.write('              <apply>\n')
            wf.write('                  <abs/>\n')
            wf.write('                  <apply>\n')
            wf.write('                      <plus/>\n')
            for weight in rhs_variables_weighting:
                wf.write(f'                     <ci>{weight}</ci>\n')
            wf.write('                  </apply>\n')
            wf.write('              </apply>\n')
            wf.write('              <cn cellml:units="m3_per_s" type="e-notation">1<sep/>-10</cn>\n')
            wf.write('          </apply>\n')
            wf.write('      </apply>\n')

        else:
            wf.write(f'            <ci>{rhs_variables_to_average[0]}</ci>\n')

        wf.write('   </apply>\n')
        wf.write('</math>\n')
