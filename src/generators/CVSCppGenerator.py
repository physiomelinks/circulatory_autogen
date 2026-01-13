'''
Created on 23/05/2023

@author: Finbar Argus
'''

import numpy as np
import re
import pandas as pd
import os
import sys
from sys import exit
generators_dir_path = os.path.dirname(__file__)
root_dir = os.path.join(generators_dir_path, '../..')
sys.path.append(os.path.join(root_dir, 'src'))
from generators.CVSCellMLGenerator import CVS0DCellMLGenerator

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

from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import DCTERMS
from urllib.parse import urlparse

class CVS0DCppGenerator(object):
    '''
    Generates Cpp files from CellML files and annotations for a 0D model

    # TODO This code is not good. it needs a complete re-write. 
    # The class could be made much much simpler. Don't try to 
    # work with this without talking to Finbar. Make him simplify it before you
    # work with this.
    '''


    def __init__(self, model, generated_model_subdir, file_prefix, resources_dir=None, 
                 solver='CVODE', couple_to_1d=False, cpp_generated_models_dir=None,
                 cpp_1d_model_config_path=None):
        '''
        Constructor
        '''
        if LIBCELLML_available == False:
            print("Error: LibCellML is not available, cannot generate Cpp files.")
            exit()
            
        self.model = model
        self.generated_model_subdir = generated_model_subdir
        if not os.path.exists(self.generated_model_subdir):
            os.mkdir(self.generated_model_subdir)
        self.file_prefix = file_prefix
        self.file_prefix_with_ids = f'{self.file_prefix}-with-ids'
        if resources_dir is None:
            self.resources_dir = os.path.join(generators_dir_path, '../../resources')
        else:
            self.resources_dir = resources_dir 
        self.generated_model_file_path = os.path.join(self.generated_model_subdir, 
                                                      self.file_prefix + '.cellml')
        self.generated_wID_model_file_path = os.path.join(self.generated_model_subdir, 
                                                      self.file_prefix_with_ids + '.cellml')
        self.annotated_model_file_path = os.path.join(self.generated_model_subdir, 
                                                      self.file_prefix_with_ids + '--annotations.ttl')
        
        self.cellml_model = None 
        self.couple_to_1d = couple_to_1d
        if self.couple_to_1d:
            self.output_cpp_file_name = "model0d" # always the same, independent of model name, to allow
                                                       # for coupling to cpp 1d model.
        else:
            self.output_cpp_file_name = self.file_prefix
        self.external_headers = []
        if self.couple_to_1d:
            self.cpp_1d_model_config_path = cpp_1d_model_config_path
            self.create_main = False
            self.external_headers += ['model1d.h']
        else:
            self.create_main = True

        if cpp_generated_models_dir is None:
            self.cpp_generated_models_dir = self.generated_model_subdir
        else:
            self.cpp_generated_models_dir = cpp_generated_models_dir
        
        self.solver = solver

        # these are for delayed variables
        self.variables_to_delay = []
        self.delayed_variables = []
        self.delay_amounts = []
        self.independent_variables = []
        # these are for two-way coupling
        self.port_input_variables = []
        self.port_output_variables = []
        self.control_variables = [] # This is for extra control_variables, e.g. resistance in a 1D FV model
        self.connection_vessel_indices = []
        self.connection_vessel_types = []
        self.connection_vessel_inlet_or_outlet = []
        self.connection_vessel_flow_or_pressure_bc = []
        # these are for one-way coupling
        self.input_variables = []
        self.output_variables = []

        # TODO should the annotations stuff be in a different class? It's not crucial for 
        # running of the model but could be useful for making sure we have annotated models
        # with the delays, coupling variables etc.

        # define some URIs for things we need

        # use this URI to identify delayed variables - not the perfect URI, but will do for now
        #     This is actually "delayed differential equation model" from the MAMO ontology
        #delay_variable_uri = URIRef('http://identifiers.org/mamo/MAMO_0000089')
        self.delay_variable_uri = URIRef('https://github.com/nickerso/libcellml-python-utils/properties.rst#delay-variable')
        self.variable_to_delay_uri = URIRef('https://github.com/nickerso/libcellml-python-utils/properties.rst#variable-to-delay')
        self.independent_variable_uri = URIRef('https://github.com/nickerso/libcellml-python-utils/properties.rst#variable-to-delay')
        self.delay_amount_uri = URIRef('https://github.com/nickerso/libcellml-python-utils/properties.rst#delay-amount')

        # use this for some random thing that I want to define - http://example.com is a good base for things that will never resolve
        self.stuff_uri = URIRef('http://example.com/cool-thing#21')

        # a "readout" variable that we maybe want to connect to something external?
        self.timecourse_readout_uri = URIRef('http://identifiers.org/mamo/MAMO_0000031')
        self.output_variable_uri = URIRef('http://identifiers.org/mamo/MAMO_0000018')
        self.input_variable_uri = URIRef('http://identifiers.org/mamo/MAMO_0000017')
        # TODO this is randomly chosen for now, update this
        self.external_variable_uri = URIRef('http://identifiers.org/mamo/MAMO_9239430432')
        self.control_variable_uri = URIRef('http://identifiers.org/mamo/MAMO_9239430433')

        self.g = Graph()

    def annotate_cellml(self):
        # TODO most of this function 
        # isn't needed for the generation anymore, but is it good to keep annotation in here? Probably.
        self.cellml_model = cellml.parse_model(self.generated_model_file_path, False)
        annotator = Annotator()
        annotator.setModel(self.cellml_model)

        if annotator.assignAllIds():
            print('Some entities have been assigned an ID, you should save the model!')
        else:
            print('Everything already had an ID.')

        duplicates = annotator.duplicateIds()
        if len(duplicates) > 0:
            print("There are some duplicate IDs, behaviour may be unreliable...")
            print(duplicates)

        # blow away all the IDs and reassign them
        annotator.clearAllIds()
        annotator.assignAllIds()

        model_string = cellml.print_model(self.cellml_model)
        print(model_string)
        
        # and save the updated model to a new file
        # - note, we need the model filename for making our annotations later
        with open(self.generated_wID_model_file_path, 'w') as f:
            f.write(model_string)
        
        # get the ID of the variables we want to annotate
        # The below for loops are the only part of annotate_cellml that is needed to run generate_cpp
        for vessel_tup in self.model.vessels_df.itertuples():
            if "delay_info" in vessel_tup._fields:
                if vessel_tup.delay_info is not "None":
                    for II in range(len(vessel_tup.delay_info["variables_to_delay"])):
                        variable_to_delay = vessel_tup.delay_info["variables_to_delay"][II]
                        delayed_variable = vessel_tup.delay_info["delayed_variables"][II]
                        delay_amount = vessel_tup.delay_info["delay_amounts"][II]
                        
                        self.variables_to_delay.append(self.cellml_model.component(vessel_tup.name \
                                                                        ).variable(variable_to_delay).id())
                        self.delayed_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                        ).variable(delayed_variable).id())
                        self.delay_amounts.append(self.cellml_model.component(vessel_tup.name \
                                                                        ).variable(delay_amount).id())
                        # TODO make functionality to delay from other variables?
                        self.independent_variables.append(self.cellml_model.component("environment" \
                                                                        ).variable('time').id())

        for vessel_tup in self.model.vessels_df.itertuples():
            for out_vessel in vessel_tup.out_vessels:
                out_vessel_row = self.model.vessels_df.loc[self.model.vessels_df["name"] == out_vessel].squeeze()
                if out_vessel_row["vessel_type"] == "FV1D_vessel":
                    # get the index from the out_vessel name which is of format "FV1D_##" or FV1D_###"
                    fv1d_vessel_index = re.findall(r'\d+$', out_vessel)[0]
                    if len(fv1d_vessel_index) != 1:
                        print("ERROR: FV1D vessel idx is not written correctly",
                              "should be of format FV1D_# or FV1D_##, etc")
                        exit()
                    self.connection_vessel_indices.append(fv1d_vessel_index)
                    self.connection_vessel_types.append("FV_1d") # TODO only option for now, can be extended to other kinds of coupling.
                    for exit_port in vessel_tup.exit_ports:
                        if exit_port["port_type"] == "vessel_port":
                            self.connection_vessel_inlet_or_outlet.append("outlet") 
                            self.connection_vessel_indices.append(fv1d_vessel_index)
                            if vessel_tup.BC_type[1] == 'v':
                                self.port_input_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                ).variable(exit_port["variables"][0]).id())
                                self.port_output_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                ).variable(exit_port["variables"][1]).id())
                            elif vessel_tup.BC_type[1] == 'p': 
                                self.port_input_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                ).variable(exit_port["variables"][1]).id())
                                self.port_output_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                ).variable(exit_port["variables"][0]).id())
                            else:
                                print("unknown BC type of {vessel_tup.BC_type} connecting to a "
                                      "1D FV model")
                        elif exit_port["port_type"] == "FV_resistance_port":
                            self.control_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                ).variable(exit_port["variables"][0]).id())
                        # TODO other port types will be added here! 
                        else:
                            print("unknown port type of {exit_port['port_type']} connecting to a "
                                  "1D FV model")
                            exit()
                
            for inp_vessel in vessel_tup.inp_vessels:
                inp_vessel_row = self.model.vessels_df.loc[self.model.vessels_df["name"] == inp_vessel].squeeze()
                if inp_vessel_row["vessel_type"] == "FV1D_vessel":
                    fv1d_vessel_index = re.findall(r'\d+$', inp_vessel)[0]
                    if len(fv1d_vessel_index) != 1:
                        print("ERROR: FV1D vessel idx is not written correctly",
                              "should be of format FV1D_# or FV1D_##, etc")
                        exit()
                    self.connection_vessel_indices.append(fv1d_vessel_index)
                    self.connection_vessel_types.append("FV_1d") # TODO only option for now, can be extended to other kinds of coupling.
                    for entrance_port in vessel_tup.entrance_ports:
                        if entrance_port["port_type"] == "vessel_port":
                            self.connection_vessel_inlet_or_outlet.append("inlet") 
                            if vessel_tup.BC_type[0] == 'v':
                                self.connection_vessel_flow_or_pressure_bc.append("flow") 
                                self.port_input_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                ).variable(entrance_port["variables"][0]).id())
                                self.port_output_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                ).variable(entrance_port["variables"][1]).id())
                                # TODO include a variable for the pressure or flow input
                            elif vessel_tup.BC_type[0] == 'p': 
                                self.connection_vessel_flow_or_pressure_bc.append("pressure") 
                                self.port_input_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                ).variable(entrance_port["variables"][1]).id())
                                self.port_output_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                ).variable(entrance_port["variables"][0]).id())
                            else:
                                print("unknown BC type of {vessel_tup.BC_type} connecting to a "
                                      "1D FV model")
                                exit()
                        elif entrance_port["port_type"] == "FV_resistance_port":
                            self.control_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                ).variable(entrance_port["variables"][0]).id())
                        # TODO other port types will be added here! 
                        else:
                            # if this port_type isn't one of the above, it wont be connected to
                            # the FV1D model
                            pass

        # self.input_variables = [model.component(vessel_tup.name).variable('v_in').id()]
        # self.output_variables = [model.component(vessel_tup.name).variable('u').id()]

        # Create an RDF URI node for our variable to use as the subject for multiple triples
        # note: we are going to serialise the RDF graph into the same folder, so we need a URI that is relative to the intended file
        variable_to_delay_name_uri = [URIRef(self.generated_wID_model_file_path + 
                                             '#' + variable) for variable in self.variables_to_delay]
        delay_variable_name_uri = [URIRef(self.generated_wID_model_file_path + 
                                          '#' + variable) for variable in self.delayed_variables]
        delay_amount_name_uri = [URIRef(self.generated_wID_model_file_path + 
                                        '#' + delay_amount) for delay_amount in self.delay_amounts]
        independent_variable_name_uri = [URIRef(self.generated_wID_model_file_path + 
                                                '#' + variable) for variable in self.independent_variables]

        port_input_variable_name_uri = [URIRef(self.generated_wID_model_file_path + 
                                               '#' + variable) for variable in self.port_input_variables]
        port_output_variable_name_uri = [URIRef(self.generated_wID_model_file_path + 
                                                '#' + variable) for variable in self.port_output_variables]
        control_variable_name_uri = [URIRef(self.generated_wID_model_file_path +
                                            '#' + variable) for variable in self.control_variables]

        external_variable_idx_uri = []                                        
        for idx, type, inlet_or_outlet, flow_or_pressure_bc in zip(self.connection_vessel_indices, self.connection_vessel_types, 
                             self.connection_vessel_inlet_or_outlet, self.connection_vessel_flow_or_pressure_bc):
            external_variable_idx_uri.append(URIRef(self.cpp_generated_models_dir) +
                                        '#' + type + '#' + idx + '#' + inlet_or_outlet + '#' + flow_or_pressure_bc)

        input_variable_name_uri = [URIRef(self.generated_wID_model_file_path + 
                                          '#' + variable) for variable in self.input_variables]
        output_variable_name_uri = [URIRef(self.generated_wID_model_file_path + 
                                           '#' + variable) for variable in self.output_variables]
        
        # Add triples using store's add() method.
        # We're using the Dublin Core term "type" to associate the variable with the delay...
        for II in range(len(variable_to_delay_name_uri)):
            self.g.add((variable_to_delay_name_uri[II], DCTERMS.type, self.variable_to_delay_uri))
            self.g.add((variable_to_delay_name_uri[II], self.delay_variable_uri, delay_variable_name_uri[II]))
            self.g.add((variable_to_delay_name_uri[II], self.independent_variable_uri, independent_variable_name_uri[II]))
            self.g.add((variable_to_delay_name_uri[II], self.delay_amount_uri, delay_amount_name_uri[II]))
        # Set coupling variables
        for II in range(len(self.port_input_variables)):
            # for this input variable, add the output variable as a coupling variable
            self.g.add((port_input_variable_name_uri[II], DCTERMS.type, self.input_variable_uri))
            self.g.add((port_input_variable_name_uri[II], self.output_variable_uri, port_output_variable_name_uri[II]))
            self.g.add((port_input_variable_name_uri[II], self.external_variable_uri, external_variable_idx_uri[II]))
            if len(control_variable_name_uri) > 0:
                self.g.add((port_input_variable_name_uri[II], self.control_variable_uri, control_variable_name_uri[II]))
        
        for II in range(len(self.output_variables)):
            self.g.add((output_variable_name_uri[II], DCTERMS.type, self.output_variable_uri))
        for II in range(len(self.input_variables)):
            self.g.add((input_variable_name_uri[II], DCTERMS.type, self.input_variable_uri))

        # print all the data in the turtle format
        print(self.g.serialize(format='ttl'))

        # and save to a file
        with open(self.annotated_model_file_path, 'w') as f:
            f.write(self.g.serialize(format='ttl'))

    def generate_cellml(self):
        print("generating CellML files, before Cpp generation")
        cellml_generator = CVS0DCellMLGenerator(self.model, self.inp_data_dict)
        cellml_generator.generate_files()

    def set_annotated_model_file_path(self, annotated_model_file_path):
        # This is for if we want to use a cellml file that it already annotated, i.e, one we didn't annotate.
        # TODO TEST THIS
        self.annotated_model_file_path = annotated_model_file_path

    def generate_cpp(self):

        print("now generating Cpp files")
        
        g = Graph().parse(self.annotated_model_file_path)

        # find all delayed variables
        variables_to_delay_info = []
        for vtd in self.g.subjects(DCTERMS.type, self.variable_to_delay_uri):
            # we only expect one delay variable for each variable to delay
            dv = self.g.value(vtd, self.delay_variable_uri)
            d_amount = self.g.value(vtd, self.delay_amount_uri)
            variables_to_delay_info.append([str(vtd), str(dv), str(d_amount)])
            
        # # find all timecourse readouts
        # readout_variables = []
        # for d in g.subjects(DCTERMS.type, timecourse_readout_uri):
        #     readout_variables.append(str(d))
            
        # print(readout_variables)

        # find input and output variables
        port_input_variable_info = []
        input_variable_info= []
        output_variable_info= []
        for d in self.g.subjects(DCTERMS.type, self.input_variable_uri):
            # TODO most of the below for-loops could be done in this one to simplify this script
            port_output_variable_readout = self.g.value(d, self.output_variable_uri)
            external_variable_readout = self.g.value(d, self.external_variable_uri)
            control_variable_readout = self.g.value(d, self.control_variable_uri)
            info_list = [str(d)]
            if port_output_variable_readout is not None:
                info_list.append(str(port_output_variable_readout))
                if external_variable_readout is not None:
                    info_list.append(str(external_variable_readout))
                    if control_variable_readout is not None:
                        info_list.append(str(control_variable_readout))
                port_input_variable_info.append(info_list)
            else:
                input_variable_info.append(info_list)

        for d in self.g.subjects(DCTERMS.type, self.output_variable_uri):
            output_variable_info.append(str(d))
            
        print('port_input_variables')
        print(port_input_variable_info)
        print('input_variables')
        print(input_variable_info)
        print('output_variables')
        print(output_variable_info)

        # We're going to use the "model" file from the first variable to delay and only continue if all annotations use the same model...

        delayed_ids = []
        for vtd, dv, d_amount in variables_to_delay_info:
            vtd_url = urlparse(vtd)
            if vtd_url.path != self.generated_wID_model_file_path:
                print("found an unexpected model file for variable to delay?!")
                exit()
            dv_url = urlparse(dv)
            if dv_url.path != self.generated_wID_model_file_path:
                print("found an unexpected model file for delay variable?!")
                exit()
            d_amount_url = urlparse(d_amount)
            if d_amount_url.path != self.generated_wID_model_file_path:
                print("found an unexpected model file for delay amount?!")
                exit()
            delayed_ids.append([vtd_url.fragment, dv_url.fragment, d_amount_url.fragment])
            
        # readout_ids = []
        # for v in readout_variables:
        #     url = urlparse(v)
        #     if url.path != model_file:
        #         print("found an unexpected model file for readout variable?!")
        #         exit
        #     readout_ids.append(url.fragment)

        port_input_variable_ids = []
        for entry in port_input_variable_info:
            if len(entry) == 2:
                port_input_var, port_output_var = entry
                external_var = None
                control_var = None
            elif len(entry) == 3:
                port_input_var, port_output_var, external_var = entry
                control_var = None
            elif len(entry) == 4:
                port_input_var, port_output_var, external_var, control_var = entry
            
            port_input_var_url = urlparse(port_input_var)
            if port_input_var_url.path != self.generated_wID_model_file_path:
                print("found an unexpected model file for readout variable?!")
                exit()
            port_output_var_url = urlparse(port_output_var)
            if port_output_var_url.path != self.generated_wID_model_file_path:
                print("found an unexpected model file for readout variable?!")
                exit()
            if external_var is not None:
                external_var_url = urlparse(external_var)
                if external_var_url.path != self.cpp_generated_models_dir:
                    print("found an unexpected model file for readout variable?!")
                    exit()
                if control_var is not None:
                    control_var_url = urlparse(control_var)
                    if control_var_url.path != self.generated_wID_model_file_path:
                        print("found an unexpected model file for readout variable?!")
                        exit()
                    port_input_variable_ids.append([port_input_var_url.fragment, port_output_var_url.fragment, 
                                                    external_var_url.fragment, control_var_url.fragment])
                else:
                    port_input_variable_ids.append([port_input_var_url.fragment, port_output_var_url.fragment, 
                                                    external_var_url.fragment])
            else:
                port_input_variable_ids.append([port_input_var_url.fragment, port_output_var_url.fragment])

        input_variable_ids = []
        for input_var in input_variable_info:
            input_var_url = urlparse(input_var)
            if input_var_url.path != self.generated_wID_model_file_path:
                print("found an unexpected model file for readout variable?!")
                exit()
            input_variable_ids.append(input_var_url.fragment)

        output_variable_ids = []
        for output_var in output_variable_info:
            output_var_url = urlparse(output_var)
            if output_var_url.path != self.generated_wID_model_file_path:
                print("found an unexpected model file for readout variable?!")
                exit()
            output_variable_ids.append(output_var_url.fragment)

        # Now we have the model file and the IDs for the variables in that model that we want to do stuff with. So we can parse the model and see if we can find the variables.
        # on windows getting a leading '/' in the filename which libCellML doesn't like...
        self.generated_wID_model_file_path= self.generated_wID_model_file_path[0:]

        # parse the model in non-strict mode to allow non CellML 2.0 models
        model = cellml.parse_model(self.generated_wID_model_file_path, False)

        # and make an annotator for this model
        # TODO this could be done before all of the loops when I turn it into one for loop.
        annotator = Annotator()
        annotator.setModel(model)

        # map our IDs to the actual variables
        annotated_variables = []
        for vtd_id, dv_id, d_amount_id in delayed_ids:
            # get the variable (will fail if id doesn't correspond to a variable in the model)
            vtd = annotator.variable(vtd_id)
            if vtd == None:
                print('Unable to find a variable to delay with the id {} in the given model...'.format(vtd_id))
                exit()
            dv = annotator.variable(dv_id)
            if dv == None:
                print('Unable to find a delay variable with the id {} in the given model...'.format(dv_id))
                exit()
            d_amount = annotator.variable(d_amount_id)
            if d_amount == None:
                print('Unable to find a delay variable with the id {} in the given model...'.format(dv_id))
                exit()
            annotated_variables.append([[vtd, dv, d_amount], self.delay_variable_uri])
            
        # for i in readout_ids:
        #     # get the variable (will fail if id doesn't correspond to a variable in the model)
        #     v = annotator.variable(i)
        #     if v == None:
        #         print('Unable to find a readout variable with the id {} in the given model...'.format(i))
        #         exit
        #     annotated_variables.append([v, timecourse_readout_uri])

        for entry in port_input_variable_ids:
            if len(entry) == 2:
                port_input_var_id, port_output_var_id = entry
            elif len(entry) == 3:
                port_input_var_id, port_output_var_id, external_var_id = entry
            elif len(entry) == 4:
                port_input_var_id, port_output_var_id, external_var_id, control_var_id = entry
                control_var = annotator.variable(control_var_id)
                if control_var == None:
                    print('Unable to find a readout variable with the id '
                          '{} in the given model...'.format(control_var_id))
                    exit()
            # get the variable (will fail if id doesn't correspond to a variable in the model)
            port_input_var = annotator.variable(port_input_var_id)
            if port_input_var == None:
                print('Unable to find a readout variable with the id {} in the given model...'.format(port_input_var_id))
                exit()
            port_output_var = annotator.variable(port_output_var_id)
            if port_output_var == None:
                print('Unable to find a readout variable with the id {} in the given model...'.format(port_output_var_id))
                exit()
            # TODO some check to see that the external variable is actually a variable within the cpp model??
            if len(entry) == 2:
                annotated_variables.append([[port_input_var, port_output_var], self.input_variable_uri])
            elif len(entry) == 3:
                annotated_variables.append([[port_input_var, port_output_var, external_var_id], self.input_variable_uri])
            elif len(entry) == 4:
                annotated_variables.append([[port_input_var, port_output_var, external_var_id, control_var], self.input_variable_uri])

        for input_var_id in input_variable_ids:
            # get the variable (will fail if id doesn't correspond to a variable in the model)
            input_var = annotator.variable(input_var_id)
            if input_var == None:
                print('Unable to find a readout variable with the id {} in the given model...'.format(input_var_id))
                exit()
            annotated_variables.append([[input_var], self.input_variable_uri])

        for output_var_id in output_variable_ids:
            # get the variable (will fail if id doesn't correspond to a variable in the model)
            output_var = annotator.variable(output_var_id)
            if output_var == None:
                print('Unable to find a readout variable with the id {} in the given model...'.format(output_var_id))
                exit()
            annotated_variables.append([[output_var], self.output_variable_uri])

        # TODO everything above needs to be simplified and put into a single for loop.
        #  Even the below loop could be in the same for loop.
        
        # Need to work out how to map the annotations through to the variables in the generated code....
        # Generate C code for the model.

        model_dir = os.path.dirname(self.generated_wID_model_file_path)

        # resolve imports, in non-strict mode
        importer = cellml.resolve_imports(model, model_dir, False)
        # need a flattened model for analysing
        flat_model = cellml.flatten_model(model, importer)
        
        model_string = cellml.print_model(flat_model)
        
        with open(os.path.join(self.generated_model_subdir, self.file_prefix + '_flat.cellml'), 'w') as f:
            f.write(model_string)

        # analyse the model
        a = Analyser()

        # set the delayed variables as external
        external_variable_info = []
        for vv, uri in annotated_variables:
            if uri == self.delay_variable_uri:
                v = vv[0]
                dv = vv[1]
                d_amount = vv[2]
                flat_variable_to_delay = flat_model.component(v.parent().name()).variable(v.name())
                flat_delay_variable = flat_model.component(dv.parent().name()).variable(dv.name())
                flat_delay_amount_variable = flat_model.component(dv.parent().name()).variable(d_amount.name())
                aev = AnalyserExternalVariable(flat_delay_variable)
                aev.addDependency(flat_variable_to_delay)
                aev.addDependency(flat_delay_amount_variable)
                #
                # TODO: really need to work out how to handle other dependencies here to make sure 
                #       all required variables are up to date...
                #
                a.addExternalVariable(aev)
                # keep track of external variable information for use in generating code
                external_variable_info.append({
                    'variable': flat_variable_to_delay,
                    'delay_variable': flat_delay_variable,
                    'delay_amount_variable': flat_delay_amount_variable,
                    'analyser_variable': aev,
                    'variable_type': 'delay'
                })
            elif uri == self.input_variable_uri:
                if len(vv) == 1:
                    v = vv
                    input_variable = flat_model.component(v.parent().name()).variable(v.name())
                    aev = AnalyserExternalVariable(input_variable)
                    a.addExternalVariable(aev)
                    external_variable_info.append({
                        'variable': input_variable,
                        'analyser_variable': aev,
                        'variable_type': 'input'
                    })
                elif len(vv) == 2:
                    input_var = vv[0]
                    output_var = vv[1]
                    input_variable = flat_model.component(input_var.parent().name()).variable(input_var.name())
                    output_variable = flat_model.component(output_var.parent().name()).variable(output_var.name())
                    aev = AnalyserExternalVariable(input_variable)
                    a.addExternalVariable(aev)
                    external_variable_info.append({
                        'variable': input_variable,
                        'port_variable': output_variable,
                        'analyser_variable': aev,
                        'variable_type': 'input'
                    })
                elif len(vv) == 3:
                    input_var = vv[0]
                    output_var = vv[1]
                    ext_variable = vv[2]
                    input_variable = flat_model.component(input_var.parent().name()).variable(input_var.name())
                    output_variable = flat_model.component(output_var.parent().name()).variable(output_var.name())
                    splt = ext_variable.split('#')
                    ext_variable_type = splt[0]
                    ext_variable_idx = splt[1]
                    ext_variable_inlet_or_outlet = splt[2]
                    ext_variable_flow_or_pressure_bc = splt[3]
            
                    aev = AnalyserExternalVariable(input_variable)
                    a.addExternalVariable(aev)
                    external_variable_info.append({
                        'variable': input_variable,
                        'port_variable': output_variable,
                        'ext_variable_idx': ext_variable_idx,
                        'ext_variable_type': ext_variable_type,
                        'coupled_to_type': 'FV_1d' if ext_variable_type.startswith('FV_1d') else 'unknown',
                        'inlet_or_outlet': ext_variable_inlet_or_outlet,
                        'bc_inlet0_or_outlet1': 0 if ext_variable_inlet_or_outlet == 'inlet' else 1,
                        'flow_or_pressure_bc': ext_variable_flow_or_pressure_bc,
                        'analyser_variable': aev,
                        'control_variable': None,
                        'control_variable_index': None,
                        'variable_type': 'input'
                    })
                elif len(vv) == 4:
                    # TODO simplify so i'm not rewriting all of this for each case
                    input_var = vv[0]
                    output_var = vv[1]
                    ext_variable = vv[2]
                    cont_var = vv[3]
                    input_variable = flat_model.component(input_var.parent().name()).variable(input_var.name())
                    output_variable = flat_model.component(output_var.parent().name()).variable(output_var.name())
                    cont_variable = flat_model.component(cont_var.parent().name()).variable(cont_var.name())
                    splt = ext_variable.split('#')
                    ext_variable_type = splt[0]
                    ext_variable_idx = splt[1]
                    ext_variable_inlet_or_outlet = splt[2]
                    ext_variable_flow_or_pressure_bc = splt[3]
            
                    aev = AnalyserExternalVariable(input_variable)
                    a.addExternalVariable(aev)
                    external_variable_info.append({
                        'variable': input_variable,
                        'port_variable': output_variable,
                        'ext_variable_idx': ext_variable_idx,
                        'ext_variable_type': ext_variable_type,
                        'coupled_to_type': 'FV_1d' if ext_variable_type.startswith('FV_1d') else 'unknown',
                        'inlet_or_outlet': ext_variable_inlet_or_outlet,
                        'bc_inlet0_or_outlet1': 0 if ext_variable_inlet_or_outlet == 'inlet' else 1,
                        'flow_or_pressure_bc': ext_variable_flow_or_pressure_bc,
                        'analyser_variable': aev,
                        'control_variable': cont_variable,
                        'control_variable_index': None,
                        'variable_type': 'input'
                    })
            elif uri == self.output_variable_uri:
                v = vv
                output_variable = flat_model.component(v.parent().name()).variable(v.name())
                external_variable_info.append({
                    'variable': output_variable,
                    'variable_type': 'output'
                })
                # TODO I need to include specification of the external variable here? i.e. the name 
                # or index of the variable from the Cpp code?

        a.analyseModel(flat_model)
        analysed_model = a.model()

        libcellml_utils.print_issues(a)
        print(analysed_model.type())
        if analysed_model.type() != AnalyserModel.Type.ODE:
            print("model is not a valid ODE model, aborting...")
            exit()
        # if not analysed_model.isValid():
        #     print("model is not valid, aborting...")
        #     exit()

        # get the information for the variables to delay
        for ext_variable in external_variable_info:
            ev = ext_variable['variable']
            avs = analysed_model.variables()

            for av in avs:
                v = av.variable()
                if analysed_model.areEquivalentVariables(v, ext_variable['variable']):
                    ext_variable['variable_index'] = av.index()
                    ext_variable['state_or_variable'] = 'variable'

                if 'port_variable' in ext_variable.keys():
                    if analysed_model.areEquivalentVariables(v, ext_variable['port_variable']):
                        ext_variable['port_variable_index'] = av.index()

                if 'control_variable' in ext_variable.keys() and ext_variable['control_variable'] != None:
                    if analysed_model.areEquivalentVariables(v, ext_variable['control_variable']):
                        ext_variable['control_variable_index'] = av.index()

                if ext_variable['variable_type'] == 'delay':
                    if analysed_model.areEquivalentVariables(v, ext_variable['delay_variable']):
                        ext_variable['delay_variable_index'] = av.index()
                    if analysed_model.areEquivalentVariables(v, ext_variable['delay_amount_variable']):
                        ext_variable['delay_amount_index'] = av.index()
            
            astates = analysed_model.states()
            for astate in astates:
                state = astate.variable()
                if state.name() == ext_variable['variable'].name(): 
                    ext_variable['state_index'] = astate.index()
                    ext_variable['state_or_variable'] = 'state'
                
                if 'port_variable' in ext_variable.keys():
                    if state.name() == ext_variable['port_variable'].name(): 
                        ext_variable['port_state_index'] = astate.index()
            

        # generate code from the analysed model
        gen = Generator()
        # using the C profile to generate C code
        # TODO BIG CHANGE
        # TODO I should create my own profile with most of the below code so I don't have to change it when changes are made to libcellml
        profile = GeneratorProfile(GeneratorProfile.Profile.C)
        profile.setInterfaceFileNameString(f'{self.output_cpp_file_name}.h')
        gen.setProfile(profile)
        gen.setModel(analysed_model)

        preHeaderStuff = f"""
        #include <stdlib.h>
        #include <memory>
        #include <map>
        #include <string>
        #include <sstream>
        #include <functional>

        """
        for external_header in self.external_headers:
            preHeaderStuff += f'#include "{external_header}"\n'

        if self.solver == 'CVODE':
            preHeaderStuff += """
        #include <cvodes/cvodes.h>
        #include <nvector/nvector_serial.h>
        #include <sunlinsol/sunlinsol_dense.h> 

        """

        interFaceCodePreClass = ''
        interFaceCodeInClass = ''
        pre_class = True
        for line in gen.interfaceCode().split('\n'):
            if 'extern ' in line:
                if 'VERSION' in line:
                    line = line.replace('VERSION', 'VERSION_')
                if 'VERSION' not in line:
                    line = line.replace('extern ', '')
            if '(* ExternalVariable)' in line:
                continue
            if ', ExternalVariable externalVariable' in line:
                line = line.replace(', ExternalVariable externalVariable', '')
            if 'const VariableInfo' in line:
                line = 'static ' + line
            if line.startswith('const size_t STATE_COUNT'):
                pre_class = False
            if pre_class:
                interFaceCodePreClass += line + '\n'
            else:
                interFaceCodeInClass += '    ' + line + '\n'
                

        classInitHeader = ""

        if self.solver == 'CVODE':
            classInitHeader += """
//forward declare the userOdeData class
class UserOdeData;
        """

        classInitHeader += """
// this is the the 0D model class definition
// it contains the model variables and the functions to compute the rates and variables
class Model0d {
public:
    // constructor
    Model0d();
    // destructor
    ~Model0d();
        """

        otherHeaderInits = f"""
    double externalVariable(double voi, double *states, double *rates, double *variables, size_t index);
    void computeNonExternalVariables(double voi, double *states, double *rates, double *variables);
    void solveOneStep(double dt);
    double voi;
    double dt;
    double eps;
    double * states;
    double * rates;
    double * variables;
    double time_dof_0d;
        """

        bufferPartsHeader = f"""
    bool buffersInitialised = false;

    void initBuffers(double dt);
    
    std::map<std::string, circular_buffer*> circular_buffer_dict;
    void storeBufferVariable(int index, double value);
    double getBufferVariable(int index, double dt_fraction, int pop_bool);
        """

        if self.couple_to_1d:
            otherHeaderInits += """
    std::map<std::string, std::map<std::string, int>> cellml_index_to_vessel1d_info;
    std::vector<std::map<std::string, int>> vessel1d_info;
    Model1d * model1d_ptr;
    int num_vessel1d_connections;
            """


        if self.solver == 'RK4':
            otherHeaderInits += """
    double * k1;
    double * k2;
    double * k3;
    double * k4;
    double * temp_states;
            """

        if self.solver == 'CVODE':
            otherHeaderInits += """
    double voiEnd;
    SUNContext context;
    void *solver;
    N_Vector y; 
    UserOdeData *userData= nullptr;
    SUNMatrix matrix;
    SUNLinearSolver linearSolver;
        """

        otherHeaderInits += """
    using FunctionType = std::function<void(double, double*, double*, double*)>;
        """

        # using computeRatesType = void (*)(double, double *, double *, double *);
        # static int func(double voi, N_Vector y, N_Vector ydot, void *userData);

        classFinisherHeader = """
};
        """

        preSourceStuff = f"""
#include <stddef.h>
#include <stdio.h>
#include <iostream>
        """
        if self.solver == 'CVODE':
            preSourceStuff += """
#include <cvodes/cvodes.h>
#include <nvector/nvector_serial.h>
#include <sunlinsol/sunlinsol_dense.h> 
        """

        # split implementation code in two so we can change it into a class
        preClassStuff = ''
        classInit = """
Model0d::Model0d() :
        """
        postClassInit = ''
        classInit += """    voi(0.0),
    dt(0.0),
    eps(1e-08),
    states(nullptr),
    rates(nullptr),
    variables(nullptr),
    time_dof_0d(-1),
        """

        if self.solver == 'RK4':
            classInit += """    k1(nullptr),
    k2(nullptr),
    k3(nullptr),
    k4(nullptr),
    temp_states(nullptr),
        """
            
        if self.couple_to_1d:
            classInit += """    model1d_ptr(nullptr),
        """
        # create mapping between external variable index and 1D vessel and 
        # BC information

        if self.couple_to_1d:
            classInit += """    cellml_index_to_vessel1d_info{
            """
            num_vessel1d_connections = 0 
            for idx, ext_variable in enumerate(external_variable_info):
                if idx != 0:
                    classInit += ',\n'
                if ext_variable['coupled_to_type'] == 'FV_1d':
                    classInit += f"""       {{ "{ext_variable["variable_index"]}", {{ 
                        {{ "vessel1d_idx", {ext_variable["ext_variable_idx"]}}}, 
                        {{ "bc_inlet0_or_outlet1", {ext_variable["bc_inlet0_or_outlet1"]} }}"""
                    if 'port_variable_index' in ext_variable.keys():
                        classInit += f""",
                        {{ "port_variable_idx", {ext_variable["port_variable_index"]}}}"""
                    elif 'port_state_index' in ext_variable.keys():
                        classInit += f""",
                        {{ "port_state_idx", {ext_variable["port_state_index"]}, }}"""
                    if 'control_variable_index' in ext_variable.keys() and ext_variable['control_variable_index'] != None:
                        classInit += f""",
                        {{ "control_variable_idx", {ext_variable["control_variable_index"]}, }}"""
                    classInit += """}}"""
                    num_vessel1d_connections += 1
            classInit += '},\n'
            classInit += f'    num_vessel1d_connections({num_vessel1d_connections}),\n'

            # now create a vessel1d_info vector of dicts for each connected variable
            classInit += """    vessel1d_info{ """
            for idx, ext_variable in enumerate(external_variable_info):
                if idx != 0:
                    classInit += ',\n'
                if ext_variable['coupled_to_type'] == 'FV_1d':
                    classInit += f"""       
                    {{  {{ "cellml_idx", {ext_variable["variable_index"]}}}, 
                        {{ "vessel1d_idx", {ext_variable["ext_variable_idx"]}}}, 
                        {{ "bc_inlet0_or_outlet1", {ext_variable["bc_inlet0_or_outlet1"]} }} """
                    if 'port_variable_index' in ext_variable.keys():
                        classInit += f""",
                        {{ "port_variable_idx", {ext_variable["port_variable_index"]}, }}
                        """
                    elif 'port_state_index' in ext_variable.keys():
                        classInit += f""",
                        {{ "port_state_idx", {ext_variable["port_state_index"]}, }}
                        """
                    if 'control_variable_index' in ext_variable.keys() and ext_variable['control_variable_index'] != None:
                        classInit += f""",
                        {{ "control_variable_idx", {ext_variable["control_variable_index"]}, }}
                        """
                    classInit += "}"
            classInit += '    },\n'


        pre_class = True
        in_class_init = False

        # TODO The below is all really susceptible to changes in libcellml, I should create my own profile
        for line in gen.implementationCode().split('\n'):
            if line.startswith('const size_t STATE_COUNT'):
                pre_class = False
                in_class_init = True

            if 'VERSION' in line:
                line = line.replace('VERSION', 'VERSION_')
            # if line.startswith('double * createStatesArray'):
            if line.startswith('double max'):
                in_class_init = False

            if pre_class:
                preClassStuff += line + '\n'
            elif in_class_init:
                if 'const size_t STATE_COUNT' in line:
                    line = line.replace('const size_t STATE_COUNT = ', 'STATE_COUNT(')
                    line = line.replace(';', '),')
                if 'const size_t VARIABLE_COUNT' in line:
                    line = line.replace('const size_t VARIABLE_COUNT = ', 'VARIABLE_COUNT(')
                    line = line.replace(';', ') {')
                classInit += '    ' + line + '\n'
            else:
                if 'createStatesArray' in line:
                    line = line.replace('createStatesArray', 'Model0d::createStatesArray')
                if 'createVariablesArray' in line:
                    line = line.replace('createVariablesArray', 'Model0d::createVariablesArray')
                if 'deleteArray' in line:
                    line = line.replace('deleteArray', 'Model0d::deleteArray')
                if 'initialiseVariables' in line:
                    line = line.replace('initialiseVariables', 'Model0d::initialiseVariables')
                if 'computeComputedConstants' in line:
                    line = line.replace('computeComputedConstants', 'Model0d::computeComputedConstants')
                if 'computeRates' in line:
                    line = line.replace('computeRates', 'Model0d::computeRates')
                if 'computeVariables' in line:
                    line = line.replace('computeVariables', 'Model0d::computeVariables')
                if 'voi' in line:
                    line = line.replace('void', 'XXX')
                    line = line.replace('voi', 'voi_')
                    line = line.replace('XXX', 'void')
                if 'states' in line:
                    line = line.replace('states', 'states_')
                if 'rates' in line:
                    line = line.replace('rates', 'rates_')
                if 'variables' in line:
                    line = line.replace('variables', 'variables_')
                if 'ExternalVariable externalVariable' in line:
                    line = line.replace(', ExternalVariable externalVariable', '')
                
                postClassInit += line + '\n'
            

        classInit += f"""
            states = createStatesArray();
            rates = createStatesArray(); // same size as states, should really be different function
            variables = createVariablesArray();
        """

        if self.solver == 'RK4':
            classInit += """
            k1= createStatesArray(); // same size as states, should really be different function
            k2= createStatesArray(); // same size as states, should really be different function
            k3= createStatesArray(); // same size as states, should really be different function
            k4= createStatesArray(); // same size as states, should really be different function
            temp_states = createStatesArray();
            """

        if self.solver == 'CVODE':
            # TODO do I have to calculate variables/states first?
            classInit += """
            // Create our SUNDIALS context.
            SUNContext_Create(NULL, &context);

            // Create our CVODE solver.
            solver = CVodeCreate(CV_BDF, context);

            // Initialise our CVODE solver.

            y = N_VMake_Serial(STATE_COUNT, states, context);

            CVodeInit(solver, func, voi, y);

            // Set our user data.

            userData = new UserOdeData(variables, std::bind(&Model0d::computeRates, this, std::placeholders::_1, 
                                                            std::placeholders::_2, std::placeholders::_3, 
                                                            std::placeholders::_4));

            CVodeSetUserData(solver, userData);

            // Set our maximum number of steps.

            CVodeSetMaxNumSteps(solver, 99999); // TODO get from user_inputs.yaml
            
            CVodeSetMaxStep(solver, 0.0001); // TODO get from user_inputs.yaml

            // Set our linear solver.

            matrix = SUNDenseMatrix(STATE_COUNT, STATE_COUNT, context);
            linearSolver = SUNLinSol_Dense(y, matrix, context);

            CVodeSetLinearSolver(solver, linearSolver, matrix);

            // Set our relative and absolute tolerances.

            CVodeSStolerances(solver, 1e-7, 1e-9); // TODO get from user_inputs.yaml
        """

        classInit += """
        }

        Model0d::~Model0d() {
            // Clean up after ourselves.

            deleteArray(states);
            deleteArray(rates);
            deleteArray(variables);

        """
        if self.solver == 'RK4':
            classInit += """
            deleteArray(k1);
            deleteArray(k2);
            deleteArray(k3);
            deleteArray(k4);
            deleteArray(temp_states);
        """
        if self.solver == 'CVODE':
            classInit += """
            // Clean up after ourselves.

            SUNLinSolFree(linearSolver);
            SUNMatDestroy(matrix);
            N_VDestroy_Serial(y);
            CVodeFree(&solver);
            SUNContext_Free(&context);
        """
        classInit += """
        }
        """
        # todo what else should I do in the destructor?

        # find the function for computeVariables to create a 
        # compute non_external variables function
        non_external_variables_function = \
                'void Model0d::computeNonExternalVariables(double voi_, double *states_, double *rates_, double *variables_) \n'
        startFunction = False
        for line in gen.implementationCode().split('\n'):
            if line.startswith('void computeVariables'):
                startFunction = True
            if startFunction:
                if line.startswith('}'):
                    non_external_variables_function += line + '\n'
                    break
                if 'externalVariable' in line:
                    continue
                if 'voi' in line:
                    line = line.replace('void', 'XXX')
                    line = line.replace('voi', 'voi_')
                    line = line.replace('XXX', 'void')
                if 'states' in line:
                    line = line.replace('states', 'states_')
                if 'rates' in line:
                    line = line.replace('rates', 'rates_')
                if 'variables' in line:
                    line = line.replace('variables', 'variables_')
                non_external_variables_function += line + '\n'


        # and generate a function to compute external variables
        computeEV = f"""
double Model0d::externalVariable(double voi_, double *states_, double *rates_, double *variables_, size_t index)
{{
    // if voi is zero we may need to calculate the non_external_variables
    // because they can be needed for the external variables
    if (voi_ == 0.0) {{
        computeNonExternalVariables(voi_, states_, rates_, variables_);
    }}
        """
        for ext_variable in external_variable_info:
            print(ext_variable.keys())

            # variable or state index
            if ext_variable["state_or_variable"] == "state":
                state_index = ext_variable['state_index']
            else:
                variable_index = ext_variable['variable_index']

            # TODO Finbar: for each external variable, include a label for its corresponding port
            # variable, ie when we need qBC, we give P_C and possibly R_T1.
            # TODO Finbar: create a config file which details the types of external variables we 
            # can have and their corresponding port variables. 
            # a new one will be created for each type of coupling.

            # TODO if time_dof_0d = 0 then we don't store the variables in the buffer, only store when it equals -1
            # set up so that when time_dof_0d is not -1 we don't step forward in the buffer, we only interpolate 
            # from it, we only step forward in the buffer when time_dof_0d is -1
            # use a time_0d variable to check where voi is in terms of time_0d
            # TODO we could run a rough CVODE where we only save at time_output points, then linearly interpolate...
            #  this would be quite easy to implement.. unsure about stability.
            if ext_variable['variable_type'] == 'delay':
                    
                delay_variable_index = ext_variable['delay_variable_index']
                delay_amount_index = ext_variable['delay_amount_index']
                computeEV += f'  double dt_fraction = (voi_ - voi) / dt;\n'
                computeEV += f'  double value;\n'
                computeEV += f'  if (index == {delay_variable_index}) {{\n'
                computeEV += f'    if (voi_ < variables_[{delay_amount_index}]) {{\n'
                computeEV += f'      if (buffersInitialised != true) {{return 0.0;}};\n'
                computeEV += f'      if (time_dof_0d == -1) {{;\n'
                if ext_variable["state_or_variable"] == "state":
                    computeEV += f'        storeBufferVariable({delay_variable_index}, states_[{state_index}]);\n'
                else:
                    computeEV += f'        storeBufferVariable({delay_variable_index}, variables_[{variable_index}]);\n'
                computeEV += f'      }};\n'
                computeEV += f'      return 0.0;\n'
                computeEV += f'    }} else {{;\n'
                computeEV += f'      if (time_dof_0d == -1) {{;\n'
                computeEV += f'        value = getBufferVariable({delay_variable_index}, 1.0, 1);\n'
                computeEV += f'      }} else {{;\n'
                computeEV += f'        value = getBufferVariable({delay_variable_index}, dt_fraction, 0);\n'
                computeEV += f'      }}\n'
                computeEV += f'      // save the current value of the variable to the circle buffer\n'
                computeEV += f'      if (time_dof_0d == -1) {{;\n'
                if ext_variable["state_or_variable"] == "state":
                    computeEV += f'        storeBufferVariable({delay_variable_index}, states_[{state_index}]);\n'
                else:
                    computeEV += f'        storeBufferVariable({delay_variable_index}, variables_[{variable_index}]);\n'
                computeEV += f'      }};\n'
                computeEV += f'      return value;\n'
                computeEV += f'    }};\n'
                computeEV += f'  }}\n'

            elif ext_variable['variable_type'] == 'input':
                if ext_variable['state_or_variable'] == 'state':
                    print('Input BC variable can not be a state variable, exiting')
                    exit()
                    
                elif ext_variable['state_or_variable'] == 'variable':
                    if ext_variable['coupled_to_type'] == 'FV_1d':
                        computeEV += f'  if (index == {variable_index}) {{\n'
                        if ext_variable['flow_or_pressure_bc'] == 'flow':
                            computeEV += f'    int vessel1d_idx = cellml_index_to_vessel1d_info["{variable_index}"]["vessel1d_idx"];\n'
                            computeEV += f'    int inlet0_or_outlet1_bc = cellml_index_to_vessel1d_info["{variable_index}"]["bc_inlet0_or_outlet1"];\n'
                            if 'port_variable_index' in ext_variable.keys():
                                port_variable_index = ext_variable['port_variable_index']
                                if 'control_variable_index' in ext_variable.keys() and ext_variable['control_variable_index'] != None:
                                    control_variable_index = ext_variable['control_variable_index']
                                    computeEV += f'    double value = get_model1d_flow(vessel1d_idx, variables_[{port_variable_index}], inlet0_or_outlet1_bc, variables_[{control_variable_index}]);\n'
                                else:
                                    computeEV += f'    double value = get_model1d_flow(vessel1d_idx, variables_[{port_variable_index}], inlet0_or_outlet1_bc);\n' 
                            elif 'port_state_index' in ext_variable.keys():
                                port_state_index = ext_variable['port_state_index']
                                if 'control_variable_index' in ext_variable.keys() and ext_variable['control_variable_index'] != None:
                                    control_variable_index = ext_variable['control_variable_index']
                                    computeEV += f'    double value = get_model1d_flow(vessel1d_idx, states_[{port_state_index}], inlet0_or_outlet1_bc, variables_[{control_variable_index}]);\n'
                                else:
                                    computeEV += f'    double value = get_model1d_flow(vessel1d_idx, states_[{port_state_index}], inlet0_or_outlet1_bc);\n'
                        elif ext_variable['flow_or_pressure_bc'] == 'pressure':
                            computeEV += f'    int vessel1d_idx = cellml_index_to_vessel1d_info["{variable_index}"]["vessel1d_idx"];\n'
                            computeEV += f'    int inlet0_or_outlet1_bc = cellml_index_to_vessel1d_info["{variable_index}"]["bc_inlet0_or_outlet1"];\n'
                            if 'port_variable_index' in ext_variable.keys():
                                port_variable_index = ext_variable['port_variable_index']
                                computeEV += f'    double value = get_model1d_pressure(vessel1d_idx, variables_[{port_variable_index}], inlet0_or_outlet1_bc);\n' 
                            elif 'port_state_index' in ext_variable.keys():
                                port_state_index = ext_variable['port_state_index']
                                computeEV += f'    double value = get_model1d_pressure(vessel1d_idx, states_[{port_state_index}], inlet0_or_outlet1_bc);\n'
                            print("Currently we don't implement the pressure coupling. ",
                                  "As it would require solving a DAE in CellML")
                            exit()
                        computeEV += f'    return value;\n'
                        computeEV += f'  }}\n'

        computeEV += f"""
return 0.0;
}}
        """

        if self.couple_to_1d:
            # external interaction functions
            externalInteractionFunctions = """
void Model0d::connect_to_model1d(Model1d* model1d){
    model1d_ptr = model1d; 
}
            """
            if 'control_variable_index' in ext_variable.keys() and ext_variable['control_variable_index'] != None:
                externalInteractionFunctions += """
double Model0d::get_model1d_flow(int vessel_idx, double p_C, int input0_output1_bc, double R_T1_wCont){
                """
            else:
                externalInteractionFunctions += """
double Model0d::get_model1d_flow(int vessel_idx, double p_C, int input0_output1_bc){
                """
            externalInteractionFunctions += """
    // get the flow from the 1d model

    // get qBC and step forward the fluctuation
            """
            
            if self.solver == 'RK4':
                externalInteractionFunctions += """
    int add_to_fluct = 1;
    double time_int_weight = 0.0;
    double time_0d = 0.0;
    if (time_dof_0d == 0) {
        time_int_weight = 1.0/6.0;
        time_0d = voi;
    }
    else if (time_dof_0d == 1 || time_dof_0d == 2) {
        time_int_weight = 1.0/3.0;
        time_0d = voi + dt/2.0;
    }
    else if (time_dof_0d == 3) {
        time_int_weight = 1.0/6.0;
        time_0d = voi + dt;
    }
    else if (time_dof_0d == -1) {
        time_int_weight = 0.0;
        add_to_fluct = 0;
        time_0d = voi + dt;
    }
    else {
        std::cout << "time_dof_0d is not set correctly" << std::endl;
        exit(1);
    }
            """

            if 'control_variable_index' in ext_variable.keys() and ext_variable['control_variable_index'] != None:
                externalInteractionFunctions += """
    double flow = model1d_ptr->getqBCAndAddToFluctWrap(vessel_idx, add_to_fluct, 
                                                time_0d, time_int_weight, 
                                                p_C, input0_output1_bc, R_T1_wCont)/pow(10,6); 
    // divide by 10^6 to get m3/s from ml/s
    return flow;
}
                """
            else:
                externalInteractionFunctions += """
    double flow = model1d_ptr->getqBCAndAddToFluctWrap(vessel_idx, add_to_fluct, 
                                                time_0d, time_int_weight, 
                                                p_C, input0_output1_bc)/pow(10,6); 
    // divide by 10^6 to get m3/s from ml/s
    return flow;
}
                """
                
            # externalInteractionFunctions += """
                
            #     // below is what I did previously
            #     // double * all_vals = (double *) malloc(10*sizeof(double));
            #     // if (input0_output1_bc == 0){
            #     //     // TODO Finbar, should be evalSolSpaceTime, where the time is an input and 
            #     //     // it interpolates the solution
            #     //     model1d_ptr->evalSol(vessel_idx, 0.0, 0, all_vals); 
            #     // } else {
            #     //     double xSample = model1d_ptr->vess[vessel_idx].L;
            #     //     int iSample = model1d_ptr->vess[vessel_idx].NCELLS - 1;
            #     //     model1d_ptr->evalSol(vessel_idx, xSample, iSample, all_vals); 
            #     // }
            #     // double flow = all_vals[1]/pow(10,6); // index 1 is the flow, divide by 10^6 to get m3/s from ml/s
            #     // // TODO make the conversion of units more generic
            #     // // std::cout << "flow: " << std::scientific << flow << std::endl;
            #     return flow; // 
            # } 
            # """

            # Currently we don't implement the pressure coupling
            # double Model0d::get_model1d_pressure(int vessel_idx, double qBC, int input0_output1_bc){
            #     // get the pressure from the 1d model
            # """ 
            
            # if self.solver == 'RK4':
            #     externalInteractionFunctions += """
                
            #     int add_to_fluct = 1;
            #     double time_int_weight = 0.0;
            #     double time_0d = 0.0;
            #     if (time_dof_0d == 0) {
            #         time_int_weight = 1.0/6.0;
            #         time_0d = voi;
            #     }
            #     else if (time_dof_0d == 1 || time_dof_0d == 2) {
            #         time_int_weight = 1.0/3.0;
            #         time_0d = voi + dt/2.0;
            #     }
            #     else if (time_dof_0d == 3) {
            #         time_int_weight = 1.0/6.0;
            #         time_0d = voi + dt;
            #     }
            #     else if (time_dof_0d == -1) {
            #         time_int_weight = 0.0;
            #         add_to_fluct = 0;
            #         time_0d = voi + dt;
            #     }
            #     else {
            #         std::cout << "time_dof_0d is not set correctly" << std::endl;
            #         exit(1);
            #     }
            # """

            # externalInteractionFunctions += """
            #     double pressure = model1d_ptr->getPressureAndAddToFluct(vessel_idx, add_to_fluct, 
            #                                                 time_0d, time_int_weight, 
            #                                                 qBC, input0_output1_bc)/10; 
            #     # divide by 10 to go from dynes/cm^2 to Pa
            #     return pressure;

            #     // below is what I did previously
            #     // double * all_vals = (double *) malloc(8*sizeof(double));
            #     // if (input0_output1_bc == 0){
            #     //     model1d_ptr->sampleMid(vessel_idx, all_vals, 0.0); 
            #     // } else {
            #     //     double xSample = model1d_ptr->vess[vessel_idx].L;
            #     //     model1d_ptr->sampleMid(vessel_idx, all_vals, xSample); 
            #     // }
            #     // return all_vals[4]; // index 4 is the pressure 
            # }
            # """

        else:
            externalInteractionFunctions = ""

        externalInteractionFunctions += """

void Model0d::initialiseVariablesAndComputeConstants() {
    initialiseVariables(voi, states, rates, variables);
    computeComputedConstants(variables);
    computeRates(voi, states, rates, variables);
    computeVariables(voi, states, rates, variables);
        """
        if self.solver == 'CVODE':
            externalInteractionFunctions += """
    // reinitialise the CVODE solver with the initialised state variables 
    y = N_VMake_Serial(STATE_COUNT, states, context);

    CVodeReInit(solver, voi, y);
        """

        externalInteractionFunctions += """
}
        """


        # external interaction headers 
        externalInteractionHeaders= """
    void initialiseVariablesAndComputeConstants();
        """
        if self.couple_to_1d:
            externalInteractionHeaders += """
    void connect_to_model1d(Model1d* model1d);
            """
            if 'control_variable_index' in ext_variable.keys() and ext_variable['control_variable_index'] != None:
                externalInteractionHeaders += """
    double get_model1d_flow(int vessel_idx, double p_C, int input0_output1_bc, double R_T1_wCont);
    double get_model1d_pressure(int vessel_idx, double qBC, int input0_output1_bc);
                """
            else:
                externalInteractionHeaders += """
    double get_model1d_flow(int vessel_idx, double p_C, int input0_output1_bc);
    double get_model1d_pressure(int vessel_idx, double qBC, int input0_output1_bc);
                """
                

        # TODO make the integration scheme implementation general
        solveOneStepFunction = """
void Model0d::solveOneStep(double dt_) {
    dt = dt_;
        """
        if self.solver == 'forward_euler':
            solveOneStepFunction += """
            
    time_dof_0d = 0;
    computeRates(voi, states, rates, variables);

    for (size_t i = 0; i < STATE_COUNT; ++i) {
        // simple forward Euler integration
        states[i] = states[i] + dt * rates[i];
    time_dof_0d = -1;
    computeVariables(voi, states, rates, variables);
    voi += dt;
    }
}
            """
        elif self.solver == 'RK4':
            solveOneStepFunction += """
    // RK4 integration
    // first step: calculate k1
    for (size_t i = 0; i < STATE_COUNT; ++i) {
        temp_states[i] = states[i];
    }
    time_dof_0d = 0;
    computeRates(voi, temp_states, k1, variables);

    // second step: calculate k2
    for (size_t i = 0; i < STATE_COUNT; ++i) {
        temp_states[i] = states[i] + dt/2.0 * k1[i];
    }
    time_dof_0d = 1;
    computeRates(voi+dt/2.0, temp_states, k2, variables);
    
    // third step: calculate k3
    for (size_t i = 0; i < STATE_COUNT; ++i) {
        temp_states[i] = states[i] + dt/2.0 * k2[i];
    }
    time_dof_0d = 2;
    computeRates(voi+dt/2.0, temp_states, k3, variables);
    
    // third step: calculate k4
    for (size_t i = 0; i < STATE_COUNT; ++i) {
        temp_states[i] = states[i] + dt * k3[i];
    }
    time_dof_0d = 3;
    computeRates(voi+dt, temp_states, k4, variables);

    for (size_t i = 0; i < STATE_COUNT; ++i) {
        rates[i] = 1.0/6.0 * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
        states[i] = states[i] + dt * rates[i];
    }
    // when we set time_dof_0d to -1, we only get q_BC
    // from the ones evaluated at the 4 time dofs. we dont step the fluct
    time_dof_0d = -1;
    computeVariables(voi, states, rates, variables);
    voi += dt;
        """
            if self.couple_to_1d:
                solveOneStepFunction += """
        // TODO Finbar, is the below correct? it was used in the 1d code.
    model1d_ptr->correctTimeLTS(voi,model1d_ptr->dtMaxLTS);
}
        """
            else:
                solveOneStepFunction += """
}
        """

        elif self.solver == 'CVODE':
            solveOneStepFunction += """ 

    voiEnd = voi + dt;
    // CVodeSetStopTime(solver, voiEnd);

    CVode(solver, voiEnd, y, &voi, CV_NORMAL);

    // Compute our variables.

    states = N_VGetArrayPointer_Serial(y);
    computeVariables(voiEnd, states, rates, variables);
    voi += dt;
    }
            """
            

        # typedef void (*computeRatesType)(double, double *, double *, double *);
        userDataHeader = """
class UserOdeData
{
public:

    explicit UserOdeData(double *pVariables, Model0d::FunctionType pComputeRates);

    double* variables() const;
    Model0d::FunctionType computeRates() const;

private:
    double *mVariables;
    Model0d::FunctionType mComputeRates;
};

        """

        UserDataCC = """
UserOdeData::UserOdeData(double *pVariables, Model0d::FunctionType pComputeRates) :
    mVariables(pVariables),
    mComputeRates(pComputeRates)
{
}

//==============================================================================

double * UserOdeData::variables() const
{
    // Return our algebraic array

    return mVariables;
}

//==============================================================================

Model0d::FunctionType UserOdeData::computeRates() const
{
    // Return our compute rates function

    return mComputeRates;
}
        """

        if self.solver == 'CVODE':
            funcCC = """
int func(double voi_, N_Vector y, N_Vector yDot, void *userData)
{
    UserOdeData *realUserData = static_cast<UserOdeData*>(userData);
    realUserData->computeRates()(voi_, N_VGetArrayPointer_Serial(y), 
                            N_VGetArrayPointer_Serial(yDot), realUserData->variables());
    return 0;
}
            
        """

        circularBufferHeader = f"""

class circular_buffer {{
private:
    int size;
    int head;
    int tail;
    double *buffer;

public:
    circular_buffer(int size);
    void put(double value);
    double get();
    double access(int index_back);
}};
        """

        circularBuffer = f"""
// circular buffer implementation
circular_buffer::circular_buffer(int size)
{{
this->size = size;
this->head = 0;
this->tail = 0;
this->buffer = new double[size];
}}

void circular_buffer::put(double value)
{{
buffer[head] = value;
head = (head + 1) % size;
// if (head == tail)
//  tail = (tail + 1) % size;
}}

double circular_buffer::get()
{{
double value = buffer[tail];
tail = (tail + 1) % size;
return value;
}}

double circular_buffer::access(int index_back)
{{
double value = buffer[tail + index_back];
return value;
}}

        """


        # generate a global singleton class to store external variables
        bufferParts = f"""

void Model0d::initBuffers(double dt)
{{
// Here I need to initialise circular buffers for each external variable   
        """
        if len(variables_to_delay_info) > 0:
            for ext_variable in external_variable_info:
                print(ext_variable.keys())
                if ext_variable['variable_type'] == 'delay':
                    index = ext_variable['delay_variable_index']
                    delay_amount_index = ext_variable['delay_amount_index']
                    bufferParts += f'  double buffer_size = static_cast<double>(variables[{delay_amount_index}])/dt;\n' 
                    bufferParts += f'  circular_buffer* var_circular_buffer;\n'
                    bufferParts += f'  var_circular_buffer = new circular_buffer(buffer_size);\n'
                    bufferParts += f'  std::ostringstream ss;\n'
                    bufferParts += f'  ss << {index};\n'
                    bufferParts += f'  circular_buffer_dict[ss.str()] = var_circular_buffer;\n'
                    bufferParts += f'  buffersInitialised = true;\n'
                    bufferParts += f'  }}\n'
        else:
            bufferParts += f'  buffersInitialised = true;\n'
            bufferParts += f'  }}\n'

        bufferParts += f""" 

void Model0d::storeBufferVariable(int index, double value)
{{
// Here I need to store the value in the correct circular buffer    
std::ostringstream ss;
ss << index;
circular_buffer_dict[ss.str()]->put(value);
}}

double Model0d::getBufferVariable(int index, double dt_fraction, int pop_bool)
{{
// Here I need to get the value from the correct circular buffer
std::ostringstream ss;
ss << index;
if (pop_bool == 1) {{
    return circular_buffer_dict[ss.str()]->get();
}} else {{
    double value_one_back = circular_buffer_dict[ss.str()]->access(-1);
    double value = circular_buffer_dict[ss.str()]->access(0);
    double interp_value = value_one_back + dt_fraction*(value - value_one_back);
    return interp_value;
}}
}}

        """

        # TODO modify below with respect to circulatory_autogen inputs
        # When coupling with a Cpp model that does the simulation, this isn't needed
        mainScript = """
int main(void){
    double end_time = 5.0;
    double dt = 0.1;
    Model0d model0d_inst;
    model0d_inst.initialiseVariablesAndComputeConstants();
    model0d_inst.initBuffers(dt);
    double eps = 1e-12;

    while (model0d_inst.voi < end_time-eps) {
        model0d_inst.solveOneStep(dt);
        std::cout << "time: " << model0d_inst.voi << " V: " << model0d_inst.states[0] << std::endl;
        std::cout << "time: " << model0d_inst.voi << " V_delay: " << model0d_inst.variables[2] << std::endl;
    }

    // TODO autogenerate a dict of variable names to print with variables

    printf("Final values:");
    printf("  time: ");
    printf("%f", model0d_inst.voi);
    printf("  states:");
    for (size_t i = 0; i < model0d_inst.STATE_COUNT; ++i) {
        printf("%f\\n", model0d_inst.states[i]);
    }
    printf("  variables:");
    for (size_t i = 0; i < model0d_inst.VARIABLE_COUNT; ++i) {
        printf("%f\\n", model0d_inst.variables[i]);
    }

return 0;
}
        """

        # save header to file
        with open(os.path.join(self.cpp_generated_models_dir, f'{self.output_cpp_file_name}.h'), 'w') as f:

            f.write(preHeaderStuff) 
            if len(variables_to_delay_info) > 0:
                f.write(circularBufferHeader)
            f.write(interFaceCodePreClass)
            f.write(classInitHeader)
            f.write(interFaceCodeInClass)
            f.write(externalInteractionHeaders)
            f.write(otherHeaderInits)
            if len(variables_to_delay_info) > 0:
                f.write(bufferPartsHeader)
            f.write(classFinisherHeader)
            f.write(userDataHeader)


            
        # and save implementation to file
        with open(os.path.join(self.cpp_generated_models_dir, f'{self.output_cpp_file_name}.cc'), 'w') as f:

            f.write(preClassStuff)
            f.write(preSourceStuff) # this has to be below the #include "model.h" in implementationCode()
            if self.solver == 'CVODE':
                f.write(funcCC)
            f.write(classInit)
            f.write(postClassInit)
            f.write(non_external_variables_function)
            f.write(externalInteractionFunctions)
            f.write(computeEV)
            f.write(solveOneStepFunction)
            f.write(UserDataCC)
            
            if len(variables_to_delay_info) > 0:
                f.write(circularBuffer)
                f.write(bufferParts)
            if self.create_main:
                f.write(mainScript)
        
        print("Cpp files generated. Check they run properly")
        return True # TODO check if they run properly

class CVS1DCppGenerator(object):
    '''
    Generates Cpp files for 1D model. This is a wrapper around Lucas Muller's code.

    WARNING: THIS CPP 1D CODE IS CURRENTLY NOT OPEN SOURCE, SO THIS CLASS WON'T WORK UNTIL IT IS MADE OPEN SOURCE.
    '''


    def __init__(self, model, output_path, file_prefix):
        '''
        Constructor
        '''
        self.model = model
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        self.file_prefix = file_prefix
        self.user_resources_path = os.path.join(generators_dir_path, '../../resources')

    
    def generate_files(self):
        print("generating 1D Cpp files")

class CVSCoupledCppGenerator(object):
    '''
    Generates Cpp files for coupled 0D and 1D models.
    '''
    
    def __init__(self, model, output_path, file_prefix):
        '''
        Constructor
        '''
        self.model = model
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        self.file_prefix = file_prefix
        self.user_resources_path = os.path.join(generators_dir_path, '../../resources')
    
    def generate_files(self):
        zeroD_generator = CVS0DCppGenerator(self.model, self.output_path, self.file_prefix)
        zeroD_generator.generate_files()

        # currently we don't generate the 1D model, we assume it is already generated.
        # oneD_generator = CVS1DCppGenerator(self.model, self.output_path, self.file_prefix)
        # oneD_generator.generate_files()
        print("now do the coupling")

