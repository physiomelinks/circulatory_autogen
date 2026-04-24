'''
Created on 23/05/2023

@author: Finbar Argus
'''

import numpy as np
import re
import pandas as pd
import os
import sys
import json
from sys import exit
generators_dir_path = os.path.dirname(__file__)
root_dir = os.path.join(generators_dir_path, '../..')
sys.path.append(os.path.join(root_dir, 'src'))
from generators.CVSCellMLGenerator import CVS0DCellMLGenerator
from parsers.PrimitiveParsers import CSVFileParser
from generators.Python1DModelFilesGenerator import generate1DPythonModelFiles, generate1DPythonSimInitFile

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
                 solver='CVODE', dtSample=1e-3, dtSolver=1e-4, nMaxSteps=5000,
                 couple_to_1d=False, cpp_generated_models_dir=None,
                 model_1d_config_path=None, create_main_0d=False,
                 conn_1d_0d_info=None, DEBUG=False):
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
        self.couple_volume_sum = False
        if self.couple_to_1d:
            self.output_cpp_file_name = "model0d" # always the same, independent of model name, 
                                                    # to allow for coupling to cpp 1d model.
            self.conn_1d_0d_info = conn_1d_0d_info
        else:
            self.output_cpp_file_name = self.file_prefix
            self.conn_1d_0d_info = None

        self.external_headers = []
        
        if self.couple_to_1d:    
            self.model_1d_config_path = model_1d_config_path
            self.create_main = create_main_0d
            # self.external_headers += ['model1d.h']
        else:
            self.create_main = True
        
        if cpp_generated_models_dir is None:
            if self.couple_to_1d:
                self.cpp_generated_models_dir = self.generated_model_subdir + "_cpp"
            else:
                self.cpp_generated_models_dir = self.generated_model_subdir
        else:
            self.cpp_generated_models_dir = cpp_generated_models_dir
            if self.couple_to_1d and 'fvm' in cpp_generated_models_dir:
                self.external_headers += ['model1d.h']
        
        if not os.path.exists(self.cpp_generated_models_dir):
            os.mkdir(self.cpp_generated_models_dir)
        
        self.solver = solver
        self.dtSample = dtSample
        self.dtSolver = dtSolver
        self.nMaxSteps = nMaxSteps

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
        # this is for volume sum one-way coupling
        self.volume_sum_variables = []
        self.volume_sum_indeces = []
        self.volume_sum_types = []

        self.DEBUG = DEBUG

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
        # TODO this is randomly chosen for now, update this
        self.volume_sum_variable_uri = URIRef('http://identifiers.org/mamo/MAMO_0000060')

        self.g = Graph()
        
    def init_from_inp_data_dict(self, inp_data_dict):
        '''
        Initialise the generator from an input data dictionary
        '''
        raise NotImplementedError("Not implemented yet")

    def annotate_cellml(self):
        print("annotating CellML files, before Cpp generation...")
        #XXX Bea: is this really necessary???
        # TODO most of this function 
        # isn't needed for the generation anymore, but is it good to keep annotation in here? Probably.
        print(self.generated_model_file_path)
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
        print("All IDs reassigned")

        model_string = cellml.print_model(self.cellml_model)
        print("Now printing the model")
        print(model_string)
        
        # and save the updated model to a new file
        # - note, we need the model filename for making our annotations later
        with open(self.generated_wID_model_file_path, 'w') as f:
            f.write(model_string)
        print("Updated model saved to a new file")
        print('\n')
        
        # get the ID of the variables we want to annotate
        #XXX The below for loops are the only part of annotate_cellml that is needed to run generate_cpp
        print("Get the ID of the variables we want to annotate")
        for vessel_tup in self.model.vessels_df.itertuples():
            if "delay_info" in vessel_tup._fields:
                # if vessel_tup.delay_info is not "None":
                if vessel_tup.delay_info != "None":
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
                        
        print('Delayed variables :: DONE')
        
        fv1d_vessel_index_list = []
        for idx0, vessel_tup in enumerate(self.model.vessels_df.itertuples()):
            if vessel_tup.vessel_type!="FV1D_vessel":
                # print(vessel_tup.name, vessel_tup.inp_vessels, vessel_tup.out_vessels)
                for out_vessel in vessel_tup.out_vessels:
                    out_vessel_row = self.model.vessels_df.loc[self.model.vessels_df["name"] == out_vessel].squeeze()
                    if out_vessel_row["vessel_type"] == "FV1D_vessel":
                        # Finbar's format: get the index from the out_vessel name which is of format "FV1D_#", or "FV1D_##", or "FV1D_###", etc
                        # # fv1d_vessel_index = re.findall(r'\d+$', out_vessel)[0]
                        # fv1d_vessel_index = re.findall(r'\d+$', out_vessel) # "\d+" -> one or more digits, "$" -> at the end of the string
                        if out_vessel.startswith('FV1D_'):
                            fv1d_vessel_index = re.findall(r'\d+$', out_vessel)
                        else:
                            fv1d_vessel_index = []
                        
                        if len(fv1d_vessel_index)==0: #XXX then, use different indexing system
                            # print("WARNING: FV1D vessel has NO idx", out_vessel, fv1d_vessel_index)
                            idx1 = -1
                            for i in range(len(self.conn_1d_0d_info)):
                                conn1d0d = self.conn_1d_0d_info[str(i+1)]
                                if conn1d0d["vess0d_idx"] == idx0:
                                    idx1 = conn1d0d["vess1d_idx"]
                                    break
                            if idx1==-1:
                                print(f"ERROR: idx of 1D FV vessel {out_vessel} connected to 0D module {vessel_tup.name} not found.")
                                exit()
                            fv1d_vessel_index = str(idx1)
                            fv1d_vessel_index_list.append([out_vessel, fv1d_vessel_index])
                            print("FV1D vessel idx found", out_vessel, fv1d_vessel_index)

                        elif len(fv1d_vessel_index)==1:
                            fv1d_vessel_index = fv1d_vessel_index[0]
                            if len(fv1d_vessel_index) != 1:
                                print("ERROR: FV1D vessel idx is not written correctly",
                                    "should be of format FV1D_#, or FV1D_##, etc",
                                    out_vessel, fv1d_vessel_index)
                                exit()
                            
                            # best to do a sanity check, for compatibility between Finbar's old code and Bea's new code
                            idx1 = -1
                            for i in range(len(self.conn_1d_0d_info)):
                                conn1d0d = self.conn_1d_0d_info[str(i+1)]
                                if conn1d0d["vess0d_idx"] == idx0:
                                    idx1 = conn1d0d["vess1d_idx"]
                                    break
                            if idx1==-1:
                                print(f"ERROR: idx of 1D FV vessel {out_vessel} connected to 0D module {vessel_tup.name} not found.")
                                exit()
                            if fv1d_vessel_index != str(idx1):
                                print(f"ERROR: different idx of 1D FV vessel {out_vessel} found: {fv1d_vessel_index} {str(idx1)}")
                                exit()
                            
                            fv1d_vessel_index_list.append([out_vessel, fv1d_vessel_index])


                        self.connection_vessel_indices.append(fv1d_vessel_index)
                        self.connection_vessel_types.append("FV_1d") # TODO only option for now, can be extended to other kinds of coupling.
                        if vessel_tup.name == 'heart':
                            FV_resistance_port = -1
                            for exit_port in vessel_tup.exit_ports:
                                if exit_port["port_type"] == "vessel_port":

                                    if (('aorta' in out_vessel or 'aortic_root' in out_vessel) and exit_port["variables"][1] == 'u_root'):
                                        self.connection_vessel_inlet_or_outlet.append("outlet") 
                                        # self.connection_vessel_indices.append(fv1d_vessel_index) #XXX Bea: I think this was copied by mistake from above
                                        if vessel_tup.BC_type[1] == 'v':
                                            self.connection_vessel_flow_or_pressure_bc.append("flow") 
                                            self.port_input_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                            ).variable(exit_port["variables"][0]).id())
                                            self.port_output_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                            ).variable(exit_port["variables"][1]).id())
                                        elif vessel_tup.BC_type[1] == 'p':
                                            self.connection_vessel_flow_or_pressure_bc.append("pressure") 
                                            self.port_input_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                            ).variable(exit_port["variables"][1]).id())
                                            self.port_output_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                            ).variable(exit_port["variables"][0]).id())
                                        else:
                                            print(f"unknown BC type of {vessel_tup.BC_type} connecting to a "
                                                "1D FV model")
                                            
                                    elif ('par' in out_vessel and exit_port["variables"][1] == 'u_par'):
                                        self.connection_vessel_inlet_or_outlet.append("outlet") 
                                        # self.connection_vessel_indices.append(fv1d_vessel_index) #XXX Bea: I think this was copied by mistake from above
                                        if vessel_tup.BC_type[1] == 'v':
                                            self.connection_vessel_flow_or_pressure_bc.append("flow") 
                                            self.port_input_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                            ).variable(exit_port["variables"][0]).id())
                                            self.port_output_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                            ).variable(exit_port["variables"][1]).id())
                                        elif vessel_tup.BC_type[1] == 'p':
                                            self.connection_vessel_flow_or_pressure_bc.append("pressure") 
                                            self.port_input_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                            ).variable(exit_port["variables"][1]).id())
                                            self.port_output_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                            ).variable(exit_port["variables"][0]).id())
                                        else:
                                            print(f"unknown BC type of {vessel_tup.BC_type} connecting to a "
                                                "1D FV model")
                                
                                elif exit_port["port_type"] == "FV_resistance_port":
                                    FV_resistance_port = 1
                                    self.control_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                        ).variable(exit_port["variables"][0]).id())
                                # TODO other port types will be added here! 
                                else:
                                    # if this port_type isn't one of the above, it wont be connected to the FV1D model
                                    pass
                                    # print(f"unknown port type of {exit_port["port_type"]} connecting to a "
                                    #       "1D FV model")
                                    # exit()

                            if FV_resistance_port == -1:
                                self.control_variables.append( None )

                        else:
                            FV_resistance_port = -1
                            for exit_port in vessel_tup.exit_ports:
                                if exit_port["port_type"] == "vessel_port":
                                    self.connection_vessel_inlet_or_outlet.append("outlet") 
                                    # self.connection_vessel_indices.append(fv1d_vessel_index) #XXX Bea: I think this was copied by mistake from above
                                    if vessel_tup.BC_type[1] == 'v':
                                        self.connection_vessel_flow_or_pressure_bc.append("flow") 
                                        self.port_input_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                        ).variable(exit_port["variables"][0]).id())
                                        self.port_output_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                        ).variable(exit_port["variables"][1]).id())
                                    elif vessel_tup.BC_type[1] == 'p':
                                        self.connection_vessel_flow_or_pressure_bc.append("pressure") 
                                        self.port_input_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                        ).variable(exit_port["variables"][1]).id())
                                        self.port_output_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                        ).variable(exit_port["variables"][0]).id())
                                    else:
                                        print(f"unknown BC type of {vessel_tup.BC_type} connecting to a "
                                            "1D FV model")
                                
                                elif exit_port["port_type"] == "FV_resistance_port":
                                    FV_resistance_port = 1
                                    self.control_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                        ).variable(exit_port["variables"][0]).id())
                                # TODO other port types will be added here! 
                                else:
                                    # if this port_type isn't one of the above, it wont be connected to the FV1D model
                                    pass
                                    # print(f"unknown port type of {exit_port["port_type"]} connecting to a "
                                    #       "1D FV model")
                                    # exit()

                            if FV_resistance_port == -1:
                                self.control_variables.append( None )
                
                for inp_vessel in vessel_tup.inp_vessels:
                    inp_vessel_row = self.model.vessels_df.loc[self.model.vessels_df["name"] == inp_vessel].squeeze()
                    if inp_vessel_row["vessel_type"] == "FV1D_vessel":
                        # # fv1d_vessel_index = re.findall(r'\d+$', inp_vessel)[0]
                        # fv1d_vessel_index = re.findall(r'\d+$', inp_vessel) # "\d+" -> one or more digits, "$" -> at the end of the string
                        if inp_vessel.startswith('FV1D_'):
                            fv1d_vessel_index = re.findall(r'\d+$', inp_vessel)
                        else:
                            fv1d_vessel_index = []
                        
                        if len(fv1d_vessel_index)==0: #XXX then, use different indexing system
                            # print("WARNING: FV1D vessel has NO idx", inp_vessel, fv1d_vessel_index)
                            idx1 = -1
                            for i in range(len(self.conn_1d_0d_info)):
                                conn1d0d = self.conn_1d_0d_info[str(i+1)]
                                if conn1d0d["vess0d_idx"] == idx0:
                                    idx1 = conn1d0d["vess1d_idx"]
                                    break
                            if idx1==-1:
                                print(f"ERROR: idx of 1D FV vessel {inp_vessel} connected to 0D module {vessel_tup.name} not found.")
                                exit()
                            fv1d_vessel_index = str(idx1)
                            fv1d_vessel_index_list.append([inp_vessel, fv1d_vessel_index])
                            print("FV1D vessel idx found", inp_vessel, fv1d_vessel_index)
                            
                        elif len(fv1d_vessel_index)==1:
                            fv1d_vessel_index = fv1d_vessel_index[0]
                            if len(fv1d_vessel_index) != 1:
                                print("ERROR: FV1D vessel idx is not written correctly",
                                    "should be of format FV1D_#, or FV1D_##, etc",
                                    inp_vessel, fv1d_vessel_index)
                                exit()

                            # best to do a sanity check, for compatibility between Finbar's old code and Bea's new code
                            idx1 = -1
                            for i in range(len(self.conn_1d_0d_info)):
                                conn1d0d = self.conn_1d_0d_info[str(i+1)]
                                if conn1d0d["vess0d_idx"] == idx0:
                                    idx1 = conn1d0d["vess1d_idx"]
                                    break
                            if idx1==-1:
                                print(f"ERROR: idx of 1D FV vessel {inp_vessel} connected to 0D module {vessel_tup.name} not found.")
                                exit()
                            if fv1d_vessel_index != str(idx1):
                                print(f"ERROR: different idx of 1D FV vessel {inp_vessel} found: {fv1d_vessel_index} {str(idx1)}")
                                exit()

                            fv1d_vessel_index_list.append([inp_vessel, fv1d_vessel_index])
                        
                        self.connection_vessel_indices.append(fv1d_vessel_index)
                        self.connection_vessel_types.append("FV_1d") # TODO only option for now, can be extended to other kinds of coupling.
                        if vessel_tup.name == 'heart':
                            FV_resistance_port = -1
                            for entrance_port in vessel_tup.entrance_ports:
                                if entrance_port["port_type"] == "vessel_port":
                                    
                                    if ('ivc' in inp_vessel and entrance_port["variables"][0] == 'v_ivc'):
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
                                            print(f"unknown BC type of {vessel_tup.BC_type} connecting to a "
                                                "1D FV model")
                                            exit()

                                    elif ('svc' in inp_vessel and entrance_port["variables"][0] == 'v_svc'):
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
                                            print(f"unknown BC type of {vessel_tup.BC_type} connecting to a "
                                                "1D FV model")
                                            exit()

                                    elif ('pvn' in inp_vessel and entrance_port["variables"][0] == 'v_pvn'):
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
                                            print(f"unknown BC type of {vessel_tup.BC_type} connecting to a "
                                                "1D FV model")
                                            exit()
                                
                                elif entrance_port["port_type"] == "FV_resistance_port":
                                    FV_resistance_port = 1
                                    self.control_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                        ).variable(entrance_port["variables"][0]).id())
                                # TODO other port types will be added here! 
                                else:
                                    # if this port_type isn't one of the above, it wont be connected to the FV1D model
                                    pass
                                    # print(f"unknown port type of {entrance_port["port_type"]} connecting to a "
                                    #       "1D FV model")
                                    # exit()

                            if FV_resistance_port == -1:
                                self.control_variables.append( None )

                        else:
                            FV_resistance_port = -1
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
                                        print(f"unknown BC type of {vessel_tup.BC_type} connecting to a "
                                            "1D FV model")
                                        exit()

                                elif entrance_port["port_type"] == "FV_resistance_port":
                                    FV_resistance_port = 1
                                    self.control_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                        ).variable(entrance_port["variables"][0]).id())
                                # TODO other port types will be added here! 
                                else:
                                    # if this port_type isn't one of the above, it wont be connected to the FV1D model
                                    pass
                                    # print(f"unknown port type of {entrance_port["port_type"]} connecting to a "
                                    #       "1D FV model")
                                    # exit()

                            if FV_resistance_port == -1:
                                self.control_variables.append( None )

        # print(fv1d_vessel_index_list)
        print('1d-0d coupled variables :: DONE')


        for idx0, vessel_tup in enumerate(self.model.vessels_df.itertuples()):
            if vessel_tup.vessel_type=="FV1D_volume_sum": # TODO only option for now, can be extended to other kinds of coupling.
                self.volume_sum_variables.append(self.cellml_model.component(vessel_tup.name \
                                            ).variable('q_sum').id())
                
                idx_sum = -1
                for i in range(len(self.conn_1d_0d_info)):
                    conn1d0d = self.conn_1d_0d_info[str(i+1)]
                    if "port_volume_sum" in conn1d0d:
                        if conn1d0d["port_volume_sum"] == 1 and conn1d0d["vess0d_idx"] == idx0:
                            idx_sum = idx0
                            break
                if idx_sum==-1:
                    print(f"ERROR: idx of 1D FV volume sum connection not found.")
                    exit()
                self.volume_sum_indeces.append(str(idx_sum))

                self.volume_sum_types.append("FV_1d_sum") 

        if len(self.volume_sum_variables)>0:
            self.couple_volume_sum = True
        
        print(f'Volume sum variables :: DONE. couple_volume_sum set to {self.couple_volume_sum}')
        # print(self.volume_sum_variables)
        # print(self.volume_sum_indeces)
        # print(self.volume_sum_types)
        # print('\n')
        
        
        # self.input_variables = [model.component(vessel_tup.name).variable('v_in').id()]
        # self.output_variables = [model.component(vessel_tup.name).variable('u').id()]

        # Create an RDF URI node for our variable to use as the subject for multiple triples
        # note: we are going to serialise the RDF graph into the same folder, so we need a URI that is relative to the intended file
        
        # print(self.generated_wID_model_file_path)
        # print(self.cpp_generated_models_dir)

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
        # control_variable_name_uri = [URIRef(self.generated_wID_model_file_path +
        #                                     '#' + variable) for variable in self.control_variables]
        control_variable_name_uri = []
        for variable in self.control_variables:
            if variable is not None:
                control_variable_name_uri.append(URIRef(self.generated_wID_model_file_path +
                                            '#' + variable))
        
        print("control_variable_name_uri:", control_variable_name_uri)

        external_variable_idx_uri = []                                        
        for idx, type, inlet_or_outlet, flow_or_pressure_bc in zip(self.connection_vessel_indices, self.connection_vessel_types, 
                             self.connection_vessel_inlet_or_outlet, self.connection_vessel_flow_or_pressure_bc):
            external_variable_idx_uri.append(URIRef(self.cpp_generated_models_dir) +
                                        '#' + type + '#' + idx + '#' + inlet_or_outlet + '#' + flow_or_pressure_bc)

        input_variable_name_uri = [URIRef(self.generated_wID_model_file_path + 
                                          '#' + variable) for variable in self.input_variables] # this is empty as long as above lines are commented
        output_variable_name_uri = [URIRef(self.generated_wID_model_file_path + 
                                           '#' + variable) for variable in self.output_variables] # this is empty as long as above lines are commented
        
        # volume_sum_variable_name_uri = [URIRef(self.generated_wID_model_file_path +
        #                                     '#' + variable) for variable in self.volume_sum_variables]
        # volume_sum_variable_name_uri = [URIRef(self.cpp_generated_models_dir +
        #                                     '#' + variable) for variable in self.volume_sum_variables]
        volume_sum_variable_name_uri = []                                        
        for variable, idx, type in zip(self.volume_sum_variables, self.volume_sum_indeces, self.volume_sum_types):
            volume_sum_variable_name_uri.append(URIRef(self.cpp_generated_models_dir) +
                                                '#' + type + '#' + idx + '#' + variable)
    
        
        # Add triples using store's add() method.
        # We're using the Dublin Core term "type" to associate the variable with the delay...
        for II in range(len(variable_to_delay_name_uri)):
            self.g.add((variable_to_delay_name_uri[II], DCTERMS.type, self.variable_to_delay_uri))
            self.g.add((variable_to_delay_name_uri[II], self.delay_variable_uri, delay_variable_name_uri[II]))
            self.g.add((variable_to_delay_name_uri[II], self.independent_variable_uri, independent_variable_name_uri[II]))
            self.g.add((variable_to_delay_name_uri[II], self.delay_amount_uri, delay_amount_name_uri[II]))
        # Set coupling variables
        IIc = 0
        for II in range(len(self.port_input_variables)):
            # for this input variable, add the output variable as a coupling variable
            self.g.add((port_input_variable_name_uri[II], DCTERMS.type, self.input_variable_uri))
            self.g.add((port_input_variable_name_uri[II], self.output_variable_uri, port_output_variable_name_uri[II]))
            self.g.add((port_input_variable_name_uri[II], self.external_variable_uri, external_variable_idx_uri[II]))
            if len(control_variable_name_uri) > 0:
                if self.control_variables[II] is not None:
                    self.g.add((port_input_variable_name_uri[II], self.control_variable_uri, control_variable_name_uri[IIc]))
                    IIc += 1

        for II in range(len(self.output_variables)):
            self.g.add((output_variable_name_uri[II], DCTERMS.type, self.output_variable_uri))
        for II in range(len(self.input_variables)):
            self.g.add((input_variable_name_uri[II], DCTERMS.type, self.input_variable_uri))

        for II in range(len(self.volume_sum_variables)):
            self.g.add((volume_sum_variable_name_uri[II], DCTERMS.type, self.volume_sum_variable_uri))

        
        # print all the data in the turtle format
        print(self.g.serialize(format='ttl'))

        # and save to a file
        with open(self.annotated_model_file_path, 'w') as f:
            f.write(self.g.serialize(format='ttl'))


    def generate_cellml(self, inp_data_dict):
        print("generating CellML files, before Cpp generation")
        cellml_generator = CVS0DCellMLGenerator(self.model, inp_data_dict)
        cellml_generator.generate_files()

    def set_annotated_model_file_path(self, annotated_model_file_path):
        # This is for if we want to use a cellml file that it already annotated, i.e, one we didn't annotate.
        # TODO TEST THIS
        self.annotated_model_file_path = annotated_model_file_path

    def generate_cpp(self):

        print("now Cpp generation can finally starts")
        
        g = Graph().parse(self.annotated_model_file_path)

        # find all delayed variables
        variables_to_delay_info = []
        for vtd in self.g.subjects(DCTERMS.type, self.variable_to_delay_uri):
            # we only expect one delay variable for each variable to delay
            dv = self.g.value(vtd, self.delay_variable_uri)
            d_amount = self.g.value(vtd, self.delay_amount_uri)
            variables_to_delay_info.append([str(vtd), str(dv), str(d_amount)])

        # print('variables_to_delay_info')
        # print(variables_to_delay_info)
        # print('\n')
            
        # # find all timecourse readouts
        # readout_variables = []
        # for d in g.subjects(DCTERMS.type, timecourse_readout_uri):
        #     readout_variables.append(str(d))
            
        # print(readout_variables)

        # find input and output variables
        port_input_variable_info = []
        input_variable_info = []
        output_variable_info = []
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

        volume_sum_variable_info = []
        for d in self.g.subjects(DCTERMS.type, self.volume_sum_variable_uri):
            volume_sum_variable_info.append(str(d))
            
        print('port_input_variable_info')
        print(port_input_variable_info)
        print('input_variable_info')
        print(input_variable_info)
        print('output_variable_info')
        print(output_variable_info)
        print('volume_sum_variable_info')
        print(volume_sum_variable_info)
        print('\n')


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

        volume_sum_variable_ids = []
        for volume_sum_var in volume_sum_variable_info:
            volume_sum_var_url = urlparse(volume_sum_var)
            # if volume_sum_var_url.path != self.generated_wID_model_file_path:
            if volume_sum_var_url.path != self.cpp_generated_models_dir:
                print("found an unexpected model file for readout variable?!")
                exit()
            vv = volume_sum_var_url.fragment
            v = vv.split("#")
            var_type = v[0]
            var_idx = v[1]
            var_id = v[2]
            volume_sum_variable_ids.append([var_id, var_type+"#"+var_idx])
            # volume_sum_variable_ids.append(volume_sum_var_url.fragment)

        print('port_input_variable_ids')
        print(port_input_variable_ids)
        print('volume_sum_variable_ids')
        print(volume_sum_variable_ids)
        print('\n')


        # Now we have the model file and the IDs for the variables in that model that we want to do stuff with. 
        # So we can parse the model and see if we can find the variables.
        # on windows getting a leading '/' in the filename which libCellML doesn't like...
        self.generated_wID_model_file_path = self.generated_wID_model_file_path[0:]

        # parse the model in non-strict mode to allow non CellML 2.0 models
        model = cellml.parse_model(self.generated_wID_model_file_path, False)

        # and make an annotator for this model
        # TODO this could be done before all of the loops when I turn it into one for loop.
        annotator = Annotator()
        annotator.setModel(model)

        #XXX map our IDs to the actual variables
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
            print("DELAY annotated variable:", annotated_variables[-1])
            
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
                    print('Unable to find a readout variable with the id {} in the given model...'.format(control_var_id))
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
            print("PORT INPUT annotated variable:", annotated_variables[-1])

        for input_var_id in input_variable_ids:
            # get the variable (will fail if id doesn't correspond to a variable in the model)
            input_var = annotator.variable(input_var_id)
            if input_var == None:
                print('Unable to find a readout variable with the id {} in the given model...'.format(input_var_id))
                exit()
            annotated_variables.append([[input_var], self.input_variable_uri])
            print("INPUT annotated variable:", annotated_variables[-1])

        for output_var_id in output_variable_ids:
            # get the variable (will fail if id doesn't correspond to a variable in the model)
            output_var = annotator.variable(output_var_id)
            if output_var == None:
                print('Unable to find a readout variable with the id {} in the given model...'.format(output_var_id))
                exit()
            annotated_variables.append([[output_var], self.output_variable_uri])
            print("OUTPUT annotated variable:", annotated_variables[-1])

        for volume_sum_var_id in volume_sum_variable_ids:
            # get the variable (will fail if id doesn't correspond to a variable in the model)
            volume_sum_var = annotator.variable(volume_sum_var_id[0])
            if volume_sum_var == None:
                print('Unable to find a readout variable with the id {} in the given model...'.format(volume_sum_var_id))
                exit()
            annotated_variables.append([[volume_sum_var, volume_sum_var_id[1]], self.volume_sum_variable_uri])
            print("VOLUME SUM annotated variable:", annotated_variables[-1])
        
        print('\n')
        

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
                    print("external variable:", ext_variable)
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
                    print("external variable:", ext_variable)
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
                print("input variable:", external_variable_info[-1], len(vv))
                print("****")
                
            elif uri == self.output_variable_uri:
                # v = vv
                v = vv[0]
                output_variable = flat_model.component(v.parent().name()).variable(v.name())
                external_variable_info.append({
                    'variable': output_variable,
                    'variable_type': 'output'
                })
                # TODO I need to include specification of the external variable here? i.e. the name 
                # or index of the variable from the Cpp code?

            elif uri == self.volume_sum_variable_uri:
                v = vv[0]
                volume_sum_var = flat_model.component(v.parent().name()).variable(v.name())
                
                volume_sum_type = vv[1].split("#")
                v_type = volume_sum_type[0]
                v_idx = volume_sum_type[1]

                aev = AnalyserExternalVariable(volume_sum_var)
                a.addExternalVariable(aev)
                external_variable_info.append({
                    'variable': volume_sum_var,
                    'ext_variable_idx': v_idx,
                    'coupled_to_type': 'FV_1d' if v_type.startswith('FV_1d') else 'unknown',
                    'analyser_variable': aev,
                    'variable_type': 'volume_sum'
                })
                print("volume sum variable:", external_variable_info[-1], len(vv))
                print("****")

        
        a.analyseModel(flat_model)
        analysed_model = a.model()

        if self.DEBUG:
            # parse_model seems to print most of the necessary issues, so we don't need to print them here
            # unless debugging... To check with Hugh
            libcellml_utils.print_issues(a)


        print(f"analysed model has type {analysed_model.type()} . Is it ODE type? {analysed_model.type()==AnalyserModel.Type.ODE}")
        # print(f"analysed model has type {analysed_model.type()} . Is it DAE type? {analysed_model.type()==AnalyserModel.Type.DAE}")
        if analysed_model.type() != AnalyserModel.Type.ODE:
            print("model is not a valid ODE model, aborting...")
            exit()
        # if not analysed_model.isValid():
        #     print("model is not valid, aborting...")
        #     exit()

        # get the information for the variables to delay
        for ext_variable in external_variable_info:
            # print("external variable:", ext_variable)
            ev = ext_variable['variable']
            ev_type = ext_variable['variable_type']
            bc_inlet_or_outlet = ext_variable['bc_inlet0_or_outlet1']

            idxev = -1  
            epv = -1
            idx1d0d = -1
            if 'coupled_to_type' in ext_variable:
                if ext_variable['coupled_to_type']=='FV_1d':
                    idxev = ext_variable['ext_variable_idx'] 
                    if ev_type == 'volume_sum':
                        for j in range(len(self.conn_1d_0d_info)):
                            if "port_volume_sum" in self.conn_1d_0d_info[str(j+1)]:
                                if (self.conn_1d_0d_info[str(j+1)]["port_volume_sum"]==1 
                                    and str(self.conn_1d_0d_info[str(j+1)]["vess0d_idx"])==idxev):
                                    idx1d0d = j+1
                                    break
                        if idx1d0d<0:
                            print(f"ERROR: 1d-0d connection not found in self.conn_1d_0d_info for volume_sum variable coupled to FV_1d type.")
                            exit()
                    else:
                        epv = ext_variable['port_variable']
                        for j in range(len(self.conn_1d_0d_info)):
                            if (str(self.conn_1d_0d_info[str(j+1)]["vess1d_idx"])==idxev
                                and self.conn_1d_0d_info[str(j+1)]["cellml_bc_in0_or_out1"]==bc_inlet_or_outlet):
                                idx1d0d = j+1
                                break
                        if idx1d0d<0:
                            print(f"ERROR: 1d-0d vessel connection not found in self.conn_1d_0d_info for ext_variable_idx {idxev} coupled to FV_1d type.")
                            exit()

            # print("idx0d1d:", idx1d0d)
            
            avs = analysed_model.variables()
            for av in avs:
                v = av.variable()
                
                # if analysed_model.areEquivalentVariables(v, ext_variable['variable']):
                if analysed_model.areEquivalentVariables(v, ev):
                    ext_variable['variable_index'] = av.index()
                    ext_variable['state_or_variable'] = 'variable'
                    if idx1d0d>0:
                        self.conn_1d_0d_info[str(idx1d0d)]["cellml_idx"] = av.index() # in the meantime, filling also self.conn_1d_0d_info
                    print(v.name(), ev.name(), " || variable index", ext_variable['variable_index'])

                if 'port_variable' in ext_variable.keys():
                    # if analysed_model.areEquivalentVariables(v, ext_variable['port_variable']):
                    if analysed_model.areEquivalentVariables(v, epv):
                        ext_variable['port_variable_index'] = av.index()
                        if idx1d0d>0:
                            self.conn_1d_0d_info[str(idx1d0d)]["port_idx"] = av.index() # in the meantime, filling also self.conn_1d_0d_info
                            self.conn_1d_0d_info[str(idx1d0d)]["port_state0_or_var1"] = 1 # variable
                        print(v.name(), epv.name(), " || port variable index", ext_variable['port_variable_index'])

                if 'control_variable' in ext_variable.keys() and ext_variable['control_variable'] != None:
                    ecv = ext_variable['control_variable']
                    # if analysed_model.areEquivalentVariables(v, ext_variable['control_variable']):
                    if analysed_model.areEquivalentVariables(v, ecv):
                        ext_variable['control_variable_index'] = av.index()
                        if idx1d0d>0:
                            self.conn_1d_0d_info[str(idx1d0d)]["R_T_variable_idx"] = av.index() # in the meantime, filling also self.conn_1d_0d_info
                        print(v.name(), ecv.name(), " || control variable index", ext_variable['control_variable_index'])

                if ext_variable['variable_type'] == 'delay':
                    if analysed_model.areEquivalentVariables(v, ext_variable['delay_variable']):
                        ext_variable['delay_variable_index'] = av.index()
                    if analysed_model.areEquivalentVariables(v, ext_variable['delay_amount_variable']):
                        ext_variable['delay_amount_index'] = av.index()
            
            astates = analysed_model.states()
            for astate in astates:
                state = astate.variable()
                
                # if state.name() == ext_variable['variable'].name(): 
                # if state.name() == ev.name(): #XXX Bea: I think this was wrong, replaced by following line
                if analysed_model.areEquivalentVariables(state, ev): 
                    ext_variable['state_index'] = astate.index()
                    ext_variable['state_or_variable'] = 'state'
                    if idx1d0d>0:
                        self.conn_1d_0d_info[str(idx1d0d)]["cellml_idx"] = astate.index() # in the meantime, filling also self.conn_1d_0d_info
                    print(state.name(), ev.name(), " || state index", ext_variable['state_index'])
                
                if 'port_variable' in ext_variable.keys():
                    # if state.name() == ext_variable['port_variable'].name(): 
                    # if state.name() == epv.name(): #XXX Bea: I think this was wrong, replaced by following line
                    if analysed_model.areEquivalentVariables(state, epv): 
                        ext_variable['port_state_index'] = astate.index()
                        if idx1d0d>0:
                            self.conn_1d_0d_info[str(idx1d0d)]["port_idx"] = astate.index() # in the meantime, filling also self.conn_1d_0d_info
                            self.conn_1d_0d_info[str(idx1d0d)]["port_state0_or_var1"] = 0 # state
                        print(state.name(), epv.name(), " || port state index", ext_variable['port_state_index'])
            print("****")


        #XXX saving final 1d-0d connectvity information in generated model directory
        if self.conn_1d_0d_info is not None and self.couple_to_1d:
            if self.file_prefix.endswith("_0d"):
                json_filename = self.file_prefix[:-3]+"_coupler1d0d.json"
            else:
                json_filename = self.file_prefix+"_coupler1d0d.json"
            
            with open(self.generated_model_subdir+"/"+json_filename, "w") as f:
                json.dump(self.conn_1d_0d_info, f, indent=4)
            with open(self.cpp_generated_models_dir+"/"+json_filename, "w") as f:
                json.dump(self.conn_1d_0d_info, f, indent=4)

        
        # generate code from the analysed model
        gen = Generator()
        # using the C profile to generate C code
        # TODO BIG CHANGE
        # TODO I should create my own profile with most of the below code so I don't have to change it when changes are made to libcellml
        profile = GeneratorProfile(GeneratorProfile.Profile.C)
        profile.setInterfaceFileNameString(f'{self.output_cpp_file_name}.h')
        gen.setProfile(profile)
        gen.setModel(analysed_model)


        #XXX HEADER FILE

        preHeaderStuff = f"""
#include <stdlib.h>
#include <memory>
#include <map>
#include <string>
#include <sstream>
#include <functional>
#include <cstring>  
#include <fstream>
#include <vector>
"""
        for external_header in self.external_headers:
            preHeaderStuff += f'#include "{external_header}"\n'

        if self.solver == 'CVODE':
            preHeaderStuff += """
#include <cvodes/cvodes.h>
#include <nvector/nvector_serial.h>
#include <sunlinsol/sunlinsol_dense.h> 
#include <sundials/sundials_types.h>

        """
        elif self.solver == 'PETSC':
            preHeaderStuff += """
#include <petscsys.h>
#include <petscts.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsnes.h>
#include <petscvec.h>

        """

        interFaceCodePreClass = ''
        interFaceCodeInClass = ''
        pre_class = True
        for line in gen.interfaceCode().split('\n'):
            # print(line)
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

            if ('initialiseVariables(' in line or 'computeComputedConstants(' in line or 'computeRates(' in line or 'computeVariables(' in line):
                # print(line)
                if ('computeRates(' in line or 'computeVariables(' in line):
                    if 'double voi' in line:
                        line = line.replace('double voi', 'double voi, double dt')
                if 'initialiseVariables(' in line:
                    if 'double voi' not in line:
                        line = line.replace('(', '(double voi, ')
                if 'double voi' in line:
                    line = line.replace('double voi', 'double voiLoc')
                if 'dt' in line:
                    line = line.replace('dt', 'dtLoc')
                if 'states' in line:
                    line = line.replace('states', 'statesLoc')
                if 'rates' in line:
                    line = line.replace('rates', 'ratesLoc')
                if 'variables' in line:
                    line = line.replace('variables', 'varLoc')
                # print(line)

            if pre_class:
                interFaceCodePreClass += line + '\n'
            else:
                interFaceCodeInClass += '    ' + line + '\n'
            # print(line)


        classInitHeader = ""

        if self.solver == 'CVODE':
            classInitHeader += """
// forward declaration for userOdeData class
class UserOdeData;
        """
        elif self.solver == 'PETSC':
            classInitHeader += """
// forward declarations for PETSc objects
struct _p_TS;
typedef struct _p_TS* TS; // abstract PETSc object that manages integrating an ODE.
struct _p_Vec;
typedef struct _p_Vec* Vec; // abstract PETSc vector object.
struct _p_Mat;
typedef struct _p_Mat* Mat; // abstract PETSc matrix object used to manage all linear operators in PETSc,
                            // even those without an explicit sparse representation (such as matrix-free operators).
struct _p_SNES;
typedef struct _p_SNES* SNES; // abstract PETSc object that manages the nonlinear solves in PETSc
struct _p_KSP;
typedef struct _p_KSP* KSP; // abstract PETSc object that manages the linear solves in PETSc 
                            // (even those such as direct factorization-based solvers that do not use Krylov accelerators).
struct _p_PC;
typedef struct _p_PC* PC; // abstract PETSc object that manages all preconditioners including direct solvers such as PCLU.

// forward declaration for userOdeData class
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
    void computeNonExternalVariables(double voiLoc, double *statesLoc, double *ratesLoc, double *varLoc);
    void getDelayVariables(double voiLoc, double dtLoc, double *statesLoc, double *varLoc);
    void getExternalVariables(double voiLoc, double dtLoc, double *statesLoc, double *varLoc);
    void getExternalVariablesVolume(double *varLoc);

    void initialiseVariablesAndComputeConstants();

    const char* initStatesPath;
    const char* initVarsPath;
    void loadFromFile(const std::string& filename, double* targetArray, const Model0d::VariableInfo infoArray[], size_t infoSize, bool useFirstRow = false);
    void initialise(double voiLoc, double *statesLoc, double *ratesLoc, double *varLoc, const char* statesPath, const char* varsPath, bool useFirstRow = false);

    std::string solver_0d;
    void set_ode_solver(std::string ODEsolver);

    std::ofstream outFileStates;
    std::ofstream outFileVars;
    std::vector<int> idx_states_to_output;
    std::vector<int> idx_vars_to_output;
    void openOutputFiles(std::string outDir);
    void writeOutput(double voiLoc);
    void closeOutputFiles();
        """

    #     if self.solver == 'CVODE':
    #         otherHeaderInits += """
    # void solveOneStepCVODE(double dtLoc);
    #     """
    #     else:
    #         otherHeaderInits += """
    # void solveOneStepExpl(double dtLoc);
    #     """
        otherHeaderInits += """
    void solveOneStep(double dtLoc);
        """

        otherHeaderInits += """
    double voi;
    double voiEnd;
    double dt;
    double eps;
    double tolTime;
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
    int N1d0d;
    int N1d0dTot;
    std::map<std::string, std::map<std::string, int>> pipe_1d_0d_info;
            """

        if self.solver == 'CVODE':
            otherHeaderInits += """
    double * states0; 
    SUNContext context;
    void * solver;
    N_Vector y;
    SUNMatrix matrix;
    SUNLinearSolver linearSolver;

    // UserOdeData* userData= nullptr;
    UserOdeData* userData;
        """
        elif self.solver == 'PETSC':
            otherHeaderInits += """
    // PETSc objects
    TS ts;
    Vec y;
    Vec rates_vec; // optional
    Mat J;
    SNES snes;
    KSP ksp;
    PC pc;

    UserOdeData* userData;
        """
        else:
            otherHeaderInits += """
    double * states0; 
    double wRK4[4];
    double * k1;
    double * k2;
    double * k3;
    double * k4;
            """


        otherHeaderInits += """
    using FunctionType = std::function<void(double, double, double*, double*, double*)>;
        """

        # using computeRatesType = void (*)(double, double *, double *, double *);
        # static int func(double voi, N_Vector y, N_Vector ydot, void *userData);

        if self.couple_to_1d:
            otherHeaderInits += """
    std::ofstream write_pipe_dt;
    std::ifstream read_pipe_dt;
    std::vector<std::ofstream> write_pipe;
    std::vector<std::ifstream> read_pipe;
    int openPipes(std::string pipePath);
    void closePipes();
    
    int DATA_LENGTH;
    double ** zero_data;
    double ** parent_data;
    double * zero_data_dt;
    double * parent_data_dt;
            """

        if self.couple_volume_sum:
            otherHeaderInits += """
    std::ifstream read_pipe_vol;
    double * parent_data_vol;
            """

        classFinisherHeader = """
};
        """


        #XXX SOURCE FILE

        preSourceStuff = f"""
#include <stddef.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <cstring>  
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <algorithm>
#include <vector>
#include <cmath>
#include <iomanip>
#include <thread>
#include <poll.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
        """

        if self.solver == 'CVODE':
            preSourceStuff += """
#include <cvodes/cvodes.h>
#include <nvector/nvector_serial.h>
#include <sunlinsol/sunlinsol_dense.h> 
#include <sundials/sundials_types.h>
        """
        elif self.solver == 'PETSC':
            preSourceStuff += """
#include <petscsys.h>
#include <petscts.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsnes.h>
#include <petscvec.h>

#define SOLVER_CN    // Crank-Nicolson
// #define SOLVER_BDF1     // BDF o1 (Backward Euler)
// #define SOLVER_BDF2  // BDF o2

        """

        with open(generators_dir_path+'/cppGeneratorTemplateFunctions.cpp', 'r') as file:
            lines_tmp = file.readlines()
        
        iStart = None
        iEnd = None
        for iL, line in enumerate(lines_tmp):
            if "Functions for index-name mapping" in line:
                iStart = iL
                break
        for iL, line in enumerate(lines_tmp[iStart+1:], start=iStart+1):
            if "End functions" in line:
                iEnd = iL
                break
        for iL, line in enumerate(lines_tmp[iStart:iEnd+1], start=iStart):
            preSourceStuff += line #+ '\n'
        preSourceStuff += '\n'

        # split implementation code in two so we can change it into a class
        preClassStuff = ''
        
        classInit = """
Model0d::Model0d() :
        """
        postClassInit = ''

        classInit += """
    voi(0.0),
    voiEnd(0.0),
    dt(0.0),
    eps(1e-08),
    tolTime(1e-11), // 1e-13
    states(nullptr),
    rates(nullptr),
    variables(nullptr),
    time_dof_0d(-1),
    solver_0d(""),

    initStatesPath(nullptr),
    initVarsPath(nullptr),
        """

        if self.solver == 'CVODE':
            classInit += """    
    states0(nullptr),
    userData(nullptr),
        """
        elif self.solver == 'PETSC':
            classInit += """    
    ts(nullptr),
    y(nullptr),
    rates_vec(nullptr),
    J(nullptr),
    snes(nullptr),
    ksp(nullptr),
    pc(nullptr),
    userData(nullptr),
        """
        else: # elif self.solver == 'RK4':
            classInit += """    
    states0(nullptr),
    k1(nullptr),
    k2(nullptr),
    k3(nullptr),
    k4(nullptr),
        """
            
        if self.couple_to_1d:
            # classInit += """    model1d_ptr(nullptr),
            N1d0dTot = len(self.conn_1d_0d_info)

            N1d0d = 0 
            for i in range (N1d0dTot):
                if "port_volume_sum" in self.conn_1d_0d_info[str(i+1)]:
                    if self.conn_1d_0d_info[str(i+1)]["port_volume_sum"]==1:
                        pass
                    else:
                        N1d0d +=1
                else:
                    N1d0d +=1
            
            classInit += f"""
    N1d0d({N1d0d}),
    N1d0dTot({N1d0dTot}),
        """ 
    #     else:
    #         N1d0d = 0
    #         N1d0dTot = 0
    #         classInit += f"""
    # N1d0d({N1d0d}),
    # N1d0dTot({N1d0dTot}),
    #     """
           
        # create mapping between external variable index and 1D vessel and BC information
        if self.couple_to_1d:
            classInit += """
    pipe_1d_0d_info{
    """
            for iC in range (N1d0dTot):
                conn1d0d = self.conn_1d_0d_info[str(iC+1)]
                port_volume_sum = -1
                if "port_volume_sum" in conn1d0d:
                    port_volume_sum = conn1d0d["port_volume_sum"]
                classInit += f"""       {{ "{str(iC+1)}", {{ 
                    {{ "vess1d_idx", {conn1d0d["vess1d_idx"]} }}, 
                    {{ "vess1d_bc_in0_or_out1", {conn1d0d["vess1d_bc_in0_or_out1"]} }}, 
                    {{ "cellml_idx", {conn1d0d["cellml_idx"]} }}, 
                    {{ "cellml_bc_in0_or_out1", {conn1d0d["cellml_bc_in0_or_out1"]} }}, 
                    {{ "cellml_bc_flow0_or_press1", {conn1d0d["cellml_bc_flow0_or_press1"]} }}, 
                    {{ "port_idx", {conn1d0d["port_idx"]} }}, 
                    {{ "port_flow0_or_press1", {conn1d0d["port_flow0_or_press1"]} }},
                    {{ "port_state0_or_var1", {conn1d0d["port_state0_or_var1"]} }},
                    {{ "R_T_variable_idx", {conn1d0d["R_T_variable_idx"]} }},
                    {{ "port_volume_sum", {port_volume_sum} }}
                }} }}"""
                if iC<N1d0dTot-1:
                    classInit += ",\n"
                else:
                    classInit += "\n"
            classInit += "    },\n"

            # classInit += """    cellml_index_to_vessel1d_info{
            # """
            # num_vessel1d_connections = 0 
            # for idx, ext_variable in enumerate(external_variable_info):
            #     if idx != 0:
            #         classInit += ',\n'
            #     if ext_variable['coupled_to_type'] == 'FV_1d':
            #         classInit += f"""       {{ "{ext_variable["variable_index"]}", {{ 
            #             {{ "vessel1d_idx", {ext_variable["ext_variable_idx"]}}}, 
            #             {{ "bc_inlet0_or_outlet1", {ext_variable["bc_inlet0_or_outlet1"]} }}"""
            #         if 'port_variable_index' in ext_variable.keys():
            #             classInit += f""",
            #             {{ "port_variable_idx", {ext_variable["port_variable_index"]}}}"""
            #         elif 'port_state_index' in ext_variable.keys():
            #             classInit += f""",
            #             {{ "port_state_idx", {ext_variable["port_state_index"]}, }}"""
            #         if 'control_variable_index' in ext_variable.keys() and ext_variable['control_variable_index'] != None:
            #             classInit += f""",
            #             {{ "control_variable_idx", {ext_variable["control_variable_index"]}, }}"""
            #         classInit += """}}"""
            #         num_vessel1d_connections += 1
            # classInit += '},\n'
            # classInit += f'    num_vessel1d_connections({num_vessel1d_connections}),\n'

            # # now create a vessel1d_info vector of dicts for each connected variable
            # classInit += """    vessel1d_info{ """
            # for idx, ext_variable in enumerate(external_variable_info):
            #     if idx != 0:
            #         classInit += ',\n'
            #     if ext_variable['coupled_to_type'] == 'FV_1d':
            #         classInit += f"""       
            #         {{  {{ "cellml_idx", {ext_variable["variable_index"]}}}, 
            #             {{ "vessel1d_idx", {ext_variable["ext_variable_idx"]}}}, 
            #             {{ "bc_inlet0_or_outlet1", {ext_variable["bc_inlet0_or_outlet1"]} }} """
            #         if 'port_variable_index' in ext_variable.keys():
            #             classInit += f""",
            #             {{ "port_variable_idx", {ext_variable["port_variable_index"]}, }}
            #             """
            #         elif 'port_state_index' in ext_variable.keys():
            #             classInit += f""",
            #             {{ "port_state_idx", {ext_variable["port_state_index"]}, }}
            #             """
            #         if 'control_variable_index' in ext_variable.keys() and ext_variable['control_variable_index'] != None:
            #             classInit += f""",
            #             {{ "control_variable_idx", {ext_variable["control_variable_index"]}, }}
            #             """
            #         classInit += "}"
            # classInit += '    },\n'

        
        pre_class = True
        in_class_init = False

        # TODO The below is all really susceptible to changes in libcellml, I should create my own profile
        lines = gen.implementationCode().split('\n')
        max_func_found = any(l.startswith('double max') for l in lines)
        
        lines_to_pass = []
        lines_to_pass_rates = []
        lines_to_pass_variables = []
        init_lines_to_mod = []
        for iL, line in enumerate(lines):
            # print(iL, line)
            if line.startswith('const size_t STATE_COUNT'):
                pre_class = False
                in_class_init = True

            if line.startswith('const VariableInfo VOI_INFO'):
                pre_class = True
                in_class_init = False

            if 'VERSION' in line:
                line = line.replace('VERSION', 'VERSION_')
            
            # max_func_found = False
            # for line2 in gen.implementationCode().split('\n'):
            #     if line2.startswith('double max'):
            #         max_func_found = True

            if max_func_found:
                if line.startswith('double max'):
                    pre_class = False
                    in_class_init = False
            else:
                if line.startswith('double * createStatesArray'):
                    pre_class = False
                    in_class_init = False
            # if line.startswith('double * createStatesArray'):
            # # if line.startswith('double max'):
            #     in_class_init = False

            if pre_class:
                if line.startswith('const VariableInfo VOI_INFO'):
                    line = line.replace('VariableInfo VOI_INFO', 'Model0d::VariableInfo Model0d::VOI_INFO')
                if line.startswith('const VariableInfo STATE_INFO'):
                    line = line.replace('VariableInfo STATE_INFO', 'Model0d::VariableInfo Model0d::STATE_INFO')
                if line.startswith('const VariableInfo VARIABLE_INFO'):
                    line = line.replace('VariableInfo VARIABLE_INFO', 'Model0d::VariableInfo Model0d::VARIABLE_INFO')
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
                # print(iL, line)
                if 'res[i] = NAN' in line:
                    line = line.replace('NAN', '0.0')

                if 'createStatesArray' in line:
                    line = line.replace('createStatesArray', 'Model0d::createStatesArray')
                if 'createVariablesArray' in line:
                    line = line.replace('createVariablesArray', 'Model0d::createVariablesArray')
                if 'deleteArray' in line:
                    line = line.replace('deleteArray', 'Model0d::deleteArray')
                
                if 'initialiseVariables' in line: 
                    line = line.replace('initialiseVariables', 'Model0d::initialiseVariables')

                    if 'initialiseVariables(' in line:
                        if 'double voi' not in line:
                            line = line.replace('(', '(double voi, ')

                    iStart = None
                    iEnd = None
                    for iL2, line2 in enumerate(lines[iL+1:], start=iL+1):
                        if line2.startswith("{"):
                            iStart = iL2
                        if line2.startswith("}"):
                            iEnd = iL2
                        if iStart is not None and iEnd is not None:
                            break
                    if iStart is None or iEnd is None:
                        print(f"Start or end line index of initialiseVariables function not found: {iStart} {iEnd}.")
                        exit()

                    for iL2, line2 in enumerate(lines[iStart:iEnd], start=iStart):
                        if 'external' in line2:
                            init_lines_to_mod.append(iL2)
                
                if 'computeComputedConstants' in line:
                    line = line.replace('computeComputedConstants', 'Model0d::computeComputedConstants')
                
                # if 'computeRates' in line: 
                if 'computeRates(double ' in line:
                    line = line.replace('computeRates', 'Model0d::computeRates')
                    line = line.replace('double voi,', 'double voi, double dt,') # Bea's addition

                    iStart = None
                    iEnd = None
                    for iL2, line2 in enumerate(lines[iL+1:], start=iL+1):
                        if line2.startswith("{"):
                            iStart = iL2
                        if line2.startswith("}"):
                            iEnd = iL2
                        if iStart is not None and iEnd is not None:
                            break
                    if iStart is None or iEnd is None:
                        print(f"Start or end line index of computeRates function not found: {iStart} {iEnd}.")
                        exit()

                    line = line + "\n"      # start from the original line
                    line += "{\n"           # add the opening brace
                    line += "    computeVariables(voi, dt, states, rates, variables);"  # add indented statement

                    lines_to_pass_rates.append(iStart)
                    pattern = re.compile(r"variables\[\d+\]\s*=\s*")
                    for iL2, line2 in enumerate(lines[iStart:iEnd], start=iStart):
                        # if 'variables' in line2:
                        if pattern.search(line2):
                            # print("lines to pass computeRates:", iL2, line2)
                            lines_to_pass_rates.append(iL2)

                # if 'computeVariables' in line:
                if 'computeVariables(double ' in line:
                    line = line.replace('computeVariables', 'Model0d::computeVariables')
                    line = line.replace('double voi,', 'double voi, double dt,') # Bea's addition

                    iStart = None
                    iEnd = None
                    for iL2, line2 in enumerate(lines[iL+1:], start=iL+1):
                        if line2.startswith("{"):
                            iStart = iL2
                        if line2.startswith("}"):
                            iEnd = iL2
                        if iStart is not None and iEnd is not None:
                            break
                    if iStart is None or iEnd is None:
                        print(f"Start or end line index of computeVariables function not found: {iStart} {iEnd}.")
                        exit()

                    line = line + "\n"      # start from the original line
                    line += "{\n"           # add the opening brace
                    line += "    computeNonExternalVariables(voi, states, rates, variables);\n"
                    # line += "    if (N1d0d>0){\n"  
                    line += "    getDelayVariables(voi, dt, states, variables);\n"
                    line += "    getExternalVariables(voi, dt, states, variables);\n"
                    # line += "    }\n"  
                    line += "}\n"  

                    lines_to_pass_variables.append(iStart)
                    pattern = re.compile(r"variables\[\d+\]\s*=\s*")
                    for iL2, line2 in enumerate(lines[iStart:iEnd], start=iStart):
                        # if 'variables' in line2:
                        if pattern.search(line2):
                            # print("lines to pass computeVariables:", iL2, line2)
                            lines_to_pass_variables.append(iL2)
                    lines_to_pass_variables.append(iEnd)
                    
                    line += "\n"
                    line += "void Model0d::computeNonExternalVariables(double voi, double *states, double *rates, double *variables)\n"
                    line += "{\n"
                    for iL2, line2 in enumerate(lines[iStart+1:iEnd], start=iStart+1):
                        if pattern.search(line2) and 'external' not in line2:
                            line += line2+"\n"
                    if len(lines_to_pass_rates)>0:
                        for iL2 in lines_to_pass_rates[1:]:
                            line2 = lines[iL2]
                            if line2 not in lines[iStart+1:iEnd]:
                                line += line2+"\n"
                                # print("line to add from computeRates:", iL2, line2)
                    line += "}" 

                if iL in init_lines_to_mod:
                    idx0 = line.find("external")
                    idx1 = line.find(";")
                    line = line.replace(line[idx0:idx1+1], '0.0; // external coupling variable computed at initial time step iteration')

                if 'ExternalVariable externalVariable' in line:
                    line = line.replace(', ExternalVariable externalVariable', '')

                if 'voi' in line:
                    line = line.replace('void', 'XXX')
                    # line = line.replace('voi', 'voi_') # I dont like this format:))
                    line = line.replace('voi', 'voiLoc')
                    line = line.replace('XXX', 'void')
                if 'dt' in line:
                    line = line.replace('dt', 'dtLoc')
                if 'states' in line:
                    # line = line.replace('states', 'states_') # I dont like this format:))
                    line = line.replace('states', 'statesLoc')
                if 'rates' in line:
                    # line = line.replace('rates', 'rates_') # I dont like this format:))
                    line = line.replace('rates', 'ratesLoc')
                if 'variables' in line:
                    # line = line.replace('variables', 'variables_') # I dont like this format:))
                    line = line.replace('variables', 'varLoc')

                # postClassInit += line + '\n'
                if (iL not in lines_to_pass and iL not in lines_to_pass_rates and iL not in lines_to_pass_variables):
                    postClassInit += line + '\n'
                else:
                    pass

        classInit += f"""
            states = createStatesArray();
            rates = createStatesArray(); // same size as states, should really be different function
            variables = createVariablesArray();
        """

        classInit += """
        }

        Model0d::~Model0d() {
            // Clean up after ourselves.
            deleteArray(states);
            deleteArray(rates);
            deleteArray(variables);
        """

        if self.solver == 'CVODE':
            classInit += """
            delete userData;
            SUNLinSolFree(linearSolver);
            SUNMatDestroy(matrix);
            N_VDestroy_Serial(y);
            CVodeFree(&solver);
            SUNContext_Free(&context);   
        """
        elif self.solver == 'PETSC':
            classInit += """
            delete userData;
            if (y) { VecDestroy(&y); y = nullptr; }
            if (rates_vec) { VecDestroy(&rates_vec); rates_vec = nullptr; }
            if (J) { MatDestroy(&J); J = nullptr; }
            if (ts) { TSDestroy(&ts); ts = nullptr; }
            // if (snes) SNESDestroy(&snes); // this is destroyed by TSDestroy
            // if (ksp) KSPDestroy(&ksp); // this is destroyed by TSDestroy
            // if (pc) PCDestroy(&pc); // this is destroyed by TSDestroy   
        """
        else:
            classInit += """
            deleteArray(states0);
            deleteArray(k1);
            deleteArray(k2);
            deleteArray(k3);
            deleteArray(k4);
        """

        if self.couple_to_1d and N1d0dTot>0:  
            classInit += """
            for (int i = 0; i < N1d0d; ++i){
                delete[] zero_data[i];
                delete[] parent_data[i];
            }
            delete[] zero_data;
            delete[] parent_data;
            delete[] zero_data_dt;
            delete[] parent_data_dt;
        """
        if self.couple_volume_sum:
            classInit += """
            delete[] parent_data_vol;
        """

        classInit += """
        }
        """
        #XXX TODO what else should I do in the destructor? # Bea: I think destructor is fine now:)

        # print("#################################################")
        # print("preClassStuff")
        # print(preClassStuff)
        # print("#################################################")
        # print("classInit")
        # print(classInit)
        # print("#################################################")
        # print("postClassInit")
        # print(postClassInit) 


        solverInitFunction = """ 
void Model0d::set_ode_solver(std::string ODEsolver)
{
    solver_0d = ODEsolver;
    """

        if self.solver == 'CVODE':
            solverInitFunction += f"""
    // Create our SUNDIALS context object.
    SUNContext_Create(SUN_COMM_NULL, &context);
    // Create our CVODE solver.
    solver = CVodeCreate(CV_BDF, context);
    // Initialise our CVODE solver.
    y = N_VMake_Serial(STATE_COUNT, states, context);
    CVodeInit(solver, func, voi, y);
    // Set our user data.
    userData = new UserOdeData(variables, std::bind(&Model0d::computeRates, this, std::placeholders::_1, 
                                                    std::placeholders::_2, std::placeholders::_3, 
                                                    std::placeholders::_4, std::placeholders::_5), solver);
    CVodeSetUserData(solver, userData);
    // Set our maximum number of internal steps (default 500).
    long int mxsteps = {self.nMaxSteps};
    CVodeSetMaxNumSteps(solver, mxsteps);
    // Set our maximum absolute step size.
    sunrealtype hmax = {self.dtSolver};
    CVodeSetMaxStep(solver, hmax);
    // Set our linear solver.
    // Create matrix object.
    matrix = SUNDenseMatrix(STATE_COUNT, STATE_COUNT, context);
    // Create linear solver object.
    linearSolver = SUNLinSol_Dense(y, matrix, context);
    // Attach linear solver module
    CVodeSetLinearSolver(solver, linearSolver, matrix);
    // Set our scalar relative and absolute tolerances.
    sunrealtype reltol = 1e-7; // TODO get this from user_inputs.yaml too
    sunrealtype abstol = 1e-9; // TODO get this from user_inputs.yaml too
    CVodeSStolerances(solver, reltol, abstol); 
"""
        elif self.solver == 'PETSC':
            solverInitFunction += f"""
    // Set up PETSc TS ODE solver (fixed-step for now)
    PetscErrorCode ierr;

    // Create solution vector
    ierr = VecCreate(PETSC_COMM_WORLD, &y);
    std::cout << "set_ode_solver :: VecCreate "<< ierr << std::endl;
    ierr = VecSetSizes(y, PETSC_DECIDE, STATE_COUNT);
    std::cout << "set_ode_solver :: VecSetSizes "<< ierr << std::endl;
    ierr = VecSetFromOptions(y);
    std::cout << "set_ode_solver :: VecSetFromOptions "<< ierr << std::endl;

    // Create TS
    ierr = TSCreate(PETSC_COMM_WORLD, &ts); // create the timestepper (TS) object
    std::cout << "set_ode_solver :: TSCreate "<< ierr << std::endl;
    ierr = TSSetProblemType(ts, TS_NONLINEAR); // TSProblemType is one of TS_LINEAR or TS_NONLINEAR
    std::cout << "set_ode_solver :: TSSetProblemType "<< ierr << std::endl;
    ierr = TSSetRHSFunction(ts, NULL, TSRHSfunc, this); // set the RHS function
    // 'this' --> pointer to your Model0d object; PETSc stores it, so that every time it calls TSRHSfunc, it passes that same pointer as ctx.
    std::cout << "set_ode_solver :: TSSetRHSFunction "<< ierr << std::endl;
    ierr = TSSetSolution(ts, y);
    std::cout << "set_ode_solver :: TSSetSolution "<< ierr << std::endl; 


#ifdef SOLVER_CN
    ierr = TSSetType(ts, TSCN); // set the solution method: Crank-Nicolson
    std::cout << "set_ode_solver :: TSSetType "<< ierr << std::endl;
    ierr = TSSetTimeStep(ts, {self.dtSolver}); // set the initial time step
    std::cout << "set_ode_solver :: TSSetTimeStep "<< ierr << std::endl;
    ierr = TSSetMaxSteps(ts, 10000000000000); // maximum number of time steps
    std::cout << "set_ode_solver :: TSSetMaxSteps "<< ierr << std::endl;
    // ierr = TSSetMaxTime(ts, 10.);
    // std::cout << "set_ode_solver :: TSSetMaxTime "<< ierr << std::endl;
    // No other options needed
#endif

#ifdef SOLVER_BDF1
    ierr = TSSetType(ts, TSBDF); // set the solution method: backward differentiation formula (BDF)
    std::cout << "set_ode_solver :: TSSetType "<< ierr << std::endl;
    ierr = TSBDFSetOrder(ts, 1);
    std::cout << "set_ode_solver :: TSBDFSetOrder "<< ierr << std::endl;
    ierr = TSSetTimeStep(ts, {self.dtSolver}); // set the initial time step
    std::cout << "set_ode_solver :: TSSetTimeStep "<< ierr << std::endl;
    ierr = TSSetMaxSteps(ts, 10000000000000); // maximum number of time steps
    std::cout << "set_ode_solver :: TSSetMaxSteps "<< ierr << std::endl;
    // ierr = TSSetMaxTime(ts, 10.);
    // std::cout << "set_ode_solver :: TSSetMaxTime "<< ierr << std::endl;

    // Options for matrix-free BDF1
    PetscOptionsSetValue(NULL, "-ts_adapt_type", "none");
    PetscOptionsSetValue(NULL, "-ts_bdf_single_jacobian", NULL);
    PetscOptionsSetValue(NULL, "-snes_mf_operator", NULL);
#endif

#ifdef SOLVER_BDF2
    ierr = TSSetType(ts, TSBDF); // set the solution method: backward differentiation formula (BDF)
    std::cout << "set_ode_solver :: TSSetType "<< ierr << std::endl;
    ierr = TSBDFSetOrder(ts, 2);
    std::cout << "set_ode_solver :: TSBDFSetOrder "<< ierr << std::endl;
    ierr = TSSetTimeStep(ts, {self.dtSolver}); // set the initial time step
    std::cout << "set_ode_solver :: TSSetTimeStep "<< ierr << std::endl;
    ierr = TSSetMaxSteps(ts, 10000000000000); // maximum number of time steps
    std::cout << "set_ode_solver :: TSSetMaxSteps "<< ierr << std::endl;
    // ierr = TSSetMaxTime(ts, 10.);
    // std::cout << "set_ode_solver :: TSSetMaxTime "<< ierr << std::endl;

    // Options for matrix-free BDF2
    PetscOptionsSetValue(NULL, "-ts_adapt_type", "none");
    PetscOptionsSetValue(NULL, "-ts_bdf_single_jacobian", NULL);
    PetscOptionsSetValue(NULL, "-snes_mf_operator", NULL);
#endif 

    // Tolerances
    // in order: scalar absolute tolerances, vector of absolute tolerances, scalar relative tolerances, vector of relative tolerances
    double reltol = 1e-7; // TODO get this from user_inputs.yaml too
    double abstol = 1e-9; // TODO get this from user_inputs.yaml too
    ierr = TSSetTolerances(ts, abstol, NULL, reltol, NULL); 
    std::cout << "set_ode_solver :: TSSetTolerances "<< ierr << std::endl;

    // Nonlinear solver
    ierr = TSGetSNES(ts, &snes); // return the SNES (nonlinear solver) associated with the TS (timestepper) context (valid only for nonlinear problems)
    std::cout << "set_ode_solver :: TSGetSNES "<< ierr << std::endl;
    ierr = SNESSetType(snes, SNESNEWTONLS); // set the algorithm/method to be used to solve the nonlinear system with the given SNES
    std::cout << "set_ode_solver :: SNESSetType "<< ierr << std::endl;

    // Linear solver (dense)
    ierr = SNESGetKSP(snes, &ksp); // return the KSP (linear solver) context for the SNES solver
    std::cout << "set_ode_solver :: SNESGetKSP "<< ierr << std::endl;
    ierr = KSPSetType(ksp, KSPGMRES); // GMRES // set the algorithm/method to be used to solve the linear system with the given KSP
    // ierr = KSPSetType(ksp, KSPFGMRES); // flexible GMRES
    // ierr = KSPSetType(ksp, KSPPREONLY); // KSPPREONLY: method that applies ONLY the preconditioner exactly once.
    std::cout << "set_ode_solver :: KSPSetType "<< ierr << std::endl;

    // Preconditioner
    ierr = KSPGetPC(ksp, &pc); // return a pointer to the preconditioner (PC) context with the KSP
    std::cout << "set_ode_solver :: KSPGetPC "<< ierr << std::endl;

    // Build PC for a particular preconditioner type
#ifdef SOLVER_CN
    // ierr = PCSetType(pc, PCNONE);
    // ierr = PCSetType(pc, PCJACOBI); // PCJACOBI: Jacobi, i.e. diagonal scaling preconditioning.
    // ierr = PCSetType(pc, PCLU); // PCLU: this uses a direct solver, based on LU factorization, as a preconditioner.
    ierr = PCSetType(pc, PCILU);
    std::cout << "set_ode_solver :: PCSetType "<< ierr << std::endl;
#endif 

#ifdef SOLVER_BDF1
    // ierr = PCSetType(pc, PCNONE);
    ierr = PCSetType(pc, PCJACOBI); // PCJACOBI: Jacobi, i.e. diagonal scaling preconditioning.
    // ierr = PCSetType(pc, PCLU); // PCLU: this uses a direct solver, based on LU factorization, as a preconditioner.
    std::cout << "set_ode_solver :: PCSetType "<< ierr << std::endl;
#endif

#ifdef SOLVER_BDF2
    // ierr = PCSetType(pc, PCNONE);
    ierr = PCSetType(pc, PCJACOBI); // PCJACOBI: Jacobi, i.e. diagonal scaling preconditioning.
    // ierr = PCSetType(pc, PCLU); // PCLU: this uses a direct solver, based on LU factorization, as a preconditioner.
    std::cout << "set_ode_solver :: PCSetType "<< ierr << std::endl;
#endif

    ierr = TSSetFromOptions(ts); // set various TS parameters from the options database
    std::cout << "set_ode_solver :: TSSetFromOptions "<< ierr << std::endl;
    ierr = TSSetUp(ts); // set up the internal data structures for the later use of TS
    std::cout << "set_ode_solver :: TSSetUp "<< ierr << std::endl;

    // User data
    userData = new UserOdeData(variables, std::bind(&Model0d::computeRates, this,
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
        std::placeholders::_4, std::placeholders::_5));
    userData->tOld = voi;

"""
        else:
            solverInitFunction += """
    states0 = createStatesArray(); // store initial state, before evolving to the next time level
    k1 = createStatesArray(); // same size as states, should really be different function
    k2 = createStatesArray(); // same size as states, should really be different function
    k3 = createStatesArray(); // same size as states, should really be different function
    k4 = createStatesArray(); // same size as states, should really be different function
    // Runge-Kutta 4 weights
    wRK4[0] = 1./6.;
    wRK4[1] = 1./3.;
    wRK4[2] = 1./3.;
    wRK4[3] = 1./6.;
"""
        solverInitFunction += """
}
"""
        
        # print("#################################################")
        # print("solverInitFunction")
        # print(solverInitFunction) 

        pipesFunctions = ''
        if self.couple_to_1d and N1d0dTot>0:

            iStart1 = None
            iEnd1 = None
            iStart2 = None
            iEnd2 = None
            for iL, line in enumerate(lines_tmp):
                if "Model0d::openPipes(" in line:
                    iStart1 = iL-1
                    break
            for iL, line in enumerate(lines_tmp[iStart1+1:], start=iStart1+1):
                if "// Open read pipe for volume sum" in line:
                    iEnd1 = iL
                    break
            for iL, line in enumerate(lines_tmp[iEnd1+1:], start=iEnd1+1):
                if "Model0d::closePipes(" in line:
                    iStart2 = iL-1
                    break
            for iL, line in enumerate(lines_tmp[iStart2+1:], start=iStart2+1):
                if "End functions" in line:
                    iEnd2 = iL
                    break
            
            for iL, line in enumerate(lines_tmp[iStart1:iStart2], start=iStart1):
                if iL==iEnd1:
                    if not self.couple_volume_sum:
                        pipesFunctions += """
    return 0;
}
"""
                        break
                    else:
                        pipesFunctions += line #+ '\n'
                else:
                    pipesFunctions += line #+ '\n'

            for iL, line in enumerate(lines_tmp[iStart2:iEnd2+1], start=iStart2):
                if ("// Close volume sum pipe" in line or "read_pipe_vol" in line):
                    if self.couple_volume_sum:
                        pipesFunctions += line #+ '\n'
                    else:
                        pass
                else:
                    pipesFunctions += line #+ '\n'
            pipesFunctions += '\n'


        # print("#################################################")
        # print("pipesFunctions")
        # print(pipesFunctions)

        outputFunctions = ''
        iStart = None
        iEnd = None
        for iL, line in enumerate(lines_tmp):
            if "Functions for import/export of model outputs" in line:
                iStart = iL
                break
        for iL, line in enumerate(lines_tmp[iStart+1:], start=iStart+1):
            if "End functions" in line:
                iEnd = iL
                break
        for iL, line in enumerate(lines_tmp[iStart:iEnd+1], start=iStart):
            if "VARIABLE_INFO[i].type == EXTERNAL ||" in line:
                if self.couple_to_1d:
                    outputFunctions += line + '\n'
                else:
                    line = line.replace("VARIABLE_INFO[i].type == EXTERNAL || ", "")
                    outputFunctions += line #+ '\n'
            else:
                outputFunctions += line #+ '\n'
        outputFunctions += '\n'


        # print("#################################################")
        # print("outputFunctions")
        # print(outputFunctions)


        # # find the function for computeVariables to create a 
        # # compute non_external variables function
        # non_external_variables_function = \
        #         'void Model0d::computeNonExternalVariables(double voi_, double *states_, double *rates_, double *variables_) \n'
        # startFunction = False
        # for line in gen.implementationCode().split('\n'):
        #     if line.startswith('void computeVariables'):
        #         startFunction = True
        #     if startFunction:
        #         if line.startswith('}'):
        #             non_external_variables_function += line + '\n'
        #             break
        #         if 'externalVariable' in line:
        #             continue
        #         if 'voi' in line:
        #             line = line.replace('void', 'XXX')
        #             line = line.replace('voi', 'voi_')
        #             line = line.replace('XXX', 'void')
        #         if 'states' in line:
        #             line = line.replace('states', 'states_')
        #         if 'rates' in line:
        #             line = line.replace('rates', 'rates_')
        #         if 'variables' in line:
        #             line = line.replace('variables', 'variables_')
        #         non_external_variables_function += line + '\n'


#         # and generate a function to compute external variables #XXX Finbar's old code
#         computeEV = f""" 
# double Model0d::externalVariable(double voi_, double *states_, double *rates_, double *variables_, size_t index)
# {{
#     // if voi is zero we may need to calculate the non_external_variables
#     // because they can be needed for the external variables
#     if (voi_ == 0.0) {{
#         computeNonExternalVariables(voi_, states_, rates_, variables_);
#     }}
#         """
#         for ext_variable in external_variable_info:
#             print(ext_variable.keys())

#             # variable or state index
#             if ext_variable["state_or_variable"] == "state":
#                 state_index = ext_variable['state_index']
#             else:
#                 variable_index = ext_variable['variable_index']

#             # TODO Finbar: for each external variable, include a label for its corresponding port
#             # variable, ie when we need qBC, we give P_C and possibly R_T1.
#             # TODO Finbar: create a config file which details the types of external variables we 
#             # can have and their corresponding port variables. 
#             # a new one will be created for each type of coupling.

#             # TODO if time_dof_0d = 0 then we don't store the variables in the buffer, only store when it equals -1
#             # set up so that when time_dof_0d is not -1 we don't step forward in the buffer, we only interpolate 
#             # from it, we only step forward in the buffer when time_dof_0d is -1
#             # use a time_0d variable to check where voi is in terms of time_0d
#             # TODO we could run a rough CVODE where we only save at time_output points, then linearly interpolate...
#             #  this would be quite easy to implement.. unsure about stability.
#             if ext_variable['variable_type'] == 'delay':
                    
#                 delay_variable_index = ext_variable['delay_variable_index']
#                 delay_amount_index = ext_variable['delay_amount_index']
#                 computeEV += f'  double dt_fraction = (voi_ - voi) / dt;\n'
#                 computeEV += f'  double value;\n'
#                 computeEV += f'  if (index == {delay_variable_index}) {{\n'
#                 computeEV += f'    if (voi_ < variables_[{delay_amount_index}]) {{\n'
#                 computeEV += f'      if (buffersInitialised != true) {{return 0.0;}};\n'
#                 computeEV += f'      if (time_dof_0d == -1) {{;\n'
#                 if ext_variable["state_or_variable"] == "state":
#                     computeEV += f'        storeBufferVariable({delay_variable_index}, states_[{state_index}]);\n'
#                 else:
#                     computeEV += f'        storeBufferVariable({delay_variable_index}, variables_[{variable_index}]);\n'
#                 computeEV += f'      }};\n'
#                 computeEV += f'      return 0.0;\n'
#                 computeEV += f'    }} else {{;\n'
#                 computeEV += f'      if (time_dof_0d == -1) {{;\n'
#                 computeEV += f'        value = getBufferVariable({delay_variable_index}, 1.0, 1);\n'
#                 computeEV += f'      }} else {{;\n'
#                 computeEV += f'        value = getBufferVariable({delay_variable_index}, dt_fraction, 0);\n'
#                 computeEV += f'      }}\n'
#                 computeEV += f'      // save the current value of the variable to the circle buffer\n'
#                 computeEV += f'      if (time_dof_0d == -1) {{;\n'
#                 if ext_variable["state_or_variable"] == "state":
#                     computeEV += f'        storeBufferVariable({delay_variable_index}, states_[{state_index}]);\n'
#                 else:
#                     computeEV += f'        storeBufferVariable({delay_variable_index}, variables_[{variable_index}]);\n'
#                 computeEV += f'      }};\n'
#                 computeEV += f'      return value;\n'
#                 computeEV += f'    }};\n'
#                 computeEV += f'  }}\n'

#             elif ext_variable['variable_type'] == 'input':
#                 if ext_variable['state_or_variable'] == 'state':
#                     print('Input BC variable can not be a state variable, exiting')
#                     exit()
                    
#                 elif ext_variable['state_or_variable'] == 'variable':
#                     if ext_variable['coupled_to_type'] == 'FV_1d':
#                         computeEV += f'  if (index == {variable_index}) {{\n'
#                         if ext_variable['flow_or_pressure_bc'] == 'flow':
#                             computeEV += f'    int vessel1d_idx = cellml_index_to_vessel1d_info["{variable_index}"]["vessel1d_idx"];\n'
#                             computeEV += f'    int inlet0_or_outlet1_bc = cellml_index_to_vessel1d_info["{variable_index}"]["bc_inlet0_or_outlet1"];\n'
#                             if 'port_variable_index' in ext_variable.keys():
#                                 port_variable_index = ext_variable['port_variable_index']
#                                 if 'control_variable_index' in ext_variable.keys() and ext_variable['control_variable_index'] != None:
#                                     control_variable_index = ext_variable['control_variable_index']
#                                     computeEV += f'    double value = get_model1d_flow(vessel1d_idx, variables_[{port_variable_index}], inlet0_or_outlet1_bc, variables_[{control_variable_index}]);\n'
#                                 else:
#                                     computeEV += f'    double value = get_model1d_flow(vessel1d_idx, variables_[{port_variable_index}], inlet0_or_outlet1_bc);\n' 
#                             elif 'port_state_index' in ext_variable.keys():
#                                 port_state_index = ext_variable['port_state_index']
#                                 if 'control_variable_index' in ext_variable.keys() and ext_variable['control_variable_index'] != None:
#                                     control_variable_index = ext_variable['control_variable_index']
#                                     computeEV += f'    double value = get_model1d_flow(vessel1d_idx, states_[{port_state_index}], inlet0_or_outlet1_bc, variables_[{control_variable_index}]);\n'
#                                 else:
#                                     computeEV += f'    double value = get_model1d_flow(vessel1d_idx, states_[{port_state_index}], inlet0_or_outlet1_bc);\n'
#                         elif ext_variable['flow_or_pressure_bc'] == 'pressure':
#                             computeEV += f'    int vessel1d_idx = cellml_index_to_vessel1d_info["{variable_index}"]["vessel1d_idx"];\n'
#                             computeEV += f'    int inlet0_or_outlet1_bc = cellml_index_to_vessel1d_info["{variable_index}"]["bc_inlet0_or_outlet1"];\n'
#                             if 'port_variable_index' in ext_variable.keys():
#                                 port_variable_index = ext_variable['port_variable_index']
#                                 computeEV += f'    double value = get_model1d_pressure(vessel1d_idx, variables_[{port_variable_index}], inlet0_or_outlet1_bc);\n' 
#                             elif 'port_state_index' in ext_variable.keys():
#                                 port_state_index = ext_variable['port_state_index']
#                                 computeEV += f'    double value = get_model1d_pressure(vessel1d_idx, states_[{port_state_index}], inlet0_or_outlet1_bc);\n'
#                             print("Currently we don't implement the pressure coupling. ",
#                                   "As it would require solving a DAE in CellML")
#                             exit()
#                         computeEV += f'    return value;\n'
#                         computeEV += f'  }}\n'

#         computeEV += f"""
# return 0.0;
# }}
#         """


        # generate a function to compute external variables #XXX Bea's new code

        # TODO Finbar: for each external variable, include a label for its corresponding port
        # variable, ie when we need qBC, we give P_C and possibly R_T1.
        # TODO Finbar: create a config file which details the types of external variables we 
        # can have and their corresponding port variables. 
        # a new one will be created for each type of coupling.
        # TODO Beatrice: check compatibility with Finbar's code for delay variables
        
        computeEV = f"""
void Model0d::getDelayVariables(double voiLoc, double dtLoc, double *statesLoc, double *varLoc)
{{
    // First, deal with delay variables
    // TODO Bea: check with Finbar if I am doing it right (from his previous code)
    // this function should manage the delay variables correctly based on the value of time_dof_0d, as originally done by Finbar
    // even if now time_dof_0d is only used for delay variables, but not for other external variables coming from 1d-0d coupling
"""
        nDelay = 0
        idx_delay_vars = []
        for iEV, ext_variable in enumerate(external_variable_info):
            # print(ext_variable.keys())
            if ext_variable['variable_type'] == 'delay':
                nDelay +=1
                idx_delay_vars.append(iEV)

        if nDelay>0:
            computeEV += """
    double dt_fraction = (voiLoc-voi)/dt;
    int index;
    double value; 
"""
            for iEV in idx_delay_vars:
                ext_variable = external_variable_info[iEV]
                # print(ext_variable.keys())
                # variable or state index
                if ext_variable["state_or_variable"] == "state":
                    state_index = ext_variable['state_index']
                else:
                    variable_index = ext_variable['variable_index']

                # TODO Finbar: if time_dof_0d = 0 then we don't store the variables in the buffer, only store when it equals -1
                # set up so that when time_dof_0d is not -1 we don't step forward in the buffer, we only interpolate 
                # from it, we only step forward in the buffer when time_dof_0d is -1
                # use a time_0d variable to check where voi is in terms of time_0d
                # TODO Finbar: we could run a rough CVODE where we only save at time_output points, then linearly interpolate...
                # this would be quite easy to implement.. unsure about stability.
             
                variable_index = ext_variable['delay_variable_index']
                delay_variable_index = ext_variable['delay_variable_index']
                delay_amount_index = ext_variable['delay_amount_index']

                computeEV += f'\n'
                computeEV += f'    value = 0.0;\n'
                computeEV += f'    index = {delay_variable_index};\n'
                computeEV += f'    if (voiLoc < varLoc[{delay_amount_index}]) {{\n'
                computeEV += f'        if (buffersInitialised != true) {{\n'
                computeEV += f'            value = 0.0;\n'
                computeEV += f'        }}\n'
                computeEV += f'        else {{\n'
                computeEV += f'            if (time_dof_0d == -1) {{\n'
                if ext_variable["state_or_variable"] == "state":
                    computeEV += f'                storeBufferVariable({delay_variable_index}, statesLoc[{state_index}]);\n'
                else:
                    computeEV += f'                storeBufferVariable({delay_variable_index}, varLoc[{variable_index}]);\n'
                computeEV += f'            }}\n'
                computeEV += f'            value = 0.0;\n'
                computeEV += f'        }}\n'
                computeEV += f'    }} else {{\n'
                computeEV += f'        if (time_dof_0d == -1) {{\n'
                computeEV += f'            value = getBufferVariable({delay_variable_index}, 1.0, 1);\n'
                computeEV += f'        }} else {{\n'
                computeEV += f'            value = getBufferVariable({delay_variable_index}, dt_fraction, 0);\n'
                computeEV += f'        }}\n'
                computeEV += f'        // save the current value of the variable to the circle buffer\n'
                computeEV += f'        if (time_dof_0d == -1) {{\n'
                if ext_variable["state_or_variable"] == "state":
                    computeEV += f'            storeBufferVariable({delay_variable_index}, statesLoc[{state_index}]);\n'
                else:
                    computeEV += f'            storeBufferVariable({delay_variable_index}, varLoc[{variable_index}]);\n'
                computeEV += f'        }}\n'
                computeEV += f'    }}\n'
                computeEV += f'    varLoc[{variable_index}] = value;\n' #XXX TODO Bea: not sure about this!!!
                
        computeEV += f"""
}}
"""

        computeEV += """
void Model0d::getExternalVariables(double voiLoc, double dtLoc, double *statesLoc, double *varLoc)
{
    // Next, deal with 1d-0d coupling
"""

        if self.couple_to_1d:
            computeEV += """
    if (N1d0d>0) {
        // Step 1: send data
        // S1: compute, write and send time & internal time step data to coupler
        zero_data_dt[0] = voiLoc;
        zero_data_dt[1] = dtLoc;
        write_pipe_dt.write(reinterpret_cast<const char*>(zero_data_dt), DATA_LENGTH * sizeof(double));
        write_pipe_dt.flush();
        
        // write...
        std::string pipeID;
        int idx_port;
        int type_port;
        int idx_res;
        int type_bc;
        for (int i = 0; i < N1d0d; ++i) {
            pipeID = std::to_string(i+1);
            if (pipe_1d_0d_info[pipeID][\"port_volume_sum\"] != 1){
                idx_port = pipe_1d_0d_info[pipeID][\"port_idx\"];
                type_port = pipe_1d_0d_info[pipeID][\"port_state0_or_var1\"];
                idx_res = pipe_1d_0d_info[pipeID][\"R_T_variable_idx\"];
                type_bc = pipe_1d_0d_info[pipeID][\"cellml_bc_flow0_or_press1\"];
                if (type_bc==0){ // flow bc --> pressure port --> pressure and resistance are passed to 1d    
                    if (type_port==0){ // state
                        zero_data[i][0] = statesLoc[idx_port];
                    } else if (type_port==1){ // variable
                        zero_data[i][0] = varLoc[idx_port];
                    }
                    // zero_data[i][1] = varLoc[idx_res];
                    if (idx_res<0){
                        zero_data[i][1] = 0.0;
                    } else{
                        zero_data[i][1] = varLoc[idx_res];
                    }
                } else if (type_bc==1){ // pressure bc --> flow port --> flow is passed to 1d
                    if (type_port==0){ // state
                        zero_data[i][0] = statesLoc[idx_port];
                    } else if (type_port==1){ // variable
                        zero_data[i][0] = varLoc[idx_port];
                    }
                }
                // S1: compute, write and send data to coupler for each 1d-0d connection
                write_pipe[i].write(reinterpret_cast<const char*>(zero_data[i]), DATA_LENGTH * sizeof(double));
                write_pipe[i].flush();
            }
            else {
                std::cerr << \"ERROR :: this is a volume sum port, use its dedicated pipes for this connection. Exiting.\" << std::endl;
                exit(1);
            }
        }
        // ...wait to receive back...
        // Step 2: receive and read data with computed 0d coupling variables
        for (int i = 0; i < N1d0d; ++i) {
            pipeID = std::to_string(i+1);
            if (pipe_1d_0d_info[pipeID][\"port_volume_sum\"] != 1){ 
                read_pipe[i].read(reinterpret_cast<char*>(parent_data[i]), DATA_LENGTH * sizeof(double));
            }
            else {
                std::cerr << \"ERROR :: this is a volume sum port, use its dedicated pipes for this connection. Exiting.\" << std::endl;
                exit(1);
            }
        } 
        int idx_cellml;
        for (int i = 0; i < N1d0d; ++i) {
            pipeID = std::to_string(i+1);
            if (pipe_1d_0d_info[pipeID][\"port_volume_sum\"] != 1){
                idx_cellml = pipe_1d_0d_info[pipeID][\"cellml_idx\"];
                varLoc[idx_cellml] = parent_data[i][0];
            }
            else {
                std::cerr << \"ERROR :: this is a volume sum port, use its dedicated pipes for this connection. Exiting.\" << std::endl;
                exit(1);
            }
        }
        // Step 3: now exit this function and compute rates
    }
"""
        computeEV += """
}
"""
        
        computeEV += """
void Model0d::getExternalVariablesVolume(double *varLoc)
{
    // Get volume sum from 1d to be added to 0d volume sum to track total volume in the system
    // This is done at the end of each time step, but not for intermediate/internal time steps taken by the solver.  
"""
        if self.couple_volume_sum:
            computeEV += """
    if (N1d0dTot>0) {
        std::string pipeID;
        int idx_cellml;
        for (int i = 0; i < N1d0dTot; ++i) {
            pipeID = std::to_string(i+1);
            if (pipe_1d_0d_info[pipeID][\"port_volume_sum\"] == 1){ 
                read_pipe_vol.read(reinterpret_cast<char*>(parent_data_vol), DATA_LENGTH * sizeof(double));
                idx_cellml = pipe_1d_0d_info[pipeID][\"cellml_idx\"];
                varLoc[idx_cellml] = parent_data_vol[0];
            }
        }
    }
"""

        computeEV += """
}
"""

        # print("#################################################")
        # print("computeEV")
        # print(computeEV)


#         if self.couple_to_1d: #XXX Finbar's old code
#             # external interaction functions
#             externalInteractionFunctions = """
# void Model0d::connect_to_model1d(Model1d* model1d){
#     model1d_ptr = model1d; 
# }
#             """
#             if 'control_variable_index' in ext_variable.keys() and ext_variable['control_variable_index'] != None:
#                 externalInteractionFunctions += """
# double Model0d::get_model1d_flow(int vessel_idx, double p_C, int input0_output1_bc, double R_T1_wCont){
#                 """
#             else:
#                 externalInteractionFunctions += """
# double Model0d::get_model1d_flow(int vessel_idx, double p_C, int input0_output1_bc){
#                 """
#             externalInteractionFunctions += """
#     // get the flow from the 1d model

#     // get qBC and step forward the fluctuation
#             """
            
#             if self.solver == 'RK4':
#                 externalInteractionFunctions += """
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
#             """

#             if 'control_variable_index' in ext_variable.keys() and ext_variable['control_variable_index'] != None:
#                 externalInteractionFunctions += """
#     double flow = model1d_ptr->getqBCAndAddToFluctWrap(vessel_idx, add_to_fluct, 
#                                                 time_0d, time_int_weight, 
#                                                 p_C, input0_output1_bc, R_T1_wCont)/pow(10,6); 
#     // divide by 10^6 to get m3/s from ml/s
#     return flow;
# }
#                 """
#             else:
#                 externalInteractionFunctions += """
#     double flow = model1d_ptr->getqBCAndAddToFluctWrap(vessel_idx, add_to_fluct, 
#                                                 time_0d, time_int_weight, 
#                                                 p_C, input0_output1_bc)/pow(10,6); 
#     // divide by 10^6 to get m3/s from ml/s
#     return flow;
# }
#                 """
                
#             # externalInteractionFunctions += """
                
#             #     // below is what I did previously
#             #     // double * all_vals = (double *) malloc(10*sizeof(double));
#             #     // if (input0_output1_bc == 0){
#             #     //     // TODO Finbar, should be evalSolSpaceTime, where the time is an input and 
#             #     //     // it interpolates the solution
#             #     //     model1d_ptr->evalSol(vessel_idx, 0.0, 0, all_vals); 
#             #     // } else {
#             #     //     double xSample = model1d_ptr->vess[vessel_idx].L;
#             #     //     int iSample = model1d_ptr->vess[vessel_idx].NCELLS - 1;
#             #     //     model1d_ptr->evalSol(vessel_idx, xSample, iSample, all_vals); 
#             #     // }
#             #     // double flow = all_vals[1]/pow(10,6); // index 1 is the flow, divide by 10^6 to get m3/s from ml/s
#             #     // // TODO make the conversion of units more generic
#             #     // // std::cout << "flow: " << std::scientific << flow << std::endl;
#             #     return flow; // 
#             # } 
#             # """

#             # Currently we don't implement the pressure coupling
#             # double Model0d::get_model1d_pressure(int vessel_idx, double qBC, int input0_output1_bc){
#             #     // get the pressure from the 1d model
#             # """ 
            
#             # if self.solver == 'RK4':
#             #     externalInteractionFunctions += """
                
#             #     int add_to_fluct = 1;
#             #     double time_int_weight = 0.0;
#             #     double time_0d = 0.0;
#             #     if (time_dof_0d == 0) {
#             #         time_int_weight = 1.0/6.0;
#             #         time_0d = voi;
#             #     }
#             #     else if (time_dof_0d == 1 || time_dof_0d == 2) {
#             #         time_int_weight = 1.0/3.0;
#             #         time_0d = voi + dt/2.0;
#             #     }
#             #     else if (time_dof_0d == 3) {
#             #         time_int_weight = 1.0/6.0;
#             #         time_0d = voi + dt;
#             #     }
#             #     else if (time_dof_0d == -1) {
#             #         time_int_weight = 0.0;
#             #         add_to_fluct = 0;
#             #         time_0d = voi + dt;
#             #     }
#             #     else {
#             #         std::cout << "time_dof_0d is not set correctly" << std::endl;
#             #         exit(1);
#             #     }
#             # """

#             # externalInteractionFunctions += """
#             #     double pressure = model1d_ptr->getPressureAndAddToFluct(vessel_idx, add_to_fluct, 
#             #                                                 time_0d, time_int_weight, 
#             #                                                 qBC, input0_output1_bc)/10; 
#             #     # divide by 10 to go from dynes/cm^2 to Pa
#             #     return pressure;

#             #     // below is what I did previously
#             #     // double * all_vals = (double *) malloc(8*sizeof(double));
#             #     // if (input0_output1_bc == 0){
#             #     //     model1d_ptr->sampleMid(vessel_idx, all_vals, 0.0); 
#             #     // } else {
#             #     //     double xSample = model1d_ptr->vess[vessel_idx].L;
#             #     //     model1d_ptr->sampleMid(vessel_idx, all_vals, xSample); 
#             #     // }
#             #     // return all_vals[4]; // index 4 is the pressure 
#             # }
#             # """

#         else:
#             externalInteractionFunctions = ""

        # externalInteractionFunctions += """
        externalInteractionFunctions = """

void Model0d::initialise(double voiLoc, double *statesLoc, double *ratesLoc, double *varLoc,
                         const char* statesPath, const char* varsPath, bool useFirstRow)
{    
    // Default initialisation
    initialiseVariables(voiLoc, statesLoc, ratesLoc, varLoc);

    auto isValid = [](const char* p) { 
        if (p == nullptr) return false;

        std::string s(p);
        return (s != \"None\" && !s.empty()); 
    };

    // Load initial state from files and overwrite defaults
    if (isValid(statesPath)) {
        std::cout << \"Loading initial states from file: \" << statesPath << std::endl;
        loadFromFile(statesPath, statesLoc, STATE_INFO, STATE_COUNT, useFirstRow);
    }
    if (isValid(varsPath)) {
        std::cout << \"Loading initial variables from file: \" << varsPath << std::endl;
        loadFromFile(varsPath, varLoc, VARIABLE_INFO, VARIABLE_COUNT, useFirstRow);
    }
}

void Model0d::initialiseVariablesAndComputeConstants() {
    // initialiseVariables(voi, states, rates, variables);
    initialise(voi, states, rates, variables, initStatesPath, initVarsPath, false);
    
    computeComputedConstants(variables);
    getExternalVariablesVolume(variables);
    computeNonExternalVariables(voi, states, rates, variables);
    // computeRates(voi, dt, states, rates, variables); // is this needed here?
    // computeVariables(voi, dt, states, rates, variables); // variables all already computed within computeRates() function
        """
        if self.solver == 'CVODE':
            externalInteractionFunctions += """
    // reinitialise the CVODE solver with the initialised state variables 
    y = N_VMake_Serial(STATE_COUNT, states, context);
    CVodeReInit(solver, voi, y);
        """
        elif self.solver == 'PETSC':
            externalInteractionFunctions += """
    PetscScalar *y_arr;
    VecGetArray(y, &y_arr);
    std::copy(states, states + STATE_COUNT, y_arr);
    VecRestoreArray(y, &y_arr);

    PetscCallVoid(TSSetSolution(ts, y)); // set the initial conditions
    PetscCallVoid(TSSetTime(ts, voi)); // set the initial time
        """

        externalInteractionFunctions += """
}
        """ 

        
        # TODO make the integration scheme implementation general
        solveOneStepFunction = """
        """
#         if self.solver == 'CVODE':
#             solveOneStepFunction += """
# void Model0d::solveOneStepCVODE(double dtLoc)
# {
# """
#         else:
#             solveOneStepFunction += """
# void Model0d::solveOneStepExpl(double dtLoc)
# {
# """
        solveOneStepFunction += """
void Model0d::solveOneStep(double dtLoc)
{
"""
        if self.solver == 'CVODE' or self.solver == 'PETSC': 
            solveOneStepFunction += f"""
    if (solver_0d != \"{self.solver}\") {{
        std::cout << \"Model0d::solveOneStep :: ODE solver chosen is not {self.solver}: \" << solver_0d << std::endl;
        exit(1);
    }}
"""
        solveOneStepFunction += """    dt = dtLoc;
"""
        if self.couple_to_1d:
            solveOneStepFunction += """
    // matching global time step between 1d and 0d
    if (N1d0dTot>0){
        zero_data_dt[0] = voi;
        zero_data_dt[1] = dt;
        write_pipe_dt.write(reinterpret_cast<const char*>(zero_data_dt), DATA_LENGTH * sizeof(double));
        write_pipe_dt.flush();
    }

    if (N1d0dTot>0){
        read_pipe_dt.read(reinterpret_cast<char*>(parent_data_dt), DATA_LENGTH * sizeof(double));
        dt = parent_data_dt[1];
    }
"""

        if self.solver == 'CVODE':
            solveOneStepFunction += """
    voiEnd = voi + dt;
    voiEnd = std::round(voiEnd / tolTime) * tolTime;
    time_dof_0d = 0;

    states = N_VGetArrayPointer_Serial(y);

    CVodeSetStopTime(solver, voiEnd);
    int flag = CVode(solver, voiEnd, y, &voi, CV_NORMAL); 
    
    if (flag<0){ // error occurred
        std::cerr << \"CVODE :: Error occurred with flag \" << flag << \". Exiting.\" << std::endl;
        exit(1);
    }
    else if (flag==0){ // CV_SUCCESS
        // Successful function return.
    }
    else if (flag==1){ // CV_TSTOP_RETURN
        // CVode succeeded by reaching the specified stopping point.
        // std::cout << "CVODE :: Stop time reached with flag " << flag << std::endl;
    }
    else if (flag==2){ // CV_ROOT_RETURN
        // CVode succeeded and found one or more roots.
        std::cout << "CVODE :: Root found with flag " << flag << std::endl;
    }
    else if (flag==99){ // CV_ROOT_RETURN
        // CVode succeeded but an unusual situation occurred.
        std::cout << "CVODE :: WARNING :: Unusual situation occurred with flag " << flag << ". CHECK YOUR OUTPUTS." << std::endl;
    }

    states = N_VGetArrayPointer_Serial(y);
    time_dof_0d = -1;
"""
            if self.couple_to_1d:
                solveOneStepFunction += """
    if (N1d0dTot>0){
        zero_data_dt[1] = -999.; // signal to parent that time step is done
        write_pipe_dt.write(reinterpret_cast<const char*>(zero_data_dt), DATA_LENGTH * sizeof(double));
        write_pipe_dt.flush();
        zero_data_dt[1] = dtLoc; // reset value
    }

    if (N1d0dTot>0){
        read_pipe_dt.read(reinterpret_cast<char*>(parent_data_dt), DATA_LENGTH * sizeof(double));
        dt = parent_data_dt[1];
        // std::cout << \"0d solver :: Received negative 0d internal time step dtLoc \" << dt << std::endl;
        dt = dtLoc; // reset value
    }
"""
            solveOneStepFunction += """
    // computeVariables(voi, states, rates, variables);
    getExternalVariablesVolume(variables);
    computeNonExternalVariables(voi, states, rates, variables);
    getDelayVariables(voi, dt, states, variables);
    userData->tOld = voi; // &voi is already incremented by dt within CVode() function
}
"""
        elif self.solver == 'PETSC':
            solveOneStepFunction += """
    voiEnd = voi + dt;
    // voiEnd = std::round(voiEnd / tolTime) * tolTime;
    time_dof_0d = 0;

    PetscCallVoid(TSSetTimeStep(ts, dt)); // set time step
    PetscCallVoid(TSSetMaxTime(ts, voiEnd)); // set final time for the current one-step solve
    PetscCallVoid(TSSolve(ts, y)); // steps the requested number of time steps
    
    PetscReal t;
    TSGetTime(ts, &t);
    voi = t;

    const PetscScalar *y_arr;
    VecGetArrayRead(y, &y_arr);
    std::copy(y_arr, y_arr + STATE_COUNT, states);
    VecRestoreArrayRead(y, &y_arr);

    time_dof_0d = -1;
"""

            if self.couple_to_1d:
                solveOneStepFunction += """
    if (N1d0dTot>0){
        zero_data_dt[1] = -999.; // signal to parent that time step is done
        write_pipe_dt.write(reinterpret_cast<const char*>(zero_data_dt), DATA_LENGTH * sizeof(double));
        write_pipe_dt.flush();
        zero_data_dt[1] = dtLoc; // reset value
    }

    if (N1d0dTot>0){
        read_pipe_dt.read(reinterpret_cast<char*>(parent_data_dt), DATA_LENGTH * sizeof(double));
        dt = parent_data_dt[1];
        // std::cout << \"0d solver :: Received negative 0d internal time step dtLoc \" << dt << std::endl;
        dt = dtLoc; // reset value
    }
"""
            solveOneStepFunction += """
    // computeVariables(voiEnd, states, rates, variables);
    getExternalVariablesVolume(variables);
    computeNonExternalVariables(voi, states, rates, variables);
    getDelayVariables(voi, dt, states, variables);
    userData->tOld = voi;
}
"""

        else:
            solveOneStepFunction += """
    time_dof_0d = 0;
    if (solver_0d == \"explEul\") {
        // Step 1
        for (size_t i = 0; i < STATE_COUNT; ++i) {
            states0[i] = states[i];
        }
        computeRates(voi, 0.0*dt, states0, k1, variables);
        for (size_t i = 0; i < STATE_COUNT; ++i) {
            rates[i] = k1[i];
            states[i] = states0[i] + dt*rates[i];
        }
        time_dof_0d = -1;  
        // //computeVariables(voi+dt, states, rates, variables);
        // getExternalVariablesVolume(variables);
        // computeNonExternalVariables(voi+dt, states, rates, variables);
        // getDelayVariables(voi, dt, states, variables);
    }
    else if (solver_0d == \"Heun\") {
        // Step 1
        for (size_t i = 0; i < STATE_COUNT; ++i) {
            states0[i] = states[i];
        }
        computeRates(voi, 0.0*dt, states0, k1, variables);
        for (size_t i = 0; i < STATE_COUNT; ++i) {
            states[i] = states0[i] + dt*k1[i];
        }
        // computeNonExternalVariables(voi+dt, states, k1, variables);

        // Step 2
        computeRates(voi+dt, dt, states, k2, variables);
        for (size_t i = 0; i < STATE_COUNT; ++i) {
            rates[i] = 0.5*(k1[i] + k2[i]);
            states[i] = states0[i] + dt*rates[i];
        }
        time_dof_0d = -1;
        // // computeVariables(voi+dt, states, rates, variables);
        // getExternalVariablesVolume(variables);
        // computeNonExternalVariables(voi+dt, states, rates, variables);
        // getDelayVariables(voi, dt, states, variables);
    }
    else if (solver_0d == \"midpoint\") {
        // Step 1
        for (size_t i = 0; i < STATE_COUNT; ++i) {
            states0[i] = states[i];
        }
        computeRates(voi, 0.0*dt, states0, k1, variables);
        for (size_t i = 0; i < STATE_COUNT; ++i) {
            states[i] = states0[i] + 0.5*dt*k1[i];
        }
        // computeNonExternalVariables(voi+0.5*dt, states, k1, variables);

        // Step 2
        computeRates(voi+0.5*dt, 0.5*dt, states, k2, variables);
        for (size_t i = 0; i < STATE_COUNT; ++i) {
            states[i] = states0[i] + dt*k2[i];
        }
        time_dof_0d = -1;
        // // computeVariables(voi+dt, states, k2, variables);
        // getExternalVariablesVolume(variables);
        // computeNonExternalVariables(voi+dt, states, k2, variables);
        // getDelayVariables(voi, dt, states, variables);
    }
    else if (solver_0d == \"RK4\") {
        // Step 1
        for (size_t i = 0; i < STATE_COUNT; ++i) {
            states0[i] = states[i];
        }
        computeRates(voi, 0.0*dt, states0, k1, variables);
        for (size_t i = 0; i < STATE_COUNT; ++i) {
            states[i] = states0[i] + 0.5*dt*k1[i];
        }
        // computeNonExternalVariables(voi+0.5*dt, states, k1, variables);
        
        // Step 2
        computeRates(voi+0.5*dt, 0.5*dt, states, k2, variables);
        for (size_t i = 0; i < STATE_COUNT; ++i) {
            states[i] = states0[i] + 0.5*dt*k2[i];
        }
        // computeNonExternalVariables(voi+0.5*dt, states, k2, variables);
    
        // Step 3
        computeRates(voi+0.5*dt, 0.5*dt, states, k3, variables);
        for (size_t i = 0; i < STATE_COUNT; ++i) {
            states[i] = states0[i] + dt*k3[i];
        }
        // computeNonExternalVariables(voi+dt, states, k3, variables);
        
        // Step 4
        computeRates(voi+dt, dt, states, k4, variables);
        for (size_t i = 0; i < STATE_COUNT; ++i) {
            rates[i] = wRK4[0]*k1[i] + wRK4[1]*k2[i] + wRK4[2]*k3[i] + wRK4[3]*k4[i];
            states[i] = states0[i] + dt*rates[i];
        }
        time_dof_0d = -1;
        // // computeVariables(voi+dt, states, rates, variables);
        // getExternalVariablesVolume(variables);
        // computeNonExternalVariables(voi+dt, states, rates, variables);
        // getDelayVariables(voi, dt, states, variables);   
    }
    else{
        std::cout << \"Model0d::solveOneStep :: explicit ODE solver not defined \" << solver_0d << std::endl;
        exit(1);
    }

    voi += dt;
    voi = std::round(voi / tolTime) * tolTime;
"""

            if self.couple_to_1d:
                solveOneStepFunction += """
    if (N1d0dTot>0){
        zero_data_dt[1] = -999.; // signal to parent that time step is done
        write_pipe_dt.write(reinterpret_cast<const char*>(zero_data_dt), DATA_LENGTH * sizeof(double));
        write_pipe_dt.flush();
        zero_data_dt[1] = dtLoc; // reset value
    }

    if (N1d0dTot>0){
        read_pipe_dt.read(reinterpret_cast<char*>(parent_data_dt), DATA_LENGTH * sizeof(double));
        dt = parent_data_dt[1];
        // std::cout << \"0d solver :: Received negative 0d internal time step dtLoc \" << dt << std::endl;
        dt = dtLoc; // reset value
    }
"""

            solveOneStepFunction += """
    getExternalVariablesVolume(variables);
    computeNonExternalVariables(voi, states, rates, variables);
    getDelayVariables(voi, dt, states, variables);
}
"""


#         if self.solver == 'forward_euler':
#             solveOneStepFunction += """
            
#     time_dof_0d = 0;
#     computeRates(voi, states, rates, variables);

#     for (size_t i = 0; i < STATE_COUNT; ++i) {
#         // simple forward Euler integration
#         states[i] = states[i] + dt * rates[i];
#     time_dof_0d = -1;
#     computeVariables(voi, states, rates, variables);
#     voi += dt;
#     }
# }
#             """
#         elif self.solver == 'RK4':
#             solveOneStepFunction += """
#     // RK4 integration
#     // first step: calculate k1
#     for (size_t i = 0; i < STATE_COUNT; ++i) {
#         temp_states[i] = states[i];
#     }
#     time_dof_0d = 0;
#     computeRates(voi, temp_states, k1, variables);

#     // second step: calculate k2
#     for (size_t i = 0; i < STATE_COUNT; ++i) {
#         temp_states[i] = states[i] + dt/2.0 * k1[i];
#     }
#     time_dof_0d = 1;
#     computeRates(voi+dt/2.0, temp_states, k2, variables);
    
#     // third step: calculate k3
#     for (size_t i = 0; i < STATE_COUNT; ++i) {
#         temp_states[i] = states[i] + dt/2.0 * k2[i];
#     }
#     time_dof_0d = 2;
#     computeRates(voi+dt/2.0, temp_states, k3, variables);
    
#     // third step: calculate k4
#     for (size_t i = 0; i < STATE_COUNT; ++i) {
#         temp_states[i] = states[i] + dt * k3[i];
#     }
#     time_dof_0d = 3;
#     computeRates(voi+dt, temp_states, k4, variables);

#     for (size_t i = 0; i < STATE_COUNT; ++i) {
#         rates[i] = 1.0/6.0 * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
#         states[i] = states[i] + dt * rates[i];
#     }
#     // when we set time_dof_0d to -1, we only get q_BC
#     // from the ones evaluated at the 4 time dofs. we dont step the fluct
#     time_dof_0d = -1;
#     computeVariables(voi, states, rates, variables);
#     voi += dt;
#         """
#             if self.couple_to_1d:
#                 solveOneStepFunction += """
#         // TODO Finbar, is the below correct? it was used in the 1d code.
#     model1d_ptr->correctTimeLTS(voi,model1d_ptr->dtMaxLTS);
# }
#         """
#             else:
#                 solveOneStepFunction += """
# }
#         """

#         elif self.solver == 'CVODE':
#             solveOneStepFunction += """ 

#     voiEnd = voi + dt;
#     // CVodeSetStopTime(solver, voiEnd);

#     CVode(solver, voiEnd, y, &voi, CV_NORMAL);

#     // Compute our variables.

#     states = N_VGetArrayPointer_Serial(y);
#     computeVariables(voiEnd, states, rates, variables);
#     voi += dt;
#     }
#             """
        

        # print("#################################################")
        # print("solveOneStepFunction")
        # print(solveOneStepFunction)
        

        # typedef void (*computeRatesType)(double, double *, double *, double *);
        if self.solver == 'CVODE':
            userDataHeader = """
class UserOdeData
{
public:

    explicit UserOdeData(double *pVariables, Model0d::FunctionType pComputeRates, void* pCvodeMem);

    double* variables() const;
    Model0d::FunctionType computeRates() const;
    void* cvodeMem() const;

    sunrealtype tOld = 0.0;
    long int nstOld = 0;

private:
    double *mVariables;
    Model0d::FunctionType mComputeRates;
    void *mCvodeMem;
};

"""
        else:
            userDataHeader = """
class UserOdeData
{
public:

    explicit UserOdeData(double *pVariables, Model0d::FunctionType pComputeRates);

    double* variables() const;
    Model0d::FunctionType computeRates() const;
    double tOld = 0.0;

private:
    double *mVariables;
    Model0d::FunctionType mComputeRates;
};

"""
        UserDataCC = """
        """

        if self.solver == 'CVODE':
            UserDataCC += """
UserOdeData::UserOdeData(double *pVariables, Model0d::FunctionType pComputeRates, void* pCvodeMem) :
    mVariables(pVariables),
    mComputeRates(pComputeRates),
    mCvodeMem(pCvodeMem)
{
}
"""
        else:
            UserDataCC += """
UserOdeData::UserOdeData(double *pVariables, Model0d::FunctionType pComputeRates) :
    mVariables(pVariables),
    mComputeRates(pComputeRates)
{
}
"""

        UserDataCC += """
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
            UserDataCC += """
//==============================================================================

void* UserOdeData::cvodeMem() const
{
    return mCvodeMem;
}
"""


        if self.solver == 'CVODE':
            funcCC = """
int func(double voi_, N_Vector y, N_Vector yDot, void *userData)
{
    // Function that computes the RHS of the ODE system, in the form of f(t, y, ydot, user_data)
    // - t : the current value of the independent variable (time)
    // - y : the current value of the dependent variable vector (the states)
    // - ydot : the output vector f(t,y) (the rates)
    // - user_data : the user_data pointer

    UserOdeData *realUserData = static_cast<UserOdeData*>(userData);
    sunrealtype t_solver;
    sunrealtype hcur;
    sunrealtype hlast;
    long int nst;
    sunrealtype dtLoc;
    double tolTime = 1e-11; // 1e-13;

    CVodeGetCurrentTime(realUserData->cvodeMem(), &t_solver);
    CVodeGetCurrentStep(realUserData->cvodeMem(), &hcur); // current internal time step
    // CVodeGetLastStep(realUserData->cvodeMem(), &hlast); // last time step taken
    CVodeGetNumSteps(realUserData->cvodeMem(), &nst); // total internal steps so far

    if (nst == realUserData->nstOld){
        // CVODE did not take a step
        dtLoc = 0.0;
        // std::cout << std::setprecision(20)
        //             << "WARNING: CVODE did not take a step at time " << voi_
        //             << " with solver time " << t_solver
        //             << std::endl;
    } else {
        dtLoc = hcur;
        realUserData->nstOld = nst;
    } 
    // std::cout << std::setprecision(20)
    //                 << "CVODE has internal time step " << dtLoc
    //                 << " at time " << voi_
    //                 << " with previous time " << realUserData->tOld
    //                 << std::endl;

    realUserData->tOld = voi_;

    realUserData->computeRates()(voi_, dtLoc, N_VGetArrayPointer_Serial(y), 
                            N_VGetArrayPointer_Serial(yDot), realUserData->variables());
    
    return 0;
}
           
        """
            
        elif self.solver == 'PETSC':
            funcCC = """
// PETSc RHS callable function
PetscErrorCode TSRHSfunc(TS ts, PetscReal t, Vec Y, Vec Ydot, void *ctx)
{
    // Function that computes the RHS of the ODE system dy/dt.
    // void *ctx : generic pointer that PETSc passes straight through this RHS function, to let you attach any data you want to the solver.

    // Recover the C++ object that owns the model data
    Model0d *model = static_cast<Model0d*>(ctx); 

    // Temporary pointers that will point into the PETSc vectors
    const PetscScalar *y_ptr; // read-only view of the current state vector
    PetscScalar *ydot_ptr; // writable view where we will store dy/dt
    
    // Get the local time step that the solver is using right now.
    PetscReal dtLoc = model->dt;
    
    // PetscReal tEndLoc;
    // TSGetMaxTime(ts, &tEndLoc); 
    // std::cout << std::setprecision(16)
    //             << "TSRHSfunc :: called at time " << t
    //             << " || dt = " << dtLoc
    //             << " || final time = " << tEndLoc
    //             << " || old time = " << model->userData->tOld
    //             << std::endl;
    
    // Get raw C-style arrays from the PETSc vectors.
    VecGetArrayRead(Y, &y_ptr); // read-only access to Y  (the state vector)
    VecGetArray(Ydot, &ydot_ptr); // read/write access to Ydot (the RHS vector)

    // Call the computeRates routine for computing the RHS, which expects plain double* pointers.
    model->computeRates(t, dtLoc, 
                        (double*)y_ptr, 
                        (double*)ydot_ptr, 
                        model->variables);

    // Restore the PETSc vectors / Release the raw arrays.
    VecRestoreArrayRead(Y, &y_ptr);
    VecRestoreArray(Ydot, &ydot_ptr);
    
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
int main(int argc, char* argv[]){
"""
        if self.solver == 'PETSC':
            mainScript += """
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc, &argv, NULL, NULL);
    std::cout << "PETSc :: initialization complete " << ierr << std::endl;

    std::cout << "Starting 0D model simulation..." << std::endl;
    {
"""
        else:
            mainScript += """
    std::cout << "Starting 0D model simulation..." << std::endl;
"""

        mainScript += f"""
    double end_time = 20.0;
    double save_time = 18.0;
    double dt = {self.dtSolver};
    double eps = 1e-12;
    std::string ODEsolver = \"{self.solver}\";
    
    Model0d model0d_inst;
    model0d_inst.set_ode_solver(ODEsolver);
    std::string resFold = \"./simulation_outputs_cpp/\";
    model0d_inst.openOutputFiles(resFold);
    model0d_inst.voi = 0.0;
    model0d_inst.dt = dt;
    model0d_inst.initialiseVariablesAndComputeConstants();
    """
        if len(variables_to_delay_info) > 0:
            mainScript += """
    model0d_inst.initBuffers(dt);
    """
        mainScript += """
    
    if (model0d_inst.voi>=save_time-eps){
        model0d_inst.writeOutput(model0d_inst.voi);
    }

    while (model0d_inst.voi < end_time-eps) {
        model0d_inst.solveOneStep(dt);
        
        if (model0d_inst.voi>=save_time-eps){
            model0d_inst.writeOutput(model0d_inst.voi);
        }

        std::cout << "time: " << model0d_inst.voi << " states[0]: " << model0d_inst.states[0] << std::endl;
        // std::cout << "time: " << model0d_inst.voi << " V_delay: " << model0d_inst.variables[2] << std::endl;
    }
    model0d_inst.closeOutputFiles();

    // TODO autogenerate a dict of variable names to print with variables

    printf("################################################################################\\n");
    printf("Final values:\\n");
    printf("  time: ");
    printf("%f\\n", model0d_inst.voi);
    printf("  states:\\n");
    for (size_t i = 0; i < model0d_inst.STATE_COUNT; ++i) {
        printf("%f\\n", model0d_inst.states[i]);
    }
    printf("  variables:\\n");
    for (size_t i = 0; i < model0d_inst.VARIABLE_COUNT; ++i) {
        printf("%f\\n", model0d_inst.variables[i]);
    }
    printf("################################################################################\\n");
"""
        if self.solver == 'PETSC':
            mainScript += """
    }

    ierr = PetscFinalize();
    std::cout << "PETSc :: finalization complete " << ierr << std::endl;

    printf("Done.\\n");
    return 0;
}
"""
        else:
            mainScript += """
    printf("Done.\\n");
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
            # f.write(externalInteractionHeaders)
            f.write(otherHeaderInits)
            if len(variables_to_delay_info) > 0:
                f.write(bufferPartsHeader)
            f.write(classFinisherHeader)
            f.write(userDataHeader)
   
        # save implementation to file
        with open(os.path.join(self.cpp_generated_models_dir, f'{self.output_cpp_file_name}.cc'), 'w') as f:

            f.write(preClassStuff)
            f.write(preSourceStuff) # this has to be below the #include "model.h" in implementationCode()
            if self.solver == 'CVODE' or self.solver == 'PETSC':
                f.write(funcCC)
            f.write(classInit)
            f.write(postClassInit)
            # f.write(non_external_variables_function)
            f.write(externalInteractionFunctions)

            f.write(solverInitFunction) #XXX Bea's addition
            f.write(pipesFunctions) #XXX Bea's addition
            f.write(outputFunctions) #XXX Bea's addition

            f.write(computeEV)
            f.write(solveOneStepFunction)
            f.write(UserDataCC)
            
            if len(variables_to_delay_info) > 0:
                f.write(circularBuffer)
                f.write(bufferParts)
            
            # if self.create_main:
            if (self.create_main and not self.couple_to_1d):
                f.write(mainScript)
        
        
        if (self.create_main and self.couple_to_1d):
            #XXX create also the main script main0d.cpp to be used to run the 0d model from the 1d-0d coupler
            with open(generators_dir_path+'/main0dTemplate.cpp', 'r') as file:
                lines = file.readlines()

            mainScript = """
"""
            
            # T0 = -1.0
            # nCC = -1
            for line in lines:
                if self.solver == 'PETSC':
                    if '// #include <petscsys.h>' in line:
                        line = line.replace('// #include <petscsys.h>', '#include <petscsys.h>')
                    if 'PetscErrorCode' in line:
                        line = line.replace('// ', '')
                    if 'PetscInitialize' in line:
                        line = line.replace('// ', '')
                    if 'PetscFinalize' in line:
                        line = line.replace('// ', '')

                if 'std::string resFold =' in line:
                    model_name = self.file_prefix
                    if self.file_prefix.endswith("_0d"):
                        model_name = self.file_prefix[:-3]
                    # results0d_folder = os.path.join(self.cpp_generated_models_dir+"/../..", f"simulation_outputs_cpp/{model_name}/")
                    results0d_folder = os.path.join(self.cpp_generated_models_dir+"/../..", f"simulation_outputs_cpp/")
                    if not os.path.exists(results0d_folder):
                        os.makedirs(results0d_folder)
                    results0d_folder = os.path.join(results0d_folder, f"{model_name}/")
                    # if not os.path.exists(results0d_folder):
                    #     os.makedirs(results0d_folder)
                    
                    results0d_folder_tmp = "./"
                    line = line.replace(f"std::string resFold = \"{results0d_folder_tmp}\";", f"std::string resFold = \"{results0d_folder}\";")
                
                # if 'ODEsolver = "' in line:
                # if 'T0 = ' in line:
                # if 'nCC = ' in line:
                # if 'modelName = "' in line:

                if 'const double tSaveRes = ' in line:
                    idx0 = line.find("=")
                    idx1 = line.find(";")
                    line = line.replace(line[idx0:idx1+1], f"= ((nCC-2)*T0 > 0.0) ? (nCC-2)*T0 : 0.0;")
                
                if 'const double dtSample = ' in line:
                    idx0 = line.find("=")
                    idx1 = line.find(";")
                    line = line.replace(line[idx0:idx1+1], f"= {self.dtSample};")

                if 'const double dt0D = ' in line:
                    idx0 = line.find("=")
                    idx1 = line.find(";")
                    line = line.replace(line[idx0:idx1+1], f"= {self.dtSolver};")

                # if 'model0d->solveOneStep(' in line:
                #     if self.solver == 'CVODE':
                #         line = line.replace("solveOneStep(", "solveOneStepCVODE(")
                #     else:
                #         line = line.replace("solveOneStep(", "solveOneStepExpl(")
                
                mainScript += line # + '\n'
            
            # save main to file    
            with open(os.path.join(self.cpp_generated_models_dir, 'main0d.cpp'), 'w') as f:
                f.write(mainScript)
        
        print("Cpp files generated. Check they run properly.")
        
        return True # TODO check if they run properly


#XXX Bea: GENERATION OF 1D INPUT FILES (for my 1D Python solver)
class CVS1DPythonGenerator(object):
    '''
    Generates Python files for 1D model.
    '''

    def __init__(self, model, file_prefix, vessels1d_csv_abs_path, parameters_csv_abs_path,
                model_1d_config_path, generated_model_subdir, cpp_generated_models_dir=None,
                solver='CVODE', dtSample=1e-3, dtSolver=1e-4, conn_1d_0d_info=None):
        '''
        Constructor
        '''

        self.model = model
        self.file_prefix = file_prefix
        self.initFile1d = model_1d_config_path
        
        self.run1dFold = os.path.dirname(model_1d_config_path)
        if not os.path.exists(self.run1dFold):
            os.mkdir(self.run1dFold)
        
        self.initFiles1dFold = self.run1dFold+'/input_files'
        if not os.path.exists(self.initFiles1dFold):
            os.mkdir(self.initFiles1dFold)

        # print(self.file_prefix)
        # print(self.initFile1d)
        # print(self.run1dFold, os.path.exists(self.run1dFold))
        # print(self.initFiles1dFold, os.path.exists(self.initFiles1dFold))

        self.ODEsolver = solver
        self.dtSample = dtSample
        self.dtSolver = dtSolver
        self.conn_1d_0d_info = conn_1d_0d_info

        self.generated_model_subdir = generated_model_subdir
        if cpp_generated_models_dir is None:
            self.cpp_generated_models_dir = self.generated_model_subdir + "_cpp"
        else:
            self.cpp_generated_models_dir = cpp_generated_models_dir

        self.csv_parser = CSVFileParser()
        self.vessels_df = self.csv_parser.get_data_as_dataframe_multistrings(vessels1d_csv_abs_path, True) 
        self.params_df = self.csv_parser.get_data_as_dataframe_multistrings(parameters_csv_abs_path, True)

        self.vessFileName = self.initFiles1dFold+f'/vess_{self.file_prefix[:-3]}.txt'
        self.nodeFileName = self.initFiles1dFold+f'/nodes_{self.file_prefix[:-3]}.txt'
        self.nameFileName = self.initFiles1dFold+f'/names_{self.file_prefix[:-3]}.csv'


    def generate_files(self):
        print("Generating 1D Python files...")

        # 1: here we need to generate  
        # - vess file in self.initFiles1dFold
        # - nodes file in self.initFiles1dFold
        # - names file in self.initFiles1dFold
        vess1d, nodes1d = generate1DPythonModelFiles(self.vessels_df, self.params_df, self.vessFileName, self.nodeFileName, self.nameFileName, self.conn_1d_0d_info)

        # 2: update the input.ini file in self.run1dFold
        computeTotBV = False
        for i in range(len(self.conn_1d_0d_info)):
            if "port_volume_sum" in self.conn_1d_0d_info[str(i+1)]:
                if self.conn_1d_0d_info[str(i+1)]["port_volume_sum"]==1:
                    computeTotBV = True
        generate1DPythonSimInitFile(self.params_df, vess1d, nodes1d, self.initFile1d, self.file_prefix, self.run1dFold, self.ODEsolver, self.dtSample, computeTotBV)


        if self.file_prefix.endswith("_1d"):
            json_filename = self.file_prefix[:-3]+"_coupler1d0d.json"
        else:
            json_filename = self.file_prefix+"_coupler1d0d.json"
        with open(self.initFiles1dFold+"/"+json_filename, "w") as f:
            json.dump(self.conn_1d_0d_info, f, indent=4)

        # 3: update the main1D.py script (IF NEEDED)

        print("1D Python files generated. Check they run properly.")
        
        return True


#XXX Finbar
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
        print("Generating 1D Cpp files...")

#XXX Finbar
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
        print("Now do the coupling...")

