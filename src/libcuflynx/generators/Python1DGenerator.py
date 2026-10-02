'''
Input files for the Python FV 1D solver (libcuflynx/solver1d), used when a cpp model is coupled
to 1D vessels (couple_to_1d with solver_1d_type: py).

Moved out of CVSCppGenerator.py unchanged.
'''

import json
import os

from libcuflynx.parsers.PrimitiveParsers import CSVFileParser
from libcuflynx.utilities.config_schemas import load_vessel_array
from libcuflynx.generators.Python1DModelFilesGenerator import generate1DPythonModelFiles, generate1DPythonSimInitFile


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
            os.makedirs(self.run1dFold)
        
        self.initFiles1dFold = self.run1dFold+'/input_files'
        if not os.path.exists(self.initFiles1dFold):
            os.makedirs(self.initFiles1dFold)

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
        # the 1D part of an already supermodule-expanded array (split_0d_1d_vessel_array)
        self.vessels_df, _ = load_vessel_array(vessels1d_csv_abs_path)
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


