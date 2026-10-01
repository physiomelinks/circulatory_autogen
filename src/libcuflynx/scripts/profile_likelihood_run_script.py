"""  
Script to run profile likelihood analysis independently.  
"""  

import sys  
import os  
from libcuflynx.utilities.mpi_utils import get_MPI as _get_MPI  
MPI = _get_MPI()
  
root_dir = os.path.join(os.path.dirname(__file__), '../..')  
sys.path.append(os.path.join(root_dir, 'src'))  
  
from param_id.paramID import CVS0DParamID  
from identifiabilty_analysis.identifiabilityAnalysis import IdentifiabilityAnalysis  
from parsers.PrimitiveParsers import YamlFileParser, CSVFileParser  
import numpy as np  

def run_profile_likelihood(inp_data_dict=None):  
    """Run profile likelihood analysis."""  
      
    yaml_parser = YamlFileParser()  
    inp_data_dict = yaml_parser.parse_user_inputs_file(  
        inp_data_dict, obs_path_needed=True,   
        do_generation_with_fit_parameters=True  
    )  
      
    comm = MPI.COMM_WORLD  
    rank = comm.Get_rank()  
      
    if rank == 0:  
        print(f"Starting profile likelihood analysis with {comm.Get_size()} MPI rank(s)")  
      
    # Initialize parameter identification  
    param_id = CVS0DParamID(  
        model_path=inp_data_dict['model_path'],  
        model_type=inp_data_dict['model_type'],  
        param_id_method=inp_data_dict['param_id_method'],  
        mcmc_instead=False,  
        file_name_prefix=inp_data_dict['file_prefix'],  
        params_for_id_path=inp_data_dict['params_for_id_path'],  
        param_id_obs_path=inp_data_dict['param_id_obs_path'],  
        sim_time=inp_data_dict['sim_time'],  
        pre_time=inp_data_dict['pre_time'],  
        solver_info=inp_data_dict['solver_info'],  
        dt=inp_data_dict['dt'],  
        optimiser_options=inp_data_dict['optimiser_options'],  
        DEBUG=inp_data_dict['DEBUG'],  
        param_id_output_dir=inp_data_dict['param_id_output_dir'],  
        resources_dir=inp_data_dict['resources_dir']  
    )  
      
    # Load best parameter values  
    csv_parser = CSVFileParser()  
    param_id_name_and_vals, _ = csv_parser.get_param_id_params_as_lists_of_tuples(  
        inp_data_dict['param_id_output_dir_abs_path']  
    )  
    best_param_vals = np.array([val for name, val in param_id_name_and_vals])  
      
    # Run identifiability analysis with profile likelihood  
    id_analysis = IdentifiabilityAnalysis(  
        model_path=inp_data_dict['model_path'],  
        model_type=inp_data_dict['model_type'],  
        file_name_prefix=inp_data_dict['file_prefix'],  
        param_id_output_dir=inp_data_dict['param_id_output_dir'],  
        resources_dir=inp_data_dict['resources_dir'],  
        param_id=param_id.param_id  
    )  
      
    id_analysis.set_best_param_vals(best_param_vals)  
    id_analysis.run(inp_data_dict['ia_options'])  
      
    if rank == 0:  
        print("Profile likelihood analysis complete")  
  
if __name__ == '__main__':  
    comm = MPI.COMM_WORLD  
    try:  
        run_profile_likelihood()  
        MPI.Finalize()  
    except Exception as e:  
        print(f"Error: {e}")  
        comm.Abort()  
        MPI.Finalize()  
        exit()