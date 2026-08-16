import sys
import os
# Not `from mpi4py import MPI`: that import initialises MPI and registers an
# atexit MPI_Finalize, and with no launcher present that finalise is what aborts
# on macOS when a NIC goes away (#396). Under mpiexec get_MPI hands back the real
# mpi4py.MPI, so a multi-rank run is unchanged.
from libcuflynx.utilities.mpi_utils import get_MPI as _get_MPI
from libcuflynx.sensitivity_analysis.sensitivityAnalysis import SensitivityAnalysis
from libcuflynx.scripts import _cli
import yaml
from libcuflynx.parsers.PrimitiveParsers import YamlFileParser

MPI = _get_MPI()

def run_SA(inp_data_dict=None):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()
    if rank == 0:
        print(f'Running sensitivity analysis with {num_procs} MPI rank(s)')

    yaml_parser = YamlFileParser()
    inp_data_dict = yaml_parser.parse_user_inputs_file(inp_data_dict, obs_path_needed=True, do_generation_with_fit_parameters=False)


    # SA_agent = SensitivityAnalysis(model_path=model_path, model_type=model_type, file_name_prefix=file_name_prefix,
    #                                DEBUG=DEBUG, model_out_names=model_out_names, solver_info=solver_info, dt=dt, 
    #                                ga_options=optimiser_options, param_id_obs_path=param_id_obs_path, params_for_id_path=params_for_id_path)
    SA_agent = SensitivityAnalysis.init_from_dict(inp_data_dict)
    if inp_data_dict.get('obs_data_dict') is not None:
        SA_agent.set_ground_truth_data(inp_data_dict['obs_data_dict'])
    if inp_data_dict.get('params_for_id') is not None:
        SA_agent.set_params_for_id(inp_data_dict['params_for_id'])
    SA_agent.run_sensitivity_analysis()

def main(argv=None):
    """Entry point for the ``cuflynx-sensitivity`` command."""
    parser = _cli.build_parser(
        'Run a Sobol sensitivity analysis of the configured observables with respect to the '
        'parameters listed for identification.')
    parser.parse_args(argv)
    return _cli.run_stage(run_SA, MPI)


if __name__ == '__main__':
    sys.exit(main())
