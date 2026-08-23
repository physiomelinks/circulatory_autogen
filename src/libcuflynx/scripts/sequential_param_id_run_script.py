'''
Created on 26/04/2022

@author: Finbar J. Argus

Staged ("sequential") calibration: fit a subset of parameters, drop the ones that turn
out to be unidentifiable, refit, repeat.

**This stage does not currently work.** The class it drives,
``libcuflynx.param_id.sequential_paramID.SequentialParamID``, is not in the tree -- the
nearest thing is the unrelated function in ``libcuflynx/obsolete/new_sequential_param_id.py``,
which is not shipped. That is long-standing (the module has been missing since well before
libcuflynx was packaged), not a packaging regression.

``cuflynx-sequential-param-id`` exists anyway, and ``--help`` says the stage is
unavailable. The alternative -- no command at all -- would mean the one documented user
route to this stage disappeared with no explanation, and whoever restores
``SequentialParamID`` would have to remember to re-declare the entry point. Instead the
import is deferred to :func:`run_sequential_param_id`, so the failure is a single sentence
naming the missing module rather than a ``ModuleNotFoundError`` traceback from inside an
MPI rank, and the day the module lands the command starts working with no packaging change.
'''

import sys
import os
import time
import numpy as np

# Not `from mpi4py import MPI`: that import initialises MPI and registers an
# atexit MPI_Finalize, and with no launcher present that finalise is what aborts
# on macOS when a NIC goes away (#396). Under mpiexec get_MPI hands back the real
# mpi4py.MPI, so a multi-rank run is unchanged.
from libcuflynx.utilities.mpi_utils import get_MPI as _get_MPI

MPI = _get_MPI()

import yaml
from libcuflynx.parsers.PrimitiveParsers import YamlFileParser
from libcuflynx.scripts import _cli

_MISSING_MESSAGE = (
    'Sequential (staged) parameter identification is not currently implemented: '
    'libcuflynx.param_id.sequential_paramID, which defines the SequentialParamID class this '
    'stage drives, is not part of libcuflynx. Use `cuflynx-param-id` (user_run_files/'
    'run_param_id.sh) for ordinary calibration. See issue #434.'
)


def _load_sequential_param_id():
    """Import ``SequentialParamID``, or explain in one line why it cannot be imported."""
    try:
        from libcuflynx.param_id.sequential_paramID import SequentialParamID
    except ImportError as exc:
        raise NotImplementedError(_MISSING_MESSAGE) from exc
    return SequentialParamID


def run_sequential_param_id(inp_data_dict=None):

    # TODO This needs to be tested for the updated user_inputs parser

    SequentialParamID = _load_sequential_param_id()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()
    if rank == 0:
        print(f'Starting sequential parameter ID with {num_procs} MPI rank(s)')

    start_time = time.time()
    yaml_parser = YamlFileParser()
    inp_data_dict = yaml_parser.parse_user_inputs_file(inp_data_dict, obs_path_needed=True,
                                                      do_generation_with_fit_parameters=True)

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
    ga_options = inp_data_dict['ga_options']
    resources_dir = inp_data_dict['resources_dir']
    param_id_output_dir = inp_data_dict['param_id_output_dir']
    plot_predictions = inp_data_dict['plot_predictions']
    do_sensitivity = inp_data_dict['do_sensitivity']
    do_uq = inp_data_dict['do_uq']
    input_params_path = inp_data_dict['input_params_path']
    num_calls_to_function = inp_data_dict['num_calls_to_function']
    UQ_options = inp_data_dict['UQ_options']

    seq_param_id = SequentialParamID(model_path, model_type, param_id_method, file_prefix,
                                     input_params_path=input_params_path,
                                     param_id_obs_path=param_id_obs_path,
                                     num_calls_to_function=num_calls_to_function,
                                     solver_info=solver_info, dt=dt, UQ_options=UQ_options,
                                     ga_options=ga_options,
                                     DEBUG=DEBUG,
                                     param_id_output_dir=param_id_output_dir,
                                     resources_dir=resources_dir)

    seq_param_id.run()

    best_param_vals = seq_param_id.param_id.get_best_param_vals()
    best_param_names = seq_param_id.get_best_param_names()

    if rank == 0:
        wall_time = time.time() - start_time
        print(f'wall time = {wall_time}')
        np.save(os.path.join(seq_param_id.param_id.output_dir, 'wall_time.npy'), wall_time)


def main(argv=None):
    """Entry point for the ``cuflynx-sequential-param-id`` command."""
    parser = _cli.build_parser(
        'Staged parameter identification: fit, drop the parameters that prove unidentifiable, '
        'refit. NOT CURRENTLY IMPLEMENTED -- ' + _MISSING_MESSAGE)
    args = parser.parse_args(argv)
    inp_data_dict = _cli.load_user_inputs(args)

    # Checked here, before any rank starts work, so that the missing implementation is one
    # line on stderr and a non-zero status -- not a traceback and an MPI_Abort from whichever
    # rank reached the import first.
    try:
        _load_sequential_param_id()
    except NotImplementedError as exc:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('ERROR: %s' % exc, file=sys.stderr)
        return 2

    return _cli.run_stage(
        lambda: run_sequential_param_id(inp_data_dict), MPI, finalize=False)


if __name__ == '__main__':
    sys.exit(main())
