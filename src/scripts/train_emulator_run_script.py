"""Train an emulator of the model's scalar observable features (issue #333).

Run through ``user_run_files/run_emulator_training.sh <num_processors>``, or directly:

    mpiexec -n 4 $python_path src/scripts/train_emulator_run_script.py

Reads the same ``user_inputs.yaml`` as every other stage. The training simulations use the
solver named by ``solver:``; ``use_emulator`` is ignored here, since training always runs the
real model.
"""
import os
import sys
import traceback

root_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir, 'src'))

# Not `from mpi4py import MPI`: that import initialises MPI and registers an
# atexit MPI_Finalize, and with no launcher present that finalise is what aborts
# on macOS when a NIC goes away (#396). Placed after the sys.path bootstrap
# above, which is what makes `utilities` importable. Under mpiexec get_MPI hands
# back the real mpi4py.MPI, so a multi-rank training run is unchanged.
from utilities.mpi_utils import get_MPI as _get_MPI

from emulators.emulator_trainer import EmulatorTrainer, require_autoemulate
from parsers.PrimitiveParsers import YamlFileParser

MPI = _get_MPI()


def train_emulator(inp_data_dict=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Checked before anything is generated or simulated: without autoemulate the training runs
    # would all complete and then have nothing to fit.
    require_autoemulate()

    yaml_parser = YamlFileParser()
    inp_data_dict = yaml_parser.parse_user_inputs_file(
        inp_data_dict, obs_path_needed=True, do_generation_with_fit_parameters=False)

    if rank == 0:
        print(f'Training an emulator with {comm.Get_size()} MPI rank(s), against solver '
              f'{inp_data_dict["solver_info"].get("solver")}')

    trainer = EmulatorTrainer.init_from_dict(inp_data_dict, comm=comm)
    bundle = trainer.train()
    if rank == 0 and bundle is not None:
        print('Set use_emulator: true in user_inputs.yaml to run calibration, sensitivity '
              'analysis, UQ or identifiability analysis against it.')
    return bundle


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    try:
        train_emulator()
        MPI.Finalize()
    except Exception:
        print(traceback.format_exc())
        comm.Abort()
        MPI.Finalize()
