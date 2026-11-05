"""Train an emulator of the model's scalar observable features (issue #333).

Run through ``user_run_files/run_emulator_training.sh <num_processors>``, or directly:

    mpiexec -n 4 cuflynx-train-emulator

Reads the same ``user_inputs.yaml`` as every other stage. The training simulations use the
solver named by ``solver:``; ``use_emulator`` is ignored here, since training always runs the
real model.

With ``emulator_settings.reuse_samples: true`` there are no simulations to run at all: the
design and features a previous training run saved beside the emulator are refitted, so a
different ``models`` or fit setting costs the fit alone. One rank is enough for that, since
only the simulations were ever spread across ranks.
"""
import os
import sys

# Not `from mpi4py import MPI`: that import initialises MPI and registers an
# atexit MPI_Finalize, and with no launcher present that finalise is what aborts
# on macOS when a NIC goes away (#396). Under mpiexec get_MPI hands back the
# real mpi4py.MPI, so a multi-rank training run is unchanged.
from libcuflynx.utilities.mpi_utils import get_MPI as _get_MPI

from libcuflynx.emulators.emulator_trainer import EmulatorTrainer, require_autoemulate
from libcuflynx.parsers.PrimitiveParsers import YamlFileParser
from libcuflynx.scripts import _cli

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


def main(argv=None):
    """Entry point for the ``cuflynx-train-emulator`` command."""
    parser = _cli.build_parser(
        "Train a surrogate (emulator) of the model's scalar observable features, so that "
        'later calibration, sensitivity or UQ runs can be driven by it instead of the solver.')
    args = parser.parse_args(argv)
    inp_data_dict = _cli.load_user_inputs(args)
    return _cli.run_stage(lambda: train_emulator(inp_data_dict), MPI)


if __name__ == '__main__':
    sys.exit(main())
