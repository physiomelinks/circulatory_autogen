"""Shared plumbing for the console entry points declared in ``[project.scripts]``.

Each stage script in this package exposes a ``main()`` that setuptools wraps into a
command (``cuflynx-param-id``, ``cuflynx-generate``, ...). Those commands are what
``user_run_files/*.sh`` invoke, so the shell scripts no longer have to know where the
package was installed.

Two things every one of them needs, defined once here:

* a parser, so that ``--help`` answers before any configuration file is opened. That
  matters more than it looks: ``--help`` has to work from a wheel install with no
  repository present, and every stage's real work begins by reading
  ``user_run_files/user_inputs.yaml``.
* the failure contract the ``if __name__ == '__main__'`` blocks have always had --
  print the traceback, then ``comm.Abort()``. The abort is not decoration: one rank
  raising while the others sit in a collective leaves the job hanging until it is
  killed by hand.
"""
import argparse
import os
import traceback

import yaml

#: Every stage takes its configuration from the same file, and none of them take
#: options. Saying so in ``--help`` is the whole of the help text people need.
CONFIG_EPILOG = (
    "Configuration is read from the file named by --user-inputs; otherwise from\n"
    "user_run_files/user_inputs.yaml under the user directory, or from the file named by\n"
    "user_inputs_path_override in it.\n"
    "\n"
    "The user directory is $CUFLYNX_USER_DIR if set; otherwise the circulatory_autogen\n"
    "checkout this libcuflynx was run from, if it is one; otherwise the current directory.\n"
    "Inputs (resources/, module_config_user/, funcs_user/) and outputs (generated_models/,\n"
    "param_id_output/) all default under it. After `pip install libcuflynx` there is no\n"
    "checkout, so set CUFLYNX_USER_DIR or run from your working directory.\n"
    "\n"
    "Run under a launcher to use more than one rank, e.g. `mpiexec -n 4 <command>`."
)


def build_parser(description, epilog=CONFIG_EPILOG):
    """An ``ArgumentParser`` for a stage command.

    ``prog`` is left to argparse, which takes it from ``sys.argv[0]`` -- so the help
    text names whichever way the user reached the script: the console command, or
    ``python -m libcuflynx.scripts...``.
    """
    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # The one option every stage shares. Without it a run is configured by editing
    # user_inputs.yaml in place, setting user_inputs_path_override inside it, or moving
    # CUFLYNX_USER_DIR -- none of which lets one checkout hold several configurations
    # (a training run and a per-dataset run, say) and choose between them per command.
    parser.add_argument(
        '--user-inputs', dest='user_inputs', metavar='PATH', default=None,
        help='read the configuration from PATH instead of the default '
             'user_run_files/user_inputs.yaml')
    return parser


def load_user_inputs(args):
    """The configuration named by ``--user-inputs``, or None for the default file.

    None is what every stage's ``inp_data_dict`` parameter already means: "go and read
    the configured file yourself". So a stage passes this straight through and behaves
    exactly as before when the option is absent.

    Every rank reads the file. That matches how the default file is already loaded, and
    a broadcast would only move the same failure to a later collective.
    """
    path = getattr(args, 'user_inputs', None)
    if not path:
        return None
    if not os.path.isfile(path):
        raise SystemExit("user inputs file not found: %s" % path)
    with open(path, 'r') as file:
        inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)
    if not isinstance(inp_data_dict, dict):
        raise SystemExit(
            "user inputs file %s did not parse to a mapping of settings" % path)
    return inp_data_dict


_TRUE = frozenset(('true', 't', 'yes', 'y', '1'))
_FALSE = frozenset(('false', 'f', 'no', 'n', '0'))


def boolean(value):
    """Parse the ``True``/``False`` words the shell scripts pass positionally.

    Replaces ``distutils.util.strtobool``, which the scripts used to call: distutils
    was removed from the standard library in Python 3.12, so importing it is a
    countdown rather than a dependency.
    """
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise argparse.ArgumentTypeError(
        "expected True or False, got %r" % (value,))


def run_stage(stage, MPI, finalize=True):
    """Run one pipeline stage and return the process exit status.

    Args:
        stage: zero-argument callable doing the work.
        MPI: the module :func:`libcuflynx.utilities.mpi_utils.get_MPI` handed back --
            real ``mpi4py.MPI`` under a launcher, the one-rank stub otherwise.
        finalize: call ``MPI.Finalize()`` after a successful run. False for the two
            stages whose ``__main__`` block never did -- plotting and sequential
            param id -- so that their behaviour is unchanged.

    Unlike the blocks this replaces, ``MPI.Finalize()`` sits outside the ``try``: a
    failure to shut MPI down after the work succeeded is not a reason to ``Abort`` and
    claim the run failed.
    """
    comm = MPI.COMM_WORLD
    try:
        stage()
    except BaseException:  # noqa: E722 - the historical bare `except:`, kept deliberately
        print(traceback.format_exc())
        # Does not return: real MPI_Abort tears the whole job down, and the one-rank
        # stub raises SystemExit(1). The status below is for an implementation that
        # somehow does return -- exiting 0 after a traceback would be the worst outcome.
        comm.Abort()
        return 1
    if finalize:
        MPI.Finalize()
    return 0
