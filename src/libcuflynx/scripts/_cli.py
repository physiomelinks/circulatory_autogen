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
import traceback

#: Every stage takes its configuration from the same file, and none of them take
#: options. Saying so in ``--help`` is the whole of the help text people need.
CONFIG_EPILOG = (
    "Configuration is read from user_run_files/user_inputs.yaml in the repository this\n"
    "libcuflynx was installed from, or from the file named by user_inputs_path_override in it.\n"
    "Run under a launcher to use more than one rank, e.g. `mpiexec -n 4 <command>`."
)


def build_parser(description, epilog=CONFIG_EPILOG):
    """An ``ArgumentParser`` for a stage command.

    ``prog`` is left to argparse, which takes it from ``sys.argv[0]`` -- so the help
    text names whichever way the user reached the script: the console command, or
    ``python -m libcuflynx.scripts...``.
    """
    return argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


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
