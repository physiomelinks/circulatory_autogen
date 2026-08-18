"""A serial run must not open MPI, and must never finalise it.

Importing mpi4py calls ``MPI_Init_thread`` and registers an ``atexit`` hook that
calls ``MPI_Finalize``. CA imported it at module scope in eighteen places, so any
import of CA opened MPI -- including a plain forward solve, in a process nothing
launched with ``mpiexec``, on a machine with no MPI installed. Closing that
process aborts on macOS (#396)::

    Abort(808576911): Fatal error in internal_Finalize
    MPIDI_OFI_handle_cq_error(593): OFI poll failed
    (default nic=en5: Input/output error)

**What these tests can and cannot do.** They cannot reproduce the abort: it needs
a NIC that fails mid-flush, and no CI runner will summon one on demand. What they
pin is the precondition -- that a serial process no longer opens MPI, and no
longer registers the finalise that aborts. Remove either and the abort becomes
possible again.

Each check runs in a fresh interpreter, because ``mpi4py.rc`` and
``sys.modules`` are process-global: importing mpi4py once to test it would
change the answer for everything after.
"""

import os
import subprocess
import sys
import textwrap

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')

from libcuflynx.utilities import mpi_utils


def _run(body, env=None):
    """Run `body` in a fresh interpreter with CA's src importable."""
    code = 'import sys; sys.path.insert(0, %r)\n' % SRC + textwrap.dedent(body)
    full_env = dict(os.environ)
    # A test runner may itself be under mpiexec; start from a clean slate so the
    # cases below control the answer.
    for var in mpi_utils.LAUNCHER_ENV_VARS:
        full_env.pop(var, None)
    full_env.update(env or {})
    proc = subprocess.run([sys.executable, '-c', code], capture_output=True,
                          text=True, timeout=300, env=full_env)
    assert proc.returncode == 0, 'stdout:\n%s\nstderr:\n%s' % (proc.stdout, proc.stderr)
    return proc.stdout


# ---------------------------------------------------------------------------
# Who counts as part of an MPI job
# ---------------------------------------------------------------------------
def test_a_bare_process_is_not_part_of_an_mpi_job():
    assert mpi_utils.launched_by_mpiexec({}) is False
    assert mpi_utils.launched_by_mpiexec({'PATH': '/usr/bin'}) is False


@pytest.mark.parametrize('var', mpi_utils.LAUNCHER_ENV_VARS)
def test_every_launcher_variable_is_recognised(var):
    """MPICH and MS-MPI set the PMI_* family, Open MPI the OMPI_* one, Hydra adds
    HYDRA_CONTROL_FD. Missing one would make a real multi-rank run look serial
    and skip a finalise that is doing actual work."""
    assert mpi_utils.launched_by_mpiexec({var: '0'}) is True


def test_rank_and_size_answer_without_opening_mpi():
    out = _run("""
        import sys
        from libcuflynx.utilities import mpi_utils
        r, s = mpi_utils.rank(), mpi_utils.size()
        print(r, s, 'mpi4py.MPI' in sys.modules)
        """)
    assert out.split() == ['0', '1', 'False']


def test_mpi_available_does_not_open_mpi():
    """It answers "is the library installed", which is what the callers mean."""
    out = _run("""
        import sys
        from libcuflynx.utilities import mpi_utils
        print(mpi_utils.mpi_available(), 'mpi4py.MPI' in sys.modules)
        """)
    installed, imported = out.split()
    assert imported == 'False'
    assert installed in ('True', 'False')


# ---------------------------------------------------------------------------
# The paths that must stay MPI-free
# ---------------------------------------------------------------------------
def test_a_forward_solve_import_never_opens_mpi():
    """The reported bug. Importing the solver wrappers is what CUFLynx does for
    live simulation, in the app's own process -- nothing there wants MPI."""
    out = _run("""
        import sys
        from libcuflynx.solver_wrappers import get_simulation_helper
        print('mpi4py.MPI' in sys.modules)
        """)
    assert out.strip() == 'False'


def test_parsing_a_config_never_opens_mpi():
    out = _run("""
        import sys
        from libcuflynx.parsers.PrimitiveParsers import CSVFileParser
        print('mpi4py.MPI' in sys.modules)
        """)
    assert out.strip() == 'False'


# ---------------------------------------------------------------------------
# The paths that need MPI must still not finalise it
# ---------------------------------------------------------------------------
def test_the_analysis_modules_never_open_mpi_serially():
    """paramID and the optimisers run real collectives -- Bcast, Scatterv,
    Gatherv -- unconditionally, even at one rank. They get those from the
    one-rank stub, so a serial calibration never initialises MPI and there is no
    MPI_Finalize for the macOS abort to happen in."""
    out = _run("""
        import sys
        from libcuflynx.param_id import paramID  # noqa: F401
        import libcuflynx.param_id.optimisers as o
        print('mpi4py.MPI' in sys.modules, o.MPI.COMM_WORLD.Get_size())
        """)
    assert out.split() == ['False', '1']


@pytest.mark.skipif(not mpi_utils.mpi_available(), reason='needs mpi4py')
def test_a_launcher_started_run_gets_the_real_mpi():
    """Under mpiexec the ranks really are talking to each other; nothing about a
    multi-rank run may change."""
    out = _run("""
        import sys
        import libcuflynx.param_id.optimisers as o
        print('mpi4py.MPI' in sys.modules, type(o.MPI).__name__)
        """, env={'PMI_RANK': '0', 'PMI_SIZE': '2'})
    assert out.split() == ['True', 'module']


@pytest.mark.skipif(not mpi_utils.mpi_available(), reason='needs mpi4py')
def test_the_real_mpi_wins_if_something_else_already_imported_it():
    """MPI is initialised either way at that point, so handing back a stub would
    make this module the odd one out."""
    out = _run("""
        from mpi4py import MPI  # noqa: F401
        from libcuflynx.utilities.mpi_utils import get_MPI
        print(type(get_MPI()).__name__)
        """)
    assert out.strip() == 'module'


# ---------------------------------------------------------------------------
# Imported is not initialised
#
# `MPI4PY_RC_INITIALIZE=0` (or `mpi4py.rc.initialize = False`) loads mpi4py and
# skips MPI_Init, so `'mpi4py.MPI' in sys.modules` can be true with MPI never
# opened. Every routine other than Is_initialized/Is_finalized is then erroneous,
# and MPICH and Microsoft MPI both answer by printing
#
#     Attempting to use an MPI routine before initializing MPI
#
# and killing the process -- not raising, so no try/except can survive it.
#
# That state is not hypothetical: CUFLynx's release build sets the variable while
# PyInstaller analyses the bundle (MPI_Init aborts in the Linux runners' UCX). All
# collected packages are imported into ONE isolated child; mpi4py.futures comes
# first and loads MPI uninitialised, and the child then died importing
# libcuflynx.solver_wrappers, whose chain reaches PrimitiveParsers' module-scope
# `rank = mpi_utils.rank()`. It failed the v0.4.0 Windows release build twice.
# ---------------------------------------------------------------------------

def _mpi_stub(initialised):
    """Source that puts a stand-in ``mpi4py.MPI`` into ``sys.modules``.

    A real uninitialised MPI cannot be used here: the process death is the thing
    under test, so it has to be observable rather than fatal to the test run. The
    stub reproduces the two behaviours that matter -- what ``Is_initialized``
    answers, and that any other routine in that state kills the process rather
    than raising.
    """
    return textwrap.dedent("""
        import importlib.machinery
        import os
        import sys
        import types

        _mpi4py = types.ModuleType('mpi4py')
        _mpi4py.__spec__ = importlib.machinery.ModuleSpec('mpi4py', None)
        _mpi4py.__path__ = []
        _MPI = types.ModuleType('mpi4py.MPI')
        _MPI.__spec__ = importlib.machinery.ModuleSpec('mpi4py.MPI', None)
        _INITIALISED = %r


        def _routine(name, answer):
            def call(*args, **kwargs):
                if not _INITIALISED:
                    print('Attempting to use an MPI routine before initializing MPI')
                    os._exit(1)
                return answer
            return staticmethod(call)


        class _Comm(object):
            Get_rank = _routine('MPI_Comm_rank', 3)
            Get_size = _routine('MPI_Comm_size', 8)


        _MPI.COMM_WORLD = _Comm()
        _MPI.Is_initialized = staticmethod(lambda: _INITIALISED)
        _MPI.Is_finalized = staticmethod(lambda: False)
        _mpi4py.MPI = _MPI
        sys.modules['mpi4py'] = _mpi4py
        sys.modules['mpi4py.MPI'] = _MPI
        """ % initialised)


def test_is_live_distinguishes_imported_from_initialised():
    """The two routines it asks with are the two the standard allows to be asked."""
    class _Uninitialised(object):
        Is_initialized = staticmethod(lambda: False)
        Is_finalized = staticmethod(lambda: False)

    class _Open(object):
        Is_initialized = staticmethod(lambda: True)
        Is_finalized = staticmethod(lambda: False)

    class _Closed(object):
        Is_initialized = staticmethod(lambda: True)
        Is_finalized = staticmethod(lambda: True)

    assert mpi_utils.mpi_is_live(_Uninitialised) is False
    assert mpi_utils.mpi_is_live(_Open) is True
    assert mpi_utils.mpi_is_live(_Closed) is False
    # An mpi4py too old to answer, or anything else that raises, is not a reason to
    # gamble a process abort on the answer.
    assert mpi_utils.mpi_is_live(object()) is False


def test_rank_does_not_call_into_an_uninitialised_mpi():
    out = _run(_mpi_stub(False) + textwrap.dedent("""
        from libcuflynx.utilities import mpi_utils
        print(mpi_utils.rank(), mpi_utils.size(),
              mpi_utils.get_MPI() is mpi_utils._SerialMPI)
        """))
    assert out.split() == ['0', '1', 'True']


def test_importing_the_solver_wrappers_survives_an_uninitialised_mpi():
    """The failure verbatim: the isolated PyInstaller child imports mpi4py first,
    then this, and the module-scope rank read in PrimitiveParsers killed it."""
    out = _run(_mpi_stub(False) + textwrap.dedent("""
        from libcuflynx.solver_wrappers import get_simulation_helper  # noqa: F401
        from libcuflynx.parsers.PrimitiveParsers import rank
        print('rank', rank)
        """))
    assert out.split() == ['rank', '0']


def test_an_initialised_mpi_is_still_the_one_that_answers():
    """The guard must cost nothing when MPI is genuinely open -- a process that
    initialised MPI is not rank 0 of 1 just because it never used mpiexec."""
    out = _run(_mpi_stub(True) + textwrap.dedent("""
        import sys
        from libcuflynx.utilities import mpi_utils
        print(mpi_utils.rank(), mpi_utils.size(),
              mpi_utils.get_MPI() is sys.modules['mpi4py.MPI'])
        """))
    assert out.split() == ['3', '8', 'True']


# ---------------------------------------------------------------------------
# No module may reintroduce the import
# ---------------------------------------------------------------------------
def test_no_module_imports_mpi4py_at_module_scope():
    """A single missed line undoes the whole fix, and does it silently: a module
    that assigns ``MPI = get_MPI()`` and *also* imports mpi4py has both the stub
    and the abort, with the later line winning. That is exactly what was left
    behind in sensitivity_analysis_run_script.py. Nothing here is subtle enough
    to warrant catching by hand on review.

    Only ``obsolete/`` is exempt, because it is not on any run path.
    ``generate_omex_analysis_script.py`` used to be too -- it matched inside the
    *text of a script it generates* -- but #435 made that generated script take
    its MPI from ``get_MPI()`` as well, after the sys.path bootstrap that makes
    libcuflynx importable. So the scan now covers generated scripts, and the last
    place mpi4py could be a hard requirement is gone.
    """
    offenders = []
    for dirpath, dirnames, filenames in os.walk(SRC):
        dirnames[:] = [d for d in dirnames if d not in ('obsolete', '__pycache__')]
        for name in filenames:
            if not name.endswith('.py'):
                continue
            path = os.path.join(dirpath, name)
            with open(path, encoding='utf-8') as handle:
                for lineno, line in enumerate(handle, 1):
                    # Module scope only: an import nested in a function has
                    # already decided it needs MPI at the moment it runs.
                    if line.startswith(('from mpi4py import', 'import mpi4py')):
                        offenders.append('%s:%d' % (os.path.relpath(path, SRC), lineno))
    assert offenders == [], (
        'these import mpi4py at module scope, which initialises MPI and registers '
        'the atexit MPI_Finalize that aborts on macOS; use '
        'libcuflynx.utilities.mpi_utils.get_MPI() instead: ' + ', '.join(offenders))


# ---------------------------------------------------------------------------
# The one-rank collectives
#
# Each is checked against what real MPI does at one rank, which is the whole
# claim: a broadcast from yourself changes nothing, a gather of your own value
# is a list of one, a scatter to yourself is a copy.
# ---------------------------------------------------------------------------
def test_rank_and_size():
    comm = mpi_utils._SerialMPI.COMM_WORLD
    assert (comm.Get_rank(), comm.Get_size()) == (0, 1)


def test_broadcast_leaves_the_buffer_alone():
    import numpy as np

    comm = mpi_utils._SerialMPI.COMM_WORLD
    buf = np.array([1.0, 2.0, 3.0])
    comm.Bcast(buf, root=0)
    assert list(buf) == [1.0, 2.0, 3.0]
    assert comm.bcast({'a': 1}, root=0) == {'a': 1}


def test_gathering_your_own_value_gives_a_list_of_one():
    comm = mpi_utils._SerialMPI.COMM_WORLD
    assert comm.gather(7, root=0) == [7]
    assert comm.allgather({'x': 1}) == [{'x': 1}]


def test_scatterv_reshapes_the_whole_population_into_the_receive_buffer():
    """The shape change is the part worth pinning: CA scatters a flat population
    into an (n, num_params) buffer, so a naive `dest[:] = src` would raise."""
    import numpy as np

    comm = mpi_utils._SerialMPI.COMM_WORLD
    send = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    recv = np.zeros((3, 2))
    comm.Scatterv([send, np.array([6]), None, mpi_utils._SerialMPI.DOUBLE], recv, root=0)
    assert recv.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


def test_gatherv_collects_the_one_rank_contribution():
    import numpy as np

    comm = mpi_utils._SerialMPI.COMM_WORLD
    send = np.array([[1.0, 2.0], [3.0, 4.0]])
    recv = np.zeros(4)
    comm.Gatherv(send, [recv, np.array([4]), None, mpi_utils._SerialMPI.DOUBLE], root=0)
    assert recv.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_allreduce_over_one_rank_is_the_value_itself():
    import numpy as np

    comm = mpi_utils._SerialMPI.COMM_WORLD
    send, recv = np.array([3.0, 1.0]), np.zeros(2)
    comm.Allreduce(send, recv, op=mpi_utils._SerialMPI.MIN)
    assert recv.tolist() == [3.0, 1.0]


def test_there_is_never_a_message_waiting():
    comm = mpi_utils._SerialMPI.COMM_WORLD
    assert comm.iprobe(source=mpi_utils._SerialMPI.ANY_SOURCE, tag=1) is False


def test_waitall_on_nothing_is_fine_and_on_something_is_not():
    """At one rank the send list is always empty -- the loop that fills it skips
    `other == rank`. A non-empty list would mean a message was sent with nobody
    to send it to, which is worth failing loudly rather than ignoring."""
    mpi_utils._SerialMPI.Request.Waitall([])
    with pytest.raises(RuntimeError, match='nothing can have been sent'):
        mpi_utils._SerialMPI.Request.Waitall(['pending'])


def test_point_to_point_refuses_rather_than_deadlocking():
    """`recv` at one rank would block forever; saying so beats hanging a run."""
    comm = mpi_utils._SerialMPI.COMM_WORLD
    with pytest.raises(RuntimeError, match='block forever'):
        comm.recv(source=0, tag=1)
    with pytest.raises(RuntimeError, match='no'):
        comm.isend(1, dest=1, tag=1)


# ---------------------------------------------------------------------------
# Ending the job
#
# The stage entry points (libcuflynx/scripts/_cli.py) end with `MPI.Finalize()`
# and, on failure, `comm.Abort()`. Both used to be missing from the stub, so a
# one-rank `cuflynx-param-id` raised AttributeError *after* finishing its work.
# ---------------------------------------------------------------------------
def test_finalizing_a_job_that_never_started_is_a_no_op():
    assert mpi_utils._SerialMPI.Finalize() is None


def test_abort_does_not_return_and_exits_non_zero():
    """`comm.Abort()` is written at every call site as the last thing that happens.

    Real MPI_Abort never returns, so the lines after it are unreachable; a stub that
    returned would let them run and a failed serial run exit 0.
    """
    comm = mpi_utils._SerialMPI.COMM_WORLD
    with pytest.raises(SystemExit) as excinfo:
        comm.Abort()
    assert excinfo.value.code == 1

    with pytest.raises(SystemExit) as excinfo:
        comm.Abort(3)
    assert excinfo.value.code == 3
