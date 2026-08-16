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
# No module may reintroduce the import
# ---------------------------------------------------------------------------
def test_no_module_imports_mpi4py_at_module_scope():
    """A single missed line undoes the whole fix, and does it silently: a module
    that assigns ``MPI = get_MPI()`` and *also* imports mpi4py has both the stub
    and the abort, with the later line winning. That is exactly what was left
    behind in sensitivity_analysis_run_script.py. Nothing here is subtle enough
    to warrant catching by hand on review.

    Two files are exempt. ``obsolete/`` is not on any run path, and
    ``generate_omex_analysis_script.py`` matches inside the *text of a script it
    generates* -- guarding generated scripts needs their own bootstrap order
    verified and belongs in its own change.
    """
    exempt = {'generate_omex_analysis_script.py'}
    offenders = []
    for dirpath, dirnames, filenames in os.walk(SRC):
        dirnames[:] = [d for d in dirnames if d not in ('obsolete', '__pycache__')]
        for name in filenames:
            if not name.endswith('.py') or name in exempt:
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
