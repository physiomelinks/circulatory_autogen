"""Answer "which rank am I?" without opening MPI when there is only one.

Importing ``mpi4py`` calls ``MPI_Init_thread`` and registers an ``atexit`` hook
that calls ``MPI_Finalize``. Because circulatory_autogen imported it at module
scope in eighteen places, *any* import of CA opened MPI -- including a plain
forward solve, in a process nothing launched with ``mpiexec``, on a machine with
no MPI installed.

That is not merely wasteful. It aborts. Reported by a CUFLynx user closing the
app (#396)::

    Abort(808576911): Fatal error in internal_Finalize: Other MPI error
    MPIDI_OFI_mpi_finalize_hook(861):
    flush_send_queue(830)...........:
    MPIDI_OFI_handle_cq_error(593)..: OFI poll failed
    (default nic=en5: Input/output error)

MPI had opened a transport over a real NIC, and closing it failed because the
NIC had gone away -- a Wi-Fi drop, a VPN, a virtual adapter disappearing. The
user's summary was exactly right: if MPI is not in use, none of this should be
happening.

Two things here, and the distinction matters:

:func:`rank` / :func:`size` / :func:`is_root` answer the common question --
"which rank am I?" -- **without importing mpi4py at all** when no launcher
started this process. Code that only needs those (the solver wrappers, the
parsers) can then run entirely MPI-free.

:func:`configure_mpi4py` is for the code that genuinely does collectives
(``Bcast``, ``Scatterv``, ``Gatherv``) and therefore must have real MPI objects.
It cannot avoid initialising MPI, so it makes the *exit* safe instead: with no
launcher present it clears ``mpi4py.rc.finalize`` before the first import, so no
``MPI_Finalize`` is ever called. Skipping the finalise in a process that is
exiting is benign -- the OS reclaims the sockets -- and it is strictly safer than
aborting in it. Under ``mpiexec`` the ranks really are talking to each other and
Finalize does real work, so nothing is changed there.

:func:`get_MPI` closes the loop for the code that *does* run collectives. CA calls
``Bcast``/``Scatterv``/``Gatherv`` unconditionally, even at one rank, so those
modules could not simply skip MPI. At one rank every collective has a trivial
definition -- a broadcast from yourself is a no-op, a gather of one value is a
list of one -- so :class:`_SerialComm` supplies them and MPI is never opened.
Under a launcher ``get_MPI`` hands back the real ``mpi4py.MPI`` and nothing about
a multi-rank run changes.
"""

import os

#: Environment variables an MPI launcher sets in every rank it spawns. MPICH and
#: Microsoft MPI use the ``PMI_*`` family, Open MPI the ``OMPI_*`` one; Hydra
#: (MPICH's launcher) adds ``HYDRA_CONTROL_FD``.
#:
#: Probed from the environment rather than asked of MPI, because the answer is
#: needed *before* mpi4py is imported -- once it has been, MPI is initialised and
#: the atexit hook is registered, and the decision has already been made.
LAUNCHER_ENV_VARS = (
    'PMI_RANK',
    'PMI_SIZE',
    'PMI_FD',
    'HYDRA_CONTROL_FD',
    'OMPI_COMM_WORLD_RANK',
    'OMPI_COMM_WORLD_SIZE',
    'MPI_LOCALNRANKS',
)


def launched_by_mpiexec(env=None):
    """Whether an MPI launcher spawned this process."""
    source = os.environ if env is None else env
    return any(var in source for var in LAUNCHER_ENV_VARS)


def configure_mpi4py(env=None):
    """Make the exit safe before anything imports ``mpi4py``.

    Call at the top of a module that will import mpi4py for collectives. With no
    launcher present this clears ``rc.finalize``, so the process never calls
    ``MPI_Finalize`` -- which is where the macOS abort happens. Returns whether
    the guard was applied.

    A no-op once mpi4py has been imported, and harmless when it is absent.
    """
    if launched_by_mpiexec(env):
        return False
    try:
        import mpi4py
    except ImportError:
        return False
    mpi4py.rc.finalize = False
    return True


def _comm_or_none():
    """``MPI.COMM_WORLD`` if MPI is already in play, else None.

    Deliberately does not *cause* an import: it uses mpi4py only when a launcher
    started this process, or when something else has already imported it (in
    which case MPI is initialised anyway and asking costs nothing).
    """
    import sys

    if not launched_by_mpiexec() and 'mpi4py.MPI' not in sys.modules:
        return None
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD
    except Exception:
        return None


def rank():
    """This process's MPI rank, or 0 when it is not part of an MPI job."""
    comm = _comm_or_none()
    if comm is None:
        return 0
    try:
        return comm.Get_rank()
    except Exception:
        return 0


def size():
    """The number of ranks, or 1 when this is not part of an MPI job."""
    comm = _comm_or_none()
    if comm is None:
        return 1
    try:
        return comm.Get_size()
    except Exception:
        return 1


def is_root():
    """Whether this process should do the work exactly one rank should do."""
    return rank() == 0


def mpi_available():
    """Whether mpi4py can be imported at all.

    Answers "is the library installed", which is what the callers using this
    flag mean; it does not initialise MPI to find out.
    """
    import importlib.util

    return importlib.util.find_spec('mpi4py') is not None


# ---------------------------------------------------------------------------
# One rank, no MPI
# ---------------------------------------------------------------------------
class _SerialRequest(object):
    """Stand-in for ``MPI.Request``.

    Only ``Waitall`` is used, and only on the list of non-blocking sends this
    rank issued to *other* ranks. At one rank that list is always empty -- the
    loop that fills it is ``for other in range(num_procs): if other != rank``,
    which has no iterations. A non-empty list here would mean the caller sent a
    message with nobody to send it to, so it is an error rather than a no-op.
    """

    @staticmethod
    def Waitall(requests):
        if requests:
            raise RuntimeError(
                'serial MPI stub: Waitall got %d pending request(s); at one rank '
                'nothing can have been sent' % len(requests))


class _SerialComm(object):
    """``MPI.COMM_WORLD`` for a process that is the whole job.

    Every collective has a trivial one-rank definition: a broadcast from
    yourself leaves the buffer alone, a gather of your own value is a list of
    one, a scatter to yourself is a copy. Implemented rather than skipped
    because circulatory_autogen calls them unconditionally -- guarding every
    call site on ``size > 1`` would be a far larger and more error-prone change
    than defining the one-rank case once, here.
    """

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def Barrier(self):
        return None

    def barrier(self):
        return None

    def bcast(self, obj, root=0):
        return obj

    def Bcast(self, buf, root=0):
        # The root is this process, so the buffer already holds the value.
        return None

    def gather(self, obj, root=0):
        return [obj]

    def allgather(self, obj):
        return [obj]

    def Gatherv(self, sendbuf, recvbuf, root=0):
        """``Gatherv(send, [recv, counts, displs, datatype], root)``."""
        _copy_into(_buffer_of(recvbuf), _buffer_of(sendbuf))
        return None

    def Scatterv(self, sendbuf, recvbuf, root=0):
        """``Scatterv([send, counts, displs, datatype], recv, root)``."""
        _copy_into(_buffer_of(recvbuf), _buffer_of(sendbuf))
        return None

    def Allreduce(self, sendbuf, recvbuf, op=None):
        # Reducing one value over one rank is that value.
        _copy_into(_buffer_of(recvbuf), _buffer_of(sendbuf))
        return None

    def iprobe(self, source=None, tag=None):
        # Nobody else exists, so there is never a message waiting.
        return False

    def recv(self, source=None, tag=None):
        raise RuntimeError(
            'serial MPI stub: recv() with one rank would block forever; the '
            'caller should have found nothing to receive')

    def isend(self, obj, dest=None, tag=None):
        raise RuntimeError(
            'serial MPI stub: isend() to rank %r with one rank has no '
            'destination' % (dest,))


def _buffer_of(spec):
    """The array out of a bare buffer or an ``[buf, counts, displs, dtype]`` spec."""
    if isinstance(spec, (list, tuple)):
        return spec[0]
    return spec


def _copy_into(dest, src):
    """``dest[:] = src``, flattened, tolerating the shape differences CA relies on.

    The send and receive buffers of a one-rank Scatterv/Gatherv hold the same
    values but need not have the same shape -- circulatory_autogen scatters a
    flat population into a ``(n, num_params)`` receive buffer, for instance.
    """
    import numpy as np

    dest_arr = np.asarray(dest)
    src_flat = np.asarray(src).reshape(-1)
    n = min(dest_arr.size, src_flat.size)
    dest_arr.reshape(-1)[:n] = src_flat[:n]


class _SerialMPI(object):
    """``mpi4py.MPI`` for a process that is the whole job.

    The datatype and operation constants exist only so call sites that name them
    keep working; the one-rank collectives above never inspect them.
    """

    COMM_WORLD = _SerialComm()
    Request = _SerialRequest

    DOUBLE = 'double'
    FLOAT = 'float'
    INT = 'int'
    LONG = 'long'
    BOOL = 'bool'
    C_BOOL = 'bool'

    SUM = 'sum'
    MIN = 'min'
    MAX = 'max'
    PROD = 'prod'
    LAND = 'land'
    LOR = 'lor'

    ANY_SOURCE = -1
    ANY_TAG = -1


def get_MPI(env=None):
    """The ``MPI`` module to use: the real one under a launcher, else the stub.

    Import this instead of ``from mpi4py import MPI`` in modules that run
    collectives. When no launcher started the process, nothing here imports
    mpi4py, so MPI is never initialised and there is no ``MPI_Finalize`` to
    abort in.

    mpi4py already being imported means MPI is initialised regardless, so the
    real module is handed back -- a stub would then be the odd one out.
    """
    import sys

    if launched_by_mpiexec(env) or 'mpi4py.MPI' in sys.modules:
        from mpi4py import MPI
        return MPI
    return _SerialMPI
