"""Collection-time availability probes that are safe under ``mpiexec``.

Importing pymc reaches arviz, whose module body writes a stamp file through a fixed temp name;
ranks importing it at once race for it. Test modules call ``_pymc_available()`` instead of
``import pymc`` at module scope, so only rank 0 imports and every rank collects the same set.
"""
try:
    from mpi4py import MPI
except ImportError:                                            # pragma: no cover
    MPI = None


def _decided_on_rank_0(probe):
    """``probe()``'s answer, taken on rank 0 and broadcast to the rest.

    Collection-time skips have the same failure mode as the assertions ``_agree`` exists for,
    and it is worse: a rank that skips a test its peers are running never enters their
    collectives at all, so the run does not fail, it **hangs** -- and the junit that survives
    shows the ranks that finished, with no sign of the one that did not.

    That is not hypothetical. ``_emulator_available`` imports
    ``libcuflynx.emulators.emulator_trainer`` on every rank at once and answers any failure with
    ``False``. Concurrent ranks importing the same module can see it half-built (Python puts a
    module in ``sys.modules`` before executing its body), and the loser silently skipped while
    the winner ran -- a 30-minute timeout with one junit file, roughly one run in three.

    Probing on rank 0 alone fixes both halves: every rank collects the same set, *and* the
    concurrent import that caused the disagreement does not happen. The broadcast also orders
    the ranks, exactly as it does in ``_agree``.
    """
    if MPI is None:
        return probe()
    comm = MPI.COMM_WORLD
    return comm.bcast(probe() if comm.Get_rank() == 0 else None, root=0)


def _pymc_available():
    def probe():
        try:
            import pymc  # noqa: F401

            return True
        except Exception:  # noqa: BLE001 - see below; ImportError alone is not enough
            # `except ImportError` was what let this take the job down. Importing pymc reaches
            # arviz, whose module body runs `_warn_once_per_day()` -- and that writes its stamp
            # through a **fixed** temp name (`~/.cache/arviz/daily_warning.tmp`). Two ranks
            # importing at once race for it, and the loser gets **FileNotFoundError**, which is
            # not an ImportError and so escaped this guard, ending collection on that rank while
            # its peer waited in a collective. 30-minute timeout, no test output, once every few
            # runs -- and only ever on the first run after the cache is cold, which is why it
            # looked random.
            #
            # Probing on rank 0 alone (below) is the actual fix: the concurrent import does not
            # happen. This stays widened anyway, because "pymc is unavailable" is the honest
            # answer to *any* failure to import it, and a collection error is never a better one.
            return False

    return _decided_on_rank_0(probe)
