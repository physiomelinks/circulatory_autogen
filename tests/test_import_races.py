"""Concurrent ranks must be able to import the calibration stack.

The failure this file exists to prevent, in full, because it is not obvious and it cost
a day:

``arviz`` 0.23.4 (and 0.23.3 -- both, checked) runs ``_warn_once_per_day()`` at module
scope. That writes a stamp file through a **fixed** temporary name::

    tmp = path.with_suffix(path.suffix + ".tmp")   # ~/.cache/arviz/daily_warning.tmp
    tmp.write_text(text)
    tmp.replace(path)

Every process computes the same temporary path, so concurrent MPI ranks race: one renames
it away and the others' ``replace()`` raises ``FileNotFoundError``. That is not an
``ImportError``, so it went straight through the ``except ImportError`` around
``import corner`` -- ours *and* the identical one inside ``corner/corner.py`` -- and killed
a rank. MPI_ABORT then killed the job.

Three properties of that bug decide how these tests are written:

1. **It needs a cold cache.** After one process writes the stamp, ``last_date == today``
   short-circuits and the whole day is clean. Measured: 33/60 ranks lost at ``-n 4`` with
   a cold cache, **0/20** with a warm one. So every test here points ``XDG_CACHE_HOME``
   at a fresh directory. Without that they would pass on the second run of the day and
   for the rest of it.

2. **It needs the ranks in lockstep, which pytest destroys.** Measured on the same
   machine, same cold cache, same ``-n 4``, the same import: an app-shaped script lost
   33/60 ranks; ``mpiexec -n 4 pytest`` with the import at module scope lost **0/40**.
   pytest's plugin loading, conftest and collection stagger the ranks by tens to hundreds
   of milliseconds and the vulnerable window is microseconds. **So these tests launch
   their own subprocesses and must never be run under ``mpiexec`` themselves** -- an
   in-process version cannot fail, and would be a test that reports a pass having
   probed nothing. (``test_uq_pymc_mpi.py`` launches its own ranks for the same reason.)

3. **It is probabilistic, and the odds are tunable.** One green run is not evidence, and
   neither is one red one. Every constant in this file was chosen by measuring the catch
   rate against deliberately broken code rather than by picking a round number -- see the
   table beside ``RANKS``. The first version of this file used four ranks and no
   pre-warm, which catches the bug three times in ten: it passed against the restored bug
   and would have shipped as a test that mostly lies.

``test_paramid_does_not_import_corner_or_arviz`` is the deterministic guard and the one
that runs everywhere; the concurrent tests are the ones that reproduce the original
failure, and they need ``arviz`` actually installed to mean anything.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = str(REPO_ROOT / "src")

#: Rank count and trial count are *measured*, not guessed. Against deliberately broken
#: code (the module-level ``import corner`` restored), the fraction of trials that caught
#: it on a 20-core machine was:
#:
#:     ranks=4, no pre-warm ->  3/10      <- what this file shipped with first; too flaky
#:     ranks=4, pre-warm    ->  9/10
#:     ranks=8, no pre-warm -> 10/10
#:     ranks=8, pre-warm    ->  9/10
#:
#: Eight ranks with the pre-warm below is ~0.9 per trial, so three trials miss a real
#: regression about once in a thousand runs. Four ranks without the pre-warm -- the
#: obvious choice -- would have missed it seven times in ten, which is a test that mostly
#: lies. --oversubscribe covers CI runners with fewer slots than RANKS.
RANKS = 8

#: Probabilistic, so a single trial proves nothing; see the table above.
TRIALS = 3


def _mpiexec():
    for name in ("mpiexec", "mpirun"):
        found = shutil.which(name)
        if found:
            return found
    return None


needs_mpiexec = pytest.mark.skipif(_mpiexec() is None, reason="no mpiexec on PATH")


def _run_on_ranks(program, env_extra=None, timeout=600):
    """Run *program* on RANKS ranks and return (returncode, combined output).

    Ranks are launched directly rather than through pytest, deliberately -- see the
    module docstring, point 2.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    # Agg so a rank without a display does not fail for an unrelated reason.
    env["MPLBACKEND"] = "Agg"
    env.update(env_extra or {})
    cmd = [_mpiexec(), "--oversubscribe", "-n", str(RANKS), sys.executable, "-c", program]
    proc = subprocess.run(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, timeout=timeout,
    )
    return proc.returncode, proc.stdout


# ---------------------------------------------------------------------------
# The deterministic guard
# ---------------------------------------------------------------------------

_IMPORT_PROBE = textwrap.dedent(
    """
    import json, sys
    import libcuflynx.param_id.paramID          # noqa: F401
    import libcuflynx.identifiabilty_analysis.identifiabilityAnalysis  # noqa: F401
    print("RESULT " + json.dumps(sorted(
        m for m in ("corner", "arviz", "xarray") if m in sys.modules)))
    """
)


def test_paramid_does_not_import_corner_or_arviz():
    """The calibration stack must not drag a plotting stack onto the import path.

    This is the fix stated as a property rather than as an implementation, so it survives
    the next refactor: whatever ``paramID`` does internally, importing it must not execute
    ``corner`` (and therefore ``arviz``, ``xarray``, ``xarray_einstats``) or any
    import-time side effect they carry.

    Deterministic and cheap, unlike everything below it, and it is the assertion that
    actually fails if someone restores the module-level ``import corner``.
    """
    env = dict(os.environ, PYTHONPATH=SRC + os.pathsep + os.environ.get("PYTHONPATH", ""),
               MPLBACKEND="Agg")
    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE], env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, universal_newlines=True, timeout=600,
    )
    assert proc.returncode == 0, f"the probe itself failed:\n{proc.stdout}"
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert line, f"probe printed no result:\n{proc.stdout}"
    leaked = json.loads(line[-1][len("RESULT "):])
    assert leaked == [], (
        f"importing the calibration stack pulled in {leaked}. corner must stay deferred "
        f"to its call sites in plot_mcmc()/plot_laplace_results() -- see "
        f"libcuflynx/utilities/lazy_imports.py for why this is not merely tidiness."
    )


# ---------------------------------------------------------------------------
# The concurrent tests -- the ones that reproduce the original failure
# ---------------------------------------------------------------------------

# BARRIER-AT-THE-IMPORT, and why a plain barrier is not enough.
#
# The obvious probe -- barrier, then import -- does not work, and shipping it would have
# been worse than shipping nothing. Measured on a 20-core machine, four ranks released
# from a barrier and then importing the calibration stack reach `import corner`
# **277-326 ms apart**, because the 2.4 s of libcellml/myokit/matplotlib imports in
# between is not identical work at identical speed. The vulnerable window inside arviz is
# the microseconds between `tmp.write_text()` and `tmp.replace()`. A 300 ms spread never
# lands in it, and the test passes against known-broken code -- verified: it did.
#
# That is the same effect that makes `mpiexec -n N pytest` a weak probe (module docstring,
# point 2), arriving from a different direction. Any work between the synchronisation
# point and the import under test destroys the probe.
#
# So the barrier is moved to the import itself. A meta-path finder sits in front of
# `sys.meta_path`, and the first time any rank asks for the target module it calls
# `comm.Barrier()` and then returns None so the ordinary finders load it as usual. Every
# rank therefore begins executing the module body within microseconds of the others, no
# matter how staggered they were beforehand.
#
# This is a deliberate **worst case**, not a simulation of production timing -- and that is
# the right thing for a test to be. It answers "if these ranks did arrive together, would
# they collide?", deterministically, instead of rolling dice on the scheduler.
#
# Deadlock note: every rank must reach the barrier or it hangs. Ranks run the same
# program, so they do -- and if a change ever makes them diverge, the subprocess timeout
# reports it as a failure, which is the correct answer to "some ranks import this and
# some do not".
_CONCURRENT_PROBE = textwrap.dedent(
    """
    import importlib.abc, sys
    # Pre-warm what the target's own module body will need. Anything it has to import
    # *after* the barrier is work that re-staggers the ranks before they reach the
    # vulnerable lines. Measured: this alone takes the catch rate at four ranks from
    # 3/10 to 9/10.
    import matplotlib, matplotlib.colors, matplotlib.pyplot  # noqa: F401
    import platformdirs, packaging.version, re, logging      # noqa: F401
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    class BarrierOn(importlib.abc.MetaPathFinder):
        '''Synchronise every rank immediately before *name*'s module body runs.'''
        def __init__(self, name):
            self.name = name
            self.fired = False
        def find_spec(self, fullname, path=None, target=None):
            if fullname == self.name and not self.fired:
                self.fired = True
                comm.Barrier()
            return None      # never claim the module; let the real finders load it

    sys.meta_path.insert(0, BarrierOn({target!r}))

    try:
        {import_line}
    except BaseException as exc:            # noqa: BLE001 - reporting, not handling
        print("FAIL rank %d %s: %s" % (comm.rank, type(exc).__name__, exc), flush=True)
        sys.exit(1)
    print("OK rank %d" % comm.rank, flush=True)
    """
)


@needs_mpiexec
def test_concurrent_ranks_import_the_calibration_stack_with_a_cold_cache(tmp_path):
    """Four ranks importing the calibration stack at once, on a cache that does not exist.

    The regression test for the original crash. ``XDG_CACHE_HOME`` points at a fresh
    directory per trial, which is what makes a once-per-day failure fire on every run
    instead of once -- the single most useful line in this file.

    Under the fix this passes for a reason worth stating: nothing on the calibration
    import path asks for ``corner`` at all, so the barrier never fires and there is no
    race to lose. Restore the module-level ``import corner`` and it fires on every rank
    at once, and the trials go red. That is the mutation this test is verified against.
    """
    program = _CONCURRENT_PROBE.format(
        target="arviz",
        import_line="import libcuflynx.param_id.paramID  # noqa: F401")
    failures = []
    for trial in range(TRIALS):
        cache = tmp_path / f"cache_{trial}"
        cache.mkdir()
        code, out = _run_on_ranks(program, {"XDG_CACHE_HOME": str(cache)})
        if code != 0 or "FAIL" in out:
            failures.append(f"--- trial {trial} (exit {code}) ---\n{out}")
    assert not failures, (
        f"{len(failures)} of {TRIALS} cold-cache trials lost a rank while importing the "
        f"calibration stack on {RANKS} ranks:\n\n" + "\n".join(failures)
    )


@needs_mpiexec
def test_concurrent_ranks_import_corner_itself_with_a_cold_cache(tmp_path):
    """The upstream defect, pinned directly, so we find out when it is fixed or spreads.

    The test above passes because ``paramID`` no longer imports ``corner``. That is the
    fix, and it is the right fix -- but it makes the upstream bug invisible to our suite,
    and ``corner`` is still a core dependency that ``plot_mcmc()`` imports on rank 0. This
    test imports ``corner`` on every rank on purpose.

    It is *expected to fail* against arviz 0.23.3/0.23.4, so it is skipped unless
    ``CUFLYNX_ASSERT_UPSTREAM_IMPORTS`` is set. Run it deliberately (and in the weekly
    dependency-upgrade job) to learn when a fixed arviz ships -- at which point this
    becomes an ordinary test and the skip comes off.
    """
    if not os.environ.get("CUFLYNX_ASSERT_UPSTREAM_IMPORTS", "").strip().strip("0"):
        pytest.skip(
            "set CUFLYNX_ASSERT_UPSTREAM_IMPORTS=1 to assert that `import corner` is "
            "safe on concurrent ranks; it is known-broken on arviz 0.23.3 and 0.23.4"
        )
    pytest.importorskip("arviz", reason="no arviz installed, so there is no race to run")
    program = _CONCURRENT_PROBE.format(
        target="arviz", import_line="import corner  # noqa: F401")
    failures = []
    for trial in range(TRIALS):
        cache = tmp_path / f"cache_{trial}"
        cache.mkdir()
        code, out = _run_on_ranks(program, {"XDG_CACHE_HOME": str(cache)})
        if code != 0 or "FAIL" in out:
            failures.append(f"--- trial {trial} (exit {code}) ---\n{out}")
    assert not failures, (
        f"`import corner` still races on {RANKS} concurrent ranks with a cold cache "
        f"({len(failures)}/{TRIALS} trials):\n\n" + "\n".join(failures)
    )
