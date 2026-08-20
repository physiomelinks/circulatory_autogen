"""Deferred imports of dependencies that are only needed to draw a plot.

Two separate problems live here, and the second one is why this module exists rather
than a bare ``try: import x`` at the top of each caller.

**1. It puts a whole scientific stack on the import path of every calibration.**
``param_id/paramID.py`` is imported by *every* calibration, whether or not it will
ever draw anything, and ``import corner`` drags in ``arviz``, ``xarray`` and
``xarray_einstats`` behind it. The *time* saved is small and worth stating honestly
-- measured at **0.05 s**, because ``paramID`` imports matplotlib and numpy anyway
and those dominate. The point is not speed. It is that three large third-party
packages, and every import-time side effect they carry, were being executed by every
rank of every run to support four call sites in one method that most runs never
reach.

**2. An import can fail with something that is not an ``ImportError``.**
This is the one that actually bit. ``arviz`` 0.23.4 runs ``_warn_once_per_day()`` at
module scope, which writes a stamp file through a *fixed* temporary name::

    tmp = path.with_suffix(path.suffix + ".tmp")   # ~/.cache/arviz/daily_warning.tmp
    tmp.write_text(text)
    tmp.replace(path)

Every MPI rank computes the same temporary path. Rank A writes it, rank B writes it,
rank A renames it away, and rank B's ``replace()`` raises::

    FileNotFoundError: '~/.cache/arviz/daily_warning.tmp' -> '~/.cache/arviz/daily_warning'

``FileNotFoundError`` is not an ``ImportError``, so the ``except ImportError`` guards
around ``import corner`` -- both the one here and the identical one inside
``corner/corner.py`` -- let it through. It killed rank 3 of a four-rank job, and
MPI_ABORT took the other three down with it.

Measured on a 20-core machine with a cold ``~/.cache/arviz``: 33 of 60 ranks lost at
``mpiexec -n 4``, 14 of 30 at ``-n 2``, and **0 of 20** once the stamp file existed.
That last number is the trap -- the failure is real for exactly one run per day, so
re-running "fixes" it and it reads as random.

So the rule this module encodes is **narrow scope, wide exception, loud log**: catch
``Exception`` rather than ``ImportError``, say exactly what was swallowed, and never
let an optional *plotting* dependency abort a calibration that had already run for
an hour.

``CUFLYNX_STRICT_STARTUP=1`` turns the swallow back into a raise. Production wants
the wide catch; CI wants to know. Set it in any job that is meant to prove the
import path is clean.
"""
import importlib
import logging
import os

logger = logging.getLogger(__name__)

#: Sentinel distinguishing "not tried yet" from "tried and failed" -- ``None`` is the
#: documented answer for the latter, so it cannot also mean the former.
_UNSET = object()

_cache = {}


def strict_startup():
    """Whether a failed optional import should raise instead of returning ``None``.

    Read at call time rather than at import time so a test can set it with
    ``monkeypatch.setenv`` without having to reload this module.
    """
    return os.environ.get("CUFLYNX_STRICT_STARTUP", "").strip() not in ("", "0")


def load_optional(name):
    """Import *name* on first use; return the module, or ``None`` if it will not load.

    The result -- including the failure -- is cached, so a broken optional dependency
    costs one import attempt per process rather than one per plot.

    Catches ``Exception``, deliberately. See the module docstring: the failure this
    was written for is a ``FileNotFoundError`` raised by a third-party module's
    import-time side effect, and an ``except ImportError`` does not see it.
    ``BaseException`` is *not* caught -- a ``KeyboardInterrupt`` during an import is
    still a ``KeyboardInterrupt``.
    """
    cached = _cache.get(name, _UNSET)
    if cached is not _UNSET:
        return cached

    try:
        module = importlib.import_module(name)
    except Exception as exc:
        if strict_startup():
            raise
        # exc_info so the traceback names the *third-party* frame that actually
        # failed. Without it this reads as "corner is missing", which sent the
        # original investigation looking in the wrong place entirely.
        logger.warning(
            "optional dependency %r could not be imported (%s: %s); anything that "
            "needs it will be skipped. This does not affect the calibration itself.",
            name, type(exc).__name__, exc, exc_info=True,
        )
        module = None

    _cache[name] = module
    return module


def load_corner():
    """``corner``, or ``None``. Import it here, not at module scope -- see above."""
    return load_optional("corner")


def require_corner(what):
    """``corner``, or raise ``ImportError`` naming *what* could not be drawn.

    For the call sites that have nothing useful to do without it, as opposed to the
    ones that can carry on and skip a figure.
    """
    module = load_corner()
    if module is None:
        raise ImportError(
            f"corner is required to {what}. It is a core dependency, so this is "
            f"usually a broken environment rather than a missing install -- run "
            f"`python -c \"import corner\"` to see the underlying error, or set "
            f"CUFLYNX_STRICT_STARTUP=1 to have it raised rather than logged."
        )
    return module


def reset_cache():
    """Forget what has been tried. For tests only."""
    _cache.clear()
