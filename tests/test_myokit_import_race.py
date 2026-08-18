"""Myokit's first import is a race, and CA has to survive losing it.

Myokit creates its user config directory at import time with a check and a create and a gap in
between::

    if not os.path.exists(DIR_USER):
        os.makedirs(DIR_USER)

Under ``mpiexec`` every rank imports at once. On a machine where ``~/.config/myokit`` does not
yet exist -- a fresh CI runner, a new HPC home directory, a user's first parallel run -- they
all see "missing", all call ``makedirs``, and every rank but one dies with::

    FileExistsError: [Errno 17] File exists: '/home/runner/.config/myokit'

surfacing as "the Myokit backend failed to import". It disappears on the next run, because by
then the directory exists, which is exactly what makes it look like flakiness worth re-running
rather than a bug worth fixing.
"""

import pytest

from solver_wrappers.myokit_helper import _import_myokit_tolerating_first_run_race

pytestmark = pytest.mark.unit


class _RacingImport:
    """Fails with FileExistsError for the first ``losses`` calls, as a losing rank does."""

    def __init__(self, losses, result='myokit'):
        self.losses = losses
        self.calls = 0
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.losses:
            raise FileExistsError(17, 'File exists', '/home/runner/.config/myokit')
        return self.result


def _run(monkeypatch, importer, **kwargs):
    """Drive the helper with ``import myokit`` replaced by ``importer``."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **rest):
        if name == 'myokit':
            return importer()
        return real_import(name, *args, **rest)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    return _import_myokit_tolerating_first_run_race(delay=0, **kwargs)


def test_a_rank_that_loses_the_race_retries_and_succeeds(monkeypatch):
    """The whole point: losing the makedirs race is recoverable, because the rank that won it
    has created the very directory this one failed to create."""
    importer = _RacingImport(losses=1)
    assert _run(monkeypatch, importer) == 'myokit'
    assert importer.calls == 2, 'it should have retried exactly once'


def test_it_gives_up_rather_than_spinning_forever(monkeypatch):
    """A FileExistsError that never clears is a real problem -- a file sitting where the
    directory should be, say -- and must surface rather than be retried out of sight."""
    importer = _RacingImport(losses=99)
    with pytest.raises(FileExistsError):
        _run(monkeypatch, importer, attempts=3)
    assert importer.calls == 3


def test_any_other_import_failure_is_raised_immediately(monkeypatch):
    """Only the race is retried. A missing Myokit, or a broken one, is the caller's to see at
    once -- and solver_wrappers reports it with its reason (#410)."""
    calls = []

    def importer():
        calls.append(1)
        raise ImportError('No module named myokit')

    with pytest.raises(ImportError):
        _run(monkeypatch, importer)
    assert len(calls) == 1, 'a genuine import failure must not be retried'


def test_the_common_case_imports_once(monkeypatch):
    """Nothing is paid when there is no race -- which is every run after the first."""
    importer = _RacingImport(losses=0)
    assert _run(monkeypatch, importer) == 'myokit'
    assert importer.calls == 1
