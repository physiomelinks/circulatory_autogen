"""``--collect-only`` must not block in the one-rank result-file wait.

Listing the suite runs no tests, but ``pytest_collection_modifyitems`` still counts the
one-rank tests (autogeneration, solver, misc) into the ``*_SUMMARY_TOTAL`` environment
variables, and pytest then exits 0 with no errors. That combination used to satisfy
``_should_wait_for_result_files``, so ``pytest_terminal_summary`` waited on three result
files that nothing would ever write -- 1800 s each, 90 minutes to answer "what tests are
there". Only the three modules that populate those totals were affected
(``test_autogeneration.py``, ``test_solvers.py``, ``test_omex_analysis_pipeline.py``);
every other file listed in about three seconds, which is why the hang looked like a slow
import rather than a wait.

A real run is unaffected -- the tests execute and append their lines, so the wait returns
on its first iteration. That is exactly why this needs a test: the bug is invisible to a
green suite.

These exercise ``_should_wait_for_result_files`` directly rather than launching a nested
pytest, for the reason ``test_manual_marker.py`` documents: an inner session's
``pytest_configure`` deletes the shared ``.pytest_*_results`` files, wiping the *outer*
run's accumulated results and sending it into the very 1800 s wait this file is about.
"""
import pytest

from conftest import _should_wait_for_result_files


class _Config:
    """The one thing ``_should_wait_for_result_files`` uses from a pytest Config."""

    def __init__(self, collectonly):
        self._collectonly = collectonly

    def getoption(self, name, default=None):
        assert name == "collectonly", name
        return self._collectonly


class _TerminalReporter:
    def __init__(self, stats=None):
        self.stats = stats or {}


@pytest.mark.unit
def test_collect_only_does_not_wait():
    """The regression: a clean --collect-only exit must skip the wait."""
    assert not _should_wait_for_result_files(
        0, _TerminalReporter(), _Config(collectonly=True)
    )


@pytest.mark.unit
@pytest.mark.parametrize("exitstatus", [0, 1])
def test_normal_run_still_waits(exitstatus):
    """A real run (pass or fail) still waits, so the rank-0 summary stays complete."""
    assert _should_wait_for_result_files(
        exitstatus, _TerminalReporter(), _Config(collectonly=False)
    )


@pytest.mark.unit
def test_collection_error_still_does_not_wait():
    """The pre-existing guard is untouched: errors never produce result files."""
    assert not _should_wait_for_result_files(
        1, _TerminalReporter({"error": [object()]}), _Config(collectonly=False)
    )


@pytest.mark.unit
@pytest.mark.parametrize("exitstatus", [2, 3, 4, 5])
def test_abnormal_exit_status_still_does_not_wait(exitstatus):
    """Interrupted / internal-error / usage-error / no-tests-collected exits."""
    assert not _should_wait_for_result_files(
        exitstatus, _TerminalReporter(), _Config(collectonly=False)
    )
