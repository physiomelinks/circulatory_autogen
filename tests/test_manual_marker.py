"""The ``manual`` marker keeps a test out of CI *and* out of a default local run.

`slow` was not enough. The full-model UQ posterior-recovery tests are deselected by
``-m "not slow"``, but CI opted back into them explicitly and a plain ``./run_pytest.sh`` runs
everything -- so they still cost 82 of the test-uq job's 89 minutes, and minutes on every local
full run, to check posteriors that ``test_uq_on_emulator.py`` checks in about five.

A marker whose gate lives in a conftest hook is easy to break silently: a refactor of
``pytest_collection_modifyitems`` that drops the deselection puts an 80-minute test back into
every run, and the only symptom is that CI got slow. These tests fail instead.
"""
import re
import subprocess
import sys

import pytest


TESTS_DIR = __file__.rsplit('/', 1)[0]


def _collect(*args):
    """Collect (never run) with pytest in a subprocess, returning its stdout."""
    proc = subprocess.run(
        [sys.executable, '-m', 'pytest', f'{TESTS_DIR}/test_UQ.py', '--collect-only', '-q',
         '-p', 'no:cacheprovider', *args],
        capture_output=True, text=True, cwd=TESTS_DIR)
    return proc.stdout + proc.stderr


def _collected_names(output):
    """The test names in a --collect-only run, in either format pytest may print.

    This repo's pytest config renders the collection tree (``<Function name>``) rather than
    node ids, and which one you get depends on ini settings a future change could flip.
    """
    names = set(re.findall(r'<Function ([^>\[]+)', output))
    names |= {line.split('::')[-1].split('[')[0].strip()
              for line in output.splitlines() if '::' in line and line.startswith('test')}
    return names


@pytest.mark.unit
def test_manual_tests_are_not_collected_by_default():
    if not sys.executable:
        pytest.skip("this interpreter cannot spawn a subprocess (OpenCOR's pythonshell "
                    'leaves sys.executable empty)')
    out = _collect()

    assert 'test_mcmc_bimodal_with_validation' not in out, (
        'a manual test was collected by a default run -- the 56-minute full-model bimodal test '
        'is back in everyones test run')
    assert 'deselected' in out


@pytest.mark.unit
def test_run_manual_opts_back_in():
    """The escape hatch has to work, or the tests are not merely un-run but unreachable."""
    if not sys.executable:
        pytest.skip("this interpreter cannot spawn a subprocess (OpenCOR's pythonshell "
                    'leaves sys.executable empty)')
    out = _collect('--run-manual')

    assert 'test_mcmc_bimodal_with_validation' in out
    assert 'test_mcmc_unimodal_with_validation' in out


@pytest.mark.unit
def test_the_full_model_uq_tests_are_the_ones_marked_manual():
    """Named explicitly: if one of them loses the marker, an 80-minute test silently rejoins
    every run, which is the failure this whole marker exists to prevent.

    Read from what pytest actually collects rather than from the source, so it stays true
    however the markers are spelled or reordered."""
    if not sys.executable:
        pytest.skip("this interpreter cannot spawn a subprocess (OpenCOR's pythonshell "
                    'leaves sys.executable empty)')
    collected = _collected_names(_collect('--run-manual'))

    assert collected == {
        'test_mcmc_unimodal_with_validation',
        'test_mcmc_unimodal_with_validation_KDE_likelihood',
        'test_mcmc_bimodal_with_validation',
    }, f'unexpected set of manual tests: {sorted(collected)}'
