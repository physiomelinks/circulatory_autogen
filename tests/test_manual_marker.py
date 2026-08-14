"""The ``manual`` marker keeps a test out of CI *and* out of a default local run.

`slow` was not enough. The full-model UQ posterior-recovery tests are deselected by
``-m "not slow"``, but CI opted back into them explicitly and a plain ``./run_pytest.sh`` runs
everything -- so they still cost 82 of the test-uq job's 89 minutes, and minutes on every local
full run, to check posteriors that ``test_uq_on_emulator.py`` checks in about five.

A marker whose gate lives in a conftest hook is easy to break silently: a refactor of
``pytest_collection_modifyitems`` that drops the deselection puts an 80-minute test back into
every run, and the only symptom is that CI got slow. These tests fail instead.

They exercise ``drop_manual_tests`` directly rather than launching a nested pytest. An inner
session is not merely slower -- ``pytest_configure`` deletes the shared ``.pytest_*_results``
files at session start, so it wipes the *outer* run's accumulated results, and the outer run
then blocks in ``_wait_for_expected_result_count`` for its full 1800 s timeout. The first
version of this file did exactly that and turned a 12-minute CI job into a 50-minute one.
"""
import pytest

from conftest import drop_manual_tests

import test_UQ


#: Every test in test_UQ.py samples the real model, so all three must carry the marker.
FULL_MODEL_UQ_TESTS = (
    'test_mcmc_unimodal_with_validation',
    'test_mcmc_unimodal_with_validation_KDE_likelihood',
    'test_mcmc_bimodal_with_validation',
)


class _Config:
    """The two things drop_manual_tests uses from a pytest Config."""

    def __init__(self, run_manual):
        self._run_manual = run_manual
        self.deselected = []
        config = self

        class _Hook:
            def pytest_deselected(self, items):
                config.deselected.extend(items)

        self.hook = _Hook()

    def getoption(self, name):
        assert name == '--run-manual', name
        return self._run_manual


class _Item:
    def __init__(self, name, *marks):
        self.nodeid = f'tests/test_x.py::{name}'
        self.keywords = {name: True, **{mark: True for mark in marks}}


def _markers_on(func):
    return {mark.name for mark in getattr(func, 'pytestmark', [])}


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_manual_tests_are_dropped_from_a_default_run():
    items = [_Item('cheap', 'unit'), _Item('expensive', 'slow', 'manual')]

    drop_manual_tests(_Config(run_manual=False), items)

    assert [item.nodeid for item in items] == ['tests/test_x.py::cheap'], (
        'a manual test survived a default run -- the 56-minute full-model bimodal test is back '
        'in everyones test run')


@pytest.mark.unit
def test_they_are_deselected_rather_than_skipped():
    """Skipping would leave them counted in the rank-assignment bookkeeping that the rest of
    pytest_collection_modifyitems builds from `items`, and would put a permanent line of noise
    in every run's summary."""
    config = _Config(run_manual=False)
    items = [_Item('expensive', 'manual')]

    drop_manual_tests(config, items)

    assert [item.nodeid for item in config.deselected] == ['tests/test_x.py::expensive']


@pytest.mark.unit
def test_run_manual_opts_back_in():
    """The escape hatch has to work, or the tests are not merely un-run but unreachable."""
    config = _Config(run_manual=True)
    items = [_Item('cheap', 'unit'), _Item('expensive', 'manual')]

    drop_manual_tests(config, items)

    assert len(items) == 2
    assert config.deselected == []


@pytest.mark.unit
def test_a_run_with_nothing_manual_is_untouched():
    config = _Config(run_manual=False)
    items = [_Item('cheap', 'unit'), _Item('slower', 'slow')]

    drop_manual_tests(config, items)

    assert len(items) == 2
    assert config.deselected == []


# ---------------------------------------------------------------------------
# and it is the full-model UQ tests that carry the marker
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize('name', FULL_MODEL_UQ_TESTS)
def test_the_full_model_uq_tests_are_marked_manual(name):
    """Named explicitly: if one of them loses the marker, an 80-minute test silently rejoins
    every run, which is the failure this whole marker exists to prevent."""
    assert 'manual' in _markers_on(getattr(test_UQ, name)), (
        f'{name} samples the real model but is no longer marked manual')


@pytest.mark.unit
def test_no_other_test_in_test_UQ_escaped_the_marker():
    """The file is entirely full-model tests. A new one added without the marker would run in
    CI, which is the mistake this catches."""
    unmarked = [name for name in dir(test_UQ)
                if name.startswith('test_') and callable(getattr(test_UQ, name))
                and 'manual' not in _markers_on(getattr(test_UQ, name))]

    assert unmarked == [], f'test_UQ.py tests missing the manual marker: {unmarked}'
