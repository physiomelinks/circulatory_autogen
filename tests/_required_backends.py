"""Turn a required backend's skip into a failure, and record which backends a run used.

``CUFLYNX_REQUIRED_BACKENDS`` is a comma-separated list of backends the job installed, e.g.
``myokit,casadi,mpi4py,cmake,sundials``. A test that skips because one of them is missing --
an ``importorskip`` of it, or a skip reason naming it as unavailable -- then fails instead:
the job set out to run it, and a green skip would hide that it did not. A skip for any other
reason (a version, a platform, an optional extra this job does not install) stays a skip.

Tests that run one backend record ``record_property('backend', <name>)``. At the end of a run,
a required backend that some collected test is about, but that no test passed with, fails the
session: its tests were all skipped or deselected.

It is a plugin rather than part of conftest.py so that tests/test_required_backends.py can load
it into a pytest run of its own. conftest.py registers it.
"""
import os
import re

import pytest

ENV = 'CUFLYNX_REQUIRED_BACKENDS'

# What a skip reason says about each backend when it is missing. The backend's own name is
# always one of them.
ALIASES = {
    'cmake': ['cmake'],
    'cc': ['c compiler', 'no compiler', 'compiler not found'],
    'sundials': ['sundials'],
    'aadc': ['aadc', 'matlogica'],
    'libopencor': ['libopencor', 'cvode_libopencor'],
}
MISSING = re.compile(r"could not import|no module named|not installed|unavailable|"
                     r"not available|not found|not licen[cs]ed|missing|\bneeds?\b",
                     re.IGNORECASE)


def required_backends(env=None):
    env = os.environ if env is None else env
    return [b.strip().lower() for b in env.get(ENV, '').split(',') if b.strip()]


def missing_backend(reason, required):
    """The required backend a skip reason says is missing, or None."""
    text = reason.lower()
    if not MISSING.search(text):
        return None
    # The backend named first: 'SUNDIALS not found by CMake' is about SUNDIALS.
    found = []
    for backend in required:
        positions = [text.find(w) for w in ALIASES.get(backend, []) + [backend] if w in text]
        if positions:
            found.append((min(positions), backend))
    if found:
        return min(found)[1]
    return None


def _skip_reason(report):
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])
    return str(longrepr)


# The backend a test records with record_property('backend', ...) -> the installed backend
# it needs, as named in CUFLYNX_REQUIRED_BACKENDS.
NEEDS = {
    'cvode_myokit': 'myokit', 'cvode_libopencor': 'libopencor', 'casadi_integrator': 'casadi',
    'be_batched_c': 'cmake', 'cvodes_batched_c': 'cmake', 'solve_ivp': 'scipy',
    'aadc_semi_implicit': 'aadc', 'cpp_cvode': 'sundials',
}

# {installed backend: number of passed tests that recorded it}
_backend_runs = {}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    required = required_backends()
    if report.skipped and required and not hasattr(report, 'wasxfail'):
        backend = missing_backend(_skip_reason(report), required)
        if backend is not None:
            report.outcome = 'failed'
            report.longrepr = (f'{ENV} requires {backend!r}, but this test skipped: '
                               f'{_skip_reason(report)}')
    for name, value in item.user_properties:
        if name == 'backend':
            key = NEEDS.get(str(value).lower(), str(value).lower())
            _backend_runs.setdefault(key, 0)
            if report.when == 'call' and report.passed:
                _backend_runs[key] += 1


def pytest_sessionfinish(session, exitstatus):
    required = required_backends()
    unrun = sorted(b for b, passed in _backend_runs.items() if b in required and passed == 0)
    if unrun:
        reporter = session.config.pluginmanager.getplugin('terminalreporter')
        if reporter is not None:
            reporter.write_line(f'{ENV}: no test passed for {", ".join(unrun)}; every one of '
                                'its tests was skipped or deselected', red=True)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
