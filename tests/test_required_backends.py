"""CUFLYNX_REQUIRED_BACKENDS (tests/_required_backends.py) and the backend header.

A CI job that installs a backend sets it, so that a test skipping for want of that backend
fails rather than passing green over code that never ran.
"""
import os
import subprocess
import sys
import textwrap

import pytest

from _backend_info import backend_info, sundials_build_type, sundials_warning
from _required_backends import missing_backend

pytestmark = pytest.mark.unit

TESTS = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize('reason, backend', [
    ("could not import 'casadi': No module named 'casadi'", 'casadi'),
    ('AADC is installed but not licensed: recording a tape raises', 'aadc'),
    ('cmake not found, which the batched C solvers need', 'cmake'),
    ('SUNDIALS not found by CMake (set SUNDIALS_DIR)', 'sundials'),
    ('needs CMake and a C compiler', 'cmake'),
    ("could not import 'libopencor'", 'libopencor'),
    # not about a missing backend, or about one this job does not require
    ('casadi 3.7 cannot differentiate a jit-compiled integrator', None),
    ("could not import 'autoemulate'", None),
    ('CVODE_libopencor is not a solver here yet (it arrives with #638)', None),
])
def test_a_skip_names_a_missing_required_backend(reason, backend):
    required = ['myokit', 'casadi', 'aadc', 'cmake', 'sundials', 'libopencor']
    assert missing_backend(reason, required) == backend


def test_a_skip_about_an_unrequired_backend_that_mentions_a_required_one_stays_a_skip():
    """The Windows job requires cmake, not SUNDIALS."""
    reason = ('CVODES_batched_c needs SUNDIALS 6 or later (CVODES): set SUNDIALS_DIR to its '
              'install prefix if CMake does not find it.')
    assert missing_backend(reason, ['myokit', 'cmake']) is None
    assert missing_backend(reason, ['myokit', 'cmake', 'sundials']) == 'sundials'


_EXAMPLE = '''
import pytest

def test_needs_casadi(record_property):
    record_property('backend', 'casadi_integrator')
    pytest.importorskip('a_module_that_is_not_there', reason="could not import 'casadi'")

def test_needs_a_version():
    pytest.skip('casadi 3.7 cannot differentiate a jit-compiled integrator')

def test_passes():
    pass
'''


def _run(tmp_path, required):
    (tmp_path / 'test_example.py').write_text(textwrap.dedent(_EXAMPLE))
    env = dict(os.environ, PYTHONPATH=TESTS + os.pathsep + os.environ.get('PYTHONPATH', ''),
               CUFLYNX_REQUIRED_BACKENDS=required)
    return subprocess.run(
        [sys.executable, '-m', 'pytest', '-p', '_required_backends', '-p', 'no:cacheprovider',
         '-rA', '--rootdir', str(tmp_path), str(tmp_path / 'test_example.py')],
        capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=120)


def test_required_backend_skip_becomes_a_failure(tmp_path):
    done = _run(tmp_path, 'myokit,casadi')
    assert done.returncode == 1, done.stdout
    assert 'FAILED test_example.py::test_needs_casadi' in done.stdout
    assert "requires 'casadi'" in done.stdout
    assert '1 failed, 1 passed, 1 skipped' in done.stdout  # the version skip stays a skip
    assert 'no test passed for casadi' in done.stdout


def test_without_the_variable_skips_stay_skips(tmp_path):
    done = _run(tmp_path, '')
    assert done.returncode == 0, done.stdout
    assert '1 passed, 2 skipped' in done.stdout


def test_backend_info_names_each_installed_backend():
    info = backend_info()
    import myokit
    assert info['myokit'] == myokit.__version__
    assert info['sundials (myokit)']
    for name in ('casadi', 'libopencor', 'aadc'):
        try:
            __import__(name)
        except ImportError:
            assert info[name] is None
        else:
            assert info[name], name


def test_an_unoptimised_sundials_is_warned_about_not_refused(tmp_path):
    include = tmp_path / 'sundials'
    include.mkdir()
    (include / 'sundials_config.h').write_text('#define SUN_BUILD_TYPE ""\n')
    assert sundials_build_type([str(tmp_path)]) == ''
    assert 'not Release' in sundials_warning('')
    assert 'Debug' in sundials_warning('Debug')
    assert sundials_warning('Release') is None
    assert sundials_warning(None) is None  # SUNDIALS 6 records no build type
