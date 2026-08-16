"""What must happen when an optional dependency is *absent* (issue #435).

`pip install libcuflynx` deliberately does not bring casadi (221 MB), autoemulate (750 MB,
mostly torch), pymc, or mpi4py. mpi4py is the awkward one: it is only 5 MB, but it compiles
against a system MPI toolchain at install time, which is the commonest pip failure on macOS
and Windows -- and a serial calibration needs none of it.

Two things have to hold for that to be a good trade, and neither is visible in an environment
that has everything installed (which every developer machine and CI runner does):

1. **The serial path really is clean.** Nothing on the import path of a calibration may need
   mpi4py. Checked by blocking the import in a fresh interpreter, since uninstalling it is not
   an option in a shared environment -- and would not prove anything about a *fresh* install
   anyway.
2. **Asking for a missing extra says which extra.** ``ModuleNotFoundError: No module named
   'casadi'`` is a true statement that tells a user nothing about how to proceed. Every
   gate here must name ``pip install "libcuflynx[...]"``.

The absence is simulated in-process or in a subprocess, never by uninstalling: these tests
have to pass in an environment where all of it *is* installed.
"""
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = str(REPO_ROOT / 'src')

from libcuflynx.utilities import mpi_utils


def _load_pyproject():
    try:
        import tomllib
    except ImportError:                                  # Python < 3.11
        tomllib = pytest.importorskip(
            'tomli', reason='needs tomllib (3.11+) or tomli to read pyproject.toml')
    return tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))


def _run_without(module, body, env=None):
    """Run ``body`` in a fresh interpreter where ``module`` looks uninstalled.

    Every finder on ``sys.meta_path`` is wrapped so that it declines this one name. That is
    what absence really looks like from Python: ``import`` raises ``ModuleNotFoundError`` and
    ``importlib.util.find_spec`` returns ``None`` -- which matters, because
    ``mpi_utils.mpi_available()`` asks the second question and a blocker that raised there
    would be testing something no user will ever hit. Inserting a finder that *raises* would
    not do: a finder is allowed to return ``None`` to mean "not mine", so the real one behind
    it would still answer.
    """
    blocker = textwrap.dedent('''
        import sys

        class _Hidden:
            """Delegates to a real finder, except for one module name."""
            def __init__(self, inner):
                self._inner = inner
            def find_spec(self, name, path=None, target=None):
                if name == %r or name.startswith(%r + '.'):
                    return None
                find_spec = getattr(self._inner, 'find_spec', None)
                if find_spec is None:
                    return None
                return find_spec(name, path, target)
            def __getattr__(self, item):
                return getattr(self._inner, item)

        sys.meta_path[:] = [_Hidden(finder) for finder in sys.meta_path]
        sys.path.insert(0, %r)
    ''' % (module, module, SRC))
    full_env = dict(os.environ)
    # The suite itself may be running under mpiexec; start from a clean slate so each case
    # controls whether this process looks like a rank.
    for var in mpi_utils.LAUNCHER_ENV_VARS:
        full_env.pop(var, None)
    full_env.update(env or {})
    return subprocess.run([sys.executable, '-c', blocker + textwrap.dedent(body)],
                          capture_output=True, text=True, timeout=600, env=full_env)


# ---------------------------------------------------------------------------
# mpi4py: the serial path must not need it
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_blocker_actually_blocks():
    """Guard the guard: if the blocker did nothing, every check below would pass vacuously."""
    proc = _run_without('mpi4py', '''
        import importlib.util
        try:
            import mpi4py
        except ModuleNotFoundError as exc:
            print('blocked', exc.name, importlib.util.find_spec('mpi4py'))
        else:
            print('NOT BLOCKED')
    ''')
    assert proc.returncode == 0, proc.stderr
    # find_spec returns None rather than raising, which is what a genuinely absent package does
    # -- and what mpi_utils.mpi_available() relies on.
    assert proc.stdout.split() == ['blocked', 'mpi4py', 'None']


@pytest.mark.unit
def test_a_serial_calibration_imports_with_no_mpi4py():
    """The whole reason mpi4py can be an extra.

    These four are what a serial ``CVODE_myokit`` calibration of a benchmark model actually
    imports: the config parser, the solver factory, the calibration class and the run script
    that ties them together. If any of them needed mpi4py, a user with no MPI toolchain could
    not run at all, and the extra would have to go back into the core dependency list.
    """
    proc = _run_without('mpi4py', '''
        import sys
        import libcuflynx.parsers.PrimitiveParsers          # noqa: F401
        import libcuflynx.solver_wrappers                   # noqa: F401
        import libcuflynx.param_id.paramID                  # noqa: F401
        import libcuflynx.scripts.param_id_run_script       # noqa: F401
        import libcuflynx.sensitivity_analysis.sensitivityAnalysis  # noqa: F401
        print('imported', 'mpi4py' in sys.modules, 'mpi4py.MPI' in sys.modules)
    ''')
    assert proc.returncode == 0, (
        'a serial import path still needs mpi4py:\n%s\n%s' % (proc.stdout, proc.stderr))
    assert proc.stdout.split() == ['imported', 'False', 'False']


@pytest.mark.unit
def test_the_serial_stub_is_what_a_launcherless_process_gets():
    """Not merely importable -- usable. ``get_MPI()`` has to hand back the one-rank stub, or
    every caller that runs a collective would still fall over."""
    proc = _run_without('mpi4py', '''
        from libcuflynx.utilities import mpi_utils
        MPI = mpi_utils.get_MPI()
        comm = MPI.COMM_WORLD
        print(comm.Get_rank(), comm.Get_size(), mpi_utils.mpi_available())
    ''')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ['0', '1', 'False']


@pytest.mark.unit
def test_a_rank_with_no_mpi4py_names_the_mpi_extra():
    """Under a launcher there is no honest fallback: the stub would answer "rank 0 of 1" in
    every one of N ranks, so N runs would silently overwrite each other's output. It must fail,
    and it must fail saying what to install -- this is the message the mpiexec-based runners
    (param id, sensitivity analysis, emulator training) produce, because ``MPI = get_MPI()`` at
    their module scope is where they all reach MPI.
    """
    proc = _run_without('mpi4py', '''
        from libcuflynx.utilities import mpi_utils
        try:
            mpi_utils.get_MPI()
        except ImportError as exc:
            print(exc)
        else:
            print('NO ERROR')
    ''', env={'PMI_RANK': '0', 'PMI_SIZE': '2'})
    assert proc.returncode == 0, proc.stderr
    assert 'pip install "libcuflynx[mpi]"' in proc.stdout
    assert 'mpi4py' in proc.stdout


@pytest.mark.unit
def test_the_param_id_run_script_fails_with_that_message_rather_than_a_bare_import_error():
    """End to end through the runner: ``mpiexec -n 2 ... param_id_run_script.py`` with no
    mpi4py must not produce ``ModuleNotFoundError: No module named 'mpi4py'`` from inside a
    rank. The check lives in ``get_MPI`` rather than in ``user_run_files/*.sh`` so it holds for
    anyone driving the entry point directly, or through a console script.
    """
    proc = _run_without('mpi4py', '''
        import libcuflynx.scripts.param_id_run_script   # noqa: F401
    ''', env={'PMI_RANK': '0', 'PMI_SIZE': '2'})
    assert proc.returncode != 0
    assert 'pip install "libcuflynx[mpi]"' in proc.stderr
    assert "No module named 'mpi4py'" not in proc.stderr


@pytest.mark.unit
def test_require_mpi4py_is_a_no_op_when_it_is_installed():
    if not mpi_utils.mpi_available():
        pytest.skip('mpi4py is not installed here, which the checks above already cover')
    assert mpi_utils.require_mpi4py() is None


# ---------------------------------------------------------------------------
# casadi
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_require_casadi_names_the_casadi_extra(monkeypatch):
    from libcuflynx.param_id import casadi_backend

    monkeypatch.setattr(casadi_backend, 'ca', None)
    with pytest.raises(ImportError) as excinfo:
        casadi_backend.require_casadi()
    assert 'pip install "libcuflynx[casadi]"' in str(excinfo.value)


@pytest.mark.unit
def test_asking_for_the_casadi_solver_without_the_extra_names_it(monkeypatch):
    """The `model_type: casadi_python` route, which is how a user meets this."""
    from libcuflynx import solver_wrappers

    monkeypatch.setattr(solver_wrappers, 'CasADiPythonSimulationHelper', None)
    monkeypatch.setitem(solver_wrappers.BACKEND_IMPORT_ERRORS, 'CasADi',
                        "ModuleNotFoundError: No module named 'casadi'")

    with pytest.raises(RuntimeError) as excinfo:
        solver_wrappers.get_simulation_helper(
            model_path='m.py', solver='casadi_integrator', model_type='casadi_python',
            dt=0.01, sim_time=1.0)
    assert 'pip install "libcuflynx[casadi]"' in str(excinfo.value)


@pytest.mark.unit
def test_a_backend_that_broke_for_another_reason_is_not_called_an_install_problem(monkeypatch):
    """The extra hint must not paper over a real error. A CasADi that is installed but whose
    import raised is not fixed by installing it again, and saying so is the point of #410."""
    from libcuflynx import solver_wrappers

    monkeypatch.setattr(solver_wrappers, 'CasADiPythonSimulationHelper', None)
    monkeypatch.setitem(solver_wrappers.BACKEND_IMPORT_ERRORS, 'CasADi',
                        'OSError: libgomp.so.1: cannot open shared object file')

    with pytest.raises(RuntimeError) as excinfo:
        solver_wrappers.get_simulation_helper(
            model_path='m.py', solver='casadi_integrator', model_type='casadi_python',
            dt=0.01, sim_time=1.0)
    message = str(excinfo.value)
    assert 'cannot open shared object file' in message
    assert 'not an installation problem' in message
    assert 'pip install' not in message


# ---------------------------------------------------------------------------
# uq and emulation
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_pymc_backend_names_the_uq_extra():
    from libcuflynx.param_id.pymc_backend import _INSTALL_HINT

    assert 'libcuflynx[uq]' in _INSTALL_HINT
    assert 'pymc' in _INSTALL_HINT


@pytest.mark.unit
def test_the_emulator_backend_names_the_emulation_extra():
    from libcuflynx.emulators.emulator_trainer import AUTOEMULATE_MISSING_MESSAGE

    assert 'pip install "libcuflynx[emulation]"' in AUTOEMULATE_MISSING_MESSAGE
    # and the size, because 750 MB is the reason it is an extra at all
    assert '750' in AUTOEMULATE_MISSING_MESSAGE


@pytest.mark.unit
def test_asking_for_an_emulator_without_the_extra_names_it(monkeypatch):
    from libcuflynx import solver_wrappers

    monkeypatch.setattr(solver_wrappers, 'EmulatorSimulationHelper', None)
    monkeypatch.delitem(solver_wrappers.BACKEND_IMPORT_ERRORS, 'emulator', raising=False)

    with pytest.raises(RuntimeError) as excinfo:
        solver_wrappers.get_simulation_helper(
            model_path='m.cellml', solver='CVODE_myokit', model_type='cellml_only',
            dt=0.01, sim_time=1.0, use_emulator=True, emulator_dir='nowhere')
    assert 'pip install "libcuflynx[emulation]"' in str(excinfo.value)


@pytest.mark.unit
def test_no_message_still_advertises_the_old_package_name():
    """The distribution is `libcuflynx`; `pip install "circulatory_autogen[emulation]"` would
    install something else entirely, or nothing."""
    offenders = []
    for path in (REPO_ROOT / 'src' / 'libcuflynx').rglob('*.py'):
        if 'obsolete' in path.parts:
            continue
        if 'circulatory_autogen[' in path.read_text(encoding='utf-8'):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], offenders


# ---------------------------------------------------------------------------
# The declaration itself
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_heavy_packages_are_not_required():
    """The list a `pip install libcuflynx` resolves. Each of these has a guarded import and a
    message naming its extra; putting any of them back here would undo that."""
    required = ' '.join(_load_pyproject()['project']['dependencies']).lower()
    for package in ('mpi4py', 'casadi', 'pymc', 'autoemulate', 'schwimmbad'):
        assert package not in required, (
            '%s is an optional extra, but it is in [project] dependencies' % package)


@pytest.mark.unit
def test_every_extra_the_epic_names_exists():
    extras = _load_pyproject()['project']['optional-dependencies']
    for name in ('mpi', 'casadi', 'uq', 'emulation', 'cpp', 'dev', 'all'):
        assert name in extras, name


@pytest.mark.unit
@pytest.mark.parametrize('extra,package', [
    ('mpi', 'mpi4py'),
    ('mpi', 'schwimmbad'),
    ('casadi', 'casadi'),
    ('uq', 'pymc'),
    ('emulation', 'autoemulate'),
])
def test_each_extra_carries_its_package(extra, package):
    extras = _load_pyproject()['project']['optional-dependencies']
    assert any(spec.lower().startswith(package) for spec in extras[extra]), extras[extra]


@pytest.mark.unit
def test_dev_carries_mpi4py():
    """`tests/conftest.py` does `from mpi4py import MPI` at module scope, so without this the
    suite does not collect -- not one test, not even the serial ones. This very file would
    never run."""
    extras = _load_pyproject()['project']['optional-dependencies']
    assert any(spec.startswith('mpi4py') for spec in extras['dev']), extras['dev']


@pytest.mark.unit
def test_all_is_every_runtime_extra():
    """`[all]` is what someone types instead of reading the size table, so leaving an extra out
    of it is a silently missing capability."""
    extras = _load_pyproject()['project']['optional-dependencies']
    referenced = {spec.split('[', 1)[1].rstrip(']') for spec in extras['all']
                  if spec.startswith('libcuflynx[')}
    assert referenced == {'mpi', 'casadi', 'uq', 'emulation', 'cpp'}
