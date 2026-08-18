"""An unavailable solver backend must say *why* it is unavailable.

Every optional backend is imported under a try/except so a missing package does not break an
install that never uses it. The cost was that a genuinely-broken import looked exactly like a
missing one: both became ``None``, and both produced "X is not available".

That is not a cosmetic difference. A fresh CI runner where two MPI ranks import myokit at once
can race on creating ``~/.config/myokit/myokit.ini``, and the resulting message told us myokit
was not installed -- when it was, and the real error had already been discarded.
"""
import pytest

from libcuflynx.solver_wrappers import (
    BACKEND_IMPORT_ERRORS, _record_import_error, _unavailable_message, get_simulation_helper)
from libcuflynx.solver_wrappers import opencor_helper
from libcuflynx.solver_wrappers.opencor_helper import (
    OpenCORUnavailableError, is_opencor_available, opencor_unavailable_message, require_opencor)


@pytest.mark.unit
def test_the_reason_an_import_failed_is_kept():
    try:
        raise ValueError('myokit.ini is missing a section header')
    except ValueError as exc:
        _record_import_error('FakeBackend', exc)

    try:
        assert 'myokit.ini is missing a section header' in BACKEND_IMPORT_ERRORS['FakeBackend']
        assert 'ValueError' in BACKEND_IMPORT_ERRORS['FakeBackend']
    finally:
        BACKEND_IMPORT_ERRORS.pop('FakeBackend', None)


@pytest.mark.unit
def test_the_message_names_the_underlying_error_when_there_is_one():
    try:
        raise RuntimeError('libsundials_cvodes.so: cannot open shared object file')
    except RuntimeError as exc:
        _record_import_error('FakeBackend', exc)

    try:
        message = _unavailable_message('FakeBackend', 'fake_solver')
        assert 'cannot open shared object file' in message
        assert 'fake_solver' in message
        # and it must say plainly that installing the package is not the fix
        assert 'not an installation problem' in message
    finally:
        BACKEND_IMPORT_ERRORS.pop('FakeBackend', None)


@pytest.mark.unit
def test_a_genuinely_absent_backend_still_reads_as_absent():
    """No recorded reason means the import never ran or the package really is missing; the
    message must not imply a hidden error that does not exist."""
    BACKEND_IMPORT_ERRORS.pop('NeverImported', None)
    message = _unavailable_message('NeverImported', 'some_solver')
    assert 'not available' in message
    assert 'failed to import' not in message


@pytest.mark.unit
def test_requesting_a_backend_that_failed_to_import_surfaces_the_reason(monkeypatch):
    """End to end through the factory: the RuntimeError a caller sees carries the cause."""
    from libcuflynx import solver_wrappers

    monkeypatch.setattr(solver_wrappers, 'MyokitSimulationHelper', None)
    monkeypatch.setitem(solver_wrappers.BACKEND_IMPORT_ERRORS, 'Myokit',
                        "ImportError: no module named 'configparser'")

    with pytest.raises(RuntimeError) as excinfo:
        get_simulation_helper(model_path='m.cellml', solver='CVODE_myokit',
                              model_type='cellml', dt=0.01, sim_time=1.0)

    message = str(excinfo.value)
    assert "no module named 'configparser'" in message
    assert 'CVODE_myokit' in message


@pytest.mark.unit
def test_every_optional_backend_records_its_failures():
    """A backend added later must be wired into the same reporting, or it reintroduces the
    silent-None behaviour this replaces."""
    import inspect

    from libcuflynx import solver_wrappers

    source = inspect.getsource(solver_wrappers)
    # Each optional import is a try/except that records, rather than swallowing.
    assert source.count('_record_import_error(') >= 4
    assert 'except:' not in source, 'a bare except would discard the reason again'


# ---------------------------------------------------------------------------
# OpenCOR is the one backend that is not merely optional but *unshippable* (issue #436).
#
# `opencor` is provided by an OpenCOR install and is not on PyPI, so `pip install libcuflynx`
# has every solver except CVODE_opencor. Before this, asking for it gave
# `ModuleNotFoundError: No module named 'opencor'` -- or, on the implicit cellml default,
# `TypeError: 'NoneType' object is not callable` -- neither of which tells a user that the
# fix is one word in user_inputs.yaml.
#
# These tests force absence with monkeypatch rather than relying on the environment, so they
# assert the same thing under plain pytest (no OpenCOR) and under ./run_pytest.sh in an
# OpenCOR shell. They are deliberately NOT marked need_opencor.
# ---------------------------------------------------------------------------


@pytest.fixture
def opencor_absent(monkeypatch):
    """Make the OpenCOR backend look uninstalled, whatever this interpreter actually has."""
    monkeypatch.setattr(opencor_helper, 'oc', None)
    monkeypatch.setattr(opencor_helper, 'OPENCOR_IMPORT_ERROR',
                        "ModuleNotFoundError: No module named 'opencor'")


@pytest.mark.unit
def test_the_opencor_helper_module_imports_without_opencor():
    """The module must be importable in a pip install: the factory imports every backend
    eagerly, and so do a couple of plotting/utility modules. Only *using* it may fail."""
    assert opencor_helper.SimulationHelper is not None
    assert is_opencor_available() is (opencor_helper.oc is not None)


@pytest.mark.unit
def test_the_unavailable_message_names_the_drop_in_alternative(opencor_absent):
    message = opencor_unavailable_message()
    # the actual fix, not just the diagnosis
    assert 'CVODE_myokit' in message
    assert 'drop-in replacement' in message
    # why it is missing, so nobody goes looking for a pip package that does not exist
    assert 'PyPI' in message
    assert 'pip install' in message
    # and where the OpenCOR route is written down
    assert opencor_helper.OPENCOR_DOCS_URL in message
    assert 'getting-started' in message


@pytest.mark.unit
def test_requesting_cvode_opencor_without_opencor_explains_itself(opencor_absent):
    """Through the factory, which is where a user's `solver: CVODE_opencor` lands."""
    with pytest.raises(OpenCORUnavailableError) as excinfo:
        get_simulation_helper(model_path='m.cellml', solver='CVODE_opencor',
                              model_type='cellml', dt=0.01, sim_time=1.0)

    message = str(excinfo.value)
    assert 'CVODE_myokit' in message
    assert "No module named 'opencor'" in message
    assert 'PyPI' in message
    # RuntimeError is what get_simulation_helper documents for an unavailable backend, and
    # what test_solvers.py's skip_on_error path catches; the new type must not break that.
    assert isinstance(excinfo.value, RuntimeError)


@pytest.mark.unit
def test_the_implicit_cellml_default_explains_itself_too(opencor_absent):
    """Naming no solver at all still routes CellML models to OpenCOR, and used to die with
    `TypeError: 'NoneType' object is not callable` from inside the factory."""
    with pytest.raises(OpenCORUnavailableError) as excinfo:
        get_simulation_helper(model_path='m.cellml', solver=None,
                              model_type='cellml', dt=0.01, sim_time=1.0)

    assert 'CVODE_myokit' in str(excinfo.value)


@pytest.mark.unit
def test_the_refusal_does_not_fire_when_opencor_is_present(monkeypatch):
    """The failure mode to guard against: breaking real OpenCOR runs, which CI cannot catch
    because CI has no OpenCOR. Standing in a sentinel for the module is the closest an
    OpenCOR-less environment can get to asserting that require_opencor() is a no-op."""
    monkeypatch.setattr(opencor_helper, 'oc', object())
    assert is_opencor_available() is True
    assert require_opencor() is None  # no raise, no return value


@pytest.mark.unit
def test_a_broken_opencor_import_is_still_reported_as_broken(opencor_absent, monkeypatch):
    """`import opencor` failing for a reason other than absence must keep saying so (#410),
    rather than being flattened into the pip-install advice."""
    monkeypatch.setattr(opencor_helper, 'OPENCOR_IMPORT_ERROR',
                        'ImportError: libQt5Core.so.5: cannot open shared object file')
    assert 'libQt5Core.so.5' in opencor_unavailable_message()
