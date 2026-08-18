"""An unavailable solver backend must say *why* it is unavailable.

Every optional backend is imported under a try/except so a missing package does not break an
install that never uses it. The cost was that a genuinely-broken import looked exactly like a
missing one: both became ``None``, and both produced "X is not available".

That is not a cosmetic difference. A fresh CI runner where two MPI ranks import myokit at once
can race on creating ``~/.config/myokit/myokit.ini``, and the resulting message told us myokit
was not installed -- when it was, and the real error had already been discarded.
"""
import pytest

from solver_wrappers import (
    BACKEND_IMPORT_ERRORS, _record_import_error, _unavailable_message, get_simulation_helper)


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
    import solver_wrappers

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

    import solver_wrappers

    source = inspect.getsource(solver_wrappers)
    # Each optional import is a try/except that records, rather than swallowing.
    assert source.count('_record_import_error(') >= 4
    assert 'except:' not in source, 'a bare except would discard the reason again'
