"""An unavailable solver backend must say *why* it is unavailable.

Every optional backend is imported under a try/except so a missing package does not break an
install that never uses it. The cost was that a genuinely-broken import looked exactly like a
missing one: both became ``None``, and both produced "X is not available".

That is not a cosmetic difference. A fresh CI runner where two MPI ranks import myokit at once
can race on creating ``~/.config/myokit/myokit.ini``, and the resulting message told us myokit
was not installed -- when it was, and the real error had already been discarded.
"""
import warnings

import pytest

from libcuflynx.solver_wrappers import (
    BACKEND_IMPORT_ERRORS, _record_import_error, _unavailable_message, get_simulation_helper)
from libcuflynx.solver_wrappers import libopencor_helper
from libcuflynx.solver_wrappers.libopencor_helper import (
    LibOpenCORUnavailableError, is_libopencor_available, libopencor_unavailable_message,
    require_libopencor)


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
# libOpenCOR (CVODE_libopencor) is an extra whose wheels need Python 3.12 or later, so asking for
# it without it installed must say how to install it and what to use instead.
#
# These tests force absence with monkeypatch, so they assert the same thing whether or not this
# interpreter has libopencor.
# ---------------------------------------------------------------------------


@pytest.fixture
def libopencor_absent(monkeypatch):
    """Make the libOpenCOR backend look uninstalled, whatever this interpreter actually has."""
    monkeypatch.setattr(libopencor_helper, 'oc', None)
    monkeypatch.setattr(libopencor_helper, 'LIBOPENCOR_IMPORT_ERROR',
                        "ModuleNotFoundError: No module named 'libopencor'")


@pytest.mark.unit
def test_the_libopencor_helper_module_imports_without_libopencor():
    """The factory imports every backend eagerly, so the module must import without
    libopencor. Only using it may fail."""
    assert libopencor_helper.SimulationHelper is not None
    assert is_libopencor_available() is (libopencor_helper.oc is not None)


@pytest.mark.unit
def test_the_unavailable_message_names_the_extra_and_the_alternative(libopencor_absent):
    message = libopencor_unavailable_message()
    assert 'pip install "libcuflynx[libopencor]"' in message
    assert '3.12' in message
    assert 'CVODE_myokit' in message


@pytest.mark.unit
def test_requesting_cvode_libopencor_without_libopencor_explains_itself(libopencor_absent):
    """Through the factory, which is where a user's `solver: CVODE_libopencor` lands."""
    with pytest.raises(LibOpenCORUnavailableError) as excinfo:
        get_simulation_helper(model_path='m.cellml', solver='CVODE_libopencor',
                              model_type='cellml', dt=0.01, sim_time=1.0)

    message = str(excinfo.value)
    assert "No module named 'libopencor'" in message
    assert 'libcuflynx[libopencor]' in message
    # RuntimeError is what get_simulation_helper documents for an unavailable backend.
    assert isinstance(excinfo.value, RuntimeError)


@pytest.mark.unit
def test_cvode_opencor_is_an_alias_for_cvode_libopencor():
    """CVODE_opencor resolves to CVODE_libopencor with one DeprecationWarning, wherever the
    name is given; it needs no libOpenCOR to resolve."""
    from libcuflynx.parsers.PrimitiveParsers import validate_solver_info
    from libcuflynx.parsers.solver_names import canonical_solver_name
    with pytest.warns(DeprecationWarning, match='solver CVODE_opencor now runs libOpenCOR; '
                                                'name it CVODE_libopencor'):
        assert canonical_solver_name('CVODE_opencor') == 'CVODE_libopencor'
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert canonical_solver_name('CVODE_libopencor') == 'CVODE_libopencor'
    with pytest.warns(DeprecationWarning):
        validate_solver_info('CVODE_opencor', {'solver': 'CVODE_opencor'})
    with pytest.warns(DeprecationWarning):
        try:
            get_simulation_helper(model_path='m.cellml', solver='CVODE_opencor',
                                  model_type='cellml', dt=0.01, sim_time=1.0)
        except Exception as exc:
            # Without libOpenCOR the factory reaches CVODE_libopencor's own error;
            # with it, the missing model file is the error.
            assert 'opencor' in str(exc).lower() or 'm.cellml' in str(exc)


@pytest.mark.unit
def test_the_implicit_cellml_default_is_myokit(monkeypatch):
    """Naming no solver routes a CellML model to CVODE_myokit, the schema's default."""
    from libcuflynx import solver_wrappers
    from libcuflynx.parsers.PrimitiveParsers import SOLVER_SCHEMA
    assert SOLVER_SCHEMA['default_solver_by_model_type']['cellml'] == 'CVODE_myokit'
    built = []
    monkeypatch.setattr(solver_wrappers, 'MyokitSimulationHelper',
                        lambda *args, **kwargs: built.append(args) or 'myokit')
    assert get_simulation_helper(model_path='m.cellml', solver=None, model_type='cellml',
                                 dt=0.01, sim_time=1.0) == 'myokit'


@pytest.mark.unit
def test_the_refusal_does_not_fire_when_libopencor_is_present(monkeypatch):
    monkeypatch.setattr(libopencor_helper, 'oc', object())
    assert is_libopencor_available() is True
    assert require_libopencor() is None


@pytest.mark.unit
def test_a_broken_libopencor_import_is_still_reported_as_broken(libopencor_absent, monkeypatch):
    """`import libopencor` failing for a reason other than absence must keep saying so (#410)."""
    monkeypatch.setattr(libopencor_helper, 'LIBOPENCOR_IMPORT_ERROR',
                        'ImportError: GLIBC_2.34 not found')
    assert 'GLIBC_2.34' in libopencor_unavailable_message()
