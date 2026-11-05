"""``utilities/lazy_imports`` -- the wide-exception, cached, optional import.

The behaviour that matters here is the one an ``except ImportError`` does not have:
surviving an import that fails with something else. See the module under test for the
arviz race that made that distinction load-bearing.
"""
from __future__ import annotations

import sys

import pytest

from libcuflynx.utilities import lazy_imports


@pytest.fixture(autouse=True)
def _clear_cache():
    lazy_imports.reset_cache()
    yield
    lazy_imports.reset_cache()


def _install_module(monkeypatch, name, exc):
    """Make ``import <name>`` raise *exc*, the way a bad import-time side effect does."""
    real_import = lazy_imports.importlib.import_module

    def fake(mod_name, *args, **kwargs):
        if mod_name == name:
            raise exc
        return real_import(mod_name, *args, **kwargs)

    monkeypatch.setattr(lazy_imports.importlib, "import_module", fake)


def test_a_missing_module_is_none():
    assert lazy_imports.load_optional("a_module_that_does_not_exist_xyzzy") is None


def test_an_import_that_fails_with_something_other_than_importerror_is_survived(monkeypatch):
    """The whole point. arviz raised FileNotFoundError from its module body.

    An ``except ImportError`` here would let this through and kill the rank -- which is
    precisely what happened, so this is the assertion that encodes the bug.
    """
    _install_module(monkeypatch, "pretend_pkg",
                    FileNotFoundError(2, "No such file or directory",
                                      "/tmp/daily_warning.tmp"))
    assert lazy_imports.load_optional("pretend_pkg") is None


@pytest.mark.parametrize("exc", [
    RuntimeError("import-time side effect blew up"),
    OSError("read-only filesystem"),
    ValueError("bad cache contents"),
])
def test_any_ordinary_exception_is_survived(monkeypatch, exc):
    _install_module(monkeypatch, "pretend_pkg", exc)
    assert lazy_imports.load_optional("pretend_pkg") is None


def test_keyboardinterrupt_is_not_swallowed(monkeypatch):
    """``except Exception`` is wide on purpose, but not that wide.

    Catching BaseException would make Ctrl-C during a slow import do nothing visible.
    """
    _install_module(monkeypatch, "pretend_pkg", KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        lazy_imports.load_optional("pretend_pkg")


def test_strict_startup_reraises(monkeypatch):
    """CI wants the failure; a queued eight-hour job wants to keep going."""
    monkeypatch.setenv("CUFLYNX_STRICT_STARTUP", "1")
    _install_module(monkeypatch, "pretend_pkg", FileNotFoundError("gone"))
    with pytest.raises(FileNotFoundError):
        lazy_imports.load_optional("pretend_pkg")


def test_strict_startup_is_off_for_the_empty_string_and_zero(monkeypatch):
    for value in ("", "0"):
        monkeypatch.setenv("CUFLYNX_STRICT_STARTUP", value)
        assert lazy_imports.strict_startup() is False
    monkeypatch.setenv("CUFLYNX_STRICT_STARTUP", "1")
    assert lazy_imports.strict_startup() is True


def test_a_failure_is_cached_so_it_is_attempted_once(monkeypatch):
    calls = []
    real_import = lazy_imports.importlib.import_module

    def fake(mod_name, *args, **kwargs):
        if mod_name == "pretend_pkg":
            calls.append(mod_name)
            raise FileNotFoundError("gone")
        return real_import(mod_name, *args, **kwargs)

    monkeypatch.setattr(lazy_imports.importlib, "import_module", fake)
    assert lazy_imports.load_optional("pretend_pkg") is None
    assert lazy_imports.load_optional("pretend_pkg") is None
    assert calls == ["pretend_pkg"], "a broken optional import must not be retried per call"


def test_a_success_is_cached_and_returns_the_module():
    first = lazy_imports.load_optional("json")
    second = lazy_imports.load_optional("json")
    assert first is second is sys.modules["json"]


def test_the_warning_names_the_underlying_error(monkeypatch, caplog):
    """The log has to name the *third-party* failure, not just 'corner is missing'.

    Reading it as a missing install is what sent the original investigation looking in
    entirely the wrong place.
    """
    _install_module(monkeypatch, "pretend_pkg",
                    FileNotFoundError(2, "No such file or directory", "/x/daily_warning.tmp"))
    with caplog.at_level("WARNING", logger=lazy_imports.__name__):
        assert lazy_imports.load_optional("pretend_pkg") is None
    text = caplog.text
    assert "pretend_pkg" in text
    assert "FileNotFoundError" in text, f"the log did not name the real error:\n{text}"


def test_require_corner_raises_naming_what_failed(monkeypatch):
    monkeypatch.setattr(lazy_imports, "load_corner", lambda: None)
    with pytest.raises(ImportError) as excinfo:
        lazy_imports.require_corner("plot the thing")
    message = str(excinfo.value)
    assert "plot the thing" in message
    assert "CUFLYNX_STRICT_STARTUP" in message, (
        "the error should say how to see the underlying cause"
    )


def test_require_corner_returns_the_module_when_it_loads():
    corner = pytest.importorskip("corner")
    assert lazy_imports.require_corner("plot") is corner
