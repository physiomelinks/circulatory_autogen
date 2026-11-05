"""Read ``pyproject.toml``, or fail -- never skip.

Several test modules assert things about the packaging metadata: the dependency lists, the
extras, ``requires-python``, the console entry points. All of them need a TOML parser, and
``tomllib`` only arrived in 3.11, so they each reached for ``pytest.importorskip('tomli')``.

``tomli`` is not declared anywhere in this project's own extras -- it was present only
because some version of pytest happened to pin it transitively. The moment that stops being
true, eleven assertions about ``pyproject.toml`` turn into eleven silent skips and a green
run stops meaning anything about the file they were written to protect. A missing parser is
a broken test environment, so it raises here and the tests error.

``[dev]`` now declares ``tomli`` for interpreters below 3.11, so a developer environment
built from this repository always has one.
"""
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"

_INSTALL_HINT = (
    "reading pyproject.toml needs tomllib (Python 3.11+) or tomli. Install the dev extra "
    "-- `pip install -e \".[dev]\"` -- which declares it."
)


def _parser():
    try:
        import tomllib
    except ImportError:                                     # Python < 3.11
        try:
            import tomli as tomllib
        except ImportError as exc:                          # pragma: no cover - env problem
            raise RuntimeError(_INSTALL_HINT) from exc
    return tomllib


def load_pyproject():
    """``pyproject.toml`` as a dict."""
    return _parser().loads(PYPROJECT.read_text(encoding="utf-8"))
