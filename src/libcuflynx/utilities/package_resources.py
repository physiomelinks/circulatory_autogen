"""Access to the data files shipped inside the ``libcuflynx`` package.

The CellML module library (``libcuflynx/generators/resources``) and the other non-Python
files that live inside the package are *package data*, not source, so they must be located
through :mod:`importlib.resources` rather than by walking up from ``__file__``. The
``__file__`` spelling only works when the package happens to be a plain directory sitting
next to the repo checkout; ``importlib.resources`` works for a normal wheel install, an
editable install and a zipimported distribution alike.

Two kinds of accessor are offered, because the consumers differ:

* :func:`package_data_file` (and its wrappers) returns a ``Traversable`` and is the right
  thing when a single file is read (``.read_text()`` / ``.open()``).
* :func:`package_data_dir` (and its wrappers) returns a **real filesystem directory path**.
  Several consumers need one: they ``os.listdir`` the module library, mix its entries into
  a single list with user-supplied directories, and hand the resulting strings to
  ``open()``, to ``shutil.copy2`` and to libCellML. Those cannot take a ``Traversable``.

:func:`package_data_dir` never stringifies a ``Traversable``: it goes through
:func:`importlib.resources.as_file`, whose context manager is held open for the life of the
process by a module-level :class:`~contextlib.ExitStack`. For a directory install (the
normal and the editable case) ``as_file`` yields the real directory and nothing is copied.
Only a zipimported distribution pays for an extraction, and only once.
"""

import atexit
import contextlib
import os
import pathlib
import shutil
import tempfile

# `files`/`as_file` are 3.9+, which is why pyproject declares requires-python = ">=3.9".
# The 3.7/3.8 fallback that used to sit here reached for `importlib_resources`, a package
# nothing declared as a dependency -- so on the interpreters it existed for, it raised.
from importlib.resources import as_file, files

# The package that owns the CellML module library, and the data directory inside it.
BUILTIN_MODULES_ANCHOR = 'libcuflynx.generators'
BUILTIN_MODULES_DIRNAME = 'resources'

# Keeps whatever `as_file` handed us alive until interpreter shutdown. Entering the context
# per call and leaving it immediately would delete an extracted directory while the caller
# still holds paths into it.
_resource_stack = contextlib.ExitStack()
atexit.register(_resource_stack.close)

_materialised_dirs = {}


def package_data_file(anchor, *parts):
    """A single shipped data file, as an :class:`importlib.abc.Traversable`.

    Use ``.read_text()`` / ``.read_bytes()`` / ``.open()`` on the result; only reach for
    :func:`package_data_dir` when a real path is unavoidable.
    """
    resource = files(anchor)
    for part in parts:
        resource = resource / part
    return resource


def package_data_dir(anchor, dirname):
    """Real filesystem path of a data directory shipped inside a package."""
    key = (anchor, dirname)
    if key not in _materialised_dirs:
        _materialised_dirs[key] = _materialise(package_data_file(anchor, dirname))
    return _materialised_dirs[key]


def builtin_modules_traversable():
    """The shipped CellML module library as a ``Traversable``."""
    return package_data_file(BUILTIN_MODULES_ANCHOR, BUILTIN_MODULES_DIRNAME)


def builtin_module_file(filename):
    """A single file of the shipped CellML module library, as a ``Traversable``."""
    return package_data_file(BUILTIN_MODULES_ANCHOR, BUILTIN_MODULES_DIRNAME, filename)


def builtin_modules_dir():
    """Real filesystem path of the shipped CellML module library.

    Prefer :func:`builtin_module_file` when a single file is being read. This exists for
    the consumers that must list the directory, or pass a path to something that only takes
    paths.
    """
    return package_data_dir(BUILTIN_MODULES_ANCHOR, BUILTIN_MODULES_DIRNAME)


def generator_template(filename):
    """A C++ template shipped next to the generators, as a ``Traversable``.

    ``cppGeneratorTemplateFunctions.cpp`` / ``main0dTemplate.cpp`` are read whole and never
    handed to anything that wants a path, so no real file is materialised for them.
    """
    return package_data_file(BUILTIN_MODULES_ANCHOR, filename)


def _materialise(traversable):
    try:
        return os.fspath(_resource_stack.enter_context(as_file(traversable)))
    except Exception:
        # `as_file` only learned to handle *directory* resources in Python 3.12, so on an
        # older interpreter a zipimported distribution lands here. Copy the (flat) directory
        # out by hand instead; it lives until interpreter shutdown, like the one `as_file`
        # would have produced.
        return _copy_out(traversable)


def _copy_out(traversable):
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix='libcuflynx-data-'))
    _resource_stack.callback(shutil.rmtree, tmpdir, ignore_errors=True)
    for entry in traversable.iterdir():
        if entry.is_file():
            (tmpdir / entry.name).write_bytes(entry.read_bytes())
    return os.fspath(tmpdir)
