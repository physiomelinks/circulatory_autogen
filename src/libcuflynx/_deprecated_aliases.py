"""Machinery behind the deprecated flat top-level import names.

Before the rename every module in this project sat at the top level of ``src/`` and was
reached with a ``sys.path.insert(0, 'src')`` followed by ``from param_id.paramID import
CVS0DParamID``. 0.4.0 moved all of it under :mod:`libcuflynx`. The shim packages in
``src/<name>/`` keep the old spellings working for exactly one release; they are removed
in 0.5.0 (see :data:`REMOVAL_VERSION`).

Object identity is the whole difficulty
---------------------------------------
``param_id.paramID.CVS0DParamID`` has to *be* ``libcuflynx.param_id.paramID.CVS0DParamID``,
not a second class object that happens to have the same source. Otherwise user code that
did ``isinstance(x, CVS0DParamID)`` against the old name, or monkeypatched an attribute on
it, silently stops matching the object the package itself uses -- the worst kind of
deprecation, one that does not raise.

A shim written as ``from libcuflynx.param_id import *`` fails that: it creates a second
module object and copies references into it, so ``param_id.paramID`` is a different module
from ``libcuflynx.param_id.paramID`` even though the classes inside happen to be shared,
and anything imported lazily or added later is simply absent.

Nor is ``sys.modules['param_id'] = libcuflynx.param_id`` enough on its own. That fixes the
package object, but the *submodules* are then resolved through that package's ``__path__``
by the ordinary path finder, which loads ``paramID.py`` a second time under the name
``param_id.paramID`` -- a genuinely distinct module holding a genuinely distinct
``CVS0DParamID``.

So the shim does both:

1. :class:`_AliasFinder` goes on ``sys.meta_path``. ``sys.meta_path`` is consulted before
   the parent package's ``__path__``, so the finder gets first refusal on *submodules* of
   a root that was aliased and answers with a loader that hands back the module that is
   *already* imported under its ``libcuflynx.`` name rather than loading anything.
2. :func:`install_shim` rebinds ``sys.modules[<root>]`` to the real package, and is the
   one-line body of each ``src/<root>/__init__.py``. Those directories exist so that the
   old names are found at all (and so setuptools' ``packages.find`` ships them); the
   finder they install is what makes everything below them resolve correctly.

Scope: the finder claims as little as it can get away with
----------------------------------------------------------
``SHIM_ROOTS`` is eleven very ordinary words -- ``models``, ``scripts``, ``utilities``,
``checks``. A finder at ``sys.meta_path[0]`` that answered for any name whose root is in
that set would mean that importing *one* shim rebinds all eleven process-wide, so a
downstream project with its own ``utilities.py`` on ``sys.path`` would silently get ours
from the moment anything did ``import param_id``. Nothing announces that; it is the same
class of failure the identity work above exists to prevent.

So :meth:`_AliasFinder.find_spec` declines twice over. It declines the *root* name, because
the physical ``src/<root>/__init__.py`` shim is the correct resolver for it and the ordinary
path finder locates that -- respecting whatever else is earlier on ``sys.path``. And it
declines a submodule whose root was not actually aliased in this process, which is the only
case where first refusal is needed at all.

Exactly one :exc:`DeprecationWarning` is emitted per shim root per process -- not one per
attribute access and not one per submodule -- because :func:`_warn_once` records the roots
it has already warned about.
"""
import importlib
import importlib.util
import sys
import warnings
from importlib.machinery import ModuleSpec

#: Release that deletes these shims. 0.3.0 is already published, the rename ships in
#: 0.4.0, so users get the whole of 0.4.x to migrate. Stated in the warning text; keep the
#: release notes saying the same number.
REMOVAL_VERSION = "0.5.0"

#: The package the flat names now live in.
PACKAGE = "libcuflynx"

#: The flat top-level names that used to be importable. Every one of these has a matching
#: ``src/<name>/__init__.py`` shim and a matching ``src/libcuflynx/<name>/`` package.
#: Deliberately not the full list of subpackages: ``coupler``, ``solver1d``,
#: ``identifiabilty_analysis`` and ``obsolete`` were never imported flat by user code.
SHIM_ROOTS = frozenset({
    "checks",
    "emulators",
    "generators",
    "models",
    "param_id",
    "parsers",
    "protocol_runners",
    "scripts",
    "sensitivity_analysis",
    "solver_wrappers",
    "utilities",
})

# Roots already warned about in this process.
_warned = set()


def _warn_once(root):
    """Emit the deprecation warning for ``root``, at most once per process."""
    if root in _warned:
        return
    _warned.add(root)
    warnings.warn(
        "`{root}` is now `{pkg}.{root}`; this shim is removed in {version}. "
        "Import `{pkg}.{root}` instead.".format(
            root=root, pkg=PACKAGE, version=REMOVAL_VERSION
        ),
        DeprecationWarning,
        # The importing frame is buried under importlib's bookkeeping, so no stacklevel
        # points reliably at the user's import statement. The message names the module.
        stacklevel=3,
    )


def _reset_warnings():
    """Forget which roots have been warned about. For tests only."""
    _warned.clear()


class _AliasLoader:
    """Loader that "loads" an alias name by handing back an already-imported module.

    ``create_module`` returning an existing module object is exactly what makes the alias
    identical to the real thing rather than a copy of it.
    """

    def __init__(self, real_name):
        self._real_name = real_name
        self._real_spec = None

    def create_module(self, spec):
        module = importlib.import_module(self._real_name)
        # importlib.util.module_from_spec() overwrites __spec__ unconditionally (unlike
        # __name__/__loader__/__package__/__path__/__file__, which it leaves alone on an
        # already-initialised module). Left as-is, the real module would end up claiming
        # the alias name in its own spec, which breaks importlib.reload() and misleads
        # anything that introspects it. Stash it and put it back in exec_module().
        self._real_spec = getattr(module, "__spec__", None)
        return module

    def exec_module(self, module):
        # Nothing to execute: the module body ran when it was first imported under its
        # real name, and running it again would create fresh class objects -- the very
        # thing this shim exists to avoid.
        if self._real_spec is not None:
            module.__spec__ = self._real_spec


class _AliasFinder:
    """Resolve ``<root>[.<sub>...]`` to the already-imported ``libcuflynx.<root>...``."""

    def find_spec(self, fullname, path=None, target=None):
        root, dot, _ = fullname.partition(".")
        if root not in SHIM_ROOTS:
            return None
        if not dot:
            # The root itself. Declining is not a gap: the physical ``src/<root>/__init__.py``
            # shim is found by the ordinary path finder and calls install_shim() itself, so
            # ``import parsers`` still resolves to ``libcuflynx.parsers``. Claiming it here
            # instead would mean that importing *one* shim silently rebinds all eleven names
            # process-wide -- ``models``, ``scripts``, ``utilities``, ``checks`` are generic
            # enough that a downstream project has its own, and it would get ours.
            return None
        if not self._root_is_aliased(root):
            # A submodule of a root this process never aliased. Someone else's ``models``
            # package with a ``models.io`` inside it is theirs, not ours; only the path
            # finder can know where it lives.
            return None
        real_name = PACKAGE + "." + fullname
        try:
            real_spec = importlib.util.find_spec(real_name)
        except (ImportError, AttributeError, ValueError):
            # No such module under libcuflynx (or its parent is not a package). Decline,
            # so the ordinary finders produce the ordinary ModuleNotFoundError.
            return None
        if real_spec is None:
            return None
        _warn_once(root)
        return ModuleSpec(
            fullname,
            _AliasLoader(real_name),
            origin=real_spec.origin,
            is_package=real_spec.submodule_search_locations is not None,
        )

    @staticmethod
    def _root_is_aliased(root):
        """Did ``install_shim`` actually take ``root`` over in this process?

        ``sys.modules[root] is sys.modules['libcuflynx.' + root]`` is the state
        :func:`install_shim` leaves behind and nothing else produces, so it answers the
        question without a second piece of bookkeeping to keep in step. Both being absent
        must not read as "yes", hence the explicit ``None`` check.
        """
        real = sys.modules.get(PACKAGE + "." + root)
        return real is not None and sys.modules.get(root) is real

    def __repr__(self):
        return "<{} for the pre-{} flat import names>".format(
            type(self).__name__, PACKAGE
        )


def install_finder():
    """Put a single :class:`_AliasFinder` at the front of ``sys.meta_path``."""
    for finder in sys.meta_path:
        if isinstance(finder, _AliasFinder):
            return finder
    finder = _AliasFinder()
    # Ahead of the path finder, so that a submodule of an aliased root is never loaded a
    # second time off disk.
    sys.meta_path.insert(0, finder)
    return finder


def install_shim(alias):
    """Body of every ``src/<alias>/__init__.py``.

    Warns, installs the finder, and replaces the half-built shim module in
    ``sys.modules`` with the real package, so that ``import parsers`` yields
    ``libcuflynx.parsers`` itself.
    """
    if alias not in SHIM_ROOTS:
        raise ImportError(
            "{!r} is not a deprecated {} import name".format(alias, PACKAGE)
        )
    install_finder()
    _warn_once(alias)
    real = importlib.import_module(PACKAGE + "." + alias)
    # Rebinding sys.modules[__name__] from inside a module body is honoured by the import
    # system: _load() re-reads sys.modules after exec_module() and returns what it finds.
    sys.modules[alias] = real
    return real
