"""The flat top-level names keep working for one release, loudly and *identically*.

``from param_id.paramID import CVS0DParamID`` was how every user script, notebook and
tutorial reached this code before the ``libcuflynx`` rename. The shims in ``src/<name>/``
buy those scripts one release. Two things have to hold, and both fail silently if we get
them wrong:

* **Identity.** ``param_id.paramID`` must be the *same module object* as
  ``libcuflynx.param_id.paramID``. If it is a second copy, a user's
  ``isinstance(x, CVS0DParamID)`` stops matching and their monkeypatch stops taking
  effect, with no error anywhere.
* **Exactly one warning per shim.** One per attribute access, or one per submodule, turns
  a migration notice into noise people filter out.
"""
import ast
import importlib
import pathlib
import sys

import pytest

from libcuflynx import _deprecated_aliases
from libcuflynx._deprecated_aliases import PACKAGE, REMOVAL_VERSION, SHIM_ROOTS

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
_TESTS = pathlib.Path(__file__).resolve().parent


def _forget(root):
    """Undo every trace of a previous import of ``root``, so it warns again.

    Only the alias entries are dropped -- the real ``libcuflynx.*`` modules stay imported,
    which is exactly the situation these shims have to cope with in a live session.
    """
    for name in [n for n in sys.modules if n == root or n.startswith(root + ".")]:
        del sys.modules[name]
    _deprecated_aliases._warned.discard(root)


@pytest.fixture
def fresh_shim():
    """Hand out shim roots that have not been imported yet in this process."""
    used = []

    def _use(root):
        _forget(root)
        used.append(root)
        return root

    yield _use
    for root in used:
        _forget(root)


@pytest.mark.unit
def test_shim_warns_once_naming_the_new_path_and_the_removal_version(fresh_shim):
    root = fresh_shim("param_id")
    with pytest.warns(DeprecationWarning) as records:
        importlib.import_module(root)
    messages = [str(r.message) for r in records if "this shim is removed" in str(r.message)]
    assert len(messages) == 1, messages
    assert "libcuflynx.param_id" in messages[0]
    assert REMOVAL_VERSION in messages[0]
    # 0.3.0 is already published, so it could never have been the removal version; the
    # rename ships in 0.4.0 and the shims go in the release after it.
    assert REMOVAL_VERSION == "0.5.0"


@pytest.mark.unit
def test_a_second_import_of_the_same_shim_is_silent(fresh_shim, recwarn):
    root = fresh_shim("parsers")
    with pytest.warns(DeprecationWarning, match="this shim is removed"):
        importlib.import_module(root)
    # Same process, everything already cached: re-importing the package, importing a
    # submodule, and reaching attributes must all stay quiet.
    recwarn.clear()
    module = importlib.import_module(root)
    importlib.import_module("parsers.PrimitiveParsers")
    module.PrimitiveParsers  # noqa: B018 -- attribute access must not warn either
    assert [str(w.message) for w in recwarn if "this shim is removed" in str(w.message)] == []


@pytest.mark.unit
def test_importing_a_submodule_first_still_warns_exactly_once(fresh_shim):
    fresh_shim("utilities")
    with pytest.warns(DeprecationWarning) as records:
        importlib.import_module("utilities.utility_funcs")
    messages = [str(r.message) for r in records if "this shim is removed" in str(r.message)]
    assert len(messages) == 1, messages


@pytest.mark.unit
def test_shim_root_is_the_real_package_not_a_copy(fresh_shim):
    root = fresh_shim("param_id")
    with pytest.warns(DeprecationWarning):
        shim = importlib.import_module(root)
    real = importlib.import_module("libcuflynx.param_id")
    assert shim is real
    # ...and the real package still knows its own name. A loader that let
    # module_from_spec() rewrite __spec__ would leave it claiming to be `param_id`.
    assert real.__name__ == "libcuflynx.param_id"
    assert real.__spec__.name == "libcuflynx.param_id"


@pytest.mark.unit
def test_every_spelling_of_a_submodule_resolves_to_one_module_object(fresh_shim):
    fresh_shim("param_id")
    real = importlib.import_module("libcuflynx.param_id.paramID")

    with pytest.warns(DeprecationWarning):
        import param_id.paramID  # noqa: F401 -- the import statement is the subject
    from param_id.paramID import CVS0DParamID

    assert sys.modules["param_id.paramID"] is real
    assert param_id.paramID is real
    assert importlib.import_module("param_id.paramID") is real
    # The point of all of it: user isinstance checks and monkeypatches keep matching.
    assert CVS0DParamID is real.CVS0DParamID
    assert real.__spec__.name == "libcuflynx.param_id.paramID"


@pytest.mark.unit
@pytest.mark.parametrize("root", sorted(SHIM_ROOTS))
def test_each_documented_root_aliases_its_libcuflynx_package(root, fresh_shim):
    fresh_shim(root)
    with pytest.warns(DeprecationWarning, match=r"this shim is removed in " + REMOVAL_VERSION):
        shim = importlib.import_module(root)
    assert shim is importlib.import_module("{}.{}".format(PACKAGE, root))


@pytest.mark.unit
@pytest.mark.parametrize("root", sorted(SHIM_ROOTS))
def test_each_root_ships_a_shim_package(root):
    """The shims must be real packages under src/, or the wheel will not carry them."""
    assert (_SRC / root / "__init__.py").is_file()


@pytest.mark.unit
def test_a_missing_submodule_still_raises_module_not_found(fresh_shim):
    fresh_shim("checks")
    with pytest.warns(DeprecationWarning):
        importlib.import_module("checks")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("checks.no_such_module")


@pytest.mark.unit
def test_only_one_finder_is_ever_installed(fresh_shim):
    for root in ("models", "generators"):
        fresh_shim(root)
        with pytest.warns(DeprecationWarning):
            importlib.import_module(root)
    finders = [f for f in sys.meta_path if isinstance(f, _deprecated_aliases._AliasFinder)]
    assert len(finders) == 1
    # Ahead of the path finder, or a submodule of an aliased root gets loaded a second
    # time off disk under the alias name and identity is lost.
    from importlib.machinery import PathFinder

    assert sys.meta_path.index(finders[0]) < sys.meta_path.index(PathFinder)


@pytest.mark.unit
def test_the_package_never_imports_its_own_deprecated_names():
    """A shim reached from inside libcuflynx would warn users about their own dependency.

    It would also mean the package depends on the shims surviving past 0.5.0.
    """
    offenders = []
    for path in sorted((_SRC / PACKAGE).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.partition(".")[0] in SHIM_ROOTS:
                    offenders.append("{}:{}: {}".format(path, node.lineno, name))
    assert offenders == []


@pytest.mark.unit
def test_the_test_suite_never_imports_the_deprecated_names_either():
    """The same rule for `tests/`, because a flat import can arrive without a conflict.

    Merging upstream into the packaging branch walked one back in: a test added on master
    used a function-local ``from parsers.PrimitiveParsers import ObsAndParamDataParser``,
    and since this branch had never touched those lines git auto-merged it cleanly. Its two
    siblings were fixed by hand only because git happened to raise them as conflicts. That
    asymmetry recurs on every merge from master while the shims exist, and the package-only
    sweep above cannot see it.

    Exercising the shims is this file's job, so this file is the one exemption. Everything
    else must import ``libcuflynx.*`` -- including function-local imports, which is why this
    walks the whole AST rather than reading the top of each file.
    """
    offenders = []
    for path in sorted(_TESTS.rglob("*.py")):
        if path.resolve() == pathlib.Path(__file__).resolve():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.partition(".")[0] in SHIM_ROOTS:
                    offenders.append("{}:{}: {}".format(path, node.lineno, name))
    assert offenders == [], (
        "these import circulatory_autogen through a deprecated flat name, which is removed "
        "in {}; import the libcuflynx.* path instead:\n  ".format(
            _deprecated_aliases.REMOVAL_VERSION) + "\n  ".join(offenders))
