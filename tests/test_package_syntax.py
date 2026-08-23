"""Every shipped module must be syntactically valid on the oldest supported Python.

`pip install` byte-compiles the package at install time, so a module that only parses on a
newer interpreter is not a latent bug -- it is a noisy install and an ImportError the first
time anything reaches it. That is invisible while the tree is merely on ``sys.path``, because
nothing imports every module.

The case that motivated this: ``scripts/generate_modules_files.py`` reused double quotes
inside an f-string expression (``f"{d["key"]}"``), which PEP 701 legalised in 3.12 and which is
a SyntaxError on everything older. It sat in the tree undetected -- the file has no test and no
importer -- and the namespace move then swept it into the distribution.
"""
import ast
import pathlib
import re

import pytest

from _pyproject import load_pyproject

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src" / "libcuflynx"


def _oldest_supported_python():
    """``requires-python`` as a ``(major, minor)`` tuple -- the floor pip will install on."""
    spec = load_pyproject()["project"]["requires-python"]
    match = re.search(r">=\s*(\d+)\.(\d+)", spec)
    assert match, "requires-python = %r has no >= floor to test against" % (spec,)
    return int(match.group(1)), int(match.group(2))


#: Read from pyproject rather than assumed, so raising or lowering the floor moves this too.
OLDEST_SUPPORTED_PYTHON = _oldest_supported_python()


def _shipped_modules():
    return sorted(_PACKAGE_ROOT.rglob("*.py"))


@pytest.mark.unit
def test_package_root_exists():
    """Guard the guard: a bad path would make the sweep below vacuously pass."""
    assert _PACKAGE_ROOT.is_dir(), _PACKAGE_ROOT
    assert len(_shipped_modules()) > 50


@pytest.mark.unit
@pytest.mark.parametrize("path", _shipped_modules(), ids=lambda p: p.name)
def test_module_compiles(path):
    """Parse, don't import -- importing would need every optional dependency present.

    ``feature_version`` is what makes this about the *oldest* supported Python rather than
    whichever one the suite happens to be running on. Plain ``compile()`` accepted the
    match statements, ``X | None`` annotations and PEP 701 f-strings of a 3.11 test runner
    and said nothing about the oldest install that pyproject promises works.
    """
    try:
        ast.parse(path.read_text(encoding="utf-8"), str(path),
                  feature_version=OLDEST_SUPPORTED_PYTHON)
    except SyntaxError as exc:
        pytest.fail(
            f"{path.relative_to(_PACKAGE_ROOT.parents[1])}:{exc.lineno} is not valid syntax on "
            f"Python {OLDEST_SUPPORTED_PYTHON[0]}.{OLDEST_SUPPORTED_PYTHON[1]}, the floor "
            f"requires-python declares: {exc.msg}"
        )


def _wheel_modules():
    """The modules that actually ship: everything under libcuflynx except obsolete/,
    which pyproject's packages.find excludes from the wheel."""
    return [p for p in _shipped_modules() if "obsolete" not in p.parts]


@pytest.mark.unit
@pytest.mark.parametrize("path", _wheel_modules(), ids=lambda p: p.name)
def test_no_shipped_module_imports_distutils(path):
    """distutils was removed from the standard library in Python 3.12, and pyproject
    declares ``requires-python = ">=3.10"`` with no upper bound -- so a shipped module that
    imports it is guaranteed broken on a supported interpreter.

    The compile sweep above cannot catch this class of bug: ``from distutils import util``
    parses fine everywhere, and only fails when the module is *imported* -- which nothing
    in the suite does for every module. Scanning the AST for the import statement catches
    it without needing every optional dependency installed. (``src/libcuflynx/obsolete/``
    still imports distutils, and may: it is excluded from the wheel.)
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        for name in _imported_names(node):
            if name == "distutils" or name.startswith("distutils."):
                pytest.fail(
                    f"{path.relative_to(_PACKAGE_ROOT.parents[1])}:{node.lineno} imports "
                    f"{name}: distutils does not exist on Python >= 3.12, which "
                    f"requires-python admits. Use shutil / libcuflynx.scripts._cli.boolean()."
                )


def _imported_names(node):
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and not node.level:
        return [node.module] if node.module else []
    return []


#: The stdlib backports a package reaches for when it claims to support an interpreter older
#: than the feature it uses. Each is a *third-party distribution*: importing one without
#: declaring it is an install that resolves and then fails on first use.
#: (`dataclasses` is not here: it is stdlib from 3.6, so the PyPI package of that name is
#: only a backport below this project's floor and an import of it is the stdlib module.)
_BACKPORTS = ("importlib_resources", "importlib_metadata", "typing_extensions")


def _declared_distributions():
    project = load_pyproject()["project"]
    declared = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        declared.extend(extra)
    # 'importlib_resources>=5 ; python_version < "3.9"' -> 'importlib_resources'
    return {re.split(r"[\s<>=!~;\[]", req, 1)[0].replace("-", "_").lower()
            for req in declared if req}


@pytest.mark.unit
@pytest.mark.parametrize("path", _wheel_modules(), ids=lambda p: p.name)
def test_no_shipped_module_imports_an_undeclared_backport(path):
    """A version-guarded fallback import still needs the fallback to be a dependency.

    ``package_resources.py`` used to answer ImportError from ``importlib.resources`` with
    ``from importlib_resources import as_file, files``. That import only ever ran on 3.7 and
    3.8 -- the two interpreters ``requires-python = ">=3.7"`` admitted and nothing declared
    ``importlib_resources`` for. So pip installed happily on exactly the versions the
    fallback existed to serve, and the first generator import died. The floor is 3.10 now and
    the fallback is gone; this stops the pattern coming back under a different name.
    """
    declared = _declared_distributions()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        for name in _imported_names(node):
            root = name.partition(".")[0]
            if root in _BACKPORTS and root.lower() not in declared:
                pytest.fail(
                    f"{path.relative_to(_PACKAGE_ROOT.parents[1])}:{node.lineno} imports the "
                    f"backport {root!r}, which no dependency or extra in pyproject.toml "
                    f"declares. Either declare it or raise requires-python past the version "
                    f"that needs it."
                )
