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
import pathlib
import sys

import pytest

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src" / "libcuflynx"


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
    """Parse, don't import -- importing would need every optional dependency present."""
    try:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    except SyntaxError as exc:
        pytest.fail(
            f"{path.relative_to(_PACKAGE_ROOT.parents[1])}:{exc.lineno} is not valid syntax on "
            f"Python {sys.version_info.major}.{sys.version_info.minor}: {exc.msg}"
        )


def _wheel_modules():
    """The modules that actually ship: everything under libcuflynx except obsolete/,
    which pyproject's packages.find excludes from the wheel."""
    return [p for p in _shipped_modules() if "obsolete" not in p.parts]


@pytest.mark.unit
@pytest.mark.parametrize("path", _wheel_modules(), ids=lambda p: p.name)
def test_no_shipped_module_imports_distutils(path):
    """distutils was removed from the standard library in Python 3.12, and pyproject
    declares ``requires-python = ">=3.7"`` with no upper bound -- so a shipped module that
    imports it is guaranteed broken on a supported interpreter.

    The compile sweep above cannot catch this class of bug: ``from distutils import util``
    parses fine everywhere, and only fails when the module is *imported* -- which nothing
    in the suite does for every module. Scanning the AST for the import statement catches
    it without needing every optional dependency installed. (``src/libcuflynx/obsolete/``
    still imports distutils, and may: it is excluded from the wheel.)
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module] if node.module else []
        else:
            continue
        offenders = [n for n in names if n == "distutils" or n.startswith("distutils.")]
        if offenders:
            pytest.fail(
                f"{path.relative_to(_PACKAGE_ROOT.parents[1])}:{node.lineno} imports "
                f"{offenders}: distutils does not exist on Python >= 3.12, which "
                f"requires-python admits. Use shutil / libcuflynx.scripts._cli.boolean()."
            )
