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
