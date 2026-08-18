"""Nothing the engine writes may assume `/tmp` specifically.

On an HPC compute node `/tmp` is routinely small, node-local, purged mid-job, or not
writable by the user at all, and a user without root cannot change that. What they *can*
do is set ``TMPDIR`` -- so every temp path the engine uses has to come from
``tempfile.gettempdir()``, which reads ``TMPDIR`` (then ``TEMP``, then ``TMP``), rather
than from a hardcoded literal.

The flattened-CellML cache is the one that matters in practice: it is written before
every Myokit compile and read on every helper construction, so a `/tmp` that vanishes
mid-job takes the run with it.
"""
import os
import pathlib
import re
import tempfile

import pytest

_PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "src" / "libcuflynx"

#: A string literal naming the absolute path, as opposed to a `/tmp/...` fragment inside
#: a message or a docstring. Matches "'/tmp'" and '"/tmp/foo"'.
_HARDCODED_TMP = re.compile(r"""['"]/tmp(?:/|['"])""")


def _shipped_modules():
    return [p for p in sorted(_PACKAGE.rglob("*.py")) if "obsolete" not in p.parts]


@pytest.mark.unit
def test_the_sweep_finds_the_package():
    """Guard the guard: a bad path would make the sweep below vacuous."""
    assert len(_shipped_modules()) > 50


@pytest.mark.unit
@pytest.mark.parametrize("path", _shipped_modules(), ids=lambda p: p.name)
def test_no_shipped_module_hardcodes_the_tmp_directory(path):
    offenders = [
        (n, line.strip())
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if _HARDCODED_TMP.search(line) and not line.lstrip().startswith("#")
    ]
    assert not offenders, (
        f"{path.relative_to(_PACKAGE.parents[1])} hardcodes /tmp: "
        + "; ".join(f"line {n}: {text}" for n, text in offenders)
        + ". Use tempfile.gettempdir() so TMPDIR works -- it is the only lever a user "
          "has on a node where /tmp is unwritable, and it needs no privileges."
    )


@pytest.mark.unit
def test_the_flattened_cellml_cache_follows_tmpdir(monkeypatch, tmp_path):
    """The path an HPC run actually depends on."""
    from libcuflynx.solver_wrappers.myokit_helper import flattened_cellml_path

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("TMPDIR", str(scratch))
    tempfile.tempdir = None  # gettempdir() caches its answer; force a re-read
    try:
        cached = flattened_cellml_path("/some/resources/model.cellml")
        assert cached.startswith(str(scratch)), cached
        assert os.path.isdir(os.path.dirname(cached)), "the cache dir was not created"
    finally:
        tempfile.tempdir = None
