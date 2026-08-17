"""The documentation has to describe the package that actually ships (issue #437).

Two things in the docs rot silently, because nothing imports them and nothing runs them:

1. **Import instructions.** Before the package existed, every documented snippet appended
   ``src/`` to the interpreter's import path and imported flat module names (``import
   parsers``). Both stopped being true when the code moved under ``libcuflynx/``: the flat
   names survive only as deprecation shims that warn now and are gone in 0.5.0, and there is
   nothing to add to the import path at all. A doc that still teaches the old way sends every
   new reader down it.

2. **Lists of commands.** The entry-point table in ``CLAUDE.md`` and the one in ``README.md``
   restate ``[project.scripts]`` in prose. Adding, renaming or removing a console command
   changes ``pyproject.toml`` and leaves the tables behind.

These are cheap to check mechanically, so they are checked mechanically. No imports, no
subprocesses -- this reads files.
"""
import pathlib
import re

import pytest

from _pyproject import load_pyproject
from _tracked_files import only_tracked

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TUTORIAL = REPO_ROOT / "tutorial"
README = REPO_ROOT / "README.md"
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"

#: Text files under tutorial/ worth reading. Images and PDFs are skipped; notebooks are
#: included, since their code cells are documentation too.
_DOC_SUFFIXES = {".md", ".py", ".ipynb", ".yml", ".yaml", ".txt"}

#: The top-level module names that used to be importable directly off ``src/``.
_FLAT_PACKAGES = (
    "solver_wrappers",
    "param_id",
    "parsers",
    "generators",
    "protocol_runners",
    "sensitivity_analysis",
    "identifiabilty_analysis",
    "emulators",
    "models",
    "utilities",
    "scripts",
    "funcs_user",
)

_FLAT_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+(%s)(?:\.|\s+import)|import\s+(%s)\b)" % ("|".join(_FLAT_PACKAGES),
                                                              "|".join(_FLAT_PACKAGES)),
    re.MULTILINE,
)


def _doc_files():
    """The documentation git actually tracks.

    An unfiltered rglob also picks up ``tutorial/**/.ipynb_checkpoints/`` and the jupytext
    ``.py`` written beside a notebook when someone opens it -- gitignored artefacts, not
    documentation, and both carry the pre-rename import lines the sweeps below forbid. They
    fail the suite on whichever machine happens to have them and are invisible everywhere
    else, which is the least useful shape a test failure can have.
    """
    files = only_tracked(
        p for p in TUTORIAL.rglob("*") if p.is_file() and p.suffix in _DOC_SUFFIXES)
    files.append(README)
    return sorted(files)


def _rel(path):
    return str(path.relative_to(REPO_ROOT))


@pytest.mark.unit
def test_no_import_path_manipulation_in_tutorial_or_readme():
    """``grep -rn "sys.path" tutorial/ README.md`` must come back empty -- issue #437.

    An installed package is importable from anywhere. Any surviving ``sys.path`` line is
    either a leftover instruction to clone-and-append (wrong since the rename) or a snippet
    that only works from a checkout, which is the opposite of what a published package
    promises. Notebooks count: their code cells are copied verbatim by readers.
    """
    offenders = []
    for path in _doc_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "sys.path" in line:
                offenders.append("%s:%d: %s" % (_rel(path), lineno, line.strip()))
    assert not offenders, (
        "documentation still manipulates the import path; import from the libcuflynx "
        "namespace instead:\n" + "\n".join(offenders)
    )


@pytest.mark.unit
def test_documented_imports_use_the_libcuflynx_namespace():
    """No documented snippet may import a flat name (they warn now, and vanish in 0.5.0)."""
    offenders = []
    for path in _doc_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in _FLAT_IMPORT_RE.finditer(text):
            line = text[match.start():text.find("\n", match.start())].strip()
            lineno = text.count("\n", 0, match.start()) + 1
            offenders.append("%s:%d: %s" % (_rel(path), lineno, line))
    assert not offenders, (
        "documented imports must be prefixed with `libcuflynx.` -- the flat names are "
        "deprecation shims removed in 0.5.0:\n" + "\n".join(offenders)
    )


@pytest.mark.unit
def test_mkdocstrings_identifiers_are_package_qualified():
    """``::: parsers.ModelParsers`` no longer resolves; every identifier needs the package."""
    bad = []
    for path in sorted((TUTORIAL / "docs" / "api").glob("*.md")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.startswith(":::") and not line.split()[1].startswith("libcuflynx."):
                bad.append("%s:%d: %s" % (_rel(path), lineno, line.strip()))
    assert not bad, "mkdocstrings identifiers must start with `libcuflynx.`:\n" + "\n".join(bad)


def _project_scripts():
    """``{command: target}`` from ``[project.scripts]``.

    Not ``pytest.importorskip("tomli")``: tomli is not a declared dependency of this
    project, so that skipped these tests -- silently -- the moment pytest stopped pinning
    it transitively. See tests/_pyproject.py.
    """
    return load_pyproject().get("project", {}).get("scripts", {})


@pytest.mark.unit
@pytest.mark.parametrize("doc", ["CLAUDE.md", "README.md"])
def test_documented_console_commands_match_project_scripts(doc):
    """Every ``cuflynx-*`` name a doc mentions exists, and every declared one is documented.

    Both files carry a table of the pipeline commands. Without this they drift the moment an
    entry point is renamed -- and a command named in the docs that does not exist is worse
    than one that is missing, because the reader has no way to tell which of the two is wrong.
    """
    declared = set(_project_scripts())
    assert declared, "[project.scripts] is empty; nothing to document"

    text = (REPO_ROOT / doc).read_text(encoding="utf-8")
    mentioned = set(re.findall(r"\bcuflynx-[a-z0-9-]+", text))

    unknown = mentioned - declared
    assert not unknown, (
        "%s documents console commands that are not in [project.scripts]: %s"
        % (doc, sorted(unknown))
    )
    missing = declared - mentioned
    assert not missing, (
        "%s does not document these declared console commands: %s" % (doc, sorted(missing))
    )


@pytest.mark.unit
def test_readme_names_both_the_repository_and_the_package():
    """Someone searching for either name has to land here (issue #437)."""
    text = README.read_text(encoding="utf-8")
    assert "circulatory_autogen" in text and "libcuflynx" in text
    assert "pip install libcuflynx" in text, "the README must show the install command"
    # The relationship has to be stated, not left to be inferred from two names on one page.
    assert re.search(r"repository is .{0,10}circulatory_autogen", text), (
        "the README must say in as many words that the repository is circulatory_autogen "
        "and the package is libcuflynx"
    )


@pytest.mark.unit
def test_deprecation_removal_version_is_stated_consistently():
    """0.5.0 is the removal version; the docs a migrator reads must all say so."""
    removal = "0.5.0"
    for path in (README, CLAUDE_MD, REPO_ROOT / "CHANGELOG.md",
                 TUTORIAL / "docs" / "api" / "index.md"):
        text = path.read_text(encoding="utf-8")
        assert removal in text, (
            "%s should name %s as the release that removes the flat-import shims"
            % (_rel(path), removal)
        )
