"""Ask git which files are actually part of the repository.

Several sweeps in this suite walk the tree with ``glob``/``rglob`` and assert something
about every file they find. Run on a machine that has ever executed the suite, that also
finds generated artefacts -- ``tests/test_outputs/**.py`` written by the OMEX and notebook
tests, ``.ipynb_checkpoints/``, the jupytext ``.py`` beside a tutorial notebook, a local
launcher a developer keeps in ``user_run_files/``. All of them are gitignored, none of
them is under review, and every one of them can fail an assertion about the repository's
contents. That is a false negative for the developer who happens to have them and a check
that passes in CI for the wrong reason.

``git ls-files`` is the authority on what is in the repository, so that is what these
sweeps ask. Outside a git work tree (an exported tarball) there is nothing to filter
against and the caller's own listing is returned unchanged -- there are no artefacts there
either.
"""
import pathlib
import subprocess

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _tracked_paths():
    """Every path git tracks, as absolute ``Path``s, or ``None`` outside a work tree."""
    try:
        result = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "ls-files", "-z"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return {
        _REPO_ROOT / name.decode("utf-8")
        for name in result.stdout.split(b"\0")
        if name
    }


_TRACKED = _tracked_paths()


def only_tracked(paths):
    """Filter ``paths`` down to the ones git tracks (a no-op outside a work tree)."""
    if _TRACKED is None:
        return list(paths)
    return [path for path in paths if pathlib.Path(path).resolve() in _TRACKED]
