"""Where the tests find circulatory-autogen-modules, the module library they run against.

Tests about module libraries use the real one (physiomelinks/circulatory-autogen-modules)
rather than one written into a temporary directory, so they check libcuflynx against the
layouts, instances and supermodules a library actually has. CI checks the library out at the
commit pinned in ``tests/MODULE_LIBRARY_REF`` (a commit of its ``devel_module_tests`` branch
until that branch reaches ``main``). Move the pin deliberately, when libcuflynx should be
tested against a newer library.

The library's ``modules/`` directory is found from, in order:

* ``CUFLYNX_MODULE_LIBRARY`` -- the checkout, or its ``modules/`` directory; when it is set,
  nowhere else is looked;
* otherwise a ``circulatory-autogen-modules`` checkout beside this repository, or beside the
  main checkout when this is a git worktree.

Without one a test that needs it is skipped, saying where it looked -- unless
``CUFLYNX_REQUIRE_MODULE_LIBRARY`` is set (CI sets it), in which case it fails: a library
that silently went missing must not turn those tests into a green run that tested nothing.
"""

import os
import pathlib
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
LIBRARY_REPO = 'circulatory-autogen-modules'
REF_FILE = REPO_ROOT / 'tests' / 'MODULE_LIBRARY_REF'
ENV_DIR = 'CUFLYNX_MODULE_LIBRARY'
ENV_REQUIRE = 'CUFLYNX_REQUIRE_MODULE_LIBRARY'


def pinned_ref():
    """The library commit CI tests against."""
    for line in REF_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            return line
    raise ValueError(f'{REF_FILE} names no commit')


def _modules_dir(path):
    path = pathlib.Path(path).expanduser()
    if (path / 'modules').is_dir():
        return path / 'modules'
    return path if path.is_dir() and path.name == 'modules' else None


def _main_checkout():
    """The main checkout's directory when this is a git worktree (whose own parent is not
    where the sibling repositories live), else None."""
    try:
        common = subprocess.run(['git', 'rev-parse', '--git-common-dir'], cwd=REPO_ROOT,
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not common:
        return None
    common = pathlib.Path(common)
    if not common.is_absolute():
        common = REPO_ROOT / common
    return common.resolve().parent


def candidates():
    """Every place looked in, in order: ``[(description, path)]``.

    A set ``$CUFLYNX_MODULE_LIBRARY`` is the only one: when it names nothing usable, quietly
    testing against some other checkout (at some other commit) would hide that.
    """
    if os.environ.get(ENV_DIR):
        return [(f'${ENV_DIR}', pathlib.Path(os.environ[ENV_DIR]))]
    found = []
    found.append(('beside this repository', REPO_ROOT.parent / LIBRARY_REPO))
    main = _main_checkout()
    if main is not None and main != REPO_ROOT:
        found.append(('beside the main checkout', main.parent / LIBRARY_REPO))
    return found


def find_modules_dir():
    """The library's ``modules/`` directory, or None."""
    for _, path in candidates():
        modules = _modules_dir(path)
        if modules is not None:
            return modules
    return None


def modules_dir_or_skip():
    """The library's ``modules/`` directory; skips (or fails, when required) without one."""
    modules = find_modules_dir()
    if modules is not None:
        return modules
    looked = '; '.join(f'{what}: {path}' for what, path in candidates())
    message = (f'the module library ({LIBRARY_REPO}) was not found ({looked}). Set {ENV_DIR} '
               f'to a checkout of it, at commit {pinned_ref()} to match CI.')
    if os.environ.get(ENV_REQUIRE):
        pytest.fail(message + f' {ENV_REQUIRE} is set, so this is a failure, not a skip.')
    pytest.skip(message)
