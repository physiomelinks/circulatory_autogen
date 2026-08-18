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
* **No collateral damage.** The shim roots are eleven generic words and the finder that
  serves them sits at ``sys.meta_path[0]``. Nobody else's ``utilities`` or ``models`` may
  start resolving to ours because something in the process imported a deprecated name.
"""
import ast
import importlib
import importlib.util
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

from _tracked_files import only_tracked
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
    # time off disk under the alias name and identity is lost. Being first is safe only
    # because find_spec declines everything it is not entitled to -- see
    # test_a_shim_import_does_not_hijack_someone_elses_top_level_module below, which is the
    # assertion that stops this position from turning into a global name grab.
    from importlib.machinery import PathFinder

    assert sys.meta_path.index(finders[0]) < sys.meta_path.index(PathFinder)


@pytest.mark.unit
def test_find_spec_declines_the_root_name_itself():
    """The root is resolved by the physical ``src/<root>/__init__.py``, not by the finder.

    Claiming it here is what would make one shim import rebind all eleven generic names.
    """
    finder = _deprecated_aliases._AliasFinder()
    for root in sorted(SHIM_ROOTS):
        assert finder.find_spec(root) is None, root


@pytest.mark.unit
def test_find_spec_declines_a_submodule_of_a_root_this_process_never_aliased(fresh_shim):
    fresh_shim("emulators")
    finder = _deprecated_aliases._AliasFinder()
    # libcuflynx.emulators.emulator_trainer exists, so this is not declined for lack of a
    # target -- it is declined because nothing made `emulators` mean libcuflynx.emulators.
    assert importlib.util.find_spec("libcuflynx.emulators.emulator_trainer") is not None
    assert finder.find_spec("emulators.emulator_trainer") is None
    with pytest.warns(DeprecationWarning):
        importlib.import_module("emulators")
    assert finder.find_spec("emulators.emulator_trainer") is not None


@pytest.mark.unit
def test_a_shim_import_does_not_hijack_someone_elses_top_level_module(tmp_path):
    """``import param_id`` must not make `utilities` mean *our* utilities everywhere.

    The shim roots are eleven very ordinary words. A downstream project that has its own
    ``utilities.py`` (or ``models/``, or ``checks/``) earlier on ``sys.path`` keeps it --
    both before and after something in the process imports a deprecated flat name. The
    identity guarantee only ever needed first refusal on submodules of a root the shim
    actually took over, and that is all it takes now.

    Run in a subprocess: the meta-path finder and the ``sys.modules`` rebinding it does are
    process-global, so proving what a *fresh* interpreter sees needs a fresh interpreter.
    """
    (tmp_path / "utilities.py").write_text("MARKER = 'the user\\'s own module'\n", encoding="utf-8")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "models" / "LumpedModels.py").write_text(
        "MARKER = 'the user\\'s own submodule'\n", encoding="utf-8")

    program = textwrap.dedent("""
        import importlib.util, sys, warnings
        warnings.simplefilter('ignore', DeprecationWarning)
        sys.path.insert(0, %r)   # the user's tree, ahead of everything
        sys.path.append(%r)      # this checkout's src/, where the shims live

        before = importlib.util.find_spec('utilities').origin
        import param_id                      # installs the finder
        after = importlib.util.find_spec('utilities').origin
        assert before == after, 'importing a shim rebound `utilities`: %%s -> %%s' %% (before, after)

        import utilities
        assert utilities.MARKER == "the user's own module", utilities

        # ...including a *submodule* of a root name the shims also use, so long as this
        # process never aliased that root.
        import models.LumpedModels
        assert models.LumpedModels.MARKER == "the user's own submodule", models.LumpedModels

        # ...and the deprecated name it did import is still the real thing, not a copy.
        import param_id.paramID, libcuflynx.param_id.paramID
        assert param_id.paramID is libcuflynx.param_id.paramID
        assert param_id is libcuflynx.param_id
        print('ok')
    """) % (str(tmp_path), str(_SRC))

    # run_pytest.sh runs the suite under mpiexec, so this process carries PMI_*/OMPI_* in
    # its environment. A child that inherits them convinces mpi_utils.get_MPI() it was
    # launched too, and importing libcuflynx.param_id then initialises MPI outside any job
    # -- which aborts in MPI_Init_thread. Same strip as tests/test_console_entry_points.py.
    from libcuflynx.utilities.mpi_utils import LAUNCHER_ENV_VARS

    env = dict(os.environ)
    for var in LAUNCHER_ENV_VARS:
        env.pop(var, None)

    result = subprocess.run([sys.executable, "-c", program], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True, timeout=180)
    assert result.returncode == 0, result.stdout


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
    # Tracked files only. `tests/test_outputs/` is gitignored generation output -- the OMEX
    # and notebook tests write .py scripts into it that legitimately use the old flat names
    # from before the rename, and an unfiltered rglob then fails this test on any machine
    # that has ever run those tests, and on no other.
    for path in sorted(only_tracked(_TESTS.rglob("*.py"))):
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
