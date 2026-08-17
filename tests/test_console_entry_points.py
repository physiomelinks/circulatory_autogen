"""The console commands declared in ``[project.scripts]`` must actually exist and answer.

A `[project.scripts]` entry is a string. Nothing checks it at build time, nothing checks it
at install time -- setuptools writes a launcher that imports the named module and calls the
named attribute, and the first anyone hears about a typo, a renamed module or a missing
``main`` is a traceback from ``cuflynx-param-id`` (or, worse, from rank 3 of an MPI job that
had already been queued for an hour).

So the table below is read out of ``pyproject.toml`` rather than restated here: every entry
point that exists is tested, and a new one cannot be added without being covered.

Each target is exercised in a subprocess. In-process would import six stage modules into the
pytest session -- opening MPI, matplotlib and the solver backends as a side effect -- and
``--help`` calls ``sys.exit``, so the isolation is worth the process spawn.
"""
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

from _tracked_files import only_tracked

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_SRC_DIR = _REPO_ROOT / "src"

#: The commands issue #434 specifies. Kept alongside the parsed table, not instead of it:
#: the parsed table catches a broken entry point, this catches a *deleted* one.
EXPECTED_ENTRY_POINTS = {
    "cuflynx-generate",
    "cuflynx-param-id",
    "cuflynx-sequential-param-id",
    "cuflynx-sensitivity",
    "cuflynx-identifiability",
    "cuflynx-train-emulator",
    "cuflynx-plot",
}


def _parse_project_scripts():
    """``{command: "module:attr"}`` from ``[project.scripts]``.

    tomllib is 3.11+, and this project supports 3.9, so fall back to reading the one
    table by hand. The fallback is deliberately strict about the shape it accepts --
    a line it cannot parse is a test failure, not a silently skipped entry point.
    """
    text = _PYPROJECT.read_text(encoding="utf-8")
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            tomllib = None

    if tomllib is not None:
        return tomllib.loads(text).get("project", {}).get("scripts", {})

    block = re.search(r"^\[project\.scripts\]\s*$(.*?)(?=^\[|\Z)", text,
                      re.MULTILINE | re.DOTALL)
    assert block, "no [project.scripts] table in pyproject.toml"
    scripts = {}
    for line in block.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r'^([A-Za-z0-9._-]+)\s*=\s*"([^"]+)"$', line)
        assert match, f"unparsable [project.scripts] line: {line!r}"
        scripts[match.group(1)] = match.group(2)
    return scripts


PROJECT_SCRIPTS = _parse_project_scripts()


def _subprocess_env():
    """A clean environment for the child: this tree on the path, no launcher pretence.

    ``run_pytest.sh`` runs the suite under ``mpiexec``, so the pytest process carries
    ``PMI_*``/``OMPI_*`` in its environment. A child that inherits them convinces
    :func:`libcuflynx.utilities.mpi_utils.get_MPI` that it was launched too, and it
    imports the real mpi4py and initialises MPI outside any job. Stripping them makes
    these tests behave the same whichever way the suite was started.
    """
    from libcuflynx.utilities.mpi_utils import LAUNCHER_ENV_VARS

    env = dict(os.environ)
    for var in LAUNCHER_ENV_VARS:
        env.pop(var, None)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(_SRC_DIR) + (os.pathsep + existing if existing else "")
    return env


def _run(code_or_args, timeout=180):
    return subprocess.run(
        [sys.executable] + code_or_args,
        env=_subprocess_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, timeout=timeout, cwd=str(_REPO_ROOT),
    )


@pytest.mark.unit
def test_expected_entry_points_are_declared():
    """Guard the guard: the sweeps below are vacuous if the table is empty or shrinks."""
    assert set(PROJECT_SCRIPTS) == EXPECTED_ENTRY_POINTS, (
        "[project.scripts] no longer matches the commands issue #434 specifies; update "
        "EXPECTED_ENTRY_POINTS deliberately if a command was genuinely added or removed"
    )


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(PROJECT_SCRIPTS))
def test_entry_point_target_is_spelled_correctly(name):
    """``module:attr``, with the module inside the shipped package."""
    target = PROJECT_SCRIPTS[name]
    assert target.count(":") == 1, f"{name} = {target!r} is not 'module:attr'"
    module, attr = target.split(":")
    assert module.startswith("libcuflynx."), (
        f"{name} points at {module!r}, which is outside the installed package"
    )
    assert attr.isidentifier(), f"{name} points at attribute {attr!r}"
    relative = pathlib.Path(*module.split(".")).with_suffix(".py")
    assert (_SRC_DIR / relative).is_file(), f"{name} points at a module that is not in the tree"


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(PROJECT_SCRIPTS))
def test_entry_point_resolves_to_a_callable_main(name):
    """Import the module and fetch the attribute, exactly as the generated launcher does."""
    module, attr = PROJECT_SCRIPTS[name].split(":")
    result = _run([
        "-c",
        "import importlib, sys\n"
        f"module = importlib.import_module({module!r})\n"
        f"target = getattr(module, {attr!r}, None)\n"
        "if target is None:\n"
        f"    sys.exit('{module}.{attr} does not exist')\n"
        "if not callable(target):\n"
        f"    sys.exit('{module}.{attr} is not callable')\n"
    ])
    assert result.returncode == 0, f"{name}: {result.stdout}"


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(PROJECT_SCRIPTS))
def test_entry_point_help_exits_zero(name):
    """``--help`` has to answer before any configuration file is opened.

    It is the one thing a user can run to find out whether the install worked, and it has
    to work from a wheel with no repository present -- so it must not need
    ``user_inputs.yaml``, a model, or MPI.
    """
    module = PROJECT_SCRIPTS[name].split(":")[0]
    result = _run(["-m", module, "--help"])
    assert result.returncode == 0, f"{name} --help exited {result.returncode}:\n{result.stdout}"
    assert "usage:" in result.stdout, f"{name} --help printed no usage line:\n{result.stdout}"
    assert "user_inputs.yaml" in result.stdout, (
        f"{name} --help should say where its configuration comes from:\n{result.stdout}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(PROJECT_SCRIPTS))
def test_entry_point_rejects_unknown_options(name):
    """A typo'd flag must fail, not be silently ignored while a long run starts."""
    module = PROJECT_SCRIPTS[name].split(":")[0]
    result = _run(["-m", module, "--definitely-not-an-option"])
    assert result.returncode != 0, f"{name} accepted an unknown option:\n{result.stdout}"


# --- the shell launchers -------------------------------------------------------------

_LAUNCHER_DIR = _REPO_ROOT / "user_run_files"

#: Scripts in user_run_files/ that are *sourced*, not run. Everything else there is a
#: launcher, and the launcher list below is derived rather than written out -- a hardcoded
#: list is exactly how `plot_param_id_without_autogen.sh` went on invoking
#: `../src/scripts/plot_param_id_script.py` (deleted in the rename) for the whole of this
#: change while its converted sibling `plot_param_id.sh` was tested.
_SOURCED_HELPERS = {
    "cuflynx_entry_point.sh",
    "python_path.sh",
    "opencor_pythonshell_path.sh",
    "load_mpi.sh",
}

#: Tracked files only: a developer's own throwaway launcher sitting in user_run_files/ is
#: not the repository's problem, and asserting things about it fails the suite on one
#: machine for reasons no reviewer can see.
_SHELL_SCRIPTS = sorted(p.name for p in only_tracked(_LAUNCHER_DIR.glob("*.sh")))

LAUNCHERS = [name for name in _SHELL_SCRIPTS if name not in _SOURCED_HELPERS]

#: launcher -> command it must invoke, for the pipeline stages CLAUDE.md documents. The
#: derived sweeps below catch a launcher that invokes *anything* by path; this catches one
#: that quietly stops driving the stage it is named after.
PIPELINE_LAUNCHERS = {
    "run_autogeneration.sh": "cuflynx-generate",
    "run_autogeneration_with_id_params.sh": "cuflynx-generate",
    "run_param_id.sh": "cuflynx-param-id",
    "run_sequential_param_id.sh": "cuflynx-sequential-param-id",
    "run_sensitivity_analysis.sh": "cuflynx-sensitivity",
    "run_identifiability_analysis.sh": "cuflynx-identifiability",
    "run_emulator_training.sh": "cuflynx-train-emulator",
    "plot_param_id.sh": "cuflynx-plot",
    "plot_param_id_without_autogen.sh": "cuflynx-plot",
}


def _executable_text(name):
    """A shell script's lines with the comments dropped.

    ``cuflynx_entry_point.sh`` quotes the old path-based invocation in its own header to
    say what it replaced, and the module-route comment names the three scripts that have no
    console command. Neither is executed, and neither should fail the sweeps below.
    """
    text = (_LAUNCHER_DIR / name).read_text(encoding="utf-8")
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


@pytest.mark.unit
def test_the_launcher_list_is_not_empty_or_stale():
    """Guard the guard: a bad glob would make every sweep below vacuous."""
    assert len(LAUNCHERS) >= 10, LAUNCHERS
    assert set(PIPELINE_LAUNCHERS) <= set(LAUNCHERS), (
        set(PIPELINE_LAUNCHERS) - set(LAUNCHERS)
    )


@pytest.mark.unit
@pytest.mark.parametrize("launcher", _SHELL_SCRIPTS)
def test_no_shell_script_reaches_into_the_source_tree(launcher):
    """No `../src/` anywhere. A wheel install has no such directory.

    The previous spelling of this assertion was ``"src/libcuflynx/scripts" not in text`` --
    the *post*-rename path, which sails straight past the pre-rename ``../src/scripts/``
    form that two launchers were still using and that has not existed in the tree since the
    package move.
    """
    text = _executable_text(launcher)
    assert re.search(r"\.\./src/", text) is None, (
        f"{launcher} invokes something by a path into src/, which stops working the moment "
        "libcuflynx is installed anywhere but a checkout"
    )


@pytest.mark.unit
@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_launcher_resolves_its_command_through_the_helper(launcher):
    """Either a console entry point or ``python -m``, and never python_path.sh."""
    text = _executable_text(launcher)
    if "cuflynx_entry_point.sh" not in text:
        # A launcher that runs no Python at all (gnuplot_fitting_convergence.sh) has
        # nothing to resolve; it just must not be reaching into src/, checked above.
        assert "libcuflynx" not in text, (
            f"{launcher} names libcuflynx but does not source cuflynx_entry_point.sh"
        )
        return
    assert re.search(r"\brequire_cuflynx(_module)?\b", text), (
        f"{launcher} sources the helper but never calls require_cuflynx/require_cuflynx_module"
    )
    assert '${cuflynx_cmd[@]}' in text, (
        f"{launcher} never runs the command require_cuflynx resolved"
    )
    assert 'source python_path.sh' not in text, (
        f"{launcher} still sources python_path.sh; that file is for the OpenCOR route only "
        "and a pip user has no reason to have configured it"
    )


@pytest.mark.unit
@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_every_module_a_launcher_names_exists(launcher):
    """``require_cuflynx_module libcuflynx.scripts.foo`` must name a real module.

    Nothing resolves these until the launcher is run, and the three utilities that take
    this route have no entry point to be caught by the table-driven tests above.
    """
    modules = re.findall(r"\blibcuflynx(?:\.[A-Za-z0-9_]+)+", _executable_text(launcher))
    for module in modules:
        relative = pathlib.Path(*module.split(".")).with_suffix(".py")
        assert (_SRC_DIR / relative).is_file(), f"{launcher} names {module}, which is not in the tree"


@pytest.mark.unit
@pytest.mark.parametrize("launcher,command", sorted(PIPELINE_LAUNCHERS.items()))
def test_launcher_calls_the_entry_point(launcher, command):
    text = _executable_text(launcher)
    assert command in text, f"{launcher} does not mention {command}"


@pytest.mark.unit
@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_launcher_is_valid_bash(launcher):
    result = subprocess.run(["bash", "-n", str(_LAUNCHER_DIR / launcher)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True)
    assert result.returncode == 0, result.stdout


@pytest.mark.unit
def test_launcher_reports_a_missing_install_before_launching_mpi():
    """Without libcuflynx installed the launcher must say so, not hand the problem to MPI.

    A bare system PATH stands in for the developer who cloned the repository and went
    straight to ``./run_param_id.sh 2``: an ordinary shell, no virtualenv activated, no
    ``pip install -e .``. What has to come back is a sentence about installing the package,
    printed before ``mpiexec`` is reached -- not an ImportError from inside a rank.
    """
    bash = shutil.which("bash")
    assert bash, "bash is needed to run the launchers at all"
    system_path = "/usr/bin:/bin"
    if shutil.which("cuflynx-param-id", path=system_path):
        pytest.skip("libcuflynx is installed system-wide, so 'not installed' cannot be staged")

    result = subprocess.run(
        [bash, "run_param_id.sh", "2"],
        cwd=str(_LAUNCHER_DIR),
        env={"PATH": system_path, "HOME": os.environ.get("HOME", "/tmp")},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, timeout=60,
    )
    assert result.returncode != 0
    assert "cuflynx-param-id" in result.stdout
    assert "pip install -e ." in result.stdout
    assert "Traceback" not in result.stdout
    assert "mpiexec" not in result.stdout, "the launcher got as far as MPI before complaining"
