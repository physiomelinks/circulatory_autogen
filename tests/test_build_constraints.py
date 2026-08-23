"""Every CI install that reaches the ``emulation`` extra has to carry the build ceiling.

``autoemulate`` depends on ``harmonic``, which ships wheels for cp311+ only, so the
Python 3.10 jobs build it from its sdist. Cython 3.3.0 made an implicit
``double -> long`` assignment an error and harmonic 1.3.0's ``model_legacy.pyx``
does exactly that, so those installs started failing outright -- in 20 seconds, at
dependency install, on branches that had touched nothing related.

``build-constraints.txt`` holds Cython below 3.3 for the *build* environments pip
creates for sdists. Nothing in these tests can stop harmonic from breaking again;
they stop the ceiling from being quietly dropped, and stop a new job that installs
the extra from being added without it.

The workflow is parsed as YAML rather than grepped, because the property is "this
step installs the extra *and this same step* passes the flag", and a regex cannot
tell a step's script from the comment above the next one.
"""
import pathlib

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONSTRAINTS = REPO_ROOT / "build-constraints.txt"
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

#: The extra that reaches harmonic. Anything installing it needs the ceiling.
EMULATION_EXTRA = "emulation"


def _install_steps():
    """``(workflow, job, step_name, script)`` for every step running a shell script."""
    steps = []
    for path in sorted(WORKFLOW_DIR.glob("*.yml")):
        doc = yaml.safe_load(path.read_text()) or {}
        for job_name, job in (doc.get("jobs") or {}).items():
            for step in job.get("steps") or []:
                script = step.get("run")
                if script:
                    steps.append(
                        (path.name, job_name, step.get("name", "<unnamed>"), script)
                    )
    return steps


def _steps_installing_the_extra():
    return [
        entry for entry in _install_steps()
        if "pip install" in entry[3] and f",{EMULATION_EXTRA}" in entry[3]
    ]


@pytest.mark.unit
def test_the_ceiling_is_present():
    assert CONSTRAINTS.is_file(), f"{CONSTRAINTS} is missing"
    text = CONSTRAINTS.read_text()
    assert "Cython<3.3" in text, (
        "the Cython ceiling is gone from build-constraints.txt. Remove it only when "
        "harmonic ships a cp310 wheel or fixes its .pyx -- not before, or every job "
        "installing the emulation extra goes red again.\n" + text
    )


@pytest.mark.unit
def test_some_job_actually_installs_the_extra():
    """Guard the guard: the sweep below is vacuous if nothing matches."""
    found = _steps_installing_the_extra()
    assert found, (
        "no workflow step installs the emulation extra, so the checks below prove "
        "nothing. Did the extra get renamed?"
    )


@pytest.mark.unit
def test_every_install_of_the_extra_passes_the_constraints():
    offenders = [
        f"{wf}:{job}:{name}"
        for wf, job, name, script in _steps_installing_the_extra()
        if "--build-constraint" not in script
    ]
    assert not offenders, (
        "these steps install the emulation extra without --build-constraint, so pip "
        "will build harmonic's sdist with whatever Cython it resolves:\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.unit
def test_every_constrained_install_upgrades_pip_first():
    """``--build-constraint`` is a pip 25.1+ option.

    Passing it to the runner image's pip is the same red build by a different
    route -- an unknown-option error instead of a Cython one.
    """
    offenders = [
        f"{wf}:{job}:{name}"
        for wf, job, name, script in _install_steps()
        if "--build-constraint" in script
        and "pip install --upgrade pip" not in script
    ]
    assert not offenders, (
        "these steps pass --build-constraint without upgrading pip first, and it is "
        "a pip 25.1+ option:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.unit
def test_the_constraints_file_says_why():
    """A bare pin gets deleted by whoever next tidies dependencies."""
    text = CONSTRAINTS.read_text()
    for expected in ("harmonic", "autoemulate", "cp310"):
        assert expected in text, (
            f"build-constraints.txt does not mention {expected!r}, so the next person "
            "cannot tell what the ceiling is holding back or when it can go"
        )
