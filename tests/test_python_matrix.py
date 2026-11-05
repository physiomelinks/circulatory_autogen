"""The tested interpreters, and the two ceilings that decide them (#459).

``requires-python`` is not what bounds this project. Wheel availability for the compiled
dependencies is, and it points two different ways:

* **The floor** is the dependencies' own ``requires-python``. numpy, scipy, matplotlib,
  scikit-learn and numdifftools all declare ``>=3.10``, so a 3.9 job resolved older pins than
  any real user gets -- or failed to resolve at all. It was testing a configuration the
  dependency tree no longer supports, and the only 3.9-specific failure it ever produced (#458)
  was a test stub relying on 3.10+ ``staticmethod`` behaviour: a fact about the scaffolding.

* **The ceiling** is ``autoemulate``, which declares ``requires-python <3.13``. libcellml 0.6.3
  has wheels to cp313 and casadi to cp314, so the base install could go further -- but the
  ``emulation`` extra cannot. That asymmetry is the thing worth pinning down: if the matrix ever
  reaches 3.13, the emulator jobs have to stay behind, and the failure if they do not is a
  resolution error deep in a CI log rather than anything that names the cause.

Parsed as YAML rather than grepped: the property is "this job installs the extra *and* this
job's interpreter is within autoemulate's range", which spans two separate steps.
"""
import pathlib
import re

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
PYPROJECT = REPO_ROOT / "pyproject.toml"

#: autoemulate 2.1.2 declares requires-python >=3.10,<3.13. Raise only when it does.
EMULATION_MAX = (3, 12)

#: The extra that pulls autoemulate in.
EMULATION_EXTRA = "emulation"


def _version(text):
    """``"3.10"`` -> ``(3, 10)``. None for a matrix expression rather than a literal."""
    m = re.fullmatch(r"(\d+)\.(\d+)", str(text).strip())
    return (int(m.group(1)), int(m.group(2))) if m else None


def _requires_python_floor():
    m = re.search(r'^requires-python\s*=\s*">=(\d+\.\d+)"', PYPROJECT.read_text(), re.M)
    assert m, "requires-python is not declared as a >= floor in pyproject.toml"
    return _version(m.group(1))


def _jobs():
    """``(workflow, job_name, job)`` for every job in every workflow."""
    for path in sorted(WORKFLOW_DIR.glob("*.yml")):
        doc = yaml.safe_load(path.read_text()) or {}
        for name, job in (doc.get("jobs") or {}).items():
            yield path.name, name, job


def _test_matrix():
    for _wf, name, job in _jobs():
        if name == "test":
            versions = job["strategy"]["matrix"]["python-version"]
            return [_version(v) for v in versions]
    pytest.fail("the `test` job is gone; this guard needs re-pointing")


def _job_python(job):
    """The literal interpreter a job pins, or None if it takes one from a matrix."""
    for step in job.get("steps") or []:
        with_ = step.get("with") or {}
        if "python-version" in with_:
            return _version(with_["python-version"])
    return None


def test_the_matrix_starts_at_requires_python():
    """A tested version below the declared floor is testing an unsupported install; one
    above it leaves the floor untested by anything."""
    matrix = _test_matrix()
    assert min(matrix) == _requires_python_floor(), (
        f"the matrix's lowest version {min(matrix)} and requires-python "
        f"{_requires_python_floor()} disagree -- one of them is wrong"
    )


def test_no_tested_version_is_below_the_declared_floor():
    floor = _requires_python_floor()
    below = [v for v in _test_matrix() if v < floor]
    assert not below, f"matrix tests {below}, below requires-python {floor}"


def test_every_emulation_job_stays_within_autoemulates_range():
    """The one that will actually catch something. autoemulate declares <3.13, so a job that
    installs `[emulation]` on 3.13 fails at dependency resolution -- 20 seconds in, with a
    message about a version conflict rather than about this ceiling."""
    offenders = []
    for wf, name, job in _jobs():
        if EMULATION_EXTRA not in yaml.dump(job):
            continue
        pinned = _job_python(job)
        if pinned is not None and pinned > EMULATION_MAX:
            offenders.append(f"{wf}:{name} pins {pinned[0]}.{pinned[1]}")
    assert not offenders, (
        "these jobs install the `emulation` extra on an interpreter autoemulate does not "
        f"support (max {EMULATION_MAX[0]}.{EMULATION_MAX[1]}): {offenders}"
    )


def test_the_matrix_does_not_outrun_the_emulation_extra_silently():
    """If the matrix goes past autoemulate's ceiling, that is allowed -- the base install has
    wheels well beyond it -- but the emulator jobs must then be pinned rather than following
    along. This fails when the matrix moves up without anyone having revisited them."""
    if max(_test_matrix()) <= EMULATION_MAX:
        pytest.skip("the matrix is still within autoemulate's range")
    unpinned = [
        f"{wf}:{name}"
        for wf, name, job in _jobs()
        if EMULATION_EXTRA in yaml.dump(job) and _job_python(job) is None
    ]
    assert not unpinned, (
        "the matrix now goes past autoemulate's ceiling, so every job installing "
        f"`emulation` needs an explicit interpreter; these take one from a matrix: {unpinned}"
    )
