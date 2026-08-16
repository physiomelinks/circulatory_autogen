"""The release workflow has to keep matching PyPI's trusted publisher (issue #438).

Publishing is configured in two places that cannot see each other: this repository's workflow
file, and a *pending publisher* registered on PyPI against the four-tuple

    owner=physiomelinks  repo=circulatory_autogen  workflow=release.yml  environment=pypi

If the workflow is renamed, the environment is renamed or dropped, or `id-token: write` is
lost, the build still goes green and only the upload fails -- on the one run nobody can retry,
because a PyPI version can never be re-uploaded. Nothing else in the repository would notice,
so these assertions stand in for the feedback the first release will not give us.

They deliberately do not run anything: no build, no network, no upload. Whether the package
actually publishes can only be established by publishing it.
"""
import pathlib
import re

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

#: Not a free choice -- PyPI's pending publisher names this exact file. See CONTRIBUTING.md.
RELEASE_WORKFLOW = WORKFLOW_DIR / "release.yml"

#: Likewise the environment the publishing job runs in.
PYPI_ENVIRONMENT = "pypi"


@pytest.fixture(scope="module")
def workflow():
    assert RELEASE_WORKFLOW.exists(), (
        "the release workflow must be %s exactly; PyPI's trusted publisher is bound to that "
        "filename" % RELEASE_WORKFLOW.relative_to(REPO_ROOT)
    )
    return yaml.safe_load(RELEASE_WORKFLOW.read_text(encoding="utf-8"))


def _triggers(workflow):
    # PyYAML reads the bare key `on` as the boolean True (YAML 1.1), which is why this is not
    # simply workflow["on"].
    return workflow.get("on", workflow.get(True))


@pytest.mark.unit
def test_release_workflow_triggers_on_version_tags(workflow):
    tags = _triggers(workflow)["push"]["tags"]
    assert any(t.startswith("v") for t in tags), (
        "the release must be cut from a v* tag, got %r" % (tags,)
    )


@pytest.mark.unit
def test_publish_job_uses_trusted_publishing(workflow):
    """OIDC, in the `pypi` environment, with no long-lived token anywhere."""
    jobs = workflow["jobs"]
    publishers = [
        job for job in jobs.values()
        if any("pypi-publish" in str(step.get("uses", "")) for step in job.get("steps", []))
    ]
    assert len(publishers) == 1, "expected exactly one job that publishes, got %d" % len(publishers)
    publish = publishers[0]

    assert publish.get("permissions", {}).get("id-token") == "write", (
        "trusted publishing needs `permissions: id-token: write` on the publishing job"
    )

    environment = publish.get("environment")
    name = environment.get("name") if isinstance(environment, dict) else environment
    assert name == PYPI_ENVIRONMENT, (
        "the publishing job must run in the environment %r that PyPI's publisher is "
        "registered against, got %r" % (PYPI_ENVIRONMENT, name)
    )

    step = next(s for s in publish["steps"] if "pypi-publish" in str(s.get("uses", "")))
    assert step["uses"].startswith("pypa/gh-action-pypi-publish@"), step["uses"]
    # A `with: password:` would mean a stored token, which is the thing being avoided.
    assert "password" not in step.get("with", {}), (
        "no API token: the upload credential comes from the OIDC exchange"
    )


@pytest.mark.unit
def test_no_pypi_token_secret_anywhere_in_the_workflows():
    """A token in any workflow would defeat the point of the OIDC setup."""
    offenders = []
    for path in sorted(WORKFLOW_DIR.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if re.search(r"secrets\.[A-Z_]*PYPI[A-Z_]*", line):
                offenders.append("%s:%d: %s" % (path.name, lineno, line.strip()))
    assert not offenders, "PyPI credentials must not come from a stored secret:\n" + "\n".join(offenders)


@pytest.mark.unit
def test_workflow_builds_both_a_wheel_and_an_sdist(workflow):
    """`python -m build` with no flags, so the sdist is not quietly dropped."""
    runs = " ".join(
        step.get("run", "")
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
    )
    assert re.search(r"python -m build(?!\s+--(sdist|wheel))", runs), (
        "the build job must run `python -m build` (both distributions), got: %s" % runs
    )


@pytest.mark.unit
def test_workflow_refuses_a_tag_that_disagrees_with_pyproject(workflow):
    """The one check that has to happen before anything is uploaded."""
    runs = "\n".join(
        step.get("run", "")
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
    )
    assert "pyproject.toml" in runs and "GITHUB_REF_NAME" in runs, (
        "the workflow must compare the tag with `version` in pyproject.toml before building; "
        "a mismatched release cannot be corrected, because PyPI never accepts a version twice"
    )


@pytest.mark.unit
def test_release_procedure_is_documented():
    """Somewhere findable, and naming the parts that cannot be undone."""
    text = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "Making a release" in text
    for needle in ("release.yml", PYPI_ENVIRONMENT, "trusted publishing", "pyproject.toml",
                   "0.5.0", "funcs_user"):
        assert needle in text, "CONTRIBUTING.md should mention %r in the release section" % needle
    assert re.search(r"never be (replaced|re-uploaded)", text), (
        "the docs must say a published PyPI version cannot be re-uploaded"
    )
