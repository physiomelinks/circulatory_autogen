"""GitHub release notes from CHANGELOG.md (``.github/scripts/release_notes.py``).

GitHub refuses a release body over 125,000 characters, which failed v0.8.0's release job (its
section was 193 kB) after PyPI had published. Over the limit, the notes keep the intro and the
leading subsections that fit, and name the rest with a link to the full section at the tag.
"""
import importlib.util
import os
import pathlib
import shutil
import subprocess

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "release_notes.py"

_spec = importlib.util.spec_from_file_location("release_notes", SCRIPT)
rn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rn)

CHANGELOG = """# Changelog

## Unreleased

- not yet

## 1.2.0 — 2030-01-01

Intro line.

### Before you upgrade

- convert your files

### Fixed

- a bug

## 1.1.0 — 2029-01-01

- older
"""


@pytest.mark.unit
def test_the_notes_are_the_version_s_section_and_an_install_footer():
    text = rn.notes(CHANGELOG, "1.2.0")
    assert text.startswith("\nIntro line.\n")
    assert "### Fixed\n\n- a bug" in text
    assert "not yet" not in text and "older" not in text
    assert text.endswith("Install: `pip install libcuflynx==1.2.0`\n"
                         "PyPI: https://pypi.org/project/libcuflynx/1.2.0/\n")


@pytest.mark.unit
def test_a_version_with_no_section_gets_a_stub():
    assert rn.notes(CHANGELOG, "9.9.9").startswith("See CHANGELOG.md.\n")


@pytest.mark.unit
def test_long_notes_keep_whole_subsections_in_order_and_list_the_rest():
    changelog = "## 2.0.0 — 2030-02-02\n\nIntro.\n\n" + "".join(
        f"### Part {i}\n\n" + "x" * 300 + "\n\n" for i in range(10)) + "## 1.0.0\n"
    limit = 1500
    text = rn.notes(changelog, "2.0.0", limit=limit)

    assert len(text) <= limit
    kept = [i for i in range(10) if f"### Part {i}\n" in text]
    assert kept == list(range(len(kept))) and 0 < len(kept) < 10
    listed = [i for i in range(10) if f"- Part {i}\n" in text]
    assert listed == list(range(len(kept), 10)), "every dropped subsection is named, in order"
    assert ("https://github.com/physiomelinks/circulatory_autogen/blob/v2.0.0/CHANGELOG.md"
            "#200--2030-02-02") in text
    assert text.endswith("PyPI: https://pypi.org/project/libcuflynx/2.0.0/\n")


@pytest.mark.unit
def test_the_0_8_0_notes_fit_and_keep_before_you_upgrade():
    """The real case: this repository's 0.8.0 section is over the limit by itself."""
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    _, body = rn.section(changelog, "0.8.0")
    assert len(body) > rn.LIMIT, "the fixture no longer exercises the limit"
    text = rn.notes(changelog, "0.8.0")
    assert len(text) <= rn.LIMIT
    assert "### Before you upgrade" in text and "### Results that change" in text
    assert "#080--2026-11-27" in text


@pytest.mark.unit
def test_the_release_workflow_step_writes_the_notes_file(tmp_path):
    """Run the step's own shell, so a broken path or argument fails here rather than on a tag."""
    workflow = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "release.yml").read_text())
    steps = workflow["jobs"]["github-release"]["steps"]
    step = next(s for s in steps if s.get("name") == "Take the notes from CHANGELOG.md")
    (tmp_path / ".github" / "scripts").mkdir(parents=True)
    shutil.copy(str(SCRIPT), str(tmp_path / ".github" / "scripts" / "release_notes.py"))
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
    result = subprocess.run(
        ["bash", "-c", step["run"]], cwd=str(tmp_path),
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "GITHUB_REF_NAME": "v1.2.0"},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout
    assert "- a bug" in (tmp_path / "notes.md").read_text(encoding="utf-8")
