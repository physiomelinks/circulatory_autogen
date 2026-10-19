"""The libcellml-from-source CI action only acts where libcuflynx's requirement admits it.

Remove with .github/actions/libcellml-from-source when libcellml 0.7.2 is on PyPI.

The action's first step reads the installed libcuflynx's libcellml requirement and decides
whether to replace the PyPI wheel with one built at a fixed commit. These tests run that
step's script, taken from action.yml, against the two requirements it has to tell apart.
"""
import importlib.metadata
import pathlib
import re

import pytest
import yaml

ACTION = (
    pathlib.Path(__file__).resolve().parents[1]
    / ".github" / "actions" / "libcellml-from-source" / "action.yml"
)


def _gate(monkeypatch, capsys, libcellml_requirement):
    """Run the gate step's Python with libcuflynx requiring ``libcellml_requirement``."""
    action = yaml.safe_load(ACTION.read_text())
    step = next(s for s in action["runs"]["steps"] if s.get("id") == "gate")
    script = re.search(r"<<'EOF'.*?\n(.*?)\n\s*EOF", step["run"], re.S).group(1)
    monkeypatch.setenv("LIBCELLML_VERSION", action["inputs"]["version"]["default"])
    monkeypatch.setattr(
        importlib.metadata, "requires",
        lambda dist: ["numpy", f"libcellml{libcellml_requirement}"] if dist == "libcuflynx" else [],
    )
    exec(compile(script, str(ACTION), "exec"), {})
    return capsys.readouterr().out.strip()


@pytest.mark.unit
@pytest.mark.parametrize("requirement,expected", [
    (">=0.6.3,<0.7.0", "build=false"),
    (">=0.7.1", "build=true"),
])
def test_gate_follows_the_libcellml_requirement(monkeypatch, capsys, requirement, expected):
    assert _gate(monkeypatch, capsys, requirement) == expected
