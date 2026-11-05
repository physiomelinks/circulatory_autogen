"""``generate_modules_files`` must create the user directories it writes into.

``utilities/paths.default_module_config_user_dir`` says so in as many words -- "callers must
tolerate it not existing -- in an install it usually will not" -- and this script is the one
that *populates* that directory. It opened two files inside it for writing with no makedirs
and parsed ``user_units.cellml`` unconditionally, so from anywhere but a checkout that
already had the directory, a conversion did all of its work and then died on a bare
FileNotFoundError.

These tests point ``$CUFLYNX_USER_DIR`` at an empty temporary directory, which is what a
``pip install libcuflynx`` looks like, and drive the writers directly: the CellML conversion
in between needs libCellML and a real model, and none of that is what is being checked here.
"""
import os
import xml.etree.ElementTree as ET

import pytest

CELLML_NS = "http://www.cellml.org/cellml/1.0#"


@pytest.fixture
def in_empty_user_dir(tmp_path, monkeypatch):
    """A user directory with nothing in it, and the module reloaded to see it.

    The paths this script writes to are module-level constants computed at import from
    ``default_module_config_user_dir()``, so the environment has to be set before the
    reload rather than before the call.
    """
    import importlib

    from libcuflynx.utilities import paths

    monkeypatch.setenv(paths.CUFLYNX_USER_DIR_ENV_VAR, str(tmp_path))
    module = importlib.reload(
        importlib.import_module("libcuflynx.scripts.generate_modules_files"))
    monkeypatch.setattr(module, "cellml_namespace", CELLML_NS, raising=False)
    module.file_prefix = "my_module"
    module.component_name = "main"
    yield module, tmp_path
    # Restore the module to the state the rest of the session expects.
    monkeypatch.undo()
    importlib.reload(module)


@pytest.mark.unit
def test_the_module_imports_without_consuming_sys_argv():
    """It used to run argparse at module scope, so importing it parsed pytest's argv."""
    import libcuflynx.scripts.generate_modules_files as module

    assert module.file_prefix is None
    assert callable(module.main)


@pytest.mark.unit
def test_the_module_config_is_written_into_a_directory_that_did_not_exist(in_empty_user_dir):
    module, user_dir = in_empty_user_dir
    assert not (user_dir / "module_config_user").exists()

    module._generate_module_config(
        variables=[], constants=[], states=[], file_prefix="my_module",
        component_name="main")

    written = user_dir / "module_config_user" / "my_module_module_config.json"
    assert written.is_file(), sorted(p.name for p in user_dir.rglob("*"))


@pytest.mark.unit
def test_user_units_is_created_when_this_is_the_first_conversion(in_empty_user_dir):
    """No user_units.cellml yet is the normal state of a fresh working directory."""
    module, user_dir = in_empty_user_dir
    assert not os.path.exists(module.user_units_cellml)

    # A model carrying one unit of its own, which _update_units_file moves out of it.
    root = ET.Element(f"{{{CELLML_NS}}}model", name="m")
    units = ET.SubElement(root, f"{{{CELLML_NS}}}units", name="my_unit")
    ET.SubElement(units, f"{{{CELLML_NS}}}unit", units="second")

    module._update_units_file(root)

    assert os.path.isfile(module.user_units_cellml)
    written = ET.parse(module.user_units_cellml).getroot()
    names = {u.get("name") for u in written.findall(f".//{{{CELLML_NS}}}units")}
    assert "my_unit" in names
    # ...and the unit is gone from the module, which is the point of moving it.
    assert root.findall(f".//{{{CELLML_NS}}}units") == []


@pytest.mark.unit
def test_an_existing_user_units_is_appended_to_not_replaced(in_empty_user_dir):
    module, user_dir = in_empty_user_dir
    config_dir = user_dir / "module_config_user"
    config_dir.mkdir()
    (config_dir / "user_units.cellml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<model name="Units" xmlns="{CELLML_NS}">\n'
        '  <units name="already_here"><unit units="second"/></units>\n'
        '</model>\n', encoding="utf-8")

    root = ET.Element(f"{{{CELLML_NS}}}model", name="m")
    units = ET.SubElement(root, f"{{{CELLML_NS}}}units", name="brand_new")
    ET.SubElement(units, f"{{{CELLML_NS}}}unit", units="second")

    module._update_units_file(root)

    names = {u.get("name")
             for u in ET.parse(module.user_units_cellml).getroot().findall(
                 f".//{{{CELLML_NS}}}units")}
    assert {"already_here", "brand_new"} <= names


@pytest.mark.unit
def test_a_missing_template_config_says_what_to_do(in_empty_user_dir):
    """The generated <prefix>_user_inputs.yaml is a copy of user_run_files/user_inputs.yaml,
    and a wheel install has no user_run_files/. A bare FileNotFoundError on a path the user
    never typed is not an answer."""
    module, user_dir = in_empty_user_dir
    assert not os.path.exists(module.user_inputs_yaml)

    with pytest.raises(FileNotFoundError) as excinfo:
        module._generate_user_inputs_yaml(str(user_dir), "my_module")
    message = str(excinfo.value)
    assert "CUFLYNX_USER_DIR" in message
    assert "user_inputs.yaml" in message
