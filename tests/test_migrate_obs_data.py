"""``cuflynx-migrate-obs-data`` -- the upgrade path for the #466 vocabulary.

The rename is not purely mechanical, which is why a command exists rather than a note in the
changelog. `variable` split into an identity and an operand, `name_for_plotting` into two
labels, and the identity has to be *unique* -- so any file that named one variable once per
feature (the mean and the max of a trace, one variable measured across experiments) collides
and needs names it never had.
"""
import json
import os
import subprocess
import sys
import warnings

import pytest

from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser
from libcuflynx.scripts.migrate_obs_data import main, plan_file

_LEGACY_TWO_FEATURES = {
    "protocol_info": {"pre_times": [0.0], "sim_times": [[1.0]], "params_to_change": {}},
    "prediction_items": [
        {"variable": "main/y", "unit": "dimensionless", "name_for_plotting": "y"}
    ],
    "data_items": [
        {"variable": "flow", "name_for_plotting": "v", "data_type": "constant",
         "operation": "mean", "operands": ["main/x"], "unit": "dimensionless",
         "weight": 1.0, "value": 1.0, "std": 0.1},
        {"variable": "flow", "name_for_plotting": "v", "data_type": "constant",
         "operation": "max", "operands": ["main/x"], "unit": "dimensionless",
         "weight": 1.0, "value": 2.0, "std": 0.1},
    ],
}


def _write(tmp_path, doc, name="study_obs_data.json"):
    path = tmp_path / name
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def _parse(path):
    """Parse with DeprecationWarning promoted: a migrated file must have no legacy keys left."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        return ObsAndParamDataParser().parse_obs_data_json(param_id_obs_path=str(path),
                                                           pre_time=0.0, sim_time=1.0)


@pytest.mark.unit
def test_a_pre_466_file_is_refused_before_migration_and_loads_after(tmp_path):
    """The whole point, end to end."""
    path = _write(tmp_path, _LEGACY_TWO_FEATURES)
    with warnings.catch_warnings():
        # the legacy keys warn on the way in; the refusal being tested comes after that
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ValueError) as excinfo:
            ObsAndParamDataParser().parse_obs_data_json(param_id_obs_path=str(path),
                                                        pre_time=0.0, sim_time=1.0)
    assert "Duplicate 'data_item_name'" in str(excinfo.value)

    assert main([str(path)]) == 0
    _parse(path)          # raises if anything is still wrong


@pytest.mark.unit
def test_colliding_names_are_told_apart_by_what_actually_differs(tmp_path):
    path = _write(tmp_path, _LEGACY_TWO_FEATURES)
    main([str(path)])
    doc = json.loads(path.read_text(encoding="utf-8"))
    names = [i["data_item_name"] for i in doc["data_items"]]
    assert names == ["mean flow", "max flow"], names
    # the shared label is what stays shared -- these are two features of one trace
    assert [i["trace_name_for_plotting"] for i in doc["data_items"]] == ["v", "v"]


@pytest.mark.unit
def test_a_prediction_item_gets_the_operand_that_variable_used_to_be(tmp_path):
    """`prediction_items` had no `operands` key at all: `variable` was the model qname."""
    path = _write(tmp_path, _LEGACY_TWO_FEATURES)
    main([str(path)])
    pred = json.loads(path.read_text(encoding="utf-8"))["prediction_items"][0]
    assert pred["operands"] == ["main/y"]
    assert pred["data_item_name"] == "main/y"


@pytest.mark.unit
def test_a_reference_to_a_renamed_item_follows_it(tmp_path):
    """An observable built from other observables names them. If the migration renames one and
    not the reference, the file parses and then resolves to nothing -- the silent failure this
    whole change exists to remove."""
    doc = json.loads(json.dumps(_LEGACY_TWO_FEATURES))
    doc["data_items"].append({
        "variable": "flow difference", "name_for_plotting": "dv", "data_type": "constant",
        "operation": "calculate_two_observable_difference", "operands": [""],
        "operation_kwargs": {"subtract_this": "v", "subtract_from": "v"},
        "unit": "dimensionless", "weight": 1.0, "value": 1.0, "std": 0.1})
    # reference the *item* names, which is what resolves post-#466
    doc["data_items"][0]["variable"] = "flow mean"
    doc["data_items"][1]["variable"] = "flow max"
    doc["data_items"][2]["operation_kwargs"] = {
        "subtract_this": "flow mean", "subtract_from": "flow max"}
    path = _write(tmp_path, doc)
    main([str(path)])
    items = json.loads(path.read_text(encoding="utf-8"))["data_items"]
    kwargs = items[2]["operation_kwargs"]
    assert kwargs == {"subtract_this": "flow mean", "subtract_from": "flow max"}, kwargs
    _parse(path)


@pytest.mark.unit
def test_dry_run_writes_nothing(tmp_path):
    path = _write(tmp_path, _LEGACY_TWO_FEATURES)
    before = path.read_text(encoding="utf-8")
    assert main([str(path), "--dry-run"]) == 0
    assert path.read_text(encoding="utf-8") == before


@pytest.mark.unit
def test_running_it_twice_changes_nothing_the_second_time(tmp_path):
    path = _write(tmp_path, _LEGACY_TWO_FEATURES)
    main([str(path)])
    once = path.read_text(encoding="utf-8")
    main([str(path)])
    assert path.read_text(encoding="utf-8") == once
    assert plan_file(str(path))[1] == []


@pytest.mark.unit
def test_it_leaves_a_file_that_is_not_obs_data_alone(tmp_path):
    path = tmp_path / "module_config_obs.json"
    path.write_text(json.dumps({"vessels": [{"variable": "x"}]}), encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    assert main([str(path)]) == 0
    assert path.read_text(encoding="utf-8") == before


@pytest.mark.unit
def test_it_walks_a_directory(tmp_path):
    (tmp_path / "nested").mkdir()
    a = _write(tmp_path, _LEGACY_TWO_FEATURES, "a_obs_data.json")
    b = _write(tmp_path / "nested", _LEGACY_TWO_FEATURES, "b_obs_data.json")
    assert main([str(tmp_path)]) == 0
    for path in (a, b):
        _parse(path)


@pytest.mark.unit
def test_the_file_keeps_its_own_formatting(tmp_path):
    """Edits are textual so the diff shows the keys that moved, not a reformat of the file."""
    path = tmp_path / "hand_formatted_obs_data.json"
    path.write_text(
        '[\n'
        '  { "variable": "flow", "name_for_plotting": "v", "data_type": "constant",\n'
        '    "operation": "mean", "operands": ["main/x"], "unit": "1",\n'
        '    "weight": 1.0, "value": 1.0, "std": 0.1 }\n'
        ']\n', encoding="utf-8")
    main([str(path)])
    text = path.read_text(encoding="utf-8")
    assert text.startswith('[\n  { "data_item_name": "flow"'), text
    assert text.count('\n') == 5, text          # same five lines it had


@pytest.mark.unit
def test_the_console_command_runs(tmp_path):
    path = _write(tmp_path, _LEGACY_TWO_FEATURES)
    result = subprocess.run([sys.executable, "-m", "libcuflynx.scripts.migrate_obs_data",
                             str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "migrated" in result.stdout
    _parse(path)
