"""Tests for the dated user_inputs.yaml archival (reproducibility snapshot)."""

import os
from datetime import date

import yaml

from libcuflynx.parsers.PrimitiveParsers import save_dated_user_inputs


def _dated_name():
    return f"user_inputs_{date.today().strftime('%y%m%d')}.yaml"


def test_save_dated_user_inputs_writes_resolved_config(tmp_path):
    cfg = {
        "resources_dir": str(tmp_path),
        "file_prefix": "3compartment",
        "model_type": "casadi_python",
        "dt": 0.01,
        "do_sensitivity": True,
        "optimiser_options": {"num_calls_to_function": 100, "cost_convergence": 1e-4},
    }
    save_dated_user_inputs(cfg)

    out = tmp_path / _dated_name()
    assert out.is_file()
    loaded = yaml.safe_load(out.read_text())
    assert loaded["file_prefix"] == "3compartment"
    assert loaded["model_type"] == "casadi_python"
    assert loaded["do_sensitivity"] is True
    assert loaded["optimiser_options"]["num_calls_to_function"] == 100


def test_save_dated_user_inputs_skips_unserialisable_values(tmp_path):
    class _Obj:
        pass

    cfg = {"resources_dir": str(tmp_path), "file_prefix": "m", "_handle": _Obj()}
    save_dated_user_inputs(cfg)  # must not raise

    loaded = yaml.safe_load((tmp_path / _dated_name()).read_text())
    assert "_handle" not in loaded  # non-serialisable entry dropped
    assert loaded["file_prefix"] == "m"


def test_save_dated_user_inputs_no_resources_dir_is_noop(tmp_path):
    # Missing/invalid resources_dir: best-effort, writes nothing, never raises.
    save_dated_user_inputs({"file_prefix": "m"})
    save_dated_user_inputs({"resources_dir": str(tmp_path / "does_not_exist")})
    assert not (tmp_path / _dated_name()).exists()
