"""``cuflynx-generate-pipeline`` writes a folder that reproduces a configuration.

The thing worth testing is not that files appear but that the folder is *movable*:
every input copied in, every path in the yaml relative to the bundle, and none of
the absolute directories that describe the machine it was written on left behind.
A bundle that only runs where it was made looks identical until someone unpacks it
somewhere else.
"""
import ast
import os
import textwrap

import pytest
import yaml

from libcuflynx.scripts import generate_pipeline_script as gps


@pytest.fixture
def study(tmp_path):
    """A configuration with every kind of input the bundler has to carry."""
    src = tmp_path / "elsewhere"
    (src / "generated_models" / "demo").mkdir(parents=True)
    (src / "resources").mkdir(parents=True)

    model = src / "generated_models" / "demo" / "demo.cellml"
    model.write_text("<model/>")
    obs = src / "resources" / "demo_obs_data.json"
    obs.write_text('{"protocol_info": {}, "data_items": [], "prediction_items": []}')
    params = src / "resources" / "demo_params_for_id.csv"
    params.write_text("vessel_name,param_name,param_type,min,max\n")
    funcs = src / "my_ops.py"
    funcs.write_text("def op():\n    return 1\n")

    return {
        "file_prefix": "demo",
        "model_type": "cellml",
        "model_path": str(model),
        "generated_models_dir": str(src / "generated_models"),
        "param_id_output_dir": str(src / "param_id_output"),
        "resources_dir": str(src / "resources"),
        "param_id_obs_path": str(obs),
        "params_for_id_path": str(params),
        "operation_funcs_external_path": str(funcs),
        "solver": "CVODE_myokit",
        "solver_info": {"solver": "CVODE_myokit"},
        "dt": 0.001,
        "pre_time": 1.0,
        "sim_time": 2.0,
        "do_simulation": True,
        "do_calibration": True,
        "do_emulation": True,
        "emulator_settings": {"num_train_samples": 8},
    }


def read_bundle_yaml(out_dir):
    names = [n for n in os.listdir(out_dir) if n.startswith("user_inputs_")]
    assert len(names) == 1, names
    with open(os.path.join(out_dir, names[0])) as file:
        return yaml.safe_load(file)


# --- the bundle ---------------------------------------------------------------------

@pytest.mark.unit
def test_bundle_carries_every_input(study, tmp_path):
    out = tmp_path / "bundle"
    written = gps.write_pipeline_bundle(study, str(out))

    for name in (
        "run_pipeline.py",
        "generated_models/demo/demo.cellml",
        "resources/demo_obs_data.json",
        "resources/demo_params_for_id.csv",
        "resources/my_ops.py",
    ):
        assert (out / name).is_file(), f"{name} missing from the bundle"
        assert name in written, f"{name} written but not reported"


@pytest.mark.unit
def test_every_path_in_the_yaml_is_relative(study, tmp_path):
    """An absolute path is the difference between a bundle and a bookmark."""
    out = tmp_path / "bundle"
    gps.write_pipeline_bundle(study, str(out))
    cfg = read_bundle_yaml(str(out))

    assert cfg["resources_dir"] == "resources"
    assert cfg["param_id_obs_path"] == "resources/demo_obs_data.json"
    assert cfg["params_for_id_file"] == "demo_params_for_id.csv"
    assert cfg["model_file"] == "demo.cellml"
    assert cfg["operation_funcs_external_path"] == "resources/my_ops.py"

    for key, value in cfg.items():
        if isinstance(value, str):
            assert not os.path.isabs(value), f"{key} is still absolute: {value}"


@pytest.mark.unit
def test_machine_specific_directories_are_dropped(study, tmp_path):
    """run_pipeline.py rebuilds these from its own location; carrying the originals
    would point a moved bundle back at the machine that wrote it."""
    out = tmp_path / "bundle"
    gps.write_pipeline_bundle(study, str(out))
    cfg = read_bundle_yaml(str(out))

    for key in ("generated_models_dir", "param_id_output_dir", "model_path"):
        assert key not in cfg, f"{key} should not survive into the bundle"


@pytest.mark.unit
def test_model_is_found_by_convention_without_model_path(study, tmp_path):
    study.pop("model_path")
    out = tmp_path / "bundle"
    gps.write_pipeline_bundle(study, str(out))
    assert (out / "generated_models" / "demo" / "demo.cellml").is_file()


# --- refusals -----------------------------------------------------------------------

@pytest.mark.unit
def test_missing_model_is_named(study, tmp_path):
    study["model_path"] = str(tmp_path / "nowhere" / "gone.cellml")
    with pytest.raises(gps.PipelineBundleError, match="gone.cellml"):
        gps.write_pipeline_bundle(study, str(tmp_path / "bundle"))


@pytest.mark.unit
def test_missing_file_prefix_is_refused(study, tmp_path):
    study.pop("file_prefix")
    with pytest.raises(gps.PipelineBundleError, match="file_prefix"):
        gps.write_pipeline_bundle(study, str(tmp_path / "bundle"))


@pytest.mark.unit
def test_unlocatable_model_says_why(study, tmp_path):
    study.pop("model_path")
    study.pop("generated_models_dir")
    with pytest.raises(gps.PipelineBundleError, match="cannot be located"):
        gps.write_pipeline_bundle(study, str(tmp_path / "bundle"))


# --- the emulator -------------------------------------------------------------------

@pytest.mark.unit
def test_a_trained_emulator_travels_with_the_bundle(study, tmp_path):
    trained = tmp_path / "trained"
    trained.mkdir()
    (trained / "emulator_metadata.json").write_text("{}")
    study["emulator_settings"] = {"emulator_dir": str(trained)}
    study["use_emulator"] = True

    out = tmp_path / "bundle"
    gps.write_pipeline_bundle(study, str(out))
    cfg = read_bundle_yaml(str(out))

    assert cfg["emulator_settings"]["emulator_dir"] == "emulator"
    assert (out / "emulator" / "emulator_metadata.json").is_file()


@pytest.mark.unit
def test_use_emulator_without_one_warns(study, tmp_path, capsys):
    """The one way this bundle can look complete and not be."""
    study["emulator_settings"] = {"emulator_dir": str(tmp_path / "never_trained")}
    study["use_emulator"] = True

    gps.write_pipeline_bundle(study, str(tmp_path / "bundle"))
    assert "no emulator was found" in capsys.readouterr().err


# --- the generated script -----------------------------------------------------------

@pytest.mark.unit
def test_generated_script_is_valid_python():
    ast.parse(gps.render_pipeline_script())


@pytest.mark.unit
@pytest.mark.parametrize("flag", [
    "do_simulation", "do_emulation", "do_sensitivity", "do_calibration", "do_ia",
])
def test_generated_script_gates_each_stage(flag):
    """Every stage is opt-in from the yaml -- otherwise enabling one enables all."""
    assert f'cfg.get("{flag}")' in gps.render_pipeline_script()


@pytest.mark.unit
def test_generated_script_trains_an_emulator():
    """The stage CUFLynx's exported pipeline never had."""
    src = gps.render_pipeline_script()
    assert "EmulatorTrainer" in src
    assert "num_train_samples" in src, (
        "weak features should say which setting to raise")


@pytest.mark.unit
def test_only_rank_zero_reports_on_the_emulator():
    """train() returns the bundle on rank 0 and None everywhere else, by design.

    Reading the return on every rank would have each non-root rank conclude that
    nothing was trained -- under mpiexec that is N-1 ranks tearing the job down
    while rank 0 is still fitting.
    """
    src = gps.render_pipeline_script()
    stage = src.split("2) Emulator training")[1].split("3) Sensitivity")[0]

    assert 'getattr(trainer, "rank", 0) == 0' in stage, (
        "the emulator stage must gate its reporting on rank 0")
    assert "sys.exit(" not in stage, (
        "a non-root rank must not exit on a None bundle -- that is the expected "
        "return there")


@pytest.mark.unit
def test_generated_script_configures_the_emulator():
    """build_inp_data_dict has to copy the emulator keys, or the stage runs unconfigured."""
    src = gps.render_pipeline_script()
    for key in ("do_emulation", "use_emulator", "emulator_settings"):
        assert f'"{key}"' in src, f"{key} never reaches inp_data_dict"


@pytest.mark.unit
def test_generated_script_finds_its_own_config(tmp_path):
    """It globs for the dated yaml beside it, so the folder needs no arguments."""
    src = gps.render_pipeline_script()
    assert "user_inputs_*.yaml" in src


@pytest.mark.unit
def test_bundle_is_runnable_end_to_end_with_every_stage_off(study, tmp_path):
    """With no stage enabled the script should still load its config and exit clean --
    that is the check that the yaml it wrote is the yaml it can read."""
    import subprocess
    import sys

    for key in list(study):
        if key.startswith("do_"):
            study[key] = False
    out = tmp_path / "bundle"
    gps.write_pipeline_bundle(study, str(out))

    env = dict(os.environ)
    src_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(gps.__file__))), "..")
    env["PYTHONPATH"] = os.path.abspath(src_dir) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "run_pipeline.py"], cwd=str(out), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout
    assert "Done." in result.stdout, result.stdout
