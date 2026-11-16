"""One working dir layout for a CA run, a pipeline bundle's run and CUFLynx.

The layout (``utilities/cuflynx_study.py``): the manifest, ``inputs/`` and the run
directories at the top of the working dir, a calibration in ``<method>_<prefix>_<obs>/``
and the MCMC that follows it in ``mcmc_<prefix>_<obs>/``, sensitivity in ``sensitivity/``
and the calibrated model as ``<prefix>_calibrated.cellml``. A pipeline bundle is the
working dir of the study it reruns.
"""
import ast
import glob
import importlib.util
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

from libcuflynx.param_id.calibrated_model import (calibrated_model_name,
                                                  calibrated_parameter_values,
                                                  write_calibrated_model)
from libcuflynx.param_id.paramID import UQ_CASE_METHOD, case_dir_name
from libcuflynx.scripts import generate_pipeline_script as gps
from libcuflynx.utilities import cuflynx_study

TEST_INPUTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_inputs")
PREFIX = "Lotka_Volterra_forced"
MODEL = os.path.join(TEST_INPUTS, PREFIX + ".cellml")
OBS = os.path.join(TEST_INPUTS, PREFIX + "_obs_data.json")
PARAMS = os.path.join(TEST_INPUTS, PREFIX + "_params_for_id.json")
OBS_PREFIX = PREFIX + "_obs_data"
#: A genetic algorithm of two small generations (a population of 8).
GA_OPTIONS = {"num_calls_to_function": 16, "cost_type": "gaussian_MLE", "num_elite": 1,
              "num_survivors": 2, "num_mutations_per_survivor": 2, "num_cross_breed": 2}


def _config(**overrides):
    config = {
        "file_prefix": PREFIX,
        "model_type": "cellml",
        "model_path": MODEL,
        "param_id_obs_path": OBS,
        "params_for_id_path": PARAMS,
        "solver": "CVODE_myokit",
        "solver_info": {"solver": "CVODE_myokit", "MaximumStep": 0.01},
        "dt": 0.01,
        "pre_time": 0.0,
        "sim_time": 5.0,
        "param_id_method": "genetic_algorithm",
        "do_simulation": False,
        "do_sensitivity": False,
        "do_calibration": False,
        "do_uq": False,
        "do_ia": False,
    }
    config.update(overrides)
    return config


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bundle_yaml(out_dir):
    (name,) = [n for n in os.listdir(out_dir) if n.startswith("user_inputs_")]
    with open(os.path.join(out_dir, name)) as handle:
        return yaml.safe_load(handle)


def _run_bundle(out_dir, timeout):
    env = dict(os.environ)
    src_dir = os.path.abspath(os.path.join(os.path.dirname(gps.__file__), "..", ".."))
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("CIRCULATORY_AUTOGEN_SRC", None)
    return subprocess.run(
        [sys.executable, "run_pipeline.py"], cwd=str(out_dir), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
        timeout=timeout)


# --- the case-dir naming rule ------------------------------------------------------------

@pytest.mark.unit
def test_a_calibration_and_its_mcmc_are_named_apart():
    assert UQ_CASE_METHOD == "mcmc"
    assert case_dir_name("genetic_algorithm", "p", "p_obs_data") == \
        "genetic_algorithm_p_p_obs_data"
    assert case_dir_name(UQ_CASE_METHOD, "p", "p_obs_data") == "mcmc_p_p_obs_data"


# --- the pipeline bundle -----------------------------------------------------------------

@pytest.mark.unit
def test_the_bundle_runs_in_its_own_folder_as_the_working_dir(tmp_path):
    """No output/ beside the script: a bundle inside a working dir would otherwise be a
    second working dir with its own manifest and inputs."""
    out = tmp_path / "bundle"
    gps.write_pipeline_bundle(_config(), str(out))

    result = _run_bundle(out, timeout=300)

    assert result.returncode == 0, result.stdout
    assert f"Outputs in {out}" in result.stdout, result.stdout
    assert not (out / "output").exists()


@pytest.mark.unit
def test_sensitivity_goes_to_the_working_dirs_sensitivity_folder(tmp_path):
    out = tmp_path / "bundle"
    gps.write_pipeline_bundle(_config(sa_options={"method": "sobol"}), str(out))
    script = _load(str(out / "run_pipeline.py"), "bundle_run_pipeline_sa")
    cfg = _bundle_yaml(str(out))

    inp = script.build_inp_data_dict(cfg, str(out))
    assert inp["sa_options"]["output_dir"] == os.path.join(str(out), "sensitivity")
    assert inp["sa_options"]["method"] == "sobol"

    cfg["sa_options"]["output_dir"] = "sobol_a"
    inp = script.build_inp_data_dict(cfg, str(out))
    assert inp["sa_options"]["output_dir"] == os.path.join(str(out), "sensitivity", "sobol_a")


@pytest.mark.unit
def test_the_bundle_drops_an_absolute_sensitivity_output_dir(tmp_path):
    """It names a folder on the machine the bundle was written on."""
    elsewhere = str(tmp_path / "somewhere" / "sensitivity")
    out = tmp_path / "bundle"
    gps.write_pipeline_bundle(
        _config(sa_options={"method": "sobol", "output_dir": elsewhere}), str(out))

    assert _bundle_yaml(str(out))["sa_options"] == {"method": "sobol"}


@pytest.mark.unit
def test_the_uq_obs_data_is_stored_with_the_working_dirs_inputs(tmp_path):
    out = tmp_path / "bundle"
    gps.write_pipeline_bundle(_config(), str(out))
    script = _load(str(out / "run_pipeline.py"), "bundle_run_pipeline_uq")

    path = script.mle_obs_data(OBS, str(out), "gaussian_MLE")

    assert os.path.dirname(os.path.dirname(path)) == os.path.join(str(out), "inputs", "obs_data")
    assert os.path.basename(path) == "uq_obs_data.json"
    with open(path) as handle:
        items = json.load(handle)["data_items"]
    assert items and all(item["cost_type"] == "gaussian_MLE" for item in items)
    assert [v["file"] for v in cuflynx_study.read_index(str(out))] == ["uq_obs_data.json"]
    assert not (out / "uq_obs_data.json").exists()


@pytest.mark.unit
def test_the_plots_read_the_chain_from_the_mcmc_run_directory(tmp_path):
    out = tmp_path / "bundle"
    gps.write_pipeline_bundle(_config(), str(out))
    util = _load(str(out / gps.PLOT_UTILITIES_NAME), "bundle_plot_utilities")
    util.load_plotting_libs()
    util.set_output_dir(str(out))
    run = out / case_dir_name(UQ_CASE_METHOD, PREFIX, OBS_PREFIX)
    run.mkdir()
    np.save(run / "mcmc_chain.npy", np.zeros((4, 3, 2)))
    (run / "param_names.csv").write_text("a/x\nb/y\n")
    (run / "posterior_predictive_coverage.json").write_text('{"num_samples": 3}')

    assert util.mcmc_chain().shape == (4, 3, 2)
    assert util.posterior_samples().shape == (6, 2)
    assert util.parameter_names() == ["a/x", "b/y"]
    assert util.posterior_predictive_coverage() == {"num_samples": 3}


@pytest.mark.unit
def test_the_module_defines_each_name_once():
    with open(gps.__file__) as handle:
        tree = ast.parse(handle.read())
    names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            names += [t.id for t in node.targets if isinstance(t, ast.Name)]
    assert sorted(n for n in set(names) if names.count(n) > 1) == []


# --- the calibrated model ----------------------------------------------------------------

def _initial_values(path, component):
    root = ET.parse(path).getroot()
    values = {}
    for comp in root.iter():
        if comp.tag.rsplit("}", 1)[-1] == "component" and comp.get("name") == component:
            for var in comp:
                if var.tag.rsplit("}", 1)[-1] == "variable" and var.get("initial_value"):
                    values[var.get("name")] = float(var.get("initial_value"))
    return values


@pytest.mark.unit
def test_a_grouped_row_sets_each_of_its_members():
    info = {"param_names": [["A/k", "B/k"], ["C/g"]]}
    assert calibrated_parameter_values(info, np.array([2.0, 3.0])) == \
        (["A/k", "B/k", "C/g"], [2.0, 2.0, 3.0])


@pytest.mark.unit
def test_the_calibrated_model_has_the_best_fit_written_in(tmp_path):
    info = {"param_names": [["Lotka_Volterra_module/alpha"], ["Lotka_Volterra_module/gamma"]]}
    out = str(tmp_path / calibrated_model_name(PREFIX))

    assert write_calibrated_model(MODEL, info, [1.25, 4.5], out) == out

    values = _initial_values(out, "Lotka_Volterra_module")
    assert values["alpha"] == 1.25 and values["gamma"] == 4.5
    assert values["beta"] == 0.2  # not calibrated, as it was
    assert calibrated_model_name(PREFIX) == PREFIX + "_calibrated.cellml"


# --- CA runs in a working dir ------------------------------------------------------------

def _param_id(out_dir, mcmc_instead=False, **settings):
    from libcuflynx.param_id.paramID import CVS0DParamID
    return CVS0DParamID(
        MODEL, "cellml", "genetic_algorithm", mcmc_instead, PREFIX,
        params_for_id_path=PARAMS, param_id_obs_path=OBS, sim_time=5.0, pre_time=0.0,
        dt=0.01, solver_info={"solver": "CVODE_myokit", "MaximumStep": 0.01},
        optimiser_options=GA_OPTIONS,
        UQ_options={"method": "mcmc", "library": "emcee", "num_steps": 4, "num_walkers": 8,
                    "cost_type": "gaussian_MLE"},
        param_id_output_dir=str(out_dir), one_rank=True, study_settings=settings)


@pytest.mark.integration
def test_a_run_in_a_working_dir_leaves_its_inputs_to_inputs(tmp_path):
    """inputs/ holds the obs_data and params_for_id with index.json; a copy in the run
    directory as well would be a second, unindexed copy."""
    in_working_dir = _param_id(tmp_path / "w")
    without = _param_id(tmp_path / "plain", cuflynx_study_copy_inputs=False)

    for stem in (OBS_PREFIX, PREFIX + "_params_for_id"):
        assert not glob.glob(os.path.join(in_working_dir.output_dir, stem + "_*.json"))
    assert {v["kind"] for v in cuflynx_study.read_index(str(tmp_path / "w"))} >= \
        {"obs_data", "params_for_id"}
    copied = sorted(os.path.basename(p) for p in
                    glob.glob(os.path.join(without.output_dir, "*_obs_data_*.json")))
    assert len(copied) == 1 and copied[0].startswith(OBS_PREFIX + "_")


@pytest.mark.integration
def test_the_mcmc_after_a_calibration_has_a_directory_of_its_own(tmp_path):
    """run_UQ on the calibration's engine: the chain goes to mcmc_<prefix>_<obs>/, the
    calibration's directory keeps its best fit, and the manifest has an entry for each."""
    param_id = _param_id(tmp_path)
    param_id.run()
    param_id.run_UQ()

    calibration = os.path.join(str(tmp_path), case_dir_name("genetic_algorithm", PREFIX,
                                                            OBS_PREFIX))
    mcmc = os.path.join(str(tmp_path), case_dir_name(UQ_CASE_METHOD, PREFIX, OBS_PREFIX))
    assert param_id.output_dir == calibration and param_id.uq_output_dir == mcmc
    assert os.path.isfile(os.path.join(calibration, "best_param_vals.npy"))
    assert not os.path.exists(os.path.join(calibration, "mcmc_chain.npy"))
    assert param_id.mcmc_chain_path() == os.path.join(mcmc, "mcmc_chain.npy")
    assert os.path.isfile(param_id.mcmc_chain_path())
    assert os.path.isfile(os.path.join(mcmc, "param_names.csv"))
    assert param_id.get_mcmc_samples() is not None

    with open(cuflynx_study.manifest_path({"param_id_output_dir": str(tmp_path),
                                           "file_prefix": PREFIX})) as handle:
        runs = json.load(handle)["runs"]
    by_dir = {run["dir"]: run for run in runs}
    assert by_dir[os.path.basename(mcmc)]["analyses"] == ["mcmc"]
    assert by_dir[os.path.basename(mcmc)]["status"] == "finished"
    assert by_dir[os.path.basename(calibration)]["analyses"] == ["genetic_algorithm"]

    # and the calibrated model beside them, which the manifest then names as the model
    path = param_id.write_calibrated_model()
    assert path == os.path.join(str(tmp_path), PREFIX + "_calibrated.cellml")
    best = param_id.get_best_param_vals()
    assert _initial_values(path, "Lotka_Volterra_module")["alpha"] == pytest.approx(best[0])
    study = cuflynx_study.read_study(cuflynx_study.manifest_path(
        {"param_id_output_dir": str(tmp_path), "file_prefix": PREFIX}))
    assert study["model"] == path


@pytest.mark.integration
def test_an_mcmc_built_for_it_reads_the_calibrations_best_fit(tmp_path):
    """mcmc_instead=True writes to mcmc_<prefix>_<obs>/ and finds the calibration's
    best_cost.npy in the calibration's directory."""
    calibration = _param_id(tmp_path)
    calibration.run()

    mcmc = _param_id(tmp_path, mcmc_instead=True)
    assert mcmc.output_dir == calibration.uq_output_dir
    assert mcmc.calibration_output_dir == calibration.output_dir
    from libcuflynx.param_id import paramID
    assert paramID.mcmc_object.calibration_best_cost() == pytest.approx(
        float(np.load(os.path.join(calibration.output_dir, "best_cost.npy"))))


# --- the pipeline, end to end ------------------------------------------------------------

@pytest.mark.integration
def test_a_bundle_run_leaves_the_working_dir_layout(tmp_path):
    """Calibration, sensitivity and MCMC through run_pipeline.py on Lotka-Volterra."""
    pytest.importorskip("emcee")
    out = tmp_path / "bundle"
    gps.write_pipeline_bundle(_config(
        do_sensitivity=True, do_calibration=True, do_uq=True,
        sa_options={"method": "sobol", "sample_type": "saltelli", "num_samples": 8},
        optimiser_options=GA_OPTIONS,
        UQ_options={"method": "mcmc", "library": "emcee", "num_steps": 4, "num_walkers": 8,
                    "cost_type": "gaussian_MLE"}), str(out))

    result = _run_bundle(out, timeout=540)
    assert result.returncode == 0, result.stdout[-4000:]

    calibration = out / case_dir_name("genetic_algorithm", PREFIX, OBS_PREFIX)
    mcmc = out / case_dir_name(UQ_CASE_METHOD, PREFIX, OBS_PREFIX)
    assert (calibration / "best_param_vals.npy").is_file()
    assert (mcmc / "mcmc_chain.npy").is_file()
    assert not (calibration / "mcmc_chain.npy").exists()
    assert glob.glob(str(out / "sensitivity" / "*Sobol_indices.csv"))
    assert glob.glob(str(out / "inputs" / "obs_data" / "*" / "uq_obs_data.json"))
    assert (out / (PREFIX + "_calibrated.cellml")).is_file()
    assert (out / (PREFIX + "_cuflynx_study.json")).is_file()
    assert not (out / "output").exists()
