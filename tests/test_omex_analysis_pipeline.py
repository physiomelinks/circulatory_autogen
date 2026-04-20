"""
Integration test for the OMEX analysis pipeline generator.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from scripts.generate_omex_analysis_script import generate_omex_analysis_script


def _venv_python(project_root):
    return os.path.join(project_root, "venv", "bin", "python")


def _sanitized_venv_env():
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env.pop("__PYVENV_LAUNCHER__", None)
    for key in list(env):
        if key.startswith(("OMPI_", "PMI", "MPI", "PMIX_", "HYDRA_")):
            env.pop(key, None)
    return env


def _cellml_backend_available(python_executable):
    completed = subprocess.run(
        [
            python_executable,
            "-c",
            (
                "import importlib.util; "
                "print(bool(importlib.util.find_spec('opencor')) or "
                "bool(importlib.util.find_spec('myokit')))"
            ),
        ],
        capture_output=True,
        text=True,
        env=_sanitized_venv_env(),
        check=True,
    )
    return completed.stdout.strip() == "True"


@pytest.mark.integration
@pytest.mark.slow
def test_generate_omex_analysis_pipeline_runs_successfully(temp_output_dir):
    project_root = os.path.join(os.path.dirname(__file__), "..")
    python_executable = _venv_python(project_root)
    omex_path = os.path.join(
        project_root,
        "tests",
        "test_inputs",
        "cardiomyocyte_with_data_omex_test.omex",
    )
    generated_script_path = os.path.join(temp_output_dir, "omex_analysis_pipeline.py")

    generated_path = generate_omex_analysis_script(omex_path, generated_script_path)
    assert os.path.exists(generated_path), f"Generated pipeline script not found: {generated_path}"

    with open(generated_path, "r", encoding="utf-8") as fh:
        generated_script = fh.read()
    assert "PARAMS_FOR_ID =" in generated_script
    assert "OBSERVABLE_SELECTIONS =" in generated_script
    assert "run_pipeline()" in generated_script
    assert '"model_type": "cellml_only"' in generated_script
    assert '"solver": "CVODE"' in generated_script
    assert "PythonGenerator(" not in generated_script

    if not os.path.exists(python_executable):
        pytest.skip(f"Expected project venv python not found: {python_executable}")
    if not _cellml_backend_available(python_executable):
        pytest.skip(f"OMEX pipeline execution requires a CellML backend in {python_executable}.")

    completed = subprocess.run(
        [python_executable, generated_path],
        cwd=project_root,
        capture_output=True,
        text=True,
        env=_sanitized_venv_env(),
        timeout=900,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "Generated OMEX analysis pipeline failed.\n"
            f"STDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
        )

    output_root = os.path.join(temp_output_dir, "pipeline_outputs")
    summary_path = os.path.join(output_root, "pipeline_summary.json")
    assert os.path.exists(summary_path), f"Pipeline summary not found: {summary_path}"

    with open(summary_path, "r", encoding="utf-8") as fh:
        summary = json.load(fh)

    sa_output_dir = summary["sa_output_dir"]
    param_id_output_dir = summary["param_id_output_dir"]
    param_id_plot_dir = summary["param_id_plot_dir"]
    best_param_vals_path = summary["best_param_vals_path"]
    laplace_mean_path = summary["laplace_mean_path"]
    laplace_covariance_path = summary["laplace_covariance_path"]

    assert os.path.exists(sa_output_dir), f"Sensitivity output directory missing: {sa_output_dir}"
    assert any(name.endswith(".csv") for name in os.listdir(sa_output_dir)), (
        f"Sensitivity output directory should contain CSV artifacts: {sa_output_dir}"
    )

    best_cost_path = os.path.join(param_id_output_dir, "best_cost.npy")
    assert os.path.exists(best_cost_path), f"Calibration cost file missing: {best_cost_path}"
    assert os.path.exists(best_param_vals_path), (
        f"Calibration parameter file missing: {best_param_vals_path}"
    )
    assert os.path.exists(param_id_plot_dir), f"Parameter ID plot directory missing: {param_id_plot_dir}"

    plot_files = os.listdir(param_id_plot_dir)
    assert any(name.startswith("reconstruct_") and name.endswith(".png") for name in plot_files), (
        f"Expected reconstruct plot PNG in {param_id_plot_dir}"
    )
    assert any(name.startswith("error_bars_") and name.endswith(".png") for name in plot_files), (
        f"Expected error bar plot PNG in {param_id_plot_dir}"
    )
    assert any(name.startswith("std_error_bars_") and name.endswith(".png") for name in plot_files), (
        f"Expected std error bar plot PNG in {param_id_plot_dir}"
    )

    best_cost = float(np.load(best_cost_path))
    assert np.isfinite(best_cost), f"Calibration cost should be finite, got {best_cost}"
    assert best_cost >= 0.0, f"Calibration cost should be non-negative, got {best_cost}"
    assert best_cost < 1.0e6, f"Calibration cost should remain below threshold, got {best_cost}"

    assert os.path.exists(laplace_mean_path), f"Laplace mean file missing: {laplace_mean_path}"
    assert os.path.exists(laplace_covariance_path), (
        f"Laplace covariance file missing: {laplace_covariance_path}"
    )

    covariance = np.load(laplace_covariance_path)
    assert covariance.shape[0] == covariance.shape[1], "Covariance matrix should be square"
    assert not np.isnan(covariance).any(), "Covariance matrix should not contain NaN values"
    assert not np.isinf(covariance).any(), "Covariance matrix should not contain Inf values"
