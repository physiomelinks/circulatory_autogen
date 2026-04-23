"""
Generate a single Python analysis pipeline script from an OMEX archive.
"""

from __future__ import annotations

import argparse
import os, sys
import pprint
import textwrap
from pathlib import Path
from string import Template

root_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir, 'src'))
from parsers.OMEXParsers import OMEXArchiveParser


def _indent_python(data, spaces: int = 4) -> str:
    return textwrap.indent(pprint.pformat(data, width=100, sort_dicts=False), " " * spaces).lstrip()


def _build_script_contents(omex_path: str, parser: OMEXArchiveParser, file_prefix: str) -> str:
    params_for_id = parser.build_default_params_for_id()
    observable_dataset_index_by_variable = {
        name: 0 for name in parser.get_direct_series_selection_options().keys()
    }
    observable_specs = parser.build_direct_series_observable_specs(observable_dataset_index_by_variable)
    observable_dataset_options = parser.get_direct_series_selection_options()
    project_root = Path(__file__).resolve().parents[2]
    observable_summary = [
        {
            "plot_index": spec.plot_index,
            "trace_index": spec.trace_index,
            "plot_name": spec.name,
            "external_expression": spec.external_expression,
            "model_expression": spec.model_expression,
            "operation_name": spec.operation_name,
        }
        for spec in observable_specs
    ]

    template = '''"""
Auto-generated OMEX analysis pipeline.

This script extracts the CellML model from an OMEX archive, maps the archive's
experimental time-series data into Circulatory Autogen's observable format, and
runs sensitivity analysis, calibration, and Laplace identifiability analysis.
"""

from __future__ import annotations

import json
import os
import sys
import importlib.util
from pathlib import Path

import numpy as np
from mpi4py import MPI

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = Path(r"$project_root")
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from identifiabilty_analysis.identifiabilityAnalysis import IdentifiabilityAnalysis
from param_id.paramID import CVS0DParamID
from parsers.OMEXParsers import OMEXArchiveParser
from sensitivity_analysis.sensitivityAnalysis import SensitivityAnalysis


OMEX_PATH = Path(r"$omex_path")
FILE_PREFIX = "$file_prefix"
OUTPUT_ROOT = THIS_FILE.parent / "pipeline_outputs"
GENERATED_MODELS_DIR = OUTPUT_ROOT / "generated_models"
PARAM_ID_OUTPUT_DIR = OUTPUT_ROOT / "param_id_output"
SENSITIVITY_OUTPUT_DIR = OUTPUT_ROOT / "sensitivity"
RESOURCES_DIR = OUTPUT_ROOT / "resources_unused"
SUMMARY_PATH = OUTPUT_ROOT / "pipeline_summary.json"


# These entries are discovered from the OMEX archive's input metadata and can be
# edited directly without writing a params_for_id.csv file.
PARAMS_FOR_ID = $params_for_id


# The generated script defaults to fitting directly to time-series data.
# To switch to derived features later, update the OMEX parser output or change
# the resulting obs_data_dict items to use a custom operation/operands pair.
OBSERVABLE_SELECTIONS = $observable_summary


# If an OMEX archive contains multiple experimental datasets for the same
# variable, choose which dataset index to use here. The default is the first
# dataset (index 0) for each duplicated variable.
OBSERVABLE_DATASET_INDEX_BY_VARIABLE = $observable_dataset_index_by_variable


# Available duplicated-dataset choices discovered from the OMEX archive.
OBSERVABLE_DATASET_OPTIONS = $observable_dataset_options


INP_DATA_DICT = {{
    "DEBUG": True,
    "file_prefix": FILE_PREFIX,
    "input_param_file": FILE_PREFIX + "_parameters.csv",
    "model_type": "cellml_only",
    "solver": "CVODE",
    "solver_info": {{
        "solver": "CVODE_myokit",
        "method": "CVODE",
        "MaximumStep": 0.1,
        "MaximumNumberOfSteps": 5000,
        "rtol": 1e-8,
        "atol": 1e-8,
    }},
    "generated_models_dir": str(GENERATED_MODELS_DIR),
    "param_id_output_dir": str(PARAM_ID_OUTPUT_DIR),
    "resources_dir": str(RESOURCES_DIR),
    "dt": 1.0,
    "pre_time": 0.0,
    "sim_time": 3000.0,
    "param_id_method": "genetic_algorithm",
    "optimiser_options": {{
        "num_calls_to_function": 2000,
        "max_patience": 10,
        "cost_convergence": 0.1,
        "cost_type": "MSE",
    }},
    "sa_options": {{
        "method": "sobol",
        "sample_type": "saltelli",
        "num_samples": 4,
        "output_dir": str(SENSITIVITY_OUTPUT_DIR),
    }},
    "do_ia": True,
    "ia_options": {{
        "method": "Laplace",
    }},
}}

OPENCOR_AVAILABLE = importlib.util.find_spec("opencor") is not None
MYOKIT_AVAILABLE = importlib.util.find_spec("myokit") is not None


def build_sensitivity_obs_data_dict(series_obs_data_dict):
    protocol_info = dict(series_obs_data_dict["protocol_info"])
    sa_data_items = []
    for item in series_obs_data_dict["data_items"]:
        values = np.asarray(item["value"], dtype=float)
        for op_name, reducer in (("mean", np.mean), ("max", np.max)):
            feature_value = float(reducer(values))
            sa_data_items.append({
                "variable": item["variable"] + "_" + op_name,
                "name_for_plotting": item["name_for_plotting"] + "_" + op_name,
                "data_type": "constant",
                "unit": item["unit"],
                "weight": item.get("weight", 1.0),
                "operands": item["operands"],
                "operation": op_name,
                "value": feature_value,
                "std": max(abs(feature_value) * 0.05, 1e-6),
                "experiment_idx": item.get("experiment_idx", 0),
                "subexperiment_idx": item.get("subexperiment_idx", 0),
            })
    return {
        "protocol_info": protocol_info,
        "data_items": sa_data_items,
        "prediction_items": series_obs_data_dict.get("prediction_items", []),
    }


def build_runtime_state():
    parser = OMEXArchiveParser(str(OMEX_PATH))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    GENERATED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PARAM_ID_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SENSITIVITY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESOURCES_DIR.mkdir(parents=True, exist_ok=True)

    model_dir = GENERATED_MODELS_DIR / FILE_PREFIX
    model_dir.mkdir(parents=True, exist_ok=True)
    cellml_path = parser.extract_model(str(model_dir), FILE_PREFIX)
    observable_specs = parser.build_direct_series_observable_specs(OBSERVABLE_DATASET_INDEX_BY_VARIABLE)
    obs_data_dict = parser.build_obs_data_dict(observable_specs)

    first_series = obs_data_dict["data_items"][0]["value"]
    obs_dt = obs_data_dict["data_items"][0]["obs_dt"]
    sim_time = obs_dt * (len(first_series) - 1) if len(first_series) > 1 else obs_dt

    inp_data_dict = dict(INP_DATA_DICT)
    inp_data_dict["generated_models_dir"] = str(GENERATED_MODELS_DIR)
    inp_data_dict["param_id_output_dir"] = str(PARAM_ID_OUTPUT_DIR)
    inp_data_dict["sim_time"] = float(sim_time)
    inp_data_dict["dt"] = float(obs_dt)
    inp_data_dict["model_path"] = str(cellml_path)
    inp_data_dict["one_rank"] = MPI.COMM_WORLD.Get_size() == 1
    inp_data_dict["solver_info"] = dict(inp_data_dict["solver_info"])
    inp_data_dict["solver_info"]["sim_time"] = float(sim_time)
    inp_data_dict["solver_info"]["pre_time"] = float(inp_data_dict["pre_time"])
    if inp_data_dict["model_type"] == "cellml_only" and not OPENCOR_AVAILABLE and MYOKIT_AVAILABLE:
        inp_data_dict["solver_info"]["solver"] = "CVODE_myokit"
    return parser, inp_data_dict, obs_data_dict, PARAMS_FOR_ID


def run_pipeline():
    parser, inp_data_dict, obs_data_dict, params_for_id = build_runtime_state()
    # Sensitivity needs scalar outputs for the current Sobol implementation, so
    # we derive simple features from the raw series here.
    sa_obs_data_dict = build_sensitivity_obs_data_dict(obs_data_dict)
    # Calibration and identifiability use the original series data directly.
    calibration_obs_data_dict = obs_data_dict

    sa_agent = SensitivityAnalysis.init_from_dict(inp_data_dict)
    sa_agent.set_ground_truth_data(sa_obs_data_dict)
    sa_agent.set_params_for_id(params_for_id)
    sa_agent.run_sensitivity_analysis(inp_data_dict["sa_options"])

    selected_param_names = sa_agent.choose_most_impactful_params_sobol(
        top_n=len(params_for_id),
        index_type="ST",
        criterion="max",
        threshold=0.0,
        use_threshold=False,
    )
    if not selected_param_names:
        selected_param_names = [f'{{entry["vessel_name"]}}/{{entry["param_name"]}}' for entry in params_for_id]
    selected_params = [
        entry for entry in params_for_id
        if f'{{entry["vessel_name"]}}/{{entry["param_name"]}}' in selected_param_names
    ]

    param_id = CVS0DParamID.init_from_dict(inp_data_dict)
    param_id.set_ground_truth_data(calibration_obs_data_dict)
    # add features here if you want to change to calibrating to features rather than the series
    # param_id.set_ground_truth_data(build_sensitivity_obs_data_dict(obs_data_dict))
    param_id.set_params_for_id(selected_params)
    param_id.run()
    if MPI.COMM_WORLD.Get_rank() == 0:
        param_id.plot_outputs()

    best_param_vals = param_id.get_best_param_vals()
    id_analysis = IdentifiabilityAnalysis(
        inp_data_dict["model_path"],
        inp_data_dict["model_type"],
        inp_data_dict["file_prefix"],
        param_id_output_dir=inp_data_dict["param_id_output_dir"],
        resources_dir=inp_data_dict["resources_dir"],
        param_id=param_id.param_id,
    )
    id_analysis.set_best_param_vals(best_param_vals)
    id_analysis.run(inp_data_dict["ia_options"])

    if MPI.COMM_WORLD.Get_rank() == 0:
        summary = {{
            "archive": parser.describe_archive(),
            "model_path": inp_data_dict["model_path"],
            "sa_output_dir": inp_data_dict["sa_options"]["output_dir"],
            "param_id_output_dir": param_id.output_dir,
            "param_id_plot_dir": param_id.plot_dir,
            "selected_param_names": selected_param_names,
            "best_cost": float(np.load(os.path.join(param_id.output_dir, "best_cost.npy"))),
            "best_param_vals_path": os.path.join(param_id.output_dir, "best_param_vals.npy"),
            "laplace_mean_path": os.path.join(OUTPUT_ROOT, f"{{FILE_PREFIX}}_laplace_mean.npy"),
            "laplace_covariance_path": os.path.join(OUTPUT_ROOT, f"{{FILE_PREFIX}}_laplace_covariance.npy"),
        }}
        with open(SUMMARY_PATH, "w", encoding="utf-8") as wf:
            json.dump(summary, wf, indent=2)
        print(f"Pipeline summary written to {{SUMMARY_PATH}}")


if __name__ == "__main__":
    run_pipeline()
'''
    template = template.replace("{{", "{").replace("}}", "}")
    return Template(template).substitute(
        project_root=str(project_root),
        omex_path=omex_path,
        file_prefix=file_prefix,
        params_for_id=_indent_python(params_for_id),
        observable_summary=_indent_python(observable_summary),
        observable_dataset_index_by_variable=_indent_python(observable_dataset_index_by_variable),
        observable_dataset_options=_indent_python(observable_dataset_options),
    )


def generate_omex_analysis_script(omex_path: str, output_script_path: str, file_prefix: str | None = None) -> str:
    parser = OMEXArchiveParser(omex_path)
    file_prefix = file_prefix or parser.get_default_file_prefix()
    output_script_path = os.path.abspath(output_script_path)
    output_dir = os.path.dirname(output_script_path)
    os.makedirs(output_dir, exist_ok=True)

    contents = _build_script_contents(os.path.abspath(omex_path), parser, file_prefix)
    with open(output_script_path, "w", encoding="utf-8") as wf:
        wf.write(contents)
    return output_script_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a Python analysis pipeline from an OMEX archive.")
    parser.add_argument("omex_path", help="Path to the OMEX archive")
    parser.add_argument("output_script_path", help="Where to write the generated Python script")
    parser.add_argument(
        "--file-prefix",
        dest="file_prefix",
        default=None,
        help="Optional model/script prefix override",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    generated = generate_omex_analysis_script(args.omex_path, args.output_script_path, file_prefix=args.file_prefix)
    print(f"Generated analysis pipeline: {generated}")
