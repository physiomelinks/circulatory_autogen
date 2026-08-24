"""A weight-0 series must change nothing about a sensitivity analysis.

Recorded traces are carried in obs_data at ``weight: 0`` so a plot has something to
draw the model against -- they are there to be looked at, not fitted. The claim that
makes that safe is that they are *inert*: a zero weight drops an item from the cost,
so every number a run produces should be identical whether the trace is in the file
or not.

That claim was worth testing rather than asserting. Three separate places had the
opposite assumption buried in them (the emulator trainer refused such items outright,
the use-time check in ``paramID`` disagreed with the trainer about them, and
``cost_calc`` interpolated them before looking at the weight), and each was found only
by running a full pipeline against a file that had them.

Sensitivity analysis is the sharpest test available: Saltelli sampling is a
deterministic Sobol sequence, so two runs over the same parameter box see exactly the
same points, and any difference in the indices is attributable to the obs_data alone.
"""
import json
import os
import shutil

import pytest

from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
from libcuflynx.scripts.sensitivity_analysis_run_script import run_SA

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - the serial stub covers the rest of the suite
    MPI = None


NUM_SAMPLES = 8


@pytest.fixture(scope="function")
def mpi_comm():
    if MPI is None:
        pytest.skip("mpi4py is required for the sensitivity analysis runner")
    return MPI.COMM_WORLD


def _recorded_trace_item():
    """A trace carried for plotting: a real series, weighted 0.

    Deliberately the same shape the generators emit -- ``data_type: series``,
    ``operands`` naming time and a model variable, and a weight of 0 -- so this
    tests the thing that actually appears in a study's obs_data.
    """
    return {
        "data_item_name": "recorded aortic root flow",
        "trace_name_for_plotting": "v_{AR} recorded",
        "data_type": "series",
        "operation": None,
        "operands": ["time", "aortic_root/v"],
        "unit": "m3_per_s",
        "weight": 0,
        "std": 1.0e-05,
        "obs_dt": 0.01,
        "value": [1.0e-04] * 32,
        "plot_type": "series",
    }


def _obs_files(resources_dir, tmp_path):
    """The same obs_data twice, once with a weight-0 trace appended."""
    source = os.path.join(resources_dir, "3compartment_obs_data.json")
    with open(source, encoding="utf-8-sig") as handle:
        items = json.load(handle)

    without = os.path.join(str(tmp_path), "without_series_obs_data.json")
    with open(without, "w") as handle:
        json.dump(items, handle)

    with_series = os.path.join(str(tmp_path), "with_series_obs_data.json")
    with open(with_series, "w") as handle:
        json.dump(list(items) + [_recorded_trace_item()], handle)

    return without, with_series


def _run_sa(base_user_inputs, resources_dir, obs_path, out_dir, generated_models_dir,
            mpi_comm):
    config = base_user_inputs.copy()
    config.update({
        "file_prefix": "3compartment",
        "input_param_file": "3compartment_parameters.csv",
        "model_type": "cellml",
        "solver": "CVODE",
        "param_id_method": "genetic_algorithm",
        "pre_time": 20,
        "sim_time": 2,
        "dt": 0.01,
        "DEBUG": True,
        "solver_info": {"MaximumStep": 0.001, "MaximumNumberOfSteps": 5000},
        "param_id_obs_path": obs_path,
        "param_id_output_dir": out_dir,
        "generated_models_dir": generated_models_dir,
        "sa_options": {
            "method": "sobol",
            "num_samples": NUM_SAMPLES,
            "sample_type": "saltelli",
            "output_dir": os.path.join(out_dir, "SA_results"),
        },
    })

    if mpi_comm.Get_rank() == 0:
        assert generate_with_new_architecture(False, config), "CellML generation failed"
    mpi_comm.Barrier()

    run_SA(config)
    return config["sa_options"]["output_dir"]


def _indices(sa_dir):
    """The Sobol table, as ``{column: [values]}``, keyed by output name."""
    import csv

    name = f"all_outputs_n{NUM_SAMPLES}_Sobol_indices.csv"
    path = os.path.join(sa_dir, name)
    assert os.path.isfile(path), f"no Sobol indices at {path}: {os.listdir(sa_dir)}"
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    return {column: [row[column] for row in rows] for column in rows[0]}


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_a_weight_zero_series_leaves_the_sobol_indices_untouched(
        base_user_inputs, resources_dir, tmp_path, temp_generated_models_dir, mpi_comm):
    """The whole point of carrying a trace at weight 0.

    Saltelli sampling is deterministic, so the two runs evaluate the model at exactly
    the same points. Every index for every output that exists in both files must
    therefore match to the digit -- not approximately, identically.
    """
    without_path, with_path = _obs_files(resources_dir, tmp_path)

    without_dir = _run_sa(base_user_inputs, resources_dir, without_path,
                          os.path.join(str(tmp_path), "without"),
                          temp_generated_models_dir, mpi_comm)
    with_dir = _run_sa(base_user_inputs, resources_dir, with_path,
                       os.path.join(str(tmp_path), "with"),
                       temp_generated_models_dir, mpi_comm)

    if mpi_comm.Get_rank() != 0:
        return

    plain = _indices(without_dir)
    with_trace = _indices(with_dir)

    assert plain["Parameter"] == with_trace["Parameter"], (
        "the parameters themselves changed, so the runs are not comparable")

    shared = [column for column in plain if column != "Parameter"]
    assert shared, "the run produced no Sobol columns to compare"

    differing = [column for column in shared
                 if column in with_trace and plain[column] != with_trace[column]]
    assert not differing, (
        f"a weight-0 series changed the sensitivity of {differing}. It is carried to be "
        f"plotted, not fitted, so it must not reach the cost at all.")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_the_trace_adds_no_output_of_its_own(
        base_user_inputs, resources_dir, tmp_path, temp_generated_models_dir, mpi_comm):
    """A zero-weighted item is not an observable the analysis reports on.

    Separate from the assertion above: indices could match on the shared columns while
    an extra column appeared for the trace, which would still mean it had reached the
    analysis. Sensitivity of something nothing is fitted to is not a meaningful number.
    """
    without_path, with_path = _obs_files(resources_dir, tmp_path)

    without_dir = _run_sa(base_user_inputs, resources_dir, without_path,
                          os.path.join(str(tmp_path), "without_cols"),
                          temp_generated_models_dir, mpi_comm)
    with_dir = _run_sa(base_user_inputs, resources_dir, with_path,
                       os.path.join(str(tmp_path), "with_cols"),
                       temp_generated_models_dir, mpi_comm)

    if mpi_comm.Get_rank() != 0:
        return

    assert sorted(_indices(without_dir)) == sorted(_indices(with_dir)), (
        "the columns differ, so the weight-0 trace reached the analysis as an output")
