"""plot_outputs with a data item whose operand is a model constant.

The solver does not log constants, so a constant operand's trace is its one value rather than
one sample per time step. Its reconstruction page draws that value as a flat line over the
subexperiment, alongside the item's own horizontal target and output lines.

Model: ``Simple_ODE_Benchmark`` (``dx/dt = -x + p``, ``dy/dt = -3y + q``); ``benchmark/p`` is
a constant.
"""
import json
import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from libcuflynx.param_id.paramID import CVS0DParamID
from libcuflynx.param_id.plot_outputs import ParamIDPlotOutputs
from libcuflynx.parsers.PrimitiveParsers import YamlFileParser

P_FIT = 1.5
SIM_TIME = 2.0


def _obs(tmp_path, resources_dir):
    with open(os.path.join(resources_dir, "Simple_ODE_Benchmark_obs_data.json")) as f:
        doc = json.load(f)
    doc["protocol_info"]["sim_times"] = [[SIM_TIME]]
    doc["data_items"].append({
        "data_item_name": "p mean",
        "trace_name_for_plotting": "p",
        "data_type": "constant",
        "operation": "mean",
        "operands": ["benchmark/p"],
        "unit": "dimensionless",
        "cost_type": "gaussian_MLE",
        "weight": 1.0,
        "value": 1.2,
        "std": 0.1,
        "experiment_idx": 0,
        "subexperiment_idx": 0,
        "plot_type": "horizontal",
    })
    path = tmp_path / "constant_operand_obs_data.json"
    path.write_text(json.dumps(doc))
    return str(path)


@pytest.mark.integration
@pytest.mark.mpi
def test_plot_outputs_draws_a_constant_operand_as_a_flat_line(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, tmp_path,
        monkeypatch):
    from libcuflynx.scripts.script_generate_with_new_architecture import \
        generate_with_new_architecture
    from libcuflynx.utilities.mpi_utils import get_MPI

    comm = get_MPI().COMM_WORLD
    config = base_user_inputs.copy()
    config.update({
        "file_prefix": "Simple_ODE_Benchmark",
        "input_param_file": "Simple_ODE_Benchmark_parameters.json",
        "model_type": "cellml",
        "solver": "CVODE_myokit",
        "param_id_method": "genetic_algorithm",
        "pre_time": 0.0,
        "sim_time": SIM_TIME,
        "dt": 0.05,
        "DEBUG": True,
        "do_uq": False,
        "do_ia": False,
        "plot_predictions": False,
        "solver_info": {"MaximumStep": 0.01},
        "param_id_obs_path": _obs(tmp_path, resources_dir),
        "params_for_id_path": os.path.join(resources_dir,
                                           "Simple_ODE_Benchmark_params_for_id.json"),
        "param_id_output_dir": temp_output_dir,
        "resources_dir": resources_dir,
        "generated_models_dir": temp_generated_models_dir,
    })
    config = YamlFileParser().parse_user_inputs_file(
        config, obs_path_needed=True, do_generation_with_fit_parameters=False)
    if comm.Get_rank() == 0:
        assert generate_with_new_architecture(False, config), "benchmark generation failed"
    comm.Barrier()
    if comm.Get_rank() != 0:
        return

    pages = {}
    save = ParamIDPlotOutputs._save_reconstruction_figure_bundle

    def _capture(plot_dir, prefix, obs_stub, plot_idx, fig, axs, *rest):
        pages[axs.get_ylabel()] = {
            line.get_label(): (np.asarray(line.get_xdata()), np.asarray(line.get_ydata()))
            for line in axs.get_lines()
        }
        save(plot_dir, prefix, obs_stub, plot_idx, fig, axs, *rest)

    monkeypatch.setattr(
        ParamIDPlotOutputs, "_save_reconstruction_figure_bundle", staticmethod(_capture))

    pid = CVS0DParamID.init_from_dict(config)
    pid.set_best_param_vals(np.array([P_FIT, 1.0]))
    try:
        pid.plot_outputs()
    finally:
        pid.close_simulation()

    page = next(lines for ylabel, lines in pages.items() if ylabel.startswith("$p$"))
    t, trace = page["output"]
    assert t[0] == pytest.approx(0.0) and t[-1] == pytest.approx(SIM_TIME)
    np.testing.assert_allclose(trace, P_FIT)
    np.testing.assert_allclose(page["mean output"][1], P_FIT)
    np.testing.assert_allclose(page["mean gt"][1], 1.2)


@pytest.mark.unit
def test_a_constant_prediction_is_padded_to_the_experiment_trace():
    """The MCMC prediction plots stack every prediction over the experiment's time axis, so a
    constant (one value per subexperiment) is padded to its segment's length."""
    from libcuflynx.param_id.paramID import ParamID

    sim_times = [1.0, 2.0]  # dt 0.5: 3 samples, then 5 sharing the first with the previous
    trace = [np.arange(3.0), np.arange(2.0, 7.0)]
    pid = ParamID.__new__(ParamID)
    pid.dt = 0.5
    pid.protocol_info = {"num_sub_per_exp": [2], "sim_times": [sim_times]}
    pid.prediction_info = {"operands": [["v/x"], ["v/k"]]}
    # [sub][pred_idx][operand_idx] -> a trace, or a constant's single value
    pid.get_pred_from_params = lambda **kwargs: [
        [[trace[sub]], [np.array([7.0])]] for sub in range(2)]

    pred = pid.get_pred_array_from_params_per_exp(None, 0)

    np.testing.assert_allclose(pred[0], [0, 1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(pred[1], np.full(7, 7.0))
