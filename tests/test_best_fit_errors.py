"""A calibration writes its best fit's error per observable (plot_outputs.ERROR_VEC_FILES) when
it ends, finished or interrupted, with no figure; a new run removes the last run's; and
``save_best_fit_errors`` writes them afterwards for a run that was killed, from its
best_param_vals.npy.

Model: ``Simple_ODE_Benchmark`` (``dx/dt = -x + p``, ``dy/dt = -3y + q``), a short GA.
"""
import glob
import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from libcuflynx.param_id.paramID import CVS0DParamID
from libcuflynx.param_id.plot_outputs import ERROR_VEC_FILES
from libcuflynx.parsers.PrimitiveParsers import YamlFileParser

#: a best fit a stopped run left on disk
BEST_SO_FAR = np.array([1.5, 1.0])


@pytest.fixture
def config(base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir):
    from libcuflynx.scripts.script_generate_with_new_architecture import \
        generate_with_new_architecture
    from libcuflynx.utilities.mpi_utils import get_MPI
    comm = get_MPI().COMM_WORLD
    config = base_user_inputs.copy()
    config.update({
        "file_prefix": "Simple_ODE_Benchmark",
        "input_param_file": "Simple_ODE_Benchmark_parameters.json",
        "model_type": "cellml", "solver": "CVODE_myokit",
        "param_id_method": "genetic_algorithm",
        "pre_time": 0.0, "sim_time": 2.0, "dt": 0.05, "DEBUG": True,
        "debug_optimiser_options": {"num_calls_to_function": 60, "cost_type": "gaussian_MLE"},
        "do_uq": False, "do_ia": False, "plot_predictions": False,
        "solver_info": {"MaximumStep": 0.01},
        "param_id_obs_path": os.path.join(resources_dir, "Simple_ODE_Benchmark_obs_data.json"),
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
    return config


def _errors(output_dir):
    return {key: np.load(os.path.join(output_dir, name), allow_pickle=True)
            for key, name in ERROR_VEC_FILES.items()}


def _stale(output_dir):
    """Error files of an earlier run, which a new run must not leave beside its best fit."""
    os.makedirs(output_dir, exist_ok=True)
    for name in ERROR_VEC_FILES.values():
        np.save(os.path.join(output_dir, name), np.array([999.0]))


def _stopping_run(pid, exception, seen=None):
    """param_id.run that saves BEST_SO_FAR as the optimisers do mid-run, then raises."""
    def run():
        if seen is not None:
            seen.extend(name for name in ERROR_VEC_FILES.values()
                        if os.path.exists(os.path.join(pid.output_dir, name)))
        np.save(os.path.join(pid.output_dir, "best_param_vals.npy"), BEST_SO_FAR)
        raise exception
    pid.param_id.run = run


@pytest.mark.integration
@pytest.mark.mpi
def test_a_finished_calibration_writes_the_errors_plot_outputs_would(config):
    """The same vectors plot_outputs writes, from one simulation, and no reconstruction page."""
    pid = CVS0DParamID.init_from_dict(config)
    try:
        pid.run()
        if pid.rank != 0:
            return
        written = _errors(pid.output_dir)
        best = np.load(os.path.join(pid.output_dir, "best_param_vals.npy"))
        assert glob.glob(os.path.join(pid.plot_dir, "reconstruct_*")) == []
        assert len(written["names"]) == len(written["percent_error"]) > 0
        pid.set_best_param_vals(best)
        pid.simulate_with_best_param_vals()
        pid.plot_outputs()
        drawn = _errors(pid.output_dir)
        for key in ("percent_error", "std_error"):
            np.testing.assert_allclose(written[key], drawn[key], rtol=1e-9)
        assert list(written["names"]) == list(drawn["names"])
    finally:
        pid.close_simulation()


@pytest.mark.integration
@pytest.mark.mpi
def test_an_interrupted_calibration_writes_the_errors_of_its_best_so_far(config):
    """Ctrl-C mid-run: the errors are those of the best_param_vals.npy it left, and an earlier
    run's files are gone before it starts."""
    pid = CVS0DParamID.init_from_dict(config)
    try:
        if pid.rank == 0:
            _stale(pid.output_dir)
        seen = []
        _stopping_run(pid, KeyboardInterrupt(), seen)
        with pytest.raises(KeyboardInterrupt):
            pid.run()
        if pid.rank != 0:
            return
        assert seen == []
        written = _errors(pid.output_dir)
        reference = os.path.join(pid.output_dir, "reference")
        os.makedirs(reference)
        pid.output_dir = reference
        pid.save_best_fit_errors(BEST_SO_FAR)
        expected = _errors(reference)
        for key in ("percent_error", "std_error"):
            np.testing.assert_allclose(written[key], expected[key], rtol=1e-9)
            assert written[key].shape != (1,)
    finally:
        pid.close_simulation()


@pytest.mark.integration
@pytest.mark.mpi
def test_a_killed_calibration_leaves_no_errors_and_they_are_written_afterwards(config):
    """A run that dies without an interrupt (killed, crashed) writes nothing; called later,
    save_best_fit_errors writes them from the best_param_vals.npy the run left."""
    pid = CVS0DParamID.init_from_dict(config)
    try:
        if pid.rank == 0:
            _stale(pid.output_dir)
        _stopping_run(pid, RuntimeError("killed"))
        with pytest.raises(RuntimeError, match="killed"):
            pid.run()
        if pid.rank != 0:
            return
        for name in ERROR_VEC_FILES.values():
            assert not os.path.exists(os.path.join(pid.output_dir, name)), name
        paths = pid.save_best_fit_errors()
        assert sorted(paths) == sorted(ERROR_VEC_FILES)
        written = _errors(pid.output_dir)
        np.testing.assert_allclose(pid.param_id.best_param_vals, BEST_SO_FAR)
        assert written["percent_error"].shape == written["std_error"].shape == \
            written["names"].shape
        assert np.all(np.isfinite(written["std_error"]))
    finally:
        pid.close_simulation()
