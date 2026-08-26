"""Can a calibration find a sodium conductance across the threshold where the cell starts firing?

This is the SN_full V_max problem reproduced small enough to run in a test. In the real study
each ``V_max`` observable is the spike peak when the cell fires and the subthreshold maximum
when it does not, so it takes one of two values ~110 mV apart and nothing in between. Scored by
a plain Gaussian against a measurement sigma of a few mV, that gap is 25-30 sigma and one wrong
branch costs more than every other observable in the study put together -- a number that comes
from pricing a *model discrepancy* event on a *measurement* scale.

``SN_simple`` shows the same thing on one parameter. Sweeping ``soma_SN/g_Na1_6`` (the transient
Nav1.6 current, the spike generator) under a weak stimulus:

    g_Na1_6   0.00   0.02   0.04   0.05  |  0.06   0.07   0.10   0.20   0.30   0.45   0.60
    V_max    -55.0  -54.4  -53.5  -52.7  | +51.6  +52.7  +52.8  +56.1  +57.1  +57.0  +58.2
    spikes       0      0      0      0  |     1      1      1      2      4      5      5

A 105 mV step between 0.05 and 0.06, and a count that is exactly zero on one side. So the test
covers both regimes by construction, and it exercises every piece the real study leans on:
``gaussian_MLE_robust`` for the jump, ``poisson_MLE`` with a background rate for the count, the
``multi_phase_`` emulator that classifies a jump column instead of regressing across it, and
MCMC recovery of the parameter that generated the data.

**dt matters more than it looks.** At ``dt=0.005`` -- fine for the smooth observables -- V_max
scatters anywhere between -55 and +57 with no visible branch structure, because a 5 ms sample
grid lands wherever it likes on a ~1 ms action potential. The bimodality above only appears at
``dt=2e-4``. A test that used the coarse grid would be measuring the sampler, not the cell.
"""
import json
import os
import shutil

import numpy as np
import pytest

from libcuflynx.param_id.paramID import CVS0DParamID
from libcuflynx.parsers.PrimitiveParsers import YamlFileParser
from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

#: The stimulus. Weak on purpose: at the -0.12 nA of the other SN_simple fixtures the cell
#: fires for every g_Na1_6 in range, so there is no threshold to cross.
I_IN = -0.02
PARAM = "soma_SN/g_Na1_6"
PARAM_MIN, PARAM_MAX = 0.0, 0.6

#: Straddling the threshold at ~0.055: one firing, one silent.
G_FIRING = 0.30
G_SILENT = 0.02

#: The knobs under test, matching what SN_full's ProcessData.py now writes.
V_MAX_STD = 6.0
V_MAX_P_OUTLIER = 0.04
V_MAX_OUTLIER_WIDTH = 170.0
COUNT_BACKGROUND_RATE = 0.01


@pytest.fixture
def mpi_comm():
    if MPI is None:
        pytest.skip("mpi4py is required for this test")
    return MPI.COMM_WORLD


def _obs_document(v_max, n_spikes):
    """One current step, scored by the two costs this test exists for."""
    return {
        "protocol_info": {
            "pre_times": [0.1],
            "sim_times": [[0.35, 0.85]],
            "params_to_change": {"soma_SN/I_in": [[0.0, I_IN]]},
            "experiment_colors": ["r"],
            "experiment_labels": ["threshold"],
            "protocol_traces": {},
        },
        "prediction_items": [],
        "data_items": [
            {
                "data_item_name": "V_max", "trace_name_for_plotting": "V_{max}",
                "data_type": "constant", "operation": "max",
                "operands": ["soma_SN/V"], "unit": "milliV", "weight": 1.0,
                "value": float(v_max), "std": V_MAX_STD,
                "cost_type": "gaussian_MLE_robust",
                "cost_kwargs": {"p_outlier": V_MAX_P_OUTLIER,
                                "outlier_width": V_MAX_OUTLIER_WIDTH},
                "experiment_idx": 0, "subexperiment_idx": 1, "plot_type": "horizontal",
            },
            {
                "data_item_name": "n_spikes", "trace_name_for_plotting": "n_{spikes}",
                "data_type": "constant", "operation": "calc_spike_count_windowed",
                "operands": ["time", "soma_SN/V"],
                "operation_kwargs": {"spike_min_thresh": -10},
                "unit": "dimensionless", "weight": 1.0,
                "cost_type": "poisson_MLE",
                "prob_dist_params": {"k": float(n_spikes)},
                "cost_kwargs": {"background_rate": COUNT_BACKGROUND_RATE},
                "experiment_idx": 0, "subexperiment_idx": 1, "plot_type": "None",
            },
        ],
    }


def _resources(resources_dir, work_dir):
    """A resources directory holding just this test's one-parameter params_for_id.

    Copied rather than pointed at the real one because ``params_for_id_path`` is always
    re-derived from ``resources_dir`` by the parser -- setting the key in the config has no
    effect at all, which is a trap worth not falling into twice.
    """
    out = os.path.join(work_dir, "resources")
    os.makedirs(out, exist_ok=True)
    for name in ("SN_simple_vessel_array.csv", "SN_simple_parameters.csv"):
        shutil.copy(os.path.join(resources_dir, name), out)
    with open(os.path.join(out, "SN_simple_params_for_id.csv"), "w") as handle:
        handle.write("vessel_name, param_name, min, max, name_for_plotting\n")
        handle.write(f"soma_SN, g_Na1_6, {PARAM_MIN}, {PARAM_MAX}, g_{{Na1.6}}\n")
    return out


def _config(base_user_inputs, work_dir, resources, obs_path, **overrides):
    config = base_user_inputs.copy()
    config.update({
        "file_prefix": "SN_simple",
        "input_param_file": "SN_simple_parameters.csv",
        "params_for_id_file": "SN_simple_params_for_id.csv",
        "model_type": "cellml",
        "solver": "CVODE_myokit",
        "param_id_method": "genetic_algorithm",
        "pre_time": 0.1, "sim_time": 1.2,
        # see the module docstring: the branch structure is invisible at 5 ms
        "dt": 2e-4,
        "DEBUG": True, "do_uq": False, "do_ia": False, "do_ad": False,
        "plot_predictions": False,
        "solver_info": {"solver": "CVODE_myokit", "MaximumStep": 2e-4},
        "resources_dir": resources,
        "generated_models_dir": os.path.join(work_dir, "generated_models"),
        "param_id_output_dir": os.path.join(work_dir, "out"),
        "param_id_obs_path": obs_path,
        "optimiser_options": {"num_calls_to_function": 4},
        "debug_optimiser_options": {"num_calls_to_function": 4},
    })
    config.update(overrides)
    return YamlFileParser().parse_user_inputs_file(
        config, obs_path_needed=True, do_generation_with_fit_parameters=False)


def _features(pid, theta):
    """The (V_max, n_spikes) the solver produces at one parameter value."""
    mapping = list(pid.obs_info["const_idx_to_obs_idx"])
    _, operands, _ = pid.get_cost_obs_and_pred_from_params(np.asarray([theta], dtype=float))
    values = np.full(len(mapping), np.nan)
    with pid.accumulating_temp_results():
        index = 0
        for exp_idx in range(pid.protocol_info["num_experiments"]):
            for sub_idx in range(pid.protocol_info["num_sub_per_exp"][exp_idx]):
                operand = operands[index] if index < len(operands) else None
                index += 1
                if operand is None:
                    continue
                with pid.evaluating_segment(exp_idx, sub_idx):
                    obs_dict = pid.get_obs_output_dict(operand)
                if obs_dict is None:
                    continue
                const = np.asarray(obs_dict["const"], dtype=float).reshape(-1)
                for const_idx in range(len(const)):
                    if pid._item_belongs_to_segment(mapping[const_idx], exp_idx, sub_idx):
                        values[const_idx] = const[const_idx]
    return values


def _posterior(output_dir, burn_in=0.5):
    """The chain, burnt in and flattened. ``mcmc_chain.npy`` is (steps, walkers, params)."""
    chain = np.load(os.path.join(output_dir, "mcmc_chain.npy"))
    start = int(burn_in * chain.shape[0])
    return chain[start:].reshape(-1, chain.shape[2])[:, 0]


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_the_firing_threshold_is_a_jump_the_plain_gaussian_cannot_afford(
        base_user_inputs, resources_dir, temp_output_dir, mpi_comm):
    """The cliff exists, and the mixture is what makes it affordable.

    Two things are asserted that no unit test can: that the *simulator* really does produce two
    branches with nothing between them, and that the difference between the two costs at a
    wrong-branch parameter is the difference between a bounded penalty and a dominating one.
    """
    pytest.importorskip("myokit")
    if mpi_comm.Get_size() > 1:
        pytest.skip("this test drives the solver directly and is single-rank")

    work = temp_output_dir
    resources = _resources(resources_dir, work)
    obs_path = os.path.join(work, "obs_probe.json")
    with open(obs_path, "w") as handle:
        json.dump(_obs_document(0.0, 0.0), handle)

    config = _config(base_user_inputs, work, resources, obs_path)
    assert generate_with_new_architecture(False, config), "SN_simple generation failed"
    config = _config(base_user_inputs, work, resources, obs_path)
    pid = CVS0DParamID.init_from_dict(config).param_id

    ladder = [0.0, 0.02, 0.04, 0.05, 0.06, 0.08, 0.20, 0.30, 0.60]
    seen = np.array([_features(pid, g) for g in ladder])
    v_max, spikes = seen[:, 0], seen[:, 1]

    silent = spikes == 0
    assert silent.any() and (~silent).any(), (
        f"the design must cover both regimes; got spike counts {spikes.tolist()}")
    # the two branches, and the empty space between them
    assert v_max[silent].max() < -40.0, f"silent branch is not subthreshold: {v_max[silent]}"
    assert v_max[~silent].min() > 40.0, f"firing branch is not a spike peak: {v_max[~silent]}"
    assert v_max[~silent].min() - v_max[silent].max() > 80.0, "no gap between the branches"
    # V_max carries no information the count does not: it is the same binary, restated
    assert np.array_equal(v_max > 0.0, ~silent)

    from libcuflynx.funcs.cost_funcs_user import gaussian_MLE, gaussian_MLE_robust

    observed = float(v_max[~silent][0])          # the cell fired
    wrong_branch = float(v_max[silent][-1])      # the model did not
    plain = gaussian_MLE(wrong_branch, observed, V_MAX_STD, 1.0)
    robust = gaussian_MLE_robust(wrong_branch, observed, V_MAX_STD, 1.0,
                                 p_outlier=V_MAX_P_OUTLIER,
                                 outlier_width=V_MAX_OUTLIER_WIDTH)
    assert plain > 100.0, f"expected the plain Gaussian to be ruinous here, got {plain}"
    assert robust <= np.log(V_MAX_OUTLIER_WIDTH / V_MAX_P_OUTLIER) + 1e-9
    assert robust < plain / 10.0


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_mcmc_over_an_emulator_recovers_the_conductance_that_made_the_data(
        base_user_inputs, resources_dir, temp_output_dir, mpi_comm):
    """Train on the jump, then infer through it, and check the answer against the truth.

    The data is the solver's own output at a known ``g_Na1_6``, so "was the inference right" has
    an answer that does not depend on the emulator's opinion of itself. Two truths are used, one
    each side of the threshold, because they fail in different ways: a firing observation
    identifies the parameter from both sides, and a silent one only bounds it from above -- every
    conductance below threshold produces exactly the same silence. A posterior that claims to
    have pinned down a silent cell is wrong, and that is worth asserting as much as recovery is.

    One emulator serves both, deliberately: ``value``, ``cost_type`` and ``cost_kwargs`` are
    outside the emulator fingerprint, so changing what the data says needs no retrain. Only the
    operations would.
    """
    pytest.importorskip("myokit")
    pytest.importorskip("autoemulate")
    pytest.importorskip("emcee")

    from libcuflynx.emulators.emulator_bundle import EmulatorBundle
    from libcuflynx.emulators.emulator_trainer import EmulatorTrainer, resolve_emulator_dir
    from libcuflynx.emulators.internal_emulators import classify_features

    work = temp_output_dir
    rank = mpi_comm.Get_rank()
    resources = _resources(resources_dir, work) if rank == 0 else os.path.join(work, "resources")
    mpi_comm.Barrier()

    probe_path = os.path.join(work, "obs_probe.json")
    if rank == 0:
        with open(probe_path, "w") as handle:
            json.dump(_obs_document(0.0, 0.0), handle)
        assert generate_with_new_architecture(
            False, _config(base_user_inputs, work, resources, probe_path)), "generation failed"
    mpi_comm.Barrier()

    emulator_settings = {
        "models": "multi_phase_RadialBasisFunctions",
        # MIN_SIDE_ROWS is 8, and the silent branch is only g <= ~0.055, i.e. 9% of the box.
        # At 40 samples the low side has 3 rows and classify_features calls the column smooth,
        # which is how the whole point of the test gets silently skipped. 160 puts ~12 silent
        # rows in the training split, and the two adaptive stages add more near the threshold.
        "num_train_samples": 160,
        "sample_type": "sobol",
        "random_seed": 0,
        "test_fraction": 0.2,
        "n_iter": 2,
        "n_splits": 2,
        # this is the tool for finding out whether an emulator is good enough, so it must be
        # able to load one that is not
        "min_r2": -1e30,
        "out_of_bounds": "clip",
    }
    train_config = _config(base_user_inputs, work, resources, probe_path,
                           do_emulation=True, emulator_settings=emulator_settings)
    trainer = EmulatorTrainer.init_from_dict(train_config, comm=mpi_comm)
    bundle = trainer.train()
    mpi_comm.Barrier()
    if rank != 0:
        return

    emulator_dir = resolve_emulator_dir(train_config)
    saved = EmulatorBundle.load(emulator_dir)
    labels = list(saved.feature_labels)
    v_idx, n_idx = labels.index("V_max"), labels.index("n_spikes")

    # Classified rather than regressed across, the jump column is nearly exact; regressed
    # across it (what happens when the design is too small to see the branches) it scored
    # R2 -30.9 on this same fixture.
    r2 = dict(zip(labels, np.asarray(saved.meta["feature_r2"], dtype=float)))
    assert r2["V_max"] > 0.9, f"held-out R2 for the jump column is {r2['V_max']:.4f}"
    assert r2["n_spikes"] > 0.9, f"held-out R2 for the count column is {r2['n_spikes']:.4f}"

    y_train = np.asarray(saved.y_train, dtype=float)
    kinds = classify_features(y_train)
    assert kinds[n_idx] == "count", (
        f"an integer spike count must be recognised as one, got {kinds[n_idx]}")
    assert kinds[v_idx] == "jump", (
        f"V_max must be recognised as a jump so it is classified rather than regressed "
        f"across, got {kinds[v_idx]}")
    assert (y_train[:, n_idx] == 0).any() and (y_train[:, n_idx] > 0).any(), (
        "the design must contain both silent and firing points")

    # The emulator must land on a branch, never between them -- a value the simulator cannot
    # produce is not a small error, it is a different answer.
    grid = np.linspace(PARAM_MIN, PARAM_MAX, 61).reshape(-1, 1)
    predicted = np.asarray(saved.predict(grid, out_of_bounds="clip"), dtype=float)
    low = y_train[y_train[:, v_idx] < 0.0, v_idx].max()
    high = y_train[y_train[:, v_idx] > 0.0, v_idx].min()
    in_the_gap = ((predicted[:, v_idx] > low + 0.1 * (high - low))
                  & (predicted[:, v_idx] < high - 0.1 * (high - low)))
    assert in_the_gap.mean() < 0.15, (
        f"{in_the_gap.sum()} of {len(grid)} sweep points predict a V_max the model never "
        f"produces (between {low:.1f} and {high:.1f} mV)")
    assert (predicted[:, n_idx] >= 0.0).all(), "a spike count cannot be negative"

    solver_config = _config(base_user_inputs, work, resources, probe_path)
    solver = CVS0DParamID.init_from_dict(solver_config).param_id

    for name, truth, identified_from_both_sides in (("firing", G_FIRING, True),
                                                    ("silent", G_SILENT, False)):
        v_max, spikes = _features(solver, truth)
        obs_path = os.path.join(work, f"obs_{name}.json")
        with open(obs_path, "w") as handle:
            json.dump(_obs_document(v_max, spikes), handle)

        out_dir = os.path.join(work, f"mcmc_{name}")
        config = _config(base_user_inputs, work, resources, obs_path,
                         param_id_output_dir=out_dir,
                         use_emulator=True,
                         emulator_settings=dict(emulator_settings,
                                                emulator_dir=emulator_dir),
                         do_uq=True,
                         UQ_options={"method": "mcmc", "library": "emcee",
                                     "num_steps": 400, "num_walkers": 16,
                                     "burn_in": 0.5, "cost_type": "gaussian_MLE"},
                         debug_UQ_options={"method": "mcmc", "library": "emcee",
                                           "num_steps": 400, "num_walkers": 16,
                                           "burn_in": 0.5, "cost_type": "gaussian_MLE"})
        runner = CVS0DParamID.init_from_dict(config)
        runner.set_best_param_vals(np.array([truth]))
        runner.run_UQ()

        samples = _posterior(runner.output_dir)
        assert samples.size > 1000, f"{name}: chain is too short to say anything"
        lower, upper = np.percentile(samples, [5.0, 95.0])
        median = float(np.median(samples))
        # printed because the interesting output of a recovery test is the interval, not the
        # pass: a tolerance chosen to be safe hides how well it actually did.
        print(f"[recovery] {name}: true g_Na1_6 {truth:.3f} -> posterior median "
              f"{median:.4f}, 5-95% [{lower:.4f}, {upper:.4f}], "
              f"P(silent) {np.mean(samples < 0.06):.2f}")

        if identified_from_both_sides:
            assert lower <= truth <= upper, (
                f"{name}: true g_Na1_6 {truth} outside the 5-95% interval "
                f"[{lower:.4f}, {upper:.4f}] (median {median:.4f})")
            # The substantive claim is the branch, and it should be certain: a firing cell
            # rules out the silent region outright. Within the firing branch the parameter is
            # only weakly identified -- V_max moves 6 mV against a 6 mV sigma over the whole
            # range and the count carries the rest -- so the interval is wide on purpose and
            # the median tolerance is loose on purpose.
            assert np.mean(samples < 0.06) < 0.05, (
                f"{name}: {100 * np.mean(samples < 0.06):.0f}% of the posterior says the cell "
                f"was silent, but it fired")
            assert abs(median - truth) < 0.2, (
                f"{name}: posterior median {median:.4f} is nowhere near {truth}")
        else:
            # Every conductance below threshold gives exactly the same silence, so the data
            # bounds the parameter from above and says nothing at all below. Asserting
            # concentration here would be asserting a claim the data cannot support -- what
            # can be asserted is that the bound lands on the threshold.
            assert np.mean(samples < 0.06) > 0.9, (
                f"{name}: only {100 * np.mean(samples < 0.06):.0f}% of the posterior is in the "
                f"silent region")
            assert upper < 0.12, (
                f"{name}: the posterior should be cut off near the ~0.055 threshold, but its "
                f"95th percentile is {upper:.4f}")
            assert lower < truth < upper, (
                f"{name}: true g_Na1_6 {truth} outside [{lower:.4f}, {upper:.4f}]")
