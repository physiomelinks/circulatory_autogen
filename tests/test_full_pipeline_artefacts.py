"""A whole study, end to end, and the files it has to leave behind.

Every stage has its own test. Nothing tested the combination, and the combination
is what the tools downstream depend on: the generated ``plot_outputs.py`` and
CUFLynx's outputs-directory loader both read a set of files produced by different
stages, found by CA's own naming rule. Each of those readers was tested against
fixtures written by hand -- which agree with the reader by construction, and
cannot notice CA renaming a file.

So this runs sensitivity, an emulator, a calibration, a chain and a posterior
predictive check on a small model, and then asserts the artefacts are there and
readable. If a stage stops writing something, or writes it somewhere else, this
fails here rather than in the app that could not open the folder.

CUFLynx imports the same builder (``libcuflynx.checks.full_pipeline_run``) and
points its own loader at the directory, so both sides of the contract are checked
against one real run rather than against each other's assumptions.
"""
import json
import os

import numpy as np
import pytest

from libcuflynx.checks import full_pipeline_run as pipeline

pytest.importorskip("autoemulate", reason="the emulator stage needs [emulation]")
pytest.importorskip("emcee", reason="the chain needs [uq]")


@pytest.fixture(scope="module")
def full_run(tmp_path_factory, resources_dir):
    """One full study, built once and read by every test below.

    Module-scoped deliberately: this is a real run of every stage, and doing it
    per test would multiply a minute by the number of assertions.
    """
    output_dir = str(tmp_path_factory.mktemp("full_pipeline"))
    result = pipeline.build_full_pipeline_run(
        output_dir=output_dir,
        resources_dir=resources_dir,
        generated_models_dir=os.path.join(output_dir, "generated_models"),
    )
    if result is None:
        pytest.skip("not rank 0")
    return result


# --- the files a full run has to leave -----------------------------------------

@pytest.mark.integration
@pytest.mark.slow
def test_every_expected_artefact_is_written(full_run):
    """The names are the contract. Anything reading a finished run finds it by
    these, so a rename is a break even when every stage still succeeds."""
    assert full_run["artefacts"]["missing"] == []


@pytest.mark.integration
@pytest.mark.slow
def test_the_run_directory_is_not_the_directory_it_was_given(full_run):
    """CA names it ``<method>_<prefix>_<obs prefix>`` underneath. Every reader has
    to find it the same way, which is why they all go through find_run_dir."""
    run_dir = full_run["run_dir"]
    assert os.path.basename(run_dir).startswith("genetic_algorithm_")
    assert run_dir != full_run["config"]["param_id_output_dir"]


@pytest.mark.integration
@pytest.mark.slow
def test_the_calibration_wrote_a_point_and_its_names(full_run):
    run_dir = full_run["run_dir"]
    best = np.load(os.path.join(run_dir, "best_param_vals.npy"), allow_pickle=True)
    with open(os.path.join(run_dir, "param_names.csv")) as handle:
        names = [line.strip() for line in handle if line.strip()]

    assert best.size > 0
    # A point estimate whose values cannot be named is not much use to a reader.
    assert len(names) >= best.size


@pytest.mark.integration
@pytest.mark.slow
def test_the_chain_is_three_dimensional(full_run):
    """``(steps, walkers, params)``. Everything downstream reshapes on that, so a
    two-dimensional chain would be silently misread as one long walker."""
    chain = np.load(os.path.join(full_run["run_dir"], "mcmc_chain.npy"),
                    allow_pickle=True)
    assert chain.ndim == 3
    assert chain.shape[0] > 0 and chain.shape[1] > 0


@pytest.mark.integration
@pytest.mark.slow
def test_the_sensitivity_stage_wrote_its_indices(full_run):
    sa_dir = full_run["config"]["sa_options"]["output_dir"]
    assert os.path.isdir(sa_dir)
    written = os.listdir(sa_dir)
    assert any(name.endswith(".csv") for name in written), written


@pytest.mark.integration
@pytest.mark.slow
def test_the_emulator_bundle_is_complete(full_run):
    """A bundle missing its metadata loads and then refuses on first use, which
    is a much worse failure than not being there at all."""
    emu_dir = full_run["config"]["emulator_settings"]["emulator_dir"]
    for name in ("emulator_metadata.json", "training_data.npz"):
        assert os.path.isfile(os.path.join(emu_dir, name)), name

    with open(os.path.join(emu_dir, "emulator_metadata.json")) as handle:
        meta = json.load(handle)
    for key in ("param_entry_labels", "feature_labels", "feature_r2", "fingerprint"):
        assert key in meta, key


# --- the posterior predictive check --------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
def test_coverage_is_reported_at_both_levels(full_run):
    coverage = full_run["coverage"]
    assert coverage["num_observables"] > 0
    for level in ("0.8", "0.95"):
        row = coverage["levels"][level]
        assert 0.0 <= row["predictive_coverage"] <= 1.0
        assert 0.0 <= row["sample_interval_coverage"] <= 1.0
        assert len(row["per_observable"]) == coverage["num_observables"] + \
            coverage["num_observables_skipped"]


@pytest.mark.integration
@pytest.mark.slow
def test_the_predictions_line_up_with_the_observables(full_run):
    """One column per observable and one row per draw; a transposed or ragged
    array would still save and would plot as nonsense."""
    path = os.path.join(full_run["run_dir"], "posterior_predictive.npz")
    with np.load(path, allow_pickle=True) as data:
        predictions = data["predictions"]
        ground_truth = data["ground_truth"]
        labels = data["labels"]

    assert predictions.shape[0] == pipeline.POSTERIOR_DRAWS
    assert predictions.shape[1] == ground_truth.size == len(labels)


@pytest.mark.integration
@pytest.mark.slow
def test_traces_are_kept_for_the_draws_that_were_asked_for(full_run):
    """Kept for a few draws only, and every trace in a block the same length --
    a block of differing lengths cannot be plotted as a fan."""
    path = os.path.join(full_run["run_dir"], pipeline.SERIES_ARTEFACT)
    assert os.path.isfile(path)

    with np.load(path, allow_pickle=True) as data:
        blocks = [data[name] for name in data.files if name.startswith("y|")]
        meta = json.loads(str(data["__meta__"]))

    assert blocks, "no traces were kept"
    for block in blocks:
        assert block.shape[0] <= pipeline.POSTERIOR_SERIES_DRAWS
        assert block.shape[1] > 1
    assert meta["observables"], "traces with nothing to draw across them"


# --- what CUFLynx will read ----------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
def test_the_directory_satisfies_the_contract_cuflynx_reads(full_run):
    """The same set CUFLynx's loader looks for.

    Asserted here as well as there because CA is where the files are written: a
    stage that stops writing one should fail in the repo that changed, not in the
    app that could no longer open the folder.
    """
    run_dir = full_run["run_dir"]
    present = set(full_run["artefacts"]["present"])

    # The calibration panel.
    assert "best_param_vals.npy" in present
    # The UQ panel, for a run CUFLynx did not produce: it has no
    # uq_posterior_samples.npy of its own to fall back on.
    assert "mcmc_chain.npy" in present
    assert "param_names.csv" in present
    # The coverage and posterior-predictive panels.
    assert "posterior_predictive_coverage.json" in present
    assert "posterior_predictive.npz" in present

    with open(os.path.join(run_dir, "posterior_predictive_coverage.json")) as handle:
        saved = json.load(handle)
    assert saved["used_emulator"] is False
    assert saved["num_samples"] == pipeline.POSTERIOR_DRAWS
