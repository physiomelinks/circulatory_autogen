"""Training an emulator against the real solver, and then calibrating on it (issue #333).

Two layers, because the two halves fail in different ways and only one of them needs a heavy
optional dependency:

* the CA half -- design over the params_for_id box, simulate it (across MPI ranks), reduce each
  run to the same features the cost uses, and record the metadata a later run validates
  against. Tested with the fit stubbed out, so it runs everywhere.
* the autoemulate half -- the fit itself, and whether an emulator trained this way actually
  reproduces the simulator well enough to calibrate on. Skipped when autoemulate is absent.
"""
import os

import numpy as np
import pytest

from emulators.emulator_bundle import EmulatorBundle
from emulators.emulator_trainer import EmulatorTrainer, resolve_emulator_dir
from param_id.paramID import CVS0DParamID
from parsers.PrimitiveParsers import YamlFileParser
from scripts.script_generate_with_new_architecture import generate_with_new_architecture

try:
    from mpi4py import MPI
except ImportError:                                       # pragma: no cover - env dependent
    MPI = None


class _LinearFit:
    """Stands in for a fitted emulator. Module level so joblib can pickle it."""

    def predict(self, x):
        return np.asarray(x, dtype=float)[:, :1] * np.ones((1, 2))


@pytest.fixture
def mpi_comm():
    if MPI is None:
        pytest.skip('mpi4py is required for the emulator training tests')
    return MPI.COMM_WORLD


def _config(base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir,
            **overrides):
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': 'Lotka_Volterra',
        'input_param_file': 'Lotka_Volterra_parameters.csv',
        'model_type': 'cellml_only',
        'solver': 'CVODE_myokit',
        'param_id_method': 'genetic_algorithm',
        'pre_time': 0.0,
        'sim_time': 5.0,
        'dt': 0.01,
        'DEBUG': True,
        'do_mcmc': False,
        'do_ia': False,
        'plot_predictions': False,
        'solver_info': {'MaximumStep': 0.001, 'MaximumNumberOfSteps': 5000},
        'param_id_obs_path': os.path.join(resources_dir, 'Lotka_Volterra_obs_data.json'),
        'params_for_id_path': os.path.join(resources_dir, 'Lotka_Volterra_params_for_id.csv'),
        'param_id_output_dir': temp_output_dir,
        'resources_dir': resources_dir,
        'generated_models_dir': temp_generated_models_dir,
        'do_emulation': True,
        'emulator_settings': {'num_train_samples': 24, 'sample_type': 'sobol',
                              'random_seed': 0, 'min_r2': 0.5, 'n_iter': 2, 'n_splits': 2},
    })
    config.update(overrides)
    return YamlFileParser().parse_user_inputs_file(
        config, obs_path_needed=True, do_generation_with_fit_parameters=False)


def _generate_model(config, comm):
    if comm.Get_rank() == 0:
        assert generate_with_new_architecture(False, config), 'Lotka_Volterra generation failed'
    comm.Barrier()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_training_set_targets_are_the_cost_features(base_user_inputs, resources_dir,
                                                    temp_output_dir,
                                                    temp_generated_models_dir, mpi_comm):
    """The emulator is trained on exactly what the calibration is fitting.

    If the training targets were computed a second way, the emulator could be a faithful
    surrogate of something the cost does not use -- and every downstream number would be
    self-consistently wrong. So the training target at a design point must equal the cost
    path's own feature vector at that point, evaluated through the real solver.
    """
    config = _config(base_user_inputs, resources_dir, temp_output_dir,
                     temp_generated_models_dir)
    _generate_model(config, mpi_comm)

    trainer = EmulatorTrainer.init_from_dict(config, comm=mpi_comm)
    design = trainer.design()
    assert design.shape == (24, len(trainer.pid.param_id_info['param_names']))
    mins = np.asarray(trainer.pid.param_id_info['param_mins'], dtype=float)
    maxs = np.asarray(trainer.pid.param_id_info['param_maxs'], dtype=float)
    assert np.all(design >= mins) and np.all(design <= maxs), 'design left the training box'

    x, y = trainer.evaluate(design)
    if mpi_comm.Get_rank() != 0:
        return
    assert y.shape[1] == len(trainer.feature_labels)
    assert np.all(np.isfinite(y))

    # The check that matters: the same point, through the ordinary cost path.
    _, operands_list, _ = trainer.pid.get_cost_obs_and_pred_from_params(x[0])
    from_cost = np.asarray(trainer.pid.get_obs_output_dict(operands_list[0])['const'],
                           dtype=float)
    assert y[0] == pytest.approx(from_cost, rel=1e-9)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_training_writes_a_checkable_artefact(base_user_inputs, resources_dir, temp_output_dir,
                                              temp_generated_models_dir, mpi_comm,
                                              monkeypatch):
    """Everything except the fit: the metadata a later run refuses on must all be there.

    The fit is stubbed so this covers the CA half without autoemulate -- what is being checked
    is the artefact, not the emulator inside it.
    """
    config = _config(base_user_inputs, resources_dir, temp_output_dir,
                     temp_generated_models_dir)
    _generate_model(config, mpi_comm)

    trainer = EmulatorTrainer.init_from_dict(config, comm=mpi_comm)

    def fake_fit(x, y):
        x_scale = EmulatorBundle.make_scale(x)
        y_scale = EmulatorBundle.make_scale(y)
        return _LinearFit(), [0.97, 0.96], [0.01, 0.02], 'LinearStub', x_scale, y_scale

    monkeypatch.setattr(trainer, 'fit', fake_fit)
    monkeypatch.setattr('emulators.emulator_trainer.require_autoemulate', lambda: None)

    bundle = trainer.train()
    if mpi_comm.Get_rank() != 0:
        assert bundle is None, 'only rank 0 writes the emulator'
        return

    saved = EmulatorBundle.load(resolve_emulator_dir(config))
    assert saved.feature_labels == trainer.feature_labels
    from parsers.PrimitiveParsers import param_entry_labels
    assert saved.param_entry_labels == param_entry_labels(trainer.pid.param_id_info)
    assert saved.meta['design']['num_train_samples'] == 24
    assert saved.meta['design']['sample_type'] == 'sobol'
    # The defaults the emulator will have to serve in place of a model.
    for entry in trainer.pid.param_id_info['param_names']:
        for name in entry:
            assert str(name) in saved.meta['param_defaults']
    # The training design is kept, so the emulator can be extended without re-simulating.
    assert saved.x_train.shape[0] == saved.y_train.shape[0]
    # ... and the artefact refuses a different problem, which is what it exists for.
    from emulators.emulator_bundle import EmulatorQualityError
    with pytest.raises(EmulatorQualityError):
        saved.check_matches({'inputs_sha256': 'a-different-model'})


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_a_trained_emulator_reports_its_accuracy_honestly(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):
    """End to end with the real library -- and the property that actually matters.

    This deliberately does *not* assert that the emulator is accurate. Lotka-Volterra's `max`
    features over the full params_for_id box are a hard thing to emulate (alpha spans 0.1-7 and
    the response spans 20-3900), and a small design genuinely produces a poor emulator: a
    128-sample Gaussian process here scores R2 around 0.2 and 0.5. Pinning a threshold would
    make this a test of how easy the fixture is.

    What must hold regardless is that the emulator's *self-report* is not optimistic. A
    scaling bug -- forgetting to invert the y transform, say -- would leave the held-out score
    looking healthy while predictions came out in the wrong units, and every refusal
    downstream would pass. So the reported held-out R2 is checked against the R2 actually
    achieved at fresh points evaluated on the simulator.
    """
    pytest.importorskip('autoemulate')

    config = _config(base_user_inputs, resources_dir, temp_output_dir,
                     temp_generated_models_dir,
                     emulator_settings={'num_train_samples': 64, 'sample_type': 'sobol',
                                        'random_seed': 0, 'min_r2': -1e9, 'n_iter': 2,
                                        'n_splits': 2, 'models': 'GaussianProcessRBF'})
    _generate_model(config, mpi_comm)

    trainer = EmulatorTrainer.init_from_dict(config, comm=mpi_comm)
    bundle = trainer.train()
    mpi_comm.Barrier()
    if mpi_comm.Get_rank() != 0:
        return

    assert bundle is not None
    reported = np.asarray(bundle.meta['feature_r2'], dtype=float)
    assert np.all(np.isfinite(reported)), 'every feature must be scored, or none can be trusted'
    assert len(bundle.feature_labels) == reported.size

    # Fresh points, evaluated on the simulator, never seen by the fit.
    mins = np.asarray(trainer.pid.param_id_info['param_mins'], dtype=float)
    maxs = np.asarray(trainer.pid.param_id_info['param_maxs'], dtype=float)
    rng = np.random.default_rng(12345)
    truths, predictions = [], []
    for _ in range(12):
        theta = mins + rng.random(mins.size) * (maxs - mins)
        _, operands_list, _ = trainer.pid.get_cost_obs_and_pred_from_params(theta)
        truths.append(np.asarray(trainer.pid.get_obs_output_dict(operands_list[0])['const'],
                                 dtype=float))
        predictions.append(bundle.predict(theta))
    truths, predictions = np.array(truths), np.array(predictions)
    assert np.all(np.isfinite(predictions))

    for col, label in enumerate(bundle.feature_labels):
        residual = float(np.sum((truths[:, col] - predictions[:, col]) ** 2))
        total = float(np.sum((truths[:, col] - np.mean(truths[:, col])) ** 2))
        actual_r2 = 1.0 - residual / total if total > 0 else float('nan')
        assert actual_r2 > reported[col] - 1.0, (
            f'{label}: the emulator reported held-out R2 {reported[col]:.3f} but achieves '
            f'{actual_r2:.3f} against the simulator -- its self-report is not to be trusted')

    # The refusal that all of this exists to support, against the score it really got.
    from emulators.emulator_bundle import EmulatorQualityError
    with pytest.raises(EmulatorQualityError, match='below the configured min_r2'):
        bundle.check_quality(float(np.max(reported)) + 0.01)

    # ... and the cost and gradient paths run on it, which is what the feature is for.
    emulator_config = dict(config)
    emulator_config['use_emulator'] = True
    emulator_config['do_emulation'] = False
    emulator_config['emulator_settings'] = dict(config['emulator_settings'])
    emulator_config['emulator_settings']['emulator_dir'] = resolve_emulator_dir(config)
    emulator_config['one_rank'] = True
    engine = CVS0DParamID.init_from_dict(emulator_config).param_id
    assert engine.emulates_features is True
    nominal = 0.5 * (mins + maxs)
    assert np.isfinite(engine.get_cost_from_params(nominal))
    assert np.all(np.isfinite(engine.get_gradient(nominal)))
