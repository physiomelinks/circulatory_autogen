"""Training an emulator against the real solver, and then calibrating on it (issue #333).

The model is ``Simple_ODE_Benchmark``, CA's analytic benchmark: ``dx/dt = -x + p`` and
``dy/dt = -3y + q``, observed as the steady-state averages of x and y. It is the right fixture
for an emulator precisely because it is dull -- smooth and monotone, with no bifurcations or
oscillations -- and because the answer is known in closed form: the features are ``p`` and
``q/3`` to within the leftover transient, measured here at 0.03 and 1e-6 absolute. So the
emulator can be checked against arithmetic rather than against another simulation, which is
what makes a scaling or ordering bug impossible to mistake for a merely mediocre fit.

Two layers, because the two halves fail in different ways and only one needs a heavy optional
dependency:

* the CA half -- design over the params_for_id box, simulate it (across MPI ranks), reduce each
  run to the same features the cost uses, and record the metadata a later run validates
  against. Tested with the fit stubbed out, so it runs everywhere.
* the autoemulate half -- the fit itself, and whether the emulator recovers ``p`` and ``q/3``.
  Skipped when autoemulate is absent.
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


def analytic_features(theta):
    """The benchmark's steady states for theta = (p, q). See the module docstring."""
    theta = np.asarray(theta, dtype=float)
    return np.array([theta[0], theta[1] / 3.0])


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
        'file_prefix': 'Simple_ODE_Benchmark',
        'input_param_file': 'Simple_ODE_Benchmark_parameters.csv',
        'model_type': 'cellml_only',
        'solver': 'CVODE_myokit',
        'param_id_method': 'genetic_algorithm',
        'pre_time': 0.0,
        'sim_time': 8.0,
        'dt': 0.05,
        'DEBUG': True,
        'do_mcmc': False,
        'do_ia': False,
        'plot_predictions': False,
        'solver_info': {'MaximumStep': 0.01, 'MaximumNumberOfSteps': 5000},
        'param_id_obs_path': os.path.join(resources_dir,
                                          'Simple_ODE_Benchmark_obs_data.json'),
        'params_for_id_path': os.path.join(resources_dir,
                                           'Simple_ODE_Benchmark_params_for_id.csv'),
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
        assert generate_with_new_architecture(False, config), 'benchmark generation failed'
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
def test_a_trained_emulator_recovers_the_analytic_steady_states(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):
    """End to end with the real library, against arithmetic.

    The benchmark's features are ``p`` and ``q/3``, so "did the emulator work" has an answer
    that does not depend on another simulation, on the emulator's own opinion of itself, or on
    how forgiving a tolerance is chosen. A forgotten inverse transform, a permuted feature
    order or a mixed-up parameter column all move the predictions off that line, and none of
    them would show in the held-out score alone.

    Measured on this fixture: 64 Sobol samples give held-out R2 of 0.99999 and 0.99997.
    """
    pytest.importorskip('autoemulate')

    config = _config(base_user_inputs, resources_dir, temp_output_dir,
                     temp_generated_models_dir,
                     emulator_settings={'num_train_samples': 64, 'sample_type': 'sobol',
                                        'random_seed': 0, 'min_r2': 0.99, 'n_iter': 2,
                                        'n_splits': 2})
    _generate_model(config, mpi_comm)

    trainer = EmulatorTrainer.init_from_dict(config, comm=mpi_comm)
    bundle = trainer.train()
    mpi_comm.Barrier()
    if mpi_comm.Get_rank() != 0:
        return

    assert bundle is not None
    reported = np.asarray(bundle.meta['feature_r2'], dtype=float)
    assert np.all(np.isfinite(reported))
    # A smooth, near-linear response is what an emulator is for; anything less than this on
    # this fixture means something is wrong with the pipeline, not with the emulator.
    assert np.all(reported > 0.99), f'held-out R2 {reported} on an almost linear response'
    bundle.check_quality(0.99)

    mins = np.asarray(trainer.pid.param_id_info['param_mins'], dtype=float)
    maxs = np.asarray(trainer.pid.param_id_info['param_maxs'], dtype=float)
    rng = np.random.default_rng(12345)
    for _ in range(8):
        theta = mins + rng.random(mins.size) * (maxs - mins)
        predicted = bundle.predict(theta)
        expected = analytic_features(theta)
        # Absolute, not relative: p is sampled down to ~0, where a relative tolerance says
        # nothing. 0.1 is generous against a feature range of 6 and the simulator's own 0.03
        # of leftover transient.
        assert predicted == pytest.approx(expected, abs=0.1), (
            f'at p={theta[0]:.4g}, q={theta[1]:.4g} the emulator predicts {predicted}, but '
            f'the benchmark\'s steady states are {expected}')

    # ... and the simulator agrees with the same arithmetic, so the line above is testing the
    # emulator rather than a shared misunderstanding of what the features mean.
    theta = 0.5 * (mins + maxs)
    _, operands_list, _ = trainer.pid.get_cost_obs_and_pred_from_params(theta)
    simulated = np.asarray(trainer.pid.get_obs_output_dict(operands_list[0])['const'],
                           dtype=float)
    assert simulated == pytest.approx(analytic_features(theta), abs=0.05)

    # ... and the cost and gradient paths run on the emulator, which is what it is for.
    emulator_config = dict(config)
    emulator_config['use_emulator'] = True
    emulator_config['do_emulation'] = False
    emulator_config['emulator_settings'] = dict(config['emulator_settings'])
    emulator_config['emulator_settings']['emulator_dir'] = resolve_emulator_dir(config)
    emulator_config['one_rank'] = True
    engine = CVS0DParamID.init_from_dict(emulator_config).param_id
    assert engine.emulates_features is True
    assert np.isfinite(engine.get_cost_from_params(theta))
    gradient = engine.get_gradient(theta)
    assert np.all(np.isfinite(gradient))
    # d(feature)/d(p) is 1 and d(feature)/d(q) is 1/3, so neither parameter can look inert.
    assert np.all(np.abs(gradient) > 0)
