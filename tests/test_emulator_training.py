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
from types import SimpleNamespace

import numpy as np
import pytest

from emulators.emulator_bundle import (EmulatorBundle, EmulatorQualityError, EmulatorReuseError,
                                       fingerprint)
from emulators.emulator_trainer import EmulatorTrainer, resolve_emulator_dir
from param_id.paramID import CVS0DParamID, emulated_feature_labels
from parsers.PrimitiveParsers import ObsAndParamDataParser, YamlFileParser, param_entry_labels
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
        'model_type': 'cellml',
        'solver': 'CVODE_myokit',
        'param_id_method': 'genetic_algorithm',
        'pre_time': 0.0,
        'sim_time': 8.0,
        'dt': 0.05,
        'DEBUG': True,
        'do_uq': False,
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
        validation = {
            'r2': [0.97, 0.96], 'rmse': [0.01, 0.02], 'mae': [0.01, 0.02],
            'bias': [0.0, 0.0], 'max_abs_error': [0.02, 0.04],
            'nrmse': [0.01, 0.02],
            'theta': x[:2], 'y_true': y[:2], 'y_pred': y[:2],
        }
        return _LinearFit(), validation, 'LinearStub', x_scale, y_scale

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


@pytest.mark.unit
def test_the_validation_report_measures_error_in_real_units():
    """The statistics and the points must be in the feature's own units.

    The emulator is fitted in a scaled space; reporting its error there would give
    a bias and an RMSE that mean nothing next to the observation they approximate,
    and a parity plot whose axes are not the quantity being emulated.
    """
    from emulators.emulator_trainer import _validation_report

    class DoublingStub:
        """Predicts exactly 0.1 (scaled) above the truth, for a known bias."""

        def predict(self, x):
            return np.asarray(x, dtype=float) + 0.1

    # y = x in the scaled space; real units are y*10 + 100, x*2 + 1.
    x_test = np.array([[0.0], [0.5], [1.0]])
    y_test = np.array([[0.0], [0.5], [1.0]])
    x_scale = {'shift': [1.0], 'span': [2.0]}
    y_scale = {'shift': [100.0], 'span': [10.0]}

    report = _validation_report(DoublingStub(), x_test, y_test, x_scale, y_scale)

    # A constant +0.1 scaled error is +1.0 in real units, on every point.
    assert report['bias'][0] == pytest.approx(1.0)
    assert report['mae'][0] == pytest.approx(1.0)
    assert report['rmse'][0] == pytest.approx(1.0)
    assert report['max_abs_error'][0] == pytest.approx(1.0)
    # And the points come back as the parameters and features they really are.
    assert report['theta'].flatten() == pytest.approx([1.0, 2.0, 3.0])
    assert report['y_true'].flatten() == pytest.approx([100.0, 105.0, 110.0])
    assert report['y_pred'].flatten() == pytest.approx([101.0, 106.0, 111.0])
    # nrmse is against the feature's own spread (10 here), so features in
    # different units can be compared.
    assert report['nrmse'][0] == pytest.approx(0.1)


@pytest.mark.unit
def test_a_degenerate_test_column_scores_nan_not_a_perfect_fit():
    from emulators.emulator_trainer import _validation_report

    class ConstantStub:
        def predict(self, x):
            return np.zeros((len(x), 1))

    report = _validation_report(
        ConstantStub(), np.array([[0.0], [1.0]]), np.array([[0.0], [0.0]]),
        {'shift': [0.0], 'span': [1.0]}, {'shift': [0.0], 'span': [1.0]},
    )
    assert np.isnan(report['r2'][0])
    assert np.isnan(report['nrmse'][0])


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
    # The error data an Analysis view is drawn from, written beside the emulator.
    saved = EmulatorBundle.load(resolve_emulator_dir(config))
    points = saved.error_points()
    assert points is not None, 'the held-out points must survive a save/load'
    assert points['theta'].shape[1] == len(saved.param_entry_labels)
    assert points['y_true'].shape == points['y_pred'].shape
    stats = saved.error_stats()
    assert len(stats) == len(saved.feature_labels)
    assert all(row['bias'] is not None for row in stats)

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


# --------------------------------------------------------------- reuse_samples (#333 follow-up)
#
# Training is two costs: the N truth-model runs, which is the whole reason an emulator exists,
# and the fit, which is seconds. ``reuse_samples`` refits the samples a previous run already
# simulated and saved, so a second emulator model or a different test_fraction costs the fit
# alone. What has to be guarded is that the samples still describe *this* problem: refitting a
# stale design produces an emulator that is confidently wrong about a study it was never
# trained for, and nothing downstream can tell.


def _benchmark_infos(base_user_inputs, resources_dir, tmp_path):
    """obs_info / protocol_info / param_id_info for the benchmark, with nothing generated.

    Parsed by the same parsers ``CVS0DParamID`` uses, so the fingerprint, the parameter labels
    and the feature labels are the real ones -- which is exactly what the reuse checks compare.
    """
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': 'Simple_ODE_Benchmark',
        'input_param_file': 'Simple_ODE_Benchmark_parameters.csv',
        'model_type': 'cellml',
        'resources_dir': resources_dir,
        'pre_time': 0.0,
        'sim_time': 8.0,
        'dt': 0.05,
        'param_id_obs_path': os.path.join(resources_dir, 'Simple_ODE_Benchmark_obs_data.json'),
        'params_for_id_path': os.path.join(resources_dir,
                                           'Simple_ODE_Benchmark_params_for_id.csv'),
        'param_id_output_dir': str(tmp_path / 'param_id_output'),
    })
    config = YamlFileParser().parse_user_inputs_file(
        config, obs_path_needed=True, do_generation_with_fit_parameters=False)

    scratch = str(tmp_path / 'parsed')
    os.makedirs(scratch, exist_ok=True)
    parser = ObsAndParamDataParser()
    parsed = parser.parse_obs_data_json(
        param_id_obs_path=config['param_id_obs_path'], pre_time=config['pre_time'],
        sim_time=config['sim_time'], model_type=config['model_type'],
        method=config['solver_info'].get('method'))
    obs_info = parser.process_obs_info(gt_df=parsed['gt_df'], output_dir=scratch, dt=config['dt'])
    protocol_info = parser.process_protocol_and_weights(
        gt_df=parsed['gt_df'], protocol_info=parsed['protocol_info'], dt=config['dt'])
    return obs_info, protocol_info, parser.get_param_id_info(config['params_for_id_path'])


def _stub_trainer(directory, obs_info, protocol_info, param_id_info, **settings):
    """A trainer over parsed infos and no solver: enough for everything reuse touches."""
    pid = SimpleNamespace(sim_helper=SimpleNamespace(emulates_features=False),
                          obs_info=obs_info, protocol_info=protocol_info,
                          param_id_info=param_id_info, model_path=None)
    trainer = EmulatorTrainer(pid, settings)
    trainer.output_dir = str(directory)
    return trainer


def _write_previous_bundle(directory, obs_info, protocol_info, param_id_info,
                           num_samples=6, with_samples=True):
    """A bundle on disk the way a previous training run would have left it."""
    labels = emulated_feature_labels(obs_info)
    mins = np.asarray(param_id_info['param_mins'], dtype=float)
    maxs = np.asarray(param_id_info['param_maxs'], dtype=float)
    x = mins + np.random.default_rng(0).random((num_samples, mins.size)) * (maxs - mins)
    y = x[:, :1] * np.arange(1.0, len(labels) + 1.0)

    meta = {
        'param_entry_labels': [str(label) for label in param_entry_labels(param_id_info)],
        'param_mins': [float(v) for v in mins],
        'param_maxs': [float(v) for v in maxs],
        'param_names': [[str(n) for n in entry] for entry in param_id_info['param_names']],
        'param_defaults': {str(name): 1.0
                           for entry in param_id_info['param_names'] for name in entry},
        'feature_labels': labels,
        'feature_r2': [0.99] * len(labels),
        'feature_rmse': [0.01] * len(labels),
        'x_scale': EmulatorBundle.make_scale(x),
        'y_scale': EmulatorBundle.make_scale(y),
        # Deliberately not this run's settings, so the test can tell a carried-through design
        # block from one rebuilt out of the current (ignored) ones.
        'design': {'sample_type': 'latin_hypercube', 'num_train_samples': num_samples + 1,
                   'num_used': num_samples, 'num_failed': 1, 'random_seed': 11,
                   'log_scale_params': False, 'reused_samples': False},
        'fingerprint': fingerprint(param_id_info, obs_info, protocol_info),
    }
    bundle = EmulatorBundle(_LinearFit(), meta,
                            x_train=x if with_samples else None,
                            y_train=y if with_samples else None)
    bundle.save(str(directory))
    return bundle


@pytest.mark.unit
def test_reuse_without_a_previous_run_refuses_and_names_the_directory(
        base_user_inputs, resources_dir, tmp_path):
    """The first run is the one that pays for the simulations; say so, and say where."""
    infos = _benchmark_infos(base_user_inputs, resources_dir, tmp_path)
    directory = tmp_path / 'emulator'
    trainer = _stub_trainer(directory, *infos, reuse_samples=True)

    with pytest.raises(EmulatorReuseError) as excinfo:
        trainer.load_previous_samples()
    message = str(excinfo.value)
    assert str(directory) in message, 'the refusal must name the directory it looked in'
    assert 'reuse_samples: false' in message


@pytest.mark.unit
def test_reuse_refuses_a_bundle_that_kept_no_samples(base_user_inputs, resources_dir, tmp_path):
    """A bundle saved before CA persisted its design has an emulator but nothing to refit."""
    infos = _benchmark_infos(base_user_inputs, resources_dir, tmp_path)
    directory = tmp_path / 'emulator'
    _write_previous_bundle(directory, *infos, with_samples=False)
    trainer = _stub_trainer(directory, *infos, reuse_samples=True)

    with pytest.raises(EmulatorReuseError) as excinfo:
        trainer.load_previous_samples()
    message = str(excinfo.value)
    assert 'training_data.npz' in message
    assert str(directory) in message and 'reuse_samples: false' in message


@pytest.mark.unit
def test_reuse_refuses_samples_simulated_for_a_different_problem(
        base_user_inputs, resources_dir, tmp_path):
    """Widened bounds mean a different theta -> features map, so the saved y are not this run's.

    Nothing about the samples changes when the study does, which is why this has to be checked
    rather than noticed -- a refit of them would look entirely healthy and be wrong.
    """
    obs_info, protocol_info, param_id_info = _benchmark_infos(
        base_user_inputs, resources_dir, tmp_path)
    directory = tmp_path / 'emulator'
    _write_previous_bundle(directory, obs_info, protocol_info, param_id_info)

    widened = dict(param_id_info)
    widened['param_maxs'] = np.asarray(param_id_info['param_maxs'], dtype=float) * 2.0
    trainer = _stub_trainer(directory, obs_info, protocol_info, widened, reuse_samples=True)

    with pytest.raises(EmulatorQualityError) as excinfo:
        trainer.load_previous_samples()
    message = str(excinfo.value)
    assert 'stale' in message
    assert 'reuse_samples: false' in message, 'the refusal must say how to get a valid emulator'


@pytest.mark.unit
def test_reuse_refuses_when_the_observables_changed(base_user_inputs, resources_dir, tmp_path):
    """Same idea one level up: the saved targets are features this run no longer computes."""
    obs_info, protocol_info, param_id_info = _benchmark_infos(
        base_user_inputs, resources_dir, tmp_path)
    directory = tmp_path / 'emulator'
    bundle = _write_previous_bundle(directory, obs_info, protocol_info, param_id_info)
    bundle.meta['feature_labels'] = ['something the obs_data no longer asks for']
    bundle.save(str(directory))

    trainer = _stub_trainer(directory, obs_info, protocol_info, param_id_info,
                            reuse_samples=True)
    with pytest.raises(EmulatorQualityError, match='trained for observables'):
        trainer.load_previous_samples()


@pytest.mark.unit
def test_reuse_carries_the_previous_design_and_reports_the_count_it_really_uses(
        base_user_inputs, resources_dir, tmp_path, capsys):
    """Provenance must not claim simulations this run did not perform.

    ``num_train_samples``, ``sample_type`` and ``log_scale_params`` describe a design reuse does
    not rebuild, so they are ignored here and the saved block is carried through as it stands --
    with ``reused_samples`` marking the fit as one that ran no simulator at all.
    """
    infos = _benchmark_infos(base_user_inputs, resources_dir, tmp_path)
    directory = tmp_path / 'emulator'
    _write_previous_bundle(directory, *infos, num_samples=6)
    trainer = _stub_trainer(directory, *infos, reuse_samples=True, num_train_samples=128,
                            sample_type='random', random_seed=7)

    x, y, design_meta = trainer.load_previous_samples()

    assert len(x) == 6 and len(y) == 6
    assert design_meta['reused_samples'] is True
    assert design_meta['num_used'] == 6
    # From the run that simulated them, not from this run's ignored settings.
    assert design_meta['sample_type'] == 'latin_hypercube'
    assert design_meta['num_train_samples'] == 7 and design_meta['num_failed'] == 1
    assert design_meta['random_seed'] == 11
    # ... and the seed that did apply here, because it still moves the fit and the split.
    assert design_meta['fit_random_seed'] == 7

    if trainer.rank == 0:
        printed = capsys.readouterr().out
        assert 'reuse_samples' in printed and '6 samples' in printed
        # The requested count disagrees with what is on disk: say which one is being used.
        assert 'num_train_samples' in printed and '128' in printed


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_reuse_refits_the_saved_samples_without_running_the_model(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm,
        monkeypatch):
    """The point of the setting: a second fit that runs no simulations at all.

    The truth-model entry point is made to raise, so a single simulation would fail the test
    rather than merely make it slow -- which is the only way to tell "reused the samples" from
    "re-ran them and got the same answer".
    """
    config = _config(base_user_inputs, resources_dir, temp_output_dir,
                     temp_generated_models_dir)
    _generate_model(config, mpi_comm)

    def fake_fit(x, y):
        validation = {'r2': [0.97, 0.96], 'rmse': [0.01, 0.02], 'mae': [0.01, 0.02],
                      'bias': [0.0, 0.0], 'max_abs_error': [0.02, 0.04], 'nrmse': [0.01, 0.02],
                      'theta': x[:2], 'y_true': y[:2], 'y_pred': y[:2]}
        return (_LinearFit(), validation, 'LinearStub',
                EmulatorBundle.make_scale(x), EmulatorBundle.make_scale(y))

    monkeypatch.setattr('emulators.emulator_trainer.require_autoemulate', lambda: None)
    trainer = EmulatorTrainer.init_from_dict(config, comm=mpi_comm)
    monkeypatch.setattr(trainer, 'fit', fake_fit)
    first = trainer.train()
    mpi_comm.Barrier()

    # A second run that changes only fit settings, and asks for the samples back.
    reuse_config = _config(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir,
        emulator_settings={'reuse_samples': True, 'num_train_samples': 512,
                           'sample_type': 'random', 'random_seed': 7, 'min_r2': 0.5,
                           'n_iter': 3, 'n_splits': 2})
    reuser = EmulatorTrainer.init_from_dict(reuse_config, comm=mpi_comm)
    assert resolve_emulator_dir(reuse_config) == resolve_emulator_dir(config)

    import param_id.fd_backend as fd_backend

    def no_simulations(*args, **kwargs):
        raise AssertionError('reuse_samples must not evaluate the truth model')

    monkeypatch.setattr(fd_backend, 'observable_features', no_simulations)
    monkeypatch.setattr(reuser, 'design', no_simulations)
    monkeypatch.setattr(reuser, 'evaluate', no_simulations)
    monkeypatch.setattr(reuser, 'fit', fake_fit)

    bundle = reuser.train()
    if mpi_comm.Get_rank() != 0:
        assert bundle is None, 'only rank 0 writes the emulator, reused or not'
        return

    assert bundle is not None
    saved = EmulatorBundle.load(resolve_emulator_dir(reuse_config))
    # A bundle indistinguishable in kind from a freshly trained one ...
    assert saved.feature_labels == reuser.feature_labels
    assert saved.x_train == pytest.approx(first.x_train)
    assert saved.y_train == pytest.approx(first.y_train)
    # ... whose provenance does not claim simulations this run did not perform.
    assert saved.meta['design']['reused_samples'] is True
    assert saved.meta['design']['num_used'] == len(first.x_train)
    assert saved.meta['design']['num_train_samples'] == 24, 'the design that was really run'
    assert saved.meta['design']['fit_random_seed'] == 7
    assert first.meta['design']['reused_samples'] is False


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_a_reused_fit_emulates_the_same_function_as_the_original(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):
    """Refitting saved samples with a different emulator must still emulate the benchmark.

    Both fits see the same simulated features, so both have to land on the same function -- and
    that function is known in closed form (``p`` and ``q/3``), so this is checked against
    arithmetic rather than against the original fit's opinion of itself.
    """
    pytest.importorskip('autoemulate')

    config = _config(base_user_inputs, resources_dir, temp_output_dir,
                     temp_generated_models_dir,
                     emulator_settings={'num_train_samples': 64, 'sample_type': 'sobol',
                                        'random_seed': 0, 'min_r2': 0.9, 'n_iter': 2,
                                        'n_splits': 2})
    _generate_model(config, mpi_comm)
    original = EmulatorTrainer.init_from_dict(config, comm=mpi_comm).train()
    mpi_comm.Barrier()

    # A different emulator and a different fit seed over the same samples: the case the setting
    # exists for, and the one that used to cost the 64 simulations again. Built on every rank,
    # like any other trainer -- the engine's constructor is collective, and reuse must leave the
    # ranks that have nothing to do returning cleanly rather than waiting on rank 0.
    reuse_config = _config(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir,
        emulator_settings={'reuse_samples': True, 'random_seed': 5, 'min_r2': 0.9,
                           'n_iter': 2, 'n_splits': 2, 'models': 'PolynomialRegression'})
    reuser = EmulatorTrainer.init_from_dict(reuse_config, comm=mpi_comm)
    refitted = reuser.train()
    if mpi_comm.Get_rank() != 0:
        assert refitted is None, 'a reused fit still writes from rank 0 alone'
        return

    assert refitted is not None
    assert 'Polynomial' in refitted.meta['model_name'], (
        f"the reused fit should have used the emulator this run asked for, not "
        f"{refitted.meta['model_name']}")
    assert refitted.x_train == pytest.approx(original.x_train)

    reported = np.asarray(refitted.meta['feature_r2'], dtype=float)
    assert np.all(np.isfinite(reported)) and np.all(reported > 0.99), (
        f'held-out R2 {reported} from a refit of an almost linear response')
    refitted.check_quality(0.99)

    mins = np.asarray(reuser.pid.param_id_info['param_mins'], dtype=float)
    maxs = np.asarray(reuser.pid.param_id_info['param_maxs'], dtype=float)
    rng = np.random.default_rng(12345)
    refit_error, disagreement = [], []
    for _ in range(16):
        theta = mins + rng.random(mins.size) * (maxs - mins)
        expected = analytic_features(theta)
        refit_error.append(np.max(np.abs(refitted.predict(theta) - expected)))
        disagreement.append(np.max(np.abs(refitted.predict(theta) - original.predict(theta))))
    # Absolute, like the fresh-training test above: p is sampled down to ~0, where a relative
    # tolerance says nothing. Measured on this fixture, the refit is off the analytic answer by
    # 0.023 and off the original fit by 0.006, against a feature range of about 6 -- so 0.1
    # leaves room for the fit's own variation while a wrong-samples, scaling or ordering bug
    # (which moves these by whole units) still fails.
    print(f'[test] refit max error {max(refit_error):.4g}, '
          f'max disagreement with the original {max(disagreement):.4g}')
    assert max(refit_error) < 0.1, (
        f'the emulator refitted from the saved samples is off the benchmark\'s analytic steady '
        f'states by up to {max(refit_error):.3g}')
    assert max(disagreement) < 0.1, (
        f'the two fits of the same samples disagree by up to {max(disagreement):.3g}')
