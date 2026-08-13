"""UQ on an emulator: the same posterior, in seconds instead of tens of minutes.

The end-to-end UQ tests in test_UQ.py sample the real model, so every likelihood evaluation is
a solver call and the cheapest of them takes ~9 minutes (the pyMC one exceeds 50 and was killed
before finishing). That is too slow to run on a pull request, which means the only tests that
check UQ *end to end* are the ones that never run.

An emulator removes exactly that cost: it is trained once from a few dozen solver calls, and
after that a likelihood evaluation is a matrix multiply. If the emulator is faithful, the
posterior sampled through it is the same posterior -- which is the claim this file exists to
check, on Simple_ODE_Benchmark, whose answer is known in closed form.

Settings were chosen by sweeping (train time, sampling time, held-out R2, posterior error):

    candidate      train s   mcmc s      r2   mean err   sd err
    RBF n=24           2.8      8.3  0.4127     0.6996   0.0929
    RBF n=48           0.7      9.1  1.0000     0.0068   0.0034   <- used here
    RBF n=96           0.8      9.0  1.0000     0.0179   0.0044
    Poly n=24          1.1      3.7 -0.8397     2.6923   2.6486
    Poly n=48          1.2      4.5 -0.3037     4.9247   0.2298
    GP-RBF n=24        4.6    174.9  0.9992     0.0136   0.0208
    GP-RBF n=48        2.8    197.3  0.9948     0.1763   0.0993

Three things that table decides:

* 24 training points is not enough (R2 0.41), and the posterior is wrong by 0.7 -- seven times
  its own standard deviation. 48 is the threshold, and the margin at 48 is large.
* PolynomialRegression cannot represent this response at all (negative R2 -- worse than
  predicting the mean), so it is not a matter of more samples.
* GaussianProcess is accurate but its prediction cost dominates: same chain, 20x the wall clock.
  Accuracy alone is the wrong thing to select an emulator on when the point is speed.
"""
import json
import os

import numpy as np
import pytest

from param_id.paramID import CVS0DParamID
from parsers.PrimitiveParsers import YamlFileParser
from scripts.script_generate_with_new_architecture import generate_with_new_architecture

try:
    from mpi4py import MPI
except ImportError:                                            # pragma: no cover
    MPI = None

# Simple_ODE_Benchmark's steady state, and the spread its observation noise implies -- the same
# constants test_UQ.py checks the full-model posterior against.
ANALYTICAL_SOLUTION = np.array([1.0, 1.0])
ANALYTICAL_STD = np.array([0.1, 0.3])

#: Walkers for emcee, chains for pymc -- the same budget either way, so the two samplers are
#: compared on equal terms rather than on how each happens to be configured.
NUM_WALKERS = 20

#: The sweep's pick: fast to train, faithful, and cheap to evaluate.
EMULATOR_SETTINGS = {
    'models': 'RadialBasisFunctions',
    'num_train_samples': 48,
    'sample_type': 'sobol',
    'random_seed': 0,
    'n_iter': 2,
    'n_splits': 2,
}


@pytest.fixture
def mpi_comm():
    if MPI is None:
        pytest.skip('mpi4py is required for the emulator UQ test')
    return MPI.COMM_WORLD


def _pymc_available():
    try:
        import pymc  # noqa: F401

        return True
    except ImportError:
        return False


SAMPLERS = [
    pytest.param('emcee', id='emcee'),
    pytest.param('pymc', id='pymc',
                 marks=pytest.mark.skipif(not _pymc_available(),
                                          reason='the pyMC backend needs the [uq] extra')),
]


def _emulator_available():
    try:
        from emulators.emulator_trainer import autoemulate_available

        return autoemulate_available()
    except Exception:
        return False


def _uq_options(library):
    options = {
        'method': 'mcmc',
        'library': library,
        'num_steps': 3000,
        'num_walkers': NUM_WALKERS,
        'burn_in': 0.5,
        'cost_type': 'gaussian_MLE',
    }
    if library == 'pymc':
        # pymc runs num_tune warm-up iterations *on top of* num_steps, per chain, and its
        # Metropolis steps one parameter at a time -- so the default 1000 would triple an
        # already heavier chain. Cheap here because each evaluation is an emulator prediction
        # rather than a solve, but there is no reason to pay for warm-up this test does not need.
        options['num_tune'] = 200
        options['num_steps'] = 1500
    return options


def _config(resources_dir, output_dir, generated_models_dir, emulator_dir,
            library='emcee', **overrides):
    config = {
        'file_prefix': 'Simple_ODE_Benchmark',
        'input_param_file': 'Simple_ODE_Benchmark_parameters.csv',
        'model_type': 'cellml_only',
        'solver': 'CVODE_myokit',
        'param_id_method': 'genetic_algorithm',
        'pre_time': 0.0,
        'sim_time': 8.0,
        'dt': 0.05,
        'DEBUG': False,
        'do_uq': True,
        'do_ia': False,
        'do_sensitivity': False,
        'plot_predictions': False,
        'solver_info': {'MaximumStep': 0.01, 'MaximumNumberOfSteps': 5000},
        'param_id_obs_path': os.path.join(resources_dir,
                                          'Simple_ODE_Benchmark_obs_data.json'),
        'params_for_id_path': os.path.join(resources_dir,
                                           'Simple_ODE_Benchmark_params_for_id.csv'),
        'param_id_output_dir': output_dir,
        'resources_dir': resources_dir,
        'generated_models_dir': generated_models_dir,
        'do_emulation': True,
        'use_emulator': True,
        'emulator_settings': {**EMULATOR_SETTINGS, 'emulator_dir': emulator_dir},
        'emulator_dir': emulator_dir,
        'UQ_options': _uq_options(library),
    }
    config.update(overrides)
    return YamlFileParser().parse_user_inputs_file(
        config, obs_path_needed=True, do_generation_with_fit_parameters=False)


def _train_emulator(config, comm):
    from emulators.emulator_trainer import EmulatorTrainer

    trainer = EmulatorTrainer.init_from_dict(config, comm=comm)
    return trainer.train()


@pytest.mark.integration
@pytest.mark.mpi
@pytest.mark.parametrize('library', SAMPLERS)
@pytest.mark.skipif(not _emulator_available(),
                    reason='autoemulate is required for the emulator UQ test')
def test_uq_on_an_emulator_recovers_the_analytic_posterior(
        library, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):
    """Train an emulator, sample through it, and check the posterior is the right one.

    Run for both samplers. Two independent samplers agreeing on a posterior whose answer is
    known in closed form is a much stronger statement than either one matching alone: a bug in
    the shared machinery below them (the cost, the priors, the emulator) would move both, but a
    bug in one sampler moves only that one.

    It is also the only affordable way to exercise the pyMC backend end to end. Against the real
    model that test exceeded 50 minutes and was killed unfinished, because pyMC's Metropolis
    takes one likelihood evaluation per parameter per step. On an emulator each of those
    evaluations is a matrix multiply, so the same chain costs seconds.

    Deliberately *not* marked slow: the whole point is that this runs on a pull request, where
    the full-model equivalents cannot. If it stops being quick, that is a regression in its own
    right -- an emulator whose sampling costs what the model costs has no reason to exist.
    """
    emulator_dir = os.path.join(temp_output_dir, 'emulator')
    config = _config(resources_dir, temp_output_dir, temp_generated_models_dir, emulator_dir,
                     library=library)

    if mpi_comm.Get_rank() == 0:
        assert generate_with_new_architecture(False, config), 'benchmark generation failed'
    mpi_comm.Barrier()

    bundle = _train_emulator(config, mpi_comm)
    if mpi_comm.Get_rank() != 0:
        return

    # The emulator has to be faithful before its posterior means anything. Checked here rather
    # than left implicit, so a bad posterior below is attributable: a broken surrogate and a
    # broken sampler fail this test in the same place otherwise.
    for entry in (bundle.error_stats() or []):
        if isinstance(entry, dict) and entry.get('r2') is not None:
            assert float(entry['r2']) > 0.99, f"emulator is not faithful: {entry}"

    param_id = CVS0DParamID.init_from_dict({**config, 'mcmc_instead': True})
    param_id.run_UQ()

    chain = np.load(os.path.join(param_id.output_dir, 'mcmc_chain.npy'))
    flat = chain[chain.shape[0] // 2:, :, :].reshape(-1, chain.shape[2])

    posterior_mean = flat.mean(axis=0)
    posterior_sd = flat.std(axis=0)
    print(f'[{library}] posterior mean {posterior_mean} (truth {ANALYTICAL_SOLUTION})')
    print(f'[{library}] posterior sd   {posterior_sd} (truth {ANALYTICAL_STD})')

    assert chain.shape[1] == NUM_WALKERS, (
        f'{library} produced {chain.shape[1]} walkers, not the {NUM_WALKERS} asked for')

    # Both moments are checked, not just the centre. A sampler that finds the right mode but
    # mis-sizes its spread has not recovered the posterior -- it has recovered a point estimate
    # with a plausible-looking error bar, which is exactly what UQ is supposed to replace.
    # Tolerances are far wider than the sweep's measured error (0.007 / 0.003), so this fails on
    # a real regression rather than on sampling noise.
    assert posterior_mean == pytest.approx(ANALYTICAL_SOLUTION, abs=0.05), (
        f'{library} posterior mean {posterior_mean} != {ANALYTICAL_SOLUTION}')
    assert posterior_sd == pytest.approx(ANALYTICAL_STD, abs=0.08), (
        f'{library} posterior sd {posterior_sd} != {ANALYTICAL_STD}')


@pytest.mark.integration
@pytest.mark.mpi
@pytest.mark.skipif(not _emulator_available(),
                    reason='autoemulate is required for the emulator UQ test')
def test_the_uq_run_writes_its_posterior_summary(
        resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):
    """The statistics file is the artifact a user reads, so a UQ run that produces a chain but
    no summary is only half a run -- and it must not have overwritten the calibration's best fit
    to produce it."""
    emulator_dir = os.path.join(temp_output_dir, 'emulator')
    config = _config(resources_dir, temp_output_dir, temp_generated_models_dir, emulator_dir)

    if mpi_comm.Get_rank() == 0:
        assert generate_with_new_architecture(False, config), 'benchmark generation failed'
    mpi_comm.Barrier()

    _train_emulator(config, mpi_comm)
    if mpi_comm.Get_rank() != 0:
        return

    param_id = CVS0DParamID.init_from_dict({**config, 'mcmc_instead': True})
    param_id.run_UQ()

    path = os.path.join(param_id.output_dir, 'mcmc_statistics.json')
    assert os.path.isfile(path), 'a UQ run must write its posterior summary'
    with open(path) as handle:
        document = json.load(handle, parse_constant=_reject_non_standard_json)

    assert document['num_samples'] > 0
    assert set(document['parameters']), 'the summary must name its parameters'
    for name, row in document['parameters'].items():
        assert row['q2.5'] <= row['median'] <= row['q97.5'], name
        assert row['sd'] > 0, name
    # A UQ run with no calibration behind it is the only case that may write a best fit, and it
    # has to say so rather than pass a posterior median off as an optimum.
    assert document['source'] in ('calibration', 'posterior_median')


def _reject_non_standard_json(value):
    raise AssertionError(f'mcmc_statistics.json contains non-standard JSON constant {value!r}')
