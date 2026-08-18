"""End-to-end uncertainty quantification: does the posterior recover a known answer? (#179)

Ported from #367. These are the tests that actually validate UQ rather than its parts: the
Simple_ODE_Benchmark model has an analytically known steady state, so a correct posterior must
centre on it with the right spread. Every unit test in test_uq_*.py can pass while the pipeline
as a whole samples the wrong distribution -- that is what these catch.

All three sample the *real model* -- a real calibration followed by a real chain -- and cost
about 80 minutes between them (the bimodal one alone took 56 on CI). They are marked ``manual``,
so they are not collected at all unless you ask for them:

    ./run_pytest.sh tests/test_UQ.py -n 4 --run-manual -v

What runs instead, on every PR and every default local run, is ``test_uq_on_emulator.py``: the
same posterior-recovery and both-modes-present assertions, sampled through an emulator, in about
five minutes. That covers everything above it in the stack -- the cost, the priors, both
samplers, the statistics. What it does not cover, and what these three exist for, is the
likelihood driven by the actual solver. Run them when you change anything on that path.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from libcuflynx.param_id.paramID import CVS0DParamID
from libcuflynx.scripts.param_id_run_script import run_param_id
# mpi_comm is a fixture defined in test_param_id, so it has to be imported here to be visible to
# these tests -- not merely referenced in their signatures.
#
# Imported as a top-level module, not as `tests.test_param_id`: tests/ has no __init__.py, so
# pytest puts it on sys.path and imports these files top-level anyway -- while `tests` as a
# package resolves against whatever else is installed. At least one dependency ships its own
# top-level `tests` package, which shadows this directory and makes the dotted form fail with
# ModuleNotFoundError in a plain pip environment (it happens to work under the OpenCOR shell,
# which has no such package -- so the dotted form passes locally and breaks in CI).
from test_param_id import _ensure_cellml_model_generated, mpi_comm  # noqa: F401

pymc_installed = True
try:
    import pymc  # noqa: F401
except ImportError:
    pymc_installed = False

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

# The Simple_ODE_Benchmark steady state, and the spread the observation noise implies.
ANALYTICAL_SOLUTION = np.array([1.0, 1.0])
ANALYTICAL_STD = np.array([0.1, 0.3])


def _uq_config(base_user_inputs, resources_dir, temp_output_dir, obs_path, library='emcee',
               num_steps=500, num_walkers=32):
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': 'Simple_ODE_Benchmark',
        'input_param_file': 'Simple_ODE_Benchmark_parameters.csv',
        'params_for_id_file': 'Simple_ODE_Benchmark_params_for_id.csv',
        'model_type': 'cellml',
        'solver': 'CVODE_myokit',
        'param_id_method': 'genetic_algorithm',
        'pre_time': 0.5,
        'sim_time': 10.0,
        'dt': 0.1,
        'DEBUG': False,
        'do_uq': True,
        'plot_predictions': False,
        'do_ia': False,
        'param_id_obs_path': obs_path,
        'param_id_output_dir': temp_output_dir,
        'UQ_options': {
            'method': 'mcmc',
            'library': library,
            'num_steps': num_steps,
            'num_walkers': num_walkers,
            'burn_in': 0.5,
        },
        'optimiser_options': {
            'num_calls_to_function': 10000,
            'cost_convergence': 0.01,
            'max_patience': 5,
            'cost_type': 'gaussian_MLE',
        },
    })
    return config


def _load_chain_after_burn_in(output_dir, burn_in):
    chain_file = os.path.join(output_dir, 'mcmc_chain.npy')
    assert os.path.exists(chain_file), 'MCMC chain file should exist'
    samples = np.load(chain_file)
    return samples[int(samples.shape[0] * burn_in):, :, :]


def _mcmc_object(config, resources_dir, temp_output_dir):
    return CVS0DParamID(
        config['model_path'], config['model_type'], config['param_id_method'], True,
        config['file_prefix'],
        params_for_id_path=config.get('params_for_id_path'),
        param_id_obs_path=config['param_id_obs_path'],
        sim_time=config['sim_time'], pre_time=config['pre_time'], dt=config['dt'],
        param_id_output_dir=temp_output_dir, resources_dir=resources_dir,
        UQ_options=config['UQ_options'], DEBUG=config['DEBUG'], one_rank=True,
    )


def _assert_chain_converged(stats, max_rhat, min_ess):
    for name, row in stats.items():
        assert np.isfinite(row['r_hat']), f'r_hat for {name} could not be computed'
        assert row['r_hat'] < max_rhat, f'r_hat for {name} is too high: {row["r_hat"]}'
        assert row['ess'] > min_ess, f'{name} has too few effective samples: {row["ess"]}'


def _assert_posterior_recovers_the_truth(stats, best_params, rtol, std_atol):
    assert np.allclose(best_params, ANALYTICAL_SOLUTION, rtol=rtol), \
        f'best params {best_params} do not recover {ANALYTICAL_SOLUTION}'

    posterior_std = np.array([row['sd'] for row in stats.values()])
    print(f'Posterior std: {posterior_std}, analytical: {ANALYTICAL_STD}')
    assert np.allclose(posterior_std, ANALYTICAL_STD, atol=std_atol), \
        f'posterior spread {posterior_std} does not match {ANALYTICAL_STD}'


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.manual
@pytest.mark.mpi
def test_mcmc_unimodal_with_validation(base_user_inputs, resources_dir, temp_output_dir,
                                       temp_generated_models_dir, mpi_comm):
    """The baseline: emcee on a unimodal posterior must recover the known steady state, with a
    converged chain and the spread the observation noise implies."""
    rank = mpi_comm.Get_rank()
    config = _uq_config(
        base_user_inputs, resources_dir, temp_output_dir,
        os.path.join(resources_dir, 'Simple_ODE_Benchmark_obs_data.json'))

    _ensure_cellml_model_generated(config, mpi_comm)
    run_param_id(config)

    if rank == 0:
        output_dir = os.path.join(
            temp_output_dir,
            f"{config['param_id_method']}_{config['file_prefix']}_Simple_ODE_Benchmark_obs_data")
        samples = _load_chain_after_burn_in(output_dir, config['UQ_options']['burn_in'])

        mcmc_obj = _mcmc_object(config, resources_dir, temp_output_dir)
        mcmc_obj.plot_mcmc()
        assert list(Path(mcmc_obj.plot_dir).glob('mcmc_cornerplot_*.pdf'))

        stats = mcmc_obj.get_posterior_stats(samples)
        _assert_chain_converged(stats, max_rhat=1.3, min_ess=50)
        _assert_posterior_recovers_the_truth(
            stats, np.load(os.path.join(output_dir, 'best_param_vals.npy')),
            rtol=0.1, std_atol=0.2)

    mpi_comm.Barrier()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.manual
@pytest.mark.mpi
@pytest.mark.skipif(not pymc_installed,
                    reason='the pyMC backend needs the optional [uq] extra')
def test_mcmc_unimodal_with_validation_KDE_likelihood(base_user_inputs, resources_dir,
                                                      temp_output_dir,
                                                      temp_generated_models_dir, mpi_comm):
    """The same posterior via the pyMC backend and a KDE likelihood.

    Two things at once, both worth having: that pyMC reaches the same answer emcee does (the
    real acceptance test for the backend -- a sampler that agrees with another sampler on a
    known posterior is working), and that kernel_density_estimation behaves as a likelihood
    when the target is given as samples rather than a named distribution.
    """
    rank = mpi_comm.Get_rank()
    config = _uq_config(
        base_user_inputs, resources_dir, temp_output_dir,
        os.path.join(_TESTS_DIR, 'test_inputs', 'Simple_ODE_Benchmark_KDE_obs_data.json'),
        library='pymc')

    _ensure_cellml_model_generated(config, mpi_comm)
    run_param_id(config)

    if rank == 0:
        output_dir = os.path.join(
            temp_output_dir,
            f"{config['param_id_method']}_{config['file_prefix']}_"
            f"Simple_ODE_Benchmark_KDE_obs_data")
        samples = _load_chain_after_burn_in(output_dir, config['UQ_options']['burn_in'])

        mcmc_obj = _mcmc_object(config, resources_dir, temp_output_dir)
        mcmc_obj.plot_mcmc()
        assert list(Path(mcmc_obj.plot_dir).glob('mcmc_cornerplot_*.pdf'))

        stats = mcmc_obj.get_posterior_stats(samples)
        _assert_chain_converged(stats, max_rhat=1.2, min_ess=50)
        _assert_posterior_recovers_the_truth(
            stats, np.load(os.path.join(output_dir, 'best_param_vals.npy')),
            rtol=0.25, std_atol=0.1)

    mpi_comm.Barrier()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.manual
@pytest.mark.mpi
def test_mcmc_bimodal_with_validation(base_user_inputs, resources_dir, temp_output_dir,
                                      temp_generated_models_dir, mpi_comm):
    """A bimodal target: the posterior must show both modes rather than collapsing onto one.

    This is the case a point estimate cannot represent at all, so it is the one that most
    justifies running UQ -- and the one an ensemble sampler is most likely to get wrong by
    parking every walker in whichever mode it found first.
    """
    rank = mpi_comm.Get_rank()
    config = _uq_config(
        base_user_inputs, resources_dir, temp_output_dir,
        os.path.join(_TESTS_DIR, 'test_inputs', 'Simple_ODE_Benchmark_bimodal_obs_data.json'))

    _ensure_cellml_model_generated(config, mpi_comm)
    run_param_id(config)

    if rank == 0:
        output_dir = os.path.join(
            temp_output_dir,
            f"{config['param_id_method']}_{config['file_prefix']}_"
            f"Simple_ODE_Benchmark_bimodal_obs_data")
        samples = _load_chain_after_burn_in(output_dir, config['UQ_options']['burn_in'])
        flat_samples = samples.reshape(-1, samples.shape[-1])

        mcmc_obj = _mcmc_object(config, resources_dir, temp_output_dir)
        mcmc_obj.plot_mcmc()
        assert list(Path(mcmc_obj.plot_dir).glob('mcmc_cornerplot_*.pdf'))

        # Both modes must be represented. A unimodal collapse shows up as all the mass on one
        # side of the midpoint, which is exactly what a mean or a best-fit point would hide.
        first_param = flat_samples[:, 0]
        midpoint = 0.5 * (first_param.min() + first_param.max())
        below = np.mean(first_param < midpoint)
        assert 0.1 < below < 0.9, (
            f'the chain collapsed onto one mode: {below:.1%} of samples below the midpoint')

    mpi_comm.Barrier()
