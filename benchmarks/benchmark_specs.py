"""Optimiser benchmark definitions, shared by the pytest tests and the standalone runner.

Each benchmark builds an OptimiserComparison, runs a set of methods, and returns a
``BenchmarkResult`` (a plain data structure of costs / runtimes / parameter errors). The
pytest tests wrap ``run_*`` with ``assert_*`` for regression; ``run_benchmarks.py`` formats
the same ``BenchmarkResult``s into the documentation.

Keeping the run logic here (not in the tests) is what lets the fast FitzHugh-Nagumo benchmark
stay a normal test AND be invoked from the benchmark runner without duplicating the setup.
"""
import os

import numpy as np

from scripts.script_generate_with_new_architecture import generate_with_new_architecture
from benchmarks.compare_optimisers import OptimiserComparison
from benchmarks.docs_results import BenchmarkResult, BenchmarkRow


# FitzHugh-Nagumo ground-truth parameters (a, b, c) the obs data was generated at.
FHN_TRUE_PARAMS = np.array([0.2, 0.2, 3.0])
FHN_PARAM_LABELS = ['a', 'b', 'c']


def aadc_license_available():
    """Whether AADC is installed AND licensed, reusing the test suite's check.

    The check must precede ``import mpi4py.MPI`` (importing MPI first breaks AADC's licence
    validation), which ``tests.conftest`` already handles at import time; we defer to it and
    fall back to False if it (or AADC) is unavailable.
    """
    try:
        from tests.conftest import AADC_LICENSE_AVAILABLE
        return bool(AADC_LICENSE_AVAILABLE)
    except Exception:
        return False


# ----------------------------------------------------------------------------------------
# FitzHugh-Nagumo (fast, non-stiff, multi-modal) -- CI-safe (no OpenCOR)
# ----------------------------------------------------------------------------------------

def fitzhugh_nagumo_config(base_config, resources_dir, output_dir, generated_models_dir,
                           param_id_method):
    config = dict(base_config)
    config.update({
        'file_prefix': 'FitzHugh_Nagumo',
        'input_param_file': 'FitzHugh_Nagumo_parameters.csv',
        'params_for_id_file': 'FitzHugh_Nagumo_params_for_id.csv',
        'model_type': 'casadi_python',
        'solver': 'casadi_integrator',
        'param_id_method': param_id_method,
        'do_ad': True,
        'pre_time': 0.0,
        'sim_time': 60.0,
        'dt': 0.2,
        'DEBUG': False,
        'do_mcmc': False,
        'plot_predictions': False,
        'do_ia': False,
        'solver_info': {
            'max_step_size': 0.01,
            'max_num_steps': 20000,
            'method': 'cvodes',
        },
        'param_id_obs_path': os.path.join(resources_dir, 'FitzHugh_Nagumo_obs_data.json'),
        'param_id_output_dir': output_dir,
        'generated_models_dir': generated_models_dir,
    })
    return config


def run_fitzhugh_nagumo(base_config, resources_dir, output_dir, generated_models_dir,
                        mpi_comm, num_calls=2000, num_starts=8, include_aadc=None):
    """Run the FitzHugh-Nagumo optimiser comparison and return a BenchmarkResult.

    Compares GA / CMA-ES against multi-start L-BFGS-B driven by finite differences, CasADi AD,
    Myokit CVODES FSA, and (if licensed) AADC AD. Non-stiff, so every gradient backend works.
    """
    if include_aadc is None:
        include_aadc = aadc_license_available()
    rank = mpi_comm.Get_rank()

    casadi_models = os.path.join(generated_models_dir, 'casadi')
    aadc_models = os.path.join(generated_models_dir, 'aadc')
    fsa_models = os.path.join(generated_models_dir, 'fsa')

    config = fitzhugh_nagumo_config(base_config, resources_dir, output_dir,
                                    casadi_models, 'genetic_algorithm')
    config['optimiser_options'] = {
        'cost_convergence': 1e-4,
        'max_patience': 500,
        'num_starts': num_starts, 'start_sampling': 'sobol', 'seed': 0,
    }

    multi_start_casadi = {
        'param_id_method': 'multi_start_sp_minimize',
        'model_type': 'casadi_python', 'solver': 'casadi_integrator', 'do_ad': True,
        'generated_models_dir': casadi_models,
    }
    multi_start_fd = dict(multi_start_casadi, do_ad=False)
    multi_start_aadc = {
        'param_id_method': 'multi_start_sp_minimize',
        'model_type': 'aadc_python', 'solver': 'aadc_semi_implicit',
        'solver_info': {'method': 'rk4'}, 'dt': 0.02, 'do_ad': True,
        'generated_models_dir': aadc_models,
    }
    multi_start_fsa = {
        'param_id_method': 'multi_start_sp_minimize',
        'model_type': 'cellml_only', 'solver': 'CVODE_myokit', 'do_ad': True,
        'solver_info': {'rtol': 1e-9, 'atol': 1e-9},
        'generated_models_dir': fsa_models,
    }

    extra = {
        'multi_start (CasADi AD)': multi_start_casadi,
        'multi_start (Myokit FSA)': multi_start_fsa,
        'multi_start (FD)': multi_start_fd,
    }
    methods = ['genetic_algorithm', 'CMA-ES', 'multi_start (FD)',
               'multi_start (CasADi AD)', 'multi_start (Myokit FSA)']
    if include_aadc:
        extra['multi_start (AADC AD)'] = multi_start_aadc
        methods.append('multi_start (AADC AD)')

    if rank == 0:
        casadi_cfg = config.copy(); casadi_cfg.update(multi_start_casadi)
        assert generate_with_new_architecture(False, casadi_cfg), \
            'CasADi model generation should succeed for FitzHugh-Nagumo'
        fsa_cfg = config.copy(); fsa_cfg.update(multi_start_fsa)
        assert generate_with_new_architecture(False, fsa_cfg), \
            'FSA (cellml_only) model generation should succeed for FitzHugh-Nagumo'
        if include_aadc:
            aadc_cfg = config.copy(); aadc_cfg.update(multi_start_aadc)
            assert generate_with_new_architecture(False, aadc_cfg), \
                'AADC model generation should succeed for FitzHugh-Nagumo'
    mpi_comm.Barrier()

    comparison = OptimiserComparison(config, methods=methods, num_calls=num_calls,
                                     extra_method_configs=extra)
    for method in methods:
        assert comparison.run_method(method) is not False, f'{method} optimisation should succeed'

    result = BenchmarkResult(
        name='fitzhugh_nagumo',
        title='FitzHugh-Nagumo (non-stiff, multi-modal)',
        description=('Gradient-free global searches (genetic algorithm, CMA-ES) vs multi-start '
                     'L-BFGS-B driven by four gradient sources. Holding the optimiser fixed and '
                     'varying only the gradient isolates what the gradient buys.'),
        env_note=(f'{mpi_comm.Get_size()} MPI rank(s); {num_calls} cost evaluations for the '
                  f'population methods; {num_starts} starts for multi-start'),
        true_params=list(FHN_TRUE_PARAMS), param_labels=FHN_PARAM_LABELS)
    if rank == 0:
        for method in methods:
            params = np.asarray(comparison.results[method]['params'], dtype=float)
            result.rows.append(BenchmarkRow(
                method=method,
                cost=float(comparison.results[method]['cost']),
                time_s=float(comparison.runtimes[method]),
                param_err=float(np.max(np.abs(params - FHN_TRUE_PARAMS))),
                params=[float(p) for p in params]))
        if not include_aadc:
            result.rows.append(BenchmarkRow(
                method='multi_start (AADC AD)',
                skipped_reason='no Matlogica licence in this environment'))
    result._comparison = comparison  # for assertions
    return result


def assert_fitzhugh_nagumo(result, mpi_comm):
    """Regression assertions for the FitzHugh-Nagumo benchmark (rank 0 only)."""
    if mpi_comm.Get_rank() != 0:
        mpi_comm.Barrier()
        return
    ran = [r for r in result.rows if r.skipped_reason is None]
    costs = {r.method: r.cost for r in ran}
    times = {r.method: r.time_s for r in ran}
    params = {r.method: np.asarray(r.params, dtype=float) for r in ran}

    for r in ran:
        assert np.isfinite(r.cost) and r.cost >= 0, \
            f'{r.method} produced a non-finite or negative cost: {r.cost}'

    multi_start_methods = [m for m in costs if m.startswith('multi_start')]
    for method in multi_start_methods:
        assert costs[method] < costs['genetic_algorithm'], (
            f'expected {method} ({costs[method]:.3e}) to beat the genetic algorithm '
            f'({costs["genetic_algorithm"]:.3e})')
        assert costs[method] < costs['CMA-ES'], (
            f'expected {method} ({costs[method]:.3e}) to beat CMA-ES ({costs["CMA-ES"]:.3e})')
        np.testing.assert_allclose(params[method], FHN_TRUE_PARAMS, atol=0.02,
                                   err_msg=f'{method} did not recover the true parameters')

    for method in multi_start_methods:
        if 'AD' in method:
            np.testing.assert_allclose(
                params[method], params['multi_start (FD)'], atol=0.02,
                err_msg=f'{method} converged somewhere different from the FD variant')
    for method in [m for m in multi_start_methods if 'AD' in m]:
        assert times[method] < times['multi_start (FD)'], (
            f'expected {method} ({times[method]:.0f}s) to beat finite differences '
            f'({times["multi_start (FD)"]:.0f}s)')
    mpi_comm.Barrier()


# ----------------------------------------------------------------------------------------
# 3compartment (slow, STIFF, long pre_time) -- local only, NOT run in CI
# ----------------------------------------------------------------------------------------

def three_compartment_config(base_config, resources_dir, output_dir, generated_models_dir):
    config = dict(base_config)
    config.update({
        'file_prefix': '3compartment',
        'input_param_file': '3compartment_parameters.csv',
        'model_type': 'cellml_only',
        'solver': 'CVODE',
        'pre_time': 20,
        'sim_time': 2,
        'dt': 0.01,
        'DEBUG': True,
        'do_mcmc': False,
        'plot_predictions': False,
        'do_ia': False,
        'solver_info': {'MaximumStep': 0.001, 'MaximumNumberOfSteps': 5000},
        'param_id_obs_path': os.path.join(resources_dir, '3compartment_obs_data.json'),
        'param_id_output_dir': output_dir,
        'generated_models_dir': generated_models_dir,
        'debug_optimiser_options': {'num_calls_to_function': 10000, 'max_patience': 500,
                                    'num_starts': 5, 'start_sampling': 'sobol', 'seed': 0},
    })
    return config


def run_three_compartment(base_config, resources_dir, output_dir, generated_models_dir,
                          mpi_comm, num_calls=10000):
    """Run the STIFF 3compartment optimiser comparison and return a BenchmarkResult.

    GA / CMA-ES vs multi-start L-BFGS-B with the two stiff-capable gradient backends (Myokit
    CVODES FSA, CasADi bdf). AADC is not run -- its fixed-step tape integrators are
    inaccurate/unstable on a stiff model -- and is recorded as a skipped row.
    """
    rank = mpi_comm.Get_rank()
    casadi_models = os.path.join(generated_models_dir, 'casadi')
    fsa_models = os.path.join(generated_models_dir, 'fsa')

    config = three_compartment_config(base_config, resources_dir, output_dir, generated_models_dir)

    multi_start_fsa = {
        'param_id_method': 'multi_start_sp_minimize',
        'model_type': 'cellml_only', 'solver': 'CVODE_myokit', 'do_ad': True,
        'solver_info': {'MaximumStep': 0.005, 'MaximumNumberOfSteps': 50000,
                        'rtol': 1e-9, 'atol': 1e-9},
        'generated_models_dir': fsa_models,
    }
    multi_start_casadi = {
        'param_id_method': 'multi_start_sp_minimize',
        'model_type': 'casadi_python', 'solver': 'casadi_integrator', 'do_ad': True,
        'solver_info': {'max_step_size': 0.001, 'max_num_steps': 50000, 'method': 'bdf'},
        'generated_models_dir': casadi_models,
    }
    extra = {
        'multi_start (Myokit FSA)': multi_start_fsa,
        'multi_start (CasADi bdf)': multi_start_casadi,
    }
    skipped = {
        'multi_start (AADC AD)':
            'not suitable for stiff models (AADC fixed-step tape integrators are '
            'inaccurate/unstable here; use CasADi bdf or Myokit FSA)',
    }

    if rank == 0:
        casadi_cfg = config.copy(); casadi_cfg.update(multi_start_casadi)
        assert generate_with_new_architecture(False, casadi_cfg), \
            'CasADi bdf model generation should succeed for 3compartment'
        fsa_cfg = config.copy(); fsa_cfg.update(multi_start_fsa)
        assert generate_with_new_architecture(False, fsa_cfg), \
            'FSA (cellml_only) model generation should succeed for 3compartment'
    mpi_comm.Barrier()

    methods = ['genetic_algorithm', 'CMA-ES',
               'multi_start (Myokit FSA)', 'multi_start (CasADi bdf)']
    comparison = OptimiserComparison(config, methods=methods, num_calls=num_calls,
                                     extra_method_configs=extra)
    for method in methods:
        assert comparison.run_method(method) is not False, f'{method} optimisation should succeed'

    result = BenchmarkResult(
        name='three_compartment',
        title='3compartment cardiovascular (stiff, 20 s warmup)',
        description=('Gradient-free global searches vs multi-start L-BFGS-B with the two '
                     'stiff-capable gradient backends. AADC is not run on this stiff model.'),
        env_note=f'{mpi_comm.Get_size()} MPI rank(s); pre_time=20 s; 5 starts for multi-start')
    if rank == 0:
        for method in methods:
            result.rows.append(BenchmarkRow(
                method=method,
                cost=float(comparison.results[method]['cost']),
                time_s=float(comparison.runtimes[method])))
        for method, reason in skipped.items():
            result.rows.append(BenchmarkRow(method=method, skipped_reason=reason))
    result._comparison = comparison
    return result


def assert_three_compartment(result, mpi_comm):
    """Regression assertions for the 3compartment benchmark (rank 0 only)."""
    if mpi_comm.Get_rank() != 0:
        mpi_comm.Barrier()
        return
    ran = [r for r in result.rows if r.skipped_reason is None]
    costs = {r.method: r.cost for r in ran}
    for r in ran:
        assert np.isfinite(r.cost) and r.cost >= 0, \
            f'{r.method} produced a non-finite or negative cost: {r.cost}'
    best_pop = min(costs['genetic_algorithm'], costs['CMA-ES'])
    for method in ('multi_start (Myokit FSA)', 'multi_start (CasADi bdf)'):
        assert costs[method] <= best_pop * 5.0 + 1e-9, (
            f"{method} cost {costs[method]:.3e} is far worse than the best population method "
            f"{best_pop:.3e}")
    mpi_comm.Barrier()


# Registry: which benchmarks run in which context.
# 'ci' benchmarks must not need OpenCOR and must finish quickly.
BENCHMARKS = {
    'fitzhugh_nagumo': {'run': run_fitzhugh_nagumo, 'assert': assert_fitzhugh_nagumo, 'ci': True},
    'three_compartment': {'run': run_three_compartment, 'assert': assert_three_compartment,
                          'ci': False},
}
