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
from benchmarks.registry import BENCHMARK_CI


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
                        mpi_comm, num_calls=30000, num_starts=16, include_aadc=None):
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
        # Run every start regardless of convergence so the same work is done at every core count
        # -- a fair parallel-scaling comparison (and a core-independent best cost).
        'no_new_starts_on_convergence': False,
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
        # Myokit (not OpenCOR) so the gradient-free baselines run in CI without OpenCOR.
        'solver': 'CVODE_myokit',
        'pre_time': 20,
        'sim_time': 2,
        'dt': 0.01,
        # DEBUG must stay False for benchmarks: DEBUG shrinks the genetic-algorithm population
        # (744 -> 28), which -- at a fixed cost budget -- changes how many generations it gets and
        # biases the optimiser comparison. Benchmark settings go in optimiser_options, not the
        # DEBUG-gated debug_optimiser_options.
        'DEBUG': False,
        'do_mcmc': False,
        'plot_predictions': False,
        'do_ia': False,
        'solver_info': {'MaximumStep': 0.001, 'MaximumNumberOfSteps': 5000},
        'param_id_obs_path': os.path.join(resources_dir, '3compartment_obs_data.json'),
        'param_id_output_dir': output_dir,
        'generated_models_dir': generated_models_dir,
        'optimiser_options': {'max_patience': 500,
                              'num_starts': 16, 'start_sampling': 'sobol', 'seed': 0,
                              # all starts run at every core count -- fair scaling comparison
                              'no_new_starts_on_convergence': False},
    })
    return config


def run_three_compartment(base_config, resources_dir, output_dir, generated_models_dir,
                          mpi_comm, num_calls=20000):
    """Run the STIFF 3compartment optimiser comparison and return a BenchmarkResult.

    GA / CMA-ES vs multi-start L-BFGS-B with the two stiff-capable gradient backends (Myokit
    CVODES FSA, CasADi bdf). AADC is not run -- its fixed-step tape integrators are
    inaccurate/unstable on a stiff model -- and is recorded as a skipped row.
    """
    rank = mpi_comm.Get_rank()
    casadi_models = os.path.join(generated_models_dir, 'casadi')
    # GA/CMA-ES (base config) and the FSA variant are all cellml_only + CVODE_myokit, so they
    # share one generated Myokit model. (The CasADi bdf variant needs its own casadi_python
    # model.) Every method's generated_models_dir must point at a model that actually gets
    # generated below, or that method fails at load with FileNotFoundError.
    myokit_models = os.path.join(generated_models_dir, 'myokit')

    config = three_compartment_config(base_config, resources_dir, output_dir, myokit_models)

    multi_start_fsa = {
        'param_id_method': 'multi_start_sp_minimize',
        'model_type': 'cellml_only', 'solver': 'CVODE_myokit', 'do_ad': True,
        'solver_info': {'MaximumStep': 0.005, 'MaximumNumberOfSteps': 50000,
                        'rtol': 1e-9, 'atol': 1e-9},
        'generated_models_dir': myokit_models,
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
    # AADC is left off this benchmark: its tape cost can only represent observables whose
    # operand is a *state* with a reimplemented operation (max/min/mean). On 3compartment 4 of
    # the 6 observables -- the aortic_root/u features (an algebraic variable, not a state) and
    # heart/q_lv's max_minus_min -- are dropped from the tape, so AADC would minimise a reduced
    # 2-of-6-observable cost rather than the same cost the other methods use. The on-tape
    # damping fix made the gradient exact for the observables it *can* tape, but full-cost
    # parity needs the algebraic variables recomputed on the tape and max_minus_min supported
    # (tracked upstream, issue #258). Until then AADC is not comparable here.
    skipped = {
        'multi_start (AADC AD)':
            "AADC's tape cost covers only state-operand observables with a reimplemented op "
            "(max/min/mean); 3compartment's algebraic-variable observables (aortic_root/u) and "
            "its max_minus_min are dropped, so AADC would optimise a reduced cost, not the full "
            "one -- excluded until it can replicate the same cost (upstream issue #258)",
    }

    if rank == 0:
        # Base cellml_only (Myokit) model, used by GA / CMA-ES and the FSA multi-start.
        assert generate_with_new_architecture(False, config), \
            'Myokit (cellml_only) model generation should succeed for 3compartment'
        casadi_cfg = config.copy(); casadi_cfg.update(multi_start_casadi)
        assert generate_with_new_architecture(False, casadi_cfg), \
            'CasADi bdf model generation should succeed for 3compartment'
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
        env_note=f'{mpi_comm.Get_size()} MPI rank(s); pre_time=20 s; 16 starts for multi-start')
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


# ----------------------------------------------------------------------------------------
# Goodwin oscillator -- an EXTERNAL CellML model (Goodwin 1965) taken straight from the
# Physiome Model Repository, not generated from CA's CSV module arrays. Demonstrates that the
# benchmarks (and CA's calibration) can consume any valid CellML placed at the expected path.
# Non-stiff, oscillatory -> multimodal (a wrong rate constant puts the oscillation out of phase
# with the data, the same trap FitzHugh-Nagumo has). CI-safe (Myokit, no OpenCOR).
# ----------------------------------------------------------------------------------------

# Ground-truth parameters the synthetic obs data was generated at (the model's own PMR values).
GOODWIN_TRUE_PARAMS = np.array([72.0, 2.0, 36.0])
GOODWIN_PARAM_LABELS = ['a_i', 'b_i', 'A_i']


def _place_external_cellml(resources_dir, generated_models_dir, file_prefix):
    """Copy a committed external CellML from resources/ to the exact path CA's loader derives
    from file_prefix ({generated_models_dir}/{file_prefix}/{file_prefix}.cellml), so it is used
    in place of the CSV->CellML generation step."""
    import shutil
    dest_dir = os.path.join(generated_models_dir, file_prefix)
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy(os.path.join(resources_dir, f'{file_prefix}.cellml'),
                os.path.join(dest_dir, f'{file_prefix}.cellml'))


def goodwin_config(base_config, resources_dir, output_dir, generated_models_dir, param_id_method):
    config = dict(base_config)
    config.update({
        'file_prefix': 'Goodwin',
        'input_param_file': 'Goodwin_parameters.csv',  # unused: x0 comes from the CellML itself
        'params_for_id_file': 'Goodwin_params_for_id.csv',
        'model_type': 'cellml_only',
        'solver': 'CVODE_myokit',
        'param_id_method': param_id_method,
        'pre_time': 0.0,
        'sim_time': 40.0,
        'dt': 0.1,
        'DEBUG': False,
        'do_mcmc': False,
        'plot_predictions': False,
        'do_ia': False,
        'cost_type': 'gaussian_MLE',
        'solver_info': {'MaximumStep': 0.01, 'MaximumNumberOfSteps': 50000,
                        'rtol': 1e-8, 'atol': 1e-8},
        'param_id_obs_path': os.path.join(resources_dir, 'Goodwin_obs_data.json'),
        'param_id_output_dir': output_dir,
        'generated_models_dir': generated_models_dir,
    })
    return config


def run_goodwin(base_config, resources_dir, output_dir, generated_models_dir,
                mpi_comm, num_calls=30000, num_starts=16):
    """Run the Goodwin-oscillator optimiser comparison and return a BenchmarkResult.

    Gradient-free global searches (GA, CMA-ES) vs multi-start L-BFGS-B driven by finite
    differences and by Myokit CVODES forward sensitivity (FSA). The CasADi/AADC AD backends do
    not apply -- they need a CA-generated symbolic model, and this is an external CellML.
    """
    rank = mpi_comm.Get_rank()
    models = os.path.join(generated_models_dir, 'goodwin')

    config = goodwin_config(base_config, resources_dir, output_dir, models, 'genetic_algorithm')
    config['optimiser_options'] = {
        'cost_convergence': 1e-6,
        'max_patience': 500,
        'num_starts': num_starts, 'start_sampling': 'sobol', 'seed': 0,
        # every start runs at every core count -- fair scaling comparison, core-independent cost.
        'no_new_starts_on_convergence': False,
    }

    multi_start_fd = {
        'param_id_method': 'multi_start_sp_minimize',
        'model_type': 'cellml_only', 'solver': 'CVODE_myokit', 'do_ad': False,
        'generated_models_dir': models,
    }
    multi_start_fsa = {
        'param_id_method': 'multi_start_sp_minimize',
        'model_type': 'cellml_only', 'solver': 'CVODE_myokit', 'do_ad': True,
        'solver_info': {'MaximumStep': 0.01, 'MaximumNumberOfSteps': 50000,
                        'rtol': 1e-9, 'atol': 1e-9},
        'generated_models_dir': models,
    }
    extra = {'multi_start (FD)': multi_start_fd, 'multi_start (Myokit FSA)': multi_start_fsa}
    methods = ['genetic_algorithm', 'CMA-ES', 'multi_start (FD)', 'multi_start (Myokit FSA)']

    # Place the external PMR CellML where CA's loader expects it (no CSV generation step).
    if rank == 0:
        _place_external_cellml(resources_dir, models, 'Goodwin')
    mpi_comm.Barrier()

    comparison = OptimiserComparison(config, methods=methods, num_calls=num_calls,
                                     extra_method_configs=extra)
    for method in methods:
        assert comparison.run_method(method) is not False, f'{method} optimisation should succeed'

    result = BenchmarkResult(
        name='goodwin',
        title='Goodwin oscillator (external PMR CellML, non-stiff, multimodal)',
        description=('Gradient-free global searches (genetic algorithm, CMA-ES) vs multi-start '
                     'L-BFGS-B (finite differences and Myokit CVODES FSA) recovering rate '
                     'constants of the Goodwin 1965 oscillator, taken directly from the Physiome '
                     'Model Repository as external CellML. Oscillatory dynamics make the '
                     'least-squares surface multimodal.'),
        env_note=(f'{mpi_comm.Get_size()} MPI rank(s); {num_calls} cost evaluations for the '
                  f'population methods; {num_starts} starts for multi-start'),
        true_params=list(GOODWIN_TRUE_PARAMS), param_labels=GOODWIN_PARAM_LABELS)
    if rank == 0:
        for method in methods:
            params = np.asarray(comparison.results[method]['params'], dtype=float)
            result.rows.append(BenchmarkRow(
                method=method,
                cost=float(comparison.results[method]['cost']),
                time_s=float(comparison.runtimes[method]),
                param_err=float(np.max(np.abs(params - GOODWIN_TRUE_PARAMS))),
                params=[float(p) for p in params]))
    result._comparison = comparison
    return result


def assert_goodwin(result, mpi_comm):
    """Regression assertions for the Goodwin benchmark (rank 0 only)."""
    if mpi_comm.Get_rank() != 0:
        mpi_comm.Barrier()
        return
    ran = [r for r in result.rows if r.skipped_reason is None]
    costs = {r.method: r.cost for r in ran}
    for r in ran:
        assert np.isfinite(r.cost) and r.cost >= 0, \
            f'{r.method} produced a non-finite or negative cost: {r.cost}'
    # The multi-start methods should reach the global basin, so they must not be beaten by the
    # gradient-free population methods (which get trapped in an out-of-phase local minimum).
    best_pop = min(costs['genetic_algorithm'], costs['CMA-ES'])
    for method in ('multi_start (FD)', 'multi_start (Myokit FSA)'):
        assert costs[method] <= best_pop + 1e-9, (
            f"{method} cost {costs[method]:.3e} is worse than the best population method "
            f"{best_pop:.3e}")
    mpi_comm.Barrier()


# ----------------------------------------------------------------------------------------
# Teusink 2000 yeast glycolysis -- the "realistic model, many parameters" case. An EXTERNAL
# CellML model from the Physiome Model Repository (itself exported there from the BioModels
# SBML), and by far the most complex benchmark: 41 components / 238 variables / 14 coupled
# metabolite states / 90 constants, vs Goodwin's 2 states. Recovers four enzyme v_max values
# from metabolite time courses -- the classic glycolysis calibration problem. CI-safe (Myokit).
# ----------------------------------------------------------------------------------------

# Ground-truth v_max values the synthetic obs data was generated at (the model's published PMR
# values), in the order of Teusink_params_for_id.csv.
TEUSINK_TRUE_PARAMS = np.array([226.452, 339.677, 1088.71, 1184.52])
TEUSINK_PARAM_LABELS = ['Vmax_GLK', 'Vmax_PGI', 'Vmax_PYK', 'Vmax_GAPDH_f']


def teusink_config(base_config, resources_dir, output_dir, generated_models_dir, param_id_method):
    config = dict(base_config)
    config.update({
        'file_prefix': 'Teusink',
        'input_param_file': 'Teusink_parameters.csv',  # unused: x0 comes from the CellML itself
        'params_for_id_file': 'Teusink_params_for_id.csv',
        'model_type': 'cellml_only',
        'solver': 'CVODE_myokit',
        'param_id_method': param_id_method,
        'pre_time': 0.0,
        'sim_time': 5.0,
        'dt': 0.05,
        'DEBUG': False,
        'do_mcmc': False,
        'plot_predictions': False,
        'do_ia': False,
        'cost_type': 'gaussian_MLE',
        # Tight tolerances: the same settings the synthetic obs data was generated with, so the
        # true parameters really are the global minimum. Glycolysis goes stiff for some parameter
        # combinations in the search box, so allow plenty of internal steps -- the few that still
        # fail return an infinite cost and the optimiser simply avoids that region.
        'solver_info': {'MaximumStep': 0.005, 'MaximumNumberOfSteps': 200000,
                        'rtol': 1e-9, 'atol': 1e-9},
        'param_id_obs_path': os.path.join(resources_dir, 'Teusink_obs_data.json'),
        'param_id_output_dir': output_dir,
        'generated_models_dir': generated_models_dir,
    })
    return config


def run_teusink(base_config, resources_dir, output_dir, generated_models_dir,
                mpi_comm, num_calls=5000, num_starts=16):
    """Run the Teusink yeast-glycolysis optimiser comparison and return a BenchmarkResult.

    Gradient-free global searches (GA, CMA-ES) vs multi-start L-BFGS-B driven by finite
    differences and by Myokit CVODES forward sensitivity (FSA).

    The CasADi/AADC AD backends are not offered. They need model_type casadi_python/aadc_python,
    which only generate_with_new_architecture produces, and that needs CA's CSV module arrays an
    external CellML does not have. It is closer than it looks -- PythonGenerator actually consumes
    a *CellML file* (the CSV step only exists to emit one) -- but it runs the libCellML Analyser,
    which is far stricter than the non-strict parse the Myokit path uses: the Analyser rejects
    these PMR files (Teusink 37 errors, Goodwin 2), starting with "W3C MathML DTD error: Syntax of
    value for attribute id of math is not valid" from their older CellML 1.0 MathML. Offering AD
    on external CellML would mean sanitising those files first.

    Note on solver settings: MaximumStep is only a *cap* -- rtol/atol govern accuracy -- and
    relaxing it 100x (0.005 -> 0.5) does not speed FSA up (37 -> 39 s over 4 starts, unchanged
    cost). FSA's cost here is intrinsic: a 14-state + 14x4-sensitivity augmented system.
    """
    rank = mpi_comm.Get_rank()
    models = os.path.join(generated_models_dir, 'teusink')

    config = teusink_config(base_config, resources_dir, output_dir, models, 'genetic_algorithm')
    config['optimiser_options'] = {
        'cost_convergence': 1e-8,
        'max_patience': 500,
        'num_starts': num_starts, 'start_sampling': 'sobol', 'seed': 0,
        # every start runs at every core count -- fair scaling comparison, core-independent cost.
        'no_new_starts_on_convergence': False,
        # A population sized for this evaluation budget (~180 generations at num_calls=5000)
        # instead of the 744-member production default, which would only get ~7 generations here
        # and leave the GA barely evolved. Configurable since the GA population landed in the
        # schema; DEBUG stays off.
        'num_elite': 4, 'num_survivors': 6, 'num_mutations_per_survivor': 2,
        'num_cross_breed': 10,
    }

    multi_start_fd = {
        'param_id_method': 'multi_start_sp_minimize',
        'model_type': 'cellml_only', 'solver': 'CVODE_myokit', 'do_ad': False,
        'generated_models_dir': models,
    }
    multi_start_fsa = {
        'param_id_method': 'multi_start_sp_minimize',
        'model_type': 'cellml_only', 'solver': 'CVODE_myokit', 'do_ad': True,
        'generated_models_dir': models,
    }
    extra = {'multi_start (FD)': multi_start_fd, 'multi_start (Myokit FSA)': multi_start_fsa}
    methods = ['genetic_algorithm', 'CMA-ES', 'multi_start (FD)', 'multi_start (Myokit FSA)']

    # Place the external PMR CellML where CA's loader expects it (no CSV generation step).
    if rank == 0:
        _place_external_cellml(resources_dir, models, 'Teusink')
    mpi_comm.Barrier()

    comparison = OptimiserComparison(config, methods=methods, num_calls=num_calls,
                                     extra_method_configs=extra)
    for method in methods:
        assert comparison.run_method(method) is not False, f'{method} optimisation should succeed'

    result = BenchmarkResult(
        name='teusink',
        title='Teusink 2000 yeast glycolysis (external PMR CellML, 14 states, stiff regions)',
        description=('The realistic, many-parameter case: recover four enzyme v_max values from '
                     'metabolite time courses of the Teusink 2000 glycolysis model, taken from '
                     'the Physiome Model Repository as external CellML (originally a BioModels '
                     'SBML export). 14 coupled metabolite states and 90 constants, with stiff '
                     'regions in the search box -- a much harder calibration than the small '
                     'oscillator benchmarks.'),
        env_note=(f'{mpi_comm.Get_size()} MPI rank(s); {num_calls} cost evaluations for the '
                  f'population methods; {num_starts} starts for multi-start'),
        true_params=list(TEUSINK_TRUE_PARAMS), param_labels=TEUSINK_PARAM_LABELS)
    if rank == 0:
        for method in methods:
            params = np.asarray(comparison.results[method]['params'], dtype=float)
            result.rows.append(BenchmarkRow(
                method=method,
                cost=float(comparison.results[method]['cost']),
                time_s=float(comparison.runtimes[method]),
                # relative error: the v_max values span 226..1185, so an absolute max would be
                # dominated by the largest parameter.
                param_err=float(np.max(np.abs(params - TEUSINK_TRUE_PARAMS)
                                       / TEUSINK_TRUE_PARAMS)),
                params=[float(p) for p in params]))
    result._comparison = comparison
    return result


def assert_teusink(result, mpi_comm):
    """Regression assertions for the Teusink benchmark (rank 0 only)."""
    if mpi_comm.Get_rank() != 0:
        mpi_comm.Barrier()
        return
    ran = [r for r in result.rows if r.skipped_reason is None]
    costs = {r.method: r.cost for r in ran}
    errs = {r.method: r.param_err for r in ran}
    for r in ran:
        assert np.isfinite(r.cost) and r.cost >= 0, \
            f'{r.method} produced a non-finite or negative cost: {r.cost}'
    # The gradient-based multi-starts should reach the true parameters on this smooth (if stiff)
    # problem, so they must not be beaten by the gradient-free population methods.
    best_pop = min(costs['genetic_algorithm'], costs['CMA-ES'])
    for method in ('multi_start (FD)', 'multi_start (Myokit FSA)'):
        assert costs[method] <= best_pop + 1e-9, (
            f"{method} cost {costs[method]:.3e} is worse than the best population method "
            f"{best_pop:.3e}")
        assert errs[method] < 0.05, (
            f"{method} did not recover the v_max values (max relative error "
            f"{errs[method]:.3f})")
    mpi_comm.Barrier()


# Registry: which benchmarks run in which context. 'ci' benchmarks must not need OpenCOR.
# All current benchmarks run on Myokit/CasADi (no OpenCOR), so all are CI-enabled; the flag
# is kept so a future OpenCOR-only benchmark can opt out of CI.
BENCHMARKS = {
    'fitzhugh_nagumo': {'run': run_fitzhugh_nagumo, 'assert': assert_fitzhugh_nagumo,
                        'ci': BENCHMARK_CI['fitzhugh_nagumo']},
    'three_compartment': {'run': run_three_compartment, 'assert': assert_three_compartment,
                          'ci': BENCHMARK_CI['three_compartment']},
    'goodwin': {'run': run_goodwin, 'assert': assert_goodwin, 'ci': BENCHMARK_CI['goodwin']},
    'teusink': {'run': run_teusink, 'assert': assert_teusink, 'ci': BENCHMARK_CI['teusink']},
}
