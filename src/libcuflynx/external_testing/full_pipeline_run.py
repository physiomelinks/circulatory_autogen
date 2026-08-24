"""Build one run directory containing every artefact a full study produces.

Sensitivity, an emulator, a calibration, a chain and a posterior predictive check
all write different files, and the tools that read them afterwards -- the
generated ``plot_outputs.py``, and CUFLynx's outputs-directory loader -- depend on
that whole set being present and named as CA names it. Nothing exercised the
combination: each stage had its own test, and the readers were tested against
fixtures written by hand, which agree with the readers by construction and cannot
catch CA renaming a file.

So this builds the real thing, small enough to run in a test:

    from libcuflynx.external_testing.full_pipeline_run import build_full_pipeline_run
    result = build_full_pipeline_run(output_dir, resources_dir, generated_models_dir)

It lives in the package rather than in ``tests/`` because the wheel ships no
``tests/``: CUFLynx resolves CA through whatever ``libcuflynx`` is installed, so a
builder it cannot import is a builder it cannot use. Shipping it lets CUFLynx check
its own loader against a directory CA actually produced, instead of against a
fixture that encodes what CUFLynx believes CA writes.

Deliberately tiny: sixteen Sobol samples, thirty-two emulator training points, a
sixty-evaluation calibration and a five-step chain. Enough that every stage runs
its real code path and writes its real files; not enough to mean anything
numerically, which is not what this is for.
"""
import os

#: The model used, and the settings that keep the whole thing to a minute or so.
FILE_PREFIX = '3compartment'
SA_SAMPLES = 16
EMULATOR_SAMPLES = 32
CALIBRATION_CALLS = 60
POSTERIOR_DRAWS = 10
POSTERIOR_SERIES_DRAWS = 4

#: The learners that fit quickly and do not need a GPU. The GaussianProcess
#: variants are left out for the same reason a real study leaves them out of a
#: quick pass: they are the slow arm, and this is not measuring accuracy.
EMULATOR_MODELS = 'RandomForest,PolynomialRegression,RadialBasisFunctions'

#: What a full run leaves behind, relative to the run directory. Every reader
#: downstream -- the plotting script, CUFLynx -- depends on these names, so they
#: are listed once here and asserted against rather than restated per test.
EXPECTED_ARTEFACTS = (
    'best_param_vals.npy',
    'best_cost.npy',
    'param_names.csv',
    'mcmc_chain.npy',
    'posterior_predictive.npz',
    'posterior_predictive_coverage.json',
)

#: Written only when traces were kept.
SERIES_ARTEFACT = 'posterior_predictive_series.npz'


def build_config(output_dir, resources_dir, generated_models_dir,
                 emulator_dir=None):
    """The user_inputs dict a small full run needs.

    Separate from running it so a caller can adjust one setting without
    reproducing the rest, and so a test can assert what was asked for.
    """
    emulator_dir = emulator_dir or os.path.join(output_dir, 'emulators', FILE_PREFIX)
    return {
        'DEBUG': True,
        'file_prefix': FILE_PREFIX,
        'input_param_file': FILE_PREFIX + '_parameters.csv',
        'model_type': 'cellml',
        # Named explicitly: a bare 'CVODE' resolves to the OpenCOR backend, whose
        # Python module is not on PyPI, so a pip-installed libcuflynx cannot run
        # it -- and this has to run wherever the tests do.
        'solver': 'CVODE_myokit',
        'solver_info': {'solver': 'CVODE_myokit',
                        'MaximumStep': 0.001, 'MaximumNumberOfSteps': 5000},
        'param_id_method': 'genetic_algorithm',
        'pre_time': 20,
        'sim_time': 2,
        'dt': 0.01,
        'do_ad': False,
        'resources_dir': resources_dir,
        'param_id_obs_path': os.path.join(
            resources_dir, FILE_PREFIX + '_obs_data.json'),
        # Named explicitly. parse_user_inputs_file fills this in by convention,
        # but anything calling init_from_dict directly -- the posterior
        # predictive check does -- bypasses that, and a missing params_for_id
        # leaves param_id_info None and every evaluation raising.
        'params_for_id_path': os.path.join(
            resources_dir, FILE_PREFIX + '_params_for_id.csv'),
        'param_id_output_dir': output_dir,
        'generated_models_dir': generated_models_dir,
        # gaussian_MLE throughout: MCMC needs ln L = -cost, and a calibration on
        # a different cost is not the point the chain starts from.
        'optimiser_options': {'num_calls_to_function': CALIBRATION_CALLS,
                              'cost_type': 'gaussian_MLE'},
        'debug_optimiser_options': {'num_calls_to_function': CALIBRATION_CALLS,
                                    'cost_type': 'gaussian_MLE'},
        'do_uq': True,
        'UQ_options': {'method': 'mcmc', 'library': 'emcee', 'num_steps': 5,
                       'num_walkers': 20, 'burn_in': 0.5,
                       'cost_type': 'gaussian_MLE'},
        'debug_UQ_options': {'method': 'mcmc', 'library': 'emcee', 'num_steps': 5,
                             'num_walkers': 20, 'burn_in': 0.5,
                             'cost_type': 'gaussian_MLE'},
        'do_ia': False,
        'do_sensitivity': True,
        'sa_options': {'method': 'sobol', 'sample_type': 'saltelli',
                       'num_samples': SA_SAMPLES,
                       'output_dir': os.path.join(output_dir, 'sensitivity')},
        'do_emulation': True,
        'use_emulator': False,
        'emulator_settings': {
            'emulator_dir': emulator_dir,
            'models': EMULATOR_MODELS,
            'num_train_samples': EMULATOR_SAMPLES,
            'reuse_samples': False,
            'sample_type': 'sobol',
            'random_seed': 0,
            'test_fraction': 0.2,
            'n_splits': 2,
            'n_iter': 2,
            # Not a quality bar here: this run is about the files, and refusing
            # a thirty-two-sample emulator would stop the stage that writes them.
            'min_r2': -1e9,
            'out_of_bounds': 'warn',
            'fd_rel_step': 1.0e-3,
        },
    }


def build_full_pipeline_run(output_dir, resources_dir, generated_models_dir,
                            config=None, with_series=True):
    """Run every stage into ``output_dir`` and report what was produced.

    Returns ``{'config', 'run_dir', 'artefacts', 'coverage'}``. ``run_dir`` is
    where CA actually wrote, which is not ``output_dir`` -- it names the
    directory after the method, prefix and obs file, and every reader has to find
    it the same way.
    """
    from libcuflynx.scripts.script_generate_with_new_architecture import (
        generate_with_new_architecture)
    from libcuflynx.scripts.sensitivity_analysis_run_script import run_SA
    from libcuflynx.scripts.train_emulator_run_script import train_emulator
    from libcuflynx.scripts.param_id_run_script import run_param_id
    from libcuflynx.param_id.posterior_predictive import posterior_predictive
    from libcuflynx.utilities.mpi_utils import get_MPI

    config = dict(config or build_config(
        output_dir, resources_dir, generated_models_dir))
    comm = get_MPI().COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0 and not generate_with_new_architecture(False, config):
        raise RuntimeError('model generation failed')
    if comm.Get_size() > 1:
        config = comm.bcast(config if rank == 0 else None, root=0)
        comm.Barrier()

    if config.get('do_sensitivity'):
        run_SA(dict(config))
    if config.get('do_emulation'):
        train_emulator(dict(config))

    # Calibration, then the chain, then identifiability -- run_param_id owns all
    # three, gated by the flags above.
    run_param_id(dict(config))

    result = posterior_predictive(
        config, num_samples=POSTERIOR_DRAWS, use_emulator=False, save=True,
        series_draws=POSTERIOR_SERIES_DRAWS if with_series else 0)

    if rank != 0:
        return None

    from libcuflynx.utilities.paths import default_param_id_output_dir  # noqa: F401

    run_dir = _find_run_dir(output_dir)
    return {
        'config': config,
        'run_dir': run_dir,
        'artefacts': present_artefacts(run_dir),
        'coverage': result.coverage if result is not None else None,
    }


def _find_run_dir(output_dir):
    """Where CA wrote this run: ``<method>_<prefix>_<obs prefix>``."""
    for entry in sorted(os.scandir(output_dir), key=lambda e: e.name):
        if entry.is_dir() and os.path.isfile(
                os.path.join(entry.path, 'best_param_vals.npy')):
            return entry.path
    return output_dir


def present_artefacts(run_dir):
    """Which of the expected files are actually there, and which are not."""
    if not run_dir or not os.path.isdir(run_dir):
        return {'present': [], 'missing': list(EXPECTED_ARTEFACTS)}
    present, missing = [], []
    for name in EXPECTED_ARTEFACTS:
        (present if os.path.isfile(os.path.join(run_dir, name)) else missing).append(name)
    if os.path.isfile(os.path.join(run_dir, SERIES_ARTEFACT)):
        present.append(SERIES_ARTEFACT)
    return {'present': present, 'missing': missing}
