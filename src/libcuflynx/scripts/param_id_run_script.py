'''
Created on 29/10/2021

@author: Finbar J. Argus
'''

import sys
import os
# Not `from mpi4py import MPI`: that import initialises MPI and registers an
# atexit MPI_Finalize, and with no launcher present that finalise is what aborts
# on macOS when a NIC goes away (#396). Under mpiexec get_MPI hands back the real
# mpi4py.MPI, so a multi-rank run is unchanged.
from libcuflynx.utilities.mpi_utils import get_MPI as _get_MPI

MPI = _get_MPI()
from libcuflynx.param_id.paramID import (CVS0DParamID, DEFAULT_OPTIMISER_OPTIONS,
                                         ensure_mle_cost_type_for_bayesian_inner, mcmc_object)
from libcuflynx.scripts import _cli
import yaml
import numpy as np
from libcuflynx.parsers.PrimitiveParsers import YamlFileParser
from libcuflynx.identifiabilty_analysis.identifiabilityAnalysis import IdentifiabilityAnalysis

def run_param_id(inp_data_dict=None):
    """Calibrate, then optionally run MCMC and identifiability analysis.

    ``inp_data_dict`` is a user_inputs dict (None reads ``user_inputs.yaml``). Besides the
    yaml keys it takes ``joint_priors``, a list of ``(names, logpdf)`` handed to
    ``set_joint_priors`` of the calibration and of the MCMC (calibration workflows use it for
    priors from an earlier step's stored posterior).

    Returns ``{'output_dir', 'uq_output_dir', 'best_param_vals', 'param_id', 'mcmc'}``:
    ``output_dir`` is the calibration's run directory and ``uq_output_dir`` the MCMC run's
    (``paramID.case_dir_name``) on rank 0 (None on the others); ``mcmc`` is None without do_uq.
    """

    yaml_parser = YamlFileParser()
    inp_data_dict = yaml_parser.parse_user_inputs_file(inp_data_dict, obs_path_needed=True, do_generation_with_fit_parameters=False)

    DEBUG = inp_data_dict['DEBUG']
    model_path = inp_data_dict['model_path']
    model_type = inp_data_dict['model_type']
    param_id_method = inp_data_dict['param_id_method']
    file_prefix = inp_data_dict['file_prefix']
    solver_info = inp_data_dict['solver_info']
    if solver_info.get('solver') == 'casadi_integrator':
        try:
            import casadi  # noqa: F401
        except ImportError as exc:
            from libcuflynx.param_id.casadi_backend import CASADI_MISSING_MESSAGE
            raise ImportError(
                "The solver is set to casadi_integrator but the casadi package is not installed. "
                + CASADI_MISSING_MESSAGE
                + " Or change the solver in your configuration."
            ) from exc
    optimiser_options = inp_data_dict.get('optimiser_options', None)
    resources_dir = inp_data_dict['resources_dir']
    param_id_output_dir = inp_data_dict['param_id_output_dir']

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()
    if rank == 0:
        if DEBUG:
            print('WARNING: DEBUG IS ON, TURN THIS OFF IF YOU WANT TO DO ANYTHING QUICKLY')
        print(f'Starting parameter identification with {num_procs} MPI rank(s)')

    param_id = _engine(inp_data_dict, mcmc_instead=False)

    if inp_data_dict.get('obs_data_dict') is not None:
        param_id.set_ground_truth_data(inp_data_dict['obs_data_dict'])
    if inp_data_dict.get('params_for_id') is not None:
        param_id.set_params_for_id(inp_data_dict['params_for_id'])
    joint_priors = inp_data_dict.get('joint_priors')
    if joint_priors:
        param_id.set_joint_priors(joint_priors)

    if rank == 0:
        if os.path.exists(os.path.join(param_id.output_dir, 'param_names_to_remove.csv')):
            os.remove(os.path.join(param_id.output_dir, 'param_names_to_remove.csv'))


    if param_id_method == 'bayesian':
        acq_func = 'PI'  # 'gp_hedge'
        n_initial_points = 5
        random_seed = 1234
        acq_func_kwargs = {'xi': 0.01, 'kappa': 0.1} # these parameters favour exploitation if they are low
                                                            # and exploration if high, see scikit-optimize docs.
                                                            # xi is used when acq_func is “EI” or “PI”,
                                                            # kappa is used when acq_func is "LCB"
                                                            # gp_hedge, chooses the best from "EI", "PI", and "LCB
                                                            # so it needs both xi and kappa
        # TODO this needs to be defined better if we want to keep bayesian optimiser functionality
        # Use optimiser_options (already merged with any legacy options by parser)
        num_calls_to_function = optimiser_options.get('num_calls_to_function')
        if num_calls_to_function is None:
            num_calls_to_function = 10000  # fallback default
        param_id.set_bayesian_parameters(num_calls_to_function, n_initial_points, acq_func,  random_seed,
                                            acq_func_kwargs=acq_func_kwargs)
    param_id.run()
    # param_id.param_id.set_best_param_vals(np.asarray([0.59779409, 0.32321317, 0.05664833, 0.35665839]))
    best_param_vals = param_id.get_best_param_vals()

    override_ia = inp_data_dict.get("override_best_param_vals_for_ia")
    if override_ia is not None:
        best_param_vals = np.asarray(override_ia, dtype=float)

    if rank == 0:
        print('param id complete with method:', param_id_method)

    # param_id.close_simulation() comment for identifiability analysis run otherwise the model will be closed before analysis
    do_uq = inp_data_dict['do_uq']

    if DEBUG:
        UQ_options = inp_data_dict['debug_UQ_options']
    else:
        UQ_options = inp_data_dict['UQ_options']

    mcmc = None
    if do_uq and (UQ_options or {}).get('method', 'mcmc') == 'laplace':
        _laplace(inp_data_dict, param_id, best_param_vals, param_id_output_dir)
    elif do_uq:

        if rank == 0:
            print('running mcmc')

        mcmc = _mcmc_at(inp_data_dict, UQ_options, best_param_vals, joint_priors)

        if rank == 0:
            print('mcmc complete')
    

    if inp_data_dict.get("do_ia") and inp_data_dict.get("ia_options", {}).get("method") == "Laplace":
        ensure_mle_cost_type_for_bayesian_inner(param_id.param_id, inp_data_dict)

    if inp_data_dict['do_ia']:
        # id_analysis = IdentifiabilityAnalysis(model_path, model_type, param_id_method, False, file_prefix,
        #                                      params_for_id_path=params_for_id_path,
        #                                      param_id_obs_path=param_id_obs_path,
        #                                      sim_time=sim_time, pre_time=pre_time,
        #                                      solver_info=solver_info, dt=dt, DEBUG=DEBUG,
        #                                      param_id_output_dir=param_id_output_dir, resources_dir=resources_dir,
        #                                      param_id=param_id.param_id) # pass in param_id object so we can use its cost functions
        id_analysis = IdentifiabilityAnalysis(model_path, model_type, file_prefix, param_id_output_dir=param_id_output_dir,
                                            resources_dir=resources_dir, param_id=param_id.param_id)  # pass in param_id object so we can use its cost functions

        id_analysis.set_best_param_vals(best_param_vals)    
        if rank == 0:
            print('Running identifiability analysis with method:', inp_data_dict['ia_options']['method'])
        #id_analysis.run_identifiability_analysis(inp_data_dict['identifiability_analysis_options'])
        id_analysis.run(inp_data_dict['ia_options'])

        if rank == 0:
            print('Identifiability analysis complete')

    return {'output_dir': param_id.output_dir, 'uq_output_dir': param_id.uq_output_dir,
            'best_param_vals': best_param_vals, 'param_id': param_id, 'mcmc': mcmc}

def _engine(inp_data_dict, mcmc_instead, UQ_options=None):
    '''The CVS0DParamID engine of a parsed user_inputs dict: a calibration's (with its
    optimiser_options and do_ad), or, with ``mcmc_instead``, an MCMC's (with ``UQ_options``).'''
    # The study manifest's name, switch, source yaml and working-dir settings
    # (utilities/cuflynx_study.py).
    from libcuflynx.utilities.cuflynx_study import STUDY_INP_KEYS
    study_kwargs = {key: inp_data_dict[key] for key in
                    ('study_name', 'write_cuflynx_study', 'user_inputs_path') if key in inp_data_dict}
    study_kwargs['study_settings'] = {key: inp_data_dict[key] for key in STUDY_INP_KEYS
                                      if key in inp_data_dict}
    # Both engines, the calibration's and the MCMC's, run on the emulator use_emulator asks for.
    use_emulator = inp_data_dict.get('use_emulator', False)
    emulator_dir = None
    if use_emulator:
        # Imported here, not at module scope: resolve_emulator_dir pulls in the emulators
        # package, and [emulation] is an optional extra.
        from libcuflynx.emulators.emulator_trainer import resolve_emulator_dir
        emulator_dir = resolve_emulator_dir(inp_data_dict)
    stage = ({'UQ_options': UQ_options} if mcmc_instead else
             {'optimiser_options': inp_data_dict.get('optimiser_options', None),
              'do_ad': inp_data_dict['do_ad']})
    return CVS0DParamID(inp_data_dict['model_path'], inp_data_dict['model_type'],
                        inp_data_dict['param_id_method'], mcmc_instead,
                        inp_data_dict['file_prefix'],
                        params_for_id_path=inp_data_dict['params_for_id_path'],
                        param_id_obs_path=inp_data_dict['param_id_obs_path'],
                        sim_time=inp_data_dict['sim_time'], pre_time=inp_data_dict['pre_time'],
                        solver_info=inp_data_dict['solver_info'], dt=inp_data_dict['dt'],
                        DEBUG=inp_data_dict['DEBUG'],
                        param_id_output_dir=inp_data_dict['param_id_output_dir'],
                        resources_dir=inp_data_dict['resources_dir'],
                        operation_funcs_external_path=inp_data_dict.get(
                            'operation_funcs_external_path', None),
                        cost_funcs_external_path=inp_data_dict.get('cost_funcs_external_path',
                                                                   None),
                        modifier_funcs_external_path=inp_data_dict.get(
                            'modifier_funcs_external_path', None),
                        **study_kwargs, **stage,
                        use_emulator=use_emulator, emulator_dir=emulator_dir,
                        emulator_settings=inp_data_dict.get('emulator_settings'))


def _mcmc_at(inp_data_dict, UQ_options, best_param_vals, joint_priors=None):
    '''MCMC started at ``best_param_vals``: the run's engine, after run_UQ.'''
    mcmc = _engine(inp_data_dict, mcmc_instead=True, UQ_options=UQ_options)
    mcmc.set_best_param_vals(best_param_vals)
    if joint_priors:
        mcmc.set_joint_priors(joint_priors)
    ensure_mle_cost_type_for_bayesian_inner(mcmc_object, inp_data_dict)
    mcmc.run_UQ()
    return mcmc


def _laplace(inp_data_dict, param_id, best_param_vals, output_dir):
    '''The Laplace approximation at ``best_param_vals`` with the calibration engine
    ``param_id`` (IdentifiabilityAnalysis, method Laplace, with the run's ia_options): ``(mean,
    covariance)`` on rank 0, written as ``<prefix>_laplace_{mean,covariance}.npy`` in
    ``output_dir``; ``(None, None)`` on the others.'''
    ia = IdentifiabilityAnalysis(inp_data_dict['model_path'], inp_data_dict['model_type'],
                                 inp_data_dict['file_prefix'], param_id_output_dir=output_dir,
                                 resources_dir=inp_data_dict['resources_dir'],
                                 param_id=param_id.param_id)
    ia.set_best_param_vals(np.asarray(best_param_vals, dtype=float))
    ensure_mle_cost_type_for_bayesian_inner(param_id.param_id, inp_data_dict)
    ia.run(dict(inp_data_dict.get('ia_options') or {}, method='Laplace'))
    return ia.mean_Laplace, ia.covariance_matrix_Laplace


def run_uq(inp_data_dict, best_param_vals, method='mcmc'):
    """Uncertainty quantification at ``best_param_vals`` (one value per params_for_id entry),
    with no calibration of its own: ``method`` ``mcmc`` samples the posterior from there,
    ``laplace`` approximates it there by a Gaussian (IdentifiabilityAnalysis, method Laplace).

    ``inp_data_dict`` is a user_inputs dict, as :func:`run_param_id` takes; its UQ_options (or
    debug_UQ_options under DEBUG) configure the MCMC, its ia_options the Laplace approximation,
    and ``joint_priors`` are the MCMC's priors. Returns ``{'method', 'uq_output_dir', 'mean',
    'covariance', 'engine'}``: ``uq_output_dir`` holds the MCMC's chain, ``mean`` and
    ``covariance`` the Laplace approximation's (on rank 0; None on the others and for mcmc).
    """
    inp_data_dict = YamlFileParser().parse_user_inputs_file(
        inp_data_dict, obs_path_needed=True, do_generation_with_fit_parameters=False)
    UQ_options = inp_data_dict['debug_UQ_options'] if inp_data_dict['DEBUG'] else \
        inp_data_dict['UQ_options']
    best = np.asarray(best_param_vals, dtype=float)
    if method == 'mcmc':
        engine = _mcmc_at(inp_data_dict, UQ_options, best, inp_data_dict.get('joint_priors'))
        return {'method': method, 'uq_output_dir': engine.uq_output_dir, 'mean': None,
                'covariance': None, 'engine': engine}
    if method == 'laplace':
        # the engine only evaluates the cost here: its optimiser defaults go unannounced
        engine = _engine(dict(inp_data_dict, optimiser_options=(
            inp_data_dict.get('optimiser_options') or dict(DEFAULT_OPTIMISER_OPTIONS))),
            mcmc_instead=False)
        mean, covariance = _laplace(inp_data_dict, engine, best,
                                    inp_data_dict['param_id_output_dir'])
        return {'method': method, 'uq_output_dir': inp_data_dict['param_id_output_dir'],
                'mean': mean, 'covariance': covariance, 'engine': engine}
    raise ValueError(f'unknown uncertainty quantification method {method!r}; one of '
                     f"['mcmc', 'laplace'].")


def main(argv=None):
    """Entry point for the ``cuflynx-param-id`` command."""
    parser = _cli.build_parser(
        'Calibrate a generated model against the observables named in the configuration, '
        'then optionally run MCMC and identifiability analysis on the result.')
    args = parser.parse_args(argv)
    inp_data_dict = _cli.load_user_inputs(args)
    return _cli.run_stage(lambda: run_param_id(inp_data_dict), MPI)


if __name__ == '__main__':
    sys.exit(main())
