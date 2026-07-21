"""
Tests for sensitivity analysis functionality.

These tests verify that sensitivity analysis works correctly for various models.
"""
import os
import pytest
from mpi4py import MPI

from scripts.script_generate_with_new_architecture import generate_with_new_architecture
from scripts.sensitivity_analysis_run_script import run_SA


@pytest.fixture(scope="function")
def mpi_comm():
    """Fixture that provides MPI communicator."""
    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        # pytest.skip("MPI tests require mpiexec with at least 2 ranks")
        print("Running param ID and Sensitivity Analysis tests with 1 rank, this is slow")
    return comm


def _ensure_cellml_model_generated(config, mpi_comm):
    """
    Ensure generated CellML exists before run_SA.

    CI checkouts omit gitignored generated_models/; local runs may already have artifacts.
    """
    if config.get("model_type") != "cellml_only":
        return
    rank = mpi_comm.Get_rank()
    if rank == 0:
        success = generate_with_new_architecture(False, config)
        prefix = config.get("file_prefix", "<unknown>")
        assert success, f"CellML autogeneration failed for {prefix}"
    mpi_comm.Barrier()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_sensitivity_analysis_3compartment_succeeds(base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):
    """
    Test that sensitivity analysis succeeds for 3compartment model.
    
    Args:
        base_user_inputs: Base user inputs configuration fixture
        resources_dir: Resources directory fixture
        temp_output_dir: Temporary output directory fixture
        mpi_comm: MPI communicator fixture
    """
    rank = mpi_comm.Get_rank()
    
    # Setup configuration
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': '3compartment',
        'input_param_file': '3compartment_parameters.csv',
        'model_type': 'cellml_only',
        'solver': 'CVODE',
        'param_id_method': 'genetic_algorithm',
        'pre_time': 20,
        'sim_time': 2,
        'dt': 0.01,
        'DEBUG': True,
        'do_mcmc': True,
        'plot_predictions': True,
        'model_out_names': ['heart/u_lv'],
        'solver_info': {
            'MaximumStep': 0.001,
            'MaximumNumberOfSteps': 5000,
        },
        'param_id_obs_path': os.path.join(resources_dir, '3compartment_obs_data.json'),
        'param_id_output_dir': temp_output_dir,
        'generated_models_dir': temp_generated_models_dir,
        'debug_optimiser_options': {'num_calls_to_function': 60, 'cost_type': 'gaussian_MLE'},
        'sa_options': {
            'method': 'sobol',
            'num_samples': 16,
            'sample_type': 'saltelli',
            'output_dir': os.path.join(temp_output_dir, '3compartment_SA_results'),
        },
    })

    _ensure_cellml_model_generated(config, mpi_comm)

    # Run sensitivity analysis
    run_SA(config)
    
    # Verify output was created (on rank 0)
    if rank == 0:
        output_dir = config['sa_options']['output_dir']
        assert os.path.exists(output_dir), f"Sensitivity analysis output directory should exist: {output_dir}"


def test_sensitivity_analysis_3compartment_extra_ops_succeeds(
    base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm
):
    """
    Test that sensitivity analysis succeeds for 3compartment_extra_ops model.
    """
    rank = mpi_comm.Get_rank()

    config = base_user_inputs.copy()
    config.update({
        'file_prefix': '3compartment_extra_ops',
        'input_param_file': '3compartment_extra_ops_parameters.csv',
        'model_type': 'cellml_only',
        'solver': 'CVODE',
        'param_id_method': 'genetic_algorithm',
        'pre_time': 20,
        'sim_time': 2,
        'dt': 0.01,
        'DEBUG': True,
        'do_mcmc': True,
        'plot_predictions': True,
        'model_out_names': ['heart/u_lv'],  
        'solver_info': {
            'MaximumStep': 0.001,
            'MaximumNumberOfSteps': 5000,
        },
        'param_id_obs_path': os.path.join(resources_dir, '3compartment_extra_ops_obs_data.json'),
        'param_id_output_dir': temp_output_dir,
        'generated_models_dir': temp_generated_models_dir,
        'debug_optimiser_options': {'num_calls_to_function': 60},
        'sa_options': {
            'method': 'sobol',
            'num_samples': 16,
            'sample_type': 'saltelli',
            'output_dir': os.path.join(temp_output_dir, '3compartment_extra_ops_SA_results'),
        },
    })

    _ensure_cellml_model_generated(config, mpi_comm)

    run_SA(config)

    if rank == 0:
        output_dir = config['sa_options']['output_dir']
        assert os.path.exists(output_dir), \
            f"Sensitivity analysis output directory should exist: {output_dir}"


def _build_local_sa_engine(base_user_inputs, resources_dir, temp_output_dir,
                           temp_generated_models_dir, mpi_comm, model_type, solver):
    """Generate the 3compartment model for `model_type` and build a CVS0DParamID engine set up
    for the backend-agnostic local-sensitivity accessor (do_ad on, q_lv_init included so the
    initial-value chain rule is exercised on the Myokit path)."""
    import json
    from parsers.PrimitiveParsers import YamlFileParser
    from param_id.paramID import CVS0DParamID

    obs_file = '3compartment_obs_data.json'
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': '3compartment',
        'input_param_file': '3compartment_parameters.csv',
        'model_type': model_type,
        'solver': solver,
        'do_ad': True,
        'pre_time': 0.3,
        'sim_time': 0.5,
        'dt': 0.01,
        'DEBUG': False,
        'do_mcmc': False,
        'plot_predictions': False,
        'solver_info': {'MaximumStep': 0.005, 'MaximumNumberOfSteps': 50000,
                        'rtol': 1e-9, 'atol': 1e-9} if model_type == 'cellml_only'
                       else {'method': 'bdf'},
        'param_id_obs_path': os.path.join(resources_dir, obs_file),
        'param_id_output_dir': temp_output_dir,
        'generated_models_dir': temp_generated_models_dir,
    })
    _ensure_cellml_model_generated(config, mpi_comm)
    if mpi_comm.Get_rank() == 0 and model_type != 'cellml_only':
        assert generate_with_new_architecture(False, config), f"generation failed for {model_type}"
    mpi_comm.Barrier()

    parsed = YamlFileParser().parse_user_inputs_file(config)
    parsed['one_rank'] = True
    engine_outer = CVS0DParamID.init_from_dict(parsed)
    with open(os.path.join(resources_dir, obs_file)) as f:
        obs_data = json.load(f)
    engine_outer.set_ground_truth_data(obs_data)
    engine_outer.set_params_for_id([
        {'vessel_name': 'global',      'param_name': 'q_lv_init', 'param_type': 'const', 'min': 200e-6, 'max': 1500e-6},
        {'vessel_name': 'aortic_root', 'param_name': 'C',         'param_type': 'const', 'min': 1e-9,   'max': 5e-8},
    ])
    return engine_outer


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_local_observable_sensitivities_match_fd(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):
    """The backend-agnostic OpencorParamID.get_observable_sensitivities returns d(feature)/d(param)
    that matches central finite differences, for the Myokit CVODES backend. (CasADi's arm is the
    exact symbolic jacobian, cross-checked against this one in
    test_local_observable_sensitivities_casadi_agrees_with_myokit.)
    """
    import numpy as np
    model_type, solver = 'cellml_only', 'CVODE_myokit'
    engine_outer = _build_local_sa_engine(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir,
        mpi_comm, model_type, solver)
    engine = engine_outer.param_id

    param_names = [n[0] if isinstance(n, list) else n
                   for n in engine.param_id_info["param_names"]]
    nominal = np.asarray(engine.sim_helper.get_init_param_vals(param_names), dtype=float).ravel()

    sens = engine.get_observable_sensitivities(nominal)   # {obs_label: {param: d(feat)/dp}}
    assert sens, "no observable sensitivities returned"
    # q_lv_init must appear with a nonzero effect on at least one observable (chain rule / symbolic).
    assert any(abs(v.get('global/q_lv_init', 0.0)) > 0 for v in sens.values()), \
        "q_lv_init has no local sensitivity"

    _OPS = {'mean': np.mean, 'max': np.max, 'min': np.min,
            'max_minus_min': lambda a: np.max(a) - np.min(a)}

    def features_at(vals):
        # Numeric feature values computed straight from the sim_helper operand traces with numpy,
        # so the FD reference is independent of each backend's (numpy vs casadi) operation funcs.
        engine.sim_helper.set_param_vals(engine.param_id_info["param_names"], list(vals))
        engine.sim_helper.reset_states()
        assert engine.sim_helper.run(), "FD reference sim failed to converge"
        operands = engine.sim_helper.get_results(engine.obs_info["operands"])
        engine.sim_helper.reset_and_clear()
        feats = []
        for k, obs_i in enumerate(engine.obs_info["const_idx_to_obs_idx"]):
            op = engine.obs_info["operations"][obs_i]
            arr = np.asarray(operands[obs_i][0], dtype=float)
            feats.append(float(_OPS[op](arr)))
        return np.asarray(feats, dtype=float)

    c2o = engine.obs_info["const_idx_to_obs_idx"]
    labels = [engine._observable_label(obs_i) for obs_i in c2o]

    checked = 0
    for jcol, pname in enumerate(param_names):
        h = 1e-3 * abs(nominal[jcol]) if nominal[jcol] != 0 else 1e-6
        vp = nominal.copy(); vp[jcol] += h
        vm = nominal.copy(); vm[jcol] -= h
        fd = (features_at(vp) - features_at(vm)) / (2 * h)
        for k, label in enumerate(labels):
            ad = float(sens[label].get(pname, 0.0))
            if abs(fd[k]) < 1e-9 * (abs(nominal[jcol]) + 1e-30) and abs(ad) < 1e-9:
                continue
            denom = max(abs(fd[k]), abs(ad), 1e-30)
            assert abs(ad - fd[k]) / denom < 0.15, (
                f"[{model_type}] d({label})/d({pname}) AD={ad:.4e} FD={fd[k]:.4e}")
            checked += 1
    assert checked > 0, "no (observable, param) pair was above the FD noise floor to check"

    if hasattr(engine_outer, 'close_simulation'):
        engine_outer.close_simulation()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_sensitivity_analysis_local_method_end_to_end(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):
    """SensitivityAnalysis with sa_options method='local' runs through the backend-agnostic engine
    (not sobol_SA), saves the CSV matrices, and exposes get_local_sensitivities()."""
    import json
    import numpy as np
    from parsers.PrimitiveParsers import YamlFileParser
    from sensitivity_analysis.sensitivityAnalysis import SensitivityAnalysis

    rank = mpi_comm.Get_rank()
    obs_file = '3compartment_obs_data.json'
    out_dir = os.path.join(temp_output_dir, '3compartment_local_SA_results')
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': '3compartment',
        'input_param_file': '3compartment_parameters.csv',
        'model_type': 'cellml_only',
        'solver': 'CVODE_myokit',
        'pre_time': 0.3,
        'sim_time': 0.5,
        'dt': 0.01,
        'DEBUG': False,
        'do_mcmc': False,
        'plot_predictions': False,
        'solver_info': {'MaximumStep': 0.005, 'MaximumNumberOfSteps': 50000,
                        'rtol': 1e-9, 'atol': 1e-9},
        'param_id_obs_path': os.path.join(resources_dir, obs_file),
        'param_id_output_dir': temp_output_dir,
        'generated_models_dir': temp_generated_models_dir,
        'sa_options': {'method': 'local', 'num_samples': 8, 'sample_type': 'saltelli',
                       'output_dir': out_dir},
    })
    _ensure_cellml_model_generated(config, mpi_comm)

    parsed = YamlFileParser().parse_user_inputs_file(config)
    sa = SensitivityAnalysis.init_from_dict(parsed)
    with open(os.path.join(resources_dir, obs_file)) as f:
        obs_data = json.load(f)
    sa.set_ground_truth_data(obs_data)
    sa.set_params_for_id([
        {'vessel_name': 'global',      'param_name': 'q_lv_init', 'param_type': 'const', 'min': 200e-6, 'max': 1500e-6},
        {'vessel_name': 'aortic_root', 'param_name': 'C',         'param_type': 'const', 'min': 1e-9,   'max': 5e-8},
    ])

    sa.run_sensitivity_analysis()  # dispatches on method='local'

    local = sa.get_local_sensitivities()
    assert local is not None and 'absolute' in local and 'relative' in local
    assert local['absolute'].shape == (len(local['output_names']), len(local['param_names']))
    assert np.all(np.isfinite(local['absolute']))
    assert local['absolute'].shape[0] > 0
    if rank == 0:
        for key in ('relative', 'absolute'):
            assert os.path.exists(os.path.join(out_dir, f'local_sensitivity_{key}.csv'))


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_local_observable_sensitivities_casadi_agrees_with_myokit(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):
    """The two backends of get_observable_sensitivities report the same d(feature)/d(param).

    Myokit (CVODES + directional derivative) is finite-difference-verified in the test above;
    CasADi is the exact symbolic jacobian. Built from the same CellML at the same nominal params,
    they must agree observable-by-observable and param-by-param. This is the backend-consistency
    guarantee the refactor exists for, and it also validates the CasADi obs_meta row -> observable
    mapping (a mapping bug would misassign sensitivities and break the agreement).
    """
    import numpy as np

    def sens_for(model_type, solver):
        eo = _build_local_sa_engine(
            base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir,
            mpi_comm, model_type, solver)
        eng = eo.param_id
        pnames = [n[0] if isinstance(n, list) else n for n in eng.param_id_info["param_names"]]
        nominal = np.asarray(eng.sim_helper.get_init_param_vals(pnames), dtype=float).ravel()
        return eng.get_observable_sensitivities(nominal), pnames

    myo, pnames_m = sens_for('cellml_only', 'CVODE_myokit')
    cas, pnames_c = sens_for('casadi_python', 'casadi_integrator')

    assert pnames_m == pnames_c
    assert set(myo.keys()) == set(cas.keys()), (sorted(myo), sorted(cas))

    # Compare only the *mean* observables. The two arms use different integrators (CVODE_myokit
    # adaptive vs CasADi fixed-step bdf), so their trajectories differ slightly; for a smooth
    # linear functional like the mean that is a sub-percent effect, but for max/min -- which pick
    # out a single argmax/argmin time -- the two solvers can land on different extremum samples
    # and legitimately disagree. So max/min cross-solver agreement is not a meaningful check; the
    # Myokit arm's max/min are finite-difference-verified in the test above, and the row->observable
    # mapping being exercised here is the same code path for every const observable.
    checked = 0
    for label in myo:
        if '(mean ' not in label:
            continue
        for p in pnames_m:
            a = float(myo[label].get(p, 0.0))
            b = float(cas[label].get(p, 0.0))
            scale = max(abs(a), abs(b))
            if scale < 1e-12:
                continue
            assert abs(a - b) / scale < 0.05, (
                f"backends disagree on d({label})/d({p}): myokit={a:.4e} casadi={b:.4e}")
            checked += 1
    assert checked > 0, "no mean-observable/param pair with a non-trivial sensitivity to compare"


@pytest.mark.unit
def test_sobolSA_generate_samples_supports_both_sample_types():
    """Both advertised sample_type choices must actually produce samples.

    `sample_type: sobol` used to raise `AttributeError: module 'SALib.analyze.sobol' has no
    attribute 'sample'` because the name `sobol` in sobolSA.py was the SALib *analyzer* import,
    not the *sampler* -- so only `saltelli` worked, even though the schema advertises both. This
    exercises generate_samples directly (no model needed) for both types.
    """
    import numpy as np
    from sensitivity_analysis.sobolSA import sobol_SA

    # generate_samples only reads num_params + SA_info, so build a bare instance to avoid the
    # heavy __init__ (which loads a model). This keeps the check a fast unit test of the dispatch.
    mgr = object.__new__(sobol_SA)
    mgr.num_params = 2
    for sample_type in ('saltelli', 'sobol'):
        mgr.SA_info = {
            'param_names': ['a', 'b'],
            'param_mins': [0.0, 0.0],
            'param_maxs': [1.0, 1.0],
            'num_samples': 8,
            'sample_type': sample_type,
        }
        samples = mgr.generate_samples()
        samples = np.asarray(samples)
        assert samples.ndim == 2 and samples.shape[1] == 2, (sample_type, samples.shape)
        assert samples.shape[0] > 0, sample_type
        assert np.all(np.isfinite(samples)), sample_type
