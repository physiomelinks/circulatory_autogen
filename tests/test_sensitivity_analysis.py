"""
Tests for sensitivity analysis functionality.

These tests verify that sensitivity analysis works correctly for various models.
"""
import json
import os

import numpy as np
import pandas as pd
from _obs_data import retimed_obs_data
import pytest
MPI = pytest.importorskip("mpi4py.MPI", reason='needs mpi4py: pip install "libcuflynx[mpi]"')

from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
from libcuflynx.scripts.sensitivity_analysis_run_script import run_SA


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
    if config.get("model_type") != "cellml":
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
        'input_param_file': '3compartment_parameters.json',
        'model_type': 'cellml',
        'solver': 'CVODE_myokit',
        'param_id_method': 'genetic_algorithm',
        'pre_time': 20,
        'sim_time': 2,
        'dt': 0.01,
        'DEBUG': True,
        'do_uq': True,
        'plot_predictions': True,
        'model_out_names': ['heart/u_lv'],
        'solver_info': {
            'MaximumStep': 0.001,
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
        # the reduction after the Sobol run, with every other option filled in by the parser
        'do_param_reduction': True,
        'param_reduction_options': {'method': 'pca'},
    })

    _ensure_cellml_model_generated(config, mpi_comm)

    # Run sensitivity analysis
    run_SA(config)
    
    # Verify output was created (on rank 0)
    if rank == 0:
        output_dir = config['sa_options']['output_dir']
        assert os.path.exists(output_dir), f"Sensitivity analysis output directory should exist: {output_dir}"
        # the samples and outputs, in Saltelli order: num_samples (2M + 2) rows, M = 4
        indices = pd.read_csv(os.path.join(output_dir, 'all_outputs_n16_Sobol_indices.csv'))
        samples = pd.read_csv(os.path.join(output_dir, 'all_outputs_n16_Sobol_samples.csv'))
        outputs = pd.read_csv(os.path.join(output_dir, 'all_outputs_n16_Sobol_outputs.csv'))
        assert len(samples) == len(outputs) == 16 * (2 * 4 + 2)
        assert list(samples.columns) == list(indices['Parameter'])
        assert [f'ST_{c}' for c in outputs.columns] == [c for c in indices.columns
                                                        if c.startswith('ST_')]
        with open(os.path.join(output_dir, 'sobol_problem.json')) as f:
            problem = json.load(f)
        # each output's obs_data std and weight (3compartment_obs_data.json's first item)
        assert problem['outputs'][0]['std'] == pytest.approx(1.0731181019893536e-05)
        assert problem['outputs'][0]['weight'] == 1.0
        # the local Jacobian: one row per parameter at each of the first 8 Saltelli base points
        jacobian = pd.read_csv(os.path.join(output_dir,
                                            'all_outputs_n16_Sobol_local_jacobian.csv'))
        rows = [10 * k for k in range(8)]          # blocks of 2M + 2 = 10 rows
        assert list(jacobian['point']) == [r for r in rows for _ in range(4)]
        assert set(jacobian['step']) == {1e-2}      # central differences, one step
        assert list(jacobian['Parameter']) == list(indices['Parameter']) * 8
        assert list(jacobian.columns[3:]) == list(outputs.columns)
        assert np.all(np.isfinite(jacobian.iloc[:, 3:].to_numpy()))
        assert problem['local_jacobian']['jacobian_scheme'] == 'central'
        assert problem['local_jacobian']['points'] == rows
        assert problem['local_jacobian']['failed_points'] == []
        reduction_dir = os.path.join(output_dir, 'param_reduction')
        assert os.path.isfile(os.path.join(reduction_dir, 'reduced_params_for_id.json'))
        with open(os.path.join(reduction_dir, 'param_reduction.json')) as f:
            report = json.load(f)
        assert report['reduction']['method'] == 'pca' and report['selected']


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
        'input_param_file': '3compartment_extra_ops_parameters.json',
        'model_type': 'cellml',
        'solver': 'CVODE_myokit',
        'param_id_method': 'genetic_algorithm',
        'pre_time': 20,
        'sim_time': 2,
        'dt': 0.01,
        'DEBUG': True,
        'do_uq': True,
        'plot_predictions': True,
        'model_out_names': ['heart/u_lv'],  
        'solver_info': {
            'MaximumStep': 0.001,
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
    from libcuflynx.parsers.PrimitiveParsers import YamlFileParser
    from libcuflynx.param_id.paramID import CVS0DParamID

    obs_file = '3compartment_obs_data.json'
    obs_path = retimed_obs_data(os.path.join(resources_dir, obs_file), 0.3, 0.5,
                                temp_output_dir)
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': '3compartment',
        'input_param_file': '3compartment_parameters.json',
        'model_type': model_type,
        'solver': solver,
        'do_ad': True,
        'pre_time': 0.3,
        'sim_time': 0.5,
        'dt': 0.01,
        'DEBUG': False,
        'do_uq': False,
        'plot_predictions': False,
        'solver_info': {'MaximumStep': 0.005,
                        'rtol': 1e-9, 'atol': 1e-9} if model_type == 'cellml'
                       else {'method': 'bdf'},
        'param_id_obs_path': obs_path,
        'param_id_output_dir': temp_output_dir,
        'generated_models_dir': temp_generated_models_dir,
    })
    _ensure_cellml_model_generated(config, mpi_comm)
    if mpi_comm.Get_rank() == 0 and model_type != 'cellml':
        assert generate_with_new_architecture(False, config), f"generation failed for {model_type}"
    mpi_comm.Barrier()

    parsed = YamlFileParser().parse_user_inputs_file(config)
    parsed['one_rank'] = True
    engine_outer = CVS0DParamID.init_from_dict(parsed)
    with open(obs_path) as f:
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
    """The backend-agnostic ParamID.get_observable_sensitivities returns d(feature)/d(param)
    that matches central finite differences, for the Myokit CVODES backend. (CasADi's arm is the
    exact symbolic jacobian, cross-checked against this one in
    test_local_observable_sensitivities_casadi_agrees_with_myokit.)
    """
    import numpy as np
    model_type, solver = 'cellml', 'CVODE_myokit'
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

    # The FD reference is only meaningful where the feature moves across the bracket by more than
    # the integrator can reproduce. CVODE here runs at rtol/atol 1e-9, so a feature that changes
    # by ~1e-8 relative between the two evaluations is measured against its own solver noise.
    # d(mean heart/u_la)/d(aortic_root/C) is the hard pair: at a step of 1e-3 relative it moves
    # ~1e-7 and the FD reads 2e6-6e6 (it depends on the heart's clock form) against an analytic
    # 3.89e6. At FD_STEP = 3e-2 relative every pair moves by more than 1e-6 and agrees with the
    # analytic value to 1% (3.92e6 for that pair; 1e-1 gives the same), while the parameters stay
    # close enough to nominal for a central difference.
    #
    # Guarding on abs(fd) does not catch noise -- a nanoscale numerator over a nanoscale step is a
    # large number. The quantity that is small is the *signal*, so guard on that as well.
    FD_STEP = 3e-2  # relative to the nominal value
    FEATURE_SIGNAL_FLOOR = 1e-7  # relative change across the 2h bracket; ~10x the 1e-8 noise
    f0 = features_at(nominal)
    checked = 0
    skipped_for_noise = []
    for jcol, pname in enumerate(param_names):
        h = FD_STEP * abs(nominal[jcol]) if nominal[jcol] != 0 else 1e-6
        vp = nominal.copy(); vp[jcol] += h
        vm = nominal.copy(); vm[jcol] -= h
        fp, fm = features_at(vp), features_at(vm)
        fd = (fp - fm) / (2 * h)
        for k, label in enumerate(labels):
            ad = float(sens[label].get(pname, 0.0))
            signal = abs(fp[k] - fm[k]) / max(abs(f0[k]), 1e-300)
            if signal < FEATURE_SIGNAL_FLOOR:
                skipped_for_noise.append(f"d({label})/d({pname}) signal={signal:.1e}")
                continue
            if abs(fd[k]) < 1e-9 * (abs(nominal[jcol]) + 1e-30) and abs(ad) < 1e-9:
                continue
            denom = max(abs(fd[k]), abs(ad), 1e-30)
            assert abs(ad - fd[k]) / denom < 0.15, (
                f"[{model_type}] d({label})/d({pname}) AD={ad:.4e} FD={fd[k]:.4e}")
            checked += 1
    # Report what was dropped rather than letting a silently-shrinking comparison set look like
    # coverage -- if this ever swallows everything interesting, it should be visible with -s.
    if skipped_for_noise:
        print(f"\n[match_fd] {len(skipped_for_noise)} pair(s) below the FD signal floor, not "
              f"compared: {'; '.join(skipped_for_noise)}")
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
    from libcuflynx.parsers.PrimitiveParsers import YamlFileParser
    from libcuflynx.sensitivity_analysis.sensitivityAnalysis import SensitivityAnalysis

    rank = mpi_comm.Get_rank()
    obs_file = '3compartment_obs_data.json'
    out_dir = os.path.join(temp_output_dir, '3compartment_local_SA_results')
    obs_path = retimed_obs_data(os.path.join(resources_dir, obs_file), 0.3, 0.5,
                                temp_output_dir)
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': '3compartment',
        'input_param_file': '3compartment_parameters.json',
        'model_type': 'cellml',
        'solver': 'CVODE_myokit',
        'pre_time': 0.3,
        'sim_time': 0.5,
        'dt': 0.01,
        'DEBUG': False,
        'do_uq': False,
        'plot_predictions': False,
        'solver_info': {'MaximumStep': 0.005,
                        'rtol': 1e-9, 'atol': 1e-9},
        'param_id_obs_path': obs_path,
        'param_id_output_dir': temp_output_dir,
        'generated_models_dir': temp_generated_models_dir,
        'sa_options': {'method': 'local', 'num_samples': 8, 'sample_type': 'saltelli',
                       'output_dir': out_dir},
    })
    _ensure_cellml_model_generated(config, mpi_comm)

    parsed = YamlFileParser().parse_user_inputs_file(config)
    sa = SensitivityAnalysis.init_from_dict(parsed)
    with open(obs_path) as f:
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
@pytest.mark.xfail(
    strict=False,
    reason="Known cross-backend discrepancy on d(mean heart/u_la)/d(aortic_root/C): myokit "
           "3.8905e+06 vs casadi 3.6519e+06 (6.1%, tolerance 5%). Not under-resolution -- both "
           "backends were swept over their own resolution controls and each converges (myokit "
           "flat to 0.01% across MaximumStep, casadi to 0.29% across max_step) to a different "
           "value. Leading suspect is how the two translations handle the conditionals behind "
           "heart/u_la (chi_a - floor(chi_a) gated by leq_func/and_func; CasADi emits "
           "ca.if_else). Tracked in issue #362 -- remove this marker when that is fixed. This "
           "observable was added here by the 3compartment identifiability work, which exposed "
           "the discrepancy but did not cause it.")
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

    myo, pnames_m = sens_for('cellml', 'CVODE_myokit')
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
    from libcuflynx.sensitivity_analysis.sobolSA import sobol_SA

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


@pytest.mark.unit
def test_init_from_all_dicts_is_a_classmethod(monkeypatch):
    """Regression for #369: the decorator was missing, so the method was an *instance* method
    and calling it on the class -- its only sensible use -- raised a TypeError before any of
    its body ran."""
    from types import SimpleNamespace
    from libcuflynx.sensitivity_analysis.sensitivityAnalysis import SensitivityAnalysis

    received = {}
    stub = SimpleNamespace(
        set_ground_truth_data=lambda v: received.setdefault('obs', v),
        set_params_for_id=lambda v: received.setdefault('params', v),
        set_sa_options=lambda v: received.setdefault('sa', v),
    )
    monkeypatch.setattr(SensitivityAnalysis, 'init_from_dict',
                        classmethod(lambda cls, d: received.setdefault('inp', d) and stub or stub))

    out = SensitivityAnalysis.init_from_all_dicts({'file_prefix': 'x'}, 'obs', 'params', 'sa')
    assert out is stub
    assert received == {'inp': {'file_prefix': 'x'}, 'obs': 'obs', 'params': 'params', 'sa': 'sa'}


@pytest.mark.integration
@pytest.mark.mpi
def test_a_failed_sobol_sample_is_filled_from_its_own_outputs(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):
    """A sample whose simulation fails gets, for each output, that output's mean over the
    samples that ran: never a value of another output, whatever its units."""
    from libcuflynx.sensitivity_analysis.sensitivityAnalysis import SensitivityAnalysis
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': '3compartment', 'input_param_file': '3compartment_parameters.json',
        'model_type': 'cellml', 'solver': 'CVODE_myokit', 'pre_time': 20, 'sim_time': 2,
        'dt': 0.01, 'solver_info': {'MaximumStep': 0.001},
        'param_id_obs_path': os.path.join(resources_dir, '3compartment_obs_data.json'),
        'param_id_output_dir': temp_output_dir, 'generated_models_dir': temp_generated_models_dir,
        'sa_options': {'method': 'sobol', 'num_samples': 2, 'sample_type': 'saltelli',
                       'output_dir': os.path.join(temp_output_dir, 'failed_samples')}})
    _ensure_cellml_model_generated(config, mpi_comm)
    if mpi_comm.Get_size() > 1:
        pytest.skip('counts the calls of one rank')
    from libcuflynx.parsers.PrimitiveParsers import YamlFileParser
    config = YamlFileParser().parse_user_inputs_file(config, obs_path_needed=True,
                                                     do_generation_with_fit_parameters=False)
    manager = SensitivityAnalysis.init_from_dict(config).SA_manager
    manager.set_sa_options(config['sa_options'])
    executor = manager._protocol_executor
    run_protocol, calls = executor.run_protocol, []

    def every_third_fails(*args, **kwargs):
        calls.append(1)
        if len(calls) % 3 == 1:
            return False, {}, None, None
        return run_protocol(*args, **kwargs)

    executor.run_protocol = every_third_fails
    samples = manager.generate_samples()[:6]
    raw = np.array(manager.generate_outputs_mpi(samples), dtype=float)
    assert np.all(np.isnan(raw[[0, 3]])) and np.all(np.isfinite(raw[[1, 2, 4, 5]]))
    with pytest.warns(UserWarning, match='2 of 6 Sobol samples'):
        filled = manager.fill_failed_outputs(raw)
    ran = raw[[1, 2, 4, 5]].mean(axis=0)
    assert filled[0] == pytest.approx(ran) and filled[3] == pytest.approx(ran)
    assert manager.failed_samples == 2


@pytest.mark.integration
@pytest.mark.mpi
def test_casadi_sobol_outputs_are_one_number_per_sample_and_output(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):
    """casadi_python's operation funcs return 1x1 matrices. Each Sobol output is one float, so
    the outputs are a (samples, outputs) array SALib can index column by column."""
    import numpy as np
    from libcuflynx.parsers.PrimitiveParsers import YamlFileParser
    from libcuflynx.sensitivity_analysis.sensitivityAnalysis import SensitivityAnalysis
    if mpi_comm.Get_size() > 1:
        pytest.skip('checks the outputs of one rank')
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': '3compartment', 'input_param_file': '3compartment_parameters.json',
        'model_type': 'casadi_python', 'solver': 'casadi_integrator', 'pre_time': 0.3,
        'sim_time': 0.5, 'dt': 0.01, 'solver_info': {'method': 'bdf'},
        'param_id_obs_path': retimed_obs_data(
            os.path.join(resources_dir, '3compartment_obs_data.json'), 0.3, 0.5, temp_output_dir),
        'param_id_output_dir': temp_output_dir, 'generated_models_dir': temp_generated_models_dir,
        'sa_options': {'method': 'sobol', 'num_samples': 2, 'sample_type': 'saltelli',
                       'output_dir': os.path.join(temp_output_dir, 'casadi_sobol')}})
    assert generate_with_new_architecture(False, config), 'casadi_python generation failed'
    config = YamlFileParser().parse_user_inputs_file(config, obs_path_needed=True,
                                                     do_generation_with_fit_parameters=False)
    manager = SensitivityAnalysis.init_from_dict(config).SA_manager
    manager.set_sa_options(config['sa_options'])
    samples = manager.generate_samples()[:3]
    outputs = manager.generate_outputs_mpi(samples)
    as_array = np.asarray(outputs)
    assert as_array.dtype == float and as_array.ndim == 2 and as_array.shape[0] == 3, \
        (as_array.dtype, as_array.shape)
    assert np.all(np.isfinite(as_array))


@pytest.mark.integration
@pytest.mark.mpi
def test_a_sobol_run_writes_its_progress_design_and_sample_status(
        base_user_inputs, resources_dir, temp_output_dir, temp_generated_models_dir, mpi_comm):
    """A Sobol run leaves progress.json (done, every sample counted, on any number of ranks),
    the design it ran, each sample's status with the failed ones marked, and the convergence
    file, beside its indices."""
    import warnings
    from libcuflynx.parsers.PrimitiveParsers import YamlFileParser
    from libcuflynx.schemas import PROGRESS_SCHEMA
    from libcuflynx.schemas.validation import validate
    from libcuflynx.sensitivity_analysis.sensitivityAnalysis import SensitivityAnalysis
    from libcuflynx.sensitivity_analysis.sobolSA import (
        SAMPLE_FAILED, SAMPLE_OK, SOBOL_CONVERGENCE_FILE, SOBOL_DESIGN_FILE,
        SOBOL_SAMPLE_STATUS_FILE)
    out_dir = os.path.join(temp_output_dir, 'sobol_progress')
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': '3compartment', 'input_param_file': '3compartment_parameters.json',
        'model_type': 'cellml', 'solver': 'CVODE_myokit', 'pre_time': 20, 'sim_time': 2,
        'dt': 0.01, 'solver_info': {'MaximumStep': 0.001},
        'param_id_obs_path': os.path.join(resources_dir, '3compartment_obs_data.json'),
        'param_id_output_dir': temp_output_dir, 'generated_models_dir': temp_generated_models_dir,
        'sa_options': {'method': 'sobol', 'num_samples': 2, 'sample_type': 'saltelli',
                       'jacobian_points': 0, 'output_dir': out_dir}})
    _ensure_cellml_model_generated(config, mpi_comm)
    config = YamlFileParser().parse_user_inputs_file(config, obs_path_needed=True,
                                                     do_generation_with_fit_parameters=False)
    analysis = SensitivityAnalysis.init_from_dict(config)
    manager = analysis.SA_manager
    manager.set_sa_options(config['sa_options'])
    executor = manager._protocol_executor
    run_protocol, calls = executor.run_protocol, []

    def every_fourth_fails(*args, **kwargs):
        calls.append(1)
        if len(calls) % 4 == 1:
            return False, {}, None, None
        return run_protocol(*args, **kwargs)

    executor.run_protocol = every_fourth_fails
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')     # the failed samples are filled, with a warning
        analysis.run_sobol_sensitivity(config['sa_options'])
    if mpi_comm.Get_rank() != 0:
        return

    samples = np.asarray(manager.samples)
    # each rank fails the first of every four of its own block
    expected, size = np.full(len(samples), SAMPLE_OK), mpi_comm.Get_size()
    per_rank, extra = divmod(len(samples), size)
    start = 0
    for rank in range(size):
        count = per_rank + (1 if rank < extra else 0)
        expected[start:start + count:4] = SAMPLE_FAILED
        start += count

    status = np.load(os.path.join(out_dir, SOBOL_SAMPLE_STATUS_FILE))
    assert status.tolist() == expected.tolist()
    assert np.load(os.path.join(out_dir, SOBOL_DESIGN_FILE)) == pytest.approx(samples)
    with open(os.path.join(out_dir, 'progress.json')) as f:
        document = json.load(f)
    validate(PROGRESS_SCHEMA, document, source='progress.json', what='progress file')
    assert document['kind'] == 'sobol' and document['state'] == 'done'
    failed = int(np.count_nonzero(expected == SAMPLE_FAILED))
    assert [(p['name'], p['state'], p['done'], p['total'], p['failed'])
            for p in document['phases']] == [
        ('sampling', 'done', len(samples), len(samples), failed), ('analysis', 'done', 1, 1, 0)]
    assert manager.failed_samples == failed
    with open(os.path.join(out_dir, SOBOL_CONVERGENCE_FILE)) as f:
        convergence = json.load(f)
    assert convergence['n'] == [2] and len(convergence['outputs']) > 0
    indices = pd.read_csv(os.path.join(out_dir, 'all_outputs_n2_Sobol_indices.csv'))
    assert len(indices) > 0, 'the indices were written before the state became done'
