"""The identifiability script hands the user-funcs paths to the engine it builds, as the
param_id and sensitivity scripts do."""
import pytest

from libcuflynx.scripts import identifiability_run_script as script


class _Stop(Exception):
    pass


@pytest.mark.unit
def test_identifiability_script_passes_user_func_paths_to_the_engine(monkeypatch):
    seen = {}

    def fake_engine(*args, **kwargs):
        seen.update(kwargs)
        raise _Stop

    inp = {'DEBUG': False, 'model_path': 'm.cellml', 'model_type': 'cellml',
           'param_id_method': 'CMA-ES', 'file_prefix': 'p', 'params_for_id_path': 'pfi.json',
           'param_id_obs_path': 'obs.json', 'sim_time': 1, 'pre_time': 0,
           'solver_info': {'solver': 'CVODE_myokit'}, 'dt': 0.1, 'optimiser_options': {},
           'resources_dir': 'r', 'param_id_output_dir': 'o',
           'operation_funcs_external_path': 'ops.py', 'cost_funcs_external_path': 'costs.py',
           'modifier_funcs_external_path': 'mods.py'}
    monkeypatch.setattr(script.YamlFileParser, 'parse_user_inputs_file',
                        lambda self, *a, **k: inp)
    monkeypatch.setattr(script, 'CVS0DParamID', fake_engine)
    with pytest.raises(_Stop):
        script.run_identifiability_analysis(inp, best_param_vals=[1.0])
    assert {k: seen.get(k) for k in ('operation_funcs_external_path',
                                     'cost_funcs_external_path',
                                     'modifier_funcs_external_path')} == {
        'operation_funcs_external_path': 'ops.py', 'cost_funcs_external_path': 'costs.py',
        'modifier_funcs_external_path': 'mods.py'}


USER_FUNC_KEYS = ('operation_funcs_external_path', 'cost_funcs_external_path',
                  'modifier_funcs_external_path')
USER_FUNCS = {'operation_funcs_external_path': 'ops.py', 'cost_funcs_external_path': 'costs.py',
              'modifier_funcs_external_path': 'mods.py'}


@pytest.mark.unit
def test_plot_param_id_script_passes_user_func_paths_to_the_engine(monkeypatch):
    from libcuflynx.scripts import plot_param_id_script as plot_script
    seen = {}

    def fake_engine(*args, **kwargs):
        seen.update(kwargs)
        raise _Stop

    inp = {'DEBUG': False, 'model_path': 'm.cellml', 'uncalibrated_model_path': 'u.cellml',
           'model_type': 'cellml', 'param_id_method': 'CMA-ES', 'file_prefix': 'p',
           'params_for_id_path': 'pfi.json', 'param_id_obs_path': 'obs.json', 'sim_time': 1,
           'pre_time': 0, 'solver_info': {'solver': 'CVODE_myokit'}, 'dt': 0.1,
           'optimiser_options': {}, 'UQ_options': {}, 'resources_dir': 'r',
           'param_id_output_dir': 'o', 'param_id_output_dir_abs_path': 'o',
           'plot_predictions': False, 'do_sensitivity': False, 'do_uq': False,
           'generated_models_subdir': 'g', 'do_ia': False, 'ia_options': {}, **USER_FUNCS}
    monkeypatch.setattr(plot_script.YamlFileParser, 'parse_user_inputs_file',
                        lambda self, *a, **k: inp)
    monkeypatch.setattr(plot_script, 'generate_with_new_architecture', lambda *a, **k: None)
    monkeypatch.setattr(plot_script, 'CVS0DParamID', fake_engine)
    with pytest.raises(_Stop):
        plot_script.plot_param_id(inp, generate=True)
    assert {k: seen.get(k) for k in USER_FUNC_KEYS} == USER_FUNCS


@pytest.mark.unit
def test_plot_mcmc_and_predictions_builds_its_engine_with_the_user_func_paths(monkeypatch):
    from libcuflynx.param_id import paramID
    seen = {}

    def fake_engine(*args, **kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(paramID, 'CVS0DParamID', fake_engine)
    plotter = object.__new__(paramID.MCMC_plotter)
    plotter.rank = 0
    for name in ('model_path', 'model_type', 'param_id_method', 'file_name_prefix',
                 'params_for_id_path', 'param_id_obs_path', 'sim_time', 'pre_time', 'dt',
                 'param_id_output_dir', 'resources_dir', 'solver_info', 'UQ_options', 'DEBUG'):
        setattr(plotter, name, None)
    plotter.operation_funcs_external_path = 'ops.py'
    plotter.cost_funcs_external_path = 'costs.py'
    plotter.modifier_funcs_external_path = 'mods.py'
    with pytest.raises(_Stop):
        plotter.plot_mcmc_and_predictions()
    assert {k: seen.get(k) for k in USER_FUNC_KEYS} == USER_FUNCS


@pytest.mark.unit
def test_mcmc_plotter_keeps_the_user_func_paths(tmp_path):
    from libcuflynx.param_id import paramID
    plotter = paramID.MCMC_plotter('m.cellml', 'cellml', 'CMA-ES', 'p', param_id_obs_path='obs.json',
                                   param_id_output_dir=str(tmp_path), **USER_FUNCS)
    assert {k: getattr(plotter, k) for k in USER_FUNC_KEYS} == USER_FUNCS


@pytest.mark.unit
def test_run_multiple_param_id_passes_user_func_paths_to_every_engine():
    """The script runs only under __main__ on fixed paths, so its CVS0DParamID calls are read
    from the source."""
    import ast
    from libcuflynx.scripts import run_multiple_param_id
    with open(run_multiple_param_id.__file__) as f:
        tree = ast.parse(f.read())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, 'id', None) == 'CVS0DParamID']
    assert calls
    for call in calls:
        assert set(USER_FUNC_KEYS) <= {k.arg for k in call.keywords}
