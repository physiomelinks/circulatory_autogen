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
