'''``cuflynx-upgrade junctions`` (libcuflynx.junction_upgrade).

The unit tests run on a small library written here (its ``renames.json`` and the configs of the
versions the junctions become). The last test uses the real module library: the fixture model
``tests/test_inputs/legacy_junctions/junction_net`` (a 2in2out junction fed by a two-inflow
source, a split, a flow_merge and a Min junction) is generated as it is, with the removed junction
versions as external modules, and after the upgrade with the library alone; the two must be the
same model.
'''
import json
import os
import shutil

import numpy as np
import pytest

from libcuflynx import upgrade
from libcuflynx.junction_upgrade import BY, upgrade_junctions

FIXTURES = os.path.join(os.path.dirname(__file__), 'test_inputs', 'legacy_junctions')

ENTRIES = [
    {'from': ['2in2out_junction', 'vv'], 'to': ['arterial', 'vv'], 'by': BY,
     'neighbour_variables': {'entrance': ['v_in_1', 'v_in_2'], 'exit': ['v_out_1', 'v_out_2']}},
    {'from': ['split_junction_simple', 'pv'], 'to': ['arterial_simple', 'pv'], 'by': BY,
     'neighbour_variables': {'exit': ['v_out_1', 'v_out_2']}},
    {'from': ['Min_junction', 'vp_simple'], 'to': ['arterial_simple', 'vp'], 'by': BY,
     'variables': {'v_in_sum': 'v_in'}},
    {'from': ['flow_merge', 'vp'], 'to': None, 'by': BY,
     'join': {'side': 'exit', 'variables': {'v_in': 0, 'v_out': 0, 'u': 1, 'u_d': 1}}},
    {'from': ['flow_sum_2', 'Safaei2016_v01'], 'to': None, 'by': BY, 'note': 'use a sum port'},
    {'from': ['inlet_flow', 'constant_2'], 'to': ['inlet_flow', 'constant'], 'by': BY,
     'split': [['v_1', 'v'], ['v_2', 'v']]},
    {'from': ['old_name', 'nn'], 'to': ['new_name', 'v01']},   # a plain rename: not ours
]


def _config(module_type, version, entrance, exit_, variables):
    def ports(vessel):
        return [{'port_type': 'vessel_port', 'variables': vessel}] if vessel else []
    return [{'module_type': module_type, 'module_subtype': version, 'entrance_ports': ports(entrance),
             'exit_ports': ports(exit_), 'variables_and_units': [[v, 'u', 'access', 'constant']
                                                                 for v in variables]}]


SIMPLE = ['R', 'C', 'I', 'q_0', 'u_0', 'u_ext']
CONFIGS = [
    _config('arterial', 'vv', ['v_in', 'u'], ['v_out', 'u_d'], ['E', 'l', 'r_0', 'u_0', 'theta', 'u_ext']),
    _config('arterial_simple', 'pv', ['v', 'u_in'], ['v_out', 'u'], SIMPLE),
    _config('arterial_simple', 'vp', ['v_in', 'u'], ['v', 'u_out'], SIMPLE),
    _config('arterial_simple', 'pp', ['v', 'u_in'], ['v_d', 'u_out'], SIMPLE),
    _config('inlet_flow', 'constant', None, ['v', 'P'], ['v']),
    _config('outlet_pressure', 'constant', ['v', 'P'], None, ['P']),
]


@pytest.fixture
def library(tmp_path):
    lib = tmp_path / 'modules'
    lib.mkdir()
    (lib / 'renames.json').write_text(json.dumps({'schema_version': 1, 'renames': ENTRIES}))
    for config in CONFIGS:
        name = f'{config[0]["module_type"]}_{config[0]["module_subtype"]}_modules_config.json'
        (lib / name).write_text(json.dumps(config))
    return str(lib)


@pytest.fixture
def model(tmp_path):
    shutil.copytree(os.path.join(FIXTURES, 'junction_net'), tmp_path / 'junction_net')
    return tmp_path / 'junction_net'


def _load(path):
    with open(path) as f:
        return json.load(f)


@pytest.mark.unit
def test_array_parameters_obs_data_and_params_for_id_are_converted(library, model):
    params = _load(model / 'junction_net_parameters.json')
    params['M']['K_tube_visco'] = {'value': 1.0, 'units': 'Js_per_m3', 'data_reference': 'test'}
    (model / 'junction_net_parameters.json').write_text(json.dumps(params))

    assert upgrade_junctions(str(model / 'junction_net_module_array.json'), [library])

    records = {r['name']: r for r in _load(model / 'junction_net_module_array.json')}
    assert (records['J']['module_type'], records['J']['module_subtype']) == ('arterial', 'vv')
    assert (records['s1']['module_type'], records['s1']['module_subtype']) == ('arterial_simple', 'pv')
    assert (records['M']['module_type'], records['M']['module_subtype']) == ('arterial_simple', 'vp')
    # the two-inflow source is two sources feeding what it fed
    assert 'inflow' not in records
    for name in ('inflow_1', 'inflow_2'):
        assert (records[name]['module_type'], records[name]['module_subtype']) == ('inlet_flow', 'constant')
        assert records[name]['out_instances'] == ['J']
    assert records['J']['inp_instances'] == ['inflow_1', 'inflow_2']
    # the flow_merge is gone and its neighbours are joined, in its place
    assert 'F' not in records
    assert records['p2']['out_instances'] == ['M'] and records['p3']['out_instances'] == ['M']
    assert records['M']['inp_instances'] == ['p2', 'p3', 'p4']

    params = _load(model / 'junction_net_parameters.json')
    assert 'inflow' not in params
    assert params['inflow_1'] == {'v': {'value': 4.0e-6, 'units': 'm3_per_s', 'data_reference': 'test value'}}
    assert params['inflow_2']['v']['value'] == 2.0e-6
    assert 'K_tube_visco' not in params['M'] and params['M']['R']['value'] == 1e7

    obs = _load(model / 'junction_net_obs_data.json')
    operands = [item['operands'][0] for item in obs['data_items']]
    # J's first outlet fed s1, whose own inflow is v; F's outflow is M's inflow
    assert operands == ['s1/v', 'M/v_in', 'M/v_in', 'J/u']
    assert list(obs['protocol_info']['params_to_change']) == ['inflow_1/v']

    targets = [p['targets'] for p in _load(model / 'junction_net_params_for_id.json')['params']]
    assert targets == [['M/R'], ['inflow_2/v']]


@pytest.mark.unit
def test_dry_run_writes_nothing(library, model):
    before = {p: (model / p).read_text() for p in os.listdir(model)}
    assert upgrade_junctions(str(model / 'junction_net_module_array.json'), [library], dry_run=True)
    assert {p: (model / p).read_text() for p in os.listdir(model)} == before


@pytest.mark.unit
def test_a_csv_array_keeps_its_columns(library, model):
    records = _load(model / 'junction_net_module_array.json')
    lines = ['name,module_type,module_subtype,inp_instances,out_instances']
    lines += [f'{r["name"]},{r["module_type"]},{r["module_subtype"]},{" ".join(r["inp_instances"])},'
              f'{" ".join(r["out_instances"])}' for r in records]
    path = model / 'junction_net_module_array.csv'
    path.write_text('\n'.join(lines) + '\n')
    (model / 'junction_net_module_array.json').unlink()
    upgrade_junctions(str(path), [library])
    text = path.read_text().splitlines()
    assert text[0] == 'name,module_type,module_subtype,inp_instances,out_instances'
    assert 'inflow_1,inlet_flow,constant,,J' in text
    assert 'M,arterial_simple,vp,p2 p3 p4,out' in text


@pytest.mark.unit
def test_a_model_without_junctions_is_left_alone(library, tmp_path):
    path = tmp_path / 'plain_module_array.json'
    path.write_text(json.dumps([{'name': 'a', 'module_type': 'arterial_simple', 'module_subtype': 'pp',
                                 'inp_instances': [], 'out_instances': []}]))
    before = path.read_text()
    assert upgrade_junctions(str(path), [library]) == []
    assert path.read_text() == before


@pytest.mark.unit
def test_a_junction_with_no_replacement_is_refused(library, tmp_path):
    path = tmp_path / 'sum_module_array.json'
    path.write_text(json.dumps([{'name': 'total', 'module_type': 'flow_sum_2',
                                 'module_subtype': 'Safaei2016_v01', 'inp_instances': [], 'out_instances': []}]))
    with pytest.raises(SystemExit, match='"total" is a flow_sum_2 Safaei2016_v01.*use a sum port'):
        upgrade_junctions(str(path), [library])


@pytest.mark.unit
def test_a_library_without_junction_entries_is_refused(tmp_path, model):
    with pytest.raises(SystemExit, match='No junction entries'):
        upgrade_junctions(str(model / 'junction_net_module_array.json'), [str(tmp_path)])


@pytest.mark.unit
def test_command_line(library, model, capsys):
    assert upgrade.main(['junctions', str(model / 'junction_net_module_array.json'),
                         '--module-library-dir', library]) == 0
    out = capsys.readouterr().out
    assert 'J: 2in2out_junction vv -> arterial vv' in out
    assert '1 of 1 file(s) changed.' in out


# ------------------------------------------------------------------------- the real library

def _generate(resources, library_dir, out_dir, external=None):
    from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
    config = {'file_prefix': 'junction_net', 'input_param_file': 'junction_net_parameters.json',
              'model_type': 'cellml', 'solver': 'CVODE_myokit', 'resources_dir': str(resources),
              'generated_models_dir': str(out_dir), 'DEBUG': False,
              'module_library_dirs': [library_dir], 'use_builtin_modules': False}
    if external:
        config['external_modules_dir'] = external
    assert generate_with_new_architecture(False, config)
    return os.path.join(str(out_dir), 'junction_net', 'junction_net.cellml')


def _states(path):
    from libcuflynx.solver_wrappers import get_simulation_helper
    helper = get_simulation_helper(model_path=path, solver='CVODE_myokit', model_type='cellml', dt=0.001,
                                   sim_time=0.5, solver_info={'rtol': 1e-10, 'atol': 1e-14}, pre_time=0.0)
    helper.run()
    names = [s.qname() for s in helper.model.states()]
    results = helper.get_results([[n.replace('.', '/', 1)] for n in names], flatten=True)
    return {n: np.asarray(r, dtype=float).ravel() for n, r in zip(names, results)}


@pytest.mark.integration
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_upgraded_model_is_the_same_model(model, tmp_path):
    from _module_library import modules_dir_or_skip
    library_dir = str(modules_dir_or_skip())
    if not os.path.exists(os.path.join(library_dir, 'renames.json')) or BY not in open(
            os.path.join(library_dir, 'renames.json')).read():
        pytest.skip('the module library has no junction entries in renames.json yet')
    original = _states(_generate(model, library_dir, tmp_path / 'original',
                                 external=os.path.join(FIXTURES, 'modules')))
    upgrade_junctions(str(model / 'junction_net_module_array.json'), [library_dir])
    upgraded = _states(_generate(model, library_dir, tmp_path / 'upgraded'))
    assert set(original) == set(upgraded)
    for name, values in original.items():
        scale = max(np.max(np.abs(values)), np.ptp(values), 1e-300)
        assert np.max(np.abs(values - upgraded[name])) / scale < 1e-8, name
