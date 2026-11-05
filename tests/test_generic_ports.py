"""A module's ports are connected the same way for every module, the heart included: its
inputs take its entrance ports of one port_type in the order its inp_instances lists them,
its outputs its exit ports in out_instances order (generators/port_order.py). Nothing is
special-cased by module type or instance name."""
import os
import re
import shutil

import pytest

from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
from _parameters import read_parameters, write_parameters

RESOURCES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resources')


def _connections(cellml_text):
    """{(component_1, component_2): [(variable_1, variable_2), ...]} of a generated model."""
    out = {}
    for block in re.findall(r'<connection>(.*?)</connection>', cellml_text, re.S):
        comps = re.search(r'component_1="([^"]+)"\s+component_2="([^"]+)"', block).groups()
        out.setdefault(comps, []).extend(
            re.findall(r'variable_1="([^"]+)"\s+variable_2="([^"]+)"', block))
    return out


def _mapped(connections, comp_a, var_a, comp_b, var_b):
    return ((var_a, var_b) in connections.get((comp_a, comp_b), []) or
            (var_b, var_a) in connections.get((comp_b, comp_a), []))


def _generate(tmp_path, prefix, array_text, parameter_rows):
    work = tmp_path / 'resources'
    work.mkdir(exist_ok=True)
    (work / f'{prefix}_module_array.csv').write_text(array_text)
    write_parameters(work / f'{prefix}_parameters.json', parameter_rows)
    config = {'file_prefix': prefix, 'input_param_file': f'{prefix}_parameters.json',
              'model_type': 'cellml', 'solver': 'CVODE_myokit', 'resources_dir': str(work),
              'generated_models_dir': str(tmp_path / 'generated'), 'DEBUG': False}
    ok = generate_with_new_architecture(False, config)
    text = (tmp_path / 'generated' / prefix / f'{prefix}.cellml').read_text()
    return ok, text


def _read(name):
    with open(os.path.join(RESOURCES, name)) as f:
        return f.read()


@pytest.mark.unit
def test_neighbour_port_index_follows_the_neighbour_list():
    from libcuflynx.generators.port_order import neighbour_port_index
    ports = [{'port_type': 'vessel_port', 'variables': ['v_a', 'u']},
             {'port_type': 'other', 'variables': ['x']},
             {'port_type': 'vessel_port', 'variables': ['v_b', 'u']},
             {'port_type': 'vessel_port', 'variables': ['v_c', 'u'], 'multi_port': 'True'}]
    types = {'p': ['vessel_port'], 'q': ['vessel_port'], 'ctrl': ['other']}
    index = lambda n: neighbour_port_index(ports, ['ctrl', 'q', 'p'], n, 'vessel_port',
                                           types.get)
    # ctrl has no vessel_port, so q is the first vessel input and p the second
    assert (index('q'), index('p')) == (0, 2)
    assert index('ctrl') is None
    # one single port of a type: the caller's default decides
    assert neighbour_port_index(ports, ['ctrl'], 'ctrl', 'other', types.get) is None


@pytest.mark.integration
def test_inputs_take_same_type_entrance_ports_in_inp_instances_order(tmp_path):
    """The rows of a merge junction's inputs do not decide which entrance port each takes:
    b's row comes first, but a is listed first, so a feeds v_in_1."""
    array = ('name,module_subtype,module_type,inp_instances,out_instances\n'
             'src_b,nn_constant,inlet_flow,,b\n'
             'b,vp,arterial,src_b,m\n'
             'src_a,nn_constant,inlet_flow,,a\n'
             'a,vp,arterial,src_a,m\n'
             'm,vp,merge_junction,a b,sink\n'
             'sink,nn_constant,outlet_pressure,m,\n')
    _, text = _generate(tmp_path, 'merge_order', array, [])
    conns = _connections(text)
    assert _mapped(conns, 'a_module', 'v', 'm_module', 'v_in_1')
    assert _mapped(conns, 'b_module', 'v', 'm_module', 'v_in_2')


@pytest.mark.integration
def test_a_heart_connects_by_its_ports_and_an_explicit_zero_flow(tmp_path):
    """3compartment's heart lists zero_flow, venous_svc and pvn, which take its three
    vessel_port entrances (v_ivc, v_svc, v_pvn) in that order, although pvn's row is first
    and zero_flow's last."""
    ok, text = _generate(tmp_path, '3compartment', _read('3compartment_module_array.csv'),
                         read_parameters(os.path.join(RESOURCES, '3compartment_parameters.json')))
    assert ok
    conns = _connections(text)
    assert _mapped(conns, 'zero_flow_module', 'v_zero', 'heart_module', 'v_ivc')
    assert _mapped(conns, 'venous_svc_module', 'v', 'heart_module', 'v_svc')
    assert _mapped(conns, 'pvn_module', 'v', 'heart_module', 'v_pvn')
    assert _mapped(conns, 'heart_module', 'u_root', 'aortic_root_module', 'u')
    assert _mapped(conns, 'heart_module', 'u_par', 'par_module', 'u')


@pytest.mark.integration
def test_a_heart_without_a_listed_zero_flow_gets_none(tmp_path, capsys):
    """Nothing adds a zero flow for a heart with two vessel inputs: they take its first two
    vessel_port entrances, and the third (v_pvn) is an unconnected port whose boundary
    condition the missing-parameters report names."""
    array = _read('3compartment_module_array.csv')
    array = '\n'.join(line for line in array.splitlines() if not line.startswith('zero_flow,'))
    array = array.replace('zero_flow venous_svc pvn,', 'venous_svc pvn,') + '\n'
    capsys.readouterr()
    ok, text = _generate(tmp_path, '3compartment', array,
                         read_parameters(os.path.join(RESOURCES, '3compartment_parameters.json')))
    assert not ok
    assert 'heart/v_pvn: a boundary condition of "heart" that no connected module' in \
        capsys.readouterr().out
    assert 'zero_flow' not in text
    conns = _connections(text)
    assert _mapped(conns, 'venous_svc_module', 'v', 'heart_module', 'v_ivc')
    assert _mapped(conns, 'pvn_module', 'v', 'heart_module', 'v_svc')


@pytest.mark.integration
def test_two_hearts_generate(tmp_path):
    """Two closed loops, each with its own heart, in one model."""
    array = _read('3compartment_module_array.csv').splitlines()
    parameters = read_parameters(os.path.join(RESOURCES, '3compartment_parameters.json'))
    names = [line.split(',')[0].strip() for line in array[1:] if line.strip()]
    loop = [n for n in names if n != 'volume_sum']

    def renamed(line):
        cells = line.split(',')
        for i in (0, 3, 4):
            words = [w + '_b' if w in loop else w for w in cells[i].split()
                     if w != 'volume_sum']
            cells[i] = ' '.join(words)
        return ','.join(cells)

    second = [renamed(line) for line in array[1:]
              if line.strip() and not line.startswith('volume_sum')]
    second_parameters = [dict(row, variable_name=row['variable_name'].replace('/', '_b/', 1))
                         for row in parameters if row['variable_name'].split('/')[0] in loop]
    ok, text = _generate(tmp_path, 'two_hearts', '\n'.join(array + second) + '\n',
                         parameters + second_parameters)
    assert ok
    conns = _connections(text)
    for suffix in ('', '_b'):
        heart = f'heart{suffix}_module'
        assert _mapped(conns, f'zero_flow{suffix}_module', 'v_zero', heart, 'v_ivc')
        assert _mapped(conns, f'venous_svc{suffix}_module', 'v', heart, 'v_svc')
        assert _mapped(conns, f'pvn{suffix}_module', 'v', heart, 'v_pvn')


@pytest.mark.unit
def test_1d_coupling_rows_come_from_the_ports_not_the_names(tmp_path):
    """convert_0d_to_1d writes the 0D neighbour's coupling variable of the port joining it
    to each 1D vessel, whatever the vessels are called."""
    from libcuflynx.scripts.convert_0d_to_1d import convert_0d_to_1d
    from libcuflynx.utilities.config_schemas import dump_vessel_records

    def rec(name, module_type, subtype, inp, out):
        return {'name': name, 'module_type': module_type, 'module_subtype': subtype,
                'inp_instances': inp, 'out_instances': out}
    records = [rec('cava_lower', 'venous', 'vp', ['loop'], ['h']),
               rec('cava_upper', 'venous', 'vp', ['loop'], ['h']),
               rec('lung_vein', 'venous', 'vp', ['lung_artery'], ['h']),
               rec('h', 'heart', 'vp', ['cava_lower', 'cava_upper', 'lung_vein'],
                   ['outflow', 'lung_artery']),
               rec('outflow', 'arterial', 'vp', ['h'], ['loop']),
               rec('lung_artery', 'arterial', 'vp', ['h'], ['lung_vein']),
               rec('loop', 'arterial', 'vp', ['outflow'], ['cava_lower', 'cava_upper'])]
    (tmp_path / 'm_0d_module_array.json').write_text(dump_vessel_records(records))
    write_parameters(tmp_path / 'm_0d_parameters.json', [])
    convert_0d_to_1d('m', str(tmp_path), 'm_0d_parameters.json',
                     vess_1d_list=['outflow', 'lung_artery', 'lung_vein', 'cava_lower'])
    rows = read_parameters(tmp_path / 'm_hybrid_parameters.json')
    h_rows = {row['variable_name']: row['units'] for row in rows}
    assert {k: v for k, v in h_rows.items() if k.startswith('h/')} == {
        'h/u_root': 'J_per_m3', 'h/u_par': 'J_per_m3',
        'h/v_pvn': 'm3_per_s', 'h/v_ivc': 'm3_per_s'}


@pytest.mark.integration
@pytest.mark.slow
def test_a_1d_vessel_couples_to_the_heart_port_it_is_connected_to(tmp_path):
    """The heart's 1D outflow vessel is coupled through the heart's first vessel_port exit
    (u_root) whatever it is called; it used to need "aorta" in its name."""
    work = tmp_path / 'resources'
    work.mkdir()
    for suffix in ('_module_array.csv', '_parameters.json'):
        text = _read('cvs_model_with_arm_hybrid' + suffix)
        (work / ('cvs_model_with_arm_hybrid' + suffix)).write_text(
            re.sub(r'\bA_aorta_ascending_1\b', 'outflow_1d', text))
    config = {'file_prefix': 'cvs_model_with_arm_hybrid',
              'input_param_file': 'cvs_model_with_arm_hybrid_parameters.json',
              'model_type': 'cpp', 'solver': 'RK4', 'solver_info': {'dt_solver': 0.0001},
              'couple_to_1d': True, 'resources_dir': str(work),
              'generated_models_dir': str(tmp_path / 'generated'),
              'cpp_generated_models_dir': str(tmp_path / 'cpp'), 'cpp_1d_model_config_path': None,
              'DEBUG': False}
    assert generate_with_new_architecture(False, config)
    model0d = (tmp_path / 'cpp' / 'model0d_core.c').read_text()
    assert re.search(r'\{"u_root_heart", "J_per_m3", "parameters", EXTERNAL\}', model0d)


@pytest.mark.unit
def test_the_inertance_diagnostic_reads_the_stiff_states_own_inertance():
    """No vessel is looked up by name: each stiff mode's dominant state <module>/<state> is
    scaled through its module's I_<module>, when it has one."""
    from types import SimpleNamespace
    from libcuflynx.utilities.casadi_solver_diagnostics import \
        _required_inertance_from_stiff_modes
    sim = SimpleNamespace(dt=0.01, solver_info={'max_num_steps': 100},
                          var_name_to_idx={'I_lung_artery': 0}, variables=[2.0e-6])
    target = 100 * 5.0 / 0.01
    modes = [{'dominant_state': 'lung_artery/v', 'eigenvalue_real': -4 * target},
             {'dominant_state': 'chamber/q', 'eigenvalue_real': -target}]
    info = _required_inertance_from_stiff_modes(sim, modes, None)
    assert info['lambda_target_s_inv'] == pytest.approx(target)
    first, second = info['modes']
    assert first['inertance'] == 'I_lung_artery'
    assert first['I_scale_up_factor'] == pytest.approx(4.0)
    assert first['I_needed_Js2_per_m6'] == pytest.approx(8.0e-6)
    assert second['inertance'] is None and 'I_needed_Js2_per_m6' not in second
