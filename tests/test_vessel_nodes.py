"""The node rule (#616): at a node of three or more vessel_port ends, the end alone on its side
must be compliant, and an n-to-m node is refused (generators/vessel_nodes.py)."""
import os

import numpy as np
import pandas as pd
import pytest

from libcuflynx.generators.vessel_nodes import check_vessel_nodes, find_vessel_nodes
from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
from libcuflynx.solver_wrappers import get_simulation_helper

SUM = ['sum', 'True']


def _port(variables, multi_port=None, port_type='vessel_port'):
    port = {'port_type': port_type, 'variables': variables}
    if multi_port is not None:
        port['multi_port'] = multi_port
    return port


def _vessel(name, subtype, inp=(), out=(), module_type='arterial'):
    """A vessel whose ``v`` ends sum (as a compliant end does) and whose ``p`` ends do not."""
    return {'name': name, 'module_type': module_type, 'module_subtype': subtype,
            'inp_instances': list(inp), 'out_instances': list(out),
            'entrance_ports': [_port(['v_in', 'u'], SUM if subtype[0] == 'v' else None),
                               _port(['u_ext'], port_type='external_pressure_port')],
            'exit_ports': [_port(['v_out', 'u_d'], SUM if subtype[1] == 'v' else None),
                           _port(['K'], port_type='stiffness_port')],
            'general_ports': []}


def _df(*rows):
    return pd.DataFrame(list(rows))


def test_refuses_a_mixed_split_naming_the_node_and_both_fixes():
    # the design page's example: a's outlet (p) is alone on its side; m's inlet is compliant
    df = _df(_vessel('a', 'vp', out=['m', 'c']),
             _vessel('m', 'vp', inp=['a']),
             _vessel('c', 'pp', inp=['a']))
    with pytest.raises(ValueError) as err:
        check_vessel_nodes(df)
    message = str(err.value)
    for part in ['the outlet of "a" (arterial vp)', 'the inlet of "m" (arterial vp)',
                 'the inlet of "c" (arterial pp)', 'is the only end on its side',
                 '"a" vp -> vv, "m" vp -> pp', 'add a lumped vessel (vv)', '#616']:
        assert part in message


def test_refuses_a_mixed_merge():
    df = _df(_vessel('x', 'vv', out=['z']),
             _vessel('y', 'pv', out=['z']),
             _vessel('z', 'pp', inp=['x', 'y']))
    with pytest.raises(ValueError, match=r'the inlet of "z" \(arterial pp\) is the only end'):
        check_vessel_nodes(df)


def test_refuses_an_n_to_m_node():
    df = _df(_vessel('p', 'vp', out=['r', 's']),
             _vessel('q', 'vp', out=['r', 's']),
             _vessel('r', 'vp', inp=['p', 'q']),
             _vessel('s', 'vp', inp=['p', 'q']))
    with pytest.raises(ValueError) as err:
        check_vessel_nodes(df)
    message = str(err.value)
    assert '2 outlets meet 2 inlets at one node' in message
    assert '2in2out_junction (vv)' in message and 'lumped vessel (vv)' in message


def test_accepts_a_split_and_a_merge_that_follow_the_rule():
    df = _df(_vessel('a', 'vv', out=['m', 'c']),
             _vessel('m', 'pp', inp=['a'], out=['z']),
             _vessel('c', 'pp', inp=['a'], out=['z']),
             _vessel('z', 'vp', inp=['m', 'c']))
    check_vessel_nodes(df)
    sizes = sorted(len(ends) for ends in find_vessel_nodes({r['name']: r for _, r in df.iterrows()}))
    assert sizes == [3, 3]


def _non_vessel(name, entrance=(), exit=(), inp=(), out=()):
    return {'name': name, 'module_type': 'chamber', 'module_subtype': 'Liang2009',
            'inp_instances': list(inp), 'out_instances': list(out),
            'entrance_ports': list(entrance), 'exit_ports': list(exit), 'general_ports': []}


# ["sum", "sum"] is a whole-port "sum" as the module config loader normalises it
@pytest.mark.parametrize('multi_port', [SUM, ['Sum', 'True'], ['sum', 'sum']])
def test_a_non_vessels_summing_port_owns_the_node(multi_port):
    # a heart chamber's [v_in, u] port sums the inflows and shares its pressure
    chamber = _non_vessel('la', entrance=[_port(['v_in', 'u'], multi_port)], inp=['pv1', 'pv2'])
    check_vessel_nodes(_df(_vessel('pv1', 'pp', out=['la']), _vessel('pv2', 'pp', out=['la']),
                           chamber))


@pytest.mark.parametrize('multi_port', [None, 'True', ['True', 'True']])
def test_a_non_vessels_port_without_a_sum_is_refused(multi_port):
    chamber = _non_vessel('la', entrance=[_port(['v_in', 'u'], multi_port)], inp=['pv1', 'pv2'])
    with pytest.raises(ValueError) as err:
        check_vessel_nodes(_df(_vessel('pv1', 'pp', out=['la']), _vessel('pv2', 'pp', out=['la']),
                               chamber))
    message = str(err.value)
    assert 'the inlet of "la" (chamber Liang2009) is the only end on its side' in message
    assert 'a "sum" entry' in message


def test_a_multiply_port_feeding_two_readers_is_refused():
    # a Multiply port broadcasts one scaled value and sums nothing, so it cannot own a node
    gain = _non_vessel('g', exit=[_port(['x'], 'Multiply')], out=['rd_a', 'rd_b'])
    gain['exit_ports'][0]['multiply_factor'] = 3
    readers = [_non_vessel(n, entrance=[_port(['y'])], inp=['g']) for n in ['rd_a', 'rd_b']]
    with pytest.raises(ValueError) as err:
        check_vessel_nodes(_df(gain, *readers))
    message = str(err.value)
    assert 'the outlet of "g" (chamber Liang2009) is the only end on its side' in message
    assert 'a "sum" entry' in message


def test_skips_a_node_with_an_fv1d_end():
    df = _df(_vessel('a', 'vp', out=['m', 'c']),
             _vessel('m', 'vp', inp=['a'], module_type='FV1D_vessel'),
             _vessel('c', 'pp', inp=['a']))
    check_vessel_nodes(df)


def test_two_ended_nodes_are_not_checked():
    # a p->p pair is the BC-letter check's business, not the node rule's
    check_vessel_nodes(_df(_vessel('a', 'pp', out=['b']), _vessel('b', 'pp', inp=['a'])))


def test_ports_are_taken_in_order():
    # a module with one port per daughter (split_junction-like) makes a two-ended node of each
    split = _vessel('s', 'pp', out=['m', 'c'])
    split['exit_ports'] = [_port(['v_out_1', 'u']), _port(['v_out_2', 'u'])]
    df = _df(split, _vessel('m', 'pp', inp=['s']), _vessel('c', 'pp', inp=['s']))
    check_vessel_nodes(df)
    assert [len(e) for e in find_vessel_nodes({r['name']: r for _, r in df.iterrows()})] == [2, 2]


def _generate(tmp_path, prefix, resources_dir):
    config = {'file_prefix': prefix,
              'input_param_file': prefix.replace('test2_', 'test_')
              + '_parameters.json',
              'model_type': 'cellml', 'solver': 'CVODE_myokit', 'DEBUG': False,
              'resources_dir': resources_dir,
              'generated_models_dir': str(tmp_path / 'generated_models')}
    assert generate_with_new_architecture(False, config)
    return tmp_path / 'generated_models' / prefix / f'{prefix}.cellml'


@pytest.mark.integration
def test_generation_refuses_the_mixed_test2_array(tmp_path, resources_dir):
    with pytest.raises(ValueError, match=r'the outlet of "aortic_arch_II" \(arterial '
                                         r'pp_nonlinear_visco\) is the only end on its side'):
        _generate(tmp_path, 'generic_junction_test2_open_loop', resources_dir)


@pytest.mark.integration
@pytest.mark.slow
def test_rule_following_version_simulates_with_conserved_node_flow(tmp_path, resources_dir):
    """generic_junction_test_open_loop is the rule-following version of test2's network: each
    split's compliance is at the parent's outlet, so the parent's outflow is the sum of its
    daughters' inflows."""
    model = _generate(tmp_path, 'generic_junction_test_open_loop', resources_dir)
    node_sum = 'generic_junction_connection/v_aortic_arch_II_sum_Nout'
    daughters = ['common_carotid_L/v', 'aortic_arch_III/v']
    helper = get_simulation_helper(model_path=str(model), solver='CVODE_myokit',
                                   model_type='cellml', dt=0.01, sim_time=0.2, pre_time=0.0)
    helper.run()
    results = helper.get_results([[v] for v in [node_sum] + daughters], flatten=True)
    total, *parts = [np.asarray(r).ravel() for r in results]
    assert np.all(np.isfinite(total)) and np.any(total != 0)
    np.testing.assert_allclose(total, parts[0] + parts[1], rtol=1e-10, atol=1e-15)
