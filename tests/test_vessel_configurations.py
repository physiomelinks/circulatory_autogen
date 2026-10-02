'''
Vessels connected in any configuration, through ports that sum over a node.

Each topology is built from circulatory-autogen-modules' ordinary vessels (``arterial_simple`` in
its four BC types, with flow and pressure sources and pressure sinks), generated as CellML and
simulated. Where vessels meet, the one compliant end (a vessel_port with
``"multi_port": ["sum", "True"]``) owns the node (libcuflynx.generators.port_nodes). The tests check:

* the CellML: a node of three or more ends gets one ``multiport_sum_<owner>_<variable>``
  component, a node of two ends none;
* the physics: at every node the owner's flow is the signed sum of the other ends' flows, and
  every end sees the owner's pressure;
* the conversion: where a junction module type expressed the same topology (the fixed versions in
  tests/test_inputs/legacy_junctions, which still go through the generator's junction handling),
  the results are the same.

Errors are tested too: two compliant ends at one node, none at a node of three or more, and a 1D
vessel in a node of three or more.
'''
import contextlib
import io
import json
import os

import numpy as np
import pandas as pd
import pytest

from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
from libcuflynx.solver_wrappers import get_simulation_helper

pytestmark = pytest.mark.integration

LEGACY_JUNCTIONS = os.path.join(os.path.dirname(__file__), 'test_inputs', 'legacy_junctions')
LIBRARY = os.environ['CUFLYNX_MODULE_LIBRARY'].split(os.pathsep)[0]

# the ordinary vessel each legacy junction version is the same as
ORDINARY = {('Min_junction', 'vp_simple'): ('arterial_simple', 'vp'),
            ('Min_junction', 'vv_simple'): ('arterial_simple', 'vv'),
            ('Nout_junction', 'pv_simple'): ('arterial_simple', 'pv'),
            ('Nout_junction', 'vv_simple'): ('arterial_simple', 'vv'),
            ('MinNout_junction', 'vv_simple'): ('arterial_simple', 'vv'),
            ('split_junction_simple', 'pv'): ('arterial_simple', 'pv'),
            ('merge_junction_simple', 'vp'): ('arterial_simple', 'vp')}
LEGACY_RENAMES = {'v_in_sum': 'v_in', 'v_out_sum': 'v_out'}

VESSEL = {'R': ('Js_per_m6', 1.0e7), 'C': ('m6_per_J', 1.0e-9), 'I': ('Js2_per_m6', 1.0e5),
          'q_0': ('m3', 0.0), 'u_0': ('J_per_m3', 0.0), 'u_ext': ('J_per_m3', 0.0)}
SIM_TIME, DT = 0.5, 1e-3
SOLVER_INFO = {'rtol': 1e-10, 'atol': 1e-14}


def vessel(name, bc, inputs, outputs, module_type='arterial_simple'):
    return {'name': name, 'module_type': module_type, 'module_subtype': bc,
            'inp_instances': list(inputs), 'out_instances': list(outputs)}


def flow_source(name, outputs):
    return {'name': name, 'module_type': 'inlet_flow', 'module_subtype': 'nn_constant',
            'inp_instances': [], 'out_instances': list(outputs)}


def pressure_source(name, outputs):
    return {'name': name, 'module_type': 'inlet_pressure', 'module_subtype': 'nn_constant',
            'inp_instances': [], 'out_instances': list(outputs)}


def sink(name, inputs):
    return {'name': name, 'module_type': 'outlet_pressure', 'module_subtype': 'nn_constant',
            'inp_instances': list(inputs), 'out_instances': []}


def parameters(records, overrides=None):
    rows = []
    for r in records:
        name = r['name']
        if r['module_type'] == 'inlet_flow':
            rows.append((f'v_{name}', 'm3_per_s', 1.0e-5))
        elif r['module_type'] in ('inlet_pressure', 'outlet_pressure'):
            rows.append((f'P_{name}', 'J_per_m3', 2000.0 if r['module_type'] == 'inlet_pressure' else 1000.0))
        else:
            rows.extend((f'{var}_{name}', units, value) for var, (units, value) in VESSEL.items())
    values = {row[0]: row for row in rows}
    for key, value in (overrides or {}).items():
        values[key] = (key, values[key][1], value)
    return pd.DataFrame([{'variable_name': n, 'units': u, 'value': v, 'data_reference': 'test'}
                         for n, u, v in values.values()])


def legacy(records, junctions):
    '''``records`` with the vessels named in ``junctions`` ({name: (module_type, version)}) made
    legacy junctions; each must be the junction whose ordinary twin the record already is.'''
    out = []
    for r in records:
        if r['name'] in junctions:
            junction = junctions[r['name']]
            assert ORDINARY[junction] == (r['module_type'], r['module_subtype']), (r, junction)
            r = dict(r, module_type=junction[0], module_subtype=junction[1])
        out.append(r)
    return out


def build(tmp_path, tag, records, params, with_legacy=False):
    '''Generate the model; returns the path of its CellML.'''
    resources = tmp_path / tag
    resources.mkdir()
    with open(resources / f'{tag}_vessel_array.json', 'w') as f:
        json.dump(records, f)
    params.to_csv(resources / f'{tag}_parameters.csv', index=False)
    config = {'file_prefix': tag, 'input_param_file': f'{tag}_parameters.csv', 'model_type': 'cellml',
              'solver': 'CVODE_myokit', 'resources_dir': str(resources),
              'generated_models_dir': str(tmp_path / 'generated'), 'DEBUG': False,
              'use_builtin_modules': False,
              'module_library_dirs': [LIBRARY] + ([LEGACY_JUNCTIONS] if with_legacy else [])}
    with contextlib.redirect_stdout(io.StringIO()):
        assert generate_with_new_architecture(False, config), f'generation of {tag} failed'
    return str(tmp_path / 'generated' / tag / f'{tag}.cellml')


def simulate(cellml_path, names=None):
    """Simulate with Myokit; returns {qualified name: values} for ``names`` ('component.variable'),
    or for every variable the model keeps. (Connected inputs are merged into their source on
    import, so only sources and outputs have names of their own.)"""
    import myokit
    with contextlib.redirect_stdout(io.StringIO()):
        model = get_simulation_helper(model_path=cellml_path, solver='CVODE_myokit', model_type='cellml',
                                      dt=DT, sim_time=SIM_TIME, solver_info=SOLVER_INFO, pre_time=0.0).model
    available = [v.qname() for v in model.variables(deep=True) if v.is_state() or not v.is_constant()]
    names = available if names is None else names
    sim = myokit.Simulation(model)
    sim.set_tolerance(SOLVER_INFO['atol'], SOLVER_INFO['rtol'])
    log = sim.run(SIM_TIME, log=[n for n in names if n in available], log_interval=DT)
    missing = [n for n in names if n not in available]
    assert not missing, f'not in the model: {missing}'
    return {n: np.asarray(log[n], dtype=float) for n in names}


def connections(cellml_path):
    """Every mapped variable pair of the CellML, as frozensets of 'component.variable'."""
    import re
    with open(cellml_path) as f:
        text = f.read()
    pairs = set()
    for block in re.findall(r'<connection>(.*?)</connection>', text, re.S):
        c1, c2 = re.search(r'component_1="([^"]+)"\s+component_2="([^"]+)"', block).groups()
        for v1, v2 in re.findall(r'variable_1="([^"]+)"\s+variable_2="([^"]+)"', block):
            pairs.add(frozenset((f'{c1}.{v1}', f'{c2}.{v2}')))
    return pairs


def qualified(name):
    """'vessel/var' -> 'vessel_module.var'; a 'multiport_sum_...' name is a component already."""
    component, variable = name.split('/')
    return f'{component}.{variable}' if component.startswith('multiport_sum_') else f'{component}_module.{variable}'


def sum_components(cellml_path):
    import re
    with open(cellml_path) as f:
        text = f.read()
    return sorted(set(re.findall(r'<component name="(multiport_sum_[^"]+)"', text)))


# Each topology: records, the expected sum components, the node checks
# [(owner flow, [(sign, other flow)], owner pressure, [other pressures])], the legacy junctions
# that express it (or None), and parameter overrides.
TOPOLOGIES = {
    # one compliant outlet feeding two vessels (Nout)
    'split': dict(
        records=[flow_source('src', ['a']), vessel('a', 'vv', ['src'], ['b', 'c']),
                 vessel('b', 'pp', ['a'], ['sink_b']), vessel('c', 'pp', ['a'], ['sink_c']),
                 sink('sink_b', ['b']), sink('sink_c', ['c'])],
        sums=['multiport_sum_a_v_out'],
        nodes=[('a/v_out', [(1, 'b/v'), (1, 'c/v')], 'a/u_d', ['b/u_in', 'c/u_in'])],
        legacy={'a': ('Nout_junction', 'vv_simple')}),
    # two vessels feeding one compliant inlet (Min)
    'merge': dict(
        records=[flow_source('src1', ['p1']), flow_source('src2', ['p2']),
                 vessel('p1', 'vp', ['src1'], ['m']), vessel('p2', 'vp', ['src2'], ['m']),
                 vessel('m', 'vv', ['p1', 'p2'], ['d']), vessel('d', 'pp', ['m'], ['sink']), sink('sink', ['d'])],
        sums=['multiport_sum_m_v_in'],
        nodes=[('m/v_in', [(1, 'p1/v'), (1, 'p2/v')], 'm/u', ['p1/u_out', 'p2/u_out'])],
        legacy={'m': ('Min_junction', 'vv_simple')}),
    # merge and split through one vessel (MinNout)
    'merge_and_split': dict(
        records=[flow_source('src1', ['p1']), flow_source('src2', ['p2']),
                 vessel('p1', 'vp', ['src1'], ['m']), vessel('p2', 'vp', ['src2'], ['m']),
                 vessel('m', 'vv', ['p1', 'p2'], ['b', 'c']), vessel('b', 'pp', ['m'], ['sink_b']),
                 vessel('c', 'pp', ['m'], ['sink_c']), sink('sink_b', ['b']), sink('sink_c', ['c'])],
        sums=['multiport_sum_m_v_in', 'multiport_sum_m_v_out'],
        nodes=[('m/v_in', [(1, 'p1/v'), (1, 'p2/v')], 'm/u', ['p1/u_out', 'p2/u_out']),
               ('m/v_out', [(1, 'b/v'), (1, 'c/v')], 'm/u_d', ['b/u_in', 'c/u_in'])],
        legacy={'m': ('MinNout_junction', 'vv_simple')}),
    # inflow and another outflow sharing one compliant inlet: the sibling c is subtracted, and
    # takes its pressure from m, a vessel it is not connected to in the array
    'mixed_owner_downstream': dict(
        records=[flow_source('src', ['a']), vessel('a', 'vp', ['src'], ['m', 'c']),
                 vessel('m', 'vp', ['a'], ['sink_m']), vessel('c', 'pp', ['a'], ['sink_c']),
                 sink('sink_m', ['m']), sink('sink_c', ['c'])],
        sums=['multiport_sum_m_v_in'],
        nodes=[('m/v_in', [(1, 'a/v'), (-1, 'c/v')], 'm/u', ['a/u_out', 'c/u_in'])],
        legacy={'m': ('Min_junction', 'vp_simple')}),
    # two inflows (one of them the owner) and two outflows at one compliant outlet
    'mixed_owner_upstream': dict(
        records=[flow_source('src1', ['a']), flow_source('src2', ['e']),
                 vessel('a', 'vv', ['src1'], ['b', 'c']), vessel('e', 'vp', ['src2'], ['b']),
                 vessel('b', 'pp', ['a', 'e'], ['sink_b']), vessel('c', 'pp', ['a'], ['sink_c']),
                 sink('sink_b', ['b']), sink('sink_c', ['c'])],
        sums=['multiport_sum_a_v_out'],
        nodes=[('a/v_out', [(1, 'b/v'), (1, 'c/v'), (-1, 'e/v')], 'a/u_d', ['b/u_in', 'c/u_in', 'e/u_out'])],
        legacy={'a': ('Nout_junction', 'vv_simple')}),
    # the fixed-arity split: a pv vessel with one outlet port instead of two
    'split_from_pv': dict(
        records=[flow_source('src', ['x']), vessel('x', 'vv', ['src'], ['a']), vessel('a', 'pv', ['x'], ['b', 'c']),
                 vessel('b', 'pp', ['a'], ['sink_b']), vessel('c', 'pp', ['a'], ['sink_c']),
                 sink('sink_b', ['b']), sink('sink_c', ['c'])],
        sums=['multiport_sum_a_v_out'],
        nodes=[('a/v_out', [(1, 'b/v'), (1, 'c/v')], 'a/u', ['b/u_in', 'c/u_in'])],
        legacy={'a': ('split_junction_simple', 'pv')}),
    # the fixed-arity merge: a vp vessel with one inlet port instead of two
    'merge_into_vp': dict(
        records=[flow_source('src1', ['p1']), flow_source('src2', ['p2']),
                 vessel('p1', 'vp', ['src1'], ['m']), vessel('p2', 'vp', ['src2'], ['m']),
                 vessel('m', 'vp', ['p1', 'p2'], ['sink']), sink('sink', ['m'])],
        sums=['multiport_sum_m_v_in'],
        nodes=[('m/v_in', [(1, 'p1/v'), (1, 'p2/v')], 'm/u', ['p1/u_out', 'p2/u_out'])],
        legacy={'m': ('merge_junction_simple', 'vp')}),
    # boundary conditions at nodes: a pressure source feeding two vessels, and two vessels
    # draining into one pressure sink (the sink owns that node)
    'boundary_condition_nodes': dict(
        records=[pressure_source('src', ['b', 'c']), vessel('b', 'pp', ['src'], ['sink']),
                 vessel('c', 'pp', ['src'], ['sink']), sink('sink', ['b', 'c'])],
        sums=['multiport_sum_sink_v', 'multiport_sum_src_v'],
        nodes=[('src/v', [(1, 'b/v'), (1, 'c/v')], 'src/P', ['b/u_in', 'c/u_in']),
               ('sink/v', [(1, 'b/v_d'), (1, 'c/v_d')], 'sink/P', ['b/u_out', 'c/u_out'])],
        legacy=None),
    # a closed loop through a split and a merge: the volume is conserved
    'closed_loop': dict(
        records=[vessel('a', 'vv', ['e'], ['b', 'c']), vessel('b', 'pp', ['a'], ['d']), vessel('c', 'pp', ['a'], ['d']),
                 vessel('d', 'vv', ['b', 'c'], ['e']), vessel('e', 'pp', ['d'], ['a'])],
        sums=['multiport_sum_a_v_out', 'multiport_sum_d_v_in'],
        nodes=[('a/v_out', [(1, 'b/v'), (1, 'c/v')], 'a/u_d', ['b/u_in', 'c/u_in']),
               ('d/v_in', [(1, 'b/v_d'), (1, 'c/v_d')], 'd/u', ['b/u_out', 'c/u_out'])],
        legacy={'a': ('Nout_junction', 'vv_simple'), 'd': ('Min_junction', 'vv_simple')},
        overrides={'u_0_a': 3000.0, 'u_0_d': 500.0}),
}


@pytest.mark.parametrize('topology', list(TOPOLOGIES))
def test_configuration(tmp_path, topology):
    spec = TOPOLOGIES[topology]
    params = parameters(spec['records'], spec.get('overrides'))
    path = build(tmp_path, topology, spec['records'], params)

    assert sum_components(path) == sorted(spec['sums'])

    pairs = connections(path)
    owner_flows = {owner: 'multiport_sum_{0}_{1}.{1}'.format(*owner.split('/')) for owner, _, _, _ in spec['nodes']}
    names = sorted(set(owner_flows.values()) | {qualified(o) for _, others, _, _ in spec['nodes'] for _, o in others})
    if topology == 'closed_loop':
        names += [f'{r["name"]}_module.q' for r in spec['records']]
    results = simulate(path, names)
    for owner, others, pressure, pressures in spec['nodes']:
        flow = results[owner_flows[owner]]
        expected = sum(sign * results[qualified(o)] for sign, o in others)
        scale = max(np.max(np.abs(expected)), np.max(np.abs(flow)), 1e-30)
        assert np.max(np.abs(flow - expected)) <= 1e-9 * scale, \
            f'{topology}: {owner} is not the signed sum of {others}'
        assert np.max(np.abs(flow)) > 0, f'{topology}: no flow through {owner}'
        assert frozenset((owner_flows[owner], qualified(owner))) in pairs, f'{topology}: {owner} not set by its sum'
        for p in pressures:
            assert frozenset((qualified(pressure), qualified(p))) in pairs, \
                f'{topology}: {p} does not take the node pressure {pressure}'
    if topology == 'closed_loop':
        volumes = [results[f'{r["name"]}_module.q'] for r in spec['records']]
        total = sum(volumes)
        assert np.max(np.abs(total - total[0])) <= 1e-9 * max(np.max(np.abs(q)) for q in volumes)

    if spec['legacy']:
        # the same topology with junction module types: every variable both models keep agrees
        legacy_path = build(tmp_path, topology + '_legacy', legacy(spec['records'], spec['legacy']), params,
                            with_legacy=True)
        new, old = simulate(path), simulate(legacy_path)
        shared = [n for n in new if n in old and '_module.' in n and not n.endswith('.t')]
        vessels = [r for r in spec['records'] if r['module_type'] == 'arterial_simple']
        assert len(shared) >= 4 * len(vessels), f'{topology}: too few shared variables {shared}'
        for name in shared:
            scale = max(np.max(np.abs(old[name])), 1e-30)
            assert np.max(np.abs(new[name] - old[name])) <= 1e-8 * scale, \
                f'{topology}: {name} differs from the model built with junction module types'


def test_two_compliant_ends_at_one_node_is_an_error(tmp_path):
    records = [flow_source('src', ['a']), vessel('a', 'vp', ['src'], ['b', 'c']),
               vessel('b', 'vp', ['a'], ['sink_b']), vessel('c', 'vp', ['a'], ['sink_c']),
               sink('sink_b', ['b']), sink('sink_c', ['c'])]
    with pytest.raises(ValueError, match='more than one end'):
        build(tmp_path, 'two_owners', records, parameters(records))


def test_a_node_nothing_sets_the_pressure_of_is_an_error(tmp_path):
    records = [flow_source('src', ['x']), vessel('x', 'vv', ['src'], ['a']), vessel('a', 'pp', ['x'], ['b', 'c']),
               vessel('b', 'pp', ['a'], ['sink_b']), vessel('c', 'pp', ['a'], ['sink_c']),
               sink('sink_b', ['b']), sink('sink_c', ['c'])]
    with pytest.raises(ValueError, match='nothing there sets the pressure'):
        build(tmp_path, 'no_owner', records, parameters(records))


@pytest.mark.unit
def test_a_1d_vessel_in_a_node_of_three_is_refused():
    from libcuflynx.generators.port_nodes import find_nodes, node_variable_pairs
    sum_port = {'port_type': 'vessel_port', 'variables': ['v_in', 'u'], 'multi_port': ['sum', 'True']}
    plain = {'port_type': 'vessel_port', 'variables': ['v', 'u_out']}
    df = pd.DataFrame([
        {'name': 'fv', 'module_format': 'external_api', 'entrance_ports': [], 'exit_ports': [plain],
         'inp_vessels': [], 'out_vessels': ['m']},
        {'name': 'a', 'module_format': 'cellml', 'entrance_ports': [], 'exit_ports': [plain],
         'inp_vessels': [], 'out_vessels': ['m']},
        {'name': 'm', 'module_format': 'cellml', 'entrance_ports': [sum_port], 'exit_ports': [],
         'inp_vessels': ['fv', 'a'], 'out_vessels': []}])
    node = [n for n in find_nodes(df) if n.owner is not None][0]
    with pytest.raises(NotImplementedError, match='fv has no CellML'):
        node_variable_pairs(node, dict(zip(df['name'], df['module_format'])))


def test_microvascular_network(tmp_path):
    '''circulatory-autogen-modules' microvasculature in the same format: an arteriole whose outlet
    owns the split into two capillaries, and a venule whose inlet owns their merge (the former
    arteriole_Nout / venule_Min, now arteriole pv_micro_noI / venule vp_micro_noI).'''
    records = [pressure_source('src', ['art']),
               vessel('art', 'pv_micro_noI', ['src'], ['cap1', 'cap2'], module_type='arteriole'),
               vessel('cap1', 'pp_micro', ['art'], ['ven'], module_type='capillary'),
               vessel('cap2', 'pp_micro', ['art'], ['ven'], module_type='capillary'),
               vessel('ven', 'vp_micro_noI', ['cap1', 'cap2'], ['sink'], module_type='venule'),
               sink('sink', ['ven'])]
    rows = [('P_src', 'J_per_m3', 9500.0), ('P_sink', 'J_per_m3', 2830.0),
            ('mu', 'Js_per_m3', 0.004), ('rho', 'Js2_per_m5', 1040.0), ('g', 'm_per_s2', 9.81),
            ('beta_g', 'dimensionless', 0.0), ('a_vessel', 'dimensionless', 0.2802), ('b_vessel', 'per_m', -505.3),
            ('c_vessel', 'dimensionless', 0.1324), ('d_vessel', 'per_m', -11.14)]
    for name, r_0, u_0 in (('art', 1.525e-05, 8000.0), ('ven', 1.83e-05, 3330.0)):
        rows += [(f'E_{name}', 'J_per_m3', 18000.0), (f'l_{name}', 'metre', 1e-4), (f'r_0_{name}', 'metre', r_0),
                 (f'u_0_{name}', 'J_per_m3', u_0), (f'u_ext_{name}', 'J_per_m3', 0.0), (f'theta_{name}', 'dimensionless', 0.0)]
    for name, length in (('cap1', 2.5e-4), ('cap2', 5e-4)):
        rows += [(f'E_{name}', 'J_per_m3', 4300.0), (f'l_{name}', 'metre', length), (f'r_{name}', 'metre', 4e-6),
                 (f'u_ext_{name}', 'J_per_m3', 0.0), (f'q_C_init_{name}', 'm3', 0.0)]
    params = pd.DataFrame([{'variable_name': n, 'units': u, 'value': v, 'data_reference': 'test'} for n, u, v in rows])
    path = build(tmp_path, 'micro', records, params)

    assert sum_components(path) == ['multiport_sum_art_v_out', 'multiport_sum_ven_v_in']
    pairs = connections(path)
    for cap in ('cap1', 'cap2'):
        assert frozenset(('art_module.u', f'{cap}_module.u_in')) in pairs
        assert frozenset(('ven_module.u', f'{cap}_module.u_out')) in pairs
    results = simulate(path, ['multiport_sum_art_v_out.v_out', 'multiport_sum_ven_v_in.v_in',
                              'cap1_module.v', 'cap2_module.v', 'cap1_module.v_d', 'cap2_module.v_d'])
    split = results['multiport_sum_art_v_out.v_out']
    merge = results['multiport_sum_ven_v_in.v_in']
    np.testing.assert_allclose(split, results['cap1_module.v'] + results['cap2_module.v'], rtol=1e-9, atol=1e-25)
    np.testing.assert_allclose(merge, results['cap1_module.v_d'] + results['cap2_module.v_d'], rtol=1e-9, atol=1e-25)
    assert np.max(split) > 0
    # the longer capillary carries less
    assert results['cap2_module.v'][-1] < results['cap1_module.v'][-1]
