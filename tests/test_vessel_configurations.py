'''
Vessels connected in any configuration, through ports that sum over a node.

Each topology is built from the module library's ordinary vessels (``arterial_simple`` in its four
BC types, with constant flow sources and pressure sinks), generated as CellML and simulated. At
every node of three or more ends, the end alone on its side is compliant (node rule D,
``generators/vessel_nodes.py``) and its vessel_port has ``"multi_port": ["sum", "True"]``. The
tests check:

* the CellML: such a node gets one ``multiport_sum_<owner>_<variable>`` component, and the owner's
  pressure is mapped to every other end;
* the physics: the owner's flow is the sum of the other ends' flows, and in a closed loop the
  volume is conserved.

These are the configurations the removed junction module types (Min/Nout/MinNout, split/merge)
had a module type each for. Refused nodes (n-to-m, a lone end that is not compliant) are tested
in test_vessel_nodes.py.
'''
import contextlib
import io
import json
import re

import numpy as np
import pytest

from _module_library import modules_dir_or_skip

pytestmark = pytest.mark.integration

VESSEL = {'R': ('Js_per_m6', 1.0e7), 'C': ('m6_per_J', 1.0e-9), 'I': ('Js2_per_m6', 1.0e5),
          'q_0': ('m3', 0.0), 'u_0': ('J_per_m3', 0.0), 'u_ext': ('J_per_m3', 0.0)}
SIM_TIME, DT = 0.5, 1e-3
SOLVER_INFO = {'rtol': 1e-10, 'atol': 1e-14}


def vessel(name, bc, inputs, outputs, module_type='arterial_simple'):
    return {'name': name, 'module_type': module_type, 'module_subtype': bc,
            'inp_instances': list(inputs), 'out_instances': list(outputs)}


def flow_source(name, outputs):
    return {'name': name, 'module_type': 'inlet_flow', 'module_subtype': 'constant',
            'inp_instances': [], 'out_instances': list(outputs)}


def sink(name, inputs):
    return {'name': name, 'module_type': 'outlet_pressure', 'module_subtype': 'constant',
            'inp_instances': list(inputs), 'out_instances': []}


def leaf(units, value):
    return {'value': value, 'units': units, 'data_reference': 'test'}


def parameters(records, overrides=None):
    doc = {'schema_version': 1}
    for r in records:
        name = r['name']
        if r['module_type'] == 'inlet_flow':
            doc[name] = {'v': leaf('m3_per_s', 1.0e-5)}
        elif r['module_type'] == 'outlet_pressure':
            doc[name] = {'P': leaf('J_per_m3', 1000.0)}
        else:
            doc[name] = {var: leaf(units, value) for var, (units, value) in VESSEL.items()}
    for (name, var), value in (overrides or {}).items():
        doc[name][var]['value'] = value
    return doc


def build(tmp_path, tag, records, params):
    '''Generate the model from the module library alone; returns the path of its CellML.'''
    from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
    resources = tmp_path / tag
    resources.mkdir()
    (resources / f'{tag}_module_array.json').write_text(json.dumps(records))
    (resources / f'{tag}_parameters.json').write_text(json.dumps(params))
    config = {'file_prefix': tag, 'input_param_file': f'{tag}_parameters.json', 'model_type': 'cellml',
              'solver': 'CVODE_myokit', 'resources_dir': str(resources),
              'generated_models_dir': str(tmp_path / 'generated'), 'DEBUG': False,
              'use_builtin_modules': False, 'module_library_dirs': [str(modules_dir_or_skip())]}
    with contextlib.redirect_stdout(io.StringIO()):
        assert generate_with_new_architecture(False, config), f'generation of {tag} failed'
    return str(tmp_path / 'generated' / tag / f'{tag}.cellml')


def simulate(cellml_path, names):
    '''{"component.variable": values} for ``names``, simulated with Myokit.'''
    import myokit
    from libcuflynx.solver_wrappers import get_simulation_helper
    with contextlib.redirect_stdout(io.StringIO()):
        model = get_simulation_helper(model_path=cellml_path, solver='CVODE_myokit', model_type='cellml',
                                      dt=DT, sim_time=SIM_TIME, solver_info=SOLVER_INFO, pre_time=0.0).model
    sim = myokit.Simulation(model)
    sim.set_tolerance(SOLVER_INFO['atol'], SOLVER_INFO['rtol'])
    log = sim.run(SIM_TIME, log=list(names), log_interval=DT)
    return {n: np.asarray(log[n], dtype=float) for n in names}


def connections(cellml_path):
    '''Every mapped variable pair of the CellML, as frozensets of "component.variable".'''
    with open(cellml_path) as f:
        text = f.read()
    pairs = set()
    for block in re.findall(r'<connection>(.*?)</connection>', text, re.S):
        c1, c2 = re.search(r'component_1="([^"]+)"\s+component_2="([^"]+)"', block).groups()
        for v1, v2 in re.findall(r'variable_1="([^"]+)"\s+variable_2="([^"]+)"', block):
            pairs.add(frozenset((f'{c1}.{v1}', f'{c2}.{v2}')))
    return pairs


def qualified(name):
    component, variable = name.split('/')
    return f'{component}_module.{variable}'


def sum_components(cellml_path):
    with open(cellml_path) as f:
        return sorted(set(re.findall(r'<component name="(multiport_sum_[^"]+)"', f.read())))


# Each topology: its records, the sum components a node of three or more ends gets, the nodes
# [(owner flow, [other flows], owner pressure, [other pressures])] and parameter overrides.
TOPOLOGIES = {
    # one compliant outlet feeding two vessels (was Nout_junction, split_junction)
    'split': dict(
        records=[flow_source('src', ['a']), vessel('a', 'vv', ['src'], ['b', 'c']),
                 vessel('b', 'pp', ['a'], ['sink_b']), vessel('c', 'pp', ['a'], ['sink_c']),
                 sink('sink_b', ['b']), sink('sink_c', ['c'])],
        sums=['multiport_sum_a_v_out'],
        nodes=[('a/v_out', ['b/v', 'c/v'], 'a/u_d', ['b/u_in', 'c/u_in'])]),
    # two vessels feeding one compliant inlet (was Min_junction, merge_junction)
    'merge': dict(
        records=[flow_source('src1', ['p1']), flow_source('src2', ['p2']),
                 vessel('p1', 'vp', ['src1'], ['m']), vessel('p2', 'vp', ['src2'], ['m']),
                 vessel('m', 'vv', ['p1', 'p2'], ['d']), vessel('d', 'pp', ['m'], ['sink']), sink('sink', ['d'])],
        sums=['multiport_sum_m_v_in'],
        nodes=[('m/v_in', ['p1/v', 'p2/v'], 'm/u', ['p1/u_out', 'p2/u_out'])]),
    # a merge and a split through one vessel (was MinNout_junction, 2in2out_junction)
    'merge_and_split': dict(
        records=[flow_source('src1', ['p1']), flow_source('src2', ['p2']),
                 vessel('p1', 'vp', ['src1'], ['m']), vessel('p2', 'vp', ['src2'], ['m']),
                 vessel('m', 'vv', ['p1', 'p2'], ['b', 'c']), vessel('b', 'pp', ['m'], ['sink_b']),
                 vessel('c', 'pp', ['m'], ['sink_c']), sink('sink_b', ['b']), sink('sink_c', ['c'])],
        sums=['multiport_sum_m_v_in', 'multiport_sum_m_v_out'],
        nodes=[('m/v_in', ['p1/v', 'p2/v'], 'm/u', ['p1/u_out', 'p2/u_out']),
               ('m/v_out', ['b/v', 'c/v'], 'm/u_d', ['b/u_in', 'c/u_in'])]),
    # three daughters from a pv vessel, which the two-port split_junction could not take
    'three_way_split_from_pv': dict(
        records=[flow_source('src', ['x']), vessel('x', 'vv', ['src'], ['a']),
                 vessel('a', 'pv', ['x'], ['b', 'c', 'e']),
                 vessel('b', 'pp', ['a'], ['sink_b']), vessel('c', 'pp', ['a'], ['sink_c']),
                 vessel('e', 'pp', ['a'], ['sink_e']),
                 sink('sink_b', ['b']), sink('sink_c', ['c']), sink('sink_e', ['e'])],
        sums=['multiport_sum_a_v_out'],
        nodes=[('a/v_out', ['b/v', 'c/v', 'e/v'], 'a/u', ['b/u_in', 'c/u_in', 'e/u_in'])]),
    # two vessels merging into a vp vessel (was merge_junction vp)
    'merge_into_vp': dict(
        records=[flow_source('src1', ['p1']), flow_source('src2', ['p2']),
                 vessel('p1', 'vp', ['src1'], ['m']), vessel('p2', 'vp', ['src2'], ['m']),
                 vessel('m', 'vp', ['p1', 'p2'], ['sink']), sink('sink', ['m'])],
        sums=['multiport_sum_m_v_in'],
        nodes=[('m/v_in', ['p1/v', 'p2/v'], 'm/u', ['p1/u_out', 'p2/u_out'])]),
    # a closed loop through a split and a merge: the volume is conserved
    'closed_loop': dict(
        records=[vessel('a', 'vv', ['e'], ['b', 'c']), vessel('b', 'pp', ['a'], ['d']), vessel('c', 'pp', ['a'], ['d']),
                 vessel('d', 'vv', ['b', 'c'], ['e']), vessel('e', 'pp', ['d'], ['a'])],
        sums=['multiport_sum_a_v_out', 'multiport_sum_d_v_in'],
        nodes=[('a/v_out', ['b/v', 'c/v'], 'a/u_d', ['b/u_in', 'c/u_in']),
               ('d/v_in', ['b/v_d', 'c/v_d'], 'd/u', ['b/u_out', 'c/u_out'])],
        overrides={('a', 'u_0'): 3000.0, ('d', 'u_0'): 500.0}),
}


@pytest.mark.parametrize('topology', list(TOPOLOGIES))
def test_configuration(tmp_path, topology):
    spec = TOPOLOGIES[topology]
    path = build(tmp_path, topology, spec['records'], parameters(spec['records'], spec.get('overrides')))
    assert [s for s in sum_components(path) if s in spec['sums']] == sorted(spec['sums'])

    pairs = connections(path)
    owner_flows = {owner: 'multiport_sum_{0}_{1}.{1}'.format(*owner.split('/')) for owner, _, _, _ in spec['nodes']}
    names = sorted(set(owner_flows.values()) | {qualified(o) for _, others, _, _ in spec['nodes'] for o in others})
    if topology == 'closed_loop':
        names += [f'{r["name"]}_module.q' for r in spec['records']]
    results = simulate(path, names)
    for owner, others, pressure, pressures in spec['nodes']:
        flow = results[owner_flows[owner]]
        expected = sum(results[qualified(o)] for o in others)
        scale = max(np.max(np.abs(expected)), np.max(np.abs(flow)), 1e-30)
        assert np.max(np.abs(flow - expected)) <= 1e-9 * scale, f'{owner} is not the sum of {others}'
        assert np.max(np.abs(flow)) > 0, f'no flow through {owner}'
        assert frozenset((owner_flows[owner], qualified(owner))) in pairs, f'{owner} not set by its sum'
        for p in pressures:
            assert frozenset((qualified(pressure), qualified(p))) in pairs, \
                f'{p} does not take the node pressure {pressure}'
    if topology == 'closed_loop':
        volumes = [results[f'{r["name"]}_module.q'] for r in spec['records']]
        total = sum(volumes)
        assert np.max(np.abs(total - total[0])) <= 1e-9 * max(np.max(np.abs(q)) for q in volumes)
