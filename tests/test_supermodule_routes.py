'''
Supermodules that are used like modules: routes, shared parameters, and templates
(libcuflynx.utilities.supermodules).
'''
import numpy as np
import pytest

from libcuflynx.parsers.ModelParsers import merge_default_parameters
from libcuflynx.utilities.config_schemas import normalise_supermodule_entry
from libcuflynx.utilities.supermodules import expand_supermodules

pytestmark = pytest.mark.unit


def _record(name, vessel_type, bc, inp=(), out=(), **extra):
    return dict({'name': name, 'vessel_type': vessel_type, 'BC_type': bc,
                 'inp_vessels': list(inp), 'out_vessels': list(out)}, **extra)


def _component(entrance=(), exit=()):
    return {'entrance_ports': [{'port_type': t, 'variables': []} for t in entrance],
            'exit_ports': [{'port_type': t, 'variables': []} for t in exit], 'general_ports': []}


# a lumped vessel: inlet compliance C, resistance R, inertance I
VESSEL = normalise_supermodule_entry({
    'vessel_type': 'lumped', 'BC_type': 'vp_lumped', 'module_format': 'supermodule',
    'shared_parameters': ['r_0', 'l'],
    'routes': {'inputs': {'vessel_port': 'C'},
               'outputs': {'vessel_port': 'I', 'volume_port': 'C'}},
    'submodules': [
        {'name': 'C', 'vessel_type': 'compliance', 'BC_type': 'vv', 'inp_vessels': [], 'out_vessels': ['R']},
        {'name': 'R', 'vessel_type': 'resistance', 'BC_type': 'pv', 'inp_vessels': ['C'], 'out_vessels': ['I']},
        {'name': 'I', 'vessel_type': 'inertance', 'BC_type': 'pp', 'inp_vessels': ['R'], 'out_vessels': []}]})
REGISTRY = {('lumped', 'vp_lumped'): VESSEL}
COMPONENTS = {('source', 'nn'): _component(exit=['vessel_port']),
              ('sink', 'nn'): _component(entrance=['vessel_port']),
              ('volume_sum', 'nn'): _component(entrance=['volume_port']),
              ('compliance', 'vv'): _component(['vessel_port'], ['vessel_port', 'volume_port']),
              ('resistance', 'pv'): _component(['vessel_port'], ['vessel_port']),
              ('inertance', 'pp'): _component(['vessel_port'], ['vessel_port'])}


def test_hosts_are_routed_by_port_type():
    records = [_record('src', 'source', 'nn', out=['ves']),
               _record('ves', 'lumped', 'vp_lumped', inp=['src'], out=['snk', 'vol']),
               _record('snk', 'sink', 'nn', inp=['ves']),
               _record('vol', 'volume_sum', 'nn', inp=['ves'])]
    expanded, _ = expand_supermodules(records, REGISTRY, component_registry=COMPONENTS)
    by = {r['name']: r for r in expanded}
    assert by['src']['out_vessels'] == ['ves_C']
    assert by['ves_C']['inp_vessels'] == ['src']
    assert by['ves_I']['out_vessels'] == ['snk']
    assert by['ves_C']['out_vessels'] == ['ves_R', 'vol']
    assert by['snk']['inp_vessels'] == ['ves_I']
    assert by['vol']['inp_vessels'] == ['ves_C']


def test_a_supermodule_host_offers_its_routes():
    records = [_record('a', 'lumped', 'vp_lumped', out=['b']),
               _record('b', 'lumped', 'vp_lumped', inp=['a'])]
    expanded, _ = expand_supermodules(records, REGISTRY, component_registry=COMPONENTS)
    by = {r['name']: r for r in expanded}
    assert by['a_I']['out_vessels'] == ['b_C']
    assert by['b_C']['inp_vessels'] == ['a_I']


def test_a_host_with_no_routed_port_is_an_error():
    records = [_record('vol', 'volume_sum', 'nn', out=['ves']), _record('ves', 'lumped', 'vp_lumped', inp=['vol'])]
    components = dict(COMPONENTS)
    components[('volume_sum', 'nn')] = _component(exit=['volume_port'])
    with pytest.raises(ValueError, match='routes\\["inputs"\\]'):
        expand_supermodules(records, REGISTRY, component_registry=components)


def test_shared_parameters_are_one_model_parameter(tmp_path):
    instances = tmp_path / 'instances' / 'default'
    instances.mkdir(parents=True)
    (instances / 'default_parameters.csv').write_text(
        'variable_name,units,value,data_reference\nr_0,metre,0.01,test\nR_R,Js_per_m6,5,test\n')
    vessel = dict(VESSEL, default_instance='default', config_path=str(tmp_path / 'lumped_modules_config.json'))
    records = [_record('ves', 'lumped', 'vp_lumped')]
    expanded, rows = expand_supermodules(records, {('lumped', 'vp_lumped'): vessel}, component_registry=COMPONENTS)
    by = {r['variable_name']: r for r in rows}
    assert by['r_0_ves']['value'] == '0.01'
    assert not any(n.startswith('r_0_ves_') for n in by)
    assert by['R_ves_R']['value'] == '5'
    # l is shared but unset: a row the host file can fill
    assert by['l_ves']['value'] is None
    # every submodule takes both, by their model names
    for rec in expanded:
        assert rec['parameter_names'] == {'r_0': 'r_0_ves', 'l': 'l_ves'}


def test_the_host_file_sets_a_shared_parameter_once():
    dtype = [('variable_name', 'U32'), ('units', 'U32'), ('value', 'U32'), ('data_reference', 'U32')]
    host = np.array([('l_ves', 'metre', '0.2', 'host'), ('R_ves_R', 'Js_per_m6', '7', 'host')], dtype=dtype)
    extra = [{'variable_name': 'l_ves', 'value': None, 'units': '', 'data_reference': '', 'shared_from': 'l_ves'},
             {'variable_name': 'r_0_ves', 'value': '0.01', 'units': 'metre', 'data_reference': 'i',
              'shared_from': 'r_0_ves'},
             {'variable_name': 'R_ves_R', 'value': '5', 'units': 'Js_per_m6', 'data_reference': 'i'},
             {'variable_name': 'm_ves', 'value': None, 'units': '', 'data_reference': '', 'shared_from': 'm_ves'}]
    merged = {str(r['variable_name']): str(r['value']) for r in merge_default_parameters(host, extra)}
    assert merged['l_ves'] == '0.2'        # the host's row
    assert merged['r_0_ves'] == '0.01'     # the instance's value
    assert merged['R_ves_R'] == '7'        # the host wins
    assert 'm_ves' not in merged           # nobody set it


def test_a_template_lists_its_slots_and_cannot_be_generated():
    template = normalise_supermodule_entry({
        'module_type': 'lumped', 'module_subtype': 'vp_empty', 'module_format': 'supermodule', 'template': True,
        'submodules': [
            {'name': 'C', 'module_type': 'compliance', 'choices': ['vv_linear', 'vv_nonlinear'],
             'inp_instances': [], 'out_instances': ['R']},
            {'name': 'R', 'module_type': 'resistance', 'inp_instances': ['C'], 'out_instances': []}]})
    assert [s['BC_type'] for s in template['submodules']] == [None, None]
    with pytest.raises(ValueError, match='empty supermodule.*C: a compliance version, one of'):
        expand_supermodules([_record('ves', 'lumped', 'vp_empty')], {('lumped', 'vp_empty'): template})


def test_a_slot_without_a_version_needs_a_template():
    with pytest.raises(ValueError, match='only allowed in a template'):
        normalise_supermodule_entry({
            'module_type': 'lumped', 'module_subtype': 'vp_x', 'module_format': 'supermodule',
            'submodules': [{'name': 'C', 'module_type': 'compliance', 'inp_instances': [], 'out_instances': []}]})


def test_a_shared_parameter_can_keep_a_monolithic_name(tmp_path):
    instances = tmp_path / 'instances' / 'default'
    instances.mkdir(parents=True)
    (instances / 'default_parameters.csv').write_text(
        'variable_name,units,value,data_reference\nC_T,m6_per_J,2e-8,test\n')
    vessel = dict(VESSEL, default_instance='default', config_path=str(tmp_path / 'lumped_modules_config.json'),
                  shared_parameters=[{'name': 'C_T', 'variable': 'C', 'submodules': ['C']}])
    expanded, rows = expand_supermodules([_record('ves', 'lumped', 'vp_lumped')], {('lumped', 'vp_lumped'): vessel},
                                         component_registry=COMPONENTS)
    by = {r['variable_name']: r for r in rows}
    assert by['C_T_ves']['value'] == '2e-8'
    recs = {r['name']: r for r in expanded}
    assert recs['ves_C']['parameter_names'] == {'C': 'C_T_ves'}
    assert 'parameter_names' not in recs['ves_R']


def test_an_aliased_shared_parameter_must_name_its_own_submodules():
    with pytest.raises(ValueError, match='shared_parameters'):
        normalise_supermodule_entry({
            'module_type': 'lumped', 'module_subtype': 'vp_x', 'module_format': 'supermodule',
            'shared_parameters': [{'name': 'C_T', 'variable': 'C', 'submodules': ['nope']}],
            'submodules': [{'name': 'C', 'module_type': 'compliance', 'module_subtype': 'vv',
                            'inp_instances': [], 'out_instances': []}]})


def test_outputs_are_exposed_under_the_instance_name():
    vessel = dict(VESSEL, outputs={'u': 'C/u', 'v': 'I/v'})
    expanded, _ = expand_supermodules([_record('ves', 'lumped', 'vp_lumped')], {('lumped', 'vp_lumped'): vessel},
                                      component_registry=COMPONENTS)
    recs = {r['name']: r for r in expanded}
    assert recs['ves_C']['output_aliases'] == {'ves': {'u': 'u'}}
    assert recs['ves_I']['output_aliases'] == {'ves': {'v': 'v'}}
    assert 'output_aliases' not in recs['ves_R']


def test_an_output_must_name_a_submodule():
    with pytest.raises(ValueError, match='"outputs"'):
        normalise_supermodule_entry({
            'module_type': 'lumped', 'module_subtype': 'vp_x', 'module_format': 'supermodule',
            'outputs': {'u': 'nope/u'},
            'submodules': [{'name': 'C', 'module_type': 'compliance', 'module_subtype': 'vv',
                            'inp_instances': [], 'out_instances': []}]})
