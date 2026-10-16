"""Path-addressed nested instances, injective flat names and supermodule interfaces (#582).

A nested instance's vessel joins its path with ``__`` (``pair/coll`` is ``pair__coll``), and no
module-array or submodule name may contain ``__`` or begin or end with ``_``, so a flat name
splits back into one path. A host may name a submodule by its path in its own lists, a
``per_submodule_*`` key may be a path into a nested instance, and a supermodule may declare an
``interface`` that hosts bind to by port type. Names joined with a single ``_`` (before #582)
are still read, with a FutureWarning, when they name one vessel.

The fixture library (``flowpair``, ``inner``, ``outer``) is test_json_module_arrays'.
"""
import copy
import os
import warnings

import myokit
import pytest

from libcuflynx.solver_wrappers.name_resolver import VariableNameResolver
from libcuflynx.utilities.config_schemas import (load_component_registry,
                                                 normalise_module_config_entry,
                                                 normalise_vessel_record)
from libcuflynx.utilities.parameter_names import (flat_name, host_model_names,
                                                  legacy_flat_name)
from test_json_module_arrays import (_expand, _flowpair_host, _links, _rec, registry,  # noqa: F401
                                     super_library_dir)

pytestmark = pytest.mark.unit


def _host_by_path(order=('pair/coll',)):
    """The flowpair host, with src_a/src_b naming the collector by path instead of through
    per_submodule_inputs."""
    return [
        _rec('src_a', 'flow_src', out=list(order)),
        _rec('src_b', 'flow_src', out=['pair/coll']),
        _rec('pair', 'flowpair', subtype='supermodule', per_submodule_outputs={'g': ['rd']}),
        _rec('rd', 'reader', inp=['pair']),
    ]


# --------------------------------------------------------------------------------------------
# path references
# --------------------------------------------------------------------------------------------

def test_a_host_names_a_submodule_by_its_path(registry):
    by_path = _links(_expand(_host_by_path(), registry)[0])
    by_per_submodule = _links(_expand(_flowpair_host(), registry)[0])
    assert by_path == by_per_submodule
    assert by_path['src_a'] == ([], ['pair__coll'])


def test_a_path_valued_per_submodule_key_reaches_the_nested_instance(registry):
    host = [_rec('src', 'flow_src', out=['o']),
            _rec('o', 'outer', subtype='supermodule', per_submodule_inputs={'in/rd': ['src']})]
    links = _links(_expand(host, registry)[0])
    assert links['src'][1] == ['o__in__rd']
    assert links['o__in__rd'][0] == ['src', 'o__in__g']


def test_a_path_two_levels_down_in_a_host_list(registry):
    host = [_rec('src', 'flow_src', out=['o/in/rd']), _rec('o', 'outer', subtype='supermodule')]
    links = _links(_expand(host, registry)[0])
    assert links['src'][1] == ['o__in__rd']
    assert links['o__in__rd'][0] == ['src', 'o__in__g']


def test_each_host_keeps_its_own_order(registry):
    """per_submodule_* gives every host the submodules in one order; a path reference stands
    where the host wrote it."""
    host = [_rec('src_a', 'flow_src', out=['pair/g', 'x', 'pair/coll']),
            _rec('src_b', 'flow_src', out=['pair/coll', 'pair/g']),
            _rec('pair', 'flowpair', subtype='supermodule'),
            _rec('x', 'reader')]
    links = _links(_expand(host, registry)[0])
    assert links['src_a'][1] == ['pair__g', 'x', 'pair__coll']
    assert links['src_b'][1] == ['pair__coll', 'pair__g']
    assert links['pair__coll'][0] == ['src_a', 'src_b']


@pytest.mark.parametrize('host, message', [
    # a path left over after expansion names nothing
    ([_rec('src', 'flow_src', out=['nope/coll'])], r'names \[\'nope/coll\'\]'),
    # into a submodule the supermodule does not have
    ([_rec('src', 'flow_src', out=['pair/nope']), _rec('pair', 'flowpair', subtype='supermodule')],
     r'"nope" is not a submodule'),
    # an instance's own list may not hold a path
    ([_rec('pair', 'flowpair', subtype='supermodule', inp=['o/src']),
      _rec('o', 'outer', subtype='supermodule')], r'its own inp_instances names the paths'),
    # a nested instance is named through one of its submodules
    ([_rec('src', 'flow_src', out=['o/in']), _rec('o', 'outer', subtype='supermodule')],
     r'which is a supermodule instance'),
])
def test_a_bad_path_is_an_error(registry, host, message):
    with pytest.raises(ValueError, match=message):
        _expand(host, registry)


def test_a_path_in_a_supermodule_s_own_submodules_is_an_error():
    with pytest.raises(ValueError, match='not submodules of this supermodule'):
        normalise_module_config_entry({
            'module_type': 'x', 'module_subtype': 'supermodule', 'module_format': 'supermodule',
            'submodules': [_rec('a', 'flow_src', out=['b/c']), _rec('b', 'collector')]})


# --------------------------------------------------------------------------------------------
# names
# --------------------------------------------------------------------------------------------

def test_flat_names_are_injective():
    """a + b_c and a_b + c joined with "_" were one name; with "__" they are two."""
    assert flat_name(('a', 'b_c')) != flat_name(('a_b', 'c'))
    assert flat_name(('a', 'b_c')).split('__') == ['a', 'b_c']


@pytest.mark.parametrize('name', ['a__b', '_a', 'a_'])
@pytest.mark.parametrize('where', ['submodule', 'top-level'])
def test_bad_names_are_rejected(name, where):
    record = _rec(name, 'flow_src')
    with pytest.raises(ValueError, match=f'name "{name}" cannot be used'):
        if where == 'top-level':
            normalise_vessel_record(record, source='m_module_array.json', index=0)
        else:
            normalise_module_config_entry({
                'module_type': 'x', 'module_subtype': 'supermodule',
                'module_format': 'supermodule', 'submodules': [record]})


_VESSELS = {'pair__g': {'mean': 'constant'}, 'src_a': {'mean': 'constant'}}


@pytest.mark.parametrize('kind', ['param_file', 'params_for_id', 'operand'])
def test_old_names_warn(kind):
    if kind == 'param_file':
        with pytest.warns(FutureWarning, match='pair__g'):
            assert host_model_names(['mean_pair_g'], _VESSELS, 'p.csv') == ['mean_pair__g']
        with pytest.warns(FutureWarning, match='pair__g'):
            assert host_model_names(['pair_g/mean'], _VESSELS, 'p.csv') == ['mean_pair__g']
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert host_model_names(['mean_src_a'], _VESSELS, 'p.csv') == ['mean_src_a']
    elif kind == 'params_for_id':
        # a calibration parameter is resolved in the model by its {var}_{vessel} name
        model = myokit.parse_model(_MMT)
        keyed = [('var', {v.qname(): v for v in model.variables(deep=True)})]
        with pytest.warns(FutureWarning, match='parameters.mean_pair__g'):
            assert VariableNameResolver.resolve_key('pair_g/mean', keyed, separator='.') == \
                ('var', 'parameters.mean_pair__g')
    else:
        model = myokit.parse_model(_MMT)
        keyed = [('state', {v.qname(): i for i, v in enumerate(model.states())}),
                 ('var', {v.qname(): v for v in model.variables(deep=True)})]
        with pytest.warns(FutureWarning, match='pair__coll_module.v'):
            assert VariableNameResolver.resolve_key('pair_coll/v', keyed, separator='.') == \
                ('state', 'pair__coll_module.v')


_MMT = """
[[model]]
pair__coll_module.v = 0
[environment]
time = 0 [s] bind time
[parameters]
mean_pair__g = 1
[pair__coll_module]
dot(v) = parameters.mean_pair__g
"""


def test_old_names_in_a_workflow_s_params_for_id_warn(tmp_path):
    from libcuflynx.calibration_workflow import load_workflow, resolve_workflow
    from libcuflynx.external_testing.workflow_library import build_workflow_library
    library = build_workflow_library(str(tmp_path / 'library'))
    path = os.path.join(library['chain/rest'], 'rest_params_for_id.csv')
    with open(path) as f:
        text = f.read()
    with open(path, 'w') as f:
        f.write(text.replace('mod__C', 'mod_C'))
    with pytest.warns(FutureWarning, match='"mod__C"'):
        resolved = resolve_workflow(load_workflow(library['chain_rest']))
    assert [p.vessel for p in resolved.steps['rest'].parameters] == ['mod__C']


def test_an_ambiguous_old_name_is_an_error():
    with pytest.raises(ValueError, match=r"could be any of \['a__b_c', 'a_b__c'\]"):
        legacy_flat_name('a_b_c', ['a__b_c', 'a_b__c'], 'p.csv')
    vessels = {'a__b_c': {'k': 'constant'}, 'a_b__c': {'k': 'constant'}}
    with pytest.raises(ValueError, match='could be any of'):
        host_model_names(['k_a_b_c'], vessels, 'p.csv')


# --------------------------------------------------------------------------------------------
# interface
# --------------------------------------------------------------------------------------------

def _with_interface(registry, interface):
    config = copy.deepcopy(registry)
    config[('flowpair', 'supermodule')]['interface'] = interface
    return config


@pytest.fixture(scope='module')
def components(super_library_dir):
    return load_component_registry(
        [os.path.join(super_library_dir, 'schema_test_modules_config.json')])


def test_a_host_binds_through_the_interface(registry, components):
    """flow_src feeds through a flow_port, reader is fed through a gain_port: each binds to
    the first interface entry of its port type, with no per_submodule_* entry."""
    from libcuflynx.utilities.supermodules import expand_supermodules
    from libcuflynx.utilities.config_schemas import normalise_vessel_records
    config = _with_interface(registry, [
        {'name': 'gain_out', 'submodule': 'g', 'port_type': 'gain_port', 'label': 'gain'},
        {'name': 'flow_in', 'submodule': 'coll', 'port_type': 'flow_port', 'label': 'inflow'},
        {'name': 'flow_in_2', 'submodule': 'g', 'port_type': 'flow_port', 'label': 'unused'}])
    host = [_rec('src_a', 'flow_src', out=['pair']), _rec('src_b', 'flow_src', out=['pair']),
            _rec('pair', 'flowpair', subtype='supermodule'), _rec('rd', 'reader', inp=['pair'])]
    expanded, _ = expand_supermodules(normalise_vessel_records(host, 'test'), config, 'test',
                                      components=components)
    assert _links(expanded) == _links(_expand(_flowpair_host(), registry)[0])
    # without a matching port it is the usual error
    with pytest.raises(ValueError, match='none of its interface entries matches its ports'):
        expand_supermodules(normalise_vessel_records(host, 'test'),
                            _with_interface(registry, [{'name': 'x', 'submodule': 'g',
                                                        'port_type': 'volume_port'}]),
                            'test', components=components)


@pytest.mark.parametrize('interface, message', [
    ([{'name': 'a', 'submodule': 'coll', 'port_type': 'flow_port'},
      {'name': 'a', 'submodule': 'g', 'port_type': 'gain_port'}], 'interface name "a" is used twice'),
    ([{'name': 'a', 'submodule': 'nope', 'port_type': 'flow_port'}], 'names submodule "nope"'),
    ([{'name': 'a', 'port_type': 'flow_port'}], r'interface\[0\] needs'),
])
def test_an_interface_is_validated(registry, interface, message):
    with pytest.raises(ValueError, match=message):
        _expand(_flowpair_host(), _with_interface(registry, interface))


# --------------------------------------------------------------------------------------------
# PhLynx's own keys
# --------------------------------------------------------------------------------------------

def test_phlynx_keys_are_ignored():
    record = normalise_vessel_record(_rec('a', 'flow_src', phlynx_position={'x': 1},
                                          phlynx_label='A'), source='m.json', index=0)
    assert not any(k.startswith('phlynx_') for k in record)
    entry = normalise_module_config_entry({
        'module_type': 'x', 'module_subtype': 'supermodule', 'module_format': 'supermodule',
        'phlynx_layout': {'nodes': []},
        'submodules': [_rec('a', 'flow_src', phlynx_position={'x': 1})]})
    assert 'phlynx_layout' not in entry and 'phlynx_position' not in entry['submodules'][0]


# --------------------------------------------------------------------------------------------
# the formats: example files, checked by the schemas and the readers
# --------------------------------------------------------------------------------------------

_TEST_INPUTS = os.path.join(os.path.dirname(__file__), 'test_inputs')


def test_the_example_files_pass_the_schemas_and_expand(components):
    import json
    jsonschema = pytest.importorskip('jsonschema')
    from libcuflynx.schemas import MODULE_ARRAY_SCHEMA, MODULE_CONFIG_SCHEMA, load_schema
    from libcuflynx.utilities.config_schemas import (load_supermodule_registry,
                                                     read_module_array_records)
    from libcuflynx.utilities.supermodules import expand_supermodules
    config = os.path.join(_TEST_INPUTS, 'nested_paths_modules_config.json')
    array = os.path.join(_TEST_INPUTS, 'nested_paths_module_array.json')
    with open(config) as f:
        jsonschema.validate(json.load(f), load_schema(MODULE_CONFIG_SCHEMA))
    with open(array) as f:
        jsonschema.validate(json.load(f), load_schema(MODULE_ARRAY_SCHEMA))
    expanded, _ = expand_supermodules(read_module_array_records(array),
                                      load_supermodule_registry([config]), array,
                                      components=components)
    links = _links(expanded)
    assert links['src_a'][1] == ['pair__coll']           # a path
    assert links['src_b'][1] == ['pair__coll']           # the interface (flow_port)
    assert links['pair__coll'][0] == ['src_b', 'src_a']
    assert links['rd'][0] == ['pair__g']
    assert links['rd2'][0] == ['o__pair__g']             # a path two levels down
    assert links['o__pair__coll'][0] == ['o__src']


@pytest.mark.parametrize('schema, document', [
    ('array', [_rec('a__b', 'flow_src')]),
    ('array', [_rec('_a', 'flow_src')]),
    ('array', [_rec('a_', 'flow_src')]),
    ('config', [{'module_type': 'x', 'module_subtype': 'supermodule',
                 'module_format': 'supermodule',
                 'submodules': [_rec('a', 'flow_src', out=['b/c']), _rec('b', 'reader')]}]),
    ('config', [{'module_type': 'x', 'module_subtype': 'supermodule',
                 'module_format': 'supermodule', 'submodules': [_rec('a', 'flow_src')],
                 'interface': [{'name': 'i', 'port_type': 'flow_port'}]}]),
])
def test_the_schemas_refuse_what_the_readers_refuse(schema, document):
    jsonschema = pytest.importorskip('jsonschema')
    from libcuflynx.schemas import MODULE_ARRAY_SCHEMA, MODULE_CONFIG_SCHEMA, load_schema
    name = MODULE_ARRAY_SCHEMA if schema == 'array' else MODULE_CONFIG_SCHEMA
    assert not jsonschema.Draft202012Validator(load_schema(name)).is_valid(document)
