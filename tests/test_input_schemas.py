"""The shipped JSON Schemas of the four input files (libcuflynx/schemas) agree with the readers.

* every schema is a valid draft 2020-12 schema, and every property it declares is described;
* the repository's obs_data and params_for_id files, the converted params_for_id CSVs and the
  examples in tests/test_inputs/schema_examples validate;
* the key sets the schemas allow are those the readers accept (read from the readers' own
  constants, or from the rule dicts in ``parse_obs_data_json``);
* inputs the readers refuse are refused by the schemas too.
"""

import ast
import glob
import inspect
import json
import os
import textwrap
import warnings

import pytest

from libcuflynx.parsers.PrimitiveParsers import (PARAM_PRIOR_TYPES, PARAMS_FOR_ID_ENTRY_KEYS,
                                                 PARAMS_FOR_ID_JSON_VERSION, ObsAndParamDataParser)
from libcuflynx.generators.multi_port import normalise_port_multi_port, validate_list_multi_port
from libcuflynx.schemas import (MODULE_ARRAY_SCHEMA, MODULE_CONFIG_SCHEMA, OBS_DATA_SCHEMA,
                                PARAMS_FOR_ID_SCHEMA, load_schema)
from libcuflynx.utilities.config_schemas import (COMPONENT_KEYS, MODULE_ARRAY_COLUMN_ORDER,
                                                 OLD_MODULE_ARRAY_KEYS, OLD_MODULE_CONFIG_KEYS,
                                                 normalise_module_config_entry)
from libcuflynx.utilities.obs_data_helpers import VALID_DATA_TYPES
from libcuflynx.utilities.protocol_shapes import EVENT_KEYS, SHAPE_KEYS, SHAPE_TYPES

jsonschema = pytest.importorskip('jsonschema')

pytestmark = pytest.mark.unit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, 'tests', 'test_inputs', 'schema_examples')
ALL_SCHEMAS = (MODULE_ARRAY_SCHEMA, MODULE_CONFIG_SCHEMA, OBS_DATA_SCHEMA, PARAMS_FOR_ID_SCHEMA)


def _validator(name):
    return jsonschema.Draft202012Validator(load_schema(name))


def _repo_files(pattern):
    return sorted(f for d in ('resources', 'tests/test_inputs', 'funcs_user')
                  for f in glob.glob(os.path.join(ROOT, d, '**', pattern), recursive=True))


# --------------------------------------------------------------------------------------------
# the schemas themselves
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize('name', ALL_SCHEMAS)
def test_schema_is_valid_draft_2020_12(name):
    schema = load_schema(name)
    assert schema['$schema'] == 'https://json-schema.org/draft/2020-12/schema'
    jsonschema.Draft202012Validator.check_schema(schema)


def _undescribed(node, path):
    """Declared properties without a description (conditional subschemas excluded)."""
    missing = []
    if isinstance(node, dict):
        for key, prop in (node.get('properties') or {}).items():
            if not (prop.get('description') or set(prop) == {'$ref'}):
                missing.append(f'{path}.{key}')
        for key, value in node.items():
            if key not in ('if', 'then', 'else', 'not'):
                missing += _undescribed(value, f'{path}/{key}')
    elif isinstance(node, list):
        for i, value in enumerate(node):
            missing += _undescribed(value, f'{path}[{i}]')
    return missing


@pytest.mark.parametrize('name', ALL_SCHEMAS)
def test_every_property_is_described(name):
    assert _undescribed(load_schema(name), name) == []


# --------------------------------------------------------------------------------------------
# real files validate
# --------------------------------------------------------------------------------------------

def test_the_examples_validate():
    kinds = {'_module_array.json': MODULE_ARRAY_SCHEMA,
             '_modules_config.json': MODULE_CONFIG_SCHEMA,
             '_obs_data.json': OBS_DATA_SCHEMA,
             '_params_for_id.json': PARAMS_FOR_ID_SCHEMA}
    seen = set()
    for path in sorted(glob.glob(os.path.join(EXAMPLES, '*.json'))):
        name = next(n for suffix, n in kinds.items() if path.endswith(suffix))
        with open(path) as f:
            _validator(name).validate(json.load(f))
        seen.add(name)
    assert seen == set(ALL_SCHEMAS)


def test_the_example_params_for_id_csv_converts_to_its_json():
    converted = ObsAndParamDataParser.params_for_id_csv_to_json(
        os.path.join(EXAMPLES, 'SN_simple_params_for_id.csv'))
    with open(os.path.join(EXAMPLES, 'SN_simple_params_for_id.json')) as f:
        assert converted == json.load(f)


def test_the_repository_obs_data_files_validate():
    files = _repo_files('*obs_data*.json')
    assert len(files) > 10
    validator = _validator(OBS_DATA_SCHEMA)
    for path in files:
        with open(path) as f:
            errors = list(validator.iter_errors(json.load(f)))
        assert not errors, (path, jsonschema.exceptions.best_match(errors).message)


def test_the_repository_params_for_id_files_validate_as_json():
    files = _repo_files('*params_for_id*.csv')
    assert len(files) > 10
    validator = _validator(PARAMS_FOR_ID_SCHEMA)
    for path in files:
        doc = ObsAndParamDataParser.params_for_id_csv_to_json(path)
        errors = list(validator.iter_errors(doc))
        assert not errors, (path, jsonschema.exceptions.best_match(errors).message)


# --------------------------------------------------------------------------------------------
# the reader fills a default for every key the schema allows
# --------------------------------------------------------------------------------------------

def _parse_obs_data_rules(variable):
    """The keys of the defaults dict ``variable`` in ``parse_obs_data_json``."""
    source = textwrap.dedent(inspect.getsource(ObsAndParamDataParser.parse_obs_data_json))
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                and any(isinstance(t, ast.Name) and t.id == variable for t in node.targets)):
            return {key.value for key in node.value.keys}
    raise AssertionError(f'no dict {variable} in parse_obs_data_json')


def test_obs_data_keys_match_the_parser():
    defs = load_schema(OBS_DATA_SCHEMA)['$defs']
    assert set(defs['dataItem']['properties']) == _parse_obs_data_rules('item_defaults')
    assert set(defs['predictionItem']['properties']) == \
        _parse_obs_data_rules('prediction_entry_defaults')
    # the required keys have no default
    assert set(defs['protocolInfo']['properties']) == \
        _parse_obs_data_rules('protocol_defaults') | set(defs['protocolInfo']['required'])
    assert set(defs['dataItem']['properties']['data_type']['enum']) == set(VALID_DATA_TYPES)

    shape = defs['shape']['oneOf'][1]
    assert set(shape['properties']) == SHAPE_KEYS
    assert set(shape['properties']['type']['enum']) == set(SHAPE_TYPES)
    assert set(defs['event']['properties']) == EVENT_KEYS


def test_params_for_id_keys_match_the_parser():
    schema = load_schema(PARAMS_FOR_ID_SCHEMA)
    props = schema['$defs']['entryKeys']['properties']
    assert set(props) == PARAMS_FOR_ID_ENTRY_KEYS
    assert set(props['prior']['anyOf'][0]['enum']) == set(PARAM_PRIOR_TYPES)

    prior_params = props['prior_params']['anyOf'][0]['properties']
    specs = {spec['name']: spec for meta in PARAM_PRIOR_TYPES.values() for spec in meta['params']}
    assert set(prior_params) == set(specs)
    for name, spec in specs.items():
        positive = 'exclusiveMinimum' in prior_params[name]['anyOf'][0]
        assert positive == spec['positive'], name

    assert PARAMS_FOR_ID_JSON_VERSION in schema['properties']['version']['enum']


def test_module_config_and_array_keys_match_the_reader():
    defs = load_schema(MODULE_CONFIG_SCHEMA)['$defs']
    assert set(defs['phlynxComponent']['required']) == set(COMPONENT_KEYS)
    refused = {r['required'][0] for r in defs['phlynxComponent']['not']['anyOf']}
    assert set(OLD_MODULE_CONFIG_KEYS) - {'default_parameters'} <= refused
    array_defs = load_schema(MODULE_ARRAY_SCHEMA)['$defs']
    assert set(MODULE_ARRAY_COLUMN_ORDER) <= set(array_defs['phlynxRecord']['properties'])
    refused = {r['required'][0] for r in array_defs['phlynxRecord']['not']['anyOf']}
    assert set(OLD_MODULE_ARRAY_KEYS) <= refused


# --------------------------------------------------------------------------------------------
# what the readers refuse, the schemas refuse
# --------------------------------------------------------------------------------------------

def _item(**changes):
    item = {'data_item_name': 'x mean', 'data_type': 'constant', 'unit': 'm',
            'operands': ['a/x'], 'operation': 'mean', 'value': 1.0, 'std': 0.1}
    item.update(changes)
    return {k: v for k, v in item.items() if v is not None}


def _obs_doc(items=(), predictions=None, **protocol):
    doc = {'protocol_info': dict({'pre_times': [0.0], 'sim_times': [[1.0]],
                                  'params_to_change': {}}, **protocol),
           'data_items': list(items) or [_item()]}
    if predictions is not None:
        doc['prediction_items'] = predictions
    return doc


def _parse_obs(doc):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return ObsAndParamDataParser().parse_obs_data_json(obs_data_dict=doc)


def test_a_valid_obs_data_document_is_accepted_by_both():
    doc = _obs_doc([_item(), _item(data_item_name='x series', data_type='series',
                                   operation=None, value=[1.0, 2.0], std=0.1, obs_dt=0.5)],
                   predictions=[{'data_item_name': 'y', 'operands': ['a/y'], 'unit': 'm'}],
                   protocol_shapes={'stim': {'events': [{'level': 1.0, 'length': 0.1}]}},
                   params_to_change={'a/I': [['stim']]})
    _parse_obs(doc)
    _validator(OBS_DATA_SCHEMA).validate(doc)


# Each refusal names where the problem is and what is wrong; the reader's error is the schema's
# (libcuflynx.schemas.validation), except where the reader advises on a removed value.
@pytest.mark.parametrize('doc, message', [
    (_obs_doc([_item(colour='red')]), r"at data_items\[0\][^:]*: .*'colour' was unexpected"),
    (_obs_doc(predictions=[{'data_item_name': 'y', 'operands': ['a/y'], 'unit': 'm',
                            'colour': 'red'}]), r"at prediction_items\[0\][^:]*: .*'colour'"),
    (_obs_doc(sim_time=[[1.0]]), r"at protocol_info: .*'sim_time' was unexpected"),
    (_obs_doc([_item(unit=None)]), r"at data_items\[0\][^:]*: 'unit' is a required property"),
    (_obs_doc([_item(value=[1.0, 2.0])]), r"(?s)at data_items\[0\][^:]*\.value: .*A constant's "
                                          r"value and std are single numbers"),
    (_obs_doc([_item(std=None)]), r"(?s)'std' is a required property.*needs a ground truth"),
    (_obs_doc([_item(data_type='series', operation=None, value=[1.0, 2.0], std=None,
                     obs_dt=0.5)]), r"(?s)'std' is a required property.*it needs std"),
    (_obs_doc([_item(data_type='series', operation=None, value=[1.0, 2.0], std=0.1)]),
     r"(?s)'obs_dt' is a required property.*Required for a series"),
    (_obs_doc(predictions=[{'data_item_name': 'y', 'operands': ['a/y'], 'unit': 'm',
                            'value': 1.0}]), r"(?s)'data_type' is a required.*needs data_type"),
    (_obs_doc(protocol_shapes={'stim': {'events': [{'level': 1.0, 'length': 0.1, 'x': 1}]}},
              params_to_change={'a/I': [['stim']]}), r"events\[0\]: .*'x' was unexpected"),
    (_obs_doc(protocol_shapes={'stim': {'type': 'ramp', 'from': 0.0}},
              params_to_change={'a/I': [['stim']]}), r"(?s)'to' is a required.*Required for a ramp"),
])
def test_obs_data_the_parser_refuses_the_schema_refuses(doc, message):
    with pytest.raises(ValueError, match=message):
        _parse_obs(doc)
    assert not _validator(OBS_DATA_SCHEMA).is_valid(doc)


def _resolve_params(doc):
    parser = ObsAndParamDataParser()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        entries = parser.resolve_params_for_id_doc(doc)
        return parser._build_param_id_info_from_entries(entries)


def _params(*entries, **top):
    return dict({'version': 1, 'params': list(entries)}, **top)


GOOD_ENTRY = {'targets': ['a/C'], 'min': 1, 'max': 2}


def test_a_valid_params_for_id_document_is_accepted_by_both():
    doc = _params(dict(GOOD_ENTRY, prior='normal', prior_params={'prior_mean': 1.5}),
                  {'name': 'k', 'modifies': ['b/R', 'c/R'], 'modifier': 'scale',
                   'min': 0.5, 'max': 2.0},
                  {'targets': 'd/E', 'unbounded': 'yes', 'prior': 'normal',
                   'prior_params': {'prior_mean': '0', 'prior_std': '1'}},
                  defaults={'name_for_plotting': 'p'})
    _resolve_params(doc)
    _validator(PARAMS_FOR_ID_SCHEMA).validate(doc)


@pytest.mark.parametrize('doc, message', [
    (_params(dict(GOOD_ENTRY, colour='red')), r"at params\[0\]: .*'colour' was unexpected"),
    (_params(GOOD_ENTRY, defaults={'colour': 'red'}), r"at defaults: .*'colour' was unexpected"),
    (_params(dict(GOOD_ENTRY, modifies=['b/R'])), r"(?s)not allowed.*targets\) or a modifier"),
    (_params(dict(GOOD_ENTRY, modifier='scale')), r'(?s)not allowed.*modifier entries only'),
    (_params({'modifies': ['b/R'], 'modifier': 'scale', 'operation': 'scale', 'min': 0.5,
              'max': 2}), r"'operation' was unexpected"),
    (_params(dict(GOOD_ENTRY, targets=['C'])), r"(?s)targets\[0\]: 'C' is not in the form.*"
                                               r"component/param"),
    (_params(dict(GOOD_ENTRY, prior='beta')), r"prior: 'beta' is not one of"),
    (_params(dict(GOOD_ENTRY, prior='normal', prior_params={'prior_std': -1})),
     r'prior_std: -1 is less than or equal to the minimum of 0'),
    (_params(dict(GOOD_ENTRY, min='low')), r"min: 'low' is not in the form"),
    (_params(dict(GOOD_ENTRY, unbounded='maybe')), r"unbounded: 'maybe' is not one of"),
    (_params(GOOD_ENTRY, version=2), r'version: 2 is not one of'),
    ({'version': 1, 'params': []}, r'params: \[\] should be non-empty'),
])
def test_params_for_id_the_parser_refuses_the_schema_refuses(doc, message):
    with pytest.raises(ValueError, match=message):
        _resolve_params(doc)
    assert not _validator(PARAMS_FOR_ID_SCHEMA).is_valid(doc)


def _component(**port):
    return {'module_type': 'v', 'module_subtype': 'nn', 'component_file': 'm.cellml', 'component_type': 'v_type',
            'module_format': 'cellml',
            'entrance_ports': [dict({'port_type': 'flow_port', 'variables': ['q']}, **port)],
            'exit_ports': [], 'variables_and_units': [['q', 'm3_per_s', 'access', 'variable']]}


@pytest.mark.parametrize('port, accepted', [
    ({'multi_port': 'Multiply', 'multiply_factor': 2}, True),
    ({'multi_port': 'multiply', 'multiply_factor': '0.5'}, True),
    ({'multi_port': ['SUM']}, True),
    ({'multiply_factor': 2}, False),
    ({'multi_port': 'True', 'multiply_factor': 2}, False),
    ({'multi_port': 'Multiply', 'multiply_factor': 'two'}, False),
])
def test_module_config_multiply_factor_in_the_reader_and_the_schema(port, accepted):
    entry = _component(**port)
    assert _validator(MODULE_CONFIG_SCHEMA).is_valid([entry]) == accepted
    if accepted:
        normalise_module_config_entry(entry)
    else:
        with pytest.raises(ValueError, match='multiply_factor'):
            normalise_module_config_entry(entry)


def test_module_config_list_multi_port_entries_are_sum_or_true():
    port = {'port_type': 'flow_port', 'variables': ['q'], 'multi_port': ['bogus']}
    with pytest.raises(ValueError, match='must be "sum" or "True"'):
        validate_list_multi_port(normalise_port_multi_port(port, 'test'), 'test')
    assert not _validator(MODULE_CONFIG_SCHEMA).is_valid([_component(multi_port=['bogus'])])


@pytest.mark.parametrize('variable', [['q', 'm3_per_s', 'access'],
                                      ['q', 'm3_per_s', 'public', 'variable'],
                                      ['q', 'm3_per_s', 'access', 'variable', 'extra']])
def test_module_config_variables_and_units_rows_have_four_fields(variable):
    entry = _component()
    entry['variables_and_units'] = [variable]
    assert not _validator(MODULE_CONFIG_SCHEMA).is_valid([entry])


def test_module_array_names_may_have_surrounding_whitespace_only():
    from libcuflynx.utilities.config_schemas import normalise_vessel_record
    validator = _validator(MODULE_ARRAY_SCHEMA)
    record = {'name': ' a ', 'module_type': 'v', 'module_subtype': 'nn'}
    assert normalise_vessel_record(record)['name'] == 'a'
    assert validator.is_valid([record])
    record = dict(record, name='a b')
    with pytest.raises(ValueError, match='contains whitespace'):
        normalise_vessel_record(record)
    assert not validator.is_valid([record])
