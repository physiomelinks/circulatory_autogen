"""PhLynx and CA read and write the same files: PhLynx builds models from module configs and
exports a module array, a module config and an obs_data that CA runs. This checks PhLynx's
side against CA's readers (which check every file against its schema on load) and against the
schemas' value lists.

Needs a PhLynx checkout: ``PHLYNX_DIR`` (CI checks out physiomelinks/phlynx main). Without one
the tests skip, unless ``CUFLYNX_REQUIRE_PHLYNX`` is set.
"""
import glob
import json
import os
import re

import pytest

from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser
from libcuflynx.schemas import MODULE_CONFIG_SCHEMA, load_schema
from libcuflynx.utilities.config_schemas import (_PORT_KEYS, VARIABLE_KINDS,
                                                 normalise_module_config,
                                                 read_module_array_records)

PHLYNX_DIR = os.environ.get('PHLYNX_DIR', '')


def _phlynx():
    if not PHLYNX_DIR or not os.path.isdir(PHLYNX_DIR):
        if os.environ.get('CUFLYNX_REQUIRE_PHLYNX'):
            raise AssertionError(f'PHLYNX_DIR ({PHLYNX_DIR!r}) is not a PhLynx checkout')
        pytest.skip('no PhLynx checkout (set PHLYNX_DIR)')
    return PHLYNX_DIR


def _files(*patterns):
    root = _phlynx()
    found = sorted({p for pattern in patterns
                    for p in glob.glob(os.path.join(root, pattern), recursive=True)})
    assert found, f'none of {patterns} in {root}: has PhLynx moved them?'
    return found


def _source_files():
    return _files('src/**/*.js')


def _definition(pattern, what):
    """The first match of ``pattern`` (whose group 1 is the definition's body) in PhLynx's src
    JavaScript, found by the definition's name wherever it lives."""
    for path in _source_files():
        with open(path, encoding='utf-8') as f:
            match = re.search(pattern, f.read(), re.S)
        if match:
            return match.group(1)
    raise AssertionError(f'{what} is not defined under src any more: has PhLynx moved it?')


def _constant_values(name):
    """The ``value``s of the option list ``export const name = [...]`` in PhLynx's src."""
    body = _definition(rf'export const {name}\s*=\s*\[(.*?)\]', name)
    return re.findall(r"value:\s*'([^']*)'", body)


def _multiport_values():
    """The multi_port values PhLynx offers: the values of ``const TYPES = {...}`` in its
    multiport.js (None, True, Sum, Multiply)."""
    for path in _source_files():
        if os.path.basename(path) != 'multiport.js':
            continue
        with open(path, encoding='utf-8') as f:
            match = re.search(r'const TYPES\s*=\s*\{(.*?)\}', f.read(), re.S)
        if match:
            return re.findall(r":\s*'([^']*)'", match.group(1))
    raise AssertionError('the multiport TYPES map is not in a multiport.js under src any more')


@pytest.mark.unit
def test_phlynx_s_module_configs_load_in_ca():
    for path in _files('src/assets/module_configs/*.json', 'tests/resources/**/*module*config*.json'):
        with open(path, encoding='utf-8-sig') as f:
            normalise_module_config(json.load(f), source=os.path.relpath(path, PHLYNX_DIR))


@pytest.mark.unit
def test_phlynx_s_module_arrays_load_in_ca():
    for path in _files('tests/resources/**/*_vessel_array.csv',
                       'tests/resources/**/*_module_array.csv',
                       'tests/resources/**/*_module_array.json'):
        assert read_module_array_records(path)


@pytest.mark.unit
def test_phlynx_s_obs_data_load_in_ca():
    for path in _files('tests/resources/**/*obs_data*.json'):
        ObsAndParamDataParser().parse_obs_data_json(param_id_obs_path=path)


@pytest.mark.unit
def test_phlynx_s_variable_kinds_are_ca_s():
    assert set(_constant_values('PARAMETER_TYPE_OPTIONS')) == set(VARIABLE_KINDS)


@pytest.mark.unit
def test_phlynx_s_port_lists_are_ca_s():
    assert set(_constant_values('PORT_TYPE_OPTIONS')) == set(_PORT_KEYS)


@pytest.mark.unit
def test_phlynx_s_multi_port_values_are_accepted_by_ca():
    values = _multiport_values()
    assert {'True', 'Sum'} <= set(values), values
    for value in values:
        port = {'port_type': 'vessel_port', 'variables': ['u']}
        # 'None' is PhLynx's menu entry for no multi_port: its export leaves the key out
        # (unparseMultiport in src/utils/config.js)
        if value != 'None':
            port['multi_port'] = value
        entry = {'module_type': 'v', 'module_subtype': 'nn', 'component_file': 'v.cellml',
                 'component_type': 'v_type', 'module_format': 'cellml',
                 'entrance_ports': [port], 'exit_ports': [],
                 'variables_and_units': [['u', 'J_per_m3', 'access', 'variable']]}
        normalise_module_config([entry], source=f'multi_port {value!r}')


def _function_body(source, name):
    """The text of ``function name`` in ``source``, up to its closing brace."""
    start = re.search(rf'function {name}\b[^{{]*\{{', source)
    if not start:
        return None
    depth, pos = 1, start.end()
    while depth and pos < len(source):
        depth += {'{': 1, '}': -1}.get(source[pos], 0)
        pos += 1
    return source[start.end():pos]


@pytest.mark.unit
def test_the_config_keys_phlynx_exports_are_in_ca_s_schema():
    # moduleConfigEntry builds each module config entry the CA export writes; it is found by name
    # wherever it lives under src/services
    body = None
    for path in _files('src/services/**/*.js'):
        with open(path, encoding='utf-8') as f:
            body = _function_body(f.read(), 'moduleConfigEntry')
        if body:
            break
    assert body, 'moduleConfigEntry is not under src/services any more'
    returned = re.search(r'return\s*\{(.*?)\n  \}', body, re.S)
    assert returned, 'moduleConfigEntry no longer returns an object literal'
    # plain keys (``key: value``) and conditional ones (``...(x && { key })`` or ``...(x && { key: v })``)
    keys = set(re.findall(r'^\s*(\w+)\s*:', returned.group(1), re.M))
    keys |= set(re.findall(r'\.\.\.\([^\n]*?&&\s*\{\s*(\w+)', returned.group(1)))
    assert keys, 'no keys found in moduleConfigEntry'
    allowed = set(load_schema(MODULE_CONFIG_SCHEMA)['$defs']['phlynxComponent']['properties'])
    assert keys <= allowed, sorted(keys - allowed)
