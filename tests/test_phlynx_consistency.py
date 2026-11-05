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


def _constant_values(name):
    """The ``value``s of the option list ``name`` in PhLynx's src/utils/constants.js."""
    with open(os.path.join(_phlynx(), 'src', 'utils', 'constants.js'), encoding='utf-8') as f:
        source = f.read()
    match = re.search(rf'export const {name}\s*=\s*\[(.*?)\]', source, re.S)
    assert match, f'{name} is not in constants.js any more'
    return re.findall(r"value:\s*'([^']*)'", match.group(1))


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
    for value in _constant_values('MULTIPORT_OPTIONS'):
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


@pytest.mark.unit
def test_the_config_keys_phlynx_exports_are_in_ca_s_schema():
    with open(os.path.join(_phlynx(), 'src', 'services', 'export', 'ca.js'),
              encoding='utf-8') as f:
        source = f.read()
    match = re.search(r'function moduleConfigEntry\b.*?return\s*\{(.*?)\n\s*\}', source, re.S)
    assert match, 'moduleConfigEntry is not in src/services/export/ca.js any more'
    keys = set(re.findall(r'^\s*(\w+)\s*:', match.group(1), re.M))
    assert keys, 'no keys found in moduleConfigEntry'
    allowed = set(load_schema(MODULE_CONFIG_SCHEMA)['$defs']['phlynxComponent']['properties'])
    assert keys <= allowed, sorted(keys - allowed)
