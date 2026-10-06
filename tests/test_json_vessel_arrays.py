"""JSON vessel arrays, CSV read through the same records, and supermodules.

A vessel array may be ``<prefix>_vessel_array.json`` -- a list of records in PhLynx keys
(``name, module_type, module_subtype, inp_instances, out_instances``) or libcuflynx keys
(``name, vessel_type, BC_type, inp_vessels, out_vessels``) -- and a CSV array is read by
converting each row to the same record (utilities/config_schemas.py).

A supermodule is a module config entry with ``"module_format": "supermodule"`` and a list of
``submodules``; an instance of it in a vessel array expands into ``<instance>_<sub>`` records
before anything else sees the array (utilities/supermodules.py). Its hosts are linked to
submodules by ``per_submodule_inputs`` / ``per_submodule_outputs``.

The fixture modules are those of test_config_schemas.py.
"""
import copy
import csv
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from test_config_schemas import (FULL_PARAMS, FULL_ROWS, _assert_same_generated_models,
                                 _prescribed_params, _simulate, _write_library, modules_config)

from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
from libcuflynx.utilities.config_schemas import (load_supermodule_registry, load_vessel_array,
                                                 main as config_schemas_main,
                                                 normalise_module_config_entry,
                                                 read_vessel_array_records, vessel_array_path,
                                                 vessel_array_to_json, vessel_records_to_frame)
from libcuflynx.utilities.supermodules import expand_supermodules

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES_DIR = os.path.join(REPO_ROOT, 'resources')


# --------------------------------------------------------------------------------------------
# fixtures: a module library with supermodules
# --------------------------------------------------------------------------------------------

def _rec(name, module_type, inp=(), out=(), subtype='nn', **extra):
    """A PhLynx-key record."""
    record = {'name': name, 'module_type': module_type, 'module_subtype': subtype,
              'inp_instances': list(inp), 'out_instances': list(out)}
    record.update(extra)
    return record


def _to_libcuflynx(record):
    keys = {'module_type': 'vessel_type', 'module_subtype': 'BC_type',
            'inp_instances': 'inp_vessels', 'out_instances': 'out_vessels'}
    return {keys.get(k, k): v for k, v in record.items()}


# "flowpair": a collector fed by the host, and a Multiply gain feeding a host reader
FLOWPAIR = {
    'module_type': 'flowpair', 'module_subtype': 'supermodule', 'module_format': 'supermodule',
    'description': 'a flow collector and a gain, for tests',
    'submodules': [_rec('coll', 'collector'), _rec('g', 'gain')],
    'default_parameters': 'flowpair_parameters.csv',
}
FLOWPAIR_DEFAULTS = [  # (variable_name, units, value)
    ('mean_g', 'm3_per_s', '1.5e-05'), ('amp_g', 'dimensionless', '0.25'),
    ('omega_g', 'per_s', '3.0'), ('some_global', 'dimensionless', '7.0'),
]
# "inner": a gain feeding a reader, internally; "outer" nests an "inner" next to a collector
INNER = {
    'vessel_type': 'inner', 'BC_type': 'supermodule', 'module_format': 'supermodule',
    'submodules': [_rec('g', 'gain', out=['rd']), _rec('rd', 'reader', inp=['g'])],
}
OUTER = {
    'module_type': 'outer', 'module_subtype': 'supermodule', 'module_format': 'supermodule',
    'submodules': [
        _rec('src', 'flow_src', out=['coll']),
        _rec('coll', 'collector', inp=['src']),
        _rec('in', 'inner', subtype='supermodule'),
    ],
}
# a cycle: loop_a contains a loop_b, which contains a loop_a
LOOP_A = {'module_type': 'loop_a', 'module_subtype': 'supermodule', 'module_format': 'supermodule',
          'submodules': [_rec('b', 'loop_b', subtype='supermodule')]}
LOOP_B = {'module_type': 'loop_b', 'module_subtype': 'supermodule', 'module_format': 'supermodule',
          'submodules': [_rec('a', 'loop_a', subtype='supermodule')]}

SUPERMODULES = [FLOWPAIR, INNER, OUTER, LOOP_A, LOOP_B]


@pytest.fixture(scope='module')
def super_library_dir(tmp_path_factory):
    directory = _write_library(tmp_path_factory.mktemp('super_modules'), {
        'schema_test_modules_config.json': modules_config(),
        'super_modules_config.json': SUPERMODULES,
    })
    with open(os.path.join(directory, 'flowpair_parameters.csv'), 'w') as f:
        f.write('variable_name,units,value,data_reference\n')
        for name, units, value in FLOWPAIR_DEFAULTS:
            f.write(f'{name},{units},{value},flowpair_default\n')
    return directory


@pytest.fixture(scope='module')
def registry(super_library_dir):
    return load_supermodule_registry([os.path.join(super_library_dir, 'super_modules_config.json')])


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------

def _write_csv(path, records):
    with open(path, 'w') as f:
        f.write('name,BC_type,vessel_type,inp_vessels,out_vessels\n')
        for r in records:
            f.write(f"{r['name']},{r['module_subtype']},{r['module_type']},"
                    f"{' '.join(r['inp_instances'])},{' '.join(r['out_instances'])}\n")


def _write_array(resources_dir, prefix, records, fmt):
    """Write ``records`` (PhLynx keys) as csv, json (PhLynx keys) or json_libcuflynx."""
    if fmt == 'csv':
        path = os.path.join(resources_dir, f'{prefix}_vessel_array.csv')
        _write_csv(path, records)
        return path
    if fmt == 'json_libcuflynx':
        records = [_to_libcuflynx(r) for r in records]
    path = os.path.join(resources_dir, f'{prefix}_vessel_array.json')
    with open(path, 'w') as f:
        json.dump(records, f, indent=1)
    return path


def _generate(work_dir, modules_dir, prefix, records, params, fmt='json', data_reference='test'):
    """Write the vessel array and parameters and generate the model; returns the CellML path."""
    work_dir = str(work_dir)
    resources_dir = os.path.join(work_dir, 'resources')
    os.makedirs(resources_dir, exist_ok=True)
    _write_array(resources_dir, prefix, records, fmt)
    with open(os.path.join(resources_dir, f'{prefix}_parameters.csv'), 'w') as f:
        f.write('variable_name,units,value,data_reference\n')
        for row in params:
            name, units, value = row[:3]
            reference = row[3] if len(row) > 3 else data_reference
            f.write(f'{name},{units},{value},{reference}\n')
    generated_dir = os.path.join(work_dir, 'generated_models')
    config = {
        'file_prefix': prefix,
        'input_param_file': f'{prefix}_parameters.csv',
        'model_type': 'cellml',
        'solver': 'CVODE_myokit',
        'resources_dir': resources_dir,
        'generated_models_dir': generated_dir,
        'external_modules_dir': modules_dir,
        'DEBUG': False,
    }
    assert generate_with_new_architecture(False, config), f'generation of {prefix} failed'
    return os.path.join(generated_dir, prefix, f'{prefix}.cellml')


def _full_records():
    return [_rec(name, vessel_type, inp, out) for name, vessel_type, inp, out in FULL_ROWS]


def _generated_parameters(cellml_path):
    prefix = os.path.splitext(os.path.basename(cellml_path))[0]
    with open(os.path.join(os.path.dirname(cellml_path), f'{prefix}_parameters.csv')) as f:
        return {row['variable_name']: row for row in csv.DictReader(f)}


def _expand(records, registry):
    from libcuflynx.utilities.config_schemas import normalise_vessel_records
    return expand_supermodules(normalise_vessel_records(records, 'test'), registry, 'test')


def read_records(records):
    from libcuflynx.utilities.config_schemas import normalise_vessel_records
    return normalise_vessel_records(records)


def _links(records):
    return {r['name']: (r['inp_vessels'], r['out_vessels']) for r in records}


# the host model the flowpair instance sits in: two sources feed its collector, its gain
# feeds a reader
def _flowpair_host(instance='pair', per_inputs=None, per_outputs=None):
    return [
        _rec('src_a', 'flow_src', out=[instance]),
        _rec('src_b', 'flow_src', out=[instance]),
        _rec(instance, 'flowpair', subtype='supermodule',
             per_submodule_inputs=per_inputs if per_inputs is not None else {'coll': ['src_a', 'src_b']},
             per_submodule_outputs=per_outputs if per_outputs is not None else {'g': ['rd']}),
        _rec('rd', 'reader', inp=[instance]),
    ]


def _flowpair_flattened(instance='pair'):
    return [
        _rec('src_a', 'flow_src', out=[f'{instance}_coll']),
        _rec('src_b', 'flow_src', out=[f'{instance}_coll']),
        _rec(f'{instance}_coll', 'collector', inp=['src_a', 'src_b']),
        _rec(f'{instance}_g', 'gain', out=['rd']),
        _rec('rd', 'reader', inp=[f'{instance}_g']),
    ]


HOST_PARAMS = (_prescribed_params('src_a', 1.0e-5, 0.5, 6.0) +
               _prescribed_params('src_b', 2.0e-5, 0.3, 4.0))


def _flowpair_defaults(instance):
    return [(name.replace('_g', f'_{instance}_g') if name.endswith('_g') else name, units, value,
             'flowpair_default') for name, units, value in FLOWPAIR_DEFAULTS]


# --------------------------------------------------------------------------------------------
# 1. JSON (both key styles) and CSV
# --------------------------------------------------------------------------------------------

@pytest.mark.integration
def test_json_in_both_key_styles_generates_the_same_model_as_csv(tmp_path, super_library_dir):
    records = _full_records()
    reference = _generate(tmp_path / 'csv', super_library_dir, 'jv_full', records, FULL_PARAMS,
                          fmt='csv')
    for fmt in ('json', 'json_libcuflynx'):
        path = _generate(tmp_path / fmt, super_library_dir, 'jv_full', records, FULL_PARAMS,
                         fmt=fmt)
        _assert_same_generated_models(reference, path)


@pytest.mark.unit
def test_json_and_csv_give_the_same_frame(tmp_path):
    records = _full_records()
    frames = {}
    for fmt in ('csv', 'json', 'json_libcuflynx'):
        directory = tmp_path / fmt
        directory.mkdir()
        frames[fmt] = load_vessel_array(_write_array(str(directory), 'm', records, fmt))[0]
    for fmt in ('json', 'json_libcuflynx'):
        assert frames[fmt].equals(frames['csv'])
    assert list(frames['csv'].columns) == ['name', 'BC_type', 'vessel_type', 'inp_vessels',
                                           'out_vessels']
    assert frames['csv'].iloc[2].tolist() == ['coll', 'nn', 'collector', ['src_a', 'src_b'], []]


@pytest.mark.unit
def test_space_separated_lists_are_accepted_in_json(tmp_path):
    path = tmp_path / 'm_vessel_array.json'
    path.write_text(json.dumps([_rec('a', 'flow_src', out=['b c'])]))
    assert read_vessel_array_records(str(path))[0]['out_vessels'] == ['b', 'c']
    path.write_text(json.dumps([{'name': 'a', 'module_type': 'x', 'module_subtype': 'nn',
                                 'out_instances': 'b  c'}]))
    record = read_vessel_array_records(str(path))[0]
    assert record['out_vessels'] == ['b', 'c'] and record['inp_vessels'] == []


@pytest.mark.unit
@pytest.mark.parametrize('record, message', [
    ({'name': 'a', 'module_type': 'x', 'module_subtype': 'nn', 'out_vessels': []},
     r'record 1 \("a"\) mixes libcuflynx keys \[\'out_vessels\'\] and PhLynx keys'),
    ({'name': 'a', 'vessel_type': 'x', 'BC_type': 'nn', 'module_subtype': 'nn'},
     'mixes libcuflynx keys'),
    ({'name': 'a', 'module_type': 'x'}, r'record 1 \("a"\): "module_subtype" is required'),
    ({'module_type': 'x', 'module_subtype': 'nn'}, r'record 1: "name" is required'),
    ({'name': 'a', 'module_type': 'x', 'module_subtype': 'nn', 'inp_instances': [1]},
     r'"inp_instances" must be a list of names'),
    ({'name': 'a', 'module_type': 'x', 'module_subtype': 'nn',
      'per_submodule_inputs': [{'p': ['h'], 'q': ['h']}]}, r'"per_submodule_inputs"\[0\] must be'),
    ('not a record', r'record 1 is a str'),
])
def test_bad_json_records_are_reported_with_file_index_and_key(tmp_path, record, message):
    path = tmp_path / 'm_vessel_array.json'
    path.write_text(json.dumps([_rec('ok', 'flow_src'), record]))
    with pytest.raises(ValueError, match=message) as info:
        read_vessel_array_records(str(path))
    assert str(path) in str(info.value)


@pytest.mark.unit
def test_vessel_array_file_is_looked_for_json_first(tmp_path):
    names = ['m_module_array.csv', 'm_module_array.json', 'm_vessel_array.csv',
             'm_vessel_array.json']
    assert vessel_array_path(str(tmp_path), 'm').endswith('m_vessel_array.csv')
    for i, name in enumerate(names):
        (tmp_path / name).write_text('')
        if i == 0:
            assert vessel_array_path(str(tmp_path), 'm').endswith(name)
        else:
            # more than one: the first is used, and the others are named in a warning (a CSV
            # edited after `to-json` wrote the JSON beside it would otherwise be ignored silently)
            with pytest.warns(UserWarning, match=f'Using {name}; the others are ignored'):
                assert vessel_array_path(str(tmp_path), 'm').endswith(name)


# --------------------------------------------------------------------------------------------
# 2. CSV -> JSON -> records
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize('style', ['phlynx', 'libcuflynx'])
def test_csv_to_json_round_trip(tmp_path, style):
    csv_path = _write_array(str(tmp_path), 'm', _full_records(), 'csv')
    json_path = vessel_array_to_json(csv_path, str(tmp_path / f'{style}.json'), style=style)
    assert read_vessel_array_records(json_path) == read_vessel_array_records(csv_path)
    with open(json_path) as f:
        text = f.read()
    raw = json.loads(text)
    assert len(text.splitlines()) == len(raw) + 2  # one record per line
    expected_keys = (['name', 'module_type', 'module_subtype', 'inp_instances', 'out_instances']
                     if style == 'phlynx' else
                     ['name', 'BC_type', 'vessel_type', 'inp_vessels', 'out_vessels'])
    assert list(raw[0]) == expected_keys


@pytest.mark.unit
def test_to_json_command_line(tmp_path):
    csv_a = _write_array(str(tmp_path), 'a', _full_records(), 'csv')
    csv_b = _write_array(str(tmp_path), 'b', _full_records()[:3], 'csv')
    assert config_schemas_main(['to-json', csv_a, csv_b, '--style', 'libcuflynx']) == 0
    assert 'vessel_type' in json.loads((tmp_path / 'b_vessel_array.json').read_text())[0]
    result = subprocess.run([sys.executable, '-m', 'libcuflynx.utilities.config_schemas',
                             'to-json', csv_a], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(tmp_path / 'a_vessel_array.json')
    assert 'module_type' in json.loads((tmp_path / 'a_vessel_array.json').read_text())[0]


# --------------------------------------------------------------------------------------------
# 3. supermodules
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_supermodule_entries_are_normalised_and_kept_out_of_the_module_dataframe(
        super_library_dir, registry):
    from libcuflynx.parsers.PrimitiveParsers import JSONFileParser
    entry = normalise_module_config_entry(FLOWPAIR)
    assert (entry['vessel_type'], entry['BC_type']) == ('flowpair', 'supermodule')
    assert entry['submodules'][0]['vessel_type'] == 'collector'
    assert normalise_module_config_entry(INNER)['vessel_type'] == 'inner'
    assert set(registry) == {('flowpair', 'supermodule'), ('inner', 'supermodule'),
                             ('outer', 'supermodule'), ('loop_a', 'supermodule'),
                             ('loop_b', 'supermodule')}
    frame = JSONFileParser().json_files_to_dataframe(
        [os.path.join(super_library_dir, f) for f in ('super_modules_config.json',
                                                      'schema_test_modules_config.json')])
    assert 'flowpair' not in set(frame['vessel_type'])
    assert 'submodules' not in frame.columns


@pytest.mark.unit
@pytest.mark.parametrize('entry, message', [
    (dict(FLOWPAIR, component_file='x.cellml'), 'has "component_file"'),
    (dict(FLOWPAIR, BC_type='supermodule'), 'mixes the libcuflynx keys'),
    (dict(FLOWPAIR, submodules=[]), 'needs "submodules"'),
    (dict(FLOWPAIR, submodules=[_rec('a', 'gain', out=['nowhere'])]), 'not submodules of this'),
    (dict(FLOWPAIR, submodules=[_rec('a', 'gain'), _rec('a', 'reader')]), 'repeated'),
])
def test_bad_supermodule_entries(entry, message):
    with pytest.raises(ValueError, match=message):
        normalise_module_config_entry(entry)


@pytest.mark.unit
def test_expansion_prefixes_and_links_hosts(registry):
    expanded, _ = _expand(_flowpair_host(), registry)
    assert [r['name'] for r in expanded] == ['src_a', 'src_b', 'pair_coll', 'pair_g', 'rd']
    flattened = read_records(_flowpair_flattened())
    assert _links(expanded) == _links(flattened)
    assert vessel_records_to_frame(expanded).equals(vessel_records_to_frame(flattened))


@pytest.mark.unit
def test_host_order_is_kept_and_hosts_are_prepended_and_appended(registry):
    config = copy.deepcopy(registry)
    # a flowpair variant whose collector has an internal input and whose gain an internal output
    config[('flowpair', 'supermodule')]['submodules'] = read_records([
        _rec('coll', 'collector', inp=['g']), _rec('g', 'gain', out=['coll'])])
    host = [
        _rec('h1', 'flow_src', out=['x', 'pair', 'y']),
        _rec('pair', 'flowpair', subtype='supermodule',
             per_submodule_inputs={'g': ['h1'], 'coll': ['h1']},
             per_submodule_outputs={'g': ['h2']}),
        _rec('h2', 'reader', inp=['a', 'pair', 'b']),
        _rec('x', 'reader'), _rec('y', 'reader'), _rec('a', 'gain'), _rec('b', 'gain'),
    ]
    links = _links(_expand(host, config)[0])
    # the instance name is replaced in place, in per_submodule_inputs order
    assert links['h1'][1] == ['x', 'pair_g', 'pair_coll', 'y']
    assert links['h2'][0] == ['a', 'pair_g', 'b']
    # hosts come before the internal inputs, and after the internal outputs
    assert links['pair_coll'] == (['h1', 'pair_g'], [])
    assert links['pair_g'] == (['h1'], ['pair_coll', 'h2'])


@pytest.mark.unit
def test_per_submodule_as_a_dict_equals_a_list_of_dicts(registry):
    as_dict = _flowpair_host(per_inputs={'coll': ['src_a', 'src_b']}, per_outputs={'g': ['rd']})
    as_list = _flowpair_host(per_inputs=[{'coll': ['src_a', 'src_b']}], per_outputs=[{'g': ['rd']}])
    assert _expand(as_dict, registry) == _expand(as_list, registry)
    with pytest.raises(ValueError, match='lists submodule "coll" twice'):
        _expand(_flowpair_host(per_inputs=[{'coll': ['src_a']}, {'coll': ['src_b']}]), registry)


@pytest.mark.integration
def test_supermodule_generates_the_same_model_as_the_flattened_array_and_runs(
        tmp_path, super_library_dir):
    params = HOST_PARAMS + _flowpair_defaults('pair')
    flattened = _generate(tmp_path / 'flat', super_library_dir, 'jv_pair', _flowpair_flattened(),
                          params)
    # the supermodule's default parameters stand in for the flattened model's gain parameters
    expanded = _generate(tmp_path / 'super', super_library_dir, 'jv_pair', _flowpair_host(),
                         HOST_PARAMS)
    _assert_same_generated_models(flattened, expanded)
    results = _simulate(expanded, ['pair_coll/v_seen', 'rd/v_seen'])
    assert results['pair_coll/v_seen'].size > 0 and results['rd/v_seen'].size > 0


@pytest.mark.integration
def test_default_parameters_apply_and_host_values_override_them(tmp_path, super_library_dir):
    host = HOST_PARAMS + [('mean_pair_g', 'm3_per_s', '9e-06', 'host_value')]
    params = _generated_parameters(_generate(tmp_path, super_library_dir, 'jv_defaults',
                                             _flowpair_host(), host))
    assert (params['mean_pair_g']['value'], params['mean_pair_g']['data_reference']) == \
        ('9e-06', 'host_value')
    assert (params['amp_pair_g']['value'], params['amp_pair_g']['data_reference']) == \
        ('0.25', 'flowpair_default')
    assert params['omega_pair_g']['value'] == '3.0'


@pytest.mark.unit
def test_default_parameters_are_renamed_and_globals_added_once(registry):
    host = _flowpair_host('p1') + [
        _rec('src_c', 'flow_src', out=['p2']),
        _rec('p2', 'flowpair', subtype='supermodule', per_submodule_inputs={'coll': ['src_c']}),
    ]
    _, rows = _expand(host, registry)
    names = [row['variable_name'] for row in rows]
    assert names == ['mean_p1_g', 'amp_p1_g', 'omega_p1_g', 'some_global',
                     'mean_p2_g', 'amp_p2_g', 'omega_p2_g']
    assert rows[0] == {'variable_name': 'mean_p1_g', 'units': 'm3_per_s', 'value': '1.5e-05',
                       'data_reference': 'flowpair_default'}


@pytest.mark.unit
def test_default_parameter_suffixes_match_the_longest_submodule_name():
    from libcuflynx.utilities.supermodules import rename_default_parameter
    subs = ['v', 'lv', 'a_lv']
    assert rename_default_parameter('E_lv', 'heart', subs) == 'E_heart_lv'
    assert rename_default_parameter('E_a_lv', 'heart', subs) == 'E_heart_a_lv'
    assert rename_default_parameter('T', 'heart', subs) == 'T'
    assert rename_default_parameter('lv', 'heart', subs) == 'lv'


@pytest.mark.unit
def test_merge_default_parameters_keeps_host_rows():
    import numpy as np
    from libcuflynx.parsers.ModelParsers import merge_default_parameters
    dtype = [(c, '<U80') for c in ('variable_name', 'units', 'value', 'data_reference')]
    host = np.array([('a', 'm', '1', 'host')], dtype=dtype)
    merged = merge_default_parameters(host, [
        {'variable_name': 'a', 'units': 'm', 'value': '2', 'data_reference': 'default'},
        {'variable_name': 'b', 'units': 'm', 'value': '3', 'data_reference': 'default'}])
    assert merged.tolist() == [('a', 'm', '1', 'host'), ('b', 'm', '3', 'default')]


@pytest.mark.unit
@pytest.mark.parametrize('change, message', [
    # an unknown submodule in per_submodule_*
    (lambda h: h[2].update(per_submodule_inputs={'nope': ['src_a', 'src_b']}),
     r'per_submodule_inputs names \[\'nope\'\], which are not submodules'),
    # a host names the instance but no per_submodule entry links them
    (lambda h: h[2].update(per_submodule_inputs={'coll': ['src_a']}),
     r'"src_b" lists "pair" in its out list, but no per_submodule_inputs'),
    (lambda h: h[2].update(per_submodule_outputs={}),
     r'"rd" lists "pair" in its inp list, but no per_submodule_outputs'),
    # a per_submodule host that does not name the instance back
    (lambda h: h[0].update(out_instances=[]),
     r'per_submodule_inputs\["coll"\] names host "src_a", but "src_a" does not list "pair"'),
    (lambda h: h[3].update(inp_instances=[]),
     r'per_submodule_outputs\["g"\] names host "rd", but "rd" does not list "pair"'),
    # a host that does not exist
    (lambda h: h[2]['per_submodule_outputs'].update(coll=['ghost']),
     r'names host "ghost", which is not in the vessel array'),
    # an expanded name that clashes with an existing record
    (lambda h: h.append(_rec('pair_g', 'reader')), r"gives the names \['pair_g'\], which are"),
    # an unknown supermodule type
    (lambda h: h[2].update(module_type='nosuch'),
     r'"pair" is a supermodule instance of type \(nosuch, supermodule\), but no supermodule'),
    (lambda h: h[2].update(module_subtype='other'),
     r'type \(flowpair, other\), but no supermodule entry'),
])
def test_expansion_errors(registry, change, message):
    host = _flowpair_host()
    change(host)
    with pytest.raises(ValueError, match=message) as info:
        _expand(host, registry)
    assert str(info.value).startswith('test: ')


@pytest.mark.unit
def test_nested_supermodules_expand_recursively(registry):
    host = [_rec('o', 'outer', subtype='supermodule')]
    expanded, _ = _expand(host, registry)
    assert _links(expanded) == {
        'o_src': ([], ['o_coll']),
        'o_coll': (['o_src'], []),
        'o_in_g': ([], ['o_in_rd']),
        'o_in_rd': (['o_in_g'], []),
    }


@pytest.mark.unit
def test_nested_supermodule_links_through_its_own_per_submodule(registry):
    config = copy.deepcopy(registry)
    config[('outer', 'supermodule')]['submodules'] = read_records([
        _rec('src', 'flow_src', out=['in']),
        _rec('in', 'inner', subtype='supermodule', per_submodule_inputs={'rd': ['src']}),
    ])
    links = _links(_expand([_rec('o', 'outer', subtype='supermodule')], config)[0])
    assert links['o_src'] == ([], ['o_in_rd'])
    assert links['o_in_rd'] == (['o_src', 'o_in_g'], [])


@pytest.mark.integration
def test_nested_supermodule_generates_the_same_model_as_the_flattened_array(
        tmp_path, super_library_dir):
    gain = _prescribed_params('o_in_g', 1.0e-5, 0.5, 3.0)
    src = _prescribed_params('o_src', 2.0e-5, 0.4, 5.0)
    flattened = _generate(tmp_path / 'flat', super_library_dir, 'jv_nested', [
        _rec('o_src', 'flow_src', out=['o_coll']),
        _rec('o_coll', 'collector', inp=['o_src']),
        _rec('o_in_g', 'gain', out=['o_in_rd']),
        _rec('o_in_rd', 'reader', inp=['o_in_g']),
    ], gain + src)
    nested = _generate(tmp_path / 'nested', super_library_dir, 'jv_nested',
                       [_rec('o', 'outer', subtype='supermodule')], gain + src)
    _assert_same_generated_models(flattened, nested)


@pytest.mark.unit
def test_nesting_cycles_are_detected(registry):
    with pytest.raises(ValueError, match=r'nest in a cycle \(loop_a/supermodule -> '
                                         r'loop_b/supermodule -> loop_a/supermodule\)'):
        _expand([_rec('x', 'loop_a', subtype='supermodule')], registry)


@pytest.mark.integration
def test_an_instance_named_heart_does_not_trigger_the_heart_special_cases(
        tmp_path, super_library_dir):
    """The parser adds pulmonary vessels to a vessel named 'heart' with fewer than two
    outputs, and the CellML generator checks a 'heart' vessel's venous inputs. Neither may
    fire for a supermodule instance named 'heart', which is gone after expansion."""
    params = HOST_PARAMS + _flowpair_defaults('heart')
    flattened = _generate(tmp_path / 'flat', super_library_dir, 'jv_heart',
                          _flowpair_flattened('heart'), params)
    expanded = _generate(tmp_path / 'super', super_library_dir, 'jv_heart',
                         _flowpair_host('heart'), HOST_PARAMS)
    _assert_same_generated_models(flattened, expanded)
    with open(expanded) as f:
        text = f.read()
    assert 'name="par"' not in text and 'name="pvn"' not in text


@pytest.mark.unit
def test_load_model_expands_before_the_heart_block(tmp_path, super_library_dir):
    from libcuflynx.parsers.ModelParsers import CSV0DModelParser
    resources = tmp_path / 'resources'
    resources.mkdir()
    _write_array(str(resources), 'm', _flowpair_host('heart'), 'json')
    (resources / 'm_parameters.csv').write_text(
        'variable_name,units,value,data_reference\n' +
        ''.join(f'{n},{u},{v},t\n' for n, u, v in HOST_PARAMS))
    parser = CSV0DModelParser({
        'vessels_csv_abs_path': str(resources / 'm_vessel_array.json'),
        'parameters_csv_abs_path': str(resources / 'm_parameters.csv'),
        'model_type': 'cellml', 'external_modules_dir': super_library_dir})
    model = parser.load_model()
    assert list(model.vessels_df['name']) == ['src_a', 'src_b', 'heart_coll', 'heart_g', 'rd']


# --------------------------------------------------------------------------------------------
# 4. every resources/ vessel array, as JSON
# --------------------------------------------------------------------------------------------

RESOURCE_PREFIXES = sorted(os.path.basename(p)[:-len('_vessel_array.csv')]
                           for p in glob.glob(os.path.join(RESOURCES_DIR, '*_vessel_array.csv')))
# 1D/0D split intermediates that tests write into resources/ (gitignored), not inputs
RESOURCE_PREFIXES = [p for p in RESOURCE_PREFIXES if not p.endswith(('_0d', '_1d'))
                     or not os.path.exists(os.path.join(RESOURCES_DIR, p[:-3] + '_vessel_array.csv'))]


def _generate_resource(work_dir, prefix, fmt):
    resources = os.path.join(str(work_dir), 'resources')
    os.makedirs(resources, exist_ok=True)
    shutil.copy(os.path.join(RESOURCES_DIR, f'{prefix}_parameters.csv'), resources)
    csv_path = os.path.join(RESOURCES_DIR, f'{prefix}_vessel_array.csv')
    if fmt == 'csv':
        shutil.copy(csv_path, resources)
    else:
        vessel_array_to_json(csv_path, os.path.join(resources, f'{prefix}_vessel_array.json'))
    config = {'file_prefix': prefix, 'input_param_file': f'{prefix}_parameters.csv',
              'model_type': 'cellml', 'solver': 'CVODE_myokit', 'resources_dir': resources,
              'generated_models_dir': os.path.join(str(work_dir), 'generated_models'),
              'DEBUG': False}
    try:
        ok = generate_with_new_architecture(False, config)
    except (Exception, SystemExit):
        ok = False
    return ok, os.path.join(str(work_dir), 'generated_models', prefix, f'{prefix}.cellml')


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize('prefix', RESOURCE_PREFIXES)
def test_every_resources_vessel_array_generates_identically_as_json(tmp_path, prefix):
    if not os.path.exists(os.path.join(RESOURCES_DIR, f'{prefix}_parameters.csv')):
        pytest.skip(f'{prefix} has no parameters file')
    ok, reference = _generate_resource(tmp_path / 'csv', prefix, 'csv')
    if not ok:
        pytest.skip(f'{prefix} does not generate from its CSV either')
    ok, converted = _generate_resource(tmp_path / 'json', prefix, 'json')
    assert ok, f'{prefix} generates from CSV but not from JSON'
    _assert_same_generated_models(reference, converted)


@pytest.mark.integration
def test_a_heart_inside_a_supermodule_is_still_the_heart(tmp_path):
    """The generators found the monolithic heart by its name, "heart", which a heart inside a
    supermodule (here "H_heart") cannot have: its ivc input was mapped to zero flow in a
    component "heart_module" that did not exist, and generation still reported success. It is
    now found by its vessel_type, and simulates exactly as the plain model."""
    library = tmp_path / 'lib'
    library.mkdir()
    with open(library / 'cardio_modules_config.json', 'w') as f:
        json.dump([{'module_type': 'cardio', 'module_subtype': 'supermodule',
                    'module_format': 'supermodule',
                    'submodules': [{'name': 'heart', 'module_type': 'heart', 'module_subtype': 'vp_Ca',
                                    'inp_instances': [], 'out_instances': []}]}], f)
    from libcuflynx.utilities.config_schemas import vessel_array_csv_to_records
    plain = vessel_array_csv_to_records(os.path.join(RESOURCES_DIR, '3compartment_vessel_array.csv'))
    params = [tuple(r) for r in pd.read_csv(os.path.join(RESOURCES_DIR, '3compartment_parameters.csv'),
                                            dtype=str).fillna('')[['variable_name', 'units', 'value']]
              .itertuples(index=False)]
    reference = _generate(tmp_path / 'plain', None, '3compartment', plain, params)

    wrapped = []
    for record in plain:
        record = dict(record)
        if record['name'] == 'heart':
            record.update(name='H', vessel_type='cardio', BC_type='supermodule',
                          per_submodule_inputs={'heart': record['inp_vessels']},
                          per_submodule_outputs={'heart': record['out_vessels']})
        else:
            record['inp_vessels'] = ['H' if n == 'heart' else n for n in record['inp_vessels']]
            record['out_vessels'] = ['H' if n == 'heart' else n for n in record['out_vessels']]
        wrapped.append(record)
    renamed = [(n[:-len('_heart')] + '_H_heart' if n.endswith('_heart') else n, u, v) for n, u, v in params]
    model = _generate(tmp_path / 'wrapped', str(library), '3compartment', wrapped, renamed)
    with open(model) as f:
        text = f.read()
    assert 'component_2="H_heart_module"' in text or 'component_1="H_heart_module"' in text
    assert '"heart_module"' not in text
    ref = _simulate(reference, ['aortic_root/u', 'heart/u_lv'], sim_time=1.0)
    new = _simulate(model, ['aortic_root/u', 'H_heart/u_lv'], sim_time=1.0)
    assert np.allclose(ref['aortic_root/u'], new['aortic_root/u'], rtol=1e-6)
    assert np.allclose(ref['heart/u_lv'], new['H_heart/u_lv'], rtol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize('style', ['phlynx', 'libcuflynx'])
def test_convert_0d_to_1d_keeps_a_json_arrays_supermodule_links(tmp_path, registry, style):
    """A CSV cannot hold per_submodule_* links, so a JSON 0D array is converted to a JSON
    hybrid array, in the same key style, with every key of every record kept."""
    from libcuflynx.scripts.convert_0d_to_1d import convert_0d_to_1d
    from libcuflynx.utilities.config_schemas import dump_vessel_records
    records = read_records(_flowpair_host() + [_rec('A', 'flow_src', out=[])])
    (tmp_path / 'm_0d_vessel_array.json').write_text(dump_vessel_records(records, style))
    (tmp_path / 'm_0d_parameters.csv').write_text('variable_name,units,value,data_reference\n')
    convert_0d_to_1d('m', str(tmp_path), 'm_0d_parameters.csv', vess_1d_list=['A'])
    hybrid = tmp_path / 'm_hybrid_vessel_array.json'
    assert hybrid.exists() and not (tmp_path / 'm_hybrid_vessel_array.csv').exists()
    raw = json.loads(hybrid.read_text())
    assert ('module_type' in raw[0]) == (style == 'phlynx')
    converted = {r['name']: r for r in read_vessel_array_records(str(hybrid))}
    assert (converted['A']['vessel_type'], converted['A']['BC_type']) == ('FV1D_vessel', 'nn')
    assert converted['pair']['per_submodule_inputs'] == {'coll': ['src_a', 'src_b']}
    assert converted['pair']['per_submodule_outputs'] == {'g': ['rd']}
    expanded, _ = expand_supermodules(list(converted.values()), registry, 'test')
    assert 'pair_coll' in {r['name'] for r in expanded}


@pytest.mark.unit
def test_the_1d_generator_reads_the_merged_parameters(tmp_path):
    """The 1D generator read the parameters file again, so a 1D vessel's parameter set by a
    supermodule's default_parameters never reached it ("Parameter l_FV1D_0 not found"). It
    now reads the model's merged parameters."""
    from types import SimpleNamespace
    from libcuflynx.generators.CVSCppGenerator import CVS1DPythonGenerator
    from libcuflynx.parsers.ModelParsers import merge_default_parameters
    from libcuflynx.parsers.PrimitiveParsers import CSVFileParser
    params = tmp_path / 'm_parameters.csv'
    params.write_text('variable_name,units,value,data_reference\nr_FV1D_0,metre,0.01,file\n')
    vessels = tmp_path / 'm_1d_vessel_array.csv'
    vessels.write_text('name,BC_type,vessel_type,inp_vessels,out_vessels\nFV1D_0,nn,FV1D_vessel,,\n')
    merged = merge_default_parameters(
        CSVFileParser().get_data_as_nparray(str(params), True),
        [{'variable_name': 'l_FV1D_0', 'units': 'metre', 'value': '0.2', 'data_reference': 'default'}])
    model = SimpleNamespace(all_parameters_array=merged)
    generator = CVS1DPythonGenerator(model, 'm_1d', str(vessels), str(params),
                                     str(tmp_path / 'run1d' / 'input.ini'), str(tmp_path / 'gen'))
    values = dict(zip(generator.params_df['variable_name'], generator.params_df['value']))
    assert values == {'r_FV1D_0': '0.01', 'l_FV1D_0': '0.2'}


@pytest.mark.unit
def test_default_parameters_reach_a_nested_supermodules_submodules(tmp_path, registry):
    """A supermodule's default_parameters row naming a submodule of a nested supermodule
    (mean_in_g: submodule g of the inner supermodule in) was treated as a global and never
    reached it."""
    defaults = tmp_path / 'outer_parameters.csv'
    defaults.write_text('variable_name,units,value,data_reference\n'
                        'mean_in_g,m3_per_s,2e-05,outer\nmean_coll,m3_per_s,1e-05,outer\n'
                        'some_global,dimensionless,3,outer\n')
    outer = dict(registry[('outer', 'supermodule')], default_parameters=str(defaults))
    _, rows = _expand([_rec('o', 'outer', subtype='supermodule')],
                      {**registry, ('outer', 'supermodule'): outer})
    names = {r['variable_name'] for r in rows}
    assert {'mean_o_in_g', 'mean_o_coll', 'some_global'} <= names
    assert 'mean_in_g' not in names


@pytest.mark.unit
def test_a_host_linked_twice_is_an_error(registry):
    """A host listing an instance twice, or a per_submodule_* list naming a host twice, gave
    out and inp lists that no longer matched (src -> [S_coll, S_g, S_coll, S_g] against
    S_coll <- [src])."""
    twice = _flowpair_host()
    twice[0]['out_instances'] = ['pair', 'pair']
    with pytest.raises(ValueError, match='"src_a" lists "pair" more than once in its out list'):
        _expand(twice, registry)
    with pytest.raises(ValueError, match=r'per_submodule_inputs\["coll"\] names \[\'src_a\'\] more than once'):
        _expand(_flowpair_host(per_inputs={'coll': ['src_a', 'src_a', 'src_b']}), registry)


@pytest.mark.unit
def test_a_bad_submodule_record_is_reported_in_its_module_config():
    with pytest.raises(ValueError) as error:
        normalise_module_config_entry({
            'module_type': 'x', 'module_subtype': 's', 'module_format': 'supermodule',
            'submodules': [{'name': 'I_p', 'module_type': 'inertance'}]},
            source='lib/x_modules_config.json')
    message = str(error.value)
    assert message.startswith('supermodule entry (x, s) in lib/x_modules_config.json, submodules, '
                              'record 0 ("I_p")')
    assert 'vessel array' not in message


def _component_entry(**port):
    return {'vessel_type': 'v', 'BC_type': 'nn', 'module_file': 'v.cellml', 'module_type': 'v_type',
            'entrance_ports': [dict({'port_type': 'p', 'variables': ['x']}, **port)],
            'exit_ports': [], 'variables_and_units': [['x', 'm', 'access', 'variable']]}


@pytest.mark.unit
@pytest.mark.parametrize('value', ['bogus', 1, 2.5, {'a': 1}])
def test_an_unknown_multi_port_is_an_error_in_the_code_and_the_schema(value):
    jsonschema = pytest.importorskip('jsonschema')
    from libcuflynx.schemas import MODULE_CONFIG_SCHEMA, load_schema
    entry = _component_entry(multi_port=value)
    with pytest.raises(ValueError, match='has multi_port'):
        normalise_module_config_entry(entry)
    assert not jsonschema.Draft202012Validator(load_schema(MODULE_CONFIG_SCHEMA)).is_valid([entry])


@pytest.mark.unit
@pytest.mark.parametrize('value, expected', [(None, None), (False, None), ('false', None),
                                             (True, 'True'), ('true', 'True'), ('True', 'True')])
def test_multi_port_true_and_false_in_any_form(value, expected):
    jsonschema = pytest.importorskip('jsonschema')
    from libcuflynx.schemas import MODULE_CONFIG_SCHEMA, load_schema
    entry = _component_entry(multi_port=value)
    port = normalise_module_config_entry(entry)['entrance_ports'][0]
    assert port.get('multi_port') == expected
    assert jsonschema.Draft202012Validator(load_schema(MODULE_CONFIG_SCHEMA)).is_valid([entry])


@pytest.mark.unit
def test_a_libcuflynx_component_entry_needs_its_cellml_file():
    entry = _component_entry()
    del entry['module_file']
    with pytest.raises(ValueError, match=r"is missing \['module_file'\]"):
        normalise_module_config_entry(entry)


@pytest.mark.unit
def test_a_null_connection_list_is_empty_in_the_code_and_the_schema(tmp_path):
    jsonschema = pytest.importorskip('jsonschema')
    from libcuflynx.schemas import VESSEL_ARRAY_SCHEMA, load_schema
    records = [{'name': 'a', 'vessel_type': 'x', 'BC_type': 'nn', 'inp_vessels': None, 'out_vessels': None}]
    assert jsonschema.Draft202012Validator(load_schema(VESSEL_ARRAY_SCHEMA)).is_valid(records)
    path = tmp_path / 'm_vessel_array.json'
    path.write_text(json.dumps(records))
    record = read_vessel_array_records(str(path))[0]
    assert record['inp_vessels'] == [] and record['out_vessels'] == []


def _two_flowpairs_in_a_row():
    """src -> S1 (collector) ... S1 (gain) -> S2 (collector) ... S2 (gain) -> rd."""
    return [
        _rec('src', 'flow_src', out=['S1']),
        _rec('S1', 'flowpair', subtype='supermodule', inp=['src'], out=['S2'],
             per_submodule_inputs={'coll': ['src']}, per_submodule_outputs={'g': ['S2']}),
        _rec('S2', 'flowpair', subtype='supermodule', inp=['S1'], out=['rd'],
             per_submodule_inputs={'coll': ['S1']}, per_submodule_outputs={'g': ['rd']}),
        _rec('rd', 'reader', inp=['S2']),
    ]


@pytest.mark.unit
@pytest.mark.parametrize('order', [[0, 1, 2, 3], [0, 2, 1, 3]])
def test_two_supermodule_instances_link_to_each_other_in_either_order(registry, order):
    """S2's per_submodule_inputs names the instance S1, and S1's per_submodule_outputs names
    S2. Expanding one used to leave the other naming an instance that no longer existed."""
    records = _two_flowpairs_in_a_row()
    expanded, _ = _expand([records[i] for i in order], registry)
    links = _links(expanded)
    assert links['src'][1] == ['S1_coll']
    assert links['S1_coll'][0] == ['src']
    assert links['S1_g'][1] == ['S2_coll']
    assert links['S2_coll'][0] == ['S1_g']
    assert links['S2_g'][1] == ['rd']
    assert links['rd'][0] == ['S2_g']


@pytest.mark.unit
def test_two_sibling_supermodule_instances_inside_a_supermodule_link(registry):
    """The same, one level down: two flowpairs as submodules of another supermodule."""
    chain = normalise_module_config_entry({
        'module_type': 'chain', 'module_subtype': 'supermodule', 'module_format': 'supermodule',
        'submodules': [
            _rec('src', 'flow_src', out=['p']),
            _rec('p', 'flowpair', subtype='supermodule', inp=['src'], out=['q'],
                 per_submodule_inputs={'coll': ['src']}, per_submodule_outputs={'g': ['q']}),
            _rec('q', 'flowpair', subtype='supermodule', inp=['p'], out=['rd'],
                 per_submodule_inputs={'coll': ['p']}, per_submodule_outputs={'g': ['rd']}),
            _rec('rd', 'reader', inp=['q']),
        ]})
    expanded, _ = _expand([_rec('c', 'chain', subtype='supermodule')],
                          {**registry, ('chain', 'supermodule'): chain})
    links = _links(expanded)
    assert links['c_p_g'][1] == ['c_q_coll']
    assert links['c_q_coll'][0] == ['c_p_g']
    assert links['c_q_g'][1] == ['c_rd']


@pytest.mark.unit
@pytest.mark.parametrize('record, message', [
    ({"name": "a b", "vessel_type": "heart", "BC_type": "vp"}, '"name" \'a b\' contains whitespace'),
    ({"name": "a", "vessel_type": "he art", "BC_type": "vp"}, '"vessel_type" \'he art\' contains whitespace'),
    ({"name": "a", "module_type": "heart", "module_subtype": "v p"},
     '"module_subtype" \'v p\' contains whitespace'),
])
def test_a_name_with_whitespace_is_an_error(record, message):
    """The generator's frame keeps a cell's first word only, so "a b" used to become "a" there
    while expansion and the connection lists kept "a b"."""
    from libcuflynx.utilities.config_schemas import normalise_vessel_record
    with pytest.raises(ValueError, match=message):
        normalise_vessel_record(record)


@pytest.mark.unit
def test_a_space_separated_string_is_still_a_list_of_names():
    from libcuflynx.utilities.config_schemas import normalise_vessel_record
    record = normalise_vessel_record({"name": "a", "vessel_type": "heart", "BC_type": "vp",
                                      "inp_vessels": "b  c"})
    assert record['inp_vessels'] == ['b', 'c']


@pytest.mark.unit
def test_the_schema_rejects_a_name_with_whitespace():
    jsonschema = pytest.importorskip('jsonschema')
    from libcuflynx.schemas import VESSEL_ARRAY_SCHEMA, load_schema
    validator = jsonschema.Draft202012Validator(load_schema(VESSEL_ARRAY_SCHEMA))
    assert not validator.is_valid([{"name": "a b", "vessel_type": "heart", "BC_type": "vp"}])
    assert validator.is_valid([{"name": "a", "vessel_type": "heart", "BC_type": "vp",
                                "inp_vessels": "b c"}])


@pytest.mark.integration
def test_extra_keys_in_a_json_array_do_not_change_the_model(tmp_path):
    """Extra keys become extra columns of the generator's frame. The rows libcuflynx appends
    itself (here the heart's pulmonary circuit, for a heart with one output) were fixed
    5-element lists, so any extra key failed with "cannot set a row with mismatched columns"."""
    prefix = 'simple_physiological'
    ok, reference = _generate_resource(tmp_path / 'csv', prefix, 'csv')
    assert ok
    resources = tmp_path / 'json' / 'resources'
    resources.mkdir(parents=True)
    shutil.copy(os.path.join(RESOURCES_DIR, f'{prefix}_parameters.csv'), resources)
    json_path = vessel_array_to_json(os.path.join(RESOURCES_DIR, f'{prefix}_vessel_array.csv'),
                                     str(resources / f'{prefix}_vessel_array.json'))
    with open(json_path) as f:
        records = json.load(f)
    for i, record in enumerate(records):
        record['label'] = f'vessel {i}'
    with open(json_path, 'w') as f:
        json.dump(records, f)
    config = {'file_prefix': prefix, 'input_param_file': f'{prefix}_parameters.csv',
              'model_type': 'cellml', 'solver': 'CVODE_myokit', 'resources_dir': str(resources),
              'generated_models_dir': str(tmp_path / 'json' / 'generated_models'), 'DEBUG': False}
    assert generate_with_new_architecture(False, config)
    _assert_same_generated_models(
        reference, str(tmp_path / 'json' / 'generated_models' / prefix / f'{prefix}.cellml'))


@pytest.mark.unit
def test_resources_vessel_arrays_and_module_configs_validate_against_the_json_schemas(
        tmp_path, super_library_dir):
    jsonschema = pytest.importorskip('jsonschema')
    from libcuflynx.schemas import MODULE_CONFIG_SCHEMA, VESSEL_ARRAY_SCHEMA, load_schema
    from libcuflynx.utilities.package_resources import builtin_modules_dir
    vessel_schema = load_schema(VESSEL_ARRAY_SCHEMA)
    module_schema = load_schema(MODULE_CONFIG_SCHEMA)
    for schema in (vessel_schema, module_schema):
        jsonschema.Draft202012Validator.check_schema(schema)
    vessel_validator = jsonschema.Draft202012Validator(vessel_schema)
    module_validator = jsonschema.Draft202012Validator(module_schema)

    for prefix in RESOURCE_PREFIXES:
        csv_path = os.path.join(RESOURCES_DIR, f'{prefix}_vessel_array.csv')
        for style in ('phlynx', 'libcuflynx'):
            json_path = vessel_array_to_json(csv_path, str(tmp_path / f'{prefix}_{style}.json'),
                                             style=style)
            with open(json_path) as f:
                vessel_validator.validate(json.load(f))
    vessel_validator.validate(_flowpair_host())
    vessel_validator.validate([_to_libcuflynx(r) for r in _flowpair_host()])
    vessel_validator.validate(_flowpair_host(per_inputs=[{'coll': ['src_a', 'src_b']}]))

    config_files = (glob.glob(os.path.join(builtin_modules_dir(), '*.json')) +
                    glob.glob(os.path.join(REPO_ROOT, 'module_config_user', '*.json')) +
                    glob.glob(os.path.join(super_library_dir, '*.json')))
    assert len(config_files) > 20
    for path in config_files:
        with open(path, encoding='utf-8-sig') as f:
            module_validator.validate(json.load(f))
    phlynx_component = {'module_type': 'a', 'module_subtype': 'nn', 'component_file': 'f.cellml',
                        'component_type': 'a_type', 'entrance_ports': [], 'exit_ports': [],
                        'variables_and_units': []}
    module_validator.validate([phlynx_component])

    # what the loaders reject, the schemas reject too
    assert not vessel_validator.is_valid([{'name': 'a', 'module_type': 'x', 'BC_type': 'nn'}])
    assert not vessel_validator.is_valid([{'name': 'a', 'module_type': 'x', 'module_subtype': 'nn',
                                           'inp_vessels': []}])
    assert not vessel_validator.is_valid([{'name': 'a', 'vessel_type': 'x', 'BC_type': 'nn',
                                           'out_instances': []}])
    assert not vessel_validator.is_valid([{'name': 'a', 'module_type': 'x'}])
    assert not module_validator.is_valid([dict(FLOWPAIR, component_file='x.cellml')])
    assert not module_validator.is_valid([dict(phlynx_component, BC_type='nn')])
    assert not module_validator.is_valid([dict(FLOWPAIR, submodules=[])])
