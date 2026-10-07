"""Module versions and instances (utilities/module_instances.py).

A module library lays a module version out as ``<module_type>/versions/<version>/`` with its
CellML, a one-entry ``*_modules_config.json`` (module_subtype == version) and named parameter
sets in ``instances/<instance>/<instance>_parameters.csv``. A module-array record (or a
supermodule's submodule) picks one with ``"instance"``; without it, the entry's
``default_instance`` is used when its file exists. Rows ``{var}`` become ``{var}_{vessel}``,
except the module's global constants, and are default parameters: the host parameters file
wins, then a supermodule's instance, its default_parameters, then submodule instances.

obs_data files may carry a top-level ``obs_data_name``; one filed under ``instances/<name>/``
is warned about when the name differs.

The fixture library: ``pulse_src`` (a prescribed flow; ``omega`` is a global constant) in
version ``v1`` with instances ``default`` and ``other``, a version ``v2`` without instances,
and a supermodule ``pair`` of two pulse sources with instances ``base`` and ``alt``. The
reader modules are those of test_config_schemas.py.
"""
import json
import os
import shutil
import warnings

import pytest

from test_config_schemas import (_assert_same_generated_models, _prescribed, _write_library,
                                 modules_config)

from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
from libcuflynx.utilities.config_schemas import (load_component_registry,
                                                 load_expanded_vessel_records,
                                                 load_supermodule_registry,
                                                 normalise_module_config_entry,
                                                 normalise_vessel_record)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES_DIR = os.path.join(REPO_ROOT, 'resources')

PULSE_CELLML = (
    "<?xml version='1.0' encoding='UTF-8'?>\n"
    '<model name="pulse_src_v1_modules" xmlns="http://www.cellml.org/cellml/1.1#" '
    'xmlns:cellml="http://www.cellml.org/cellml/1.1#">\n'
    + _prescribed('pulse_src_type', 'v', 'm3_per_s')
    + '</model>\n'
)


def _pulse_entry(version, **extra):
    entry = {
        'module_type': 'pulse_src', 'module_subtype': version,
        'component_file': f'pulse_src_{version}_modules.cellml',
        'component_type': 'pulse_src_type', 'module_format': 'cellml',
        'entrance_ports': [], 'exit_ports': [{'port_type': 'flow_port', 'variables': ['v']}],
        'general_ports': [],
        'variables_and_units': [['v', 'm3_per_s', 'access', 'variable'],
                                ['mean', 'm3_per_s', 'access', 'constant'],
                                ['amp', 'dimensionless', 'access', 'constant'],
                                ['omega', 'per_s', 'access', 'global_constant']],
    }
    entry.update(extra)
    return entry


def _sub(name, instance=None):
    record = {'name': name, 'module_type': 'pulse_src', 'module_subtype': 'v1',
              'inp_instances': [], 'out_instances': []}
    if instance is not None:
        record['instance'] = instance
    return record


PAIR = {'module_type': 'pair', 'module_subtype': 'v1', 'module_format': 'supermodule',
        'submodules': [_sub('a', instance='other'), _sub('b')],
        'default_instance': 'base', 'default_parameters': 'pair_defaults.csv'}

# instance -> rows (variable_name, units, value)
PULSE_INSTANCES = {
    'default': [('mean', 'm3_per_s', '1e-05'), ('amp', 'dimensionless', '0.5'),
                ('omega', 'per_s', '6.0')],
    'other': [('mean', 'm3_per_s', '2e-05'), ('amp', 'dimensionless', '0.3'),
              ('omega', 'per_s', '4.0')],
}
PAIR_INSTANCES = {
    'base': [('amp_a', 'dimensionless', '0.9'), ('omega', 'per_s', '5.0')],
    'alt': [('amp_b', 'dimensionless', '0.1')],
}
PAIR_DEFAULTS = [('amp_a', 'dimensionless', '0.77'), ('mean_b', 'm3_per_s', '3e-05')]


def _write_parameters(path, rows, reference, sourced=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('variable_name,units,value,data_reference' + (',sourced' if sourced else '') + '\n')
        for name, units, value in rows:
            f.write(f'{name},{units},{value},{reference}' + (',yes' if sourced else '') + '\n')


def _write_instances(version_dir, instances):
    for instance, rows in instances.items():
        _write_parameters(os.path.join(version_dir, 'instances', instance,
                                       f'{instance}_parameters.csv'),
                          rows, f'{instance}_ref', sourced=instance == 'other')


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    """``(module_library_dir, external_modules_dir)``"""
    root = tmp_path_factory.mktemp('instance_library')
    modules = os.path.join(str(root), 'modules')
    for version, extra in (('v1', {'default_instance': 'default'}),
                           ('v2', {'default_instance': 'missing'})):
        version_dir = os.path.join(modules, 'sources', 'pulse_src', 'versions', version)
        os.makedirs(version_dir)
        with open(os.path.join(version_dir, f'pulse_src_{version}_modules.cellml'), 'w') as f:
            f.write(PULSE_CELLML.replace('pulse_src_v1_modules', f'pulse_src_{version}_modules'))
        _write_json(os.path.join(version_dir, f'pulse_src_{version}_modules_config.json'),
                    [_pulse_entry(version, **extra)])
    _write_instances(os.path.join(modules, 'sources', 'pulse_src', 'versions', 'v1'),
                     PULSE_INSTANCES)
    pair_dir = os.path.join(modules, 'systems', 'pair', 'versions', 'v1')
    _write_json(os.path.join(pair_dir, 'pair_v1_modules_config.json'), [PAIR])
    _write_parameters(os.path.join(pair_dir, 'pair_defaults.csv'), PAIR_DEFAULTS, 'pair_defaults')
    _write_instances(pair_dir, PAIR_INSTANCES)
    readers = _write_library(str(root / 'readers'),
                             {'schema_test_modules_config.json': modules_config()})
    return modules, readers


def _config_files(library):
    modules, readers = library
    found = []
    for directory, _, files in os.walk(modules):
        found += [os.path.join(directory, f) for f in sorted(files)
                  if f.endswith('_modules_config.json')]
    return sorted(found) + [os.path.join(readers, 'schema_test_modules_config.json')]


def _rec(name, module_type, subtype='nn', inp=(), out=(), **extra):
    record = {'name': name, 'module_type': module_type, 'module_subtype': subtype,
              'inp_instances': list(inp), 'out_instances': list(out)}
    record.update(extra)
    return record


def _load(tmp_path, library, records):
    path = str(tmp_path / 'm_module_array.json')
    _write_json(path, records)
    files = _config_files(library)
    return load_expanded_vessel_records(path, load_supermodule_registry(files),
                                        load_component_registry(files))


def _values(rows):
    return {row['variable_name']: (row['value'], row['data_reference']) for row in rows}


def _two_sources(first_instance='other', second_instance=None):
    s1 = _rec('s1', 'pulse_src', 'v1', out=['coll'])
    s2 = _rec('s2', 'pulse_src', 'v1', out=['coll'])
    if first_instance is not None:
        s1['instance'] = first_instance
    if second_instance is not None:
        s2['instance'] = second_instance
    return [s1, s2, _rec('coll', 'collector', inp=['s1', 's2'])]


def _pair_host(**instance):
    return [_rec('pr', 'pair', 'v1', out=['coll'],
                 per_submodule_outputs={'a': ['coll'], 'b': ['coll']}, **instance),
            _rec('coll', 'collector', inp=['pr'])]


# --------------------------------------------------------------------------------------------
# component instances
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_a_named_instance_and_the_default_instance_are_applied_with_the_vessel_suffix(
        tmp_path, library):
    _, rows = _load(tmp_path, library, _two_sources('other', None))
    assert _values(rows) == {
        'mean_s1': ('2e-05', 'other_ref'), 'amp_s1': ('0.3', 'other_ref'),
        # a global constant keeps its plain name and is added once, the first record's
        'omega': ('4.0', 'other_ref'),
        'mean_s2': ('1e-05', 'default_ref'), 'amp_s2': ('0.5', 'default_ref'),
    }
    assert [r['variable_name'] for r in rows].count('omega') == 1
    # only the four parameter columns are kept (the "sourced" column is ignored)
    assert all(set(r) == {'variable_name', 'units', 'value', 'data_reference'} for r in rows)


@pytest.mark.unit
def test_the_first_record_sets_a_global_shared_by_several_instances(tmp_path, library):
    _, rows = _load(tmp_path, library, _two_sources(None, 'other'))
    assert _values(rows)['omega'] == ('6.0', 'default_ref')
    assert _values(rows)['mean_s2'] == ('2e-05', 'other_ref')


@pytest.mark.unit
def test_nothing_is_loaded_without_an_instance_or_an_existing_default(tmp_path, library):
    # v2 declares default_instance "missing" but has no instances/: nothing, as before
    records = [_rec('s', 'pulse_src', 'v2', out=['coll']), _rec('coll', 'collector', inp=['s'])]
    _, rows = _load(tmp_path, library, records)
    assert rows == []


@pytest.mark.unit
def test_an_unknown_instance_names_the_version_directory_and_its_instances(tmp_path, library):
    with pytest.raises(ValueError) as error:
        _load(tmp_path, library, _two_sources('nope'))
    message = str(error.value)
    assert 'unknown instance "nope"' in message and '"s1"' in message
    assert os.path.join('pulse_src', 'versions', 'v1') in message
    assert "['default', 'other']" in message


@pytest.mark.unit
def test_an_instance_of_a_config_without_instances_is_an_error(tmp_path, library):
    records = [_rec('s', 'pulse_src', 'v2', out=['coll'], instance='default'),
               _rec('coll', 'collector', inp=['s'])]
    with pytest.raises(ValueError, match=r'has no instances/ directory'):
        _load(tmp_path, library, records)


@pytest.mark.unit
def test_an_instance_of_a_type_with_no_config_is_an_error(tmp_path, library):
    records = [_rec('s', 'pulse_src', 'v9', instance='default')]
    with pytest.raises(ValueError, match=r'no module config entry of that type'):
        _load(tmp_path, library, records)


@pytest.mark.unit
@pytest.mark.parametrize('value', ['', '   ', 3, 'a/b', '..'])
def test_a_bad_instance_name_is_rejected(value):
    with pytest.raises(ValueError, match='instance'):
        normalise_vessel_record(_rec('s', 'pulse_src', 'v1', instance=value), 'test', 0)


@pytest.mark.unit
@pytest.mark.parametrize('entry', [_pulse_entry('v1', default_instance=''),
                                   dict(PAIR, default_instance=['base'])])
def test_a_bad_default_instance_is_rejected(entry):
    with pytest.raises(ValueError, match='default_instance'):
        normalise_module_config_entry(entry, 'test')


@pytest.mark.unit
def test_merge_default_parameters_uses_the_first_of_repeated_extra_rows():
    import numpy as np
    from libcuflynx.parsers.ModelParsers import merge_default_parameters
    dtype = [(c, '<U80') for c in ('variable_name', 'units', 'value', 'data_reference')]
    host = np.array([('a', 'm', '1', 'host')], dtype=dtype)
    merged = merge_default_parameters(host, [
        {'variable_name': 'g', 'units': 'm', 'value': '2', 'data_reference': 'first'},
        {'variable_name': 'g', 'units': 'm', 'value': '3', 'data_reference': 'second'}])
    assert merged['variable_name'].tolist() == ['a', 'g']
    assert merged['value'].tolist() == ['1', '2']


# --------------------------------------------------------------------------------------------
# supermodules
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_supermodule_instance_beats_its_default_parameters_which_beat_submodule_instances(
        tmp_path, library):
    records, rows = _load(tmp_path, library, _pair_host())
    assert [r['name'] for r in records] == ['pr_a', 'pr_b', 'coll']
    assert _values(rows) == {
        # the supermodule's default instance "base"
        'amp_pr_a': ('0.9', 'base_ref'), 'omega': ('5.0', 'base_ref'),
        # its default_parameters, where base does not set the name
        'mean_pr_b': ('3e-05', 'pair_defaults'),
        # submodule a's own instance "other", then b's default instance
        'mean_pr_a': ('2e-05', 'other_ref'),
        'amp_pr_b': ('0.5', 'default_ref'),
    }


@pytest.mark.unit
def test_a_named_supermodule_instance_replaces_its_default_instance(tmp_path, library):
    _, rows = _load(tmp_path, library, _pair_host(instance='alt'))
    assert _values(rows) == {
        'amp_pr_b': ('0.1', 'alt_ref'),
        'amp_pr_a': ('0.77', 'pair_defaults'), 'mean_pr_b': ('3e-05', 'pair_defaults'),
        'mean_pr_a': ('2e-05', 'other_ref'), 'omega': ('4.0', 'other_ref'),
    }


@pytest.mark.unit
def test_an_unknown_supermodule_instance_lists_its_instances(tmp_path, library):
    with pytest.raises(ValueError) as error:
        _load(tmp_path, library, _pair_host(instance='nope'))
    message = str(error.value)
    assert 'unknown instance "nope"' in message and "['alt', 'base']" in message
    assert os.path.join('pair', 'versions', 'v1') in message


@pytest.mark.unit
def test_an_unknown_submodule_instance_is_an_error(tmp_path, library):
    directory = tmp_path / 'bad_pair'
    bad_pair = {k: v for k, v in PAIR.items() if k not in ('default_instance', 'default_parameters')}
    _write_json(str(directory / 'badpair_modules_config.json'),
                [dict(bad_pair, module_type='badpair', submodules=[_sub('a', instance='nope')])])
    path = str(tmp_path / 'm_module_array.json')
    _write_json(path, [_rec('pr', 'badpair', 'v1', out=['coll'],
                            per_submodule_outputs={'a': ['coll']}),
                       _rec('coll', 'collector', inp=['pr'])])
    files = _config_files(library) + [str(directory / 'badpair_modules_config.json')]
    with pytest.raises(ValueError, match=r'"pr_a".*unknown instance "nope"'):
        load_expanded_vessel_records(path, load_supermodule_registry(files),
                                     load_component_registry(files))


# --------------------------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------------------------

def _generate(work_dir, library, prefix, records, params):
    modules, readers = library
    resources = os.path.join(str(work_dir), 'resources')
    _write_json(os.path.join(resources, f'{prefix}_module_array.json'), records)
    _write_parameters(os.path.join(resources, f'{prefix}_parameters.csv'), params, 'host_value')
    generated = os.path.join(str(work_dir), 'generated_models')
    config = {'file_prefix': prefix, 'input_param_file': f'{prefix}_parameters.csv',
              'model_type': 'cellml', 'solver': 'CVODE_myokit', 'resources_dir': resources,
              'generated_models_dir': generated, 'external_modules_dir': readers,
              'module_library_dirs': [modules], 'DEBUG': False}
    assert generate_with_new_architecture(False, config), f'generation of {prefix} failed'
    import csv
    with open(os.path.join(generated, prefix, f'{prefix}_parameters.csv')) as f:
        return {row['variable_name']: (row['value'], row['data_reference'])
                for row in csv.DictReader(f)}


@pytest.mark.integration
def test_generation_uses_instances_and_the_host_file_wins(tmp_path, library):
    records = [_rec('s1', 'pulse_src', 'v1', out=['coll'], instance='other'),
               _rec('s2', 'pulse_src', 'v1', out=['coll']),
               _rec('pr', 'pair', 'v1', out=['coll'],
                    per_submodule_outputs={'a': ['coll'], 'b': ['coll']}),
               _rec('coll', 'collector', inp=['s1', 's2', 'pr'])]
    host = [('mean_s1', 'm3_per_s', '9e-06'), ('amp_pr_a', 'dimensionless', '0.11'),
            ('omega', 'per_s', '7.0')]
    params = _generate(tmp_path, library, 'mi_gen', records, host)
    assert params['mean_s1'] == ('9e-06', 'host_value')
    assert params['amp_s1'] == ('0.3', 'other_ref')
    assert params['mean_s2'] == ('1e-05', 'default_ref')
    assert params['amp_pr_a'] == ('0.11', 'host_value')
    assert params['mean_pr_a'] == ('2e-05', 'other_ref')
    assert params['mean_pr_b'] == ('3e-05', 'pair_defaults')
    assert params['omega'] == ('7.0', 'host_value')
    assert 'omega_s1' not in params and 'omega_pr_a' not in params


# --------------------------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_the_json_schemas_accept_instances_and_obs_data_names(library):
    jsonschema = pytest.importorskip('jsonschema')
    from libcuflynx.schemas import (MODULE_CONFIG_SCHEMA, OBS_DATA_SCHEMA, MODULE_ARRAY_SCHEMA,
                                    load_schema)
    validators = {}
    for name in (MODULE_ARRAY_SCHEMA, MODULE_CONFIG_SCHEMA, OBS_DATA_SCHEMA):
        schema = load_schema(name)
        jsonschema.Draft202012Validator.check_schema(schema)
        validators[name] = jsonschema.Draft202012Validator(schema)
    vessels, configs, obs = (validators[MODULE_ARRAY_SCHEMA], validators[MODULE_CONFIG_SCHEMA],
                             validators[OBS_DATA_SCHEMA])

    vessels.validate(_two_sources('other') + _pair_host(instance='alt'))
    for bad in ('', ' ', 'a/b', '..', 3):
        assert not vessels.is_valid([_rec('s', 'pulse_src', 'v1', instance=bad)]), bad
    for path in _config_files(library):
        with open(path) as f:
            configs.validate(json.load(f))
    assert not configs.is_valid([_pulse_entry('v1', default_instance='a/b')])
    assert not configs.is_valid([dict(PAIR, submodules=[_sub('a', instance='')])])

    obs.validate({'obs_data_name': 'other', 'data_items': [], 'protocol_info': {}})
    obs.validate({'data_items': []})
    obs.validate([{'data_item_name': 'x'}])
    assert not obs.is_valid({'obs_data_name': 3, 'data_items': []})
    assert not obs.is_valid({'obs_data_name': '', 'data_items': []})


# --------------------------------------------------------------------------------------------
# obs_data_name
# --------------------------------------------------------------------------------------------

def _obs_doc(**top):
    doc = {'protocol_info': {'pre_times': [0.0], 'sim_times': [[1.0]], 'params_to_change': {}},
           'data_items': [{'data_item_name': 'mean of x', 'data_type': 'constant',
                           'unit': 'dimensionless', 'operands': ['main/x'], 'operation': 'mean',
                           'weight': 1.0, 'value': 1.0, 'std': 0.1}]}
    doc.update(top)
    return doc


def _parse(doc=None, path=None):
    from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser
    return ObsAndParamDataParser().parse_obs_data_json(param_id_obs_path=path, obs_data_dict=doc,
                                                       pre_time=0.0, sim_time=1.0)


def _name_warnings(caught):
    return [w for w in caught if 'obs_data_name' in str(w.message)]


@pytest.mark.unit
def test_obs_data_name_is_accepted_and_returned():
    assert _parse(_obs_doc(obs_data_name='other'))['obs_data_name'] == 'other'
    parsed = _parse(_obs_doc())
    assert parsed['obs_data_name'] is None and len(parsed['gt_df']) == 1


@pytest.mark.unit
@pytest.mark.parametrize('bad', [3, '', ['other']])
def test_a_bad_obs_data_name_is_rejected(bad):
    with pytest.raises(ValueError, match='obs_data_name'):
        _parse(_obs_doc(obs_data_name=bad))


@pytest.mark.unit
@pytest.mark.parametrize('top, warns', [({'obs_data_name': 'other'}, False),
                                        ({'obs_data_name': 'default'}, True),
                                        ({}, True)])
def test_obs_data_in_an_instance_directory_is_checked_against_its_name(tmp_path, top, warns):
    path = str(tmp_path / 'v1' / 'instances' / 'other' / 'other_obs_data.json')
    _write_json(path, _obs_doc(**top))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        parsed = _parse(path=path)
    assert len(parsed['gt_df']) == 1
    found = _name_warnings(caught)
    if warns:
        assert len(found) == 1 and "instance 'other'" in str(found[0].message)
    else:
        assert found == []


@pytest.mark.unit
def test_obs_data_outside_an_instance_directory_is_not_checked(tmp_path):
    path = str(tmp_path / 'resources' / 'm_obs_data.json')
    _write_json(path, _obs_doc(obs_data_name='anything'))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert _parse(path=path)['obs_data_name'] == 'anything'
    assert _name_warnings(caught) == []


# --------------------------------------------------------------------------------------------
# existing models are unchanged
# --------------------------------------------------------------------------------------------

def _resource_prefixes():
    from test_json_module_arrays import RESOURCE_PREFIXES
    return RESOURCE_PREFIXES


def _generate_resource(work_dir, prefix):
    resources = os.path.join(str(work_dir), 'resources')
    os.makedirs(resources, exist_ok=True)
    for suffix in ('_parameters.csv', '_module_array.csv'):
        shutil.copy(os.path.join(RESOURCES_DIR, prefix + suffix), resources)
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
@pytest.mark.parametrize('prefix', _resource_prefixes())
def test_resources_models_generate_identically_without_the_instance_lookup(
        tmp_path, monkeypatch, prefix):
    if not os.path.exists(os.path.join(RESOURCES_DIR, f'{prefix}_parameters.csv')):
        pytest.skip(f'{prefix} has no parameters file')
    ok, with_lookup = _generate_resource(tmp_path / 'instances', prefix)
    if not ok:
        pytest.skip(f'{prefix} does not generate')
    import libcuflynx.parsers.ModelParsers as model_parsers
    monkeypatch.setattr(model_parsers, 'load_component_registry', lambda files: None)
    ok, without = _generate_resource(tmp_path / 'plain', prefix)
    assert ok
    _assert_same_generated_models(with_lookup, without)


# --- rows libcuflynx appends itself must fit module arrays with extra columns ("instance", ...) ---

def test_appended_vessel_rows_fill_extra_columns():
    import pandas as pd
    from libcuflynx.parsers.ModelParsers import _append_vessel_row
    df = pd.DataFrame(columns=['name', 'BC_type', 'vessel_type', 'inp_vessels', 'out_vessels', 'instance'])
    _append_vessel_row(df, 'par', 'vp', 'arterial_simple', ['heart'], ['pvn'])
    assert list(df.loc[0]) == ['par', 'vp', 'arterial_simple', ['heart'], ['pvn'], '']
    # a frame with the columns in another order still gets each value in its column
    df2 = pd.DataFrame(columns=['instance', 'name', 'BC_type', 'vessel_type', 'inp_vessels', 'out_vessels'])
    _append_vessel_row(df2, 'volume_sum_1D', 'nn', 'FV1D_volume_sum', '', 'total')
    assert df2.loc[0, 'name'] == 'volume_sum_1D' and df2.loc[0, 'instance'] == ''

def _generate_cpp_1d(work_dir, module_array_name, records=None):
    import contextlib
    import io
    res = os.path.join(work_dir, 'res')
    os.makedirs(res, exist_ok=True)
    shutil.copy(os.path.join(RESOURCES_DIR, 'aortic_bif_1d_parameters.csv'), res)
    if records is None:
        shutil.copy(os.path.join(RESOURCES_DIR, 'aortic_bif_1d_module_array.csv'), res)
    else:
        _write_json(os.path.join(res, module_array_name), records)
    cfg = {'file_prefix': 'aortic_bif_1d', 'input_param_file': 'aortic_bif_1d_parameters.csv',
           'model_type': 'cpp', 'solver': 'RK4', 'couple_to_1d': True, 'resources_dir': res,
           'generated_models_dir': os.path.join(work_dir, 'gen'), 'cpp_generated_models_dir': os.path.join(work_dir, 'cpp'),
           'cpp_1d_model_config_path': None, 'dt': 0.001,
           'solver_info': {'dt_solver': 1e-4, 'MaximumNumberOfSteps': 5000, 'solver': 'RK4'}, 'DEBUG': False}
    with contextlib.redirect_stdout(io.StringIO()):
        assert generate_with_new_architecture(False, cfg)
    with open(os.path.join(work_dir, 'cpp', 'model0d.cc')) as f:
        return f.read()


def test_0d_1d_split_accepts_records_with_extra_keys(tmp_path):
    '''split_0d_1d_module_array appends volume_sum_1D / 1D-coupling rows; with an extra record key
    (such as "instance") a positional 5-value row used to raise "cannot set a row with mismatched
    columns". The generated C++ is the same as from the plain array.'''
    from libcuflynx.utilities.config_schemas import read_module_array_records
    plain = _generate_cpp_1d(str(tmp_path / 'plain'), None)
    records = [dict(r, comment='extra column') for r in
               read_module_array_records(os.path.join(RESOURCES_DIR, 'aortic_bif_1d_module_array.csv'))]
    extra = _generate_cpp_1d(str(tmp_path / 'extra'), 'aortic_bif_1d_module_array.json', records)
    assert extra == plain


def test_an_empty_optional_csv_cell_is_not_set(tmp_path):
    '''An empty "instance" cell in a CSV module array (e.g. the 0D part libcuflynx writes itself
    for the 1D split, where appended rows have no instance) leaves the key unset rather than
    giving an invalid empty instance name.'''
    from libcuflynx.utilities.config_schemas import read_module_array_records
    p = tmp_path / 'm_module_array.csv'
    p.write_text('name,BC_type,vessel_type,inp_vessels,out_vessels,instance\n'
                 'a,nn,pulse_src,,b,other\n'
                 'b,nn,volume_sum,a,,\n')
    recs = read_module_array_records(str(p))
    assert recs[0]['instance'] == 'other'
    assert 'instance' not in recs[1]
