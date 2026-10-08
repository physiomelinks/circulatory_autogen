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

Most tests run against the real module library, circulatory-autogen-modules, at the commit CI
pins (``tests/MODULE_LIBRARY_REF``; found as ``tests/_module_library.py`` describes, skipped
without it). They read what they expect from the library's own instance files, so a value that
is recalibrated there changes nothing here. The modules they use:

* ``i_M`` (``Argus2026_v01``): instances ``default`` and ``davis2020_wistar`` among others, whose
  ``rho_M`` differ; ``F`` and ``T`` are global constants;
* ``soma`` (``sympathetic``): a supermodule with an instance, whose submodules name instances;
* ``neuron`` (``sympathetic``): a supermodule with ``soma`` nested in it.

A small synthetic library is kept for what a real library cannot contain -- its own structure
tests forbid it: a ``default_instance`` whose file is missing and a config without an
``instances/`` directory (``pulse_src`` v2), a supermodule with legacy ``default_parameters``
(``pair``, instances ``base`` and ``alt``), and two instances that set one global constant to
different values (``pulse_src`` v1 ``default`` and ``other``; no real module's instances
disagree on a global). Its reader modules are those of test_config_schemas.py.
"""
import csv
import json
import os
import shutil
import types
import warnings

import pytest

import _module_library as module_library
from test_config_schemas import (_assert_same_generated_models, _prescribed, _write_library,
                                 modules_config)

from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
from libcuflynx.utilities.config_schemas import (load_component_registry,
                                                 load_expanded_vessel_records,
                                                 load_supermodule_registry,
                                                 normalise_module_config_entry,
                                                 normalise_vessel_record)
from libcuflynx.utilities.module_instances import (ConflictingGlobalWarning,
                                                   available_instances, global_constants,
                                                   instance_parameters_path, read_parameter_rows)
from libcuflynx.utilities.supermodules import rename_default_parameter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES_DIR = os.path.join(REPO_ROOT, 'resources')

# --------------------------------------------------------------------------------------------
# the real module library
# --------------------------------------------------------------------------------------------

I_M = ('i_M', 'Argus2026_v01')
I_M_NAMED = 'davis2020_wistar'
SOMA = ('soma', 'sympathetic')
NEURON = ('neuron', 'sympathetic')


@pytest.fixture(scope='module')
def real_library():
    """The library's ``modules`` dir, its config files and both registries."""
    modules = str(module_library.modules_dir_or_skip())
    from libcuflynx.utilities.module_library import ModuleSources
    # the library holds its own copies of the built-in modules, so it is used alone, as its
    # own harness does
    files = ModuleSources({'module_library_dirs': [modules],
                           'use_builtin_modules': False}).config_files
    return types.SimpleNamespace(modules=modules, files=files,
                                 components=load_component_registry(files),
                                 supermodules=load_supermodule_registry(files))


def _entry(lib, key):
    return lib.supermodules.get(key) or lib.components[key]


def _instance_values(lib, key, instance):
    """``{variable_name: (value, data_reference)}`` of one real instance's file."""
    path = instance_parameters_path(_entry(lib, key)['config_path'], instance)
    return {row['variable_name']: (row['value'], row['data_reference'])
            for row in read_parameter_rows(path, 'test')}


def _real_load(tmp_path, lib, records):
    path = str(tmp_path / 'm_module_array.json')
    _write_json(path, records)
    return load_expanded_vessel_records(path, lib.supermodules, lib.components)


# --------------------------------------------------------------------------------------------
# the synthetic library (only what a real library cannot contain; see the module docstring)
# --------------------------------------------------------------------------------------------

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
        tmp_path, real_library):
    lib = real_library
    named = _instance_values(lib, I_M, I_M_NAMED)
    default = _instance_values(lib, I_M, _entry(lib, I_M)['default_instance'])
    assert named['rho_M'] != default['rho_M'], 'the library no longer tells them apart'
    globals_ = global_constants(_entry(lib, I_M))
    assert globals_, 'i_M has no global constants left to test with'

    _, rows = _real_load(tmp_path, lib, [_rec('s1', *I_M, instance=I_M_NAMED), _rec('s2', *I_M)])

    expected = {}
    for vessel, values in (('s1', named), ('s2', default)):
        for name, value in values.items():
            # a global keeps its plain name and comes once, from the first record
            expected.setdefault(name if name in globals_ else f'{name}_{vessel}', value)
    assert _values(rows) == expected
    names = [r['variable_name'] for r in rows]
    for name in globals_ & set(named):
        assert names.count(name) == 1 and f'{name}_s1' not in names
    # only the four parameter columns are kept (the library's "sourced" column is ignored)
    assert all(set(r) == {'variable_name', 'units', 'value', 'data_reference'} for r in rows)


def _conflicts(caught):
    return [w.message for w in caught if isinstance(w.message, ConflictingGlobalWarning)]


def _load_caught(tmp_path, library, records):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        records, rows = _load(tmp_path, library, records)
    return rows, _conflicts(caught)


@pytest.mark.unit
def test_the_first_record_sets_a_global_shared_by_several_instances(tmp_path, library):
    # synthetic: no two instances in the real library set a global to different values
    rows, conflicts = _load_caught(tmp_path, library, _two_sources(None, 'other'))
    assert _values(rows)['omega'] == ('6.0', 'default_ref')
    assert _values(rows)['mean_s2'] == ('2e-05', 'other_ref')
    # ...and the disagreement is reported, naming both values and who set them
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict.name == 'omega'
    assert [v for v, _, _ in conflict.values] == ['6.0', '4.0']
    message = str(conflict)
    assert '"s1" (instance "default")' in message and '"s2" (instance "other")' in message
    assert 'host parameters file' in message


@pytest.mark.unit
def test_instances_that_agree_on_a_global_do_not_warn(tmp_path, library):
    _, conflicts = _load_caught(tmp_path, library, _two_sources('other', 'other'))
    assert not conflicts


@pytest.mark.unit
def test_a_global_a_supermodule_instance_sets_is_not_a_conflict_of_its_submodules(tmp_path,
                                                                                  library):
    # pair's submodules' instances set omega to 4.0 (a: other) and 6.0 (b: default), but pair's
    # own instance sets 5.0, which outranks both
    rows, conflicts = _load_caught(tmp_path, library, _pair_host())
    assert _values(rows)['omega'][0] == '5.0'
    assert not conflicts


@pytest.mark.unit
def test_a_supermodule_s_global_used_for_modules_outside_it_warns(tmp_path, library):
    # pr1 (instance base) sets omega 5.0; pr2 (alt) sets none, so its submodules' instances
    # chose 4.0 and 6.0 -- but a global is one value for the model, and pr2's modules run at 5.0
    records = [_rec('pr1', 'pair', 'v1', out=['coll'], instance='base',
                    per_submodule_outputs={'a': ['coll'], 'b': ['coll']}),
               _rec('pr2', 'pair', 'v1', out=['coll'], instance='alt',
                    per_submodule_outputs={'a': ['coll'], 'b': ['coll']}),
               _rec('coll', 'collector', inp=['pr1', 'pr2'])]
    rows, conflicts = _load_caught(tmp_path, library, records)
    assert _values(rows)['omega'][0] == '5.0'
    assert len(conflicts) == 1
    assert [v for v, _, _ in conflicts[0].values] == ['5.0', '4.0', '6.0']
    message = str(conflicts[0])
    assert 'supermodule "pr1" (instance "base")' in message and '"pr2_a"' in message
    # pr1's own submodules (pr1_a: 4.0, pr1_b: 6.0) are overridden on purpose: not listed
    assert '"pr1_a"' not in message and '"pr1_b"' not in message


@pytest.mark.unit
def test_the_host_file_settles_a_conflicting_global():
    from libcuflynx.utilities.module_instances import reissue_warnings, warn_conflicting_globals
    settings = [('T', '295.15', 'kelvin', '"a"', 'a'), ('T', '310', 'kelvin', '"b"', 'b')]
    for settled, expected in (((), 1), ({'T'}, 0)):
        with warnings.catch_warnings(record=True) as held:
            warnings.simplefilter('always')
            warn_conflicting_globals(settings, 'm')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            reissue_warnings(held, settled=settled)
        assert len(_conflicts(caught)) == expected
    # different units: most likely two quantities sharing a name
    with warnings.catch_warnings(record=True) as held:
        warnings.simplefilter('always')
        warn_conflicting_globals([('T', '295.15', 'kelvin', '"cell"', 'cell'),
                                  ('T', '1', 'second', '"clock"', 'clock')], 'm')
    assert 'units differ' in str(_conflicts(held)[0])
    # an outer supermodule overriding the global of one nested in it is not a conflict
    with warnings.catch_warnings(record=True) as held:
        warnings.simplefilter('always')
        warn_conflicting_globals([('T', '1', 'K', 'outer', 'h'), ('T', '2', 'K', 'inner', 'h_soma')],
                                 'm')
    assert not _conflicts(held)


@pytest.mark.unit
def test_nothing_is_loaded_without_an_instance_or_an_existing_default(tmp_path, library):
    # synthetic: v2 declares default_instance "missing" but has no instances/: nothing, as
    # before (the real library's structure test forbids such a version)
    records = [_rec('s', 'pulse_src', 'v2', out=['coll']), _rec('coll', 'collector', inp=['s'])]
    _, rows = _load(tmp_path, library, records)
    assert rows == []


@pytest.mark.unit
def test_an_unknown_instance_names_the_version_directory_and_its_instances(tmp_path,
                                                                            real_library):
    with pytest.raises(ValueError) as error:
        _real_load(tmp_path, real_library, [_rec('s1', *I_M, instance='nope')])
    message = str(error.value)
    assert 'unknown instance "nope"' in message and '"s1"' in message
    assert os.path.join('i_M', 'versions', 'Argus2026_v01') in message
    existing = available_instances(_entry(real_library, I_M)['config_path'])
    assert I_M_NAMED in existing and str(existing) in message


@pytest.mark.unit
def test_an_instance_of_a_config_without_instances_is_an_error(tmp_path, library):
    # synthetic: every real module version has an instances/ directory
    records = [_rec('s', 'pulse_src', 'v2', out=['coll'], instance='default'),
               _rec('coll', 'collector', inp=['s'])]
    with pytest.raises(ValueError, match=r'has no instances/ directory'):
        _load(tmp_path, library, records)


@pytest.mark.unit
def test_an_instance_of_a_type_with_no_config_is_an_error(tmp_path, real_library):
    with pytest.raises(ValueError, match=r'no module config entry of that type'):
        _real_load(tmp_path, real_library, [_rec('s', 'i_M', 'v9', instance='default')])


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
def test_a_supermodule_instance_beats_its_submodules_instances(tmp_path, real_library):
    lib = real_library
    soma = lib.supermodules[SOMA]
    subs = [s['name'] for s in soma['submodules']]
    records, rows = _real_load(tmp_path, lib, [_rec('soma', *SOMA)])
    assert [r['name'] for r in records] == [f'soma_{s}' for s in subs]
    values = _values(rows)

    own = {rename_default_parameter(name, 'soma', subs): value for name, value in
           _instance_values(lib, SOMA, soma['default_instance']).items()}
    for name, value in own.items():
        assert values[name] == value, name

    overridden = 0
    for sub in soma['submodules']:
        key = (sub['vessel_type'], sub['BC_type'])
        globals_ = global_constants(_entry(lib, key))
        instance = sub.get('instance') or _entry(lib, key).get('default_instance')
        for name, value in _instance_values(lib, key, instance).items():
            if name in globals_:
                continue
            model_name = f'{name}_soma_{sub["name"]}'
            if model_name in own:
                overridden += 1          # the supermodule's instance wins
                assert values[model_name] == own[model_name]
            else:
                assert values[model_name] == value, model_name
    assert overridden, 'soma\'s instance sets nothing its submodules\' instances set'


@pytest.mark.unit
def test_a_nested_supermodule_s_instance_is_renamed_through_the_nesting(tmp_path, real_library):
    lib = real_library
    soma = lib.supermodules[SOMA]
    subs = [s['name'] for s in soma['submodules']]
    neuron = lib.supermodules[NEURON]
    soma_path = next(s['name'] for s in neuron['submodules']
                     if (s['vessel_type'], s['BC_type']) == SOMA)
    records, rows = _real_load(tmp_path, lib, [_rec('neuron', *NEURON)])
    names = [r['name'] for r in records]
    assert f'neuron_{soma_path}_{subs[0]}' in names
    values = _values(rows)
    neuron_own = {rename_default_parameter(n, 'neuron', [s['name'] for s in neuron['submodules']])
                  for n in _instance_values(lib, NEURON, neuron['default_instance'])}
    checked = 0
    for name, value in _instance_values(lib, SOMA, soma['default_instance']).items():
        renamed = rename_default_parameter(name, 'soma', subs)
        if renamed == name:
            continue                     # a global
        nested = rename_default_parameter(name, f'neuron_{soma_path}', subs)
        if nested in neuron_own:
            continue                     # the outer supermodule sets it
        assert values[nested] == value, nested
        checked += 1
    assert checked


@pytest.mark.unit
def test_supermodule_instance_beats_its_default_parameters_which_beat_submodule_instances(
        tmp_path, library):
    # synthetic: no real supermodule has legacy default_parameters
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
    # synthetic: the real supermodules have one instance each, and this one has
    # default_parameters too
    _, rows = _load(tmp_path, library, _pair_host(instance='alt'))
    assert _values(rows) == {
        'amp_pr_b': ('0.1', 'alt_ref'),
        'amp_pr_a': ('0.77', 'pair_defaults'), 'mean_pr_b': ('3e-05', 'pair_defaults'),
        'mean_pr_a': ('2e-05', 'other_ref'), 'omega': ('4.0', 'other_ref'),
    }


@pytest.mark.unit
def test_an_unknown_supermodule_instance_lists_its_instances(tmp_path, real_library):
    with pytest.raises(ValueError) as error:
        _real_load(tmp_path, real_library, [_rec('soma', *SOMA, instance='nope')])
    message = str(error.value)
    existing = available_instances(real_library.supermodules[SOMA]['config_path'])
    assert 'unknown instance "nope"' in message and str(existing) in message
    assert os.path.join('soma', 'versions', 'sympathetic') in message


@pytest.mark.unit
def test_an_unknown_submodule_instance_is_an_error(tmp_path, real_library):
    lib = real_library
    # a supermodule config of its own whose one submodule is the real i_M, naming an instance
    # i_M does not have
    config = tmp_path / 'probe' / 'probe_v1_modules_config.json'
    _write_json(str(config), [{
        'module_type': 'probe', 'module_subtype': 'v1', 'module_format': 'supermodule',
        'submodules': [{'name': 'm', 'module_type': I_M[0], 'module_subtype': I_M[1],
                        'instance': 'nope', 'inp_instances': [], 'out_instances': []}]}])
    files = lib.files + [str(config)]
    path = str(tmp_path / 'm_module_array.json')
    _write_json(path, [_rec('pr', 'probe', 'v1')])
    with pytest.raises(ValueError, match=r'"pr_m".*unknown instance "nope"'):
        load_expanded_vessel_records(path, load_supermodule_registry(files),
                                     load_component_registry(files))


# --------------------------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------------------------

@pytest.mark.integration
def test_generation_uses_instances_and_the_host_file_wins(tmp_path, real_library):
    lib = real_library
    named = _instance_values(lib, I_M, I_M_NAMED)
    default = _instance_values(lib, I_M, _entry(lib, I_M)['default_instance'])
    units = {row['variable_name']: row['units'] for row in read_parameter_rows(
        instance_parameters_path(_entry(lib, I_M)['config_path'], I_M_NAMED), 'test')}
    prefix = 'mi_gen'
    resources = tmp_path / 'resources'
    _write_json(str(resources / f'{prefix}_module_array.json'),
                [_rec('s1', *I_M, instance=I_M_NAMED), _rec('s2', *I_M)])
    # host values that differ from every instance's, under the module's own units
    _write_parameters(str(resources / f'{prefix}_parameters.csv'),
                      [('tau_w_num_s1', units['tau_w_num'], '999'),
                       ('T', units['T'], '300')], 'host_value')
    generated = tmp_path / 'generated_models'
    config = {'file_prefix': prefix, 'input_param_file': f'{prefix}_parameters.csv',
              'model_type': 'cellml', 'solver': 'CVODE_myokit', 'resources_dir': str(resources),
              'generated_models_dir': str(generated), 'module_library_dirs': [lib.modules],
              'use_builtin_modules': False, 'DEBUG': False}
    assert generate_with_new_architecture(False, config), f'generation of {prefix} failed'
    with open(generated / prefix / f'{prefix}_parameters.csv') as f:
        params = {row['variable_name']: (row['value'], row['data_reference'])
                  for row in csv.DictReader(f)}
    assert params['tau_w_num_s1'] == ('999', 'host_value')
    assert params['T'] == ('300', 'host_value')
    # values only: generation rewrites the library's data_reference text
    assert params['rho_M_s1'][0] == named['rho_M'][0]
    assert params['rho_M_s2'][0] == default['rho_M'][0]
    assert params['tau_w_num_s2'][0] == default['tau_w_num'][0]
    assert 'T_s1' not in params and 'T_s2' not in params


# --------------------------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_the_json_schemas_accept_instances_and_obs_data_names(real_library):
    jsonschema = pytest.importorskip('jsonschema')
    from libcuflynx.schemas import (MODULE_CONFIG_SCHEMA, OBS_DATA_SCHEMA, MODULE_ARRAY_SCHEMA,
                                    load_schema)
    validators = {}
    for name in (MODULE_ARRAY_SCHEMA, MODULE_CONFIG_SCHEMA, OBS_DATA_SCHEMA):
        schema = load_schema(name)
        jsonschema.Draft202012Validator.check_schema(schema)
        validators[name] = jsonschema.Draft202012Validator(schema)
    arrays, configs, obs = (validators[MODULE_ARRAY_SCHEMA], validators[MODULE_CONFIG_SCHEMA],
                            validators[OBS_DATA_SCHEMA])

    # every module config in the library, and every instance obs_data
    for path in real_library.files:
        with open(path) as f:
            configs.validate(json.load(f))
    instance_obs = []
    for directory, _, files in os.walk(real_library.modules):
        if os.path.basename(os.path.dirname(directory)) == 'instances':
            instance_obs += [os.path.join(directory, f) for f in files
                             if f.endswith('_obs_data.json')]
    assert instance_obs, 'the library has no instance obs_data to check'
    for path in instance_obs:
        with open(path) as f:
            obs.validate(json.load(f))

    arrays.validate([_rec('s1', *I_M, instance=I_M_NAMED), _rec('s2', *I_M),
                     _rec('soma', *SOMA, instance='default')])
    for bad in ('', ' ', 'a/b', '..', 3):
        assert not arrays.is_valid([_rec('s', *I_M, instance=bad)]), bad
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
def test_a_real_instance_s_obs_data_names_its_instance(real_library):
    path = os.path.join(os.path.dirname(instance_parameters_path(
        _entry(real_library, I_M)['config_path'], I_M_NAMED)), f'{I_M_NAMED}_obs_data.json')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert _parse(path=path)['obs_data_name'] == I_M_NAMED
    assert _name_warnings(caught) == []


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
@pytest.mark.usefixtures('skip_generated_model_checks')
@pytest.mark.parametrize('prefix', _resource_prefixes())
def test_resources_models_generate_identically_without_the_instance_lookup(
        tmp_path, monkeypatch, prefix):
    if not os.path.exists(os.path.join(RESOURCES_DIR, f'{prefix}_parameters.csv')):
        pytest.skip(f'{prefix} has no parameters file')
    ok, with_lookup = _generate_resource(tmp_path / 'instances', prefix)
    if not ok:
        pytest.skip(f'{prefix} does not generate')
    # load_model reads the registry through ModuleSources; None means no instance lookup
    from libcuflynx.utilities.module_library import ModuleSources
    monkeypatch.setattr(ModuleSources, 'component_registry', lambda self: None)
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
