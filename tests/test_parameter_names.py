"""Parameter names written <vessel>/<variable> (utilities/parameter_names.py), and
cuflynx-migrate-parameter-names.

A model names a module's constant {var}_{vessel}; a file a person writes may name it
<vessel>/<variable> instead, which is never ambiguous -- {var}_{vessel} cannot be split back
when names contain "_". The supermodule tests use the real module library (see
tests/_module_library.py): its soma's instance, migrated, must give exactly what the old
form gave.
"""
import csv
import os
import shutil
import warnings

import pytest

import _module_library as module_library
from libcuflynx.utilities.config_schemas import (load_component_registry,
                                                 load_expanded_vessel_records,
                                                 load_supermodule_registry)
from libcuflynx.utilities.parameter_names import (host_model_names, model_name, split_path_name,
                                                  supermodule_row_name)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES_DIR = os.path.join(REPO_ROOT, 'resources')

# --------------------------------------------------------------------------------------------
# the names
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize('written, expected', [
    ('aortic_root/C', 'C_aortic_root'),
    ('heart/lv/E', 'E_heart_lv'),
    ('global/R', 'R'),
    ('R', 'R'),
    ('C_aortic_root', 'C_aortic_root'),      # the model's own name, read as before
    (' lv / E ', 'E_lv'),
])
def test_a_written_name_becomes_the_model_name(written, expected):
    assert model_name(written) == expected


@pytest.mark.unit
@pytest.mark.parametrize('bad', ['/C', 'lv/', 'lv//E', '/'])
def test_a_slash_name_needs_both_sides(bad):
    with pytest.raises(ValueError, match='nothing empty'):
        split_path_name(bad)


PATHS = [('lv',), ('rv',), ('soma',), ('soma', 'i_M'), ('i_M',), ('leak_Na',), ('Na',)]


@pytest.mark.unit
@pytest.mark.parametrize('written, expected, old', [
    ('lv/E', 'E_heart_lv', False),
    ('soma/i_M/rho_M', 'rho_M_heart_soma_i_M', False),
    ('i_M/rho_M', 'rho_M_heart_i_M', False),
    ('global/T', 'T', False),
    ('T', 'T', False),
    # the older form, split at the longest submodule suffix and flagged
    ('E_lv', 'E_heart_lv', True),
    ('rho_M_soma_i_M', 'rho_M_heart_soma_i_M', True),
    # ...which is a guess: g_leak_Na reads as g of leak_Na; g_leak of Na needs "Na/g_leak"
    ('g_leak_Na', 'g_heart_leak_Na', True),
    ('Na/g_leak', 'g_leak_heart_Na', False),
])
def test_supermodule_rows(written, expected, old):
    assert supermodule_row_name(written, 'heart', PATHS) == (expected, old)


@pytest.mark.unit
def test_a_supermodule_row_naming_no_submodule_is_an_error():
    with pytest.raises(ValueError, match='"lv/x" is not a submodule'):
        supermodule_row_name('lv/x/E', 'heart', PATHS)
    # the error lists the submodules, nested ones written with "/"
    with pytest.raises(ValueError, match="'soma/i_M'"):
        supermodule_row_name('nope/E', 'heart', PATHS)


VESSELS = {'aortic_root': {'C': 'constant', 'T': 'global_constant'},
           'heart_lv': {'E': 'constant'}}


@pytest.mark.unit
def test_host_names_are_checked_against_the_model():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        names = host_model_names(['aortic_root/C', 'heart/lv/E', 'global/R', 'C_x'],
                                 VESSELS, 'p.csv')
    assert names == ['C_aortic_root', 'E_heart_lv', 'R', 'C_x']
    assert not [w for w in caught if 'not used' in str(w.message)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        host_model_names(['aortc_root/C', 'aortic_root/Cx', 'aortic_root/T'], VESSELS, 'p.csv')
    message = str(next(w for w in caught if 'not used' in str(w.message)).message)
    assert 'aortc_root/C (no vessel "aortc_root"' in message
    assert 'has no variable "Cx"' in message
    assert 'aortic_root/T ("T" is a global constant: write global/T or T)' in message


@pytest.mark.unit
def test_one_parameter_set_twice_is_an_error():
    with pytest.raises(ValueError, match='"aortic_root/C" and "C_aortic_root" both set'):
        host_model_names(['aortic_root/C', 'C_aortic_root'], VESSELS, 'p.csv')


# --------------------------------------------------------------------------------------------
# generation, on a model of this repository
# --------------------------------------------------------------------------------------------

PREFIX = '3compartment'


def _copy_resources(directory):
    os.makedirs(directory)
    for suffix in ('_module_array.csv', '_parameters.csv'):
        shutil.copy(os.path.join(RESOURCES_DIR, PREFIX + suffix), directory)
    return os.path.join(directory, f'{PREFIX}_parameters.csv')


def _generate(resources, generated):
    from libcuflynx.scripts.script_generate_with_new_architecture import \
        generate_with_new_architecture
    config = {'file_prefix': PREFIX, 'input_param_file': f'{PREFIX}_parameters.csv',
              'model_type': 'cellml', 'solver': 'CVODE_myokit', 'resources_dir': resources,
              'generated_models_dir': generated, 'DEBUG': False}
    assert generate_with_new_architecture(False, config)
    return os.path.join(generated, PREFIX)


def _read(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


@pytest.mark.integration
def test_a_migrated_host_file_generates_the_same_model(tmp_path):
    from libcuflynx.scripts import migrate_parameter_names
    _copy_resources(str(tmp_path / 'old'))
    migrated = _copy_resources(str(tmp_path / 'new'))
    assert migrate_parameter_names.main(
        ['--module-array', str(tmp_path / 'new' / f'{PREFIX}_module_array.csv')]) == 0
    rows = _read(migrated)
    slashed = [r['variable_name'] for r in rows if '/' in r['variable_name']]
    assert slashed and 'aortic_root/C' in slashed
    assert len(rows) == len(_read(os.path.join(RESOURCES_DIR, f'{PREFIX}_parameters.csv')))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        new = _generate(str(tmp_path / 'new'), str(tmp_path / 'gen_new'))
    assert not [w for w in caught if 'not used by the model' in str(w.message)]
    old = _generate(str(tmp_path / 'old'), str(tmp_path / 'gen_old'))
    for name in sorted(os.listdir(old)):
        if name.endswith('.cellml'):
            with open(os.path.join(old, name)) as a, open(os.path.join(new, name)) as b:
                assert a.read() == b.read(), name


@pytest.mark.integration
def test_a_misspelt_vessel_is_reported_rather_than_silently_unused(tmp_path):
    params = _copy_resources(str(tmp_path / 'r'))
    with open(params) as f:
        text = f.read()
    columns = text.splitlines()[0].count(',') + 1
    row = ['aortic_rot/C', 'm6_per_N', '1e-9', 'typo'] + [''] * (columns - 4)
    with open(params, 'w') as f:
        f.write(text.rstrip('\n') + '\n' + ','.join(row) + '\n')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _generate(str(tmp_path / 'r'), str(tmp_path / 'g'))
    assert any('aortic_rot/C (no vessel "aortic_rot"' in str(w.message) for w in caught)


# --------------------------------------------------------------------------------------------
# supermodules, on the real library
# --------------------------------------------------------------------------------------------

SOMA = ('soma', 'sympathetic')


@pytest.fixture(scope='module')
def library_files():
    modules = str(module_library.modules_dir_or_skip())
    from libcuflynx.utilities.module_library import ModuleSources
    files = ModuleSources({'module_library_dirs': [modules],
                           'use_builtin_modules': False}).config_files
    return modules, files


def _soma_rows(tmp_path, files):
    path = str(tmp_path / 'm_module_array.json')
    import json
    with open(path, 'w') as f:
        json.dump([{'name': 'soma', 'module_type': SOMA[0], 'module_subtype': SOMA[1],
                    'inp_instances': [], 'out_instances': []}], f)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _, rows = load_expanded_vessel_records(path, load_supermodule_registry(files),
                                               load_component_registry(files))
    old_form = [w for w in caught if issubclass(w.category, FutureWarning)
                and '{variable}_{submodule}' in str(w.message)]
    return {r['variable_name']: (r['value'], r['data_reference']) for r in rows}, old_form


@pytest.mark.integration
def test_a_migrated_supermodule_instance_gives_the_same_parameters(tmp_path, library_files):
    from libcuflynx.scripts import migrate_parameter_names
    modules, files = library_files
    original = next(f for f in files if f.endswith('soma_sympathetic_modules_config.json'))
    os.makedirs(tmp_path / 'before')
    before, warned = _soma_rows(tmp_path / 'before', files)
    assert warned, 'the library\'s soma no longer uses the old form; this test has nothing to do'

    # a copy of the soma's version directory, migrated; the library supplies its submodules
    copy = tmp_path / 'soma_copy'
    shutil.copytree(os.path.dirname(original), copy)
    config = str(copy / os.path.basename(original))
    assert migrate_parameter_names.main(['--supermodule-config', config, '--module-library-dir',
                                         modules, '--no-builtin-modules']) == 0
    instance = copy / 'instances' / 'default' / 'default_parameters.csv'
    names = [r['variable_name'] for r in _read(str(instance))]
    assert any(n.startswith('membrane/') for n in names) and 'R' in names

    files_with_copy = [f for f in files if f != original] + [config]
    os.makedirs(tmp_path / 'after')
    after, warned = _soma_rows(tmp_path / 'after', files_with_copy)
    assert not warned
    assert after == before


@pytest.mark.unit
def test_a_component_instance_names_its_variables_without_a_vessel(tmp_path):
    from libcuflynx.utilities.module_instances import component_instance_rows
    directory = tmp_path / 'pulse' / 'versions' / 'v1'
    os.makedirs(directory / 'instances' / 'x')
    (directory / 'instances' / 'x' / 'x_parameters.csv').write_text(
        'variable_name,units,value,data_reference\ns1/mean,m3_per_s,1,r\n')
    entry = {'vessel_type': 'pulse_src', 'BC_type': 'v1', 'variables_and_units': [],
             'config_path': str(directory / 'pulse_v1_modules_config.json')}
    record = {'name': 's1', 'vessel_type': 'pulse_src', 'BC_type': 'v1', 'instance': 'x'}
    with pytest.raises(ValueError, match='named without a vessel'):
        component_instance_rows([record], {('pulse_src', 'v1'): entry})
