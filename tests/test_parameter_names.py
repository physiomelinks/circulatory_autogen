"""Parameter names written <vessel>/<variable> (utilities/parameter_names.py), and
cuflynx-upgrade parameter-names.

A model names a module's constant {var}_{vessel}; a file a person writes names it
<vessel>/<variable>, which is never ambiguous -- {var}_{vessel} cannot be split back when
names contain "_". The older {var}_{vessel} (host file) and {var}_{submodule} (supermodule
rows) forms are refused, naming the converter; the converter tests write a file in the old
form, check it is refused, convert it and check it reads as the current one.
"""
import csv
import os
import shutil

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
    ('heart/lv/E', 'E_heart__lv'),
    ('global/R', 'R'),
    ('R', 'R'),
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
@pytest.mark.parametrize('written, expected', [
    ('lv/E', 'E_heart__lv'),
    ('soma/i_M/rho_M', 'rho_M_heart__soma__i_M'),
    ('i_M/rho_M', 'rho_M_heart__i_M'),
    ('global/T', 'T'),
    ('T', 'T'),
    ('Na/g_leak', 'g_leak_heart__Na'),
])
def test_supermodule_rows(written, expected):
    assert supermodule_row_name(written, 'heart', PATHS) == expected


@pytest.mark.unit
@pytest.mark.parametrize('written, suggested', [('E_lv', 'lv/E'), ('rho_M_soma_i_M', 'soma/i_M/rho_M'),
                                                ('rho_M_soma__i_M', 'soma/i_M/rho_M')])
def test_a_supermodule_row_in_the_older_form_is_refused(written, suggested):
    with pytest.raises(ValueError, match=rf'"{written}" is written .*\({suggested}\).*'
                                         rf'cuflynx-upgrade parameter-names'):
        supermodule_row_name(written, 'heart', PATHS)


@pytest.mark.unit
def test_a_supermodule_row_naming_no_submodule_is_an_error():
    with pytest.raises(ValueError, match='"lv/x" is not a submodule'):
        supermodule_row_name('lv/x/E', 'heart', PATHS)
    # the error lists the submodules, nested ones written with "/"
    with pytest.raises(ValueError, match="'soma/i_M'"):
        supermodule_row_name('nope/E', 'heart', PATHS)


VESSELS = {'aortic_root': {'C': 'constant', 'T': 'global_constant'},
           'heart__lv': {'E': 'constant'}}


@pytest.mark.unit
def test_host_names_are_checked_against_the_model():
    names = host_model_names(['aortic_root/C', 'heart/lv/E', 'global/R', 'C_x'], VESSELS, 'p.csv')
    assert names == ['C_aortic_root', 'E_heart__lv', 'R', 'C_x']

    with pytest.raises(ValueError) as raised:
        host_model_names(['aortc_root/C', 'aortic_root/Cx', 'aortic_root/T'], VESSELS, 'p.csv')
    message = str(raised.value)
    assert '3 parameter(s) name something the model does not have' in message
    assert 'aortc_root/C (no vessel "aortc_root"' in message
    assert 'has no variable "Cx"' in message
    assert 'aortic_root/T ("T" is a global constant: write global/T or T)' in message


@pytest.mark.unit
def test_one_parameter_set_twice_is_an_error():
    with pytest.raises(ValueError, match='"aortic_root/C" and "aortic_root / C" both set'):
        host_model_names(['aortic_root/C', 'aortic_root / C'], VESSELS, 'p.csv')


@pytest.mark.unit
def test_a_host_name_in_the_older_form_is_refused():
    with pytest.raises(ValueError, match=r'1 parameter\(s\) are written \{variable\}_\{vessel\} '
                                         r'\(e.g. "C_aortic_root"\).*cuflynx-upgrade parameter-names'):
        host_model_names(['C_aortic_root', 'R'], VESSELS, 'p.csv')
    with pytest.raises(ValueError, match='"E_heart_lv"'):
        host_model_names(['E_heart_lv'], VESSELS, 'p.csv')


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


def _write_old_form(path, rename):
    rows = _read(path)
    for row in rows:
        row['variable_name'] = rename(row['variable_name'])
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.integration
def test_a_host_file_in_the_older_form_is_refused_and_converted(tmp_path):
    from libcuflynx.upgrade import main as upgrade
    current = _read(os.path.join(RESOURCES_DIR, f'{PREFIX}_parameters.csv'))
    assert any(r['variable_name'] == 'aortic_root/C' for r in current)
    params = _copy_resources(str(tmp_path / 'r'))
    _write_old_form(params, model_name)
    with pytest.raises(ValueError, match='cuflynx-upgrade parameter-names'):
        _generate(str(tmp_path / 'r'), str(tmp_path / 'g_old'))
    assert upgrade(['parameter-names', '--module-array',
                    str(tmp_path / 'r' / f'{PREFIX}_module_array.csv')]) == 0
    assert _read(params) == current
    _generate(str(tmp_path / 'r'), str(tmp_path / 'g'))


@pytest.mark.integration
def test_a_misspelt_vessel_is_an_error(tmp_path):
    params = _copy_resources(str(tmp_path / 'r'))
    with open(params) as f:
        text = f.read()
    columns = text.splitlines()[0].count(',') + 1
    row = ['aortic_rot/C', 'm6_per_N', '1e-9', 'typo'] + [''] * (columns - 4)
    with open(params, 'w') as f:
        f.write(text.rstrip('\n') + '\n' + ','.join(row) + '\n')
    with pytest.raises(ValueError, match=r'aortic_rot/C \(no vessel "aortic_rot"'):
        _generate(str(tmp_path / 'r'), str(tmp_path / 'g'))


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
    _, rows = load_expanded_vessel_records(path, load_supermodule_registry(files),
                                           load_component_registry(files))
    return {r['variable_name']: (r['value'], r['data_reference']) for r in rows}


def _old_row_name(name):
    '''<sub>/<var> written {var}_{sub} (nested levels joined with "_").'''
    *path, variable = name.split('/')
    return f'{variable}_{"_".join(path)}' if path else name


@pytest.mark.integration
def test_a_supermodule_parameterisation_in_the_older_form_is_refused_and_converted(
        tmp_path, library_files):
    from libcuflynx.upgrade import main as upgrade
    modules, files = library_files
    original = next(f for f in files if f.endswith('soma_sympathetic_modules_config.json'))
    copy = tmp_path / 'soma_copy'
    shutil.copytree(os.path.dirname(original), copy)
    config = str(copy / os.path.basename(original))
    parameterisation = str(copy / 'parameterisations' / 'default' / 'default_parameters.csv')
    if not any('/' in r['variable_name'] for r in _read(parameterisation)):
        # the library is not converted yet: convert the copy to have a reference
        assert upgrade(['parameter-names', '--supermodule-config', config,
                        '--module-library-dir', modules, '--no-builtin-modules']) == 0
    current = _read(parameterisation)
    assert any(r['variable_name'].startswith('membrane/') for r in current)
    files_with_copy = [f for f in files if f != original] + [config]
    os.makedirs(tmp_path / 'current')
    expected = _soma_rows(tmp_path / 'current', files_with_copy)

    _write_old_form(parameterisation, _old_row_name)
    os.makedirs(tmp_path / 'old')
    with pytest.raises(ValueError, match='cuflynx-upgrade parameter-names --supermodule-config'):
        _soma_rows(tmp_path / 'old', files_with_copy)
    assert upgrade(['parameter-names', '--supermodule-config', config, '--module-library-dir',
                    modules, '--no-builtin-modules']) == 0
    assert _read(parameterisation) == current
    os.makedirs(tmp_path / 'after')
    assert _soma_rows(tmp_path / 'after', files_with_copy) == expected


@pytest.mark.unit
def test_a_parameterisation_names_its_variables_without_a_vessel(tmp_path):
    from libcuflynx.utilities.parameterisations import component_parameterisation_rows
    directory = tmp_path / 'pulse' / 'versions' / 'v1'
    os.makedirs(directory / 'parameterisations' / 'x')
    (directory / 'parameterisations' / 'x' / 'x_parameters.csv').write_text(
        'variable_name,units,value,data_reference\ns1/mean,m3_per_s,1,r\n')
    entry = {'module_type': 'pulse_src', 'module_subtype': 'v1', 'variables_and_units': [],
             'config_path': str(directory / 'pulse_v1_modules_config.json')}
    record = {'name': 's1', 'module_type': 'pulse_src', 'module_subtype': 'v1', 'parameterisation': 'x'}
    with pytest.raises(ValueError, match='named without a vessel'):
        component_parameterisation_rows([record], {('pulse_src', 'v1'): entry})
