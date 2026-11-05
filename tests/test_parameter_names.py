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
from libcuflynx.utilities.supermodules import write_instance_parameters
from libcuflynx.utilities.parameter_names import (host_model_names, model_name, output_model_name,
                                                  split_path_name,
                                                  supermodule_row_name)
from _parameters import MODEL, MODULE, read_parameters, write_parameters

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
@pytest.mark.parametrize('written, expected', [
    ('SN/soma/membrane/V', 'SN__soma__membrane/V'),
    (' heart / lv / q ', 'heart__lv/q'),
    ('heart__lv/q', 'heart__lv/q'),
    ('aortic_root/u', 'aortic_root/u'),
    ('t', 't'),
])
def test_an_output_written_by_its_path_becomes_the_model_name(written, expected):
    assert output_model_name(written) == expected


@pytest.mark.unit
def test_the_resolver_and_candidates_take_an_output_by_its_path():
    from libcuflynx.parsers.PrimitiveParsers import model_qname_candidates
    from libcuflynx.solver_wrappers.name_resolver import VariableNameResolver
    resolver = VariableNameResolver([{'component': 'SN__soma__membrane_module', 'name': 'V'}], [])
    assert resolver.resolve('SN/soma/membrane/V') == resolver.resolve('SN__soma__membrane/V') == ('state', 0)
    keyed = [('state', {'SN__soma__membrane_module.V': 0})]
    assert VariableNameResolver.resolve_key('SN/soma/membrane/V', keyed, separator='.') == \
        ('state', 'SN__soma__membrane_module.V')
    assert model_qname_candidates('heart/lv/E') == model_qname_candidates('heart__lv/E')
    assert 'parameters/E_heart__lv' in model_qname_candidates('heart/lv/E')


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
           'heart__lv': {'E': 'constant', 'R': 'global_constant'}}


@pytest.mark.unit
def test_host_names_are_checked_against_the_model():
    names = host_model_names(['aortic_root/C', 'heart/lv/E', 'global/R', 'T'], VESSELS, 'p.csv')
    assert names == ['C_aortic_root', 'E_heart__lv', 'R', 'T']

    with pytest.raises(ValueError) as raised:
        host_model_names(['aortc_root/C', 'aortic_root/Cx', 'aortic_root/T'], VESSELS, 'p.csv')
    message = str(raised.value)
    assert '3 parameter(s) name something the model does not have' in message
    assert 'aortc_root/C (no vessel "aortc_root"' in message
    assert 'has no variable "Cx"' in message
    assert 'aortic_root/T ("T" is a global constant: write global/T or T)' in message


@pytest.mark.unit
def test_a_global_name_no_module_declares_is_refused():
    # r_0_par: par's module has no r_0, so it is neither a global nor the older form of a row
    with pytest.raises(ValueError) as raised:
        host_model_names(['r_0_par', 'global/Q', 'T'], VESSELS, 'p.csv')
    message = str(raised.value)
    assert '2 parameter(s) name something the model does not have' in message
    assert 'r_0_par (no module in the model has a global constant "r_0_par")' in message
    assert 'global/Q (no module in the model has a global constant "Q")' in message
    # a model with 1D vessels: the 1D generator reads rows such as rho by name
    assert host_model_names(['r_0_par', 'T'], VESSELS, 'p.csv', check_globals=False) == ['r_0_par', 'T']


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
    for suffix in ('_module_array.csv', '_parameters.json'):
        shutil.copy(os.path.join(RESOURCES_DIR, PREFIX + suffix), directory)
    return os.path.join(directory, f'{PREFIX}_parameters.json')


def _generate(resources, generated):
    from libcuflynx.scripts.script_generate_with_new_architecture import \
        generate_with_new_architecture
    config = {'file_prefix': PREFIX, 'input_param_file': f'{PREFIX}_parameters.json',
              'model_type': 'cellml', 'solver': 'CVODE_myokit', 'resources_dir': resources,
              'generated_models_dir': generated, 'DEBUG': False}
    assert generate_with_new_architecture(False, config)
    return os.path.join(generated, PREFIX)


def _read(path, kind=MODEL):
    return read_parameters(path, kind)


def _write_old_form(path, rename, kind=MODEL):
    write_parameters(path, [dict(row, variable_name=rename(row['variable_name']))
                            for row in _read(path, kind)], kind)


@pytest.mark.integration
@pytest.mark.parametrize('form', ['json', 'csv'])
def test_a_host_file_in_the_older_form_is_refused_and_converted(tmp_path, form):
    """Names written {variable}_{vessel} are refused naming parameter-names, which rewrites
    them in the nested JSON file, or in the parameters CSV they were first written in, which
    ``parameters`` then converts."""
    from libcuflynx.upgrade import main as upgrade
    current = _read(os.path.join(RESOURCES_DIR, f'{PREFIX}_parameters.json'))
    assert any(r['variable_name'] == 'aortic_root/C' for r in current)
    params = _copy_resources(str(tmp_path / 'r'))
    _write_old_form(params, model_name)
    with pytest.raises(ValueError, match='cuflynx-upgrade parameter-names'):
        _generate(str(tmp_path / 'r'), str(tmp_path / 'g_old'))
    if form == 'csv':
        rows = _read(params)
        os.remove(params)
        csv_path = params[:-len('.json')] + '.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    assert upgrade(['parameter-names', '--module-array',
                    str(tmp_path / 'r' / f'{PREFIX}_module_array.csv')]) == 0
    if form == 'csv':
        assert upgrade(['parameters', csv_path]) == 0
        assert not os.path.exists(csv_path)
    assert _read(params) == current
    _generate(str(tmp_path / 'r'), str(tmp_path / 'g'))


@pytest.mark.integration
def test_a_misspelt_vessel_is_an_error(tmp_path):
    params = _copy_resources(str(tmp_path / 'r'))
    write_parameters(params, _read(params) + [{'variable_name': 'aortic_rot/C',
                                               'units': 'm6_per_N', 'value': '1e-9',
                                               'data_reference': 'typo'}])
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
    """The soma's default rows (its parameterisation's globals, its submodules' defaults) in a
    model of one soma instance in ``tmp_path``, made from its version."""
    path = str(tmp_path / 'm_module_array.json')
    import json
    with open(path, 'w') as f:
        json.dump([{'name': 'soma', 'module_type': SOMA[0], 'module_subtype': SOMA[1],
                    'inp_instances': [], 'out_instances': []}], f)
    supermodules = load_supermodule_registry(files)
    write_instance_parameters(path, supermodules)
    _, rows = load_expanded_vessel_records(path, supermodules, load_component_registry(files))
    return {r['variable_name']: (r['value'], r['data_reference']) for r in rows}


def _old_row_name(name):
    '''<sub>/<var> written {var}_{sub} (nested levels joined with "_").'''
    if name.startswith('global/'):
        return name
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
    parameterisation = str(copy / 'parameterisations' / 'default' / 'default_parameters.json')
    current = _read(parameterisation, MODULE)
    assert any(r['variable_name'].startswith('membrane/') for r in current)
    files_with_copy = [f for f in files if f != original] + [config]
    os.makedirs(tmp_path / 'current')
    expected = _soma_rows(tmp_path / 'current', files_with_copy)

    _write_old_form(parameterisation, _old_row_name, MODULE)
    os.makedirs(tmp_path / 'old')
    with pytest.raises(ValueError, match='cuflynx-upgrade parameter-names --supermodule-config'):
        _soma_rows(tmp_path / 'old', files_with_copy)
    assert upgrade(['parameter-names', '--supermodule-config', config, '--module-library-dir',
                    modules, '--no-builtin-modules']) == 0
    assert sorted(_read(parameterisation, MODULE), key=lambda r: r['variable_name']) == \
        sorted(current, key=lambda r: r['variable_name'])
    os.makedirs(tmp_path / 'after')
    assert _soma_rows(tmp_path / 'after', files_with_copy) == expected


@pytest.mark.unit
def test_a_parameterisation_names_its_variables_without_a_vessel(tmp_path):
    from libcuflynx.utilities.parameterisations import component_parameterisation_rows
    directory = tmp_path / 'pulse' / 'versions' / 'v1'
    os.makedirs(directory / 'parameterisations' / 'x')
    write_parameters(directory / 'parameterisations' / 'x' / 'x_parameters.json',
                     'variable_name,units,value,data_reference\ns1/mean,m3_per_s,1,r\n', MODULE)
    entry = {'module_type': 'pulse_src', 'module_subtype': 'v1', 'variables_and_units': [],
             'config_path': str(directory / 'pulse_v1_modules_config.json')}
    record = {'name': 's1', 'module_type': 'pulse_src', 'module_subtype': 'v1', 'parameterisation': 'x'}
    with pytest.raises(ValueError, match="the module's own variables, at the top of its file"):
        component_parameterisation_rows([record], {('pulse_src', 'v1'): entry})
