"""cuflynx-upgrade (libcuflynx/upgrade.py): each command converts a file in an older form into
one the readers accept, leaves a current file alone, and ``--dry-run`` writes nothing."""
import json
import os
import shutil

import pytest

from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser
from libcuflynx.upgrade import main as upgrade
from libcuflynx.utilities.config_schemas import (load_supermodule_registry, normalise_module_config,
                                                 read_module_array_records)
from libcuflynx.utilities import parameters_file
from libcuflynx.utilities.supermodules import instance_values_from_version

pytestmark = pytest.mark.unit

OLD_COMPONENT = {'vessel_type': 'src', 'BC_type': 'nn', 'module_format': 'cellml',
                 'module_file': 'm.cellml', 'module_type': 'src_type', 'default_instance': 'p',
                 'entrance_ports': [],
                 'exit_ports': [{'port_type': 'flow_port', 'variables': ['v'],
                                 'multi_port': 'False'}],
                 'general_ports': [], 'variables_and_units': [['v', 'm3_per_s', 'access', 'variable']]}
OLD_SUPERMODULE = {'vessel_type': 'pair', 'BC_type': 'supermodule', 'module_format': 'supermodule',
                   'default_parameters': 'pair_defaults.csv',
                   'submodules': [{'name': 'a', 'vessel_type': 'src', 'BC_type': 'nn',
                                   'out_vessels': ['b']},
                                  {'name': 'b', 'vessel_type': 'src', 'BC_type': 'nn',
                                   'inp_vessels': ['a']}]}


def _write(path, obj):
    path.write_text(json.dumps(obj, indent=2))
    return str(path)


def test_a_module_config_is_converted(tmp_path):
    path = _write(tmp_path / 'm_modules_config.json', [OLD_COMPONENT, OLD_SUPERMODULE])
    (tmp_path / 'pair_defaults.csv').write_text('variable_name,units,value,data_reference\n'
                                                'a/v,m3_per_s,1,r\n')
    with pytest.raises(ValueError, match='cuflynx-upgrade module-config'):
        normalise_module_config(json.loads(open(path).read()), path)
    assert upgrade(['module-config', path]) == 0
    entries = normalise_module_config(json.loads(open(path).read()), path)
    component, supermodule = entries
    assert (component['module_type'], component['module_subtype'],
            component['component_file'], component['component_type']) == \
        ('src', 'nn', 'm.cellml', 'src_type')
    assert component['default_parameterisation'] == 'p'
    assert 'multi_port' not in component['exit_ports'][0]
    assert supermodule['default_parameterisation'] == 'default'
    # the inline submodules: their versions in the entry, the records in its module array
    assert 'submodules' not in supermodule
    assert supermodule['submodule_versions'] == [{'module_type': 'src', 'module_subtype': 'nn'}]
    records = read_module_array_records(str(tmp_path / 'pair_supermodule_module_array.json'))
    assert [(r['name'], r['module_type'], r['module_subtype'], r['inp_instances'],
             r['out_instances']) for r in records] == [('a', 'src', 'nn', [], ['b']),
                                                       ('b', 'src', 'nn', ['a'], [])]
    registered = load_supermodule_registry([path])[('pair', 'supermodule')]
    assert [s['name'] for s in registered['submodules']] == ['a', 'b']
    # the default_parameters row a/v is a's value in the template, the default parameterisation
    assert instance_values_from_version(registered) == {
        'a': {'v': {'value': '1', 'units': 'm3_per_s', 'data_reference': 'r'}}, 'b': {}}
    moved = tmp_path / 'parameterisations' / 'default' / 'default_parameters.json'
    assert moved.is_file() and not moved.with_suffix('.csv').exists()
    assert not (tmp_path / 'pair_defaults.csv').exists()
    before = open(path).read()
    assert upgrade(['module-config', path]) == 0
    assert open(path).read() == before


def test_a_module_array_is_converted_and_renamed(tmp_path):
    json_path = _write(tmp_path / 'm_vessel_array.json', [
        {'name': 'heart', 'vessel_type': 'heart', 'BC_type': 'vp',
         'inp_vessels': 'venous_svc', 'out_vessels': ['aortic_root']},
        {'name': 'aortic_root', 'vessel_type': 'arterial', 'BC_type': 'vv', 'inp_vessels': ['heart']}])
    csv_path = tmp_path / 'c_vessel_array.csv'
    # cells padded with spaces, as many CSV arrays are
    csv_path.write_text('name,BC_type,vessel_type,inp_vessels,out_vessels\n'
                        'a,  nn, src,,b   \nb,nn,src,a,\nh, vp, heart_LVprop, a, b\n')
    assert upgrade(['module-array', json_path, str(csv_path)]) == 0
    assert not (tmp_path / 'm_vessel_array.json').exists()
    assert not csv_path.exists()
    records = read_module_array_records(str(tmp_path / 'm_module_array.json'))
    assert [(r['name'], r['module_type'], r['inp_instances'], r['out_instances'])
            for r in records] == [('heart', 'heart', ['venous_svc'], ['aortic_root']),
                                  ('aortic_root', 'arterial', ['heart'], [])]
    rows = read_module_array_records(str(tmp_path / 'c_module_array.csv'))
    assert [(r['name'], r['module_type'], r['out_instances']) for r in rows] == \
        [('a', 'src', ['b']), ('b', 'src', []), ('h', 'heart_LVprop', ['b'])]


@pytest.mark.integration
def test_a_one_output_heart_is_not_given_par_and_pvn(tmp_path, capsys):
    """A heart is a module like any other: neither the converter nor the generator adds a
    pulmonary artery and vein for a heart with one output. Its pulmonary ports are left
    unconnected, so generation stops at their boundary conditions and names them."""
    from libcuflynx.scripts.script_generate_with_new_architecture import \
        generate_with_new_architecture
    resources = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'resources')
    work = tmp_path / 'resources'
    work.mkdir()
    for suffix in ('_module_array.csv', '_parameters.json'):
        shutil.copy(os.path.join(resources, 'cvs_model_0d' + suffix), work)
    array = work / 'cvs_model_0d_module_array.csv'
    rows = [r for r in array.read_text().splitlines() if not r.startswith(('par,', 'pvn,'))]
    rows = [r.replace(' pvn,', ',').replace(' par', '') if r.startswith('heart,') else r
            for r in rows]
    array.write_text('\n'.join(rows) + '\n')
    parameters = work / 'cvs_model_0d_parameters.json'
    document = json.loads(parameters.read_text())
    for instance in ('par', 'pvn'):
        document.pop(instance, None)
    parameters.write_text(json.dumps(document, indent=2))
    before = array.read_text()
    assert upgrade(['module-array', str(array)]) == 0
    assert array.read_text() == before
    config = {'file_prefix': 'cvs_model_0d', 'input_param_file': 'cvs_model_0d_parameters.json',
              'model_type': 'cellml', 'solver': 'CVODE_myokit', 'resources_dir': str(work),
              'generated_models_dir': str(tmp_path / 'generated'), 'DEBUG': False}
    capsys.readouterr()
    assert not generate_with_new_architecture(False, config)
    out = capsys.readouterr().out
    for variable in ('u_par', 'v_pvn'):
        assert f'heart/{variable}: a boundary condition of "heart" that no connected module' in out
    cellml = (tmp_path / 'generated' / 'cvs_model_0d' / 'cvs_model_0d.cellml').read_text()
    assert 'name="par"' not in cellml and 'name="pvn"' not in cellml
    assert 'zero_flow' not in cellml


@pytest.mark.parametrize('doc', [
    [{'data_item_name': 'x', 'data_type': 'constant', 'unit': 'm', 'operands': ['a/x'],
      'value': 1.0, 'std': 0.1}],
    {'data_items': [{'data_item_name': 'x', 'data_type': 'constant', 'unit': 'm',
                     'operands': ['a/x'], 'value': 1.0, 'std': 0.1}]},
])
def test_an_obs_data_without_protocol_info_is_converted(tmp_path, doc):
    path = _write(tmp_path / 'm_obs_data.json', doc)
    with pytest.raises(ValueError, match='no protocol_info.*cuflynx-upgrade obs-data'):
        ObsAndParamDataParser().parse_obs_data_json(param_id_obs_path=path)
    inputs = tmp_path / 'user_inputs.yaml'
    inputs.write_text('pre_time: 20\nsim_time: 2\n')
    assert upgrade(['obs-data', path, '--dry-run', '--user-inputs', str(inputs)]) == 0
    assert json.loads(open(path).read()) == doc
    assert upgrade(['obs-data', path, '--user-inputs', str(inputs)]) == 0
    parsed = ObsAndParamDataParser().parse_obs_data_json(param_id_obs_path=path)
    assert parsed['protocol_info']['pre_times'] == [20.0]
    assert parsed['protocol_info']['sim_times'] == [[2.0]]
    assert len(parsed['gt_df']) == 1


def test_obs_data_needs_the_times(tmp_path):
    path = _write(tmp_path / 'm_obs_data.json', [])
    with pytest.raises(SystemExit, match='needs pre_time and sim_time'):
        upgrade(['obs-data', path])


def test_explicit_times_win_over_user_inputs(tmp_path):
    path = _write(tmp_path / 'm_obs_data.json', [])
    inputs = tmp_path / 'user_inputs.yaml'
    inputs.write_text('pre_time: 20\nsim_time: 2\n')
    assert upgrade(['obs-data', path, '--user-inputs', str(inputs), '--pre-time', '5']) == 0
    protocol_info = json.load(open(path))['protocol_info']
    assert protocol_info['pre_times'] == [5.0]
    assert protocol_info['sim_times'] == [[2.0]]


def test_a_heart_with_no_outputs_is_left_alone(tmp_path):
    """A heart module tested on its own has no neighbours, and its array is left as it is."""
    path = _write(tmp_path / 'm_module_array.json',
                  [{'name': 'mod', 'module_type': 'heart', 'module_subtype': 'vp_simple'}])
    before = open(path).read()
    assert upgrade(['module-array', path]) == 0
    assert open(path).read() == before


# ------------------------------------------------------------------- supermodule submodules

def _rows(path):
    """``[(name, value)]`` of a parameterisation's file, its globals ``global/<variable>``."""
    return [(r['variable_name'], r['value'])
            for r in parameters_file.read_parameter_rows(str(path), kind=parameters_file.MODULE)]


def _parameters(path, rows):
    """A parameterisation's parameters file (nested JSON) of ``rows`` ``(name, value)``."""
    parameters_file.write_parameter_rows(str(path), [
        {'variable_name': n, 'units': 'u', 'value': v, 'data_reference': 'ref'} for n, v in rows],
        parameters_file.MODULE)


def _library(root):
    """A module library: component ``src`` v1 (``k`` local, ``T`` a global constant) with
    parameterisations ``default`` and ``fast``, supermodule ``inner`` v1 (current form)
    with parameterisations ``d`` (its default: its template) and ``p``, and a version still in an
    older form, which the readers refuse and the converter reads past."""
    src = root / 'src' / 'versions' / 'v1'
    src.mkdir(parents=True)
    _write(src / 'src_v1_modules_config.json', [{
        'module_type': 'src', 'module_subtype': 'v1', 'component_file': 'src.cellml',
        'component_type': 'src_type', 'default_parameterisation': 'default',
        'variables_and_units': [['k', 'm', 'access', 'constant'],
                                ['T', 'K', 'access', 'global_constant']]}])
    _parameters(src / 'parameterisations' / 'default' / 'default_parameters.json',
                [('k', '1'), ('global/T', '300')])
    _parameters(src / 'parameterisations' / 'fast' / 'fast_parameters.json',
                [('k', '5'), ('global/T', '310'), ('global/G', '2')])
    inner = root / 'inner' / 'versions' / 'v1'
    inner.mkdir(parents=True)
    _write(inner / 'inner_v1_modules_config.json', [{
        'module_type': 'inner', 'module_subtype': 'v1', 'module_format': 'supermodule',
        'default_parameterisation': 'd',
        'submodule_versions': [{'module_type': 'src', 'module_subtype': 'v1'}]}])
    _write(inner / 'inner_v1_module_array.json', [{'name': 'g', 'module_type': 'src',
                                                    'module_subtype': 'v1'}])
    _parameters(inner / 'parameterisations' / 'd' / 'd_parameters.json', [('g/k', '1')])
    _parameters(inner / 'parameterisations' / 'p' / 'p_parameters.json',
                [('g/k', '7'), ('global/T', '320')])
    legacy = root / 'legacy' / 'versions' / 'v1'
    legacy.mkdir(parents=True)
    _write(legacy / 'legacy_v1_modules_config.json', [dict(OLD_COMPONENT, BC_type='v1')])
    return str(root)


def _csv_parameters(path, rows):
    """A parameterisation in the older form, a parameters CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('variable_name,units,value,data_reference\n'
                    + ''.join(f'{n},u,{v},ref\n' for n, v in rows))


def _sub(name, module_type='src', parameterisation=None, **extra):
    record = {'name': name, 'module_type': module_type, 'module_subtype': 'v1', **extra}
    if parameterisation is not None:
        record['parameterisation'] = parameterisation
    return record


def _supermodule(tmp_path, submodules, **entry):
    directory = tmp_path / 'sup' / 'versions' / 'v1'
    directory.mkdir(parents=True)
    return _write(directory / 'sup_v1_modules_config.json', [dict(
        {'module_type': 'sup', 'module_subtype': 'v1', 'module_format': 'supermodule',
         'submodules': submodules}, **entry)]), directory


def _flat(node, path=()):
    out = {}
    for key, value in node.items():
        if parameters_file.is_leaf(value):
            out['/'.join(path + (key,))] = value['value']
        else:
            out.update(_flat(value, path + (key,)))
    return out


def _template(directory, name='default'):
    """``{instance: {variable or path: value}}`` of the converted supermodule's parameterisation
    ``name`` (its template, when it is the default), without its globals."""
    doc = parameters_file.read_parameters_file(
        str(directory / 'parameterisations' / name / f'{name}_parameters.json'),
        kind=parameters_file.MODULE)
    return {k: _flat(v) for k, v in doc.items() if k not in ('schema_version', 'global')}


def test_each_instance_s_values_are_written_to_the_template(tmp_path):
    library = _library(tmp_path / 'lib')
    path, directory = _supermodule(
        tmp_path, [_sub('a', parameterisation='fast', out_instances=['b']),
                   _sub('b', parameterisation='default', inp_instances=['a']), _sub('c')],
        default_parameterisation='base')
    # the supermodule's own parameterisations, in the older form (CSV), converted with it
    _csv_parameters(directory / 'parameterisations' / 'base' / 'base_parameters.csv',
                    [('c/k', '9'), ('T', '305')])
    _csv_parameters(directory / 'parameterisations' / 'alt' / 'alt_parameters.csv', [('a/k', '8')])
    assert upgrade(['module-config', path, '--module-library-dir', library,
                    '--no-builtin-modules']) == 0
    entry = json.loads(open(path).read())[0]
    assert entry['submodule_versions'] == [{'module_type': 'src', 'module_subtype': 'v1'}]
    assert entry['default_parameterisation'] == 'base'
    records = read_module_array_records(str(directory / 'sup_v1_module_array.json'))
    assert [r['name'] for r in records] == ['a', 'b', 'c']
    assert not any('parameterisation' in r for r in records)
    assert records[0]['out_instances'] == ['b']
    # a's values from "fast", b's and c's from src's default, c/k of the supermodule's default
    # parameterisation over c's: the template, in "base"
    assert _template(directory, 'base') == {'a': {'k': '5'}, 'b': {'k': '1'}, 'c': {'k': '9'}}
    # with base's global, and a's globals from "fast" where they are missing
    assert [r for r in _rows(directory / 'parameterisations' / 'base' / 'base_parameters.json')
            if r[0].startswith('global/')] == [('global/T', '305'), ('global/G', '2')]
    # another parameterisation keeps its instances' values and gets the globals too; it
    # replaces the template for a model instance naming it, so it takes the template's values
    # it does not set
    assert _rows(directory / 'parameterisations' / 'alt' / 'alt_parameters.json') == [
        ('global/T', '310'), ('global/G', '2'), ('a/k', '8'), ('b/k', '1'), ('c/k', '9')]
    assert not list(directory.glob('parameterisations/*/*.csv'))
    load_supermodule_registry([path])


def test_a_supermodule_without_parameterisations_gets_a_default_one(tmp_path):
    library = _library(tmp_path / 'lib')
    path, directory = _supermodule(tmp_path, [_sub('a', parameterisation='fast')])
    assert upgrade(['module-config', path, '--module-library-dir', library,
                    '--no-builtin-modules']) == 0
    entry = json.loads(open(path).read())[0]
    assert entry['default_parameterisation'] == 'default'
    assert _template(directory) == {'a': {'k': '5'}}
    assert _rows(directory / 'parameterisations' / 'default' / 'default_parameters.json') == [
        ('global/T', '310'), ('global/G', '2'), ('a/k', '5')]


def test_a_submodule_at_its_default_is_written_with_its_default_s_values(tmp_path):
    library = _library(tmp_path / 'lib')
    path, directory = _supermodule(tmp_path, [_sub('a', parameterisation='default')])
    assert upgrade(['module-config', path, '--module-library-dir', library,
                    '--no-builtin-modules']) == 0
    entry = json.loads(open(path).read())[0]
    assert entry['default_parameterisation'] == 'default'
    assert 'parameterisation' not in read_module_array_records(
        str(directory / 'sup_v1_module_array.json'))[0]
    # its version's default sets no global the supermodule's must carry
    assert _rows(directory / 'parameterisations' / 'default' / 'default_parameters.json') == [
        ('a/k', '1')]


def test_a_nested_supermodule_s_values_are_paths_into_it(tmp_path):
    library = _library(tmp_path / 'lib')
    path, directory = _supermodule(
        tmp_path, [_sub('in', module_type='inner', parameterisation='p'), _sub('s')])
    assert upgrade(['module-config', path, '--module-library-dir', library,
                    '--no-builtin-modules']) == 0
    assert json.loads(open(path).read())[0]['submodule_versions'] == [
        {'module_type': 'inner', 'module_subtype': 'v1'},
        {'module_type': 'src', 'module_subtype': 'v1'}]
    # g/k is inner's submodule g; T is a global and goes to the supermodule's parameterisation
    assert _template(directory) == {'in': {'g/k': '7'}, 's': {'k': '1'}}
    assert ('global/T', '320') in _rows(
        directory / 'parameterisations' / 'default' / 'default_parameters.json')


def test_a_submodule_version_without_its_library_is_refused_and_nothing_written(tmp_path):
    path, directory = _supermodule(tmp_path, [_sub('a', parameterisation='fast')])
    before = open(path).read()
    with pytest.raises(SystemExit, match=r'submodule "a" is of '
                                         r"\('src', 'v1'\).*pass --module-library-dir"):
        upgrade(['module-config', path, '--no-builtin-modules'])
    assert open(path).read() == before
    assert sorted(os.listdir(directory)) == ['sup_v1_modules_config.json']


def test_an_existing_module_array_is_not_overwritten(tmp_path):
    path, directory = _supermodule(tmp_path, [_sub('a')])
    (directory / 'sup_v1_module_array.json').write_text('[]')
    with pytest.raises(SystemExit, match='exists already'):
        upgrade(['module-config', path])
    assert (directory / 'sup_v1_module_array.json').read_text() == '[]'


def test_a_dry_run_writes_no_module_array(tmp_path):
    library = _library(tmp_path / 'lib')
    path, directory = _supermodule(tmp_path, [_sub('a', parameterisation='fast')])
    before = open(path).read()
    assert upgrade(['module-config', path, '--dry-run', '--module-library-dir', library,
                    '--no-builtin-modules']) == 0
    assert open(path).read() == before
    assert sorted(os.listdir(directory)) == ['sup_v1_modules_config.json']


# ------------------------------------------------------------------------ supermodule-instances

def _model_library(tmp_path):
    """The converted ``sup`` (instances a, from src's "fast", and in, an inner from its "p") over
    a library of src and inner, with sup's parameterisation "alt" setting a/k."""
    library = _library(tmp_path / 'lib')
    shutil.rmtree(os.path.join(library, 'legacy'))
    path, directory = _supermodule(tmp_path, [_sub('a', parameterisation='fast'),
                                              _sub('in', module_type='inner', parameterisation='p')])
    _parameters(directory / 'parameterisations' / 'alt' / 'alt_parameters.json', [('a/k', '8')])
    assert upgrade(['module-config', path, '--module-library-dir', library,
                    '--no-builtin-modules']) == 0
    model = tmp_path / 'model'
    model.mkdir()
    array = _write(model / 'm_module_array.json', [
        {'name': 'S', 'module_type': 'sup', 'module_subtype': 'v1'},
        {'name': 'S2', 'module_type': 'sup', 'module_subtype': 'v1', 'parameterisation': 'alt'},
        {'name': 'k', 'module_type': 'src', 'module_subtype': 'v1'}])
    return array, ['--module-library-dir', library, '--module-library-dir', str(tmp_path / 'sup'),
                   '--no-builtin-modules']


def _instance_values(model, name):
    """``{variable path: value}`` of the instance ``name`` in the model's parameters file."""
    return _flat(parameters_file.read_parameters_file(str(model / 'm_parameters.json'))[name])


def test_a_model_s_supermodule_instances_get_their_values(tmp_path):
    array, sources = _model_library(tmp_path)
    model = tmp_path / 'model'
    assert upgrade(['supermodule-instances', array, '--dry-run'] + sources) == 0
    assert not (model / 'm_parameters.json').exists()
    assert upgrade(['supermodule-instances', array] + sources) == 0
    assert not (model / 'm_supermodule_parameters').exists()
    # a copy of the version's template, with the nested inner's values as paths under sup's own
    assert _instance_values(model, 'S') == {'a/k': '5', 'in/g/k': '7'}
    # S2's parameterisation "alt", which module-config completed with the default's in/g/k
    assert _instance_values(model, 'S2') == {'a/k': '8', 'in/g/k': '7'}
    # the component k has no entry: its values come from its parameterisation
    assert 'k' not in parameters_file.read_parameters_file(str(model / 'm_parameters.json'))


def test_an_instance_s_values_are_kept_unless_replaced(tmp_path, capsys):
    array, sources = _model_library(tmp_path)
    model = tmp_path / 'model'
    assert upgrade(['supermodule-instances', array] + sources) == 0
    path = str(model / 'm_parameters.json')
    doc = parameters_file.read_parameters_file(path)
    doc['S']['a']['k']['value'] = '42'
    parameters_file.write_parameters_file(path, doc)
    capsys.readouterr()
    assert upgrade(['supermodule-instances', array] + sources) == 0
    assert '"S" has values in the model\'s parameters file; kept' in capsys.readouterr().out
    assert _instance_values(model, 'S')['a/k'] == '42'
    assert upgrade(['supermodule-instances', array, '--instance', 'S', '--replace'] + sources) == 0
    assert _instance_values(model, 'S')['a/k'] == '5'
    with pytest.raises(SystemExit, match=r"\['k'\] are not supermodule instances there"):
        upgrade(['supermodule-instances', array, '--instance', 'k'] + sources)


# ---------------------------------------------------------------------------------- parameters

def _doc(path, kind=parameters_file.MODEL):
    return parameters_file.read_parameters_file(str(path), kind=kind)


def test_a_model_s_parameters_csv_becomes_nested_json(tmp_path, capsys):
    csv_path = tmp_path / 'm_parameters.csv'
    csv_path.write_text('variable_name,units,value,data_reference,sourced\n'
                        'aortic_root/C,m3_per_Pa,1.2e-8,Liang2009,yes\n'
                        'heart/lv/E_A,J_per_m6,366575000,ref,\n'
                        'heart__ra/E_A,J_per_m6,7998000,ref,\n'
                        'R,J_per_mol_K,8.314,known,\n'
                        'global/T,kelvin,310,known,\n'
                        'aortic_root/C,m3_per_Pa,9,second,\n')
    with pytest.raises(ValueError, match=r'm_parameters\.csv: libcuflynx reads parameter values '
                                         r'from nested JSON.*cuflynx-upgrade parameters'):
        parameters_file.read_parameter_rows(str(csv_path))
    assert upgrade(['parameters', str(csv_path), '--dry-run']) == 0
    assert csv_path.exists() and not (tmp_path / 'm_parameters.json').exists()
    capsys.readouterr()
    assert upgrade(['parameters', str(csv_path)]) == 0
    assert "rows set twice, the first kept: ['aortic_root/C']" in capsys.readouterr().out
    assert not csv_path.exists()
    text = (tmp_path / 'm_parameters.json').read_text()
    # a number keeps the text it was written with
    assert '"value": 1.2e-8' in text
    assert _doc(tmp_path / 'm_parameters.json') == {
        'schema_version': 1,
        'global': {'R': {'value': '8.314', 'units': 'J_per_mol_K', 'data_reference': 'known'},
                   'T': {'value': '310', 'units': 'kelvin', 'data_reference': 'known'}},
        'aortic_root': {'C': {'value': '1.2e-8', 'units': 'm3_per_Pa',
                              'data_reference': 'Liang2009', 'sourced': 'yes'}},
        'heart': {'lv': {'E_A': {'value': '366575000', 'units': 'J_per_m6',
                                 'data_reference': 'ref'}},
                  'ra': {'E_A': {'value': '7998000', 'units': 'J_per_m6',
                                 'data_reference': 'ref'}}}}
    # converted already: nothing to do
    assert upgrade(['parameters', str(tmp_path)]) == 0


def test_a_model_s_older_instance_files_move_into_its_parameters_file(tmp_path):
    (tmp_path / 'm_parameters.csv').write_text('variable_name,units,value,data_reference\n'
                                               'S/a/k,m,9,host\n')
    directory = tmp_path / 'm_supermodule_parameters'
    directory.mkdir()
    _write(directory / 'S.json', {'schema_version': 1, 'instances': {
        'a': {'parameterisation': 'fast', 'parameters': {
            'k': {'value': 5, 'units': 'm', 'data_reference': 'file'},
            'j': {'value': 1.5e-08, 'units': 'm', 'data_reference': 'file'}}},
        'in': {'parameterisation': 'p', 'parameters': {
            'g/k': {'value': 7, 'units': 'm', 'data_reference': 'file'}}},
        'e': {'parameterisation': 'default', 'parameters': {}}}})
    assert upgrade(['parameters', str(tmp_path)]) == 0
    assert not directory.exists()
    doc = _doc(tmp_path / 'm_parameters.json')
    # the model's own row wins over the instance file's
    assert _flat(doc['S']) == {'a/k': '9', 'a/j': '1.5e-08', 'in/g/k': '7'}
    assert doc['S']['e'] == {}


def test_a_library_s_parameterisations_and_version_files_are_converted(tmp_path):
    src = tmp_path / 'lib' / 'src' / 'versions' / 'v1'
    src.mkdir(parents=True)
    _write(src / 'src_v1_modules_config.json', [{
        'module_type': 'src', 'module_subtype': 'v1', 'component_file': 'src.cellml',
        'component_type': 'src_type', 'default_parameterisation': 'default',
        'variables_and_units': [['k', 'm', 'access', 'constant'],
                                ['T', 'K', 'access', 'global_constant']]}])
    _csv_parameters(src / 'parameterisations' / 'default' / 'default_parameters.csv',
                    [('k', '1'), ('T', '300')])
    _csv_parameters(src / 'parameterisations' / 'fit' / 'fit_calibrated_parameters.csv',
                    [('k', '2.5')])
    sup = tmp_path / 'lib' / 'sup' / 'versions' / 'v1'
    sup.mkdir(parents=True)
    _write(sup / 'sup_v1_modules_config.json', [{
        'module_type': 'sup', 'module_subtype': 'v1', 'module_format': 'supermodule',
        'default_parameterisation': 'default',
        'submodule_versions': [{'module_type': 'src', 'module_subtype': 'v1'}]}])
    _write(sup / 'sup_v1_module_array.json', [{'name': 'a', 'module_type': 'src',
                                               'module_subtype': 'v1'},
                                              {'name': 'b', 'module_type': 'src',
                                               'module_subtype': 'v1'}])
    _csv_parameters(sup / 'parameterisations' / 'default' / 'default_parameters.csv',
                    [('global/T', '305')])
    _write(sup / 'sup_v1_parameters.json', {'schema_version': 1, 'instances': {
        'a': {'parameterisation': 'default', 'parameters': {
            'k': {'value': 3, 'units': 'm', 'data_reference': 'sup', 'sourced': 'no'}}},
        'b': {'parameterisation': 'default', 'parameters': {}}}})
    with pytest.raises(ValueError, match=r'sup_v1_parameters\.json holds its instances\' values '
                                         r'in the older form.*cuflynx-upgrade parameters'):
        load_supermodule_registry([str(sup / 'sup_v1_modules_config.json')])
    assert upgrade(['parameters', str(tmp_path / 'lib')]) == 0
    assert not list((tmp_path / 'lib').rglob('*.csv'))
    assert not (sup / 'sup_v1_parameters.json').exists()
    # a component's global constant goes under "global", its own variable at the top
    assert _rows(src / 'parameterisations' / 'default' / 'default_parameters.json') == [
        ('global/T', '300'), ('k', '1')]
    assert _rows(src / 'parameterisations' / 'fit' / 'fit_calibrated_parameters.json') == [
        ('k', '2.5')]
    # the version's instance values are its template, in its default parameterisation
    assert _doc(sup / 'parameterisations' / 'default' / 'default_parameters.json',
                parameters_file.MODULE) == {
        'schema_version': 1,
        'global': {'T': {'value': '305', 'units': 'u', 'data_reference': 'ref'}},
        'a': {'k': {'value': '3', 'units': 'm', 'data_reference': 'sup', 'sourced': 'no'}},
        'b': {}}
    registry = load_supermodule_registry([str(sup / 'sup_v1_modules_config.json')])
    assert _flat(instance_values_from_version(registry[('sup', 'v1')], registry)) == {'a/k': '3'}


def test_a_parameters_file_read_and_written_again_reads_back_the_same(tmp_path):
    """schema_version stays the integer the schema requires; every value keeps its text."""
    path = tmp_path / 'm_parameters.json'
    path.write_text('{"schema_version": 1, "global": {"R": {"value": 8.314, "units": "u"}},\n'
                    ' "a": {"C": {"value": 1.2e-8, "units": "m", "data_reference": "r"},\n'
                    '       "n": {"value": 366575000, "units": "m", "sourced": "yes"}}}\n')
    doc = parameters_file.read_parameters_file(str(path))
    assert doc['schema_version'] == 1 and doc['a']['C']['value'] == '1.2e-8'
    parameters_file.write_parameters_file(str(path), doc)
    assert '"schema_version": 1,' in path.read_text() and '"value": 1.2e-8' in path.read_text()
    assert parameters_file.read_parameters_file(str(path)) == doc


def test_a_parameter_without_a_value_is_read_as_unset(tmp_path):
    """A blank value (a blank CSV cell, converted) is read as EMPTY_MUST_BE_FILLED, which
    generation reports as missing, rather than as a value."""
    path = tmp_path / 'm_parameters.csv'
    path.write_text('variable_name,units,value,data_reference\na/q_init,m3,,\na/C,m,1,r\n')
    assert upgrade(['parameters', str(path)]) == 0
    rows = {r['variable_name']: r['value']
            for r in parameters_file.read_parameter_rows(str(tmp_path / 'm_parameters.json'))}
    assert rows == {'a/q_init': 'EMPTY_MUST_BE_FILLED', 'a/C': '1'}
