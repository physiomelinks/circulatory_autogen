"""cuflynx-upgrade (libcuflynx/upgrade.py): each command converts a file in an older form into
one the readers accept, leaves a current file alone, and ``--dry-run`` writes nothing."""
import json
import os
import shutil

import pytest

from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser
from libcuflynx.upgrade import main as upgrade
from libcuflynx.utilities.config_schemas import (normalise_module_config, read_module_array_records)

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
    assert [s['out_instances'] for s in supermodule['submodules']] == [['b'], []]
    moved = tmp_path / 'parameterisations' / 'default' / 'default_parameters.csv'
    assert moved.read_text().endswith('a/v,m3_per_s,1,r\n')
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
    heart = records[0]
    assert heart['inp_instances'] == ['venous_svc', 'pvn']
    assert heart['out_instances'] == ['aortic_root', 'par']
    assert [(r['name'], r['module_type'], r['inp_instances'], r['out_instances'])
            for r in records[2:]] == [('par', 'arterial_simple', ['heart'], ['pvn']),
                                       ('pvn', 'arterial_simple', ['par'], ['heart'])]
    rows = read_module_array_records(str(tmp_path / 'c_module_array.csv'))
    assert [(r['name'], r['module_type'], r['out_instances']) for r in rows] == \
        [('a', 'src', ['b']), ('b', 'src', []), ('h', 'heart_LVprop', ['b', 'par']),
         ('par', 'arterial_simple', ['pvn']), ('pvn', 'arterial_simple', ['h'])]


@pytest.mark.integration
def test_a_one_output_heart_is_refused_until_converted(tmp_path):
    from libcuflynx.scripts.script_generate_with_new_architecture import \
        generate_with_new_architecture
    resources = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'resources')
    work = tmp_path / 'resources'
    work.mkdir()
    for suffix in ('_module_array.csv', '_parameters.csv'):
        shutil.copy(os.path.join(resources, 'cvs_model_0d' + suffix), work)
    array = work / 'cvs_model_0d_module_array.csv'
    rows = [r for r in array.read_text().splitlines() if not r.startswith(('par,', 'pvn,'))]
    rows = [r.replace(' pvn,', ',').replace(' par', '') if r.startswith('heart,') else r
            for r in rows]
    array.write_text('\n'.join(rows) + '\n')
    config = {'file_prefix': 'cvs_model_0d', 'input_param_file': 'cvs_model_0d_parameters.csv',
              'model_type': 'cellml', 'solver': 'CVODE_myokit', 'resources_dir': str(work),
              'generated_models_dir': str(tmp_path / 'generated'), 'DEBUG': False}
    with pytest.raises(ValueError, match='heart "heart" has one output.*cuflynx-upgrade module-array'):
        generate_with_new_architecture(False, config)
    assert upgrade(['module-array', str(array)]) == 0
    assert any(r.startswith('par,') for r in array.read_text().splitlines())


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


def test_a_heart_with_no_outputs_is_left_alone(tmp_path):
    """A heart module tested on its own has no neighbours; only a one-output heart lacks its
    pulmonary circuit."""
    path = _write(tmp_path / 'm_module_array.json',
                  [{'name': 'mod', 'module_type': 'heart', 'module_subtype': 'vp_simple'}])
    before = open(path).read()
    assert upgrade(['module-array', path]) == 0
    assert open(path).read() == before
