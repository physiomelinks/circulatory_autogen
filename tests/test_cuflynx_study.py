"""The study manifest every analysis records itself in (``utilities/cuflynx_study.py``).

The file is a contract with CUFLynx: it opens a study from it, believing what it says.
So what is checked here is what a reader would get wrong if it drifted -- which entry is
first, which paths are relative, that a second analysis does not erase the first, and
that a broken write never fails the run it describes.
"""
import json
import os

import pytest

from libcuflynx.utilities import cuflynx_study

pytestmark = pytest.mark.unit


@pytest.fixture
def study(tmp_path):
    """A config for a study whose outputs live in ``tmp_path/out``, inputs elsewhere."""
    out = tmp_path / 'out'
    out.mkdir()
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    for name in ('model.cellml', 'obs.json', 'params.csv', 'ops.py', 'user_inputs.yaml'):
        (inputs / name).write_text('x')
    return {
        'param_id_output_dir': str(out),
        'file_prefix': 'demo',
        'model_path': str(inputs / 'model.cellml'),
        'param_id_obs_path': str(inputs / 'obs.json'),
        'params_for_id_path': str(inputs / 'params.csv'),
        'operation_funcs_external_path': str(inputs / 'ops.py'),
        'user_inputs_path': str(inputs / 'user_inputs.yaml'),
        'solver_info': {'solver': 'CVODE_myokit', 'MaximumStep': 1e-4},
        'dt': 0.01,
        'pre_time': 100,
        'sim_time': 10,
    }


def _raw(path):
    with open(path) as handle:
        return json.load(handle)


def test_named_after_the_study_in_its_output_dir(study):
    path = cuflynx_study.update_study(study)
    assert path == os.path.join(study['param_id_output_dir'], 'demo_cuflynx_study.json')
    study['study_name'] = 'other'
    assert cuflynx_study.manifest_path(study).endswith('other_cuflynx_study.json')


def test_records_the_inputs_funcs_and_solver(study):
    raw = _raw(cuflynx_study.update_study(study))
    assert raw['schema'] == 1
    assert raw['file_prefix'] == 'demo' and raw['study_name'] == 'demo'
    # every input is copied into the working dir and recorded relative to the manifest
    assert 'study_dir' not in raw
    assert raw['model'] == raw['base_model'] == os.path.join('inputs', 'model', 'model.cellml')
    assert raw['obs_data'] == os.path.join('inputs', 'obs_data', 'obs.json')
    assert raw['params_for_id'] == os.path.join('inputs', 'params_for_id', 'params.csv')
    assert raw['user_inputs'] == os.path.join('inputs', 'user_inputs', 'user_inputs.yaml')
    assert raw['user_funcs'] == {'operation': os.path.join('inputs', 'funcs', 'ops.py')}
    for relative in (raw['model'], raw['obs_data'], raw['user_funcs']['operation']):
        assert os.path.isfile(os.path.join(study['param_id_output_dir'], relative))
    # dt folded into solver_info, the shape CUFLynx adopts
    assert raw['solver_info'] == {'solver': 'CVODE_myokit', 'MaximumStep': 1e-4, 'dt': 0.01}
    assert raw['pre_time'] == 100 and raw['sim_time'] == 10
    assert raw['updated']


def test_copy_inputs_off_records_inputs_where_they_are(study):
    study['cuflynx_study_copy_inputs'] = False
    raw = _raw(cuflynx_study.update_study(study))
    assert raw['study_dir'] == os.path.abspath(study['param_id_output_dir'])
    # outside the study, so absolute: they keep resolving wherever the study is moved
    assert raw['model'] == raw['base_model'] == study['model_path']
    assert raw['obs_data'] == study['param_id_obs_path']
    assert raw['user_funcs'] == {'operation': study['operation_funcs_external_path']}
    assert not os.path.exists(os.path.join(study['param_id_output_dir'], 'inputs'))


def _all_files(directory):
    return sorted(os.path.relpath(os.path.join(root, name), directory)
                  for root, _, names in os.walk(directory) for name in names)


def test_inputs_are_copied_once_per_content(study):
    out = study['param_id_output_dir']
    cuflynx_study.update_study(study)
    first = _all_files(os.path.join(out, 'inputs'))
    cuflynx_study._BROUGHT.clear()        # as a second process would see it
    cuflynx_study.update_study(study)
    assert _all_files(os.path.join(out, 'inputs')) == first

    # a different file of the same name goes beside it, the first copy kept
    with open(study['param_id_obs_path'], 'w') as handle:
        handle.write('changed')
    raw = _raw(cuflynx_study.update_study(study))
    assert raw['obs_data'] != os.path.join('inputs', 'obs_data', 'obs.json')
    assert os.path.dirname(os.path.dirname(raw['obs_data'])) == os.path.join('inputs', 'obs_data')
    with open(os.path.join(out, raw['obs_data'])) as handle:
        assert handle.read() == 'changed'
    with open(os.path.join(out, 'inputs', 'obs_data', 'obs.json')) as handle:
        assert handle.read() == 'x'


def test_a_cellml_model_brings_the_files_it_imports(study, tmp_path):
    models = tmp_path / 'models'
    (models / 'shared').mkdir(parents=True)
    (models / 'shared' / 'units.cellml').write_text(
        '<model xmlns="http://www.cellml.org/cellml/2.0#" name="units"/>')
    (models / 'm.cellml').write_text(
        '<model xmlns="http://www.cellml.org/cellml/2.0#" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" name="m">'
        '<import xlink:href="shared/units.cellml"><import_units name="u" units_ref="u"/>'
        '</import></model>')
    study['model_path'] = str(models / 'm.cellml')
    raw = _raw(cuflynx_study.update_study(study))
    out = study['param_id_output_dir']
    assert raw['model'] == os.path.join('inputs', 'model', 'm.cellml')
    assert os.path.isfile(os.path.join(out, 'inputs', 'model', 'shared', 'units.cellml'))


def test_a_generated_models_system_is_copied(study, tmp_path):
    resources = tmp_path / 'resources'
    resources.mkdir()
    (resources / 'demo_module_array.json').write_text('{}')
    (resources / 'demo_parameters.json').write_text('{}')
    study['vessels_csv_abs_path'] = str(resources / 'demo_module_array.json')
    study['parameters_csv_abs_path'] = str(resources / 'demo_parameters.json')
    raw = _raw(cuflynx_study.update_study(study))
    assert raw['system'] == {
        'module_array': os.path.join('inputs', 'system', 'demo_module_array.json'),
        'parameters': os.path.join('inputs', 'system', 'demo_parameters.json')}
    read = cuflynx_study.read_study(study['param_id_output_dir'])
    assert all(os.path.isfile(p) for p in read['system'].values())


def test_a_nested_study_copies_into_the_larger_working_dir(study, tmp_path):
    working = tmp_path / 'out'
    study['param_id_output_dir'] = str(working / 'step' / 'param_id')
    study['cuflynx_working_dir'] = str(working)
    elsewhere = working / 'step' / 'sensitivity'
    raw = _raw(cuflynx_study.update_study(study, 'sensitivity',
                                          {'dir': str(elsewhere), 'method': 'sobol'}))
    assert raw['obs_data'] == os.path.join('..', '..', 'inputs', 'obs_data', 'obs.json')
    assert raw['sensitivity'][0]['dir'] == os.path.join('..', 'sensitivity')
    assert os.path.isfile(working / 'inputs' / 'obs_data' / 'obs.json')


def test_an_output_outside_the_working_dir_stays_absolute(study, tmp_path):
    elsewhere = str(tmp_path / 'sa_elsewhere')
    raw = _raw(cuflynx_study.update_study(study, 'sensitivity',
                                          {'dir': elsewhere, 'method': 'sobol'}))
    assert raw['sensitivity'][0]['dir'] == elsewhere


def test_a_zipped_and_moved_working_dir_resolves_inside_itself(study, tmp_path):
    import shutil
    out = study['param_id_output_dir']
    cuflynx_study.update_study(study, 'runs', {'dir': os.path.join(out, 'a'), 'method': 'm'})
    os.makedirs(os.path.join(out, 'a'))
    with open(cuflynx_study.manifest_path(study)) as handle:
        assert '"/' not in handle.read()
    archive = shutil.make_archive(str(tmp_path / 'study'), 'zip', out)
    moved = tmp_path / 'moved'
    shutil.unpack_archive(archive, str(moved))
    shutil.rmtree(tmp_path / 'inputs')    # the originals are gone too
    shutil.rmtree(out)
    _assert_resolves_inside(cuflynx_study.read_study(str(moved)), str(moved))


def _assert_resolves_inside(read, root):
    paths = [read[k] for k in cuflynx_study.PATH_KEYS if read.get(k)]
    paths += list((read.get('user_funcs') or {}).values())
    paths += list((read.get('system') or {}).values())
    paths += [e[k] for s in cuflynx_study.ENTRY_SECTIONS for e in read[s]
              for k in ('dir', 'obs_data') if e.get(k)]
    assert paths
    for path in paths:
        assert os.path.commonpath([path, root]) == root, path
        assert os.path.exists(path), path


def test_paths_inside_the_study_are_relative(study):
    run = os.path.join(study['param_id_output_dir'], 'genetic_algorithm_demo_obs')
    os.makedirs(run)
    raw = _raw(cuflynx_study.update_study(study, 'runs', {'dir': run, 'method': 'genetic_algorithm'}))
    assert raw['runs'][0]['dir'] == 'genetic_algorithm_demo_obs'


def test_analyses_accumulate_instead_of_overwriting(study):
    out = study['param_id_output_dir']
    run = os.path.join(out, 'genetic_algorithm_demo_obs')
    sa = os.path.join(out, 'sensitivity')
    cuflynx_study.update_study(study, 'runs', {'dir': run, 'method': 'genetic_algorithm'})
    cuflynx_study.update_study(study, 'sensitivity', {'dir': sa, 'method': 'sobol'})
    raw = _raw(cuflynx_study.update_study(study, 'runs', {'dir': run, 'method': 'mcmc'}))

    # one entry per directory, holding both analyses the directory contains
    assert len(raw['runs']) == 1
    assert raw['runs'][0]['method'] == 'mcmc'
    assert raw['runs'][0]['analyses'] == ['genetic_algorithm', 'mcmc']
    assert raw['sensitivity'][0]['dir'] == 'sensitivity'


def test_newest_run_first(study):
    out = study['param_id_output_dir']
    cuflynx_study.update_study(study, 'runs', {'dir': os.path.join(out, 'a'), 'method': 'm'})
    cuflynx_study.update_study(study, 'runs', {'dir': os.path.join(out, 'b'), 'method': 'm'})
    raw = _raw(cuflynx_study.update_study(study, 'runs', {'dir': os.path.join(out, 'a'), 'method': 'm'}))
    # CUFLynx opens a study on its first declared run
    assert [run['dir'] for run in raw['runs']] == ['a', 'b']


def test_two_studies_share_an_output_dir(study):
    cuflynx_study.update_study(study, 'runs', {'dir': os.path.join(study['param_id_output_dir'], 'a')})
    other = dict(study, study_name='second')
    cuflynx_study.update_study(other, 'runs', {'dir': os.path.join(study['param_id_output_dir'], 'b')})
    first = _raw(cuflynx_study.manifest_path(study))
    second = _raw(cuflynx_study.manifest_path(other))
    assert [r['dir'] for r in first['runs']] == ['a']
    assert [r['dir'] for r in second['runs']] == ['b']


def test_unknown_keys_survive_an_update(study):
    path = cuflynx_study.update_study(study)
    raw = _raw(path)
    raw['written_by_someone_else'] = {'keep': True}
    with open(path, 'w') as handle:
        json.dump(raw, handle)
    assert _raw(cuflynx_study.update_study(study))['written_by_someone_else'] == {'keep': True}


def test_calibrated_model_is_preferred_once_written(study):
    calibrated = os.path.join(study['param_id_output_dir'], 'demo_calibrated.cellml')
    with open(calibrated, 'w') as handle:
        handle.write('x')
    raw = _raw(cuflynx_study.update_study(study))
    assert raw['model'] == 'demo_calibrated.cellml'
    assert raw['base_model'] == os.path.join('inputs', 'model', 'model.cellml')


def test_emulator_is_declared(study, tmp_path):
    bundle = tmp_path / 'shared' / 'emulator'
    bundle.mkdir(parents=True)
    (bundle / 'emulator_metadata.json').write_text('{}')
    raw = _raw(cuflynx_study.update_study(study, emulator=str(bundle)))
    # a shared emulator outside the working dir is copied in, directory and all
    assert raw['emulator'] == os.path.join('inputs', 'emulator', 'emulator')
    assert os.path.isfile(os.path.join(study['param_id_output_dir'], raw['emulator'],
                                       'emulator_metadata.json'))


def test_turned_off(study):
    study['write_cuflynx_study'] = False
    assert cuflynx_study.update_study(study) is None
    assert not os.path.exists(cuflynx_study.manifest_path(study))


def test_a_corrupt_manifest_is_replaced_not_fatal(study, capsys):
    path = cuflynx_study.manifest_path(study)
    with open(path, 'w') as handle:
        handle.write('{"schema": 1, "runs": [')
    raw = _raw(cuflynx_study.update_study(study))
    assert raw['file_prefix'] == 'demo'
    assert 'could not read' in capsys.readouterr().out


def test_a_failed_write_warns_and_does_not_raise(study, monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(cuflynx_study, '_write_atomic', boom)
    assert cuflynx_study.update_study(study) is None
    assert 'disk full' in capsys.readouterr().out


def test_readable_like_any_other_output(study):
    path = cuflynx_study.update_study(study)
    with open(os.path.join(study['param_id_output_dir'], 'plain.txt'), 'w') as handle:
        handle.write('x')
    plain = os.stat(os.path.join(study['param_id_output_dir'], 'plain.txt')).st_mode & 0o777
    assert os.stat(path).st_mode & 0o777 == plain


def test_no_temporary_files_left_behind(study):
    cuflynx_study.update_study(study)
    leftovers = [n for n in os.listdir(study['param_id_output_dir']) if n.startswith('.cuflynx_study_')]
    assert leftovers == []


def test_bad_entries_are_refused_without_raising(study, capsys):
    assert cuflynx_study.update_study(study, 'nonsense', {'dir': 'x'}) is None
    assert cuflynx_study.update_study(study, 'runs', {}) is None
    assert 'WARNING' in capsys.readouterr().out


def test_read_resolves_paths(study):
    out = study['param_id_output_dir']
    cuflynx_study.update_study(study, 'runs', {'dir': os.path.join(out, 'a'), 'method': 'm'})
    read = cuflynx_study.read_study(out)
    assert read['runs'][0]['dir'] == os.path.join(os.path.abspath(out), 'a')
    assert read['user_funcs']['operation'] == os.path.join(os.path.abspath(out), 'inputs',
                                                           'funcs', 'ops.py')


def test_a_moved_copy_resolves_against_study_dir(study, tmp_path):
    study['cuflynx_study_copy_inputs'] = False
    out = study['param_id_output_dir']
    path = cuflynx_study.update_study(study, 'runs', {'dir': os.path.join(out, 'a')})
    elsewhere = tmp_path / 'downloads' / 'demo_cuflynx_study.json'
    elsewhere.parent.mkdir()
    elsewhere.write_text(open(path).read())
    assert cuflynx_study.read_study(str(elsewhere))['runs'][0]['dir'] == os.path.join(
        os.path.abspath(out), 'a')


def test_parser_fills_in_the_study_keys(tmp_path):
    from libcuflynx.parsers.PrimitiveParsers import YamlFileParser
    inp = YamlFileParser().parse_user_inputs_file({
        'file_prefix': 'demo',
        'input_param_file': 'demo_parameters.json',
        'resources_dir': str(tmp_path / 'resources'),
        'param_id_output_dir': str(tmp_path / 'out'),
        'generated_models_dir': str(tmp_path / 'generated_models'),
    })
    assert inp['study_name'] == 'demo'
    assert inp['write_cuflynx_study'] is True


def _param_id_with(study, engine_run):
    '''A CVS0DParamID with only what run() reads, whose engine runs ``engine_run``.'''
    from libcuflynx.param_id.paramID import CVS0DParamID
    pid = CVS0DParamID.__new__(CVS0DParamID)
    run_dir = os.path.join(study['param_id_output_dir'], 'CMA-ES_demo_obs')
    pid.__dict__.update({'rank': 0, '_study_inp': study, 'output_dir': run_dir,
                         'param_id_method': 'CMA-ES',
                         'param_id': type('E', (), {'run': staticmethod(engine_run)})()})
    pid._require_data_items = lambda what: None
    pid._check_info_available = lambda: None
    return pid, cuflynx_study.manifest_path(study)


def test_a_run_is_recorded_when_it_starts_and_marked_when_it_ends(study):
    seen = []

    def engine_run():
        # mid-run: the run is already in the manifest, so a run cut short keeps its results
        seen.append(_raw(manifest)['runs'][0]['status'])

    pid, manifest = _param_id_with(study, engine_run)
    pid.run()
    assert seen == ['running']
    assert [(r['dir'], r['status']) for r in _raw(manifest)['runs']] == \
        [('CMA-ES_demo_obs', 'finished')]


@pytest.mark.parametrize('error, status', [(KeyboardInterrupt, 'stopped'),
                                           (RuntimeError, 'failed')])
def test_a_run_cut_short_stays_in_the_manifest(study, error, status):
    def engine_run():
        raise error('cut short')

    pid, manifest = _param_id_with(study, engine_run)
    with pytest.raises(error):
        pid.run()
    assert [(r['dir'], r['status']) for r in _raw(manifest)['runs']] == \
        [('CMA-ES_demo_obs', status)]
