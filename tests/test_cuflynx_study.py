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
    assert raw['study_dir'] == os.path.abspath(study['param_id_output_dir'])
    # outside the study, so absolute: they keep resolving wherever the study is moved
    assert raw['model'] == raw['base_model'] == study['model_path']
    assert raw['obs_data'] == study['param_id_obs_path']
    assert raw['user_funcs'] == {'operation': study['operation_funcs_external_path']}
    # dt folded into solver_info, the shape CUFLynx adopts
    assert raw['solver_info'] == {'solver': 'CVODE_myokit', 'MaximumStep': 1e-4, 'dt': 0.01}
    assert raw['pre_time'] == 100 and raw['sim_time'] == 10
    assert raw['updated']


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
    assert raw['base_model'] == study['model_path']


def test_emulator_is_declared(study, tmp_path):
    bundle = tmp_path / 'shared' / 'emulator'
    bundle.mkdir(parents=True)
    raw = _raw(cuflynx_study.update_study(study, emulator=str(bundle)))
    assert raw['emulator'] == str(bundle)


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
    assert read['user_funcs']['operation'] == study['operation_funcs_external_path']


def test_a_moved_copy_resolves_against_study_dir(study, tmp_path):
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
