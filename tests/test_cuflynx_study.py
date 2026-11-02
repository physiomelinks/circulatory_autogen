"""The study manifest every analysis records itself in (``utilities/cuflynx_study.py``).

The file is a contract with CUFLynx: it opens a study from it, believing what it says.
So what is checked here is what a reader would get wrong if it drifted -- which entry is
first, which paths are relative, that a second analysis does not erase the first, and
that a broken write never fails the run it describes.
"""
import json
import os
import re
import zipfile
from datetime import datetime

import pytest
import yaml

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


STAMP = re.compile(r'^\d{4}-\d{2}-\d{2}_\d{6}(_\d+)?$')


def _dated(path, kind):
    """The ``inputs/<kind>/<yyyy-mm-dd_HHMMSS>`` folder ``path`` (as the manifest stores it)
    is in, and its path within that folder."""
    parts = path.split(os.sep)
    assert parts[:2] == ['inputs', kind] and STAMP.match(parts[2]), path
    return os.path.join(*parts[:3]), os.path.join(*parts[3:])


@pytest.fixture
def clock(monkeypatch):
    """Set the time an input is first used: ``clock(datetime(...))``."""
    def at(when):
        monkeypatch.setattr(cuflynx_study, '_stamp', lambda: when)
    return at


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
    # each in a folder named for when it was first used
    assert raw['model'] == raw['base_model']
    assert _dated(raw['model'], 'model')[1] == 'model.cellml'
    assert _dated(raw['obs_data'], 'obs_data')[1] == 'obs.json'
    assert _dated(raw['params_for_id'], 'params_for_id')[1] == 'params.csv'
    assert _dated(raw['user_inputs'], 'user_inputs')[1] == 'user_inputs.yaml'
    assert list(raw['user_funcs']) == ['operation']
    assert _dated(raw['user_funcs']['operation'], 'funcs')[1] == 'ops.py'
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


def test_each_version_of_an_input_has_its_own_dated_folder(study, clock):
    """An edited obs_data goes into a second dated folder; no copy sits outside one, so the
    file under a plain name is never an earlier run's taken for the current one."""
    out = study['param_id_output_dir']
    clock(datetime(2026, 10, 31, 9, 15, 2))
    cuflynx_study.update_study(study, 'runs', {'dir': os.path.join(out, 'run1'), 'method': 'm'})
    with open(study['param_id_obs_path'], 'w') as handle:
        handle.write('changed')
    clock(datetime(2026, 10, 31, 9, 40, 0))
    raw = _raw(cuflynx_study.update_study(study, 'runs',
                                          {'dir': os.path.join(out, 'run2'), 'method': 'm'}))

    obs = os.path.join(out, 'inputs', 'obs_data')
    assert sorted(os.listdir(obs)) == ['2026-10-31_091502', '2026-10-31_094000']
    assert all(os.path.isdir(os.path.join(obs, name)) for name in os.listdir(obs))
    # each run points at the copy it ran with
    runs = {run['dir']: run['obs_data'] for run in raw['runs']}
    assert runs == {'run1': os.path.join('inputs', 'obs_data', '2026-10-31_091502', 'obs.json'),
                    'run2': os.path.join('inputs', 'obs_data', '2026-10-31_094000', 'obs.json')}
    for name, content in (('run1', 'x'), ('run2', 'changed')):
        with open(os.path.join(out, runs[name])) as handle:
            assert handle.read() == content
    assert raw['obs_data'] == runs['run2']

    versions = [v for v in cuflynx_study.read_index(out) if v['kind'] == 'obs_data']
    assert [(v['file'], v['folder'], v['first_used'], v['runs']) for v in versions] == [
        ('obs.json', os.path.join('inputs', 'obs_data', '2026-10-31_091502'),
         '2026-10-31T09:15:02', ['run1']),
        ('obs.json', os.path.join('inputs', 'obs_data', '2026-10-31_094000'),
         '2026-10-31T09:40:00', ['run2'])]
    assert versions[0]['hash'] != versions[1]['hash']
    # an unchanged input both runs used lists both
    params = [v for v in cuflynx_study.read_index(out) if v['kind'] == 'params_for_id']
    assert len(params) == 1 and params[0]['runs'] == ['run1', 'run2']


def test_a_rerun_with_unchanged_inputs_copies_nothing(study, clock):
    out = study['param_id_output_dir']
    clock(datetime(2026, 10, 31, 9, 15, 2))
    cuflynx_study.update_study(study, 'runs', {'dir': os.path.join(out, 'run1'), 'method': 'm'})
    first = _all_files(os.path.join(out, 'inputs'))
    cuflynx_study._BROUGHT.clear()        # as a second process would see it
    clock(datetime(2026, 11, 2, 8, 0, 0))
    raw = _raw(cuflynx_study.update_study(study, 'runs',
                                          {'dir': os.path.join(out, 'run2'), 'method': 'm'}))
    assert _all_files(os.path.join(out, 'inputs')) == first
    assert {run['obs_data'] for run in raw['runs']} == {
        os.path.join('inputs', 'obs_data', '2026-10-31_091502', 'obs.json')}
    assert all(v['runs'] == ['run1', 'run2'] for v in cuflynx_study.read_index(out))


def test_two_versions_first_used_in_one_second_are_told_apart(study, clock, tmp_path):
    clock(datetime(2026, 10, 31, 9, 15, 2))
    raw = _raw(cuflynx_study.update_study(study))
    other = tmp_path / 'elsewhere' / 'obs.json'
    other.parent.mkdir()
    other.write_text('other')
    raw2 = _raw(cuflynx_study.update_study(dict(study, param_id_obs_path=str(other))))
    assert _dated(raw['obs_data'], 'obs_data')[0].endswith('2026-10-31_091502')
    assert _dated(raw2['obs_data'], 'obs_data')[0].endswith('2026-10-31_091502_2')


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
    folder, name = _dated(raw['model'], 'model')
    assert name == 'm.cellml'
    assert os.path.isfile(os.path.join(out, folder, 'shared', 'units.cellml'))


def test_a_generated_models_system_is_copied(study, tmp_path):
    resources = tmp_path / 'resources'
    resources.mkdir()
    (resources / 'demo_module_array.json').write_text('{}')
    (resources / 'demo_parameters.json').write_text('{}')
    study['vessels_csv_abs_path'] = str(resources / 'demo_module_array.json')
    study['parameters_csv_abs_path'] = str(resources / 'demo_parameters.json')
    raw = _raw(cuflynx_study.update_study(study))
    folder, name = _dated(raw['system']['module_array'], 'system')
    assert raw['system'] == {
        'module_array': os.path.join(folder, 'demo_module_array.json'),
        'parameters': os.path.join(folder, 'demo_parameters.json')}
    [version] = [v for v in cuflynx_study.read_index(study['param_id_output_dir'])
                 if v['kind'] == 'system']
    assert version['folder'] == folder and version['file'] == 'demo_module_array.json'
    read = cuflynx_study.read_study(study['param_id_output_dir'])
    assert all(os.path.isfile(p) for p in read['system'].values())


def test_a_nested_study_copies_into_the_larger_working_dir(study, tmp_path):
    working = tmp_path / 'out'
    study['param_id_output_dir'] = str(working / 'step' / 'param_id')
    study['cuflynx_working_dir'] = str(working)
    elsewhere = working / 'step' / 'sensitivity'
    raw = _raw(cuflynx_study.update_study(study, 'sensitivity',
                                          {'dir': str(elsewhere), 'method': 'sobol'}))
    assert raw['obs_data'].startswith(os.path.join('..', '..', 'inputs', 'obs_data', ''))
    folder, name = _dated(os.path.relpath(raw['obs_data'], os.path.join('..', '..')), 'obs_data')
    assert name == 'obs.json'
    assert raw['sensitivity'][0]['dir'] == os.path.join('..', 'sensitivity')
    assert os.path.isfile(working / folder / 'obs.json')
    # the index is the working dir's, its paths relative to it
    [version] = [v for v in cuflynx_study.read_index(str(working)) if v['kind'] == 'obs_data']
    assert version['folder'] == folder
    assert version['runs'] == [os.path.join('step', 'sensitivity')]


def test_an_output_outside_the_working_dir_is_warned_about_flagged_and_listed(study, tmp_path,
                                                                             capsys):
    """An output directory outside the working dir is allowed, but a zip of the working dir does
    not hold it: the run warns once naming it, its manifest entry keeps the absolute path with
    ``"outside": true``, and ``outside_outputs`` lists it for an app exporting the working dir."""
    elsewhere = str(tmp_path / 'sa_elsewhere')
    inside = os.path.join(study['param_id_output_dir'], 'sensitivity')
    for _ in range(2):  # a running analysis updates its entry more than once
        cuflynx_study.update_study(study, 'sensitivity', {'dir': elsewhere, 'method': 'sobol'})
    location = cuflynx_study.update_study(study, 'sensitivity', {'dir': inside, 'method': 'local'})

    warnings = [line for line in capsys.readouterr().out.splitlines()
                if 'outside the working dir' in line]
    assert len(warnings) == 1
    assert elsewhere in warnings[0] and 'will not travel with the working dir' in warnings[0]

    entries = {e['method']: e for e in _raw(location)['sensitivity']}
    assert entries['sobol']['dir'] == elsewhere and entries['sobol']['outside'] is True
    assert entries['local']['dir'] == 'sensitivity' and 'outside' not in entries['local']

    expected = [{'section': 'sensitivity', 'dir': elsewhere, 'method': 'sobol'}]
    assert cuflynx_study.outside_outputs(cuflynx_study.read_study(location)) == expected
    assert cuflynx_study.outside_outputs(location) == expected


def test_a_moved_study_never_points_a_rerun_at_its_outside_output(study, tmp_path):
    """Nothing re-runs a study from its manifest into a recorded output directory: moved, the
    inside output resolves into the new working dir, the outside one stays flagged (so an app
    can say it is missing), and ``generation_inputs`` -- the manifest's only route back to a
    run -- names no output directory at all."""
    import shutil
    out = study['param_id_output_dir']
    elsewhere = str(tmp_path / 'sa_elsewhere')
    os.makedirs(os.path.join(out, 'sensitivity'))
    cuflynx_study.update_study(study, 'sensitivity', {'dir': elsewhere, 'method': 'sobol'})
    cuflynx_study.update_study(study, 'sensitivity',
                               {'dir': os.path.join(out, 'sensitivity'), 'method': 'local'})
    moved = tmp_path / 'moved'
    shutil.copytree(out, moved)
    shutil.rmtree(out)

    read = cuflynx_study.read_study(str(moved))
    dirs = {e['method']: e['dir'] for e in read['sensitivity']}
    assert dirs['local'] == str(moved / 'sensitivity')
    assert [o['dir'] for o in cuflynx_study.outside_outputs(read)] == [elsewhere]

    system = {'module_array': str(moved / 'inputs' / 'system' / 'a.json'),
              'parameters': str(moved / 'inputs' / 'system' / 'p.json')}
    again = cuflynx_study.generation_inputs(dict(read, system=system))
    assert set(again) == {'file_prefix', 'resources_dir', 'input_param_file'}
    assert not any(elsewhere in str(value) for value in again.values())


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
    for value in (read.get('system') or {}).values():
        paths += value if isinstance(value, list) else [value] if isinstance(value, str) else []
    paths += [e[k] for s in cuflynx_study.ENTRY_SECTIONS for e in read[s]
              for k in ('dir', 'obs_data') if e.get(k)]
    assert paths
    for path in paths:
        assert os.path.commonpath([path, root]) == root, path
        assert os.path.exists(path), path


# -- export_working_dir ---------------------------------------------------------------------

def _snapshot(directory):
    return {path: open(os.path.join(directory, path), 'rb').read()
            for path in _all_files(directory)}


@pytest.fixture
def exported(study, tmp_path, capsys):
    """A working dir whose user_inputs yaml sends Sobol output and generated models outside it,
    with a dated resolved yaml, exported: ``{w, before, zip, unzipped, left, out}``."""
    w = study['param_id_output_dir']
    sa_runs, models = str(tmp_path / 'scratch' / 'sa_runs'), str(tmp_path / 'scratch' / 'models')
    with open(study['user_inputs_path'], 'w') as handle:
        handle.write(f"""# my study
file_prefix: demo
input_param_file: demo_parameters.json
resources_dir: {tmp_path / 'inputs'}
param_id_output_dir: {w}
generated_models_dir: {models}
sa_options:
  method: sobol
  output_dir: {sa_runs}
""")
    cuflynx_study.update_study(study, 'sensitivity', {'dir': sa_runs, 'method': 'sobol'})
    inside = os.path.join(w, 'sensitivity')
    os.makedirs(inside)
    cuflynx_study.update_study(study, 'sensitivity', {'dir': inside, 'method': 'local'})
    (tmp_path / 'out' / 'inputs' / 'user_inputs' / 'user_inputs_261102.yaml').write_text(
        yaml.safe_dump({'file_prefix': 'demo', 'param_id_output_dir': w,
                        'generated_models_dir': models,
                        'generated_models_subdir': os.path.join(models, 'demo'),
                        'user_inputs_path': study['user_inputs_path'],
                        'sa_options': {'output_dir': inside}}))
    capsys.readouterr()

    before = _snapshot(w)
    target = str(tmp_path / 'shared' / 'study.zip')
    left = cuflynx_study.export_working_dir(w, target)
    unzipped = tmp_path / 'unzipped'
    with zipfile.ZipFile(target) as archive:
        archive.extractall(unzipped)
    return {'w': w, 'before': before, 'zip': target, 'unzipped': str(unzipped), 'left': left,
            'out': capsys.readouterr().out, 'sa_runs': sa_runs, 'models': models}


def _given_yaml(root):
    versions = [v for v in cuflynx_study.read_index(root) if v['kind'] == 'user_inputs']
    assert len(versions) == 1
    return os.path.join(root, versions[0]['folder'], versions[0]['file'])


def test_export_leaves_the_working_dir_as_it_is(exported):
    """The rewrite is in the zip only: the yaml, manifest and index keep the outside path, so
    the user's own runs and a zip made by hand are as they were."""
    w = exported['w']
    assert _snapshot(w) == exported['before']
    entries = {e['method']: e for e in _raw(cuflynx_study.manifest_path(
        {'param_id_output_dir': w, 'file_prefix': 'demo'}))['sensitivity']}
    assert entries['sobol']['dir'] == exported['sa_runs'] and entries['sobol']['outside'] is True
    assert exported['sa_runs'] in open(_given_yaml(w)).read()
    assert [o['dir'] for o in cuflynx_study.outside_outputs(w)] == [exported['sa_runs']]


def test_the_zip_holds_the_working_dir_in_sorted_relative_paths(exported):
    with zipfile.ZipFile(exported['zip']) as archive:
        names = archive.namelist()
    assert names == sorted(names) and not any(n.startswith('/') or '..' in n for n in names)
    assert 'demo_cuflynx_study.json' in names and 'inputs/index.json' in names
    assert set(_all_files(exported['unzipped'])) == set(exported['before'])


def test_the_zips_yamls_name_relative_output_dirs(exported):
    root = exported['unzipped']
    given = _given_yaml(root)
    data = yaml.safe_load(open(given).read())
    assert data['sa_options'] == {'method': 'sobol', 'output_dir': 'sa_runs'}
    assert data['param_id_output_dir'] == os.path.join('..', '..', '..')
    assert data['generated_models_dir'] == os.path.join('..', '..', '..', 'generated_models')
    assert data['input_param_file'] == 'demo_parameters.json'    # the rest is untouched

    dated = yaml.safe_load(
        open(os.path.join(root, 'inputs', 'user_inputs', 'user_inputs_261102.yaml')).read())
    assert dated['param_id_output_dir'] == os.path.join('..', '..')
    assert dated['generated_models_dir'] == os.path.join('..', '..', 'generated_models')
    assert dated['sa_options']['output_dir'] == '.'    # already in sensitivity/, so no warning
    assert 'generated_models_subdir' not in dated and 'user_inputs_path' not in dated


def test_the_zips_manifest_and_index_hold_no_outside_entry_or_absolute_path(exported):
    root = exported['unzipped']
    manifest = os.path.join(root, 'demo_cuflynx_study.json')
    text = open(manifest).read()
    assert '"/' not in text and 'outside' not in text and exported['sa_runs'] not in text
    assert [e['method'] for e in _raw(manifest)['sensitivity']] == ['local']

    index = open(os.path.join(root, 'inputs', 'index.json')).read()
    assert exported['sa_runs'] not in index
    # the hashes are of the files the zip holds, so a same content is still found in it
    version = next(v for v in cuflynx_study.read_index(root) if v['kind'] == 'user_inputs')
    assert version['files'] == {version['file']: cuflynx_study._sha256(_given_yaml(root))}
    assert version['hash'] == cuflynx_study._content_hash(version['files'])


def test_read_study_works_on_the_unzipped_archive(exported):
    read = cuflynx_study.read_study(exported['unzipped'])
    _assert_resolves_inside(read, exported['unzipped'])
    assert cuflynx_study.outside_outputs(read) == []


def test_a_run_of_the_unzipped_yaml_writes_inside_the_unzipped_working_dir(exported, tmp_path,
                                                                           monkeypatch):
    """Through the route a yaml is run by, ``user_inputs_path_override``, every output goes in
    the recipient's working dir, and the run's own manifest entry is not outside it."""
    from libcuflynx.parsers import PrimitiveParsers
    root = exported['unzipped']
    runner = tmp_path / 'user_run_files'
    runner.mkdir()
    (runner / 'user_inputs.yaml').write_text(
        yaml.safe_dump({'user_inputs_path_override': _given_yaml(root)}))
    monkeypatch.setattr(PrimitiveParsers, 'user_inputs_dir', str(runner))
    (tmp_path / 'inputs' / 'demo_parameters.json').write_text('{}')

    inp = PrimitiveParsers.YamlFileParser().parse_user_inputs_file(None)

    output_dir = os.path.abspath(inp['sa_options']['output_dir'])
    assert output_dir == os.path.join(root, 'sensitivity', 'sa_runs')
    assert os.path.abspath(inp['generated_models_subdir']).startswith(root + os.sep)
    assert cuflynx_study.working_dir(inp) == root
    assert not os.path.exists(exported['sa_runs'])
    location = cuflynx_study.update_study(inp, 'sensitivity', {'dir': output_dir, 'method': 'sobol'})
    assert location == os.path.join(root, 'demo_cuflynx_study.json')
    entry = next(e for e in _raw(location)['sensitivity'] if e['method'] == 'sobol')
    assert entry['dir'] == os.path.join('sensitivity', 'sa_runs') and 'outside' not in entry


def _rewrite_given_yaml(study, **values):
    """Give ``study``'s yaml the content ``values`` (a new version of it)."""
    with open(study['user_inputs_path'], 'w') as handle:
        yaml.safe_dump({'file_prefix': 'demo', 'input_param_file': 'demo_parameters.json',
                        **values}, handle)


def _export(study, tmp_path):
    """``(the unzipped folder, what export_working_dir returned)``."""
    target, unzipped = tmp_path / 'w.zip', tmp_path / 'w_unzipped'
    left = cuflynx_study.export_working_dir(study['param_id_output_dir'], str(target))
    with zipfile.ZipFile(target) as archive:
        archive.extractall(unzipped)
    return str(unzipped), left


def _stored(root, kind, copy):
    """The stored file of ``kind`` a yaml in ``inputs/user_inputs/...`` names as ``copy``."""
    version = next(v for v in cuflynx_study.read_index(root) if v['kind'] == kind
                   and copy.endswith(os.path.join(v['folder'], v['file'])))
    return os.path.join(version['folder'], version['file'])


def test_the_zips_yamls_name_their_inputs_by_their_stored_copies(exported):
    """Each input a yaml reads is a path to its copy from the yaml's own folder, as the parser
    takes it, so nothing points back at this machine."""
    root = exported['unzipped']
    given = _given_yaml(root)
    data = yaml.safe_load(open(given).read())
    folder = os.path.dirname(given)
    for key, kind in (('param_id_obs_path', 'obs_data'), ('generated_model_path', 'model'),
                      ('operation_funcs_external_path', 'funcs')):
        path = os.path.normpath(os.path.join(folder, data[key]))
        assert os.path.isfile(path) and not os.path.isabs(data[key])
        assert _stored(root, kind, path), key
    resources = os.path.join(folder, data['resources_dir'])
    assert os.path.isfile(os.path.join(resources, data['params_for_id_file']))
    assert '/tmp' not in str(data) and str(exported['w']) not in str(data)


def test_the_zips_yaml_keeps_nothing_the_parser_works_out_again():
    raw = {'model_path': '/a/m.cellml', 'uncalibrated_model_path': '/a/m.cellml',
           'vessels_csv_abs_path': '/a/v', 'parameters_csv_abs_path': '/a/p',
           'params_for_id_path': '/a/pfi.json', 'param_id_output_dir': '/w'}
    out = cuflynx_study._portable_user_inputs(raw, os.path.join('inputs', 'user_inputs'), '/w',
                                              {}, {'base_model': 'inputs/model/d/m.cellml'}, set())
    assert sorted(out) == ['generated_model_path', 'generated_models_dir', 'param_id_output_dir']


def test_export_points_each_yaml_at_the_version_its_own_run_used(study, tmp_path, clock,
                                                                 monkeypatch):
    """An edited obs_data gives a second version, and going back to the first content a third
    run: each dated yaml and each given yaml names what its own run read, not the latest, even
    when the run directory and the file name are the same for all three."""
    w = study['param_id_output_dir']
    obs = tmp_path / 'inputs' / 'obs.json'
    archive = os.path.join(w, 'inputs', 'user_inputs')
    os.makedirs(archive, exist_ok=True)
    for day, (when, content) in enumerate([(datetime(2026, 11, 1, 9), 'first'),
                                           (datetime(2026, 11, 2, 9), 'second'),
                                           (datetime(2026, 11, 3, 9), 'first')]):
        clock(when)
        obs.write_text(content)
        _rewrite_given_yaml(study, param_id_obs_path=str(obs), run=day)
        name = f'user_inputs_26110{day + 1}.yaml'
        monkeypatch.setattr(cuflynx_study, 'dated_user_inputs_name', lambda: name)
        with open(os.path.join(archive, name), 'w') as handle:
            yaml.safe_dump({'file_prefix': 'demo', 'param_id_obs_path': str(obs)}, handle)
        cuflynx_study.update_study(study, 'runs', {'dir': os.path.join(w, 'run'), 'method': 'ga'})

    root, _ = _export(study, tmp_path)

    def read_obs(path):
        folder = os.path.dirname(path)
        relative = yaml.safe_load(open(path).read())['param_id_obs_path']
        assert not os.path.isabs(relative)
        return open(os.path.join(folder, relative)).read()

    dated = os.path.join(root, 'inputs', 'user_inputs')
    assert [read_obs(os.path.join(dated, f'user_inputs_26110{n}.yaml')) for n in (1, 2, 3)] == [
        'first', 'second', 'first']
    yamls = [v for v in cuflynx_study.read_index(root) if v['kind'] == 'user_inputs']
    assert [read_obs(os.path.join(root, v['folder'], v['file'])) for v in yamls] == [
        'first', 'second', 'first']


def test_export_warns_about_each_rewritten_input(exported):
    rewritten = [item for item in exported['left'] if item.get('input')]
    assert rewritten and all(item['relative'].startswith('inputs') for item in rewritten)
    resources = next(item for item in rewritten if item['original'].endswith('inputs'))
    assert f"copy {resources['relative']} " in exported['out']


def test_an_input_with_no_copy_keeps_its_path_and_is_warned_about(study, tmp_path, capsys):
    """Left out, it would fall back on a default without a word, so it stays absolute and a run
    elsewhere stops at it. A path only 1D coupling reads is not warned about without it."""
    cost = str(tmp_path / 'elsewhere' / 'cost.py')
    _rewrite_given_yaml(study, cost_funcs_external_path=cost, some_table='/data/table.csv',
                        cpp_1d_model_config_path='/data/input1d.dat',
                        solver_info={'build_dir': '/tmp/build'})
    cuflynx_study.update_study(study)
    capsys.readouterr()

    root, left = _export(study, tmp_path)

    data = yaml.safe_load(open(_given_yaml(root)).read())
    assert data['cost_funcs_external_path'] == cost and data['some_table'] == '/data/table.csv'
    out = capsys.readouterr().out
    missing = {item['original'] for item in left if item.get('input') and not item['relative']}
    assert missing == {cost, '/data/table.csv'}
    assert any(f'input {cost} has no copy' in line for line in out.splitlines())


def test_a_yaml_no_run_recorded_keeps_its_input_paths(study, tmp_path, capsys):
    cuflynx_study.update_study(study)
    archive = os.path.join(study['param_id_output_dir'], 'inputs', 'user_inputs')
    os.makedirs(archive, exist_ok=True)
    with open(os.path.join(archive, 'user_inputs_250101.yaml'), 'w') as handle:
        yaml.safe_dump({'file_prefix': 'demo', 'resources_dir': '/old/resources',
                        'model_path': '/old/demo.cellml'}, handle)
    root, _ = _export(study, tmp_path)
    data = yaml.safe_load(open(os.path.join(root, 'inputs', 'user_inputs',
                                            'user_inputs_250101.yaml')).read())
    assert data['resources_dir'] == '/old/resources'
    assert data['generated_model_path'] == '/old/demo.cellml'
    assert 'input /old/resources has no copy' in capsys.readouterr().out


def test_the_zips_yaml_finds_the_system_and_its_module_dirs(study, tmp_path):
    resources = tmp_path / 'resources'
    (resources / 'sub').mkdir(parents=True)
    (resources / 'demo_module_array.json').write_text('{}')
    (resources / 'sub' / 'demo_parameters.json').write_text('{}')
    library = tmp_path / 'library'
    library.mkdir()
    (library / 'demo_units.cellml').write_text('<model/>')
    external = tmp_path / 'external'
    external.mkdir()
    (external / 'mine_modules_config.json').write_text('[]')
    study.update({'vessels_csv_abs_path': str(resources / 'demo_module_array.json'),
                  'parameters_csv_abs_path': str(resources / 'sub' / 'demo_parameters.json'),
                  'module_library_dirs': [str(library)], 'external_modules_dir': str(external),
                  'use_builtin_modules': True})
    _rewrite_given_yaml(study, input_param_file='sub/demo_parameters.json',
                        resources_dir=str(resources), module_library_dirs=[str(library)],
                        external_modules_dir=str(external))
    cuflynx_study.update_study(study)

    root, _ = _export(study, tmp_path)

    given = _given_yaml(root)
    data = yaml.safe_load(open(given).read())
    folder = os.path.dirname(given)
    system = os.path.normpath(os.path.join(folder, data['resources_dir']))
    assert os.path.dirname(system) == os.path.join(root, 'inputs', 'system')
    assert os.path.isfile(os.path.join(system, 'demo_module_array.json'))
    assert data['input_param_file'] == 'demo_parameters.json'
    assert os.path.isfile(os.path.join(system, data['input_param_file']))
    assert [os.path.normpath(os.path.join(folder, d)) for d in data['module_library_dirs']] == [
        os.path.join(system, 'modules')]
    assert os.path.normpath(os.path.join(folder, data['external_modules_dir'])) == os.path.join(
        system, 'external_modules')


@pytest.mark.parametrize('use_emulator', [True, False])
def test_the_zips_yaml_uses_the_stored_emulator_only_with_use_emulator(study, tmp_path, monkeypatch,
                                                                       use_emulator):
    """With ``use_emulator`` the emulator is an input, so the yaml reads its copy, relative to
    the yaml; without, ``emulator_dir`` is where a training run writes, so it takes the default
    in the working dir."""
    from libcuflynx.parsers import PrimitiveParsers
    emulator = tmp_path / 'trained' / 'demo_emulator'
    emulator.mkdir(parents=True)
    (emulator / 'bundle.json').write_text('{}')
    study.update({'use_emulator': use_emulator, 'emulator_dir': str(emulator)})
    _rewrite_given_yaml(study, use_emulator=use_emulator,
                        emulator_settings={'emulator_dir': str(emulator), 'min_r2': 0.9})
    cuflynx_study.update_study(study)

    root, _ = _export(study, tmp_path)

    given = _given_yaml(root)
    settings = yaml.safe_load(open(given).read())['emulator_settings']
    assert settings['min_r2'] == 0.9
    if not use_emulator:
        assert 'emulator_dir' not in settings
        return
    assert not os.path.isabs(settings['emulator_dir'])
    runner = tmp_path / 'user_run_files'
    runner.mkdir()
    (runner / 'user_inputs.yaml').write_text(yaml.safe_dump({'user_inputs_path_override': given}))
    monkeypatch.setattr(PrimitiveParsers, 'user_inputs_dir', str(runner))
    inp = PrimitiveParsers.YamlFileParser().parse_user_inputs_file(None)
    resolved = os.path.normpath(inp['emulator_settings']['emulator_dir'])
    assert resolved.startswith(os.path.join(root, 'inputs', 'emulator') + os.sep)
    assert os.path.isfile(os.path.join(resolved, 'bundle.json'))


@pytest.mark.parametrize('model_type, key', [('cellml', 'generated_model_path'),
                                             ('external_python', 'external_model_path')])
def test_the_zips_yaml_names_the_model_it_ran(study, tmp_path, model_type, key):
    _rewrite_given_yaml(study, model_type=model_type)
    cuflynx_study.update_study(study)
    root, _ = _export(study, tmp_path)
    given = _given_yaml(root)
    data = yaml.safe_load(open(given).read())
    assert os.path.isfile(os.path.join(os.path.dirname(given), data[key]))


def test_generation_from_a_yaml_archives_its_obs_data_path_resolved(exported, tmp_path, monkeypatch):
    """Generation does not read obs_data, but the dated archive it writes names it, and that
    file sits in another folder from the yaml it was parsed from."""
    from libcuflynx.parsers import PrimitiveParsers
    root = exported['unzipped']
    given = _given_yaml(root)
    runner = tmp_path / 'user_run_files'
    runner.mkdir()
    (runner / 'user_inputs.yaml').write_text(yaml.safe_dump({'user_inputs_path_override': given}))
    monkeypatch.setattr(PrimitiveParsers, 'user_inputs_dir', str(runner))
    PrimitiveParsers.YamlFileParser().parse_user_inputs_file(None)
    archive = os.path.join(root, 'inputs', 'user_inputs', cuflynx_study.dated_user_inputs_name())
    obs = yaml.safe_load(open(archive).read())['param_id_obs_path']
    assert os.path.isfile(obs) and os.path.isabs(obs)


@pytest.mark.parametrize('fit_parameters', [False, True])
def test_the_parser_runs_the_generated_model_path_it_is_given(tmp_path, fit_parameters):
    """A model kept elsewhere is run in place of the generated one; generating with fitted
    parameters still writes the generated one, from the model kept elsewhere."""
    from libcuflynx.parsers.PrimitiveParsers import YamlFileParser
    kept = str(tmp_path / 'kept' / 'demo.cellml')
    obs = tmp_path / 'demo_obs.json'
    obs.write_text('{}')
    generated = tmp_path / 'generated'
    inp = YamlFileParser().parse_user_inputs_file(
        {'file_prefix': 'demo', 'input_param_file': 'demo_parameters.json',
         'resources_dir': str(tmp_path), 'param_id_output_dir': str(tmp_path / 'out'),
         'generated_models_dir': str(generated), 'param_id_obs_path': str(obs),
         'cuflynx_study_copy_inputs': False, 'generated_model_path': kept},
        do_generation_with_fit_parameters=fit_parameters)
    assert inp['uncalibrated_model_path'] == kept
    assert inp['model_path'] == (str(generated / 'demo_demo_obs' / 'demo.cellml')
                                 if fit_parameters else kept)


def test_export_warns_once_for_each_rewritten_path(exported):
    warnings = [line for line in exported['out'].splitlines() if 'WARNING' in line]
    outputs = [line for line in warnings if 'output directory' in line]
    assert len(outputs) == 2    # the yaml, the dated yaml and the manifest name these two
    expected = {exported['models']: 'generated_models',
                exported['sa_runs']: os.path.join('sensitivity', 'sa_runs')}
    for original, relative in expected.items():
        line = next(w for w in outputs if original in w)
        assert f'relative path {relative} ' in line and 'not in the zip' in line
    assert [item for item in exported['left'] if not item.get('input')] == [
        {'original': o, 'relative': r} for o, r in sorted(expected.items())]


def test_export_rewrites_nested_manifests_and_drops_a_written_file(study, tmp_path, capsys):
    """A workflow's step has its own manifest, and the ``--write-calibrated`` file beside the
    target is an absolute path in the workflow block."""
    w = study['param_id_output_dir']
    target = str(tmp_path / 'library' / 'calibrated.json')
    cuflynx_study.update_study(study, workflow={'result': os.path.join(w, 'result.json'),
                                                'calibrated_parameters': target})
    elsewhere = str(tmp_path / 'sa_elsewhere')
    step = dict(study, param_id_output_dir=os.path.join(w, 'step'), cuflynx_working_dir=w)
    cuflynx_study.update_study(step, 'sensitivity', {'dir': elsewhere, 'method': 'sobol'})
    capsys.readouterr()

    left = cuflynx_study.export_working_dir(w, str(tmp_path / 'w.zip'))

    with zipfile.ZipFile(tmp_path / 'w.zip') as archive:
        top = json.loads(archive.read('demo_cuflynx_study.json'))
        nested = json.loads(archive.read('step/demo_cuflynx_study.json'))
    assert top['workflow'] == {'result': 'result.json'}
    assert nested['sensitivity'] == []
    assert [item for item in left if not item.get('input')] == [
        {'original': elsewhere, 'relative': os.path.join('sensitivity', 'sa_elsewhere')},
        {'original': target, 'relative': None}]
    assert sum(target in line for line in capsys.readouterr().out.splitlines()) == 1


def test_export_can_write_into_the_working_dir_it_zips(study):
    cuflynx_study.update_study(study)
    target = os.path.join(study['param_id_output_dir'], 'study.zip')
    for _ in range(2):    # the second run must not hold the first zip
        cuflynx_study.export_working_dir(study['param_id_output_dir'], target)
    with zipfile.ZipFile(target) as archive:
        assert 'study.zip' not in archive.namelist()


def test_export_leaves_a_yaml_it_cannot_read_as_it_is(study, tmp_path, capsys):
    with open(study['user_inputs_path'], 'w') as handle:
        handle.write('a: [unclosed\n')
    cuflynx_study.update_study(study)
    cuflynx_study.export_working_dir(study['param_id_output_dir'], str(tmp_path / 'w.zip'))
    assert 'could not read' in capsys.readouterr().out
    with zipfile.ZipFile(tmp_path / 'w.zip') as archive:
        name = next(n for n in archive.namelist() if n.endswith('user_inputs.yaml'))
        assert archive.read(name) == b'a: [unclosed\n'


def test_export_refuses_what_is_not_a_self_contained_working_dir(study, tmp_path):
    empty = tmp_path / 'empty'
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        cuflynx_study.export_working_dir(str(empty), str(tmp_path / 'w.zip'))
    cuflynx_study.update_study(dict(study, cuflynx_study_copy_inputs=False))
    with pytest.raises(ValueError, match='cuflynx_study_copy_inputs'):
        cuflynx_study.export_working_dir(study['param_id_output_dir'], str(tmp_path / 'w.zip'))
    assert not (tmp_path / 'w.zip').exists()


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
    assert _dated(raw['base_model'], 'model')[1] == 'model.cellml'


def test_emulator_is_declared(study, tmp_path):
    bundle = tmp_path / 'shared' / 'emulator'
    bundle.mkdir(parents=True)
    (bundle / 'emulator_metadata.json').write_text('{}')
    raw = _raw(cuflynx_study.update_study(study, emulator=str(bundle)))
    # a shared emulator outside the working dir is copied in, directory and all
    assert _dated(raw['emulator'], 'emulator')[1] == 'emulator'
    assert os.path.isfile(os.path.join(study['param_id_output_dir'], raw['emulator'],
                                       'emulator_metadata.json'))


def test_turned_off(study):
    study['write_cuflynx_study'] = False
    assert cuflynx_study.update_study(study) is None
    assert not os.path.exists(cuflynx_study.manifest_path(study))


def test_store_input_keeps_each_saved_content_in_its_own_dated_folder(tmp_path, clock):
    """An app saving an edited input: a new content gets a new dated folder and the earlier
    one stays; saving a content again reuses its folder and makes it the latest."""
    work = tmp_path / 'work'
    clock(datetime(2026, 10, 31, 9, 0, 0))
    first = cuflynx_study.store_input(work, 'obs_data', b'{"a": 1}', 'demo_obs_data.json')
    assert first == os.path.join('inputs', 'obs_data', '2026-10-31_090000', 'demo_obs_data.json')
    clock(datetime(2026, 10, 31, 9, 5, 0))
    second = cuflynx_study.store_input(work, 'obs_data', b'{"a": 2}', 'demo_obs_data.json')
    assert second == os.path.join('inputs', 'obs_data', '2026-10-31_090500', 'demo_obs_data.json')
    assert (work / first).read_bytes() == b'{"a": 1}'
    assert cuflynx_study.latest_input(work, 'obs_data', 'demo_obs_data.json') == str(work / second)

    clock(datetime(2026, 10, 31, 9, 9, 0))
    again = cuflynx_study.store_input(work, 'obs_data', b'{"a": 1}', 'demo_obs_data.json')
    assert again == first
    assert sorted(os.listdir(work / 'inputs' / 'obs_data')) == ['2026-10-31_090000',
                                                                 '2026-10-31_090500']
    assert cuflynx_study.latest_input(work, 'obs_data', 'demo_obs_data.json') == str(work / first)
    assert cuflynx_study.latest_input(work, 'obs_data', 'other.json') is None
    assert cuflynx_study.latest_input(work, 'params_for_id') is None
    assert [v['kind'] for v in cuflynx_study.read_index(work)] == ['obs_data', 'obs_data']


def test_store_input_takes_a_file_a_renamed_file_or_a_folder(tmp_path, clock):
    work = tmp_path / 'work'
    source = tmp_path / 'src'
    (source / 'sub').mkdir(parents=True)
    (source / 'calibration_workflow.json').write_text('{}')
    (source / 'sub' / 'obs.json').write_text('[]')
    clock(datetime(2026, 10, 31, 9, 0, 0))
    as_is = cuflynx_study.store_input(work, 'obs_data', source / 'sub' / 'obs.json')
    assert as_is.endswith(os.path.join('2026-10-31_090000', 'obs.json'))
    renamed = cuflynx_study.store_input(work, 'obs_data', source / 'sub' / 'obs.json',
                                        'demo_obs_data.json')
    assert renamed.endswith(os.path.join('2026-10-31_090000_2', 'demo_obs_data.json'))

    tree = cuflynx_study.store_input(work, 'workflows/wf', source, 'calibration_workflow.json')
    folder = os.path.join('inputs', 'workflows', 'wf', '2026-10-31_090000')
    assert tree == os.path.join(folder, 'calibration_workflow.json')
    assert (work / folder / 'sub' / 'obs.json').read_text() == '[]'
    assert cuflynx_study.latest_input(work, 'workflows/wf') == str(work / tree)

    # already in the working dir: not copied again
    assert cuflynx_study.store_input(work, 'obs_data', work / as_is) == as_is
    with pytest.raises(ValueError):
        cuflynx_study.store_input(work, '../x', b'', 'a')
    with pytest.raises(ValueError):
        cuflynx_study.store_input(work, 'obs_data', b'x')


def test_store_input_leaves_out_a_folders_hidden_files_and_folders(tmp_path, clock):
    """A workflow folder is stored without its hidden folders (a checkout's ``.git``, an
    editor's ``.settings``) and files, and a change only inside them reuses the stored version."""
    work = tmp_path / 'work'
    source = tmp_path / 'wf'
    (source / '.git' / 'objects').mkdir(parents=True)
    (source / '.git' / 'HEAD').write_text('ref: refs/heads/main')
    (source / '.git' / 'objects' / 'ab').write_text('history')
    (source / '.settings').mkdir()
    (source / '.settings' / 'solver.json').write_text('{}')
    (source / '.hidden').write_text('x')
    (source / 'calibration_workflow.json').write_text('{}')
    clock(datetime(2026, 10, 31, 9, 0, 0))
    stored = cuflynx_study.store_input(work, 'workflows/wf', source, 'calibration_workflow.json')
    folder = work / os.path.dirname(stored)
    assert _all_files(folder) == ['calibration_workflow.json']
    [version] = cuflynx_study.read_index(work)
    assert sorted(version['files']) == ['calibration_workflow.json']

    (source / '.git' / 'HEAD').write_text('ref: refs/heads/other')
    (source / '.settings' / 'solver.json').write_text('{"changed": true}')
    clock(datetime(2026, 10, 31, 9, 5, 0))
    again = cuflynx_study.store_input(work, 'workflows/wf', source, 'calibration_workflow.json')
    assert again == stored
    assert os.listdir(work / 'inputs' / 'workflows' / 'wf') == ['2026-10-31_090000']


def test_a_stored_folder_is_in_the_zip_as_it_is_in_inputs(study, tmp_path):
    """store_input and export_working_dir leave out the same names, so the zip of a working dir
    holds every stored input file and the index lists no file the zip lacks."""
    cuflynx_study.update_study(study)
    w = study['param_id_output_dir']
    source = tmp_path / 'wf'
    (source / '.settings').mkdir(parents=True)
    (source / '.settings' / 'solver.json').write_text('{}')
    (source / '.git').mkdir()
    (source / '.git' / 'HEAD').write_text('ref: refs/heads/main')
    (source / 'sub').mkdir()
    (source / 'sub' / 'obs.json').write_text('[]')
    (source / 'calibration_workflow.json').write_text('{}')
    stored = cuflynx_study.store_input(w, 'workflows/wf', source, 'calibration_workflow.json')
    folder = os.path.dirname(stored)
    on_disk = _all_files(os.path.join(w, folder))
    [version] = [v for v in cuflynx_study.read_index(w) if v['kind'] == 'workflows/wf']

    unzipped, _ = _export(study, tmp_path)
    in_zip = _all_files(os.path.join(unzipped, folder))
    assert in_zip == on_disk == sorted(version['files'])
    assert on_disk == ['calibration_workflow.json', os.path.join('sub', 'obs.json')]


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
    folder, name = _dated(_raw(cuflynx_study.manifest_path(study))['user_funcs']['operation'],
                          'funcs')
    assert read['user_funcs']['operation'] == os.path.join(os.path.abspath(out), folder, name)


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
