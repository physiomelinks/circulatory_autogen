"""A calibration run leaves one working dir that holds every input and output of its study.

The working dir is ``param_id_output_dir``: the study manifest there records every path
relative to itself, and each input from elsewhere (model, obs_data, params_for_id, the
user_inputs yaml, user funcs, the module array, parameters file and module versions) is copied
into its ``inputs/`` folder (``utilities/cuflynx_study.py``). So ``zip -r`` of that one folder is
the whole study: unzipped anywhere, with the originals gone, every path in it still resolves.
"""
import json
import os
import re
import shutil
from datetime import date

import pytest
import yaml

from libcuflynx.utilities import cuflynx_study

PREFIX = '3compartment'
MODEL_FILES = (f'{PREFIX}_module_array.csv', f'{PREFIX}_parameters.json',
               f'{PREFIX}_params_for_id.json', f'{PREFIX}_obs_data.json')


def _workspace(comm, tmp_path):
    """Rank 0's ``tmp_path``, shared by every rank (each rank's own ``tmp_path`` differs)."""
    root = comm.bcast(str(tmp_path) if comm.Get_rank() == 0 else None, root=0)
    if comm.Get_rank() == 0:
        os.makedirs(os.path.join(root, 'resources'))
    comm.Barrier()
    return root


def _files(directory):
    return sorted(os.path.relpath(os.path.join(root, name), directory)
                  for root, _, names in os.walk(directory) for name in names)


def _resolved_paths(study):
    paths = [study[key] for key in cuflynx_study.PATH_KEYS if study.get(key)]
    paths += list((study.get('user_funcs') or {}).values())
    for value in (study.get('system') or {}).values():
        paths += value if isinstance(value, list) else [value] if isinstance(value, str) else []
    for section in cuflynx_study.ENTRY_SECTIONS:
        paths += [entry[key] for entry in study[section] for key in ('dir', 'obs_data')
                  if entry.get(key)]
    return paths


def _generated_study(comm, workspace, base_user_inputs, resources_dir):
    """The config of a generated 3compartment study with every input outside its working dir
    ``<workspace>/study``: resources, generated models, an external funcs file and the yaml."""
    from libcuflynx.scripts.script_generate_with_new_architecture import (
        generate_with_new_architecture)

    resources = os.path.join(workspace, 'resources')
    funcs = os.path.join(workspace, 'funcs', 'my_ops.py')
    user_inputs = os.path.join(workspace, 'user_inputs.yaml')
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': PREFIX, 'input_param_file': f'{PREFIX}_parameters.json',
        'model_type': 'cellml', 'solver': 'CVODE_myokit',
        'param_id_method': 'genetic_algorithm',
        'pre_time': 2, 'sim_time': 1, 'dt': 0.01, 'DEBUG': True,
        'do_uq': False, 'do_ia': False, 'plot_predictions': False,
        'solver_info': {'MaximumStep': 0.001},
        'resources_dir': resources,
        'generated_models_dir': os.path.join(workspace, 'generated_models'),
        'param_id_output_dir': os.path.join(workspace, 'study'),
        'param_id_obs_path': os.path.join(resources, f'{PREFIX}_obs_data.json'),
        'operation_funcs_external_path': funcs,
        'user_inputs_path': user_inputs,
        'debug_optimiser_options': {'num_calls_to_function': 40, 'max_patience': 500},
    })
    if comm.Get_rank() == 0:
        for name in MODEL_FILES:
            shutil.copy2(os.path.join(resources_dir, name), os.path.join(resources, name))
        os.makedirs(os.path.dirname(funcs))
        with open(funcs, 'w') as handle:
            handle.write('"""No operations of my own yet."""\n')
        with open(user_inputs, 'w') as handle:
            yaml.safe_dump({key: value for key, value in config.items()
                            if key != 'user_inputs_path'}, handle)
        assert generate_with_new_architecture(False, dict(config))
    comm.Barrier()
    return config


@pytest.mark.integration
@pytest.mark.mpi
def test_a_calibration_leaves_a_working_dir_that_can_be_moved(base_user_inputs, resources_dir,
                                                               tmp_path):
    MPI = pytest.importorskip("mpi4py.MPI",
                              reason='needs mpi4py: pip install "libcuflynx[mpi]"')
    from libcuflynx.scripts.param_id_run_script import run_param_id

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    workspace = _workspace(comm, tmp_path)
    resources = os.path.join(workspace, 'resources')
    working = os.path.join(workspace, 'study')
    user_inputs = os.path.join(workspace, 'user_inputs.yaml')
    config = _generated_study(comm, workspace, base_user_inputs, resources_dir)

    run_param_id(dict(config))
    comm.Barrier()

    manifest = cuflynx_study.manifest_path(config)
    if rank == 0:
        with open(manifest) as handle:
            text = handle.read()
        assert '"/' not in text, 'the manifest records an absolute path'
        raw = json.loads(text)
        assert 'study_dir' not in raw
        inputs = os.path.join(working, 'inputs')
        for kind in ('model', 'obs_data', 'params_for_id', 'user_inputs', 'funcs', 'system'):
            assert os.listdir(os.path.join(inputs, kind)), kind
            # every copy is in a folder dated when it was first used; the only plain files are
            # the dated user_inputs archives
            for name in os.listdir(os.path.join(inputs, kind)):
                assert re.match(r'\d{4}-\d{2}-\d{2}_\d{6}$', name) or (
                    kind == 'user_inputs' and name.startswith('user_inputs_')), (kind, name)
        run_dirs = [os.path.normpath(run['dir']) for run in raw['runs']]
        for version in cuflynx_study.read_index(working):
            assert version['runs'] == run_dirs, version
        assert os.path.isfile(os.path.join(
            inputs, 'user_inputs', f"user_inputs_{date.today().strftime('%y%m%d')}.yaml"))
        assert not any(name.startswith('user_inputs_') for name in os.listdir(resources))
        first = _files(inputs)
    comm.Barrier()

    # a second run copies nothing new: every input is there already, with the same content
    run_param_id(dict(config))
    comm.Barrier()

    if rank == 0:
        assert _files(os.path.join(working, 'inputs')) == first

        archive = shutil.make_archive(os.path.join(workspace, 'study'), 'zip', working)
        moved = os.path.join(workspace, 'moved', 'study')
        shutil.unpack_archive(archive, moved)
        for original in ('resources', 'generated_models', 'funcs', 'study'):
            shutil.rmtree(os.path.join(workspace, original))
        os.remove(user_inputs)

        study = cuflynx_study.read_study(moved)
        assert study['runs'] and study['user_funcs'] and study['system']
        # 3compartment's modules are all built in, so they are recorded, not copied
        assert study['system']['use_builtin_modules'] is True
        assert not os.path.exists(os.path.join(moved, 'inputs', 'system', 'modules'))
        for path in _resolved_paths(study):
            assert os.path.commonpath([path, moved]) == moved, path
            assert os.path.exists(path), path
    comm.Barrier()


def _files_outside(workspace):
    """The files in ``workspace`` besides the unzipped study and the yaml that starts a run."""
    return {path for path in _files(workspace) if not path.startswith(('elsewhere', 'runner'))}


def _run_from(yaml_path, monkeypatch, runner):
    """Make ``yaml_path`` the file a run reads, the way a user does: through
    ``user_inputs_path_override`` in the default user_inputs.yaml, in the folder ``runner``."""
    from libcuflynx.parsers import PrimitiveParsers
    with open(os.path.join(runner, 'user_inputs.yaml'), 'w') as handle:
        yaml.safe_dump({'user_inputs_path_override': yaml_path}, handle)
    monkeypatch.setattr(PrimitiveParsers, 'user_inputs_dir', runner)


@pytest.mark.integration
@pytest.mark.mpi
def test_the_zips_yaml_runs_with_the_originals_gone(base_user_inputs, resources_dir, tmp_path,
                                                    monkeypatch):
    """Exported, unzipped elsewhere and with every original deleted, either yaml in the zip
    generates and calibrates with no edits, reading and writing only in the unzipped folder."""
    MPI = pytest.importorskip("mpi4py.MPI",
                              reason='needs mpi4py: pip install "libcuflynx[mpi]"')
    from libcuflynx.scripts.param_id_run_script import run_param_id
    from libcuflynx.scripts.script_generate_with_new_architecture import (
        generate_with_new_architecture)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    workspace = _workspace(comm, tmp_path)
    config = _generated_study(comm, workspace, base_user_inputs, resources_dir)
    run_param_id(dict(config))
    comm.Barrier()

    working = config['param_id_output_dir']
    unzipped = os.path.join(workspace, 'elsewhere', 'study')
    runner = os.path.join(workspace, 'runner')
    run_dir = os.path.join(unzipped, f'genetic_algorithm_{PREFIX}_{PREFIX}_obs_data')
    given = dated = outside = None
    if rank == 0:
        before = {path: open(os.path.join(working, path), 'rb').read() for path in _files(working)}
        target = os.path.join(workspace, 'shared', 'study.zip')
        cuflynx_study.export_working_dir(working, target)
        assert before == {path: open(os.path.join(working, path), 'rb').read()
                          for path in _files(working)}
        shutil.unpack_archive(target, unzipped)
        for original in ('resources', 'generated_models', 'funcs', 'study', 'shared'):
            shutil.rmtree(os.path.join(workspace, original))
        os.remove(config['user_inputs_path'])
        os.makedirs(runner)
        [version] = [v for v in cuflynx_study.read_index(unzipped) if v['kind'] == 'user_inputs']
        given = os.path.join(unzipped, version['folder'], version['file'])
        dated = os.path.join(unzipped, 'inputs', 'user_inputs',
                             f"user_inputs_{date.today().strftime('%y%m%d')}.yaml")
        for name in (given, dated):
            with open(name) as handle:
                assert workspace not in handle.read(), name
        outside = _files_outside(workspace)
    given, dated, outside = comm.bcast((given, dated, outside), root=0)

    for name in (given, dated):
        if rank == 0:
            shutil.rmtree(run_dir)
        comm.Barrier()
        _run_from(name, monkeypatch, runner)
        run_param_id(None)
        comm.Barrier()
        if rank == 0:
            assert os.path.isfile(os.path.join(run_dir, 'best_param_vals.npy'))
            assert generate_with_new_architecture(False, None)
            assert os.path.isfile(os.path.join(unzipped, 'generated_models', PREFIX,
                                               f'{PREFIX}.cellml'))
            shutil.rmtree(os.path.join(unzipped, 'generated_models'))
            assert _files_outside(workspace) == outside
        comm.Barrier()

    if rank == 0:
        study = cuflynx_study.read_study(unzipped)
        assert [run['status'] for run in study['runs']] == ['finished']
        for path in _resolved_paths(study):
            assert os.path.commonpath([path, unzipped]) == unzipped, path
            assert os.path.exists(path), path
    comm.Barrier()


@pytest.mark.integration
def test_the_zips_yaml_generates_a_library_model_with_the_library_gone(tmp_path, monkeypatch):
    """The zip's yaml names the stored module versions, so a model built from a module library
    generates again, the same, from nothing but the unzipped folder."""
    import zipfile

    from libcuflynx.external_testing.workflow_library import build_workflow_library
    from libcuflynx.parsers.PrimitiveParsers import YamlFileParser
    from libcuflynx.scripts.script_generate_with_new_architecture import (
        generate_with_new_architecture)

    library = build_workflow_library(str(tmp_path / 'library'))
    working = tmp_path / 'study'
    generated = tmp_path / 'generated_models'
    config = {'file_prefix': 'chain', 'input_param_file': 'chain_parameters.json',
              'model_type': 'cellml', 'resources_dir': library['arrays'],
              'generated_models_dir': str(generated),
              'module_library_dirs': [library['modules']],
              'param_id_output_dir': str(working), 'DEBUG': False}
    assert generate_with_new_architecture(False, dict(config))
    model = generated / 'chain' / 'chain.cellml'
    original = model.read_text()
    inp = YamlFileParser().parse_user_inputs_file(dict(config))
    cuflynx_study.update_study(inp)
    cuflynx_study.export_working_dir(str(working), str(tmp_path / 'study.zip'))
    unzipped = tmp_path / 'unzipped'
    with zipfile.ZipFile(tmp_path / 'study.zip') as archive:
        archive.extractall(unzipped)
    shutil.rmtree(tmp_path / 'library')
    shutil.rmtree(generated)
    shutil.rmtree(working)

    dated = unzipped / 'inputs' / 'user_inputs' / cuflynx_study.dated_user_inputs_name()
    _run_from(str(dated), monkeypatch, str(tmp_path))
    assert generate_with_new_architecture(False, None)
    assert (unzipped / 'generated_models' / 'chain' / 'chain.cellml').read_text() == original


@pytest.mark.unit
@pytest.mark.parametrize('copy_inputs', [True, False])
def test_sensitivity_output_defaults_into_the_working_dir(tmp_path, copy_inputs):
    """Sobol output goes in ``<working dir>/sensitivity``, or where it did before with
    ``cuflynx_study_copy_inputs: false``; an absolute ``output_dir`` is kept as given."""
    from libcuflynx.parsers.PrimitiveParsers import YamlFileParser
    from libcuflynx.utilities.paths import default_sensitivity_outputs_dir

    def parsed(**extra):
        return YamlFileParser().parse_user_inputs_file({
            'file_prefix': PREFIX, 'input_param_file': f'{PREFIX}_parameters.json',
            'resources_dir': str(tmp_path / 'resources'),
            'param_id_output_dir': str(tmp_path / 'out'),
            'generated_models_dir': str(tmp_path / 'generated_models'),
            'cuflynx_study_copy_inputs': copy_inputs, **extra})['sa_options']['output_dir']

    expected = (str(tmp_path / 'out' / 'sensitivity') if copy_inputs else
                os.path.join(default_sensitivity_outputs_dir(), f'{PREFIX}_SA_results'))
    assert parsed() == expected
    elsewhere = str(tmp_path / 'elsewhere')
    assert parsed(sa_options={'output_dir': elsewhere}) == elsewhere
    if copy_inputs:
        assert parsed(sa_options={'output_dir': 'mine'}) == str(tmp_path / 'out' / 'sensitivity'
                                                                 / 'mine')


@pytest.mark.integration
def test_a_generated_models_system_regenerates_it_from_the_working_dir_alone(tmp_path):
    """``inputs/system/`` holds the module array, its parameters file and each module library
    version the array uses, so the model generates again, the same, with only the working dir
    left. Built-in modules are not copied: they come with libcuflynx."""
    from libcuflynx.external_testing.workflow_library import build_workflow_library
    from libcuflynx.parsers.PrimitiveParsers import YamlFileParser
    from libcuflynx.scripts.script_generate_with_new_architecture import (
        generate_with_new_architecture)

    library = build_workflow_library(str(tmp_path / 'library'))
    working = tmp_path / 'study'
    generated = tmp_path / 'generated_models'
    config = {'file_prefix': 'chain', 'input_param_file': 'chain_parameters.json',
              'model_type': 'cellml', 'resources_dir': library['arrays'],
              'generated_models_dir': str(generated),
              'module_library_dirs': [library['modules']],
              'param_id_output_dir': str(working), 'DEBUG': False}
    assert generate_with_new_architecture(False, dict(config))
    model = generated / 'chain' / 'chain.cellml'
    original = model.read_text()
    inp = YamlFileParser().parse_user_inputs_file(dict(config))
    cuflynx_study.update_study(dict(inp, model_path=str(model)))

    with open(cuflynx_study.manifest_path(inp)) as handle:
        raw = json.load(handle)
    system_dir = os.path.dirname(raw['system']['module_array'])
    assert os.path.dirname(system_dir) == os.path.join('inputs', 'system')
    assert raw['system'] == {
        'module_array': os.path.join(system_dir, 'chain_module_array.json'),
        'parameters': os.path.join(system_dir, 'chain_parameters.json'),
        'module_library_dirs': [os.path.join(system_dir, 'modules')],
        'use_builtin_modules': True}
    versions = sorted(os.path.relpath(root, working / system_dir / 'modules')
                      for root, _, names in os.walk(working / system_dir / 'modules')
                      if any(n.endswith('_modules_config.json') for n in names))
    # the versions the array uses (lin_a and lin_c), not the rest of the library
    assert versions == [os.path.join(t, 'versions', 'v1') for t in ('lin_a', 'lin_c')]

    archive = shutil.make_archive(str(tmp_path / 'study'), 'zip', str(working))
    moved = tmp_path / 'moved'
    shutil.unpack_archive(archive, str(moved))
    for original_dir in ('library', 'generated_models', 'study'):
        shutil.rmtree(tmp_path / original_dir)

    again = cuflynx_study.generation_inputs(cuflynx_study.read_study(str(moved)))
    assert os.path.commonpath([again['resources_dir'], str(moved)]) == str(moved)
    regenerated = tmp_path / 'regenerated'
    assert generate_with_new_architecture(False, dict(again, model_type='cellml',
                                                      generated_models_dir=str(regenerated)))
    assert (regenerated / 'chain' / 'chain.cellml').read_text() == original
