"""A calibration run leaves one working dir that holds every input and output of its study.

The working dir is ``param_id_output_dir``: the study manifest there records every path
relative to itself, and each input from elsewhere (model, obs_data, params_for_id, the
user_inputs yaml, user funcs, the module array and parameters file) is copied into its
``inputs/`` folder (``utilities/cuflynx_study.py``). So ``zip -r`` of that one folder is the
whole study: unzipped anywhere, with the originals gone, every path in it still resolves.
"""
import json
import os
import shutil
import tempfile
from datetime import date

import pytest
import yaml
from mpi4py import MPI

from libcuflynx.utilities import cuflynx_study

PREFIX = '3compartment'
MODEL_FILES = (f'{PREFIX}_module_array.csv', f'{PREFIX}_parameters.json',
               f'{PREFIX}_params_for_id.json', f'{PREFIX}_obs_data.json')


def _workspace(comm, name):
    """A fresh directory under the temp dir, the same on every rank (``tmp_path`` is not)."""
    root = os.path.join(tempfile.gettempdir(), 'libcuflynx_working_dir', name)
    if comm.Get_rank() == 0:
        shutil.rmtree(root, ignore_errors=True)
        os.makedirs(os.path.join(root, 'resources'))
    comm.Barrier()
    return root


def _files(directory):
    return sorted(os.path.relpath(os.path.join(root, name), directory)
                  for root, _, names in os.walk(directory) for name in names)


def _resolved_paths(study):
    paths = [study[key] for key in cuflynx_study.PATH_KEYS if study.get(key)]
    for block in ('user_funcs', 'system'):
        paths += list((study.get(block) or {}).values())
    for section in cuflynx_study.ENTRY_SECTIONS:
        paths += [entry[key] for entry in study[section] for key in ('dir', 'obs_data')
                  if entry.get(key)]
    return paths


@pytest.mark.integration
@pytest.mark.mpi
def test_a_calibration_leaves_a_working_dir_that_can_be_moved(base_user_inputs, resources_dir):
    from libcuflynx.scripts.param_id_run_script import run_param_id
    from libcuflynx.scripts.script_generate_with_new_architecture import (
        generate_with_new_architecture)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    workspace = _workspace(comm, 'calibration')
    resources = os.path.join(workspace, 'resources')
    working = os.path.join(workspace, 'study')
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
        'param_id_output_dir': working,
        'param_id_obs_path': os.path.join(resources, f'{PREFIX}_obs_data.json'),
        'operation_funcs_external_path': funcs,
        'user_inputs_path': user_inputs,
        'debug_optimiser_options': {'num_calls_to_function': 40, 'max_patience': 500},
    })
    if rank == 0:
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
        for path in _resolved_paths(study):
            assert os.path.commonpath([path, moved]) == moved, path
            assert os.path.exists(path), path
    comm.Barrier()


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
