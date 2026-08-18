"""The package must never be written into (issue #431).

Installed, ``libcuflynx`` lives in ``site-packages``. Every ``__file__``-relative "repo
root" the code used to compute resolved *there* — a meaningless place to look for a
user's model inputs, and a writable one, so a run would quietly scatter generated
models, calibration output and dated ``user_inputs_<yymmdd>.yaml`` snapshots through
the installed package instead of failing loudly.

The load-bearing test here is a hash sweep: run a real generation and a real (short)
calibration with ``resources_dir``/``generated_models_dir``/``param_id_output_dir``
pointed outside the repo entirely, and assert the package tree is byte-identical
afterwards. It fires for *any* new write site, not only the ones known today, which is
why it is worth its runtime.
"""

import hashlib
import os
import shutil
import tempfile
from datetime import date

import pytest
import yaml
from mpi4py import MPI

from libcuflynx.parsers.PrimitiveParsers import save_dated_user_inputs
from libcuflynx.utilities import paths
from libcuflynx.utilities.paths import package_dir


# ---------------------------------------------------------------------------
# hash sweep
# ---------------------------------------------------------------------------

def _sweep_package(root=None):
    """{relative path -> sha256} for every file in the package tree.

    ``__pycache__`` is excluded: CPython writes bytecode caches next to the source of
    any importable package, installed or not, and that is the interpreter's doing
    rather than ours. Everything else is fair game.
    """
    root = root or package_dir()
    digests = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != '__pycache__']
        for filename in filenames:
            full = os.path.join(dirpath, filename)
            rel = os.path.relpath(full, root)
            try:
                with open(full, 'rb') as fh:
                    digests[rel] = hashlib.sha256(fh.read()).hexdigest()
            except OSError:
                # A file we cannot read is still a file that exists; record that much
                # so an appearing-then-unreadable file is not silently ignored.
                digests[rel] = '<unreadable>'
    return digests


def _describe_package_changes(before, after):
    """A message naming the offending files, or '' when the tree is untouched.

    Naming them is the whole point: "the package changed" tells a future maintainer
    nothing about which write site regressed.
    """
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    modified = sorted(p for p in set(before) & set(after) if before[p] != after[p])
    if not (added or removed or modified):
        return ''
    parts = [f'the libcuflynx package tree at {package_dir()} was modified by this run; '
             'nothing may write inside the installed package (#431)']
    if added:
        parts.append('  files CREATED inside the package:\n' +
                     '\n'.join(f'    {p}' for p in added))
    if modified:
        parts.append('  files MODIFIED inside the package:\n' +
                     '\n'.join(f'    {p}' for p in modified))
    if removed:
        parts.append('  files DELETED from the package:\n' +
                     '\n'.join(f'    {p}' for p in removed))
    return '\n'.join(parts)


def _assert_package_unchanged(before):
    message = _describe_package_changes(before, _sweep_package())
    assert not message, message


# ---------------------------------------------------------------------------
# defaults never point inside the package
# ---------------------------------------------------------------------------

_DEFAULT_PATH_FUNCS = (
    'default_resources_dir',
    'default_generated_models_dir',
    'default_param_id_output_dir',
    'default_sensitivity_outputs_dir',
    'default_user_inputs_dir',
    'default_module_config_user_dir',
    'default_funcs_user_dir',
)


@pytest.mark.unit
@pytest.mark.parametrize('func_name', _DEFAULT_PATH_FUNCS)
def test_default_dirs_are_never_inside_the_package(func_name):
    """Every default user-data directory resolves outside the package, in a checkout."""
    resolved = os.path.abspath(getattr(paths, func_name)())
    pkg = os.path.abspath(package_dir())
    assert os.path.commonpath([resolved, pkg]) != pkg, \
        f'{func_name}() resolved to {resolved}, which is inside the package at {pkg}'


@pytest.mark.unit
@pytest.mark.parametrize('func_name', _DEFAULT_PATH_FUNCS)
def test_default_dirs_fall_back_to_cwd_when_not_a_checkout(func_name, tmp_path, monkeypatch):
    """With no checkout around the package — the pip-install case — defaults follow the cwd.

    Simulated by making ``repo_root()`` answer ``None``, which is exactly what it does
    when ``libcuflynx`` sits in ``site-packages`` rather than under a ``src/`` beside a
    ``pyproject.toml``.
    """
    monkeypatch.setattr(paths, 'repo_root', lambda: None)
    monkeypatch.delenv(paths.CUFLYNX_USER_DIR_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)

    resolved = os.path.abspath(getattr(paths, func_name)())
    assert resolved.startswith(os.path.abspath(str(tmp_path)) + os.sep), \
        f'{func_name}() resolved to {resolved}, expected it under the cwd {tmp_path}'


@pytest.mark.unit
def test_user_data_root_honours_the_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.CUFLYNX_USER_DIR_ENV_VAR, str(tmp_path))
    assert paths.user_data_root() == os.path.abspath(str(tmp_path))


@pytest.mark.unit
def test_repo_root_refuses_a_directory_without_the_markers(tmp_path, monkeypatch):
    """A ``src/libcuflynx`` layout alone is not a checkout — the markers have to be there."""
    fake_pkg = tmp_path / 'src' / 'libcuflynx'
    fake_pkg.mkdir(parents=True)
    monkeypatch.setattr(paths, '_PACKAGE_DIR', str(fake_pkg))
    assert paths.repo_root() is None

    (tmp_path / 'pyproject.toml').write_text('')
    (tmp_path / 'user_run_files').mkdir()
    assert paths.repo_root() == str(tmp_path)


# ---------------------------------------------------------------------------
# save_dated_user_inputs
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_save_dated_user_inputs_writes_to_resources_dir_not_the_package(tmp_path):
    """The dated run-config snapshot lands in the user's resources_dir, and only there.

    ``save_dated_user_inputs`` runs on *every* parse of a config, so if it ever resolved
    its output against the package it would be the single fastest way to pollute an
    install.
    """
    before = _sweep_package()

    resources_dir = tmp_path / 'somewhere' / 'outside' / 'resources'
    resources_dir.mkdir(parents=True)
    save_dated_user_inputs({'resources_dir': str(resources_dir), 'file_prefix': '3compartment'})

    expected = resources_dir / f"user_inputs_{date.today().strftime('%y%m%d')}.yaml"
    assert expected.is_file(), f'expected the dated config archive at {expected}'
    assert yaml.safe_load(expected.read_text())['file_prefix'] == '3compartment'

    _assert_package_unchanged(before)


@pytest.mark.unit
def test_parsed_config_archives_into_the_configured_resources_dir(tmp_path):
    """Going through the parser (not the writer directly) still archives outside the package."""
    from libcuflynx.utilities.utility_funcs import get_default_inp_data_dict

    before = _sweep_package()

    resources_dir = tmp_path / 'resources'
    resources_dir.mkdir()
    inp = get_default_inp_data_dict('3compartment', '3compartment_parameters.csv',
                                    str(resources_dir))

    assert inp['resources_dir'] == str(resources_dir)
    dated = resources_dir / f"user_inputs_{date.today().strftime('%y%m%d')}.yaml"
    assert dated.is_file(), \
        f'parse_user_inputs_file should archive its resolved config at {dated}'

    _assert_package_unchanged(before)


@pytest.mark.unit
def test_config_defaults_follow_the_user_dir_when_nothing_is_configured(tmp_path, monkeypatch):
    """With no ``*_dir`` keys in the config at all, the parser's defaults hang off the
    user's directory — never off the package.

    This is the path an installed run takes when the user configures nothing, and it is
    the one that used to resolve into ``site-packages``.
    """
    from libcuflynx.parsers.PrimitiveParsers import YamlFileParser

    monkeypatch.setenv(paths.CUFLYNX_USER_DIR_ENV_VAR, str(tmp_path))
    (tmp_path / 'resources').mkdir()

    before = _sweep_package()
    parsed = YamlFileParser().parse_user_inputs_file(
        {'file_prefix': _MODEL_PREFIX, 'input_param_file': f'{_MODEL_PREFIX}_parameters.csv'},
        obs_path_needed=False)

    for key in ('resources_dir', 'generated_models_dir', 'param_id_output_dir'):
        resolved = os.path.abspath(parsed[key])
        assert resolved.startswith(os.path.abspath(str(tmp_path)) + os.sep), \
            f'default {key} resolved to {resolved}, expected it under {tmp_path}'
    assert os.path.abspath(parsed['sa_options']['output_dir']).startswith(
        os.path.abspath(str(tmp_path)) + os.sep)

    _assert_package_unchanged(before)


# ---------------------------------------------------------------------------
# the real thing: generation + calibration, entirely outside the repo
# ---------------------------------------------------------------------------

_MODEL_PREFIX = '3compartment'
_MODEL_FILES = (
    f'{_MODEL_PREFIX}_vessel_array.csv',
    f'{_MODEL_PREFIX}_parameters.csv',
    f'{_MODEL_PREFIX}_params_for_id.csv',
    f'{_MODEL_PREFIX}_obs_data.json',
)


@pytest.fixture(scope="function")
def mpi_comm():
    """The world communicator (matching the other integration test modules)."""
    return MPI.COMM_WORLD


def _outside_repo_workspace(mpi_comm, name):
    """A workspace under the system temp dir — deliberately nowhere near the checkout.

    ``tmp_path`` is per-process, so it would differ between MPI ranks; this is derived
    from a fixed name instead and created once on rank 0.
    """
    root = os.path.join(tempfile.gettempdir(), 'libcuflynx_431_outside_repo', name)
    if mpi_comm.Get_rank() == 0:
        shutil.rmtree(root, ignore_errors=True)
        for sub in ('resources', 'generated_models', 'param_id_output'):
            os.makedirs(os.path.join(root, sub), exist_ok=True)
    mpi_comm.Barrier()
    return root


@pytest.mark.integration
@pytest.mark.mpi
def test_generation_and_calibration_never_write_into_the_package(
        base_user_inputs, resources_dir, mpi_comm):
    """Generate and calibrate with every path outside the repo; the package must not change.

    This is the acceptance test for #431. It is deliberately end-to-end and deliberately
    blunt: any write site anywhere under ``libcuflynx`` — present or future — that
    resolves against the package rather than against a config-supplied path will show up
    here as a created or modified file, named in the failure message.

    Kept out of the ``slow`` set: 3compartment with a one-generation genetic algorithm is
    the cheapest run that still exercises generation, the CellML/Myokit solver path,
    optimisation, output archival and the dated config snapshot.
    """
    from libcuflynx.scripts.param_id_run_script import run_param_id
    from libcuflynx.scripts.script_generate_with_new_architecture import (
        generate_with_new_architecture)

    rank = mpi_comm.Get_rank()
    workspace = _outside_repo_workspace(mpi_comm, 'generation_and_calibration')
    user_resources_dir = os.path.join(workspace, 'resources')

    if rank == 0:
        for filename in _MODEL_FILES:
            shutil.copy2(os.path.join(resources_dir, filename),
                         os.path.join(user_resources_dir, filename))
    mpi_comm.Barrier()

    # Nothing in this config points at the repo, let alone at the package.
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': _MODEL_PREFIX,
        'input_param_file': f'{_MODEL_PREFIX}_parameters.csv',
        'model_type': 'cellml',
        'solver': 'CVODE_myokit',
        'param_id_method': 'genetic_algorithm',
        'pre_time': 2,
        'sim_time': 1,
        'dt': 0.01,
        'DEBUG': True,
        'do_uq': False,
        'do_ia': False,
        'plot_predictions': False,
        'solver_info': {'MaximumStep': 0.001, 'MaximumNumberOfSteps': 5000},
        'resources_dir': user_resources_dir,
        'generated_models_dir': os.path.join(workspace, 'generated_models'),
        'param_id_output_dir': os.path.join(workspace, 'param_id_output'),
        'param_id_obs_path': os.path.join(user_resources_dir,
                                          f'{_MODEL_PREFIX}_obs_data.json'),
        'debug_optimiser_options': {'num_calls_to_function': 40, 'max_patience': 500},
    })

    before = _sweep_package()

    if rank == 0:
        assert generate_with_new_architecture(False, config), \
            'generation into an out-of-repo generated_models_dir should succeed'
    mpi_comm.Barrier()

    run_param_id(config)
    mpi_comm.Barrier()

    if rank == 0:
        # The run really happened where we asked it to -- otherwise "the package is
        # unchanged" would be trivially true and prove nothing.
        model_path = os.path.join(workspace, 'generated_models', _MODEL_PREFIX,
                                  f'{_MODEL_PREFIX}.cellml')
        assert os.path.isfile(model_path), f'expected the generated model at {model_path}'
        out_dir = os.path.join(workspace, 'param_id_output',
                               f'genetic_algorithm_{_MODEL_PREFIX}_{_MODEL_PREFIX}_obs_data')
        assert os.path.isfile(os.path.join(out_dir, 'best_cost.npy')), \
            f'expected calibration output under {out_dir}'
        dated = os.path.join(user_resources_dir,
                             f"user_inputs_{date.today().strftime('%y%m%d')}.yaml")
        assert os.path.isfile(dated), \
            f'the dated config archive belongs in the user resources_dir, expected {dated}'

        _assert_package_unchanged(before)
    mpi_comm.Barrier()


@pytest.mark.integration
@pytest.mark.mpi
def test_generation_works_with_no_checkout_directories_present(
        resources_dir, mpi_comm, tmp_path, monkeypatch):
    """Generation succeeds when only the *package* and the user's resources exist.

    A pip install has no ``module_config_user/`` and no ``funcs_user/`` -- those are
    checkout directories. Pointing ``$CUFLYNX_USER_DIR`` at a bare directory reproduces
    that layout without needing an actual install, and catches the class of bug where the
    code assumes a checkout directory is always there (the old ``os.listdir`` on
    ``module_config_user`` raised ``FileNotFoundError`` in an install).

    It also pins the units file staying well-formed with a *single* units script: the
    concatenation used to drop ``</model>`` from the first file and only re-add it from
    the last, which silently produced an unterminated CellML when there was only one.
    """
    from libcuflynx.scripts.script_generate_with_new_architecture import (
        generate_with_new_architecture)
    from libcuflynx.utilities.utility_funcs import get_default_inp_data_dict

    if mpi_comm.Get_rank() != 0:
        mpi_comm.Barrier()
        return

    user_dir = tmp_path / 'user_dir'
    user_resources = user_dir / 'resources'
    user_resources.mkdir(parents=True)
    for filename in _MODEL_FILES:
        shutil.copy2(os.path.join(resources_dir, filename), user_resources / filename)
    assert not (user_dir / 'module_config_user').exists()
    assert not (user_dir / 'funcs_user').exists()

    monkeypatch.setenv(paths.CUFLYNX_USER_DIR_ENV_VAR, str(user_dir))
    before = _sweep_package()

    config = get_default_inp_data_dict(_MODEL_PREFIX, f'{_MODEL_PREFIX}_parameters.csv',
                                       str(user_resources))
    config.update({'model_type': 'cellml', 'solver': 'CVODE_myokit',
                   'pre_time': 2, 'sim_time': 1, 'dt': 0.01})
    for key in ('generated_models_dir', 'param_id_output_dir'):
        assert str(user_dir) in config[key], \
            f'{key} should default under $CUFLYNX_USER_DIR, got {config[key]}'

    assert generate_with_new_architecture(False, config), \
        'generation should succeed with no module_config_user/ or funcs_user/ present'

    units_path = os.path.join(config['generated_models_dir'], _MODEL_PREFIX,
                              f'{_MODEL_PREFIX}_units.cellml')
    units_text = open(units_path).read()
    assert units_text.rstrip().endswith('</model>'), \
        f'{units_path} is not a closed CellML model; the units concatenation dropped </model>'

    _assert_package_unchanged(before)
    mpi_comm.Barrier()
