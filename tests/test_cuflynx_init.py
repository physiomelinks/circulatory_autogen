"""``cuflynx-init`` writes a starter study that the stage commands run from unchanged."""
import fnmatch
import os
import pathlib
import re

import pytest

from libcuflynx.scripts import init_script

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_STARTER_DIR = _REPO_ROOT / 'src' / 'libcuflynx' / 'scripts' / 'starter'


@pytest.mark.unit
@pytest.mark.parametrize('name', [n for n, _ in init_script.STARTER_FILES
                                  if n != 'user_inputs.yaml'])
def test_the_starter_resources_are_the_repository_s_3compartment_files(name):
    """The starter is the main test model: a change to resources/ must be copied here."""
    assert (_STARTER_DIR / name).read_bytes() == (_REPO_ROOT / 'resources' / name).read_bytes(), (
        f'src/libcuflynx/scripts/starter/{name} differs from resources/{name}: copy it across')


@pytest.mark.unit
def test_every_starter_file_ships_in_the_wheel():
    """Each file must match a [tool.setuptools.package-data] pattern of libcuflynx.scripts."""
    text = (_REPO_ROOT / 'pyproject.toml').read_text()
    entry = re.search(r'^"libcuflynx\.scripts" = \[(.*?)\]', text, re.S | re.M).group(1)
    patterns = re.findall(r'"([^"]+)"', entry)
    names = sorted(p.name for p in _STARTER_DIR.iterdir())
    assert names == sorted(n for n, _ in init_script.STARTER_FILES)
    for name in names:
        assert any(fnmatch.fnmatch(f'starter/{name}', pat) for pat in patterns), name


@pytest.mark.unit
def test_init_writes_the_yaml_and_resources(tmp_path):
    written = init_script.write_starter(tmp_path)
    assert sorted(pathlib.Path(p).relative_to(tmp_path).as_posix() for p in written) == [
        'resources/3compartment_module_array.csv', 'resources/3compartment_obs_data.json',
        'resources/3compartment_parameters.json', 'resources/3compartment_params_for_id.json',
        'user_run_files/user_inputs.yaml']
    assert (tmp_path / 'user_run_files' / 'user_inputs.yaml').read_bytes() == \
        (_STARTER_DIR / 'user_inputs.yaml').read_bytes()


@pytest.mark.unit
def test_init_overwrites_nothing_without_force(tmp_path, capsys):
    yaml_path = tmp_path / 'user_run_files' / 'user_inputs.yaml'
    yaml_path.parent.mkdir()
    yaml_path.write_text('mine: true\n')

    assert init_script.main([str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert str(yaml_path) in err and '--force' in err
    assert yaml_path.read_text() == 'mine: true\n'
    assert not (tmp_path / 'resources').exists(), 'nothing is written when one file exists'

    assert init_script.main([str(tmp_path), '--force']) == 0
    assert yaml_path.read_bytes() == (_STARTER_DIR / 'user_inputs.yaml').read_bytes()


@pytest.mark.unit
def test_init_says_to_set_the_user_dir_when_it_is_elsewhere(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('CUFLYNX_USER_DIR', str(tmp_path / 'elsewhere'))
    assert init_script.main([str(tmp_path / 'study')]) == 0
    out = capsys.readouterr().out
    assert f'export CUFLYNX_USER_DIR={tmp_path / "study"}' in out
    assert 'cuflynx-generate' in out


@pytest.fixture
def study(tmp_path, monkeypatch):
    """A directory holding the starter, set as the user directory and the current directory.

    The parser fixes where it looks for user_inputs.yaml when it is imported, so a process
    that set CUFLYNX_USER_DIR afterwards (this one) points that at the study too.
    """
    from libcuflynx.parsers import PrimitiveParsers

    init_script.write_starter(tmp_path)
    monkeypatch.setenv('CUFLYNX_USER_DIR', str(tmp_path))
    monkeypatch.setattr(PrimitiveParsers, 'user_inputs_dir', str(tmp_path / 'user_run_files'))
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.unit
def test_the_starter_yaml_parses_with_its_paths_under_the_study(study):
    from libcuflynx.parsers.PrimitiveParsers import YamlFileParser

    tmp_path = study
    parsed = YamlFileParser().parse_user_inputs_file(None, obs_path_needed=True)

    assert parsed['param_id_obs_path'] == 'resources/3compartment_obs_data.json', \
        'the starter yaml was not the one read'
    assert os.path.realpath(parsed['resources_dir']) == os.path.realpath(tmp_path / 'resources')
    assert os.path.isfile(parsed['param_id_obs_path'])
    assert os.path.realpath(parsed['params_for_id_path']) == os.path.realpath(
        tmp_path / 'resources' / '3compartment_params_for_id.json')


@pytest.mark.unit
def test_the_starter_calibrates_with_cma_es(study):
    """CMA-ES runs on any budget; the genetic algorithm needs 744 evaluations a generation."""
    from libcuflynx.parsers.PrimitiveParsers import YamlFileParser

    parsed = YamlFileParser().parse_user_inputs_file(None, obs_path_needed=True)
    assert parsed['param_id_method'] == 'CMA-ES'


@pytest.mark.integration
def test_a_model_generates_from_the_starter(study):
    from libcuflynx.scripts.script_generate_with_new_architecture import \
        generate_with_new_architecture

    tmp_path = study
    assert generate_with_new_architecture(False, None)
    assert (tmp_path / 'generated_models' / '3compartment' / '3compartment.cellml').is_file()
