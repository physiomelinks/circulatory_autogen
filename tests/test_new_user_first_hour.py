"""What a new user meets in the first hour: a missing or mistyped configuration, a forgotten
module library, a name the model does not have, a missing compiler header.

Each test is one mistake, and checks that the message says what to change.
"""
import argparse
import glob
import os
import re
import shutil
import warnings

import pytest
import yaml

from libcuflynx.parsers import PrimitiveParsers
from libcuflynx.parsers.PrimitiveParsers import (KNOWN_USER_INPUTS_KEYS, UserInputsError,
                                                YamlFileParser, ObsAndParamDataParser,
                                                unknown_user_inputs_keys,
                                                warn_unknown_user_inputs_keys)
from libcuflynx.scripts import _cli
from libcuflynx.utilities import paths

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, 'src', 'libcuflynx')
SHIPPED_YAML = os.path.join(REPO, 'user_run_files', 'user_inputs.yaml')


def _study(tmp_path, *suffixes):
    resources = tmp_path / 'resources'
    resources.mkdir(exist_ok=True)
    for suffix in suffixes or ('_module_array.csv', '_parameters.json'):
        shutil.copy(os.path.join(REPO, 'resources', '3compartment' + suffix), resources)
    return {'file_prefix': '3compartment', 'input_param_file': '3compartment_parameters.json',
            'model_type': 'cellml', 'resources_dir': str(resources),
            'generated_models_dir': str(tmp_path / 'generated'),
            'param_id_output_dir': str(tmp_path / 'param_id_output'),
            'dt': 0.01, 'pre_time': 0.0, 'sim_time': 1.0}


# ---- a missing user_inputs.yaml ----------------------------------------------------------

@pytest.mark.unit
def test_a_missing_user_inputs_yaml_names_the_user_directory_and_what_to_do(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.CUFLYNX_USER_DIR_ENV_VAR, str(tmp_path))
    monkeypatch.setattr(PrimitiveParsers, 'user_inputs_dir', paths.default_user_inputs_dir())
    with pytest.raises(FileNotFoundError) as exc:
        YamlFileParser().parse_user_inputs_file(None)
    message = str(exc.value)
    assert '\n' not in message
    assert str(tmp_path) in message and '$CUFLYNX_USER_DIR' in message
    assert 'user_run_files/user_inputs.yaml' in message and '--user-inputs' in message


@pytest.mark.unit
def test_the_commands_report_a_missing_user_inputs_yaml_in_one_line(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.CUFLYNX_USER_DIR_ENV_VAR, str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        _cli.load_user_inputs(argparse.Namespace(user_inputs=None))
    message = str(exc.value.code)
    assert '\n' not in message and str(tmp_path) in message and '--user-inputs' in message


@pytest.mark.unit
def test_the_user_directory_message_says_how_it_was_chosen(tmp_path, monkeypatch):
    monkeypatch.delenv(paths.CUFLYNX_USER_DIR_ENV_VAR, raising=False)
    monkeypatch.setattr(paths, 'repo_root', lambda: None)
    monkeypatch.chdir(tmp_path)
    assert paths.user_data_root_source() == 'the current directory'
    monkeypatch.setattr(paths, 'repo_root', lambda: str(tmp_path))
    assert 'checkout' in paths.user_data_root_source()


# ---- the shipped user_inputs.yaml ---------------------------------------------------------

@pytest.mark.unit
def test_the_shipped_user_inputs_yaml_names_no_path_of_one_machine():
    with open(SHIPPED_YAML) as f:
        text = f.read()
    assert '/home/' not in text and '/hpc/' not in text
    config = yaml.safe_load(text)
    assert not os.path.isabs(config['param_id_obs_path'])
    assert os.path.isfile(os.path.join(REPO, config['param_id_obs_path']))
    for key in ('cpp_generated_models_dir', 'cpp_1d_model_config_path'):
        assert key not in config


@pytest.mark.unit
def test_a_relative_obs_path_is_found_in_the_user_directory(tmp_path, monkeypatch):
    """`cd user_run_files && ./run_param_id.sh` starts in a directory the path is not relative
    to: it is then taken from the user directory."""
    study = tmp_path / 'study'
    study.mkdir()
    config = _study(study, '_module_array.csv', '_parameters.json', '_obs_data.json',
                    '_params_for_id.json')
    config['param_id_obs_path'] = 'resources/3compartment_obs_data.json'
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv(paths.CUFLYNX_USER_DIR_ENV_VAR, str(study))
    parsed = YamlFileParser().parse_user_inputs_file(config, obs_path_needed=True)
    assert parsed['param_id_obs_path'] == str(study / 'resources' / '3compartment_obs_data.json')


@pytest.mark.unit
def test_a_relative_obs_path_from_the_current_directory_is_kept(tmp_path, monkeypatch):
    config = _study(tmp_path, '_module_array.csv', '_parameters.json', '_obs_data.json',
                    '_params_for_id.json')
    config['param_id_obs_path'] = 'resources/3compartment_obs_data.json'
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(paths.CUFLYNX_USER_DIR_ENV_VAR, str(tmp_path / 'not_here'))
    parsed = YamlFileParser().parse_user_inputs_file(config, obs_path_needed=True)
    assert parsed['param_id_obs_path'] == 'resources/3compartment_obs_data.json'


@pytest.mark.unit
@pytest.mark.parametrize('absolute_override', [True, False])
def test_a_relative_obs_path_not_beside_the_override_is_found_in_the_user_directory(
        tmp_path, monkeypatch, absolute_override):
    """With user_inputs_path_override, a relative obs path is first taken from the override's
    directory, then from the user directory."""
    study = tmp_path / 'study'
    study.mkdir()
    config = _study(study, '_module_array.csv', '_parameters.json', '_obs_data.json',
                    '_params_for_id.json')
    config['param_id_obs_path'] = 'resources/3compartment_obs_data.json'
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs' / 'other.yaml').write_text(yaml.safe_dump(config))
    default_dir = tmp_path / 'default'
    default_dir.mkdir()
    override = (str(tmp_path / 'configs' / 'other.yaml') if absolute_override
                else os.path.join('configs', 'other.yaml'))
    (default_dir / 'user_inputs.yaml').write_text(
        yaml.safe_dump({'user_inputs_path_override': override}))
    monkeypatch.setattr(PrimitiveParsers, 'user_inputs_dir', str(default_dir))
    monkeypatch.setenv(paths.CUFLYNX_USER_DIR_ENV_VAR, str(study))
    monkeypatch.chdir(tmp_path)
    parsed = YamlFileParser().parse_user_inputs_file(None, obs_path_needed=True)
    assert parsed['param_id_obs_path'] == str(study / 'resources' / '3compartment_obs_data.json')


# ---- a mistyped user_inputs key -----------------------------------------------------------

@pytest.mark.unit
def test_an_unknown_user_inputs_key_is_named_with_the_closest_key():
    unknown = unknown_user_inputs_keys({'file_prefix': 'x', 'param_id_methd': 'CMA-ES',
                                        'sim_tme': 2, 'zzz_nothing_like_it': 1})
    assert unknown == {'param_id_methd': 'param_id_method', 'sim_tme': 'sim_time',
                       'zzz_nothing_like_it': None}


@pytest.mark.unit
@pytest.mark.parametrize('key', ['module_array_path', 'parameters_path', 'model_pth',
                                 'param_id_output_dir_abs', 'vessels_csv_path'])
def test_a_key_the_parser_derives_is_never_suggested(key):
    assert unknown_user_inputs_keys({key: 1})[key] not in PrimitiveParsers.DERIVED_USER_INPUTS_KEYS


@pytest.mark.unit
def test_a_key_the_parser_derives_does_not_warn():
    assert not unknown_user_inputs_keys(dict.fromkeys(PrimitiveParsers.DERIVED_USER_INPUTS_KEYS))


@pytest.mark.unit
@pytest.mark.parametrize('key', ['num_calls_to_function', 'cost_convergence', 'num_elite',
                                 'sigma0', 'num_starts'])
def test_an_optimiser_option_at_the_top_level_is_pointed_to_optimiser_options(key):
    assert unknown_user_inputs_keys({key: 100}) == {key: 'optimiser_options'}
    with pytest.warns(UserWarning, match=f"'{key}'.*an optimiser option: put it under "
                                         f"optimiser_options"):
        warn_unknown_user_inputs_keys({key: 100}, 'user_inputs.yaml')


@pytest.mark.unit
def test_a_user_inputs_file_with_a_mistyped_key_warns(tmp_path):
    path = tmp_path / 'user_inputs.yaml'
    path.write_text('file_prefix: 3compartment\nparam_id_methd: CMA-ES\n')
    with pytest.warns(UserWarning, match="'param_id_methd'.*Did you mean 'param_id_method'"):
        loaded = _cli.load_user_inputs(argparse.Namespace(user_inputs=str(path)))
    assert loaded['param_id_methd'] == 'CMA-ES'


@pytest.mark.unit
def test_the_default_user_inputs_file_with_a_mistyped_key_warns(tmp_path, monkeypatch):
    config = _study(tmp_path)
    config['sim_tme'] = config.pop('sim_time')
    (tmp_path / 'user_inputs.yaml').write_text(yaml.safe_dump(config))
    monkeypatch.setattr(PrimitiveParsers, 'user_inputs_dir', str(tmp_path))
    with pytest.warns(UserWarning, match="'sim_tme'.*Did you mean 'sim_time'"):
        YamlFileParser().parse_user_inputs_file(None)


@pytest.mark.unit
def test_a_dict_built_in_code_is_not_checked_for_unknown_keys(tmp_path):
    config = dict(_study(tmp_path), a_key_of_the_caller=1)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        YamlFileParser().parse_user_inputs_file(config)


@pytest.mark.unit
def test_the_shipped_user_inputs_yaml_has_no_unknown_key():
    with open(SHIPPED_YAML) as f:
        text = f.read()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        warn_unknown_user_inputs_keys(yaml.safe_load(text), SHIPPED_YAML)
    # the keys it offers commented out, at the top level
    commented = set(re.findall(r'^# ([A-Za-z_][A-Za-z0-9_]*):', text, re.M))
    assert commented and not unknown_user_inputs_keys(dict.fromkeys(commented - {'NOTE'}))


@pytest.mark.unit
def test_every_user_inputs_key_the_code_reads_is_known():
    read = set()
    for path in glob.glob(os.path.join(SRC, '**', '*.py'), recursive=True):
        with open(path) as f:
            source = f.read()
        read.update(re.findall(
            r"inp_data_dict(?:\.get\(|\[|\.setdefault\(|\.pop\()\s*['\"]([A-Za-z0-9_]+)['\"]", source))
        read.update(re.findall(r"['\"]([A-Za-z0-9_]+)['\"]\s+(?:not\s+)?in\s+inp_data_dict", source))
    read -= set(PrimitiveParsers.REMOVED_USER_INPUTS_KEYS)
    assert len(read) > 50
    assert not read - KNOWN_USER_INPUTS_KEYS, (
        'add these to KNOWN_USER_INPUTS_KEYS, or a user_inputs.yaml that sets them warns: '
        f'{sorted(read - KNOWN_USER_INPUTS_KEYS)}')


@pytest.mark.unit
def test_the_settings_a_calibration_workflow_takes_are_known_user_inputs_keys():
    from libcuflynx.calibration_workflow.spec import ALLOWED_SETTINGS
    assert not set(ALLOWED_SETTINGS) - KNOWN_USER_INPUTS_KEYS


# ---- a configuration the parser refuses: an exception, not exit() -------------------------

@pytest.mark.unit
def test_an_invalid_solver_raises_with_the_valid_solvers(tmp_path):
    config = dict(_study(tmp_path), solver='CVODE_myokitt')
    with pytest.raises(UserInputsError, match='Invalid solver: CVODE_myokitt') as exc:
        YamlFileParser().parse_user_inputs_file(config)
    assert 'CVODE_myokit' in str(exc.value).split('Valid solvers', 1)[1]


@pytest.mark.unit
def test_an_unsupported_solver_info_key_raises(tmp_path):
    config = dict(_study(tmp_path), model_type='cpp', solver='CVODE',
                  solver_info={'MaximumStep': 1e-4})
    with pytest.raises(UserInputsError, match=r"not supported by solver 'CVODE': \['MaximumStep'\]"):
        YamlFileParser().parse_user_inputs_file(config)


@pytest.mark.unit
def test_a_budget_below_the_genetic_algorithm_population_raises():
    from libcuflynx.param_id.optimisers import GeneticAlgorithmOptimiser
    optimiser = GeneticAlgorithmOptimiser.__new__(GeneticAlgorithmOptimiser)
    optimiser.comm, optimiser.rank, optimiser.num_procs, optimiser.DEBUG = None, 0, 1, False
    optimiser.optimiser_options = {'num_calls_to_function': 40}
    with pytest.raises(ValueError, match=r'num_calls_to_function=40 .*num_pop=\d+.*at least \d+'):
        optimiser.run()


@pytest.mark.unit
def test_more_ranks_than_the_genetic_algorithm_population_raises():
    from libcuflynx.param_id.optimisers import GeneticAlgorithmOptimiser
    optimiser = GeneticAlgorithmOptimiser.__new__(GeneticAlgorithmOptimiser)
    optimiser.comm, optimiser.rank, optimiser.num_procs, optimiser.DEBUG = None, 0, 10000, False
    optimiser.optimiser_options = {'num_calls_to_function': 100000}
    with pytest.raises(ValueError, match=r'10000 MPI ranks .*num_pop=\d+.*at most \d+ ranks'):
        optimiser.run()


def _default_user_inputs(tmp_path, monkeypatch, settings):
    user_run_files = tmp_path / 'user_run_files'
    user_run_files.mkdir()
    (user_run_files / 'user_inputs.yaml').write_text(yaml.safe_dump(settings))
    monkeypatch.setattr(PrimitiveParsers, 'user_inputs_dir', str(user_run_files))
    return user_run_files / 'user_inputs.yaml'


@pytest.mark.unit
def test_a_missing_user_inputs_path_override_raises_in_one_line(tmp_path, monkeypatch):
    missing = str(tmp_path / 'nowhere' / 'user_inputs.yaml')
    _default_user_inputs(tmp_path, monkeypatch, {'user_inputs_path_override': missing})
    with pytest.raises(FileNotFoundError) as exc:
        YamlFileParser().parse_user_inputs_file(None)
    message = str(exc.value)
    assert '\n' not in message and missing in message and 'user_inputs_path_override' in message


@pytest.mark.parametrize('settings, error', [
    ({'model_type': 'cellml'}, "UserInputsError: .*has no file_prefix"),
    ({'file_prefix': 'x', 'model_type': 'cellmll'}, "UserInputsError: Invalid model_type: cellmll"),
    ({'file_prefix': 'x', 'dt': 1}, "UserInputsError: dt must be a float"),
    ({'file_prefix': 'x', 'module_library_dirs': 'no_such_dir'},
     "FileNotFoundError: module_library_dirs entry .*no_such_dir does not exist"),
])
@pytest.mark.unit
def test_a_refused_user_inputs_file_raises_not_exits(tmp_path, monkeypatch, settings, error):
    """exit() would raise SystemExit(None), which the commands turn into exit status 0."""
    config = dict(_study(tmp_path), **settings)
    if 'file_prefix' not in settings:
        del config['file_prefix']
    _default_user_inputs(tmp_path, monkeypatch, config)
    with pytest.raises((UserInputsError, FileNotFoundError)) as exc:
        YamlFileParser().parse_user_inputs_file(None)
    assert re.match(error, f'{type(exc.value).__name__}: {exc.value}')
    assert '\n' not in str(exc.value)


@pytest.mark.unit
def test_cuflynx_generate_with_a_missing_override_exits_1(tmp_path):
    import subprocess
    import sys
    missing = str(tmp_path / 'nowhere' / 'user_inputs.yaml')
    (tmp_path / 'user_run_files').mkdir()
    (tmp_path / 'user_run_files' / 'user_inputs.yaml').write_text(
        yaml.safe_dump({'user_inputs_path_override': missing}))
    env = dict(os.environ, **{paths.CUFLYNX_USER_DIR_ENV_VAR: str(tmp_path)})
    result = subprocess.run(
        [sys.executable, '-m', 'libcuflynx.scripts.script_generate_with_new_architecture'],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 1, result.stdout + result.stderr
    last_line = result.stderr.strip().splitlines()[-1]
    assert last_line.startswith('FileNotFoundError: ') and missing in last_line


@pytest.mark.unit
def test_a_cpp_cvode_model_without_dt_solver_says_which_key_to_add(tmp_path):
    from libcuflynx.scripts.script_generate_with_new_architecture import \
        generate_with_new_architecture
    config = dict(_study(tmp_path), model_type='cpp', solver='CVODE', solver_info={'rtol': 1e-6})
    with pytest.raises(ValueError, match=r'solver_info\.dt_solver'):
        generate_with_new_architecture(False, config)


# ---- a forgotten module library ------------------------------------------------------------

@pytest.mark.unit
def test_a_module_with_no_library_says_how_to_give_one():
    from libcuflynx.utilities.parameterisations import component_parameterisation_rows
    records = [{'name': 'A', 'module_type': 'lin_a', 'module_subtype': 'v1',
                'parameterisation': 'default'}]
    with pytest.raises(ValueError) as exc:
        component_parameterisation_rows(records, {}, source='a_only_module_array.json')
    message = str(exc.value)
    assert 'module_library_dirs' in message and '--module-library-dir' in message


def test_a_workflow_run_writes_nothing_in_the_current_directory(tmp_path, monkeypatch):
    from libcuflynx.calibration_workflow import run_calibration_workflow
    from libcuflynx.external_testing.workflow_library import build_workflow_library
    build_workflow_library(str(tmp_path / 'wf'))
    cwd = tmp_path / 'cwd'
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.delenv(paths.CUFLYNX_USER_DIR_ENV_VAR, raising=False)
    monkeypatch.setattr(paths, 'repo_root', lambda: None)
    run_calibration_workflow(str(tmp_path / 'wf' / 'arrays' / 'chain_array.json'),
                             output_dir=str(tmp_path / 'out'))
    assert os.listdir(cwd) == []


# ---- params_for_id in code -----------------------------------------------------------------

@pytest.mark.unit
def test_set_params_for_id_takes_the_file_form_too():
    """The entries of a params_for_id file, the whole document and the vessel_name/param_name
    entries give the same param_id_info."""
    parser = ObsAndParamDataParser()
    by_vessel = parser.get_param_id_info_from_entries([
        {'vessel_name': 'aortic_root', 'param_name': 'C', 'min': 1e-9, 'max': 5e-8,
         'name_for_plotting': 'C_{ao}'},
        {'vessel_name': ['venous_svc', 'venous_ivc'], 'param_name': 'R', 'min': 1e5, 'max': 1e7,
         'name_for_plotting': 'R_v'}])
    entries = [
        {'targets': ['aortic_root/C'], 'min': 1e-9, 'max': 5e-8, 'name_for_plotting': 'C_{ao}'},
        {'targets': ['venous_svc/R', 'venous_ivc/R'], 'min': 1e5, 'max': 1e7,
         'name_for_plotting': 'R_v'}]
    by_targets = parser.get_param_id_info_from_entries(entries)
    by_document = parser.get_param_id_info_from_entries({'version': 1, 'params': entries})
    for info in (by_targets, by_document):
        assert info['param_names'] == by_vessel['param_names']
        assert list(info['param_mins']) == list(by_vessel['param_mins'])
        assert list(info['param_maxs']) == list(by_vessel['param_maxs'])
        assert list(info['param_names_for_plotting']) == list(by_vessel['param_names_for_plotting'])


# ---- a name the model does not have --------------------------------------------------------

QNAMES = ['parameters.C_aortic_root', 'parameters.E_A_heart__lv', 'parameters_global.q_lv_init',
          'aortic_root_module.v', 'aortic_root_module.u', 'environment.time']


@pytest.mark.unit
@pytest.mark.parametrize('what, name, suggestion, source', [
    ('parameter', 'aortic_rot/C', 'aortic_root/C', 'params_for_id'),
    ('parameter', 'heart/lv/E_B', 'heart/lv/E_A', 'params_for_id'),
    ('parameter', 'global/q_lv_int', 'global/q_lv_init', 'params_for_id'),
    ('variable', 'aortic_root/vv', 'aortic_root/v', 'obs_data'),
])
def test_a_name_not_in_the_model_gets_the_closest_names_and_its_file(what, name, suggestion,
                                                                     source):
    from libcuflynx.solver_wrappers.name_resolver import VariableNameResolver
    message = VariableNameResolver.not_found_message(what, name, QNAMES, 'model.cellml')
    assert message.startswith(f'{what} {name} not found in the model model.cellml.')
    assert suggestion in message.split('Did you mean', 1)[1]
    assert source in message


@pytest.mark.unit
def test_the_myokit_helper_reports_a_missing_parameter_with_suggestions(
        generated_cellml_model_factory):
    from libcuflynx.solver_wrappers import get_simulation_helper
    model = generated_cellml_model_factory('3compartment')
    helper = get_simulation_helper(model_path=model, model_type='cellml', solver='CVODE_myokit',
                                   dt=0.01, sim_time=0.01, pre_time=0.0,
                                   solver_info={'MaximumStep': 0.001})
    with pytest.raises(ValueError, match='aortic_rot/C not found in the model .*Did you mean '
                                         'aortic_root/C'):
        helper.get_init_param_vals(['aortic_rot/C'])
    with pytest.raises(ValueError, match='variable aortic_root/vv not found.*aortic_root/v'):
        helper._extract('aortic_root/vv')


# ---- a missing header when Myokit compiles -------------------------------------------------

@pytest.mark.unit
def test_a_missing_header_gives_the_install_line_not_the_compiler_traceback():
    from libcuflynx.generators.CVSCellMLGenerator import myokit_compile_hint
    python_h = ('Unable to compile.\nError message:\nerror: command gcc failed\nTraceback ...\n'
                'source.c:3:10: fatal error: Python.h: No such file or directory')
    hint = myokit_compile_hint(python_h)
    assert 'python3-dev' in hint and 'Traceback' not in hint and '\n' not in hint
    sundials = 'source.c:9:10: fatal error: cvode/cvode.h: No such file or directory'
    assert 'libsundials-dev' in myokit_compile_hint(sundials)
    assert myokit_compile_hint('Simulation returned False.') is None


@pytest.mark.unit
def test_the_generated_cmake_says_how_to_give_sundials():
    template = os.path.join(SRC, 'generators', 'cpp', 'templates', 'CMakeLists.txt.j2')
    with open(template) as f:
        text = f.read()
    not_found = text.split('if(NOT SUNDIALS_INCLUDE_DIR)', 1)[1].split('endif()', 1)[0]
    assert 'FATAL_ERROR' in not_found and '-DSUNDIALS_DIR' in not_found
    assert 'libsundials-dev' in not_found


# ---- messages that said the wrong thing ----------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize('relative_path, absent', [
    ('param_id/paramID.py', 'be a snail'),
    ('param_id/optimisers.py', 'from CSV'),
    ('scripts/param_id_run_script.py', 'IF YOU WANT TO DO ANYTHING QUICKLY'),
    ('scripts/identifiability_run_script.py', 'IF YOU WANT TO DO ANYTHING QUICKLY'),
    ('scripts/run_multiple_param_id.py', 'IF YOU WANT TO DO ANYTHING QUICKLY'),
    ('scripts/plot_param_id_script.py', 'doing ia for rank'),
])
def test_messages_that_misled_are_gone(relative_path, absent):
    with open(os.path.join(SRC, relative_path)) as f:
        assert absent not in f.read()
