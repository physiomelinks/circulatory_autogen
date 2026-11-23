"""Bad input to the parsers and generators raises CAInputError with its message (#706).

A caller running libcuflynx in-process (a notebook, the CUFLynx server, a calibration
workflow step) catches these with ``except Exception`` and can show the message;
``exit()`` raised ``SystemExit``, which passes through that and printed the message only.
"""
import ast
import os
import shutil
from types import SimpleNamespace

import pandas as pd
import pytest

from libcuflynx.errors import CAInputError
from libcuflynx.parsers.PrimitiveParsers import ObsDataError, YamlFileParser

pytestmark = pytest.mark.unit

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'libcuflynx')
RESOURCES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resources')

# Modules a caller runs in-process; none of them may exit the interpreter.
NO_EXIT_MODULES = [
    'parsers/PrimitiveParsers.py',
    'parsers/ModelParsers.py',
    'generators/CVSCellMLGenerator.py',
    'generators/Python1DModelFilesGenerator.py',
]


def _exit_calls(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in ('exit', 'quit'):
            yield node.lineno
        elif (isinstance(func, ast.Attribute) and func.attr == 'exit'
              and isinstance(func.value, ast.Name) and func.value.id in ('sys', 'os')):
            yield node.lineno


@pytest.mark.parametrize('module', NO_EXIT_MODULES)
def test_the_module_never_exits(module):
    with open(os.path.join(SRC, module)) as f:
        tree = ast.parse(f.read())
    assert list(_exit_calls(tree)) == [], f'{module} calls exit(); raise CAInputError instead'


def test_generate_with_new_architecture_never_exits():
    """Its main() keeps the exit status; the function a caller runs in-process does not exit."""
    path = os.path.join(SRC, 'scripts', 'script_generate_with_new_architecture.py')
    with open(path) as f:
        tree = ast.parse(f.read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == 'generate_with_new_architecture')
    assert list(_exit_calls(fn)) == []


def test_input_errors_are_ordinary_exceptions():
    assert issubclass(CAInputError, ValueError)
    assert not issubclass(CAInputError, SystemExit)
    assert issubclass(ObsDataError, CAInputError)


# --- the user_inputs parser -----------------------------------------------------------------


def _inputs(tmp_path, **overrides):
    resources = tmp_path / 'resources'
    resources.mkdir(exist_ok=True)
    for suffix in ('_module_array.csv', '_parameters.json'):
        shutil.copy(os.path.join(RESOURCES, '3compartment' + suffix), resources)
    inp = {'file_prefix': '3compartment', 'input_param_file': '3compartment_parameters.json',
           'model_type': 'cellml', 'solver': 'CVODE_myokit', 'resources_dir': str(resources),
           'generated_models_dir': str(tmp_path / 'generated'),
           'dt': 0.01, 'pre_time': 0.0, 'sim_time': 1.0}
    for key, value in overrides.items():
        if value is None:
            inp.pop(key, None)
        else:
            inp[key] = value(tmp_path) if callable(value) else value
    return inp


BAD_USER_INPUTS = {
    'no file_prefix': ({'file_prefix': None}, 'file_prefix'),
    'unknown model_type': ({'model_type': 'fortran', 'solver': None},
                           'Invalid model.type: fortran'),
    'unknown solver': ({'solver': 'NOT_A_SOLVER'}, 'Invalid solver: NOT_A_SOLVER'),
    'dt not a float': ({'dt': 1}, 'dt must be a float'),
    'python model, CellML solver': ({'model_type': 'python'},
                                    'cannot be used with Python models'),
    'casadi model, CellML solver': ({'model_type': 'casadi_python'},
                                    'cannot be used with CasADi Python models'),
    'aadc model, CellML solver': ({'model_type': 'aadc_python'},
                                  'cannot be used with AADC Python models'),
    'external model, CellML solver': ({'model_type': 'external_python'},
                                      'cannot be used with external Python models'),
    'external solver, CellML model': ({'solver': 'external'},
                                      'requires model_type to be "external_python"'),
    'CellML solver, unknown method': ({'solver_info': {'method': 'nope'}},
                                      'solver method nope not compatible with solver CVODE_myokit'),
    'cpp model, python solver': ({'model_type': 'cpp', 'solver': 'solve_ivp'},
                                 'solver solve_ivp not compatible with model_type cpp'),
    'solver_info key of another solver': ({'solver_info': {'not_a_key': 1}},
                                          "not supported by solver 'CVODE_myokit'"),
    'sp_minimize without a gradient': ({'param_id_method': 'sp_minimize', 'do_ad': False},
                                       'sp_minimize requires model_type'),
}

# A path that does not exist may also be refused as a FileNotFoundError: still an ordinary
# exception with its message, which is what a caller needs.
MISSING_PATH_ERRORS = (CAInputError, FileNotFoundError)

MISSING_PATHS = {
    'external_modules_dir missing': ({'external_modules_dir': lambda p: str(p / 'no_ext')},
                                     'external_modules_dir=.*no_ext does not exist'),
    'module library missing': ({'module_library_dirs': lambda p: [str(p / 'no_lib')]},
                               'module_library_dirs entry .*no_lib does not exist'),
}


@pytest.mark.parametrize('overrides, message', list(BAD_USER_INPUTS.values()),
                         ids=list(BAD_USER_INPUTS))
def test_a_bad_user_inputs_config_raises_with_its_message(tmp_path, overrides, message):
    with pytest.raises(CAInputError, match=message):
        YamlFileParser().parse_user_inputs_file(_inputs(tmp_path, **overrides),
                                                obs_path_needed=False)


@pytest.mark.parametrize('overrides, message', list(MISSING_PATHS.values()),
                         ids=list(MISSING_PATHS))
def test_a_missing_path_raises_with_its_message(tmp_path, overrides, message):
    with pytest.raises(MISSING_PATH_ERRORS, match=message):
        YamlFileParser().parse_user_inputs_file(_inputs(tmp_path, **overrides),
                                                obs_path_needed=False)


def test_a_missing_obs_data_file_raises(tmp_path):
    inp = _inputs(tmp_path, param_id_obs_path=str(tmp_path / 'no_obs.json'))
    with pytest.raises(MISSING_PATH_ERRORS, match='param_id_obs_path=.*no_obs.json does not exist'):
        YamlFileParser().parse_user_inputs_file(inp, obs_path_needed=True)


# --- parameters files -------------------------------------------------------------------------


def _reduce(header, rows):
    from libcuflynx.parsers.ModelParsers import CSV0DModelParser
    from libcuflynx.parsers.PrimitiveParsers import CSVFileParser
    from libcuflynx.utilities.parameters_file import parameters_array

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from _parameters import rows_from_csv
    finally:
        sys.path.pop(0)
    vessels_df = pd.DataFrame([{'name': 'vessel',
                                'variables_and_units': [['R', 'Js_per_m6', 'access', 'constant']]}])
    parser = CSV0DModelParser.__new__(CSV0DModelParser)
    parser.parameter_filename = 'x_parameters.json'
    parser.csv_parser = CSVFileParser()
    return parser._CSV0DModelParser__reduce_parameters_array(
        parameters_array(rows_from_csv('\n'.join([header] + rows))), vessels_df)


def test_a_parameters_unit_mismatch_raises():
    with pytest.raises(CAInputError, match='units of m3 in the parameters file does not'):
        _reduce('variable_name,units,value,data_reference', ['R_vessel,m3,1,ref'])


# --- generators -------------------------------------------------------------------------------


def test_modules_without_a_matching_port_raise():
    from libcuflynx.generators.CVSCellMLGenerator import CVS0DCellMLGenerator
    check = CVS0DCellMLGenerator._CVS0DCellMLGenerator__check_input_output_ports
    with pytest.raises(CAInputError, match=r'upstream module a and downstream module b do not '
                                           r'have a matching port(.|\n)*exit ports: \[.flow.\]'):
        check(None, [{'port_type': 'flow'}], [{'port_type': 'signal'}], [{'port_type': 'pressure'}], [], 'a', 'b')


BAD_1D_PARAMETERS = {
    'type undetermined': ([], "Artery/vein type couldn't be determined for 1D vessel seg"),
    'type not 0 or 1': ([('art_ven_type_seg', 2)], 'art_ven_type_seg must be 1 .* got 2'),
    'no length': ([('art_ven_type_seg', 1)], 'Parameter l_seg not found'),
    'one radius only': ([('art_ven_type_seg', 1), ('l_seg', 0.1), ('rProx_0_seg', 0.01)],
                        'found in parameters dataframe, but not parameter rDist_0_seg'),
}


@pytest.mark.parametrize('params, message', list(BAD_1D_PARAMETERS.values()),
                         ids=list(BAD_1D_PARAMETERS))
def test_bad_1d_vessel_parameters_raise(tmp_path, params, message):
    from libcuflynx.generators.Python1DModelFilesGenerator import generate1DPythonModelFiles
    df_vess = pd.DataFrame({'name': ['seg']})
    df_params = pd.DataFrame({'variable_name': [p[0] for p in params] or pd.Series(dtype=str),
                              'value': [p[1] for p in params] or pd.Series(dtype=float)})
    with pytest.raises(CAInputError, match=message):
        generate1DPythonModelFiles(df_vess, df_params, str(tmp_path / 'vess.txt'),
                                   str(tmp_path / 'nodes.txt'), str(tmp_path / 'names.csv'), {})


# --- the calibration workflow -----------------------------------------------------------------


def test_a_workflow_step_reports_the_input_error():
    from libcuflynx.calibration_workflow.runner import WorkflowStepError, _Run

    def refuse():
        raise CAInputError('Invalid solver: NOT_A_SOLVER')

    with pytest.raises(WorkflowStepError, match='step "s1": the param_id configuration was '
                                                'refused: Invalid solver: NOT_A_SOLVER'):
        _Run._run_stage(None, SimpleNamespace(id='s1'), 'param_id', refuse)
