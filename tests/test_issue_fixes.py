"""Regression tests for a batch of small closed issues:

* #83  -- macOS AppleDouble ``._*`` files must be skipped by the module readers.
* #99  -- ``*_parameters_unfinished.csv`` must be written to the configured ``resources_dir``.
* #155 -- the duplicated input-flow BC modules were removed from the microvasculature config.
* #157 -- solver Make files are copied into each generated model directory.
* #159 -- parameter CSV columns are read by header name, not by position.
* #398 -- solver_info rtol/atol reach the generated C++ instead of being hardcoded.
* #167 -- sensitivity-analysis plot filenames are sanitised (no ``{}``/spaces/backslashes).

These are deliberately light-weight: heavy optional deps are imported inside the test bodies so a
missing analysis stack cannot break collection of the whole file.
"""
import json
import os

import pytest


# ---------------------------------------------------------------------------
# #83 -- skip macOS AppleDouble '._*.json' sidecar files
# ---------------------------------------------------------------------------
def test_is_json_module_file_skips_appledouble():
    from libcuflynx.parsers.PrimitiveParsers import JSONFileParser
    assert JSONFileParser._is_json_module_file('boundary_condition_modules_config.json')
    assert not JSONFileParser._is_json_module_file('._boundary_condition_modules_config.json')
    assert not JSONFileParser._is_json_module_file('notes.txt')


def test_json_reader_ignores_appledouble_sidecar(tmp_path):
    from libcuflynx.parsers.PrimitiveParsers import JSONFileParser
    good = tmp_path / 'good'
    empty = tmp_path / 'empty'
    good.mkdir()
    empty.mkdir()
    (good / 'module.json').write_text(json.dumps([{'vessel_type': 'x', 'BC_type': 'y'}]))
    # An AppleDouble sidecar is binary; if it were read as JSON it would raise.
    (good / '._module.json').write_bytes(b'\x00\x05\x16\x07Mac OS X\x00binary junk')

    df = JSONFileParser().json_to_dataframe_with_user_dir(str(good), str(empty), None)
    assert list(df['vessel_type']) == ['x']


# ---------------------------------------------------------------------------
# #99 -- unfinished parameters CSV goes to the configured resources_dir
# ---------------------------------------------------------------------------
class _StubModel:
    def __init__(self):
        # minimum shape __generate_parameters_csv needs: a DataFrame-able array with const_type.
        self.parameters_array = [
            {'variable_name': 'a', 'const_type': 'global', 'value': 'EMPTY_MUST_BE_FILLED'},
            {'variable_name': 'b', 'const_type': 'global', 'value': '1.0'},
        ]


def _bare_generator(**attrs):
    """A CVS0DCellMLGenerator with only the attributes a single method needs (skips __init__,
    which would build a whole model)."""
    from libcuflynx.generators.CVSCellMLGenerator import CVS0DCellMLGenerator
    gen = CVS0DCellMLGenerator.__new__(CVS0DCellMLGenerator)
    for k, v in attrs.items():
        setattr(gen, k, v)
    return gen


def test_unfinished_parameters_csv_written_to_resources_dir(tmp_path):
    resources = tmp_path / 'my_resources'
    output = tmp_path / 'generated'
    resources.mkdir()
    output.mkdir()
    gen = _bare_generator(all_parameters_defined=False, resources_dir=str(resources),
                          output_dir=str(output), file_prefix='demo', model=_StubModel())

    gen._CVS0DCellMLGenerator__generate_parameters_csv()

    assert (resources / 'demo_parameters_unfinished.csv').is_file(), \
        'unfinished CSV must land in the configured resources_dir (#99)'
    assert not (output / 'demo_parameters_unfinished.csv').exists()


def test_finished_parameters_csv_written_to_output_dir(tmp_path):
    resources = tmp_path / 'my_resources'
    output = tmp_path / 'generated'
    resources.mkdir()
    output.mkdir()
    gen = _bare_generator(all_parameters_defined=True, resources_dir=str(resources),
                          output_dir=str(output), file_prefix='demo', model=_StubModel())

    gen._CVS0DCellMLGenerator__generate_parameters_csv()

    assert (output / 'demo_parameters.csv').is_file()


# ---------------------------------------------------------------------------
# #157 -- solver Make files are copied into the generated model directory
# ---------------------------------------------------------------------------
def test_solver_make_files_copied_into_model_dir(tmp_path):
    from libcuflynx.generators.CVSCellMLGenerator import solver_make_files_dir
    if not os.path.isdir(solver_make_files_dir):
        pytest.skip('solver Make_files directory not present in this checkout')
    expected = [f for f in os.listdir(solver_make_files_dir)
                if os.path.isfile(os.path.join(solver_make_files_dir, f))
                and not f.startswith('._')]
    assert expected, 'expected some Make files to copy'

    output = tmp_path / 'generated'
    output.mkdir()
    gen = _bare_generator(output_dir=str(output))
    gen._CVS0DCellMLGenerator__copy_solver_make_files()

    for f in expected:
        assert (output / f).is_file(), f'{f} should have been copied into the model dir (#157)'


# ---------------------------------------------------------------------------
# #155 -- duplicated input-flow BC modules removed from the microvasculature config
# ---------------------------------------------------------------------------
def test_microvasculature_config_has_no_duplicate_input_flow_modules():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    micro = os.path.join(root, 'module_config_user',
                         'microvasculature_network_module_config.json')
    with open(micro) as rf:
        entries = json.load(rf)
    vessel_types = {e.get('vessel_type') for e in entries}
    # the duplicated input BC modules are gone (use the boundary_condition modules instead, #155)
    assert 'P_inlet' not in vessel_types
    assert 'Q_inlet' not in vessel_types

    bc = os.path.join(root, 'src', 'libcuflynx', 'generators', 'resources',
                      'boundary_condition_modules_config.json')
    with open(bc) as rf:
        bc_entries = json.load(rf)
    bc_module_types = {e.get('module_type') for e in bc_entries}
    # the canonical replacements still exist
    assert {'constant_flow_BC_type', 'constant_pressure_BC_type'} <= bc_module_types


# ---------------------------------------------------------------------------
# #167 -- sanitise sensitivity-analysis plot filenames
# ---------------------------------------------------------------------------
def test_sanitize_for_filename_strips_unsafe_characters():
    from libcuflynx.sensitivity_analysis.sobolSA import sanitize_for_filename
    raw = r"u_{A_{R}} - experiment0, subexperiment0"
    safe = sanitize_for_filename(raw)
    for bad in '{}\\/ ,':
        assert bad not in safe, f'{bad!r} should not survive sanitisation'
    assert safe  # non-empty
    # a name that is nothing but unsafe characters still yields a usable stem
    assert sanitize_for_filename('***') == 'output'
    # already-safe names are preserved (dots and dashes are allowed)
    assert sanitize_for_filename('flow_rate-1.0') == 'flow_rate-1.0'


# ---------------------------------------------------------------------------
# #159 -- parameter CSV columns are matched by header, not by position
# ---------------------------------------------------------------------------
def _reduce(tmp_path, header, rows, variables_and_units=None):
    """Run __reduce_parameters_array over a hand-written parameters CSV.

    Builds the parser with __new__ so only the two attributes this method needs are set -- the
    real __init__ would parse a whole model.
    """
    import pandas as pd
    from libcuflynx.parsers.ModelParsers import CSV0DModelParser
    from libcuflynx.parsers.PrimitiveParsers import CSVFileParser

    csv_path = tmp_path / 'x_parameters.csv'
    csv_path.write_text('\n'.join([header] + rows) + '\n')

    if variables_and_units is None:
        # one required constant, 'R', which gets the vessel name appended -> 'R_vessel'
        variables_and_units = [['R', 'Js_per_m6', 'access', 'constant']]
    vessels_df = pd.DataFrame([{'name': 'vessel',
                                'variables_and_units': variables_and_units}])

    parser = CSV0DModelParser.__new__(CSV0DModelParser)
    parser.parameter_filename = str(csv_path)
    parser.csv_parser = CSVFileParser()

    parameters_array_orig = parser.csv_parser.get_data_as_nparray(str(csv_path), True)
    return parser._CSV0DModelParser__reduce_parameters_array(
        parameters_array_orig, vessels_df)


def test_parameters_csv_extra_column_does_not_shift_the_value(tmp_path):
    """The bug: rows were flattened into a positional list, so any column the code did not expect
    shifted every field to its right.

    resources/FTU_wCVS_parameters.csv is a real file with a 'comp_env' column between units and
    value; before the fix its parameter values parsed as the comp_env string ('heart') and its
    data_references as the values.
    """
    out = _reduce(tmp_path,
                  'variable_name,units,comp_env,value,data_reference',
                  ['R_vessel,Js_per_m6,heart,1333000,Blanco_2013'])
    assert out['value'][0] == '1333000', 'the comp_env column was read as the value'
    assert out['data_reference'][0] == 'Blanco_2013'
    assert out['units'][0] == 'Js_per_m6'
    assert out['const_type'][0] == 'constant'


def test_parameters_csv_columns_may_be_in_any_order(tmp_path):
    """Header-based reading means column order is not part of the file format."""
    reordered = _reduce(tmp_path,
                        'data_reference,value,variable_name,units',
                        ['Blanco_2013,1333000,R_vessel,Js_per_m6'])
    canonical = _reduce(tmp_path,
                        'variable_name,units,value,data_reference',
                        ['R_vessel,Js_per_m6,1333000,Blanco_2013'])
    for field in ('variable_name', 'units', 'const_type', 'value', 'data_reference'):
        assert reordered[field][0] == canonical[field][0], field


def test_parameters_csv_header_whitespace_is_tolerated(tmp_path):
    """A space after a comma in the header row must not make a column unfindable under the name
    the user can see in their file."""
    out = _reduce(tmp_path,
                  'variable_name, units, value, data_reference',
                  ['R_vessel,Js_per_m6,1333000,Blanco_2013'])
    assert out['value'][0] == '1333000'
    assert out['units'][0] == 'Js_per_m6'


def test_parameters_csv_missing_required_column_is_reported(tmp_path):
    """Positional reading turned a missing column into a silent EMPTY_MUST_BE_FILLED; by name it
    is a named error."""
    with pytest.raises(SystemExit):
        _reduce(tmp_path,
                'variable_name,units,value',
                ['R_vessel,Js_per_m6,1333000'])


def test_parameters_csv_missing_row_still_flags_it_as_unfilled(tmp_path):
    """Unchanged behaviour: a required parameter with no row in the CSV is carried through as
    EMPTY_MUST_BE_FILLED so the unfinished-parameters file can list it."""
    out = _reduce(tmp_path,
                  'variable_name,units,value,data_reference',
                  ['something_else,Js_per_m6,1333000,Blanco_2013'])
    assert out['variable_name'][0] == 'R_vessel'
    assert out['value'][0] == 'EMPTY_MUST_BE_FILLED'


def test_parameters_csv_unit_mismatch_still_exits(tmp_path):
    """Unchanged behaviour: units that disagree with the module config are a hard error."""
    with pytest.raises(SystemExit):
        _reduce(tmp_path,
                'variable_name,units,value,data_reference',
                ['R_vessel,m3,1333000,Blanco_2013'])


# ---------------------------------------------------------------------------
# #398 -- solver_info rtol/atol reach the generated C++
# ---------------------------------------------------------------------------
def _cpp_solver_init(solver, reltol=1e-7, abstol=1e-9):
    """The emitted set_ode_solver body, built without generating a whole model."""
    from libcuflynx.generators.CVSCppGenerator import CVS0DCppGenerator

    gen = CVS0DCppGenerator.__new__(CVS0DCppGenerator)
    gen.solver = solver
    gen.reltol = reltol
    gen.abstol = abstol
    gen.dtSolver = 1e-4
    gen.nMaxSteps = 5000
    return gen._build_solver_init_function()


@pytest.mark.parametrize('solver', ['CVODE', 'PETSC'])
def test_cpp_generator_emits_the_configured_tolerances(solver):
    """The two tolerance literals in the emitted C++ carried a standing
    'TODO get this from user_inputs.yaml too'; a cpp user could only change the accuracy of a run
    by hand-editing generated code."""
    emitted = _cpp_solver_init(solver, reltol=1.5e-9, abstol=2.5e-11)
    assert 'reltol = 1.5e-09' in emitted, emitted
    assert 'abstol = 2.5e-11' in emitted, emitted
    # the values that used to be baked in are gone unless the user asks for them
    assert 'reltol = 1e-7' not in emitted
    assert 'abstol = 1e-9' not in emitted
    assert 'TODO get this from user_inputs' not in emitted


@pytest.mark.parametrize('solver', ['CVODE', 'PETSC'])
def test_cpp_generator_defaults_reproduce_the_previous_output(solver):
    """A config that sets neither tolerance must generate exactly what it did before, so wiring
    the settings up does not silently change anybody's existing model."""
    emitted = _cpp_solver_init(solver)
    assert 'reltol = 1e-07' in emitted, emitted
    assert 'abstol = 1e-09' in emitted, emitted


def test_cpp_rk4_has_no_tolerances_to_configure():
    """RK4 is a fixed-step scheme with no tolerance knobs -- it must not grow one just because
    the settings now exist."""
    emitted = _cpp_solver_init('RK4')
    assert 'reltol' not in emitted
    assert 'abstol' not in emitted
    assert 'wRK4' in emitted, 'expected the RK4 branch'


def test_generate_script_reads_tolerances_from_solver_info():
    """The generator only gets the user's values if the cpp branch of the generate script passes
    them, and the defaults there are what keeps existing configs byte-identical."""
    import inspect
    from libcuflynx.scripts import script_generate_with_new_architecture as gen_script

    source = inspect.getsource(gen_script.generate_with_new_architecture)
    assert "solver_info.get('rtol', 1e-7)" in source
    assert "solver_info.get('atol', 1e-9)" in source
    # both CVS0DCppGenerator constructions (coupled and uncoupled) must forward them
    assert source.count('reltol=reltol, abstol=abstol') == 2
