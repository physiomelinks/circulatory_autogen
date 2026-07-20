"""Regression tests for a batch of small closed issues:

* #83  -- macOS AppleDouble ``._*`` files must be skipped by the module readers.
* #99  -- ``*_parameters_unfinished.csv`` must be written to the configured ``resources_dir``.
* #155 -- the duplicated input-flow BC modules were removed from the microvasculature config.
* #157 -- solver Make files are copied into each generated model directory.
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
    from parsers.PrimitiveParsers import JSONFileParser
    assert JSONFileParser._is_json_module_file('boundary_condition_modules_config.json')
    assert not JSONFileParser._is_json_module_file('._boundary_condition_modules_config.json')
    assert not JSONFileParser._is_json_module_file('notes.txt')


def test_json_reader_ignores_appledouble_sidecar(tmp_path):
    from parsers.PrimitiveParsers import JSONFileParser
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
    from generators.CVSCellMLGenerator import CVS0DCellMLGenerator
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
    from generators.CVSCellMLGenerator import solver_make_files_dir
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

    bc = os.path.join(root, 'src', 'generators', 'resources',
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
    from sensitivity_analysis.sobolSA import sanitize_for_filename
    raw = r"u_{A_{R}} - experiment0, subexperiment0"
    safe = sanitize_for_filename(raw)
    for bad in '{}\\/ ,':
        assert bad not in safe, f'{bad!r} should not survive sanitisation'
    assert safe  # non-empty
    # a name that is nothing but unsafe characters still yields a usable stem
    assert sanitize_for_filename('***') == 'output'
    # already-safe names are preserved (dots and dashes are allowed)
    assert sanitize_for_filename('flow_rate-1.0') == 'flow_rate-1.0'
