"""The calibrated-model writer (``param_id.calibrated_model``): values substituted into a CellML
file, flattened or as it is, with a report of where each value went. CUFLynx's calibrated model
and PhLynx send use it, as do the workflow's CellML steps."""
import os
import shutil
import warnings
from xml.etree import ElementTree as ET

import pytest

from libcuflynx.calibration_workflow.cellml_files import read_cellml_constants
from libcuflynx.param_id.calibrated_model import (flattened_cellml_text,
                                                  substitute_parameter_values,
                                                  write_calibrated_model, write_cellml_values)

TEST_INPUTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_inputs')
LV = os.path.join(TEST_INPUTS, 'Lotka_Volterra_forced.cellml')
THREE_DIR = os.path.join(TEST_INPUTS, '3compartment_unflattened')
THREE = os.path.join(THREE_DIR, '3compartment.cellml')


def _values(path):
    return {name: c['value'] for name, c in read_cellml_constants(path).items()}


@pytest.fixture(scope='module')
def flat_text():
    """3compartment flattened, as a GUI loads it: one file."""
    return flattened_cellml_text(THREE)


@pytest.mark.unit
def test_substitutes_best_fit_into_flat_model_parameters(flat_text, tmp_path):
    best = {'aortic_root/C': 9.99e-09, 'global/E_lv_A': 3.33e8, 'global/q_lv_init': 1.234e-4}
    new_text, report = substitute_parameter_values(flat_text, best)

    assert report['updated'] == list(best)
    assert report['unresolved'] == []
    assert report['resolved'] == {'aortic_root/C': 'parameters/C_aortic_root',
                                  'global/E_lv_A': 'parameters_global/E_lv_A',
                                  'global/q_lv_init': 'parameters_global/q_lv_init'}
    path = tmp_path / 'calibrated.cellml'
    path.write_text(new_text)
    values = _values(path)
    assert values['parameters/C_aortic_root'] == pytest.approx(9.99e-09)
    assert values['parameters_global/E_lv_A'] == pytest.approx(3.33e8)
    assert values['parameters_global/q_lv_init'] == pytest.approx(1.234e-4)
    assert values['parameters/R_pvn'] == pytest.approx(1333000.0)  # not given: as it was
    ET.fromstring(new_text)


@pytest.mark.unit
def test_only_the_targeted_lines_change(flat_text):
    new_text, _ = substitute_parameter_values(flat_text, {'aortic_root/C': 1.0,
                                                          'global/E_lv_A': 2.0})
    old_lines, new_lines = flat_text.splitlines(), new_text.splitlines()
    assert len(old_lines) == len(new_lines)
    assert sum(a != b for a, b in zip(old_lines, new_lines)) == 2


@pytest.mark.unit
def test_component_variable_names_resolve_to_themselves(flat_text):
    _, report = substitute_parameter_values(flat_text, {'parameters/C_aortic_root': 1.0})
    assert report['resolved'] == {'parameters/C_aortic_root': 'parameters/C_aortic_root'}


@pytest.mark.unit
def test_direct_qname_model_substitutes_too(tmp_path):
    """A model that is not generated (Lotka-Volterra): written by the vessel/param name, and
    the file keeps its form."""
    out = tmp_path / 'lv.cellml'
    report = write_cellml_values(LV, {'Lotka_Volterra_module/alpha': 7.5}, out)
    assert report['updated'] == ['Lotka_Volterra_module/alpha']
    assert report['resolved'] == {'Lotka_Volterra_module/alpha': 'Lotka_Volterra_module/alpha'}
    assert report['flattened'] is False and report['path'] == out
    assert _values(out)['Lotka_Volterra_module/alpha'] == pytest.approx(7.5)
    with open(LV) as f:
        old = f.read().splitlines()
    new = out.read_text().splitlines()
    assert len(old) == len(new) and sum(a != b for a, b in zip(old, new)) == 1


@pytest.mark.unit
def test_unresolved_params_are_reported_not_dropped(flat_text):
    new_text, report = substitute_parameter_values(flat_text, {'nonexistent_vessel/nope': 1.0})
    assert report == {'updated': [], 'unresolved': ['nonexistent_vessel/nope'], 'resolved': {}}
    assert new_text == flat_text


@pytest.mark.unit
def test_a_model_with_imports_is_flattened_by_default(tmp_path):
    """A generated model imports its parameters: written as it is, none is found; flattened
    (the default for a file with imports), each is, and the file runs by itself."""
    model_dir = tmp_path / 'model'
    shutil.copytree(THREE_DIR, model_dir)
    best = {'aortic_root/C': 2.0e-8, 'global/E_lv_A': 4.0e8}

    kept = write_cellml_values(model_dir / '3compartment.cellml', best, tmp_path / 'kept.cellml',
                               flatten=False)
    assert kept['unresolved'] == list(best) and kept['flattened'] is False
    assert '<import' in (tmp_path / 'kept.cellml').read_text()

    out = tmp_path / 'elsewhere' / 'flat.cellml'
    flat = write_cellml_values(model_dir / '3compartment.cellml', best, out)
    assert flat['flattened'] is True and flat['unresolved'] == []
    assert '<import' not in out.read_text()
    values = _values(out)
    assert values['parameters/C_aortic_root'] == pytest.approx(2.0e-8)
    assert values['parameters_global/E_lv_A'] == pytest.approx(4.0e8)


@pytest.mark.unit
def test_an_import_does_not_hide_the_components_after_it(tmp_path):
    """An import's ``<component .../>`` has no body: the component defined after it is still
    written."""
    text = ('<model xmlns="http://www.cellml.org/cellml/1.1#" '
            'xmlns:xlink="http://www.w3.org/1999/xlink" name="m">'
            '<import xlink:href="other.cellml"><component name="a" component_ref="a"/></import>'
            '<component name="b"><variable name="k" units="dimensionless" initial_value="1"/>'
            '</component></model>')
    new_text, report = substitute_parameter_values(text, {'b/k': 3.0})
    assert report['resolved'] == {'b/k': 'b/k'}
    assert 'initial_value="3.0"' in new_text


@pytest.mark.unit
def test_write_calibrated_model_reports_and_warns(tmp_path):
    info = {'param_names': [['Lotka_Volterra_module/alpha'], ['Lotka_Volterra_module/nope']]}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        report = write_calibrated_model(LV, info, [1.25, 4.5], tmp_path / 'c.cellml',
                                        return_report=True)
    assert report['updated'] == ['Lotka_Volterra_module/alpha']
    assert report['unresolved'] == ['Lotka_Volterra_module/nope']
    assert any('Lotka_Volterra_module/nope' in str(w.message) for w in caught)


@pytest.mark.unit
def test_the_workflow_writes_a_single_file_model_in_its_own_form(tmp_path):
    """A workflow's CellML steps (generate_cellml, write_calibrated_cellml) write through the
    same writer: a model with no imports keeps its form."""
    from libcuflynx.calibration_workflow.cellml_files import write_cellml_with_values
    out = tmp_path / 'step.cellml'
    write_cellml_with_values(LV, {'Lotka_Volterra_module/alpha': 4.5,
                                  'Lotka_Volterra_module/gamma': 2.25}, str(out))
    with open(LV) as f:
        old = f.read().splitlines()
    new = out.read_text().splitlines()
    assert len(old) == len(new) and sum(a != b for a, b in zip(old, new)) == 2
    assert _values(out)['Lotka_Volterra_module/gamma'] == 2.25
