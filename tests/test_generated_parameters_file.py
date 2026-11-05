"""Every generated model has one parameters file beside it, ``<prefix>_parameters.json``, and no
parameters CSV: every value it uses, in the form of a model's parameters file
(utilities/parameters_file.py), a value the model's own file does not set naming the
parameterisation it came from at the start of its data_reference (``from <parameterisation>: <its
own data_reference>``). Generating again from that file alone gives the same model, the readers
of a generated model's parameters read it (``parameters_file.model_parameter_rows``), and a
calibration workflow's calibrated parameters are built on it, so a result reproduces the run
without the module library.

The supermodule model is the module library's 3compartment system model (its heart a
supermodule instance), run against the library CI pins (``tests/MODULE_LIBRARY_REF``; skipped
without it). The host-only model is ``resources/3compartment`` on the built-in modules.
"""
import json
import os
import re
import shutil

import numpy as np
import pytest

import _module_library as module_library
from _parameters import read_parameters, write_parameters
from test_config_schemas import _assert_same_generated_models

from libcuflynx.calibration_workflow import (WorkflowError, run_calibration_workflow,
                                             write_calibrated_parameters_to)
from libcuflynx.external_testing.workflow_library import (K_SOURCE, Y_A, Z_C,
                                                          build_workflow_library)
from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
from libcuflynx.utilities import parameters_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES_DIR = os.path.join(REPO_ROOT, 'resources')
PREFIX = 'm'


def _generate(work_dir, module_array, parameters, library_dirs=None):
    """Generate ``module_array`` (a path) with the parameters document or file ``parameters`` as
    the model's parameters file; returns the generated model's directory."""
    resources = os.path.join(str(work_dir), 'resources')
    os.makedirs(resources, exist_ok=True)
    extension = os.path.splitext(module_array)[1]
    shutil.copy(module_array, os.path.join(resources, f'{PREFIX}_module_array{extension}'))
    target = os.path.join(resources, parameters_file.parameters_file_name(PREFIX))
    if isinstance(parameters, dict):
        with open(target, 'w') as f:
            json.dump(parameters, f, indent=1)
    else:
        shutil.copy(parameters, target)
    config = {'file_prefix': PREFIX, 'input_param_file': os.path.basename(target),
              'model_type': 'cellml', 'solver': 'CVODE_myokit', 'resources_dir': resources,
              'generated_models_dir': os.path.join(str(work_dir), 'generated_models'),
              'DEBUG': False}
    if library_dirs:
        config.update(module_library_dirs=list(library_dirs), use_builtin_modules=False)
    assert generate_with_new_architecture(False, config), 'generation failed'
    return os.path.join(str(work_dir), 'generated_models', PREFIX)


def _generated_file(model_dir):
    """The parameters file generation writes beside a generated model."""
    return os.path.join(model_dir, f'{PREFIX}_parameters.json')


_CELLML_VARIABLE = re.compile(r'<variable initial_value="([^"]*)" name="([^"]*)"[^>]* '
                              r'units="([^"]*)"/>')


def _cellml_constants(model_dir):
    """``{model name: (value, units)}`` of the generated model's parameters CellML."""
    with open(os.path.join(model_dir, f'{PREFIX}_parameters.cellml')) as f:
        return {name: (float(value), units)
                for value, name, units in _CELLML_VARIABLE.findall(f.read())}


def _assert_every_value_generated_file(model_dir):
    """The generated parameters file, the only one beside the model, holds exactly the
    generated model's constants, each with its value and units. Returns its rows by model
    name, as ``model_parameter_rows`` reads them."""
    assert sorted(f for f in os.listdir(model_dir) if '_parameters' in f) == \
        [f'{PREFIX}_parameters.cellml', f'{PREFIX}_parameters.json']
    used = {r['variable_name']: r
            for r in parameters_file.model_parameter_rows(_generated_file(model_dir))}
    constants = _cellml_constants(model_dir)
    assert constants and sorted(used) == sorted(constants)
    for name, (value, units) in constants.items():
        assert (float(used[name]['value']), used[name]['units']) == (value, units), name
    return used


@pytest.fixture(scope='module')
def three_compartment():
    """The library's modules dir and its 3compartment system model's module array and
    parameters file."""
    modules = str(module_library.modules_dir_or_skip())
    model = os.path.join(os.path.dirname(modules), 'system_models', 'closed_loop_cvs',
                         '3compartment')
    if not os.path.isdir(model):
        pytest.skip(f'the module library at {modules} has no 3compartment system model')
    return modules, os.path.join(model, '3compartment_module_array.json'), \
        os.path.join(model, '3compartment_parameters.json')


def _empty_heart(three_compartment):
    _, _, parameters = three_compartment
    with open(parameters) as f:
        doc = json.load(f)
    assert 'heart' in doc
    doc['heart'] = {}
    return doc


@pytest.mark.integration
def test_an_empty_supermodule_instance_generates_and_its_generated_parameters_file_has_every_value(
        tmp_path, three_compartment):
    """``"heart": {}``: every heart value comes from its submodules' version defaults, and the
    generated parameters file has each of them under heart, naming the parameterisation it came
    from."""
    modules, array, _ = three_compartment
    model_dir = _generate(tmp_path, array, _empty_heart(three_compartment), [modules])
    used = _assert_every_value_generated_file(model_dir)
    heart = {n: r for n, r in used.items() if n.endswith(tuple(
        f'_heart__{s}' for s in ('ra', 'rv', 'la', 'lv', 'trv', 'puv', 'miv', 'aov')))}
    assert len(heart) > 20
    # a data_reference is kept whole, however long
    assert max(len(r['data_reference']) for r in heart.values()) > 80
    for name, row in heart.items():
        assert row['data_reference'].startswith('from ') and \
            ' parameterisation ' in row['data_reference'], name
    doc = parameters_file.read_parameters_file(_generated_file(model_dir))
    e_a = doc['heart']['lv']['E_A']
    assert e_a['data_reference'].startswith('from chamber/')
    assert ' parameterisation lv: liang2009multi; ' in e_a['data_reference']
    # no key but a parameter's own, and the host's own values name no source
    assert all(set(leaf) <= {'value', 'units', 'data_reference', 'origin', 'reason'}
               for _, leaf in parameters_file.leaves(doc))
    assert not any(r['data_reference'].startswith('from ')
                   for n, r in used.items() if n not in heart)


@pytest.mark.integration
def test_a_host_only_model_s_generated_parameters_file_has_every_value(tmp_path):
    model_dir = _generate(tmp_path, os.path.join(RESOURCES_DIR, '3compartment_module_array.csv'),
                          os.path.join(RESOURCES_DIR, '3compartment_parameters.json'))
    used = _assert_every_value_generated_file(model_dir)
    assert 'T' in used and not any(r['data_reference'].startswith('from ')
                                   for r in used.values())


@pytest.mark.integration
def test_a_values_origin_and_reason_reach_the_generated_parameters_file(tmp_path):
    with open(os.path.join(RESOURCES_DIR, '3compartment_parameters.json')) as f:
        doc = json.load(f)
    doc['global']['T'].update(origin='set_by_hand', reason='a one second cycle')
    model_dir = _generate(tmp_path, os.path.join(RESOURCES_DIR, '3compartment_module_array.csv'),
                          doc)
    generated = parameters_file.read_parameters_file(_generated_file(model_dir))
    assert generated['global']['T']['origin'] == 'set_by_hand'
    assert generated['global']['T']['reason'] == 'a one second cycle'
    others = [leaf for path, leaf in parameters_file.leaves(generated) if path != ['global', 'T']]
    assert others and not any('origin' in leaf or 'reason' in leaf for leaf in others)


@pytest.mark.unit
def test_a_value_set_by_hand_must_give_its_reason(tmp_path):
    path = tmp_path / 'm_parameters.json'
    leaf = {'value': 1, 'units': 'second', 'origin': 'set_by_hand'}
    path.write_text(json.dumps({'schema_version': 1, 'global': {'T': leaf}}))
    with pytest.raises(ValueError, match='reason'):
        parameters_file.read_parameters_file(str(path))
    leaf['origin'] = 'guessed'
    path.write_text(json.dumps({'schema_version': 1, 'global': {'T': leaf}}))
    with pytest.raises(ValueError, match='origin'):
        parameters_file.read_parameters_file(str(path))


@pytest.mark.unit
def test_a_value_identified_by_parameter_id_is_marked_calibrated():
    """Generation with fit parameters (cuflynx-generate True) writes an identified value with
    origin calibrated, so a derived or hand-set origin never stays on a fitted value."""
    from types import SimpleNamespace
    from libcuflynx.generators.CVSCellMLGenerator import CVS0DCellMLGenerator
    fields = ('variable_name', 'units', 'const_type', 'value', 'data_reference', 'origin',
              'reason')
    rows = [('R_T0_Systemic_T', 'J_s_per_m6', 'constant', '1.5e8', 'derived', 'derived', ''),
            ('T', 'second', 'global_constant', '1.0', 't', 'set_by_hand', 'a one second cycle'),
            ('C_T', 'm3_per_Pa', 'constant', '1e-8', 'c', 'measured', '')]
    array = np.array(rows, dtype=[(f, 'U80') for f in fields])
    generator = object.__new__(CVS0DCellMLGenerator)
    generator.model = SimpleNamespace(parameters_array=array, param_id_date='261106',
                                      param_id_name_and_vals=[('R_T0_Systemic_T', 2e8),
                                                              ('T', 0.9)])
    generator._CVS0DCellMLGenerator__modify_parameters_array_from_param_id()
    by_name = {r['variable_name']: r for r in array}
    for name, value in (('R_T0_Systemic_T', 2e8), ('T', 0.9)):
        row = by_name[name]
        assert (float(row['value']), row['origin'], row['reason']) == (value, 'calibrated', '')
        assert row['data_reference'] == '261106_identified'
    assert (by_name['C_T']['origin'], by_name['C_T']['value']) == ('measured', '1e-8')


@pytest.mark.integration
@pytest.mark.parametrize('model', ['supermodule', 'host_only'])
def test_generating_from_the_generated_parameters_file_alone_gives_the_same_model(tmp_path, model,
                                                                                   request):
    if model == 'supermodule':
        three = request.getfixturevalue('three_compartment')
        args = (three[1], _empty_heart(three), [three[0]])
    else:
        args = (os.path.join(RESOURCES_DIR, '3compartment_module_array.csv'),
                os.path.join(RESOURCES_DIR, '3compartment_parameters.json'), None)
    first = _generate(tmp_path / 'first', args[0], args[1], args[2])
    again = _generate(tmp_path / 'again', args[0], _generated_file(first), args[2])
    # every generated file, the generated parameters file included
    _assert_same_generated_models(os.path.join(first, f'{PREFIX}.cellml'),
                                  os.path.join(again, f'{PREFIX}.cellml'))


# --------------------------------------------------------------------------------------------
# calibrated results
# --------------------------------------------------------------------------------------------

def _tree(directory):
    """``{relative path: bytes}`` of every file under ``directory``."""
    out = {}
    for root, _, files in os.walk(directory):
        for name in files:
            path = os.path.join(root, name)
            with open(path, 'rb') as f:
                out[os.path.relpath(path, directory)] = f.read()
    return out


@pytest.fixture
def chain(tmp_path):
    """The workflow test library, its chain target's parameters file without C/k, which its
    model then takes from lin_c's default parameterisation."""
    library = build_workflow_library(str(tmp_path / 'library'))
    path = os.path.join(library['arrays'], 'chain_parameters.json')
    write_parameters(path, [r for r in read_parameters(path) if r['variable_name'] != 'C/k'])
    return library


def _assert_complete_result(path):
    rows = {r['variable_name']: r for r in read_parameters(path)}
    assert sorted(rows) == ['A/p', 'C/k', 'C/q']
    # the calibrated values
    assert float(rows['A/p']['value']) == pytest.approx(np.sqrt(Y_A / 2), rel=1e-4)
    assert float(rows['C/q']['value']) == pytest.approx(np.sqrt(Z_C - Y_A), rel=1e-4)
    assert rows['C/q']['data_reference'].startswith('calibrated by workflow ')
    # the value the target's file does not set, with its source
    assert rows['C/k']['value'] == '1.0'
    assert rows['C/k']['data_reference'] == f'from lin_c/v1 parameterisation default: {K_SOURCE}'
    # a calibrated value says so in its origin; the value the target's file does not set has none
    assert {n: set(r) for n, r in rows.items()} == {
        'A/p': {'variable_name', 'value', 'units', 'data_reference', 'origin'},
        'C/q': {'variable_name', 'value', 'units', 'data_reference', 'origin'},
        'C/k': {'variable_name', 'value', 'units', 'data_reference'}}
    assert rows['A/p']['origin'] == rows['C/q']['origin'] == 'calibrated'
    return rows


@pytest.mark.integration
def test_a_calibrated_result_holds_every_value_and_the_calibrated_ones(chain, tmp_path):
    out = str(tmp_path / 'run')
    result = run_calibration_workflow(chain['chain_array'], output_dir=out,
                                      write_calibrated=True)
    _assert_complete_result(result['calibrated_parameters_file'])
    # the target's generated model has its parameters file, with the calibrated values
    assert sorted(result['target_model']) == ['model_path', 'parameters_path']
    target_parameters = result['target_model']['parameters_path']
    assert os.path.basename(target_parameters) == 'target_parameters.json'
    assert {r['variable_name'] for r in read_parameters(target_parameters)} == \
        {'A/p', 'C/k', 'C/q'}
    with open(os.path.join(str(tmp_path / 'run'), 'workflow_result.json')) as f:
        assert json.load(f)['target_model']['parameters_path'] == target_parameters


@pytest.mark.integration
def test_write_calibrated_parameters_to_writes_a_complete_result_to_its_path_only(chain,
                                                                                  tmp_path):
    out, path = str(tmp_path / 'run'), str(tmp_path / 'send' / 'chain_parameters.json')
    run_calibration_workflow(chain['chain_array'], output_dir=out)
    library_root = os.path.dirname(chain['modules'])
    before, run_before = _tree(library_root), _tree(out)
    assert write_calibrated_parameters_to(chain['chain_array'], out, path) == path
    _assert_complete_result(path)
    # nothing else is written: not the target, the module library or the run
    assert _tree(library_root) == before
    assert _tree(out) == run_before
    assert os.listdir(os.path.dirname(path)) == ['chain_parameters.json']



@pytest.mark.integration
def test_a_result_written_later_holds_the_values_the_run_generated_file(chain, tmp_path):
    """write_calibrated_parameters_to builds on the run's own generated target parameters: a library
    default changed after the run does not reach the result. With no target model in the run,
    the target is generated again, against the library as it is then."""
    out, path = str(tmp_path / 'run'), str(tmp_path / 'send' / 'chain_parameters.json')
    run_calibration_workflow(chain['chain_array'], output_dir=out)
    default = os.path.join(chain['modules'], 'lin_c', 'versions', 'v1', 'parameterisations',
                           'default', 'default_parameters.json')
    write_parameters(default, [dict(r, value='2.0') if r['variable_name'] == 'k' else r
                               for r in read_parameters(default, parameters_file.MODULE)],
                     parameters_file.MODULE)
    write_calibrated_parameters_to(chain['chain_array'], out, path)
    assert _assert_complete_result(path)['C/k']['value'] == '1.0'
    shutil.rmtree(os.path.join(out, '_target'))
    write_calibrated_parameters_to(chain['chain_array'], out, path)
    assert {r['variable_name']: r['value'] for r in read_parameters(path)}['C/k'] == '2.0'


@pytest.mark.integration
def test_a_library_module_s_calibrated_result_has_only_a_library_s_keys(tmp_path):
    """A library target's result, in its parameterisation directory, holds every value its model
    uses with only a parameterisation's keys (value, units, data_reference, sourced, and origin
    calibrated on a calibrated value):
    C/k, which neither the chain's default nor "rest" sets, names lin_c's default in its
    data_reference."""
    library = build_workflow_library(str(tmp_path / 'library'))
    chain_dir = os.path.join(library['modules'], 'chain', 'versions', 'v1', 'parameterisations')
    for name in ('default', 'rest'):
        path = os.path.join(chain_dir, name, f'{name}_parameters.json')
        write_parameters(path, [r for r in read_parameters(path, parameters_file.MODULE)
                                if r['variable_name'] != 'C/k'], parameters_file.MODULE)
    result = run_calibration_workflow(library['chain_rest'], output_dir=str(tmp_path / 'run'),
                                      write_calibrated=True)
    path = result['calibrated_parameters_file']
    assert os.path.dirname(path) == os.path.join(chain_dir, 'rest')
    rows = {r['variable_name']: r for r in read_parameters(path, parameters_file.MODULE)}
    assert sorted(rows) == ['A/p', 'C/k', 'C/q']
    assert all(set(r) - {'variable_name'} <= {'value', 'units', 'data_reference', 'sourced',
                                              'origin'} for r in rows.values())
    assert rows['C/q']['origin'] == 'calibrated' and 'origin' not in rows['C/k']
    assert rows['C/k']['data_reference'] == f'from lin_c/v1 parameterisation default: {K_SOURCE}'
    assert rows['C/q']['data_reference'].startswith('calibrated by workflow chain_rest, step rest')


@pytest.mark.unit
def test_a_sourced_reference_is_prefixed_once():
    source = 'chamber/v1 parameterisation lv'
    assert parameters_file.sourced_reference(source, 'doe2020; Table 1') == \
        'from chamber/v1 parameterisation lv: doe2020; Table 1'
    assert parameters_file.sourced_reference(source, '') == 'from chamber/v1 parameterisation lv'
    once = parameters_file.sourced_reference(source, 'doe2020')
    assert parameters_file.sourced_reference('other/v1 parameterisation x', once) == once


@pytest.mark.integration
def test_a_run_that_records_a_generated_parameters_csv_is_refused(chain, tmp_path):
    """No reader takes a generated parameters CSV: a run whose workflow_result.json names one
    for its target model (an older libcuflynx's) is refused when its result is written."""
    out, path = str(tmp_path / 'run'), str(tmp_path / 'send' / 'chain_parameters.json')
    run_calibration_workflow(chain['chain_array'], output_dir=out)
    result_path = os.path.join(out, 'workflow_result.json')
    with open(result_path) as f:
        result = json.load(f)
    csv_path = os.path.join(os.path.dirname(result['target_model']['parameters_path']),
                            'target_parameters.csv')
    with open(csv_path, 'w') as f:
        f.write('variable_name,units,value,data_reference\nk_mod__C,dimensionless,1.0,x\n')
    result['target_model']['parameters_path'] = csv_path
    with open(result_path, 'w') as f:
        json.dump(result, f)
    with pytest.raises(WorkflowError, match='no longer reads'):
        write_calibrated_parameters_to(chain['chain_array'], out, path)
    assert not os.path.exists(path)
