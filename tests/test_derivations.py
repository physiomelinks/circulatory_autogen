"""Parameter derivations (libcuflynx.derivations, cuflynx-derive) and obs_data input_items.

The toy (tests/test_inputs/derivations): ``Q_T0 = CO * frac`` by a Python function, then
``R_T0 = (MAP - CVP) / Q_T0`` by a CellML model in Pa, with MAP a data_item and CVP an
input_item in mmHg, so the chain, the unit conversion and both input kinds are exercised.
"""
import copy
import json
import os
import shutil
import warnings

import numpy as np
import pytest

from libcuflynx.derivations import (DerivationError, DerivationFunctionError,
                                    DerivationModelError, check_derivations, derivation_files,
                                    derivations_file_for, evaluate_cellml, evaluate_derivations,
                                    load_derivations, order_derivations, parameters_file_for,
                                    read_parameters)
from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser
from libcuflynx.schemas import DERIVATIONS_SCHEMA, OBS_DATA_SCHEMA
from libcuflynx.schemas.validation import problems
from libcuflynx.scripts.derive_script import main as derive_main
from libcuflynx.utilities.obs_data_helpers import ObsDataCreator

TOY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_inputs', 'derivations')
MMHG = 133.322
Q = 1e-4 * 0.8
R = (93.0 - 5.0) * MMHG / Q


def _read(path):
    with open(path) as f:
        return json.load(f)


def _write(path, doc):
    with open(path, 'w') as f:
        json.dump(doc, f, indent=1)


@pytest.fixture
def toy(tmp_path):
    """A copy of the toy, whose files a test may change."""
    root = tmp_path / 'toy'
    shutil.copytree(TOY, str(root))
    return {'dir': str(root), 'derivations': str(root / 'toy_derivations.json'),
            'obs': str(root / 'toy_obs_data.json'),
            'parameters': str(root / 'toy_parameters.json'),
            'cellml': str(root / 'derivations' / 'systemic_resistance.cellml')}


def _evaluate(toy, derivations=None, obs=None):
    derivations = load_derivations(derivations or toy['derivations'])
    doc = read_parameters(toy['parameters'])
    return evaluate_derivations(derivations, doc, obs or toy['obs']), doc


def _values(results):
    return {out['parameter']: out['value'] for r in results for out in r.outputs.values()}


def _parse_obs(doc):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return ObsAndParamDataParser().parse_obs_data_json(obs_data_dict=copy.deepcopy(doc))


# ----------------------------------------------------------------------------------- input_items

@pytest.mark.unit
def test_input_items_are_parsed_and_returned_beside_the_data_items():
    doc = _read(os.path.join(TOY, 'toy_obs_data.json'))
    parsed = _parse_obs(doc)
    assert set(parsed['input_items']) == {'CVP_toy', 'CO_toy'}
    cvp = parsed['input_items']['CVP_toy']
    assert (cvp['value'], cvp['std'], cvp['unit'], cvp['data_reference']) == \
        (5.0, 1.0, 'mmHg', 'Toy2026')
    assert cvp['species'] is None
    assert parsed['gt_df']['data_item_name'].tolist() == ['MAP']


@pytest.mark.unit
def test_input_items_change_nothing_a_calibration_reads(tmp_path):
    doc = _read(os.path.join(TOY, 'toy_obs_data.json'))
    without = copy.deepcopy(doc)
    del without['input_items']
    parser = ObsAndParamDataParser()
    a, b = _parse_obs(doc), _parse_obs(without)
    assert a['gt_df'].equals(b['gt_df'])
    assert a['prediction_info'] == b['prediction_info']
    assert a['protocol_info'] == b['protocol_info']
    assert b['input_items'] == {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        info_a = parser.process_obs_info(a["gt_df"], str(tmp_path), 0.01)
        info_b = parser.process_obs_info(b["gt_df"], str(tmp_path), 0.01)
    assert info_a.keys() == info_b.keys()
    for key in info_a:
        assert repr(info_a[key]) == repr(info_b[key]), key


@pytest.mark.unit
@pytest.mark.parametrize('change, message', [
    (lambda d: d['input_items'][0].update(data_item_name='MAP'),
     r"Duplicate 'data_item_name'.*'MAP' x2 \(in data_items, input_items\)"),
    (lambda d: d['input_items'].append(dict(d['input_items'][0])), 'CVP_toy'),
    (lambda d: d['input_items'][0].update(operands=['a/b']), "'operands' was unexpected"),
    (lambda d: d['input_items'][0].pop('unit'), "'unit' is a required property"),
])
def test_an_input_item_the_parser_refuses(change, message):
    doc = _read(os.path.join(TOY, 'toy_obs_data.json'))
    change(doc)
    with pytest.raises(ValueError, match=message):
        _parse_obs(doc)


@pytest.mark.unit
def test_obs_data_creator_adds_input_items():
    obs = ObsDataCreator()
    obs.add_protocol_info([0.0], [[1.0]], {})
    obs.add_input_item({'data_item_name': 'CVP', 'value': 5.0, 'unit': 'mmHg', 'std': 1.0,
                        'source': 'toy'})
    with pytest.raises(ValueError, match='already used'):
        obs.add_input_item({'data_item_name': 'CVP', 'value': 6.0, 'unit': 'mmHg'})
    with pytest.raises(ValueError, match='not input item keys'):
        obs.add_input_item({'data_item_name': 'x', 'value': 1.0, 'unit': 'm', 'weight': 2})
    doc = obs.get_obs_data_dict()
    assert problems(OBS_DATA_SCHEMA, doc) == []
    assert _parse_obs(doc)['input_items']['CVP']['value'] == 5.0


# -------------------------------------------------------------------------------- the file

@pytest.mark.unit
def test_the_files_beside_each_other_are_named_from_one_another():
    assert derivations_file_for('/a/BVC_parameters.json') == '/a/BVC_derivations.json'
    assert parameters_file_for('/a/BVC_derivations.json') == '/a/BVC_parameters.json'


@pytest.mark.unit
@pytest.mark.parametrize('change, message', [
    (lambda d: d['derivations'][0].update(python='f.py:f'), 'is valid under each of'),
    (lambda d: d['derivations'][0].pop('cellml'), "'cellml' is a required property"),
    (lambda d: d['derivations'][0]['inputs']['MAP'].update(input_item='MAP'),
     'has too many properties'),
    (lambda d: d['derivations'][0]['inputs']['MAP'].update(model='x'), "'model' was unexpected"),
    (lambda d: d['derivations'][0].update(outputs={}), 'should be non-empty'),
])
def test_a_derivations_file_the_schema_refuses(toy, change, message):
    doc = _read(toy['derivations'])
    change(doc)
    found = problems(DERIVATIONS_SCHEMA, doc)
    assert found and any(message in str(p) for p in found), found
    with pytest.raises(ValueError):
        load_derivations(doc, toy['dir'])


@pytest.mark.unit
def test_a_derivation_name_given_twice_is_refused(toy):
    doc = _read(toy['derivations'])
    doc['derivations'][1]['name'] = 'R_T0'
    with pytest.raises(DerivationError, match="'R_T0' is given twice"):
        load_derivations(doc, toy['dir'])


@pytest.mark.unit
def test_the_files_a_derivations_file_reads(toy):
    assert derivation_files(toy['derivations']) == [
        toy['cellml'], os.path.join(toy['dir'], 'derivations', 'derivation_funcs.py')]


# -------------------------------------------------------------------------------- evaluation

@pytest.mark.unit
def test_the_toy_derives_its_parameters_and_its_cache_is_current(toy):
    results, doc = _evaluate(toy)
    assert _values(results) == pytest.approx({'Systemic_T/Q_T0': Q, 'Systemic_T/R_T0': R},
                                             rel=1e-12)
    assert check_derivations(results, doc) == []
    inputs = {r.derivation.name: r.inputs for r in results}
    assert inputs['R_T0']['CVP']['data_reference'] == 'Toy2026'
    assert inputs['R_T0']['MAP']['std'] == 5.0
    assert inputs['R_T0']['MAP']['units'] == 'mmHg'


@pytest.mark.unit
def test_a_cellml_derivation_converts_its_inputs_to_the_models_units(toy):
    in_pa = evaluate_cellml(toy['cellml'], {'MAP': (93.0 * MMHG, 'Pa'), 'CVP': (5.0 * MMHG, 'Pa'),
                                            'Q': (Q, 'm3_per_s')}, {'R': 'J_s_per_m6'})
    in_mmhg = evaluate_cellml(toy['cellml'], {'MAP': (93.0, 'mmHg'), 'CVP': (5.0, 'mmHg'),
                                              'Q': (Q, 'm3_per_s')}, {'R': None})
    assert in_pa['R'] == pytest.approx(R, rel=1e-12)
    assert in_mmhg['R'] == pytest.approx(R, rel=1e-12)


@pytest.mark.unit
def test_a_cellml_equation_whose_units_do_not_match_is_refused(toy):
    with open(toy['cellml']) as f:
        text = f.read()
    with open(toy['cellml'], 'w') as f:
        f.write(text.replace('<variable name="CVP" units="Pa"', '<variable name="CVP" units="mmHg"'))
    with pytest.raises(DerivationModelError, match="(?s)analysis found.*'MAP-CVP'.*not equivalent"):
        _evaluate(toy)


@pytest.mark.unit
@pytest.mark.parametrize('inputs, outputs, message', [
    ({'MAP': (93.0, 'second')}, {'R': None},
     "input 'MAP' is in 'second' and the model's 'MAP' in 'Pa'.*not units of one dimension"),
    ({}, {'R': 'Pa'}, "output 'R' is in 'J_s_per_m6' and its parameter in 'Pa'"),
    ({'R': (1.0, 'J_s_per_m6')}, {}, "input 'R' is a computed constant"),
    ({}, {'MAP': None}, "output 'MAP' is a constant"),
    ({'X': (1.0, 'Pa')}, {}, "input 'X' is not a variable"),
])
def test_a_cellml_derivation_refuses_inputs_and_outputs_it_cannot_take(toy, inputs, outputs,
                                                                        message):
    with pytest.raises(DerivationModelError, match=message):
        evaluate_cellml(toy['cellml'], inputs, outputs)


@pytest.mark.unit
def test_an_input_items_units_must_be_the_models_dimension(toy):
    obs = _read(toy['obs'])
    obs['input_items'][0]['unit'] = 'm3_per_s'
    _write(toy['obs'], obs)
    with pytest.raises(DerivationModelError, match="input 'CVP' is in 'm3_per_s'"):
        _evaluate(toy)


@pytest.mark.unit
def test_a_python_derivation_and_its_errors(toy):
    funcs = os.path.join(toy['dir'], 'derivations', 'derivation_funcs.py')
    with open(funcs, 'a') as f:
        f.write('\n\ndef wrong_outputs(inputs, units):\n    return {"q": 1.0}\n'
                '\n\ndef sees_units(inputs, units):\n    return {"Q": float(len(units))}\n')
    doc = _read(toy['derivations'])
    doc['derivations'] = [doc['derivations'][1]]
    _write(toy['derivations'], doc)
    results, _ = _evaluate(toy)
    assert _values(results) == {'Systemic_T/Q_T0': pytest.approx(Q, rel=1e-15)}

    for spec, error in [('derivations/derivation_funcs.py:wrong_outputs',
                         r"returned \['q'\]; its outputs are \['Q'\]"),
                        ('derivations/derivation_funcs.py:nope', "has no function 'nope'"),
                        ('derivations/derivation_funcs.py', 'names a file but no function')]:
        doc['derivations'][0]['python'] = spec
        _write(toy['derivations'], doc)
        with pytest.raises(DerivationFunctionError, match=error):
            _evaluate(toy)

    # units carries every input and output symbol
    doc['derivations'][0]['python'] = 'derivations/derivation_funcs.py:sees_units'
    _write(toy['derivations'], doc)
    assert _values(_evaluate(toy)[0]) == {'Systemic_T/Q_T0': 3.0}


@pytest.mark.unit
def test_chained_derivations_run_in_the_order_their_parameters_need(toy):
    doc = read_parameters(toy['parameters'])
    derivations = load_derivations(toy['derivations'])
    assert [d.name for d in derivations] == ['R_T0', 'Q_T0']
    assert [d.name for d in order_derivations(derivations, doc)] == ['Q_T0', 'R_T0']
    # R_T0 reads the derived Q_T0, not the stale value cached in the file
    params = _read(toy['parameters'])
    params['Systemic_T']['Q_T0']['value'] = 1.0
    params['Systemic_T']['frac']['value'] = 0.4
    _write(toy['parameters'], params)
    results, _ = _evaluate(toy)
    assert _values(results)['Systemic_T/R_T0'] == pytest.approx(2 * R, rel=1e-12)


@pytest.mark.unit
def test_a_loop_of_derivations_is_refused(toy):
    doc = _read(toy['derivations'])
    doc['derivations'][1]['inputs']['frac'] = {'parameter': 'Systemic_T/R_T0'}
    with pytest.raises(DerivationError, match='derivations loop: (R_T0 -> Q_T0 -> R_T0|'
                                              'Q_T0 -> R_T0 -> Q_T0)'):
        order_derivations(load_derivations(doc, toy['dir']), read_parameters(toy['parameters']))


@pytest.mark.unit
def test_a_parameter_set_by_two_derivations_is_refused(toy):
    doc = _read(toy['derivations'])
    doc['derivations'][1]['outputs'] = {'Q': 'Systemic_T/R_T0'}
    with pytest.raises(DerivationError, match="'Systemic_T/R_T0' is set by derivations "
                                              "'R_T0' and 'Q_T0'"):
        order_derivations(load_derivations(doc, toy['dir']), read_parameters(toy['parameters']))


@pytest.mark.unit
@pytest.mark.parametrize('change, message', [
    (lambda d: d['derivations'][0]['inputs'].update(CVP={'input_item': 'nope'}),
     "no obs_data has a input_item named 'nope'"),
    (lambda d: d['derivations'][0]['outputs'].update(R='Systemic_T/nope'),
     "has no parameter 'Systemic_T/nope'"),
    (lambda d: d['derivations'][1]['inputs'].update(frac={'data_item': 'CO_toy'}),
     "no obs_data has a data_item named 'CO_toy'"),
])
def test_an_input_or_output_that_does_not_resolve_is_named(toy, change, message):
    doc = _read(toy['derivations'])
    change(doc)
    _write(toy['derivations'], doc)
    with pytest.raises(DerivationError, match=message):
        _evaluate(toy)


@pytest.mark.unit
def test_the_cache_check_fails_after_an_input_changes(toy):
    obs = _read(toy['obs'])
    obs['input_items'][0]['value'] = 5.0 * (1 + 1e-6)
    _write(toy['obs'], obs)
    results, doc = _evaluate(toy)
    found = check_derivations(results, doc)
    assert len(found) == 1 and found[0].startswith("Systemic_T/R_T0: cached 146654199")
    # within rtol 1e-9 is current
    obs['input_items'][0]['value'] = 5.0 * (1 + 1e-12)
    _write(toy['obs'], obs)
    assert check_derivations(*_evaluate(toy)) == []


@pytest.mark.unit
def test_a_parameter_naming_a_derivation_that_does_not_set_it_is_reported(toy):
    params = _read(toy['parameters'])
    params['Systemic_T']['C_T']['derivation'] = 'R_T0'
    _write(toy['parameters'], params)
    found = check_derivations(*_evaluate(toy))
    assert found == ["Systemic_T/C_T: names derivation 'R_T0', which does not set it."]


# --------------------------------------------------------------------------------------- CLI

@pytest.mark.unit
def test_derive_write_round_trips(toy, capsys):
    before = _read(toy['parameters'])
    args = [toy['derivations'], '--obs', toy['obs']]
    assert derive_main(args) == 0
    obs = _read(toy['obs'])
    obs['data_items'][0]['value'] = 103.0
    _write(toy['obs'], obs)
    assert derive_main(args) == 1
    assert 'rerun with --write' in capsys.readouterr().out
    assert derive_main(args + ['--write']) == 0
    assert derive_main(args) == 0
    after = _read(toy['parameters'])
    assert after['Systemic_T']['R_T0']['value'] == pytest.approx(R * 98.0 / 88.0, rel=1e-12)
    # every other key and parameter is kept
    after['Systemic_T']['R_T0']['value'] = before['Systemic_T']['R_T0']['value']
    assert after == before
    with open(toy['parameters']) as f:
        assert '"C_T": {"value": 1e-8,' in f.read()       # numbers keep their text


@pytest.mark.unit
def test_derive_write_names_each_output_by_its_derivation(toy):
    params = _read(toy['parameters'])
    del params['Systemic_T']['R_T0']['derivation']
    _write(toy['parameters'], params)
    assert derive_main([toy['derivations'], '--obs', toy['obs'], '--write',
                        '--parameters', toy['parameters']]) == 0
    assert _read(toy['parameters'])['Systemic_T']['R_T0']['derivation'] == 'R_T0'


# ------------------------------------------------------------------------------ packing

@pytest.mark.unit
def test_a_packed_workflow_carries_the_targets_derivations(tmp_path):
    from libcuflynx.calibration_workflow import load_workflow, pack_workflow
    from libcuflynx.external_testing.workflow_library import build_workflow_library
    library = build_workflow_library(str(tmp_path / 'library'))
    target = load_workflow(library['chain_array']).target.parameters
    here = os.path.dirname(target)
    shutil.copytree(os.path.join(TOY, 'derivations'), os.path.join(here, 'derivations'))
    shutil.copyfile(os.path.join(TOY, 'toy_derivations.json'), derivations_file_for(target))

    packed = pack_workflow(library['chain_array'], str(tmp_path / 'pack'))
    target_dir = os.path.join('inputs', '_target')
    stem = os.path.basename(target)[:-len('_parameters.json')]
    for relative in (f'{stem}_derivations.json',
                     os.path.join('derivations', 'systemic_resistance.cellml'),
                     os.path.join('derivations', 'derivation_funcs.py')):
        assert os.path.join(target_dir, relative) in packed['files']
    packed_derivations = os.path.join(str(tmp_path / 'pack'), target_dir,
                                      f'{stem}_derivations.json')
    assert derivation_files(packed_derivations) == [
        os.path.join(os.path.dirname(packed_derivations), 'derivations', name)
        for name in ('systemic_resistance.cellml', 'derivation_funcs.py')]

    # a file outside the derivations file's directory cannot be packed beside it
    doc = _read(derivations_file_for(target))
    doc['derivations'][0]['cellml'] = '../elsewhere.cellml'
    _write(derivations_file_for(target), doc)
    with pytest.raises(ValueError, match='outside its directory'):
        pack_workflow(library['chain_array'], str(tmp_path / 'pack2'))


# ---------------------------------------------------- a calibration ignores the input_items

@pytest.mark.integration
@pytest.mark.mpi
def test_a_calibrations_cost_ignores_input_items(base_user_inputs, resources_dir,
                                                 temp_output_dir, temp_generated_models_dir,
                                                 tmp_path):
    """The cost of one parameter set is the same with and without input_items in the
    obs_data, on the real solver (Simple_ODE_Benchmark)."""
    from libcuflynx.param_id.paramID import CVS0DParamID
    from libcuflynx.parsers.PrimitiveParsers import YamlFileParser
    from libcuflynx.scripts.script_generate_with_new_architecture import \
        generate_with_new_architecture
    from libcuflynx.utilities.mpi_utils import get_MPI
    comm = get_MPI().COMM_WORLD

    doc = _read(os.path.join(resources_dir, 'Simple_ODE_Benchmark_obs_data.json'))
    with_inputs = copy.deepcopy(doc)
    with_inputs['input_items'] = [
        {'data_item_name': 'p_measured', 'value': 1.0, 'std': 0.1, 'unit': 'dimensionless',
         'source': 'a number only a derivation reads'}]
    paths = {}
    for name, obs in (('without', doc), ('with', with_inputs)):
        paths[name] = str(tmp_path / f'{name}_obs_data.json')
        _write(paths[name], obs)

    def config(obs_path):
        inputs = base_user_inputs.copy()
        inputs.update({
            'file_prefix': 'Simple_ODE_Benchmark',
            'input_param_file': 'Simple_ODE_Benchmark_parameters.json',
            'model_type': 'cellml', 'solver': 'CVODE_myokit',
            'param_id_method': 'genetic_algorithm', 'pre_time': 0.0, 'sim_time': 8.0,
            'dt': 0.05, 'DEBUG': True, 'do_uq': False, 'do_ia': False,
            'plot_predictions': False, 'solver_info': {'MaximumStep': 0.01},
            'param_id_obs_path': obs_path,
            'params_for_id_path': os.path.join(resources_dir,
                                               'Simple_ODE_Benchmark_params_for_id.json'),
            'param_id_output_dir': temp_output_dir, 'resources_dir': resources_dir,
            'generated_models_dir': temp_generated_models_dir})
        return YamlFileParser().parse_user_inputs_file(
            inputs, obs_path_needed=True, do_generation_with_fit_parameters=False)

    if comm.Get_rank() == 0:
        assert generate_with_new_architecture(False, config(paths['without']))
    comm.Barrier()
    costs = {}
    for name, path in paths.items():
        engine = CVS0DParamID.init_from_dict(config(path)).param_id
        theta = np.asarray(engine.param_id_info['param_mins'], dtype=float) * 1.3 + 0.1
        costs[name] = engine.get_cost_obs_and_pred_from_params(theta)[0]
        assert engine.obs_info['data_item_names'] == ['benchmark/x', 'benchmark/y']
    assert np.isfinite(costs['without'])
    assert costs['with'] == costs['without']
