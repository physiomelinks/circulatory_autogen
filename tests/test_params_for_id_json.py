"""params_for_id as JSON, with the CSV converted on read.

The CSV stays readable forever; it is converted to the JSON structure at the front door and
everything downstream sees only resolved entries. The whole backwards-compatibility risk lives in
that conversion: a silent error there would misparameterise a model without failing, so the
round-trip is asserted exhaustively over every fixture in resources/ rather than by spot checks.
"""
import json
import pathlib

import numpy as np
import pytest

from libcuflynx.parsers.PrimitiveParsers import (
    ObsAndParamDataParser, PARAMS_FOR_ID_ENTRY_KEYS, PARAMS_FOR_ID_JSON_VERSION)

RESOURCES = pathlib.Path(__file__).resolve().parent.parent / 'resources'
FIXTURES = sorted(RESOURCES.glob('*params_for_id.csv'))


def _parser():
    return ObsAndParamDataParser()


def _assert_info_equal(a, b, context):
    assert set(a) == set(b), f'{context}: key sets differ'
    for key in a:
        left, right = a[key], b[key]
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            left, right = np.asarray(left), np.asarray(right)
            assert left.shape == right.shape, f'{context}: {key} shape'
            if left.dtype.kind == 'f':
                assert np.allclose(left, right, equal_nan=True), f'{context}: {key}'
            else:
                assert list(left) == list(right), f'{context}: {key}'
        else:
            assert left == right, f'{context}: {key}'


@pytest.mark.unit
def test_there_are_fixtures_to_check():
    """A glob that silently matched nothing would make the whole suite below vacuous."""
    assert len(FIXTURES) >= 15, [f.name for f in FIXTURES]


@pytest.mark.unit
@pytest.mark.parametrize('csv_path', FIXTURES, ids=lambda p: p.stem)
def test_every_fixture_converts_to_json_and_back_to_the_same_param_id_info(csv_path, tmp_path):
    """CSV -> JSON -> param_id_info must equal CSV -> param_id_info, key by key.

    This is the backwards-compatibility guarantee. Every existing user study is a CSV.
    """
    parser = _parser()
    from_csv = parser.get_param_id_info(str(csv_path))

    doc = parser.params_for_id_csv_to_json(str(csv_path))
    json_path = tmp_path / f'{csv_path.stem}.json'
    json_path.write_text(json.dumps(doc))
    from_json = parser.get_param_id_info(str(json_path))

    _assert_info_equal(from_csv, from_json, csv_path.name)


@pytest.mark.unit
@pytest.mark.parametrize('csv_path', FIXTURES, ids=lambda p: p.stem)
def test_the_converter_output_is_json_serialisable_and_well_formed(csv_path):
    """CUFLynx reads this straight into an editor, so it must be plain JSON with known keys."""
    doc = _parser().params_for_id_csv_to_json(str(csv_path))
    json.dumps(doc)  # raises on numpy scalars or NaN-typed leftovers

    assert doc['version'] == PARAMS_FOR_ID_JSON_VERSION
    assert isinstance(doc['params'], list) and doc['params']
    for entry in doc['params']:
        assert set(entry) <= PARAMS_FOR_ID_ENTRY_KEYS, set(entry) - PARAMS_FOR_ID_ENTRY_KEYS
        assert entry['targets'] and all('/' in t for t in entry['targets'])
        assert entry['name']


@pytest.mark.unit
def test_a_grouped_row_becomes_one_entry_with_several_targets():
    """vessel_name='a b' is one calibrated value over two qnames -- one entry, two targets."""
    csv_text = ('vessel_name,param_name,min,max\n'
                'a b,C,1e-9,5e-8\n')
    doc = _parser().params_for_id_csv_to_json(csv_text)
    assert len(doc['params']) == 1
    assert doc['params'][0]['targets'] == ['a/C', 'b/C']


@pytest.mark.unit
def test_the_global_plus_named_vessel_gen_names_survive_conversion():
    """#368/#350: a row mixing 'global' with named vessels must emit one gen name per vessel.

    Deciding the whole row from the first vessel dropped every vessel after it, while param_names
    kept them, so the two positional lists stopped describing the same parameters.
    """
    csv_text = ('vessel_name,param_name,min,max\n'
                'global a b,C,1e-9,5e-8\n')
    parser = _parser()
    info = parser._build_param_id_info_from_entries(
        parser.resolve_params_for_id_doc(parser.params_for_id_csv_to_json(csv_text)))

    assert info['param_names'] == [['global/C', 'a/C', 'b/C']]
    # 'global' contributes the bare param name; a named vessel gets the suffix
    assert info['param_names_for_gen'] == [['C', 'C_a', 'C_b']]


@pytest.mark.unit
def test_targets_may_mix_different_parameter_names():
    """The reason for JSON: a CSV row has one param_name for all its vessels, so an arbitrary
    group could not be written at all."""
    parser = _parser()
    doc = {'version': 1, 'params': [
        {'name': 'mixed', 'targets': ['aortic_root/C', 'heart/E_lv_A'], 'min': 0.5, 'max': 2.0},
    ]}
    info = parser._build_param_id_info_from_entries(parser.resolve_params_for_id_doc(doc))
    assert info['param_names'] == [['aortic_root/C', 'heart/E_lv_A']]
    assert info['param_names_for_gen'] == [['C_aortic_root', 'E_lv_A_heart']]


@pytest.mark.unit
def test_defaults_apply_and_an_entry_overrides_them():
    """The 'easier to change general priors' ask: set the family once at the top."""
    parser = _parser()
    doc = {'version': 1,
           'defaults': {'prior': 'uniform', 'param_type': 'const'},
           'params': [
               {'name': 'a', 'targets': ['x/C'], 'min': 1.0, 'max': 2.0},
               {'name': 'b', 'targets': ['y/C'], 'min': 1.0, 'max': 2.0, 'prior': 'normal',
                'prior_params': {'prior_mean': 1.5, 'prior_std': 0.1}},
           ]}
    info = parser._build_param_id_info_from_entries(parser.resolve_params_for_id_doc(doc))
    assert list(info['param_prior_types']) == ['uniform', 'normal']


@pytest.mark.unit
def test_prior_params_merge_per_key_rather_than_wholesale():
    """A defaults block setting one hyper-parameter must not wipe an entry's other one."""
    parser = _parser()
    doc = {'version': 1,
           'defaults': {'prior': 'normal', 'prior_params': {'prior_std': 0.25}},
           'params': [{'name': 'a', 'targets': ['x/C'], 'min': 1.0, 'max': 2.0,
                       'prior_params': {'prior_mean': 1.5}}]}
    entries = parser.resolve_params_for_id_doc(doc)
    assert entries[0]['prior_params'] == {'prior_std': 0.25, 'prior_mean': 1.5}


@pytest.mark.unit
def test_an_unknown_entry_key_is_refused():
    """Same stance as operation_kwargs/cost_kwargs: a key nothing reads changes nothing and gives
    no sign it was ignored."""
    parser = _parser()
    with pytest.raises(ValueError, match='unknown key'):
        parser.resolve_params_for_id_doc(
            {'version': 1, 'params': [{'targets': ['x/C'], 'min': 1, 'max': 2, 'mn': 0.1}]})


@pytest.mark.unit
def test_an_unknown_defaults_key_is_refused():
    parser = _parser()
    with pytest.raises(ValueError, match='unknown key'):
        parser.resolve_params_for_id_doc(
            {'version': 1, 'defaults': {'priorr': 'uniform'},
             'params': [{'targets': ['x/C'], 'min': 1, 'max': 2}]})


@pytest.mark.unit
def test_duplicate_entry_names_are_refused():
    parser = _parser()
    with pytest.raises(ValueError, match='reuses the name'):
        parser.resolve_params_for_id_doc({'version': 1, 'params': [
            {'name': 'dup', 'targets': ['x/C'], 'min': 1, 'max': 2},
            {'name': 'dup', 'targets': ['y/C'], 'min': 1, 'max': 2},
        ]})


@pytest.mark.unit
def test_a_target_without_a_component_is_refused():
    """The most common typo, and it used to fail much later and less clearly."""
    parser = _parser()
    with pytest.raises(ValueError, match="not a 'component/param' name"):
        parser.resolve_params_for_id_doc(
            {'version': 1, 'params': [{'targets': ['C'], 'min': 1, 'max': 2}]})


@pytest.mark.unit
def test_an_unsupported_version_is_refused():
    parser = _parser()
    with pytest.raises(ValueError, match='version'):
        parser.resolve_params_for_id_doc(
            {'version': 99, 'params': [{'targets': ['x/C'], 'min': 1, 'max': 2}]})


@pytest.mark.unit
def test_name_defaults_to_the_first_target():
    parser = _parser()
    entries = parser.resolve_params_for_id_doc(
        {'version': 1, 'params': [{'targets': ['a/C', 'b/C'], 'min': 1, 'max': 2}]})
    assert entries[0]['name'] == 'a/C'


@pytest.mark.unit
def test_idxs_to_ignore_filters_entries():
    parser = _parser()
    doc = {'version': 1, 'params': [
        {'name': 'a', 'targets': ['x/C'], 'min': 1, 'max': 2},
        {'name': 'b', 'targets': ['y/C'], 'min': 3, 'max': 4},
    ]}
    entries = parser.resolve_params_for_id_doc(doc)
    info = parser._build_param_id_info_from_entries(entries, idxs_to_ignore=[0])
    assert info['param_names'] == [['y/C']]
    assert info['param_entry_names'] == ['b']


@pytest.mark.unit
def test_the_programmatic_dict_api_still_works_and_agrees_with_the_csv():
    """The in-code params_for_id form is documented in CLAUDE.md and must keep working."""
    parser = _parser()
    from_entries = parser.get_param_id_info_from_entries([
        {'vessel_name': 'aortic_root', 'param_name': 'C', 'min': 1e-9, 'max': 5e-8},
        {'vessel_name': ['a', 'b'], 'param_name': 'R', 'min': 1.0, 'max': 2.0},
    ])
    assert from_entries['param_names'] == [['aortic_root/C'], ['a/R', 'b/R']]
    assert from_entries['param_names_for_gen'] == [['C_aortic_root'], ['R_a', 'R_b']]
