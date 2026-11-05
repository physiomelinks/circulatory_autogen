"""params_for_id as JSON, the only form libcuflynx reads.

A params_for_id CSV is refused, naming ``cuflynx-upgrade params-for-id``, which converts it once
(``utilities/params_for_id_file.csv_to_document``). A silent error in that conversion would
misparameterise a model without failing, so the converter is checked against the row reading
the CSV had (the in-code vessel_name/param_name form, which reads rows as the CSV reader did),
and every converted fixture in resources/ is read.
"""
import io
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from libcuflynx.parsers.PrimitiveParsers import (
    ObsAndParamDataParser, PARAMS_FOR_ID_ENTRY_KEYS, PARAMS_FOR_ID_JSON_VERSION)
from libcuflynx.upgrade import main as upgrade
from libcuflynx.utilities import params_for_id_file as pfi

RESOURCES = pathlib.Path(__file__).resolve().parent.parent / 'resources'
FIXTURES = sorted(RESOURCES.glob('*params_for_id.json'))

#: a CSV with every column the converter reads: grouped, nested and global vessels, priors,
#: unbounded, number text
RICH_CSV = (
    'vessel_name,param_name,param_type,min,max,name_for_plotting,prior,prior_mean,prior_std,'
    'unbounded,comment\n'
    'aortic_root,C,const,1e-9,5e-8,C_{ao},,,,,\n'
    'a b,R,const,1.0,2.0,R,normal,1.5,0.1,,shared\n'
    'heart__lv,E_A,const,1e8,5e8,E,,,,,\n'
    'global,q_lv_init,const,200e-6,1500e-6,q,,,,,\n'
    'global a,k,const,0.5,0.75,k,normal,0.6,0.05,true,\n')


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
        elif key == 'param_prior_params':
            assert [{k: float(v) for k, v in d.items()} for d in left] == \
                [{k: float(v) for k, v in d.items()} for d in right], f'{context}: {key}'
        else:
            assert left == right, f'{context}: {key}'


@pytest.mark.unit
def test_there_are_fixtures_to_check():
    """A glob that silently matched nothing would make the whole suite below vacuous."""
    assert len(FIXTURES) >= 15, [f.name for f in FIXTURES]


@pytest.mark.unit
@pytest.mark.parametrize('json_path', FIXTURES, ids=lambda p: p.stem)
def test_every_fixture_is_json_with_known_keys_and_reads(json_path):
    """CUFLynx reads these straight into an editor, so they must be plain JSON with known keys."""
    doc = json.loads(json_path.read_text())
    assert doc['version'] == PARAMS_FOR_ID_JSON_VERSION
    assert isinstance(doc['params'], list) and doc['params']
    for entry in doc['params']:
        assert set(entry) <= PARAMS_FOR_ID_ENTRY_KEYS, set(entry) - PARAMS_FOR_ID_ENTRY_KEYS
        assert entry['targets'] and all('/' in t for t in entry['targets'])
    info = _parser().get_param_id_info(str(json_path))
    assert len(info['param_names']) == len(doc['params'])


@pytest.mark.unit
def test_a_converted_csv_reads_as_its_rows_did(tmp_path):
    """CSV -> JSON (the converter) -> param_id_info equals the rows read as vessel_name /
    param_name entries, key by key, but for the labels of a nested vessel, which the JSON names
    by its path."""
    parser = _parser()
    from_rows = parser._build_param_id_info_from_df(pd.read_csv(io.StringIO(RICH_CSV), dtype=str))
    json_path = pfi.write_params_for_id(tmp_path / 'm_params_for_id.json',
                                        pfi.csv_to_document(RICH_CSV))
    from_json = parser.get_param_id_info(str(json_path))
    for info in (from_rows, from_json):
        info.pop('param_names_for_plotting')
    _assert_info_equal(from_rows, from_json, 'rich CSV')


@pytest.mark.unit
def test_the_converter_writes_targets_as_users_name_them_and_numbers_as_written():
    doc = pfi.csv_to_document(RICH_CSV)
    assert [e['targets'] for e in doc['params']] == [
        ['aortic_root/C'], ['a/R', 'b/R'], ['heart/lv/E_A'], ['global/q_lv_init'],
        ['global/k', 'a/k']]
    text = pfi.dump_params_for_id(doc)
    assert '"min": 200e-6, "max": 1500e-6' in text and '"min": 1e-9' in text
    assert json.loads(text)['params'][1] == {
        'targets': ['a/R', 'b/R'], 'min': 1.0, 'max': 2.0, 'param_type': 'const',
        'name_for_plotting': 'R', 'prior': 'normal',
        'prior_params': {'prior_mean': 1.5, 'prior_std': 0.1}, 'comment': 'shared'}
    assert json.loads(text)['params'][4]['unbounded'] is True
    # the entry's name defaults to its first target, so the converter writes none
    assert all('name' not in e for e in doc['params'])


@pytest.mark.unit
def test_a_nested_vessel_s_path_is_the_model_s_flat_vessel():
    """heart/lv/E_A (what users write) is the model's heart__lv/E_A, as written in a CSV."""
    parser = _parser()
    info = parser._build_param_id_info_from_entries(parser.resolve_params_for_id_doc(
        {'params': [{'targets': ['heart/lv/E_A', 'heart__rv/E_A'], 'min': 1, 'max': 2}]}))
    assert info['param_names'] == [['heart__lv/E_A', 'heart__rv/E_A']]
    assert info['param_names_for_gen'] == [['E_A_heart__lv', 'E_A_heart__rv']]


@pytest.mark.unit
def test_a_params_for_id_csv_is_refused_naming_the_converter(tmp_path):
    path = tmp_path / 'm_params_for_id.csv'
    path.write_text('vessel_name,param_name,min,max\na,k,1,2\n')
    with pytest.raises(ValueError, match='cuflynx-upgrade params-for-id .*m_params_for_id.csv'):
        _parser().get_param_id_info(str(path))


@pytest.mark.unit
def test_user_inputs_naming_a_csv_or_with_only_a_csv_are_refused(tmp_path):
    from libcuflynx.parsers.PrimitiveParsers import YamlFileParser
    (tmp_path / 'm_params_for_id.csv').write_text('vessel_name,param_name,min,max\na,k,1,2\n')
    (tmp_path / 'm_obs_data.json').write_text('{}')
    for extra in ({'params_for_id_file': 'm_params_for_id.csv'}, {}):
        config = dict({'file_prefix': 'm', 'input_param_file': 'm_parameters.json',
                       'resources_dir': str(tmp_path), 'param_id_obs_path': str(
                           tmp_path / 'm_obs_data.json'),
                       'generated_models_dir': str(tmp_path / 'gen'),
                       'param_id_output_dir': str(tmp_path / 'out')}, **extra)
        with pytest.raises(ValueError, match='cuflynx-upgrade params-for-id'):
            YamlFileParser().parse_user_inputs_file(config, obs_path_needed=True,
                                                    do_generation_with_fit_parameters=False)


@pytest.mark.unit
def test_cuflynx_upgrade_params_for_id_writes_the_json_beside_and_removes_the_csv(tmp_path,
                                                                                 capsys):
    (tmp_path / 'sub').mkdir()
    one = tmp_path / 'm_params_for_id.csv'
    one.write_text(RICH_CSV)
    other = tmp_path / 'sub' / 'chosen.csv'
    other.write_text('vessel_name, param_name, min, max\nx, k, 1, 2\n')
    unrelated = tmp_path / 'sub' / 'm_parameters.csv'
    unrelated.write_text('variable_name,units,value,data_reference\nx/k,m,1,r\n')

    assert upgrade(['params-for-id', str(one), '--dry-run']) == 0
    assert one.exists() and not (tmp_path / 'm_params_for_id.json').exists()
    assert upgrade(['params-for-id', str(tmp_path)]) == 0
    assert not one.exists() and not other.exists() and unrelated.exists()
    assert pfi.read_params_for_id(tmp_path / 'm_params_for_id.json') == \
        json.loads(pfi.dump_params_for_id(pfi.csv_to_document(RICH_CSV)))
    assert pfi.read_params_for_id(tmp_path / 'sub' / 'chosen.json')['params'] == [
        {'targets': ['x/k'], 'min': 1, 'max': 2}]
    assert 'written as chosen.json' in capsys.readouterr().out
    with pytest.raises(SystemExit, match='not a params_for_id CSV'):
        upgrade(['params-for-id', str(unrelated)])


@pytest.mark.unit
def test_a_grouped_row_becomes_one_entry_with_several_targets():
    """vessel_name='a b' is one calibrated value over two qnames -- one entry, two targets."""
    doc = _parser().params_for_id_csv_to_json('vessel_name,param_name,min,max\na b,C,1e-9,5e-8\n')
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
    with pytest.raises(ValueError, match='was unexpected'):
        parser.resolve_params_for_id_doc(
            {'version': 1, 'params': [{'targets': ['x/C'], 'min': 1, 'max': 2, 'mn': 0.1}]})


@pytest.mark.unit
def test_an_unknown_defaults_key_is_refused():
    parser = _parser()
    with pytest.raises(ValueError, match='was unexpected'):
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
    with pytest.raises(ValueError, match=r"(?s)'C' is not in the form.*<vessel>/<variable>"):
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
def test_the_programmatic_dict_api_still_works():
    """The in-code params_for_id form is documented in CLAUDE.md and must keep working."""
    parser = _parser()
    from_entries = parser.get_param_id_info_from_entries([
        {'vessel_name': 'aortic_root', 'param_name': 'C', 'min': 1e-9, 'max': 5e-8},
        {'vessel_name': ['a', 'b'], 'param_name': 'R', 'min': 1.0, 'max': 2.0},
    ])
    assert from_entries['param_names'] == [['aortic_root/C'], ['a/R', 'b/R']]
    assert from_entries['param_names_for_gen'] == [['C_aortic_root'], ['R_a', 'R_b']]
