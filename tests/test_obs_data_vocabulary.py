"""The obs_data entry vocabulary introduced by #466.

``variable`` named the item *and* stood in as its operand; ``name_for_plotting`` named the trace
*and* the scalar feature drawn from it. Both jobs of both keys are now separate fields:

    variable            -> data_item_name  (identity, unique) + operands (the model variable)
    name_for_plotting   -> trace_name_for_plotting (axis label, may repeat)
                         + item_name_for_plotting  (the feature's label)

The identity matters beyond tidiness: ``data_item_name`` is the key an ``operation_kwargs`` value
resolves against when one item is built from another, so a repeat silently picks whichever item
was evaluated last. See ``test_the_shipped_extra_ops_example_differences_two_distinct_items``.
"""
import json
import os
import warnings

import pandas as pd
import pytest

from libcuflynx.parsers.PrimitiveParsers import (ObsAndParamDataParser,
                                                 check_data_item_names_unique,
                                                 default_item_name_for_plotting)
from libcuflynx.utilities.obs_data_helpers import (LEGACY_OBS_INFO_KEYS,
                                                   LEGACY_OBS_ITEM_KEYS,
                                                   LEGACY_PREDICTION_INFO_KEYS,
                                                   migrate_legacy_obs_item_keys,
                                                   normalise_obs_info,
                                                   normalise_prediction_info,
                                                   obs_item_labels, obs_item_names,
                                                   obs_trace_labels)

_ROOT = os.path.join(os.path.dirname(__file__), '..')


def _item(**overrides):
    item = {'data_item_name': 'mean of x', 'data_type': 'constant', 'unit': 'dimensionless',
            'operands': ['main/x'], 'operation': 'mean', 'weight': 1.0,
            'value': 1.0, 'std': 0.1}
    item.update(overrides)
    return item


def _doc(data_items, prediction_items=None):
    doc = {'protocol_info': {'pre_times': [0.0], 'sim_times': [[1.0]], 'params_to_change': {}},
           'data_items': data_items}
    if prediction_items is not None:
        doc['prediction_items'] = prediction_items
    return doc


def _parse(doc):
    return ObsAndParamDataParser().parse_obs_data_json(obs_data_dict=doc,
                                                       pre_time=0.0, sim_time=1.0)


# ------------------------------------------------------------------ the legacy keys migrate

@pytest.mark.unit
def test_a_legacy_variable_becomes_the_data_item_name_and_warns_about_both_of_its_jobs():
    doc = _doc([{'variable': 'mean of x', 'data_type': 'constant', 'unit': 'dimensionless',
                 'operands': ['main/x'], 'operation': 'mean', 'weight': 1.0,
                 'value': 1.0, 'std': 0.1}])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = _parse(doc)
    assert out['gt_df']['data_item_name'].tolist() == ['mean of x']
    messages = ' '.join(str(w.message) for w in caught
                        if issubclass(w.category, DeprecationWarning))
    # both replacements named, because `variable` did both jobs
    assert 'data_item_name' in messages and 'operands' in messages


@pytest.mark.unit
def test_a_legacy_name_for_plotting_becomes_the_trace_name_and_warns_about_both_labels():
    doc = _doc([_item(name_for_plotting='x_{m}', **{})])
    doc['data_items'][0].pop('data_item_name', None)
    doc['data_items'][0]['data_item_name'] = 'mean of x'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = _parse(doc)
    assert out['gt_df']['trace_name_for_plotting'].tolist() == ['x_{m}']
    messages = ' '.join(str(w.message) for w in caught
                        if issubclass(w.category, DeprecationWarning))
    assert 'trace_name_for_plotting' in messages and 'item_name_for_plotting' in messages


@pytest.mark.unit
@pytest.mark.parametrize('legacy,current', sorted(LEGACY_OBS_ITEM_KEYS.items()))
def test_setting_a_legacy_key_and_its_replacement_together_is_an_error(legacy, current):
    """No precedence rule: silently picking one would fit whichever the author did not mean."""
    with pytest.raises(ValueError) as excinfo:
        migrate_legacy_obs_item_keys([{legacy: 'a', current: 'b'}])
    assert legacy in str(excinfo.value) and current in str(excinfo.value)


@pytest.mark.unit
def test_migration_does_not_mutate_the_caller_s_dicts():
    """An obs_data dict handed in by a user (or built by ObsDataCreator) is theirs."""
    original = {'variable': 'x', 'operands': ['main/x']}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        migrate_legacy_obs_item_keys([original])
    assert original == {'variable': 'x', 'operands': ['main/x']}


# ------------------------------------------------------------------------- the two labels

@pytest.mark.unit
def test_the_item_label_defaults_to_the_trace_label_and_the_operation():
    out = _parse(_doc([_item(trace_name_for_plotting='x')]))
    assert out['gt_df']['item_name_for_plotting'].tolist() == ['x (mean)']


@pytest.mark.unit
def test_an_item_with_no_operation_keeps_the_trace_label_unchanged():
    """The item *is* the series, so a dangling '(None)' would be worse than no suffix."""
    assert default_item_name_for_plotting('x', None) == 'x'
    assert default_item_name_for_plotting('x', 'mean') == 'x (mean)'


@pytest.mark.unit
def test_the_trace_label_defaults_to_the_model_variable_being_reduced():
    out = _parse(_doc([_item()]))
    assert out['gt_df']['trace_name_for_plotting'].tolist() == ['main/x']


@pytest.mark.unit
def test_two_items_may_share_a_trace_label_but_not_an_item_name():
    """The whole point of splitting the key: the mean and the max of one trace are one series
    with two features, so they share an axis label and differ in identity."""
    out = _parse(_doc([_item(data_item_name='mean of x', operation='mean',
                             trace_name_for_plotting='x'),
                       _item(data_item_name='max of x', operation='max',
                             trace_name_for_plotting='x')]))
    assert out['gt_df']['trace_name_for_plotting'].tolist() == ['x', 'x']
    assert out['gt_df']['item_name_for_plotting'].tolist() == ['x (mean)', 'x (max)']


# ------------------------------------------------------------------------------ uniqueness

@pytest.mark.unit
def test_a_repeated_data_item_name_is_rejected_and_the_offenders_are_named():
    with pytest.raises(ValueError) as excinfo:
        _parse(_doc([_item(data_item_name='x'), _item(data_item_name='x', operation='max')]))
    msg = str(excinfo.value)
    assert "'x'" in msg and 'data_item_name' in msg
    # and it must not send the author off renaming their axis labels
    assert 'trace_name_for_plotting' in msg


@pytest.mark.unit
def test_a_name_shared_between_a_data_item_and_a_prediction_item_is_rejected():
    """A reference does not care which list an item came from, so neither does the rule."""
    with pytest.raises(ValueError) as excinfo:
        _parse(_doc([_item(data_item_name='x')],
                    [{'data_item_name': 'x', 'operands': ['main/x'], 'unit': 'dimensionless'}]))
    assert 'prediction_items' in str(excinfo.value)


@pytest.mark.unit
def test_the_uniqueness_rule_is_importable_on_its_own():
    """CUFLynx validates an edited obs_data before handing it back, so the rule has to be
    callable without going through a full parse."""
    df = pd.DataFrame([{'data_item_name': 'a'}, {'data_item_name': 'a'}])
    with pytest.raises(ValueError):
        check_data_item_names_unique(df)
    check_data_item_names_unique(pd.DataFrame([{'data_item_name': 'a'}]))


# --------------------------------------------------------------- the removed operand fallback

@pytest.mark.unit
def test_obs_type_no_longer_takes_its_operand_from_the_item_name():
    """`obs_type: max` used to mean "max of whatever `variable` names". `variable` no longer
    names a model variable, so the operand has to be stated."""
    item = _item(operation=None, obs_type='max')
    item['operands'] = []
    parser = ObsAndParamDataParser()
    parsed = parser.parse_obs_data_json(obs_data_dict=_doc([item]), pre_time=0.0, sim_time=1.0)
    with pytest.raises(ValueError) as excinfo:
        parser.process_obs_info(parsed['gt_df'], '/tmp', 0.01)
    assert 'operands' in str(excinfo.value)


@pytest.mark.unit
def test_a_missing_data_item_name_says_what_replaced_variable():
    item = _item()
    del item['data_item_name']
    with pytest.raises(ValueError) as excinfo:
        _parse(_doc([item]))
    msg = str(excinfo.value)
    assert 'data_item_name' in msg and "'variable'" in msg


# --------------------------------------------------------------------------- prediction items

@pytest.mark.unit
def test_a_legacy_prediction_item_gets_its_operand_from_the_variable_it_named():
    """There was no `operands` key on a prediction_item at all -- `variable` was the qname."""
    out = _parse(_doc([_item()],
                      [{'variable': 'main/y', 'unit': 'dimensionless'}]))
    pred = out['prediction_info']
    assert pred['operands'] == [['main/y']]
    assert pred['data_item_names'] == ['main/y']


# ------------------------------------------------------------------- the shipped fixtures

def _authored_obs_data_files():
    """Committed obs_data files only.

    Deliberately git-tracked rather than a filesystem walk: a local run writes generated
    obs_data into resources/ (the 3compartment synthetic benchmark data, for one), and those
    are artifacts of somebody's run, not inputs this repo is responsible for.
    """
    import subprocess
    listed = subprocess.run(['git', 'ls-files', '-z'], cwd=_ROOT,
                            capture_output=True, text=True, check=True).stdout.split('\0')
    return sorted(os.path.join(_ROOT, f) for f in listed
                  if f.endswith('.json') and 'obs' in os.path.basename(f))


@pytest.mark.unit
@pytest.mark.parametrize('path', _authored_obs_data_files(),
                         ids=lambda p: os.path.basename(p))
def test_every_authored_obs_data_file_uses_the_current_vocabulary(path):
    with open(path, encoding='utf-8-sig') as fh:
        doc = json.load(fh)
    items = doc if isinstance(doc, list) else (doc.get('data_items') or doc.get('data_item') or [])
    preds = [] if isinstance(doc, list) else (doc.get('prediction_items') or [])
    items = [i for i in items if isinstance(i, dict)]
    preds = [i for i in preds if isinstance(i, dict)]
    if not items:
        pytest.skip('not an obs_data file')
    legacy = sorted({k for i in items + preds for k in i if k in LEGACY_OBS_ITEM_KEYS})
    assert not legacy, f'{os.path.relpath(path, _ROOT)} still uses {legacy}'

    names = [i.get('data_item_name') for i in items + preds]
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, f'{os.path.relpath(path, _ROOT)} repeats data_item_name {dupes}'
    assert all(n for n in names), f'{os.path.relpath(path, _ROOT)} has an item with no name'


@pytest.mark.unit
def test_the_shipped_extra_ops_example_differences_two_distinct_items():
    """The bug this vocabulary exists to prevent, pinned on the file that had it.

    ``3compartment_extra_ops`` asks for the max minus the mean of one trace. Both source items
    were labelled ``v_{AR}``, ``temp_results`` was keyed on that label, so both operands resolved
    to the same number and the difference was a constant 0.0 against a ground truth of 4e-4 --
    with every test still passing, because nothing asserted a value.
    """
    path = os.path.join(_ROOT, 'resources', '3compartment_extra_ops_obs_data.json')
    with open(path, encoding='utf-8-sig') as fh:
        items = json.load(fh)

    by_name = {i['data_item_name']: i for i in items}
    diff = next(i for i in items
                if i.get('operation') == 'calculate_two_observable_difference')
    pred1, pred2 = diff['operation_kwargs']['pred1'], diff['operation_kwargs']['pred2']

    assert pred1 != pred2, 'both operands name the same item, so the difference is always 0'
    assert {pred1, pred2} <= set(by_name), 'an operand names no data_item in this file'
    # and they must be different features, not two spellings of one
    assert by_name[pred1]['operation'] != by_name[pred2]['operation']
    # the function returns pred2 - pred1, so the stated ground truth has to match
    assert diff['value'] == pytest.approx(by_name[pred2]['value'] - by_name[pred1]['value'])


# --- the shipped examples ------------------------------------------------------------------

_RESOURCES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resources')


def _shipped_obs_data_files():
    return sorted(f for f in os.listdir(_RESOURCES)
                  if 'obs_data' in f and f.endswith('.json'))


@pytest.mark.unit
def test_no_shipped_obs_data_uses_the_deprecated_obs_type():
    """The files people copy must be written the way people should write them.

    `obs_type` is read only when `operation` is null, and then only `min`/`max`/`mean` name an
    operation at all -- `series`/`frequency` were saying what `data_type` already says. Five
    shipped files still spelled it the old way after the #466 migration, so someone starting
    from `simple_physiological_obs_data.json` would learn the deprecated key and get the
    deprecation path.

    A file is checked as *text*: the parser accepts `obs_type` on purpose (that is what makes
    an existing study keep working), so parsing one proves nothing about how it is spelled.
    """
    offenders = []
    for name in _shipped_obs_data_files():
        with open(os.path.join(_RESOURCES, name), encoding='utf-8') as f:
            if '"obs_type"' in f.read():
                offenders.append(name)
    assert not offenders, (
        "these shipped obs_data files still use the deprecated 'obs_type': %s. Use 'operation' "
        "for min/max/mean; 'series'/'frequency' belong in 'data_type'." % offenders)


@pytest.mark.unit
def test_every_shipped_obs_data_item_has_a_unique_name():
    """`data_item_name` is the item's identity and what an operation_kwargs reference resolves
    against (#466), so a repeat makes one of them unreachable -- and silently resolves a
    reference to whichever was recorded last. That is the bug #466 was opened for: two items
    labelled `v_{AR}` made `max - mean` compute `max - max` and score a constant 0.0.
    """
    bad = {}
    for name in _shipped_obs_data_files():
        with open(os.path.join(_RESOURCES, name), encoding='utf-8') as f:
            doc = json.load(f)
        items = doc if isinstance(doc, list) else [
            i for key in ('data_items', 'data_item', 'prediction_items')
            for i in (doc.get(key) or [])]
        names = [i['data_item_name'] for i in items
                 if isinstance(i, dict) and 'data_item_name' in i]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            bad[name] = dupes
    assert not bad, "shipped obs_data files with repeated data_item_name: %s" % bad


# ---------------------------------------------------------------------------
# The parsed obs_info / prediction_info vocabulary -- a different layer from the
# entry keys above, with its own tables.
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_two_info_tables_disagree_on_purpose():
    """``names_for_plotting`` means the ITEM label in one dict and the TRACE label in the other.

    This is the test to read before merging the two tables into one, which is the tidy-up
    somebody will eventually attempt. A shared mapping cannot be right for both: whichever
    target it picked, the other dict's labels would move into the wrong slot, and nothing
    would fail -- an axis would just be captioned with the wrong text.
    """
    assert LEGACY_OBS_INFO_KEYS['names_for_plotting'] == 'item_names_for_plotting'
    assert LEGACY_PREDICTION_INFO_KEYS['names_for_plotting'] == 'trace_names_for_plotting'
    assert LEGACY_OBS_INFO_KEYS != LEGACY_PREDICTION_INFO_KEYS


@pytest.mark.unit
@pytest.mark.parametrize('legacy,current', sorted(LEGACY_OBS_INFO_KEYS.items()))
def test_an_obs_info_setting_both_spellings_is_an_error(legacy, current):
    with pytest.raises(ValueError) as excinfo:
        normalise_obs_info({legacy: ['a'], current: ['b']})
    assert legacy in str(excinfo.value) and current in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.parametrize('legacy,current', sorted(LEGACY_PREDICTION_INFO_KEYS.items()))
def test_a_prediction_info_setting_both_spellings_is_an_error(legacy, current):
    with pytest.raises(ValueError) as excinfo:
        normalise_prediction_info({legacy: ['a'], current: ['b']})
    assert legacy in str(excinfo.value) and current in str(excinfo.value)


@pytest.mark.unit
def test_normalising_does_not_mutate_the_caller_s_dict():
    original = {'obs_names': ['a'], 'names_for_plotting': ['A']}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        normalise_obs_info(original)
    assert original == {'obs_names': ['a'], 'names_for_plotting': ['A']}


@pytest.mark.unit
def test_one_warning_per_legacy_key_not_per_item():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        normalise_obs_info({'obs_names': ['a', 'b', 'c'], 'names_for_plotting': ['A', 'B', 'C']})
    messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(messages) == 2, messages


@pytest.mark.unit
def test_a_prediction_infos_flat_names_become_operand_lists():
    """The rename carries a shape change no table can express.

    ``operands`` is a list *per item* on obs_info, so a prediction_info that spelled it
    ``names`` -- flat -- has to be reshaped, or one key would name two shapes.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        out = normalise_prediction_info({'names': ['main/y', 'main/z']})
    assert out['operands'] == [['main/y'], ['main/z']]


@pytest.mark.unit
def test_none_prediction_info_passes_through():
    """It is genuinely optional -- CUFLynx hands ParamID `prediction_info=None`."""
    assert normalise_prediction_info(None) is None


@pytest.mark.unit
def test_the_accessors_read_a_canonical_prediction_info():
    """One set of accessors for both dicts is the point of mirroring their key names."""
    pred = {'data_item_names': ['y'], 'item_names_for_plotting': ['Y (max)'],
            'trace_names_for_plotting': ['Y']}
    assert obs_item_names(pred) == ['y']
    assert obs_item_labels(pred) == ['Y (max)']
    assert obs_trace_labels(pred) == ['Y']
