"""An observable built from an observable in another experiment or sub-experiment (#466, #127).

An ``operation_kwargs`` value naming another ``data_item_name`` is how one observable is built
from others. Until now the table those names resolve against -- ``temp_results`` -- was cleared
once per (experiment, sub-experiment), so a reference could only ever see the segment being
evaluated: the quantity you actually know, "how much did the forcing move the peak", could not
be written down at all. The table now spans one whole cost evaluation, and the segments are
visited in order, so a reference reaches backwards across experiments.

The fixture is the smallest thing that shows it: the same Lotka-Volterra model run unforced
(experiment 0) and forced (experiment 1), with a third item in experiment 1 that differences the
two peaks.
"""
import json
import os

import numpy as np
import pytest

from libcuflynx.param_id.paramID import CVS0DParamID

_TESTS = os.path.dirname(__file__)
_INPUTS = os.path.join(_TESTS, 'test_inputs')
_OBS = os.path.join(_INPUTS, 'Lotka_Volterra_forced_cross_exp_obs_data.json')

UNFORCED = 'peak prey, unforced'
FORCED = 'peak prey, forced'
RESPONSE = 'peak prey, forcing response'


def _param_id(output_dir, obs_path=_OBS):
    return CVS0DParamID(
        os.path.join(_INPUTS, 'Lotka_Volterra_forced.cellml'), 'cellml', 'genetic_algorithm',
        False, 'Lotka_Volterra_forced',
        params_for_id_path=os.path.join(_INPUTS, 'Lotka_Volterra_forced_params_for_id.csv'),
        param_id_obs_path=obs_path,
        sim_time=5.0, pre_time=0.0,
        solver_info={'solver': 'CVODE_myokit', 'MaximumStep': 0.05,
                     'MaximumNumberOfSteps': 50000},
        dt=0.01, optimiser_options=None, do_ad=False, DEBUG=False,
        param_id_output_dir=str(output_dir), resources_dir=_INPUTS)


def _midpoint(engine):
    mins = np.asarray(engine.param_id_info['param_mins'], dtype=float)
    maxs = np.asarray(engine.param_id_info['param_maxs'], dtype=float)
    return 0.5 * (mins + maxs)


@pytest.mark.integration
@pytest.mark.solver
def test_an_item_can_difference_two_observables_from_different_experiments(temp_output_dir):
    """The load-bearing assertion: the cross-experiment item equals the difference of the two
    per-experiment peaks, and those peaks are genuinely different numbers.

    The second half matters as much as the first. If the reference silently collapsed to one
    segment -- which is exactly what used to happen when two items shared a name -- both
    operands would be the same value and the difference would be 0.0 while still 'working'.
    """
    engine = _param_id(temp_output_dir).param_id
    cost, _obs, _pred = engine.get_cost_obs_and_pred_from_params(_midpoint(engine))

    results = engine.temp_results
    assert set(results) >= {UNFORCED, FORCED, RESPONSE}, sorted(results)

    unforced, forced = float(results[UNFORCED]), float(results[FORCED])
    assert forced != pytest.approx(unforced, rel=1e-3), (
        'the two experiments produced the same peak, so this fixture cannot tell a working '
        'cross-experiment reference from a collapsed one')
    # calculate_two_observable_difference returns subtract_from - subtract_this
    assert float(results[RESPONSE]) == pytest.approx(forced - unforced, rel=1e-9)
    assert np.isfinite(cost)


@pytest.mark.integration
@pytest.mark.solver
def test_the_referenced_value_is_the_referenced_experiment_s_own(temp_output_dir):
    """Not merely 'some number': each peak must be the max of *its* experiment's trace.

    Every data_item is evaluated against every segment, and which one counts is decided later by
    the zeroed weights -- so a reference that read whatever was computed last would pick up this
    item's operation applied to the wrong experiment's trace, and still look plausible.
    """
    engine = _param_id(temp_output_dir).param_id
    params = _midpoint(engine)
    engine.get_cost_obs_and_pred_from_params(params)
    results = dict(engine.temp_results)

    _cost, operands_by_segment, _pred = engine.get_cost_obs_and_pred_from_params(params)
    obs = engine.obs_info
    names = list(obs['data_item_names'])
    num_sub = engine.protocol_info['num_sub_per_exp']
    for name in (UNFORCED, FORCED):
        idx = names.index(name)
        exp = int(obs['experiment_idxs'][idx])
        sub = int(obs['subexperiment_idxs'][idx])
        flat = int(sum(num_sub[:exp]) + sub)
        trace = np.asarray(operands_by_segment[flat][idx][0], dtype=float)
        assert float(results[name]) == pytest.approx(float(trace.max()), rel=1e-9), (
            f'{name} did not come from experiment {exp}, sub-experiment {sub}')


@pytest.mark.integration
@pytest.mark.solver
def test_a_forward_reference_says_which_item_to_move(tmp_path, temp_output_dir):
    """A reference to an item that has not been computed yet used to fall through as a plain
    string, so the operation received `'peak prey, forced'` and raised a TypeError on `str - str`
    -- or, for an operation that tolerates it, quietly produced a number."""
    with open(_OBS, encoding='utf-8-sig') as fh:
        doc = json.load(fh)
    # put the difference first, so both the items it names come after it
    doc['data_items'] = [doc['data_items'][2], doc['data_items'][0], doc['data_items'][1]]
    bad = tmp_path / 'forward_reference_obs_data.json'
    bad.write_text(json.dumps(doc, indent=1), encoding='utf-8')

    engine = _param_id(temp_output_dir, obs_path=str(bad)).param_id
    with pytest.raises(ValueError) as excinfo:
        engine.get_cost_obs_and_pred_from_params(_midpoint(engine))
    message = str(excinfo.value)
    assert 'has not been computed yet' in message
    assert RESPONSE in message


@pytest.mark.unit
def test_the_cross_segment_items_are_reported_for_refusal():
    """The FSA and CasADi arms build each observable from one sub-experiment's operands, so they
    cannot differentiate a cross-segment reference. They refuse rather than return a gradient
    for a different feature than the cost -- which needs this to spot one."""
    from libcuflynx.param_id.paramID import ParamID

    pid = ParamID.__new__(ParamID)
    pid.obs_info = {
        'data_item_names': ['a', 'b', 'diff'],
        'experiment_idxs': [0, 1, 1],
        'subexperiment_idxs': [0, 0, 0],
        'operation_kwargs': [{}, {}, {'subtract_this': 'a', 'subtract_from': 'b'}],
    }
    assert pid.cross_segment_reference_items() == [('diff', 'a')]
    with pytest.raises(NotImplementedError) as excinfo:
        pid._refuse_cross_segment_references('The Myokit CVODES FSA gradient')
    assert 'finite differences' in str(excinfo.value)

    # a reference that stays inside one segment is ordinary and must not be refused
    pid.obs_info['experiment_idxs'] = [1, 1, 1]
    assert pid.cross_segment_reference_items() == []
    pid._refuse_cross_segment_references('anything')


@pytest.mark.integration
@pytest.mark.solver
def test_a_standalone_segment_evaluation_gets_its_own_table(temp_output_dir):
    """``get_cost_from_operands`` works for a caller that did not come through ``get_cost``.

    ``temp_results`` used to be created *only* by the loop in
    ``get_cost_obs_and_pred_from_params``, while ``get_cost_from_operands`` unconditionally
    enters ``evaluating_segment`` -- which tells ``get_obs_output_dict`` not to reset the table.
    So every caller evaluating one segment at a time raised ``AttributeError:
    'ParamID' object has no attribute 'temp_results'`` on the first item it recorded.

    Not a hypothetical entry point: CUFLynx's ``obs_cost`` scores an emulator's predictions
    through exactly this call, and ``evaluating_segment``'s own docstring lists the gradient
    backends, ``plot_outputs`` and CUFLynx as callers it must not break. It surfaced as a
    silently missing emulator cost rather than an error, because that caller reads a failure
    here as "no cost available".
    """
    engine = _param_id(temp_output_dir).param_id
    _cost, operands, _pred = engine.get_cost_obs_and_pred_from_params(_midpoint(engine))

    # A fresh engine, so nothing has set the table up -- the state a standalone caller is in.
    fresh = _param_id(temp_output_dir).param_id
    assert not hasattr(fresh, 'temp_results') or fresh.temp_results == {}

    cost = fresh.get_cost_from_operands(operands[0], exp_idx=0, sub_idx=0)
    assert np.isfinite(cost)
    # And it recorded into a table of its own rather than raising.
    assert UNFORCED in fresh.temp_results


@pytest.mark.integration
@pytest.mark.solver
def test_a_standalone_call_does_not_read_the_previous_call_s_values(temp_output_dir):
    """Each standalone segment evaluation starts from an empty table.

    Accumulating across segments belongs to one cost evaluation (#466). A caller evaluating
    single segments in a loop of its own must not have a reference resolve against whatever the
    *previous* call left behind -- that is a stale number presented as this segment's.
    """
    engine = _param_id(temp_output_dir).param_id
    _cost, operands, _pred = engine.get_cost_obs_and_pred_from_params(_midpoint(engine))

    fresh = _param_id(temp_output_dir).param_id
    fresh.get_cost_from_operands(operands[0], exp_idx=0, sub_idx=0)
    fresh.temp_results['a name no data_item has'] = 123.0

    fresh.get_cost_from_operands(operands[0], exp_idx=0, sub_idx=0)
    assert 'a name no data_item has' not in fresh.temp_results
