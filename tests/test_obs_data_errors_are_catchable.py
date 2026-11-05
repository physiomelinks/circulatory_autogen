"""The obs_data guards raise, so an application embedding the parser can report them.

``print()`` then ``exit()`` is fine in a script: the message lands on the terminal the user
is already looking at. Anywhere else it is the worst of both worlds. ``exit()`` raises
``SystemExit``, which derives from ``BaseException`` rather than ``Exception``, so it goes
straight through the ``except Exception`` an embedding application wraps a parse in -- and
the explanation, being a bare ``print``, goes to that application's stdout where nobody is
reading.

That is not hypothetical. CUFLynx's ``POST /api/protocol/run`` returned a bare "Internal
Server Error" on an SN_full study whose series carry ``obs_dt = 1e-4`` while the app
evaluates at its default ``dt = 0.01``. The sentence explaining it existed, and reached only
the uvicorn log.

These drive the real ``get_ground_truth_values`` rather than a copy of its four lines, so
they fail if the guard is changed. Each asserts three things: it raises, what it raises is an
ordinary ``Exception`` and not ``SystemExit``, and the message carries the numbers needed to
act rather than only the names of the keys involved.
"""
import numpy as np
import pandas as pd
import pytest

from libcuflynx.parsers.PrimitiveParsers import (ObsAndParamDataParser, ObsDataError,
                                                 _obs_item_label, _series_is_scored)

#: The columns ``get_ground_truth_values`` reads before it reaches the obs_dt guard. Kept
#: minimal on purpose: a full study would need a model, a protocol and a params file to
#: reach four lines of validation.
def series_row(name='Vgt_{B0}', obs_dt=1e-4, **extra):
    row = {
        'data_type': 'series', 'data_item_name': name, 'operands': ['soma/V'],
        'operation': 'series', 'value': [0.0, 1.0], 'std': 1.0, 'weight': 1.0,
        'prob_dist_params': None, 'obs_dt': obs_dt,
    }
    if obs_dt is None:
        del row['obs_dt']
    row.update(extra)
    return row


def run(rows, dt, output_dir=None):
    """``output_dir`` matters only past the guard: the function writes ground_truth npy
    files there once it has validated. The raising tests never reach it."""
    obs_info = {'num_obs': len(rows),
                'data_types': [r['data_type'] for r in rows]}
    return ObsAndParamDataParser().get_ground_truth_values(
        pd.DataFrame(rows), obs_info, output_dir, dt)


def test_a_dt_finer_than_the_solver_step_raises_rather_than_exiting():
    """The SN_full case: obs_dt 1e-4 against an application's default dt of 0.01."""
    with pytest.raises(ObsDataError) as excinfo:
        run([series_row()], dt=0.01)
    assert isinstance(excinfo.value, Exception)
    assert not isinstance(excinfo.value, SystemExit), \
        'SystemExit passes through `except Exception`, which is the bug'


def test_the_message_carries_both_numbers_and_the_offending_item():
    with pytest.raises(ObsDataError) as excinfo:
        run([series_row()], dt=0.01)
    message = str(excinfo.value)
    assert '0.01' in message, 'the solver dt the caller used must appear'
    assert '0.0001' in message, 'the obs_dt that conflicts with it must appear'
    assert 'Vgt_{B0}' in message, 'and which item it came from'


def test_many_offenders_are_summarised_not_listed_in_full():
    with pytest.raises(ObsDataError) as excinfo:
        run([series_row(f'Vgt_{{s{i}}}') for i in range(9)], dt=0.01)
    message = str(excinfo.value)
    assert 'Vgt_{s0}' in message
    assert 'and 5 more' in message, 'four named, the rest counted'
    assert 'Vgt_{s8}' not in message


def test_only_the_items_that_actually_conflict_are_named():
    with pytest.raises(ObsDataError) as excinfo:
        run([series_row('Vgt_{fine}', 1e-4), series_row('Vgt_{coarse}', 0.5)], dt=0.01)
    message = str(excinfo.value)
    assert 'Vgt_{fine}' in message
    assert 'Vgt_{coarse}' not in message, 'a series coarser than dt is not the problem'


def test_a_series_without_obs_dt_raises_and_names_itself():
    with pytest.raises(ObsDataError, match=r"Vgt_\{nodt\}"):
        run([series_row('Vgt_{nodt}', obs_dt=None)], dt=0.01)


def test_a_dt_equal_to_obs_dt_is_fine(tmp_path):
    """The boundary is ``<``, not ``<=``: sampling the solver exactly at the data's spacing
    is the normal, intended case and must not trip the guard."""
    out = run([series_row()], dt=1e-4, output_dir=str(tmp_path))
    assert np.allclose(out['obs_dt'], [1e-4])


def test_constants_alone_do_not_trip_it(tmp_path):
    """An obs_data with no series has nothing to compare a dt against."""
    out = run([{'data_type': 'constant', 'data_item_name': 'V_{max}', 'operands': ['soma/V'],
                'operation': 'max', 'value': 1.0, 'std': 1.0, 'weight': 1.0,
                'prob_dist_params': None}], dt=0.01, output_dir=str(tmp_path))
    assert len(out['obs_dt']) == 0


def test_the_label_falls_back_when_an_item_has_no_name():
    df = pd.DataFrame([{'data_type': 'series', 'operands': ['soma/V_sensed']}])
    assert _obs_item_label(df, 0) == 'soma/V_sensed'
    bare = pd.DataFrame([{'data_type': 'series'}])
    assert _obs_item_label(bare, 0) == 'item 0'
    assert _obs_item_label(bare, 7) == 'item 7', 'an out-of-range row must not raise'


# --------------------------------------------------- only a scored series constrains dt

def test_an_empty_zero_weighted_series_does_not_block_the_study(tmp_path):
    """The SN_full case exactly: eight placeholder series at obs_dt 1e-4, weight 0 and no
    samples, in a study whose scored observables are all constants. They cannot be compared
    against anything, so they must not stop the evaluation."""
    rows = [series_row(f'Vgt_{{s{i}}}', 1e-4, weight=0.0, value=[]) for i in range(8)]
    rows.append({'data_type': 'constant', 'data_item_name': 'V_{max}', 'operands': ['soma/V'],
                 'operation': 'max', 'value': 1.0, 'std': 1.0, 'weight': 1.0,
                 'prob_dist_params': None})
    out = run(rows, dt=0.01, output_dir=str(tmp_path))
    assert len(out['obs_dt']) == 8, 'obs_dt stays parallel to the series for plot_outputs'


def test_a_scored_series_still_constrains_dt():
    """The guard must not have been softened into uselessness."""
    with pytest.raises(ObsDataError, match=r"Vgt_\{real\}"):
        run([series_row('Vgt_{real}', 1e-4, weight=1.0, value=[0.0, 1.0])], dt=0.01)


def test_a_weighted_but_empty_series_does_not_constrain_dt(tmp_path):
    run([series_row('Vgt_{empty}', 1e-4, weight=1.0, value=[])], dt=0.01,
        output_dir=str(tmp_path))


def test_a_zero_weighted_series_with_samples_does_not_constrain_dt(tmp_path):
    run([series_row('Vgt_{off}', 1e-4, weight=0.0, value=[0.0, 1.0])], dt=0.01,
        output_dir=str(tmp_path))


def test_only_the_scored_offender_is_named():
    with pytest.raises(ObsDataError) as excinfo:
        run([series_row('Vgt_{placeholder}', 1e-4, weight=0.0, value=[]),
             series_row('Vgt_{real}', 1e-4, weight=1.0, value=[0.0, 1.0])], dt=0.01)
    message = str(excinfo.value)
    assert 'Vgt_{real}' in message
    assert 'Vgt_{placeholder}' not in message


def test_series_is_scored_reads_weight_and_samples():
    df = pd.DataFrame([
        series_row('a', weight=1.0, value=[0.0]),
        series_row('b', weight=0.0, value=[0.0]),
        series_row('c', weight=1.0, value=[]),
        series_row('d', weight=None, value=[0.0]),
    ])
    assert [_series_is_scored(df, i) for i in range(4)] == [True, False, False, True]
