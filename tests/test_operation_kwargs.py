"""Unit tests for the per-``data_item`` ``operation_kwargs`` contract (issue #304).

``operation_kwargs`` is the public obs_data.json field the CUFLynx GUI writes when a user tunes
an operation func's keyword arguments. These tests lock in how it is validated and forwarded --
``operation(*operands, **operation_kwargs)`` -- so calibration, MCMC/UQ and sensitivity analysis
stay in step. No model, solver or MPI needed.
"""
import json
import os
import textwrap

import numpy as np
import pytest

from libcuflynx.param_id.operation_funcs import (
    RESERVED_OPERATION_KWARGS,
    check_operation_kwargs,
    get_operation_kwarg_spec,
    resolve_operation_kwargs,
    validate_operation_kwargs,
)
from libcuflynx.utilities.obs_data_helpers import obs_item_names
from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser, scriptFunctionParser


# ---------------------------------------------------------------------------
# Sample operation funcs (mirroring the shapes that exist in the real registries)
# ---------------------------------------------------------------------------

def op_windowed(x, start_frac=0.0, end_frac=1.0, series_output=False):
    """Keyword-argument op: one operand + tunables (like ``mean_in_range``)."""
    if series_output:
        return x
    start_idx = int(start_frac * (len(x) - 1))
    end_idx = int(end_frac * (len(x) - 1))
    return float(np.mean(x[start_idx:end_idx]))


def op_index(x, window=10, series_output=False):
    """Op whose kwarg default is an ``int`` and is used as a slice index."""
    if series_output:
        return x
    return float(np.mean(x[:window]))


def op_two_operands(x1, x2, scale=1.0):
    return (x1 - x2) * scale


def op_var_kwargs(x=None, series_output=False, **kwargs):
    """``**kwargs`` op (like ``calculate_two_observable_difference``): accepts any key."""
    if series_output:
        return x
    return kwargs["pred2"] - kwargs["pred1"]


def op_no_kwargs(x):
    return float(np.max(x))


_X = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])


# ---------------------------------------------------------------------------
# Signature introspection
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_get_operation_kwarg_spec_separates_operands_from_kwargs():
    accepted, from_operands, accepts_any = get_operation_kwarg_spec(op_windowed)
    assert accepted == ['x', 'start_frac', 'end_frac', 'series_output']
    assert from_operands == ['x']          # only 'x' has no default -> filled from operands
    assert accepts_any is False


@pytest.mark.unit
def test_get_operation_kwarg_spec_detects_var_keyword():
    _accepted, _from_operands, accepts_any = get_operation_kwarg_spec(op_var_kwargs)
    assert accepts_any is True


# ---------------------------------------------------------------------------
# Default / empty case -- the overwhelmingly common one
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize('raw', [{}, None, float('nan'), np.nan, 'not a dict', 0.0])
def test_missing_or_empty_operation_kwargs_resolves_to_empty_dict(raw):
    """The schema default is ``{}``; a NaN/None left by a partially filled obs_data file must
    also be tolerated rather than raising ``argument after ** must be a mapping``."""
    kwargs = resolve_operation_kwargs(raw, op_windowed, operation_name='op_windowed')
    assert kwargs == {}
    # and the resulting call is the plain, default-valued one
    assert op_windowed(_X, **kwargs) == pytest.approx(float(np.mean(_X[:9])))


@pytest.mark.unit
def test_op_with_no_kwargs_at_all_still_works_with_empty_operation_kwargs():
    assert resolve_operation_kwargs({}, op_no_kwargs, operation_name='op_no_kwargs') == {}


# ---------------------------------------------------------------------------
# Scalar kwargs
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_scalar_kwargs_are_forwarded():
    raw = {'start_frac': 0.5, 'end_frac': 1.0}
    kwargs = resolve_operation_kwargs(raw, op_windowed, operation_name='op_windowed',
                                      num_operands=1)
    assert kwargs == {'start_frac': 0.5, 'end_frac': 1.0}
    assert op_windowed(_X, **kwargs) == pytest.approx(float(np.mean(_X[4:9])))
    # the caller's dict is never mutated
    assert raw == {'start_frac': 0.5, 'end_frac': 1.0}


@pytest.mark.unit
def test_kwargs_dict_is_a_copy_not_the_obs_info_entry():
    raw = {'start_frac': 0.5}
    kwargs = resolve_operation_kwargs(raw, op_windowed, operation_name='op_windowed')
    kwargs['start_frac'] = 0.9
    assert raw['start_frac'] == 0.5


# ---------------------------------------------------------------------------
# String-valued kwargs (reference to an earlier observable)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_string_kwarg_matching_an_earlier_observable_is_substituted():
    temp_results = {'v_{ARmean}': 2.0, 'v_{ARmax}': 5.0}
    raw = {'pred1': 'v_{ARmean}', 'pred2': 'v_{ARmax}'}
    kwargs = resolve_operation_kwargs(raw, op_var_kwargs, operation_name='op_var_kwargs',
                                      temp_results=temp_results)
    assert kwargs == {'pred1': 2.0, 'pred2': 5.0}
    assert op_var_kwargs(**kwargs) == pytest.approx(3.0)


@pytest.mark.unit
def test_string_kwarg_matching_nothing_is_passed_through_unchanged():
    """Plain string options must keep working; only names of earlier observables substitute."""
    kwargs = resolve_operation_kwargs({'pred1': 'literal'}, op_var_kwargs,
                                      operation_name='op_var_kwargs',
                                      temp_results={'something_else': 1.0})
    assert kwargs == {'pred1': 'literal'}


@pytest.mark.unit
def test_substituted_array_value_is_not_coerced():
    series = np.arange(5.0)
    kwargs = resolve_operation_kwargs({'window': 'earlier'}, op_index,
                                      operation_name='op_index',
                                      temp_results={'earlier': series})
    assert kwargs['window'] is series


# ---------------------------------------------------------------------------
# series_output branch
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_series_output_is_reserved_and_rejected():
    assert 'series_output' in RESERVED_OPERATION_KWARGS
    with pytest.raises(ValueError) as excinfo:
        resolve_operation_kwargs({'series_output': True}, op_windowed,
                                 operation_name='op_windowed', data_item_name='u_{AR}')
    msg = str(excinfo.value)
    assert 'series_output' in msg
    assert "u_{AR}" in msg
    assert 'op_windowed' in msg


@pytest.mark.unit
def test_resolved_kwargs_compose_with_the_series_output_call():
    """The plotting path calls ``func(*operands, series_output=True, **kwargs)``; the resolved
    kwargs must never contain series_output so that call cannot get it twice."""
    kwargs = resolve_operation_kwargs({'start_frac': 0.5}, op_windowed,
                                      operation_name='op_windowed', num_operands=1)
    assert 'series_output' not in kwargs
    out = op_windowed(_X, series_output=True, **kwargs)
    assert np.array_equal(out, _X)


# ---------------------------------------------------------------------------
# Error path: unknown / colliding keys
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_unknown_kwarg_raises_an_actionable_error():
    with pytest.raises(ValueError) as excinfo:
        resolve_operation_kwargs({'start_fraction': 0.5}, op_windowed,
                                 operation_name='op_windowed', data_item_name='u_{AR}',
                                 num_operands=1)
    msg = str(excinfo.value)
    assert 'start_fraction' in msg          # the offending key
    assert 'op_windowed' in msg             # the operation func
    assert "u_{AR}" in msg                  # the data item
    assert 'start_frac' in msg              # the accepted keys / close-match suggestion
    assert 'operation_kwargs' in msg        # where to fix it


@pytest.mark.unit
def test_unknown_kwarg_suggests_a_close_match():
    with pytest.raises(ValueError, match="Did you mean 'end_frac'"):
        resolve_operation_kwargs({'end_frak': 1.0}, op_windowed, operation_name='op_windowed')


@pytest.mark.unit
def test_kwarg_already_filled_from_operands_raises():
    with pytest.raises(ValueError) as excinfo:
        resolve_operation_kwargs({'x': [1.0]}, op_windowed, operation_name='op_windowed',
                                 data_item_name='u_{AR}', num_operands=1)
    assert 'operands' in str(excinfo.value)


@pytest.mark.unit
def test_second_operand_name_is_only_reserved_when_that_operand_is_supplied():
    # only one operand given -> x2 is still a legitimate keyword argument
    assert resolve_operation_kwargs({'x2': 3.0}, op_two_operands, operation_name='op_two_operands',
                                    num_operands=1) == {'x2': 3.0}
    # both operands given -> x2 comes from operands, so passing it again is an error
    with pytest.raises(ValueError):
        resolve_operation_kwargs({'x2': 3.0}, op_two_operands, operation_name='op_two_operands',
                                 num_operands=2)


@pytest.mark.unit
def test_non_string_key_raises():
    with pytest.raises(ValueError, match='keys must be strings'):
        resolve_operation_kwargs({3: 0.5}, op_windowed, operation_name='op_windowed')


@pytest.mark.unit
def test_var_kwargs_func_accepts_any_key():
    """An op that declares **kwargs opts out of the unknown-key check."""
    kwargs = resolve_operation_kwargs({'anything_at_all': 1.0}, op_var_kwargs,
                                      operation_name='op_var_kwargs')
    assert kwargs == {'anything_at_all': 1.0}


# ---------------------------------------------------------------------------
# Basic type coercion
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_integral_float_is_coerced_to_int_when_the_default_is_an_int():
    """JSON has one number type, so a GUI can write 3.0 where an int index is wanted."""
    kwargs = resolve_operation_kwargs({'window': 3.0}, op_index, operation_name='op_index',
                                      num_operands=1)
    assert isinstance(kwargs['window'], int) and kwargs['window'] == 3
    assert op_index(_X, **kwargs) == pytest.approx(1.0)


@pytest.mark.unit
def test_int_is_coerced_to_float_when_the_default_is_a_float():
    kwargs = resolve_operation_kwargs({'start_frac': 0, 'end_frac': 1}, op_windowed,
                                      operation_name='op_windowed', num_operands=1)
    assert isinstance(kwargs['start_frac'], float)
    assert isinstance(kwargs['end_frac'], float)


@pytest.mark.unit
def test_non_integral_float_is_not_coerced_to_int():
    kwargs = resolve_operation_kwargs({'window': 3.5}, op_index, operation_name='op_index')
    assert kwargs['window'] == 3.5


@pytest.mark.unit
def test_bools_and_other_types_are_not_coerced():
    kwargs = resolve_operation_kwargs({'anything': True, 'other': None, 'l': [1, 2]},
                                      op_var_kwargs, operation_name='op_var_kwargs')
    assert kwargs == {'anything': True, 'other': None, 'l': [1, 2]}


# ---------------------------------------------------------------------------
# Eager validation over a whole obs_info
# ---------------------------------------------------------------------------

def _obs_info(operations, operands, operation_kwargs, names):
    return {'operations': operations, 'operands': operands,
            'operation_kwargs': operation_kwargs, 'data_item_names': names}


@pytest.mark.unit
def test_validate_operation_kwargs_flags_the_offending_data_item():
    obs_info = _obs_info(['op_windowed', 'op_windowed'],
                         [['main/x'], ['main/x']],
                         [{'start_frac': 0.5}, {'start_frak': 0.5}],
                         ['good', 'bad_item'])
    with pytest.raises(ValueError) as excinfo:
        validate_operation_kwargs(obs_info, {'op_windowed': op_windowed})
    assert 'bad_item' in str(excinfo.value)


@pytest.mark.unit
def test_validate_operation_kwargs_accepts_a_valid_obs_info():
    obs_info = _obs_info(['op_windowed', None, 'op_var_kwargs'],
                         [['main/x'], ['main/y'], []],
                         [{'start_frac': 0.5}, {}, {'pred1': 'good', 'pred2': 'good'}],
                         ['good', 'plain', 'delta'])
    validate_operation_kwargs(obs_info, {'op_windowed': op_windowed,
                                         'op_var_kwargs': op_var_kwargs})


@pytest.mark.unit
def test_validate_operation_kwargs_skips_ops_not_yet_registered():
    """A user func may be registered later via add_user_operation_func; eager validation must not
    pre-emptively fail on it."""
    obs_info = _obs_info(['not_registered_yet'], [['main/x']], [{'foo': 1.0}], ['item'])
    validate_operation_kwargs(obs_info, {'op_windowed': op_windowed})


@pytest.mark.unit
def test_validate_operation_kwargs_tolerates_empty_obs_info():
    validate_operation_kwargs(None, {})
    validate_operation_kwargs({}, {})


# ---------------------------------------------------------------------------
# The contract helpers must not be registered as observable operations
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_contract_helpers_are_not_registered_as_operations():
    ops = scriptFunctionParser().get_operation_funcs_dict('numpy')
    for name in ['resolve_operation_kwargs', 'check_operation_kwargs',
                 'validate_operation_kwargs', 'get_operation_kwarg_spec']:
        assert name not in ops
    assert 'mean' in ops        # sanity: the real operations are still registered


# ---------------------------------------------------------------------------
# Regression: a user op with kwargs driven from an obs_data.json file (issue #304 request 4)
# ---------------------------------------------------------------------------

_EXTERNAL_OPS = textwrap.dedent('''
    from libcuflynx.param_id.operation_funcs import series_to_constant
    from libcuflynx.param_id.differentiable import differentiable
    from libcuflynx.param_id.math_backend import make_math_backend
    mb = make_math_backend("numpy")

    @differentiable
    @series_to_constant
    def scaled_mean(x, scale=1.0, offset=0.0, series_output=False):
        if series_output:
            return x
        return mb.mean(x) * scale + offset
''')


def _obs_data_json(operation_kwargs_for_second_item):
    return {
        "protocol_info": {"pre_times": [0.0], "sim_times": [[1.0]], "params_to_change": {}},
        "data_items": [
            {"data_item_name": "x plain", "trace_name_for_plotting": "x_{plain}", "data_type": "constant",
             "operation": "scaled_mean", "operands": ["main/x"], "unit": "dimensionless",
             "weight": 1.0, "value": 1.0, "std": 0.1},
            {"data_item_name": "x scaled", "trace_name_for_plotting": "x_{scaled}", "data_type": "constant",
             "operation": "scaled_mean", "operands": ["main/x"], "unit": "dimensionless",
             "operation_kwargs": operation_kwargs_for_second_item,
             "weight": 1.0, "value": 1.0, "std": 0.1},
        ],
    }


def _parse_obs_info(obs_data_dict, tmp_path):
    parser = ObsAndParamDataParser()
    parsed = parser.parse_obs_data_json(obs_data_dict=obs_data_dict, pre_time=0.0, sim_time=1.0)
    return parser.process_obs_info(gt_df=parsed["gt_df"], output_dir=str(tmp_path), dt=0.01)


@pytest.mark.unit
def test_external_user_op_receives_operation_kwargs_from_obs_data_json(tmp_path):
    """End-to-end for the field: obs_data.json -> parser -> obs_info -> operation call, using a
    user op loaded through ``operation_funcs_external_path`` (#303)."""
    op_path = os.path.join(str(tmp_path), 'my_ops.py')
    with open(op_path, 'w') as fh:
        fh.write(_EXTERNAL_OPS)
    ops = scriptFunctionParser(operation_funcs_external_path=op_path).get_operation_funcs_dict('numpy')
    assert 'scaled_mean' in ops

    obs_info = _parse_obs_info(_obs_data_json({"scale": 3.0, "offset": 1.0}), tmp_path)

    # the parser gives the first (kwarg-free) item the schema default {}
    assert obs_info["operation_kwargs"][0] == {}
    assert obs_info["operation_kwargs"][1] == {"scale": 3.0, "offset": 1.0}

    # ...and it validates cleanly and forwards through to the external func
    validate_operation_kwargs(obs_info, ops)
    x = np.array([1.0, 2.0, 3.0])
    results = []
    for idx in range(len(obs_info["operations"])):
        func = ops[obs_info["operations"][idx]]
        kwargs = resolve_operation_kwargs(
            obs_info["operation_kwargs"][idx], func,
            operation_name=obs_info["operations"][idx],
            data_item_name=obs_item_names(obs_info)[idx],
            temp_results={}, num_operands=1)
        results.append(func(x, **kwargs))
    assert results[0] == pytest.approx(2.0)
    assert results[1] == pytest.approx(2.0 * 3.0 + 1.0)


@pytest.mark.unit
def test_stale_operation_kwargs_in_obs_data_json_fails_with_a_clear_error(tmp_path):
    op_path = os.path.join(str(tmp_path), 'my_ops.py')
    with open(op_path, 'w') as fh:
        fh.write(_EXTERNAL_OPS)
    ops = scriptFunctionParser(operation_funcs_external_path=op_path).get_operation_funcs_dict('numpy')

    obs_info = _parse_obs_info(_obs_data_json({"scaling": 3.0}), tmp_path)
    with pytest.raises(ValueError) as excinfo:
        validate_operation_kwargs(obs_info, ops)
    msg = str(excinfo.value)
    assert 'scaling' in msg and 'scaled_mean' in msg and 'x scaled' in msg
    assert 'scale' in msg


@pytest.mark.unit
def test_shipped_extra_ops_obs_data_json_still_validates(tmp_path):
    """resources/3compartment_extra_ops_obs_data.json uses operation_kwargs with a **kwargs op;
    the new validation must not reject it."""
    resources_dir = os.path.join(os.path.dirname(__file__), '..', 'resources')
    obs_path = os.path.join(resources_dir, '3compartment_extra_ops_obs_data.json')
    with open(obs_path, encoding='utf-8-sig') as fh:
        data_items = json.load(fh)

    parser = ObsAndParamDataParser()
    parsed = parser.parse_obs_data_json(obs_data_dict=data_items, pre_time=0.0, sim_time=1.0)
    obs_info = parser.process_obs_info(gt_df=parsed["gt_df"], output_dir=str(tmp_path), dt=0.01)
    validate_operation_kwargs(obs_info, scriptFunctionParser().get_operation_funcs_dict('numpy'))


@pytest.mark.unit
def test_check_operation_kwargs_is_a_noop_for_empty_kwargs():
    check_operation_kwargs({}, op_windowed, 'op_windowed')
    check_operation_kwargs(None, op_windowed, 'op_windowed')
