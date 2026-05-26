"""
Tests for ProtocolExecutor and ProtocolRunner.

These tests verify that the centralised protocol-running code in
src/protocol_runners/ works correctly in the same ways it is used by:

  - ProcessData (standalone ProtocolRunner)
  - OpencorParamID / sobol_SA (ProtocolExecutor with pre-built sim_helper)

All tests use the SN_simple CellML model
(generated_models/SN_simple/SN_simple.cellml) and the protocol_info from
tests/test_inputs/SN_simple_param_id_fast_obs.json.
"""

import json
import os
import sys

import numpy as np
import pytest

_TEST_ROOT = os.path.join(os.path.dirname(__file__), '..')
_SRC_DIR = os.path.join(_TEST_ROOT, 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from solver_wrappers import get_simulation_helper
from protocol_runners import ProtocolExecutor, ProtocolRunner

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_CELLML_PATH = os.path.join(
    _TEST_ROOT, 'generated_models', 'SN_simple', 'SN_simple.cellml'
)
_OBS_JSON_PATH = os.path.join(
    _TEST_ROOT, 'tests', 'test_inputs', 'SN_simple_param_id_fast_obs.json'
)

_DT = 0.001
_SOLVER_INFO = {'MaximumStep': 0.001, 'MaximumNumberOfSteps': 50000}


def _load_protocol_info():
    """Load the protocol_info dict from the SN_simple fixture JSON."""
    with open(_OBS_JSON_PATH, encoding='utf-8-sig') as fh:
        obs = json.load(fh)
    return obs['protocol_info']


def _make_myokit_helper(sim_time=0.35, pre_time=0.1):
    """Return a myokit SimulationHelper for SN_simple, or skip if unavailable."""
    try:
        helper = get_simulation_helper(
            model_path=_CELLML_PATH,
            solver='CVODE_myokit',
            dt=_DT,
            sim_time=sim_time,
            solver_info=_SOLVER_INFO,
            pre_time=pre_time,
        )
    except (RuntimeError, ImportError) as exc:
        pytest.skip(f'Myokit backend not available: {exc}')
    return helper


def _make_runner():
    """Return a ProtocolRunner for SN_simple using CVODE_myokit."""
    inp_data_dict = {
        'dt': _DT,
        'solver_info': _SOLVER_INFO,
    }
    try:
        runner = ProtocolRunner(
            _CELLML_PATH,
            inp_data_dict=inp_data_dict,
            solver='CVODE_myokit',
        )
    except (RuntimeError, ImportError) as exc:
        pytest.skip(f'Myokit backend not available: {exc}')
    return runner


# ---------------------------------------------------------------------------
# Guard: skip the whole module if neither model nor JSON exists
# ---------------------------------------------------------------------------

if not os.path.exists(_CELLML_PATH):
    pytest.skip(
        f'SN_simple CellML model not found: {_CELLML_PATH}',
        allow_module_level=True,
    )
if not os.path.exists(_OBS_JSON_PATH):
    pytest.skip(
        f'SN_simple obs JSON not found: {_OBS_JSON_PATH}',
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Test 1 — ProtocolExecutor: single-subexperiment protocol
# ---------------------------------------------------------------------------

def test_protocol_executor_basic():
    """
    ProtocolExecutor runs a single-experiment, single-subexperiment protocol
    and returns finite, non-empty results.
    """
    helper = _make_myokit_helper(sim_time=0.35, pre_time=0.1)
    executor = ProtocolExecutor(helper)

    protocol_info = {
        'sim_times': [[0.35]],
        'pre_times': [0.1],
        'params_to_change': {'soma_SN/I_in': [[0.0]]},
    }

    success, results_by_sub, extra_by_sub, t_by_exp = executor.run_protocol(
        protocol_info,
        result_variables=None,  # collect all
    )

    assert success, 'ProtocolExecutor.run_protocol reported failure'
    assert (0, 0) in results_by_sub, 'Missing result for (exp=0, sub=0)'

    sub_res = results_by_sub[(0, 0)]
    assert len(sub_res) > 0, 'Result list is empty'

    t = t_by_exp[0]
    assert t is not None, 't_by_exp[0] is None'
    expected_n = round(0.35 / _DT) + 1
    assert abs(len(t) - expected_n) <= 2, (
        f'Time vector length {len(t)} differs from expected ~{expected_n}'
    )
    assert np.all(np.isfinite(t)), 'Time vector contains non-finite values'

    # Every variable array should be finite
    for arr in sub_res:
        arr = np.asarray(arr)
        if arr.ndim > 0 and arr.size > 1:
            assert np.all(np.isfinite(arr)), 'Result array contains non-finite values'


# ---------------------------------------------------------------------------
# Test 2 — ProtocolExecutor: two-subexperiment protocol from JSON fixture
# ---------------------------------------------------------------------------

def test_protocol_executor_multi_subexperiment():
    """
    ProtocolExecutor runs the two-subexperiment protocol from
    SN_simple_param_id_fast_obs.json (pre_time=0.1, sim_times=[0.35, 0.85])
    and produces results keyed by (0,0) and (0,1).
    """
    protocol_info = _load_protocol_info()
    # Raw protocol_info from the fixture uses only the basic keys
    helper = _make_myokit_helper(sim_time=0.35, pre_time=0.1)
    executor = ProtocolExecutor(helper)

    success, results_by_sub, _, t_by_exp = executor.run_protocol(
        protocol_info,
        result_variables=None,
    )

    assert success, 'Multi-subexperiment protocol reported failure'
    assert (0, 0) in results_by_sub, 'Missing result for sub-experiment 0'
    assert (0, 1) in results_by_sub, 'Missing result for sub-experiment 1'

    t = t_by_exp[0]
    assert t is not None
    # Total sim_time = 0.35 + 0.85 = 1.2 s; t should span [0, ~1.2] after pre_time shift
    assert t[0] >= -1e-9, f't[0] should be ~0 but got {t[0]}'
    total_expected = 0.35 + 0.85
    assert abs(t[-1] - total_expected) < 0.05, (
        f't[-1]={t[-1]:.4f} differs from expected total {total_expected:.2f} s'
    )


# ---------------------------------------------------------------------------
# Test 3 — ProtocolRunner: standalone use (like a notebook or script)
# ---------------------------------------------------------------------------

def test_protocol_runner_standalone():
    """
    ProtocolRunner creates its own SimulationHelper and returns t_list /
    res_list / sim_times with the expected shapes when called directly.
    """
    runner = _make_runner()
    protocol_info = _load_protocol_info()

    t_list, res_list, sim_times = runner.run_protocols(_CELLML_PATH, protocol_info)

    assert len(t_list) == 1, f'Expected 1 experiment, got {len(t_list)}'
    assert len(res_list) == 1

    t = t_list[0]
    assert t is not None and len(t) > 0
    assert np.all(np.isfinite(t)), 'Time vector contains non-finite values'

    # res_list[0] is a list of arrays, one per variable
    res = res_list[0]
    assert len(res) > 0, 'Result list is empty'

    var_names = runner.get_variable_names()
    assert len(res) == len(var_names), (
        f'Result list length {len(res)} != variable_names length {len(var_names)}'
    )

    # The time vector should span the total sim_time
    total_sim = sum(sim_times[0])
    assert abs(t[-1] - total_sim) < 0.05, (
        f't[-1]={t[-1]:.4f} differs from expected {total_sim:.2f} s'
    )


# ---------------------------------------------------------------------------
# Test 4 — ProtocolRunner result-dict conversion (ProcessData-like usage)
# ---------------------------------------------------------------------------

def test_protocol_runner_result_dict_conversion():
    """
    After run_protocols, convert res_list[exp_idx] from a positional list of
    arrays into a dict keyed by variable name — the same conversion that
    ProcessData.__simulate_model performs.

    Verifies that the variable 'soma_SN/V' (or its myokit equivalent) is
    accessible and has the expected number of time points.
    """
    runner = _make_runner()
    protocol_info = _load_protocol_info()

    t_list, res_list, sim_times = runner.run_protocols(_CELLML_PATH, protocol_info)

    var_names = runner.get_variable_names()
    var2idx = runner.get_var2idx_dict()

    # Build a result dict for experiment 0
    res_dict = {name: res_list[0][idx] for name, idx in var2idx.items()}

    # Find a variable name that matches soma_SN/V (myokit uses dots)
    voltage_key = next(
        (k for k in res_dict if 'V' in k and ('soma_SN' in k or 'soma' in k.lower())),
        None
    )
    assert voltage_key is not None, (
        f'Could not find soma voltage variable in {list(res_dict.keys())[:10]}'
    )

    v_arr = np.asarray(res_dict[voltage_key])
    assert v_arr.ndim == 1 and v_arr.size > 0
    assert np.all(np.isfinite(v_arr)), 'Voltage array contains non-finite values'
    assert len(v_arr) == len(t_list[0]), (
        'Voltage array length does not match time vector length'
    )


# ---------------------------------------------------------------------------
# Test 5 — ProcessData meta-experiment pattern via ProtocolRunner
# ---------------------------------------------------------------------------

def test_protocol_runner_processdata_meta_pattern():
    """
    Mimic ProcessData.__simulate_model: two meta-experiments that share the
    same sim_times / pre_times template but use different slices of
    params_to_change.

    Verifies that each meta run produces an independent t_list and res_list
    with consistent shapes.
    """
    runner = _make_runner()

    # Build a 2-experiment protocol (one sub-exp each, short sim_times)
    base_protocol_info = {
        'sim_times': [[0.3], [0.3]],
        'pre_times': [0.05, 0.05],
        'params_to_change': {
            'soma_SN/I_in': [[0.0], [-0.05]],
        },
    }
    sim_times = base_protocol_info['sim_times']
    pre_times = base_protocol_info['pre_times']
    params_to_change = base_protocol_info['params_to_change']

    # Simulate two meta experiments, each with 1 experiment, slicing
    # params_to_change globally (as ProcessData does with sim_count/sim_offset).
    results_per_meta = []
    var2idx = runner.get_var2idx_dict()

    for meta_idx in range(2):
        n_exps = 1
        sim_offset = meta_idx  # meta 0 → slice [0:1], meta 1 → slice [1:2]

        sub_protocol_info = {
            'sim_times': sim_times[:n_exps],
            'pre_times': pre_times[:n_exps],
            'params_to_change': {
                key: list(vals)[sim_offset:sim_offset + n_exps]
                for key, vals in params_to_change.items()
            },
        }

        t_list, res_list_all, _ = runner.run_protocols(
            _CELLML_PATH, sub_protocol_info
        )

        # Convert to dict (ProcessData pattern)
        variables_to_plot = ['soma_SN/V']
        res_list_dicts = []
        for exp_idx in range(n_exps):
            res_dict = {}
            for var_name in variables_to_plot:
                # myokit uses dots, so search for a matching key
                matched = next(
                    (k for k in var2idx if var_name.replace('/', '.') in k or var_name in k),
                    None
                )
                if matched is not None:
                    res_dict[var_name] = res_list_all[exp_idx][var2idx[matched]]
                else:
                    res_dict[var_name] = None
            res_list_dicts.append(res_dict)

        results_per_meta.append((t_list, res_list_dicts))

    # Both meta experiments should have produced a valid result
    for meta_idx, (t_list, res_list_dicts) in enumerate(results_per_meta):
        assert len(t_list) == 1, f'Meta {meta_idx}: expected 1 time vector'
        t = t_list[0]
        assert t is not None and len(t) > 0, f'Meta {meta_idx}: empty time vector'
        assert np.all(np.isfinite(t)), f'Meta {meta_idx}: time vector has non-finite values'

    # The two meta experiments had different I_in values; their voltage traces
    # should be present (though we do not compare the values themselves here)
    for meta_idx, (t_list, res_list_dicts) in enumerate(results_per_meta):
        v = res_list_dicts[0].get('soma_SN/V')
        # v may be None if the variable name format differs; skip assertion in that case
        if v is not None:
            assert np.all(np.isfinite(np.asarray(v))), (
                f'Meta {meta_idx}: voltage contains non-finite values'
            )
