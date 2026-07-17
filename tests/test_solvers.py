"""
Tests for different solver implementations.

These tests verify that OpenCOR (CVODE_opencor), Myokit (CVODE_myokit), and Python BDFsolvers
work correctly for various models.

"""
import os
import re
import sys
import pytest
import numpy as np
import shutil

# Ensure src is on sys.path
_TEST_ROOT = os.path.join(os.path.dirname(__file__), '..')
_SRC_DIR = os.path.join(_TEST_ROOT, 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from solver_wrappers import get_simulation_helper
from generators.PythonGenerator import PythonGenerator
from scripts.script_generate_with_new_architecture import generate_with_new_architecture
import xml.etree.ElementTree as ET

# _MODEL_INPUT_FILES and the generated_cellml_model_factory fixture now live in
# tests/conftest.py so they are shared with test_protocol_state_continuity.py.


def _normalize_variable_name(var_name, solver_type):
    """
    Normalize variable names for comparison across solvers.
    
    Args:
        var_name: Variable name from solver
        solver_type: Type of solver ('myokit', 'opencor', 'python')
    
    Returns:
        Normalized (component, variable) tuple
    """
    if solver_type == 'myokit':
        # Format: component_module.variable
        parts = var_name.split('.')
        if len(parts) >= 2:
            comp = parts[0].replace('_module', '')
            var = '.'.join(parts[1:])
            return comp, var
        return None, var_name
    elif solver_type == 'opencor' or solver_type == 'python':
        # Format: component/variable
        parts = var_name.split('/')
        if len(parts) == 2:
            return parts[0], parts[1]
        return None, var_name
    # elif solver_type == 'python':
    #     # Format: variable (no component prefix typically)
    #     return None, var_name
    return None, var_name


def _match_variables(ref_vars, ref_type, other_vars, other_type):
    """
    Match variables between two solvers.
    
    Args:
        ref_vars: List of reference variable names
        ref_type: Type of reference solver
        other_vars: List of other solver variable names
        other_type: Type of other solver
    
    Returns:
        Dictionary mapping reference variable names to other variable names
    """
    mapping = {}
    
    # Build normalized lookup for other solver
    other_normalized = {}
    for var in other_vars:
        comp, var_name = _normalize_variable_name(var, other_type)
        key = (comp, var_name) if comp else (None, var_name)
        if key not in other_normalized:
            other_normalized[key] = var
    
    # First pass: collect exact matches and build fallback candidates
    fallback_candidates = {}  # ref_var -> other_var (via short-name fallback)
    fallback_claims = {}      # other_var -> count of ref_vars that would claim it
    for ref_var in ref_vars:
        ref_comp, ref_var_name = _normalize_variable_name(ref_var, ref_type)
        ref_key = (ref_comp, ref_var_name) if ref_comp else (None, ref_var_name)

        if ref_key in other_normalized:
            mapping[ref_var] = other_normalized[ref_key]
        else:
            # Fallback: match by variable name only (component-agnostic)
            var_only_key = (None, ref_var_name)
            if var_only_key in other_normalized:
                other_var = other_normalized[var_only_key]
                fallback_candidates[ref_var] = other_var
                fallback_claims[other_var] = fallback_claims.get(other_var, 0) + 1

    # Second pass: only include unambiguous fallback matches (one ref_var per other_var)
    for ref_var, other_var in fallback_candidates.items():
        if fallback_claims[other_var] == 1:
            mapping[ref_var] = other_var

    return mapping


def _check_initial_states(myokit_helper, opencor_helper, model_name):
    """
    Check that all initial STATE variables are correctly defined in the Myokit modified model
    and compare with OpenCOR initial states.

    Args:
        myokit_helper: Myokit SimulationHelper instance
        opencor_helper: OpenCOR SimulationHelper instance
        model_name: Name of the model for reporting

    Returns:
        Dictionary with check results and any mismatches
    """
    results = {
        "myokit_initial_states": {},
        "opencor_initial_states": {},
        "cellml_initial_states": {},
        "mismatches": [],
        "missing_in_myokit": [],
        "missing_in_opencor": []
    }

    # Get Myokit STATE variables from simulation default_state (should be CellML values)
    if hasattr(myokit_helper, 'state_index'):
        try:
            myokit_state = myokit_helper.simulation.default_state()
            print(f"DEBUG: Getting Myokit states from simulation.default_state(): {len(myokit_helper.state_index)} variables")
        except:
            myokit_state = myokit_helper.simulation.state()
            print(f"DEBUG: default_state() failed, using simulation.state(): {len(myokit_helper.state_index)} variables")

        for qname, idx in list(myokit_helper.state_index.items())[:3]:
            val = float(myokit_state[idx])
            results["myokit_initial_states"][qname] = val
            print(f"  {qname}: {val}")
        # Get the rest
        for qname, idx in list(myokit_helper.state_index.items())[3:]:
            results["myokit_initial_states"][qname] = float(myokit_state[idx])

    # Get OpenCOR STATE variables - need to reset and use get_init_param_vals
    if hasattr(opencor_helper, 'reset_states') and hasattr(opencor_helper, 'get_init_param_vals'):
        opencor_helper.reset_states()

        # Get all state variable names from OpenCOR
        all_vars = opencor_helper.get_all_variable_names()
        state_var_names = []
        for var_name in all_vars:
            if hasattr(opencor_helper, 'data') and hasattr(opencor_helper.data, 'states') and var_name in opencor_helper.data.states():
                state_var_names.append(var_name)

        # Get initial values for state variables
        if state_var_names:
            try:
                init_vals = opencor_helper.get_init_param_vals(state_var_names)
                for var_name, init_val_list in zip(state_var_names, init_vals):
                    if init_val_list and len(init_val_list) > 0:
                        val = init_val_list[0] if isinstance(init_val_list, list) else init_val_list
                        try:
                            # Handle numpy arrays and other types
                            if hasattr(val, 'item'):
                                results["opencor_initial_states"][var_name] = float(val.item())
                            elif isinstance(val, (int, float, np.number)):
                                results["opencor_initial_states"][var_name] = float(val)
                            else:
                                results["opencor_initial_states"][var_name] = float(val)
                        except (TypeError, ValueError, AttributeError):
                            pass
            except Exception as e:
                print(f"Warning: Could not get OpenCOR initial states: {e}")
    
    # Get CellML initial states from processed model
    if hasattr(myokit_helper, 'processed_cellml_path') and myokit_helper.processed_cellml_path:
        processed_path = myokit_helper.processed_cellml_path
        if os.path.exists(processed_path):
            try:
                tree = ET.parse(processed_path)
                root = tree.getroot()
                ns = {'c': 'http://www.cellml.org/cellml/2.0#',
                      'm': 'http://www.w3.org/1998/Math/MathML'}
                
                for var_elem in root.findall('.//c:variable', ns):
                    var_name = var_elem.get('name')
                    initial_value_str = var_elem.get('initial_value')
                    if initial_value_str is not None:
                        try:
                            # Get component name
                            parent_component = var_elem.find('..')
                            if parent_component is not None and 'component' in parent_component.tag:
                                comp_name = parent_component.get('name', '')
                                if comp_name:
                                    full_name = f"{comp_name}.{var_name}"
                                else:
                                    full_name = var_name
                            else:
                                full_name = var_name
                            
                            results["cellml_initial_states"][full_name] = float(initial_value_str)
                        except ValueError:
                            pass
            except Exception as e:
                print(f"Warning: Could not parse CellML initial states: {e}")
    
    # Compare initial states
    print(f"\n{'='*80}")
    print(f"INITIAL STATE CHECK - {model_name} model")
    print("="*80)
    print(f"Myokit initial states: {len(results['myokit_initial_states'])}")
    print(f"OpenCOR initial states: {len(results['opencor_initial_states'])}")

    # Create mapping from Myokit names to OpenCOR names for comparison
    myokit_vars = sorted(results['myokit_initial_states'].keys())
    opencor_vars = sorted(results['opencor_initial_states'].keys())

    myokit_to_opencor = {}
    for mk_var in myokit_vars:
        if '.' in mk_var:
            comp_mod, var = mk_var.split('.', 1)
            if comp_mod.endswith('_module'):
                comp = comp_mod[:-7]  # Remove '_module'
                opencor_equiv = f'{comp}/{var}'
                myokit_to_opencor[mk_var] = opencor_equiv

    # Find matching variables for comparison
    matching_vars = {}
    for mk_var, oc_equiv in myokit_to_opencor.items():
        if oc_equiv in opencor_vars:
            matching_vars[mk_var] = oc_equiv

    print(f"\\n=== COMPARING MATCHING STATE VARIABLES ({len(matching_vars)}) ===")
    mismatches = []
    for mk_var, oc_var in sorted(matching_vars.items()):
        mk_val = results["myokit_initial_states"][mk_var]
        oc_val = results["opencor_initial_states"][oc_var]
        diff_pct = abs(mk_val - oc_val) / max(abs(mk_val), abs(oc_val), 1e-15) * 100
        status = "✓" if diff_pct < 0.01 else "✗"
        print(f"  {mk_var} <-> {oc_var}: {mk_val:.6e} vs {oc_val:.6e} ({diff_pct:.3f}%) {status}")
        if diff_pct >= 0.01:
            mismatches.append((mk_var, oc_var, mk_val, oc_val, diff_pct))

    # Only compare matching variables
    results["myokit_initial_states"] = {k: v for k, v in results["myokit_initial_states"].items() if k in matching_vars}
    results["opencor_initial_states"] = {oc_var: results["opencor_initial_states"][oc_var] for mk_var, oc_var in matching_vars.items() if oc_var in results["opencor_initial_states"]}

    results["mismatches"] = mismatches
    return results

    


def _to_numpy(data):
    """Convert data to a 1-D numpy float64 array, handling CasADi DM/SX objects."""
    if isinstance(data, np.ndarray):
        return data.astype(float)
    try:
        import casadi as _ca
        if isinstance(data, (_ca.DM, _ca.SX, _ca.MX)):
            return np.array(_ca.DM(data)).flatten().astype(float)
    except (ImportError, Exception):
        pass
    return np.array([data], dtype=float)


def _get_python_model_initial_states(model_path):
    """
    Load a Python model file fresh (without any CasADi patching) and return
    numeric initial state values.

    Returns:
        dict mapping state name (component/variable) -> float initial value
    """
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location("_init_check_model", model_path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    states = mod.create_states_array()
    rates = mod.create_states_array()
    variables = mod.create_variables_array()
    mod.initialise_variables(states, rates, variables)
    mod.compute_computed_constants(variables)
    def _qname(info):
        comp = info.get("component", "")
        if comp.endswith("_module"):
            comp = comp[:-7]
        return f"{comp}/{info['name']}" if comp else info["name"]
    return {_qname(info): float(states[idx]) for idx, info in enumerate(mod.STATE_INFO)}


def _check_python_family_initial_states(helpers, model_name, tolerance=0.01):
    """
    Compare Python-family helpers (solve_ivp_BDF, casadi_integrator_cvodes) initial
    states against a reference built from OpenCOR (preferred) or Myokit (fallback).

    The reference uses OpenCOR-style component/variable keys. Myokit keys are
    converted to the same format when OpenCOR is unavailable so CI can still
    validate Python-family helpers.

    Returns:
        dict with 'mismatches' list of
        (state_name, helper_key, ref_val, other_val, diff_pct)
    """
    python_family_keys = [
        k for k in ("solve_ivp_BDF", "casadi_integrator_cvodes") if k in helpers
    ]
    if not python_family_keys:
        return {"mismatches": []}

    # Build reference state map in component/variable format
    ref_state_map = {}
    ref_label = None
    if "CVODE_opencor" in helpers:
        oc = helpers["CVODE_opencor"]
        for name, val in oc.data.states().items():
            # OpenCOR returns DataStore::DataStoreValue objects; extract numeric value
            ref_state_map[name] = float(val.value() if hasattr(val, "value") else val)
        ref_label = "CVODE_opencor"
    elif "CVODE_myokit" in helpers:
        mk = helpers["CVODE_myokit"]
        try:
            mk_state = mk.simulation.default_state()
        except Exception:
            mk_state = mk.simulation.state()
        for qname, idx in mk.state_index.items():
            if "." in qname:
                comp_mod, var = qname.split(".", 1)
                comp = comp_mod[:-7] if comp_mod.endswith("_module") else comp_mod
                ref_state_map[f"{comp}/{var}"] = float(mk_state[idx])
        ref_label = "CVODE_myokit"

    if not ref_state_map:
        return {"mismatches": []}

    mismatches = []
    print(f"\n{'='*80}")
    print(
        f"PYTHON-FAMILY INITIAL STATE CHECK - {model_name} (ref: {ref_label})"
    )
    print("=" * 80)

    for helper_key in python_family_keys:
        # Reload the model numerically (avoids CasADi symbolic patching)
        model_path = helpers[helper_key].model_path
        try:
            python_model_initial = _get_python_model_initial_states(model_path)
        except Exception as e:
            print(f"  ⊘ {helper_key}: could not load model numerically: {e}")
            continue

        print(f"\n  {helper_key}:")
        for state_name, ref_val in sorted(ref_state_map.items()):
            if state_name not in python_model_initial:
                print(f"    ⊘ {state_name}: missing in {helper_key} model")
                continue
            other_val = python_model_initial[state_name]
            denom = max(abs(ref_val), abs(other_val), 1e-15)
            diff_pct = abs(ref_val - other_val) / denom * 100
            status = "✓" if diff_pct < tolerance else "✗"
            print(
                f"    {status} {state_name}: {ref_val:.6e} vs {other_val:.6e} "
                f"({diff_pct:.3f}%)"
            )
            if diff_pct >= tolerance:
                mismatches.append(
                    (state_name, helper_key, ref_val, other_val, diff_pct)
                )

    return {"mismatches": mismatches}


def _compare_solver_results(ref_helper, ref_name, other_helper, other_name, tolerance=0.01):
    """
    Compare results between two solvers.
    
    Args:
        ref_helper: Reference solver SimulationHelper
        ref_name: Name of reference solver
        other_helper: Other solver SimulationHelper
        other_name: Name of other solver
        tolerance: Maximum allowed relative error percentage (default 0.01%)
    
    Returns:
        Dictionary with comparison results and any failures
    """
    ref_vars = ref_helper.get_all_variable_names()
    other_vars = other_helper.get_all_variable_names()
    
    ref_results = ref_helper.get_all_results(flatten=False)
    other_results = other_helper.get_all_results(flatten=False)
    
    # Build result dictionaries
    ref_dict = {}
    for i, var in enumerate(ref_vars):
        ref_dict[var] = ref_results[i][0]
    
    other_dict = {}
    for i, var in enumerate(other_vars):
        other_dict[var] = other_results[i][0]
    
    # Match variables
    ref_type = 'myokit' if 'myokit' in ref_name else ('opencor' if 'opencor' in ref_name.lower() else 'python')
    other_type = 'myokit' if 'myokit' in other_name else ('opencor' if 'opencor' in other_name.lower() else 'python')
    
    var_mapping = _match_variables(ref_vars, ref_type, other_vars, other_type)
    
    # Compare matched variables
    comparisons = []
    max_rel_error = 0.0
    failed_vars = []
    
    for ref_var, other_var in var_mapping.items():
        ref_data = ref_dict[ref_var]
        other_data = other_dict[other_var]
        
        # Ensure numpy float arrays (handles CasADi DM/SX and scalars)
        ref_data = _to_numpy(ref_data)
        other_data = _to_numpy(other_data)
        
        # Skip scalars (constants) - only compare time series
        if len(ref_data) <= 1 or len(other_data) <= 1:
            continue
        
        # Handle different lengths
        min_len = min(len(ref_data), len(other_data))
        ref_data = ref_data[:min_len]
        other_data = other_data[:min_len]
        
        # Calculate relative error
        abs_diff = np.abs(ref_data - other_data)
        ref_abs = np.abs(ref_data)
        other_abs = np.abs(other_data)
        max_abs = np.maximum(ref_abs, other_abs)
        max_abs = np.maximum(max_abs, 1e-10)  # Avoid division by zero
        
        rel_error = (abs_diff / max_abs) * 100
        has_nan = bool(np.any(np.isnan(rel_error)))
        max_rel_error_var = np.nanmax(rel_error) if not has_nan else float('nan')
        mean_rel_error_var = np.nanmean(rel_error)
        
        comparisons.append({
            'ref_var': ref_var,
            'other_var': other_var,
            'max_rel_error': max_rel_error_var,
            'mean_rel_error': mean_rel_error_var,
            'max_abs_diff': np.nanmax(abs_diff),
            'mean_abs_diff': np.nanmean(abs_diff)
        })
        
        if has_nan or max_rel_error_var > max_rel_error:
            max_rel_error = max_rel_error_var
        
        # NaN counts as a failure (comparison undefined — likely a conversion bug)
        if has_nan or max_rel_error_var > tolerance:
            failed_vars.append({
                'ref_var': ref_var,
                'other_var': other_var,
                'max_rel_error': max_rel_error_var
            })
    
    return {
        'comparisons': comparisons,
        'max_rel_error': max_rel_error,
        'failed_vars': failed_vars,
        'matched_count': len(var_mapping),
        'compared_count': len(comparisons)
    }


@pytest.mark.integration
@pytest.mark.solver
def test_myokit_multi_trace_protocol():
    """
    Verify multi-experiment forcing: protocol traces vs constants, with pace rebounding.

    Uses tests/test_inputs/Lotka_Volterra_forced.cellml (flat CellML 2.0 with u_alpha
    and u_gamma forcing inputs) and resources/Lotka_Volterra_forced_multi_trace_obs_data.json:

      - Experiment 0: pre_time=1 s, two subexperiments [0.25 s, 5 s] — constants then
        u_alpha trace — exercises cumulative protocol times vs Myokit simulation.reset().
      - Experiment 1: u_alpha = 0, u_gamma step trace
      - Experiment 2: u_alpha fixed constant, u_gamma = 0 (both numeric — no trace)
      - Experiment 3: u_alpha = 0, u_gamma fixed constant

    This exercises myokit_helper rebinding the 'pace' label when switching which
    input is driven by TimeSeriesProtocol, multi-subexperiment update_times/run loops,
    and runs where both inputs use plain constants only.
    """
    import json

    tests_dir = os.path.dirname(__file__)
    cellml_path = os.path.join(tests_dir, "test_inputs", "Lotka_Volterra_forced.cellml")
    obs_data_path = os.path.join(_TEST_ROOT, "resources", "Lotka_Volterra_forced_multi_trace_obs_data.json")

    assert os.path.exists(cellml_path), f"CellML model not found: {cellml_path}"
    assert os.path.exists(obs_data_path), f"obs_data not found: {obs_data_path}"

    with open(obs_data_path, encoding="utf-8-sig") as fh:
        obs_data = json.load(fh)
    protocol_info = obs_data["protocol_info"]

    dt = 0.01
    solver_info = {"MaximumStep": 0.05, "MaximumNumberOfSteps": 50000}

    try:
        helper = get_simulation_helper(
            model_path=cellml_path,
            model_type="cellml_only",
            solver="CVODE_myokit",
            dt=dt,
            sim_time=1.0,   # overridden per-experiment below
            solver_info=solver_info,
            pre_time=0.0,
        )
    except RuntimeError as exc:
        pytest.skip(f"Myokit backend not available: {exc}")

    helper.set_protocol_info(protocol_info)

    sim_times = protocol_info["sim_times"]
    pre_times = protocol_info["pre_times"]
    params_to_change = protocol_info["params_to_change"]
    param_keys = list(params_to_change.keys())

    all_results = {}

    for exp_idx in range(len(sim_times)):
        current_time = 0.0
        for sub_idx, sim_time in enumerate(sim_times[exp_idx]):
            if sub_idx == 0:
                helper.update_times(dt, current_time, sim_time, pre_times[exp_idx])
                current_time += pre_times[exp_idx]
            else:
                helper.update_times(dt, current_time, sim_time, pre_time=0.0)

            param_vals = [params_to_change[k][exp_idx][sub_idx] for k in param_keys]
            helper.set_param_vals(param_keys, param_vals)
            ok = helper.run()
            assert ok, f"Simulation failed for experiment {exp_idx}, sub-experiment {sub_idx}"
            current_time += sim_time

        # Collect results after last sub-experiment
        var_names = helper.get_all_variable_names()
        x_name = next((n for n in var_names if n.endswith(".x")), None)
        y_name = next((n for n in var_names if n.endswith(".y")), None)
        assert x_name and y_name, f"Could not find x/y in {var_names[:10]}"

        x_series = np.asarray(helper.get_results([x_name], flatten=True)[0], dtype=float)
        y_series = np.asarray(helper.get_results([y_name], flatten=True)[0], dtype=float)

        assert np.all(np.isfinite(x_series)), f"x series contains non-finite values in exp {exp_idx}"
        assert np.all(np.isfinite(y_series)), f"y series contains non-finite values in exp {exp_idx}"
        assert np.all(x_series >= 0), f"Prey (x) went negative in exp {exp_idx}"
        assert np.all(y_series >= 0), f"Predator (y) went negative in exp {exp_idx}"

        all_results[exp_idx] = {"x": x_series, "y": y_series}
        helper.reset_and_clear()

    # Distinct regimes: pacing u_alpha vs u_gamma produces different dynamics
    assert not np.allclose(all_results[0]["x"], all_results[1]["x"], rtol=1e-3), (
        "Experiments 0 vs 1: x trajectories should differ when different inputs are paced."
    )
    # Pure-constant forcings differ between exp 2 (boost u_alpha) and exp 3 (boost u_gamma)
    assert not np.allclose(all_results[2]["x"], all_results[3]["x"], rtol=1e-3), (
        "Experiments 2 vs 3: different constant forcings should produce different dynamics."
    )
    # Trace on u_alpha (exp 0) should differ from flat constant u_alpha (exp 2)
    assert not np.allclose(all_results[0]["x"], all_results[2]["x"], rtol=1e-3), (
        "Experiments 0 vs 2: time-varying u_alpha should differ from constant u_alpha."
    )


def _lotka_sim_helper(generated_cellml_model_factory, temp_model_dir, model_type, dt, sim_time):
    """A Lotka-Volterra simulation helper on the given backend (Myokit or CasADi).

    CasADi needs a generated (casadi_compat) Python model; Myokit runs the cellml directly. Both
    are non-stiff CVODE-family integrations at tight tolerance so their trajectories agree.
    """
    cellml = generated_cellml_model_factory("Lotka_Volterra", "Lotka_Volterra_parameters.csv")
    if model_type == "casadi_python":
        pytest.importorskip("casadi")
        model_path = PythonGenerator(
            cellml, output_dir=temp_model_dir, module_name="Lotka_Volterra_casadi",
            casadi_compat=True).generate()
        solver = "casadi_integrator"
        solver_info = {"method": "cvodes", "max_step_size": 0.005, "max_num_steps": 50000}
    else:
        model_path = cellml
        solver = "CVODE_myokit"
        solver_info = {"MaximumStep": 0.005, "MaximumNumberOfSteps": 50000,
                       "rtol": 1e-9, "atol": 1e-9}
    return get_simulation_helper(
        model_path=model_path, model_type=model_type, solver=solver,
        dt=dt, sim_time=sim_time, solver_info=solver_info, pre_time=0.0)


@pytest.mark.integration
@pytest.mark.solver
@pytest.mark.parametrize("model_type", ["cellml_only", "casadi_python"])
def test_params_to_change_affects_targeted_subexperiment_output(
        model_type, generated_cellml_model_factory, temp_model_dir):
    """A params_to_change value must change the simulated output of the sub-experiment it targets,
    and only that one -- on both the Myokit (cellml_only) and CasADi backends.

    Lotka-Volterra with gamma (the predator decay rate) changed between two sub-experiments: gamma
    is the model default in sub 0 and set per-run in sub 1. Running two different sub-1 values,
    sub 0 (same gamma in both) must be identical and sub 1 (differing gamma) must visibly differ.
    A silently-ignored params_to_change would leave sub 1 identical too.
    """
    dt, T = 0.02, 2.0
    GAMMA, Y = "Lotka_Volterra/gamma", "Lotka_Volterra/y"

    def run_protocol(gamma_sub1):
        h = _lotka_sim_helper(generated_cellml_model_factory, temp_model_dir, model_type, dt, T)
        h.set_protocol_info({"pre_times": [0.0], "sim_times": [[T, T]],
                             "params_to_change": {GAMMA: [[3.0, gamma_sub1]]}})
        h.reset_states()
        traces, cur = [], 0.0
        for sub_idx, st in enumerate([T, T]):
            h.update_times(dt, cur, st, 0.0)
            h.set_param_vals([GAMMA], [3.0 if sub_idx == 0 else gamma_sub1])
            assert h.run(), f"sub-experiment {sub_idx} failed"
            traces.append(np.asarray(h.get_results([[Y]], flatten=True)[0]).flatten())
            cur += st
        h.close_simulation()
        return traces

    base = run_protocol(3.0)      # gamma unchanged in sub 1
    changed = run_protocol(6.0)   # gamma changed in sub 1

    n0 = min(len(base[0]), len(changed[0]))
    np.testing.assert_allclose(
        base[0][:n0], changed[0][:n0], rtol=1e-6, atol=1e-6,
        err_msg="sub 0 output changed even though its params_to_change value did not")

    n1 = min(len(base[1]), len(changed[1]))
    denom = np.max(np.abs(base[1][:n1])) + 1e-9
    max_rel_diff = float(np.max(np.abs(changed[1][:n1] - base[1][:n1])) / denom)
    assert max_rel_diff > 0.05, (
        f"params_to_change (gamma) had no meaningful effect on its target sub-experiment: "
        f"max relative difference {max_rel_diff:.4g}")


@pytest.mark.integration
@pytest.mark.solver
def test_trace_step_input_equals_discrete_subexperiment_change(
        generated_cellml_model_factory, temp_model_dir):
    """A step-function trace input over ONE sub-experiment gives the same output as the same step
    encoded as a DISCRETE parameter change across TWO sub-experiments.

    "gamma steps 3 -> 5 at t=T", compared against a Myokit step-trace reference:
      - Myokit, one sub-experiment, gamma driven by a step trace (a 0-order step at t=T);
      - Myokit, two sub-experiments, gamma = 3 then 5 (discrete);
      - CasADi, two sub-experiments, gamma = 3 then 5 (discrete).
    CasADi has no trace/pace mechanism, so it can only express the discrete two-sub encoding; that
    it matches the Myokit trace-step is the check that CasADi's multi-sub state carry is correct.
    The two-sub versions restart the integrator at the boundary while the trace crosses the
    discontinuity continuously, so they agree to solver tolerance, not bits.
    """
    dt, T, G0, G1 = 0.02, 2.0, 3.0, 5.0
    GAMMA, Y = "Lotka_Volterra/gamma", "Lotka_Volterra/y"

    def discrete_2sub(model_type):
        h = _lotka_sim_helper(generated_cellml_model_factory, temp_model_dir, model_type, dt, T)
        h.set_protocol_info({"pre_times": [0.0], "sim_times": [[T, T]],
                             "params_to_change": {GAMMA: [[G0, G1]]}})
        h.reset_states()
        ys, cur = [], 0.0
        for sub_idx, st in enumerate([T, T]):
            h.update_times(dt, cur, st, 0.0)
            h.set_param_vals([GAMMA], [G0 if sub_idx == 0 else G1])
            assert h.run(), f"{model_type} sub-experiment {sub_idx} failed"
            ys.append(np.asarray(h.get_results([[Y]], flatten=True)[0]).flatten())
            cur += st
        h.close_simulation()
        return np.concatenate([ys[0], ys[1][1:]])

    # Reference: Myokit, one sub-experiment, gamma driven by a step trace (0-order step at t=T).
    hT = _lotka_sim_helper(generated_cellml_model_factory, temp_model_dir, "cellml_only", dt, 2 * T)
    hT.set_protocol_info({"pre_times": [0.0], "sim_times": [[2 * T]],
                          "params_to_change": {GAMMA: [["g_step"]]},
                          "protocol_traces": {"g_step": {"t": [0.0, T, T, 2 * T],
                                                         "values": [G0, G0, G1, G1]}}})
    hT.reset_states()
    hT.update_times(dt, 0.0, 2 * T, 0.0)
    hT.set_param_vals([GAMMA], ["g_step"])
    assert hT.run(), "myokit trace-step run failed"
    trace = np.asarray(hT.get_results([[Y]], flatten=True)[0]).flatten()
    hT.close_simulation()

    for model_type in ("cellml_only", "casadi_python"):
        disc = discrete_2sub(model_type)
        n = min(len(trace), len(disc))
        assert n > 10 and len(trace) == len(disc), \
            f"{model_type}: grid mismatch {len(trace)} vs {len(disc)}"
        np.testing.assert_allclose(
            disc[:n], trace[:n], rtol=1e-3, atol=1e-3,
            err_msg=f"{model_type} two-sub discrete gamma differs from the Myokit step trace")


@pytest.mark.integration
@pytest.mark.solver
def test_myokit_set_constant_preserved_after_rebind():
    """
    Regression test for issue #219: cost mismatch when two protocol_traces are active.

    Root cause: _rebind_pace_to creates a new myokit.Simulation(self.model). Myokit clones
    self.model at construction time, so any set_constant calls made on the *old* simulation
    (e.g. ID parameters applied at the start of each experiment) were silently discarded.
    The fix syncs self.model via Variable.set_rhs so the next _recreate_simulation picks
    them up.

    Protocol: two experiments, each driven by a different trace parameter.
      - exp0: u_alpha trace  (set_protocol_info pre-binds u_alpha → no rebind here)
      - exp1: u_gamma trace  (u_gamma ≠ u_alpha → rebind triggered)

    A non-default alpha value (7.0 vs model default 5.0) is applied as an ID param at the
    start of each experiment.  exp1's results should reflect alpha=7.0; before the fix they
    reverted to alpha=5.0 (template model default) after the rebind, making them
    indistinguishable from a run where alpha was never changed.
    """
    tests_dir = os.path.dirname(__file__)
    cellml_path = os.path.join(tests_dir, "test_inputs", "Lotka_Volterra_forced.cellml")

    assert os.path.exists(cellml_path), f"CellML model not found: {cellml_path}"

    dt = 0.01
    sim_time = 5.0
    solver_info = {"MaximumStep": 0.05}

    # Minimal two-experiment protocol: exp0 paces u_alpha, exp1 paces u_gamma.
    # Both traces are identically zero so they don't alter dynamics directly;
    # only alpha (the ID param) changes dynamics between Run A and Run B.
    zero_trace = {"t": [0.0, sim_time], "values": [0.0, 0.0]}
    protocol_info = {
        "pre_times": [0.0, 0.0],
        "sim_times": [[sim_time], [sim_time]],
        "params_to_change": {
            "Lotka_Volterra_module/u_alpha": [["u_alpha_trace"], [0.0]],
            "Lotka_Volterra_module/u_gamma": [[0.0], ["u_gamma_trace"]],
        },
        "protocol_traces": {
            "u_alpha_trace": zero_trace,
            "u_gamma_trace": zero_trace,
        },
    }

    def _run_protocol(id_alpha):
        """Run both experiments applying id_alpha as the ID param for alpha."""
        try:
            helper = get_simulation_helper(
                model_path=cellml_path,
                model_type="cellml_only",
                solver="CVODE_myokit",
                dt=dt,
                sim_time=sim_time,
                solver_info=solver_info,
                pre_time=0.0,
            )
        except RuntimeError as exc:
            pytest.skip(f"Myokit backend not available: {exc}")

        helper.set_protocol_info(protocol_info)

        var_names_out = None
        results_by_exp = {}

        for exp_idx in range(2):
            # Step A: apply ID params (the alpha value under test)
            helper.set_param_vals(
                ["Lotka_Volterra_module/alpha"],
                [id_alpha],
            )
            helper.reset_states()

            # Step C: update times + run the single sub-experiment
            helper.update_times(dt, 0.0, sim_time, 0.0)

            ptc_vals = [
                protocol_info["params_to_change"][k][exp_idx][0]
                for k in protocol_info["params_to_change"]
            ]
            helper.set_param_vals(
                list(protocol_info["params_to_change"].keys()),
                ptc_vals,
            )

            ok = helper.run()
            assert ok, f"Simulation failed for exp {exp_idx} with alpha={id_alpha}"

            var_names = helper.get_all_variable_names()
            x_name = next((n for n in var_names if n.endswith(".x")), None)
            assert x_name, f"x variable not found in {var_names[:6]}"
            var_names_out = var_names

            results_by_exp[exp_idx] = np.asarray(
                helper.get_results([x_name], flatten=True)[0], dtype=float
            )
            helper.reset_and_clear()

        return results_by_exp

    # Run A: alpha overridden to 7.0 (non-default)
    results_alpha7 = _run_protocol(id_alpha=7.0)
    # Run B: alpha left at model default (5.0)
    results_alpha5 = _run_protocol(id_alpha=5.0)

    # exp0 (u_alpha trace, NO rebind): alpha differs → dynamics differ
    assert not np.allclose(results_alpha7[0], results_alpha5[0], rtol=1e-3), (
        "exp0 (no rebind): alpha=7 and alpha=5 should produce different x trajectories."
    )

    # exp1 (u_gamma trace, REBIND triggered): alpha must still differ.
    # Before the fix, alpha reverted to 5.0 after rebind in both runs,
    # making results_alpha7[1] == results_alpha5[1].
    assert not np.allclose(results_alpha7[1], results_alpha5[1], rtol=1e-3), (
        "exp1 (rebind triggered): alpha=7.0 should still be active after rebind — "
        "if this fails, set_constant changes are being lost in _rebind_pace_to "
        "(issue #219)."
    )


def test_init_states_myokit(generated_cellml_model_factory):
    """
    Repro for computed-constant initial state values via Myokit wrapper.

    - Uses resources/test_init_states_vessel_array.csv
    - Uses resources/test_init_states_parameters.csv where a_test_vessel = 3
    - Module defines x0 = 2 * a and x has initial_value=\"x0\"
    - Expect x(0) == 6 and y(0) == 1
    """
    cellml_path = generated_cellml_model_factory(
        "test_init_states",
        input_param_file="test_init_states_parameters.csv",
        solver="CVODE_myokit",
    )

    dt = 0.01
    sim_time = 0.1
    solver_info = {"MaximumStep": 0.001, "MaximumNumberOfSteps": 5000}
    helper = get_simulation_helper(model_path=cellml_path, model_type="cellml_only", dt=dt, sim_time=sim_time, 
                                   solver_info=solver_info, pre_time=0.0, solver="CVODE_myokit")
    result = helper.run()
    assert result, "Myokit simulation failed for init_states_test"

    # Find x and y state series
    names = helper.get_all_variable_names()
    x_name = next((n for n in names if n.endswith(".x")), None)
    y_name = next((n for n in names if n.endswith(".y")), None)
    assert x_name is not None and y_name is not None, f"Could not find x/y. Sample: {names[:20]}"

    x0 = float(np.asarray(helper.get_results([x_name], flatten=True)[0][0]))
    y0 = float(np.asarray(helper.get_results([y_name], flatten=True)[0][0]))
    assert np.isclose(x0, 6.0, rtol=0, atol=1e-12), f"Expected x(0)=6.0, got {x0} ({x_name})"
    assert np.isclose(y0, 1.0, rtol=0, atol=1e-12), f"Expected y(0)=1.0, got {y0} ({y_name})"


def _find_state_series_name(helper, state_basename):
    candidates = helper.get_all_variable_names()
    for name in candidates:
        if name == state_basename:
            return name
        if name.endswith(f"/{state_basename}") or name.endswith(f".{state_basename}"):
            return name
    raise AssertionError(f"Could not find state '{state_basename}' in variable names: {candidates[:20]}")


def _run_and_get_initial_state(helper, state_name):
    ok = helper.run()
    assert ok, "Simulation run failed"
    state_series = helper.get_results([state_name], flatten=True)[0]
    return float(np.asarray(state_series)[0])


@pytest.mark.integration
@pytest.mark.solver
@pytest.mark.parametrize("solver,model_type,solver_info", [
    ("CVODE_myokit", "cellml_only", {"MaximumStep": 0.001, "MaximumNumberOfSteps": 5000}),
    pytest.param(
        "CVODE_opencor",
        "cellml_only",
        {"MaximumStep": 0.001, "MaximumNumberOfSteps": 5000},
        marks=pytest.mark.need_opencor,
    ),
])
def test_set_param_vals_updates_state_init_for_cellml_solvers(solver, model_type, solver_info, generated_cellml_model_factory):
    """
    For 3compartment, q_lv initial state is controlled by q_lv_init.
    Verify set_param_vals + reset_states updates state initialization consistently.
    """
    model_path = generated_cellml_model_factory("3compartment", "3compartment_parameters.csv", solver=solver)

    dt = 0.01
    sim_time = 0.1
    pre_time = 0.0

    try:
        helper = get_simulation_helper(
            model_path=model_path,
            model_type=model_type,
            solver=solver,
            dt=dt,
            sim_time=sim_time,
            solver_info=solver_info,
            pre_time=pre_time,
        )
    except RuntimeError as e:
        pytest.skip(f"{solver} backend not available: {e}")

    q_name = _find_state_series_name(helper, "q_lv")

    # Case 1: q_lv_init = 2e-4 -> q_lv(0) = 2e-4
    helper.update_times(dt, 0.0, sim_time, pre_time)
    helper.set_param_vals(["global/q_lv_init"], [2e-4])
    helper.reset_states()
    q0_lo = _run_and_get_initial_state(helper, q_name)
    assert np.isclose(q0_lo, 2e-4, rtol=0.0, atol=1e-10), (
        f"{solver}: expected q_lv(0)=2e-4, got {q0_lo}"
    )

    # get_all_results_dict should still be available after reset via cache.
    _ = helper.get_all_results_dict()
    helper.reset_and_clear()
    cached_results = helper.get_all_results_dict()
    assert q_name in cached_results, f"{solver}: cached results missing {q_name}"
    if solver == "CVODE_myokit":
        assert "environment.time" in cached_results, (
            "CVODE_myokit: expected normalized time key 'environment.time' "
            "to be present in cached results"
        )

    # Case 2: q_lv_init = 8e-4 -> q_lv(0) = 8e-4
    helper.update_times(dt, 0.0, sim_time, pre_time)
    helper.set_param_vals(["global/q_lv_init"], [8e-4])
    helper.reset_states()
    q0_hi = _run_and_get_initial_state(helper, q_name)
    assert np.isclose(q0_hi, 8e-4, rtol=0.0, atol=1e-10), (
        f"{solver}: expected q_lv(0)=8e-4, got {q0_hi}"
    )


@pytest.mark.integration
@pytest.mark.solver
def test_set_param_vals_updates_state_init_for_python_solver(temp_model_dir, generated_cellml_model_factory):
    """
    Same state-init parameter update check for Python solver helper.
    """
    cellml_path = generated_cellml_model_factory("3compartment", "3compartment_parameters.csv")

    py_generator = PythonGenerator(
        cellml_path,
        output_dir=temp_model_dir,
        module_name="three_compartment_py",
    )
    python_model_path = py_generator.generate()

    dt = 0.01
    sim_time = 0.1
    pre_time = 0.0
    solver_info = {"method": "BDF", "rtol": 1e-6, "atol": 1e-8}

    helper = get_simulation_helper(
        model_path=python_model_path,
        model_type="python",
        solver="solve_ivp",
        dt=dt,
        sim_time=sim_time,
        solver_info=solver_info,
        pre_time=pre_time,
    )

    q_name = _find_state_series_name(helper, "q_lv")

    helper.update_times(dt, 0.0, sim_time, pre_time)
    helper.set_param_vals(["global/q_lv_init"], [2e-4])
    helper.reset_states()
    q0_lo = _run_and_get_initial_state(helper, q_name)
    assert np.isclose(q0_lo, 2e-4, rtol=0.0, atol=1e-10), (
        f"Python solver: expected q_lv(0)=2e-4, got {q0_lo}"
    )

    _ = helper.get_all_results_dict()
    helper.reset_and_clear()
    cached_results = helper.get_all_results_dict()
    assert q_name in cached_results, "Python solver: cached results missing q_lv state series"

    helper.update_times(dt, 0.0, sim_time, pre_time)
    helper.set_param_vals(["global/q_lv_init"], [8e-4])
    helper.reset_states()
    q0_hi = _run_and_get_initial_state(helper, q_name)
    assert np.isclose(q0_hi, 8e-4, rtol=0.0, atol=1e-10), (
        f"Python solver: expected q_lv(0)=8e-4, got {q0_hi}"
    )


@pytest.fixture(scope="function")
def temp_model_dir(request):
    """Create a persistent per-test directory for generated Python models."""
    output_root = os.path.join(os.path.dirname(__file__), "test_outputs")
    safe_nodeid = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.nodeid).strip("._") or "unnamed_test"
    model_dir = os.path.join(output_root, safe_nodeid, "python_models")
    shutil.rmtree(model_dir, ignore_errors=True)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


@pytest.mark.parametrize("model_name,input_param_file", [
    ("3compartment", "3compartment_parameters.csv"),
    ("SN_simple", "SN_simple_parameters.csv"),
])
@pytest.mark.parametrize("solver,solver_info", [
    pytest.param(
        "CVODE_opencor",
        {"MaximumStep": 0.0001},
        marks=pytest.mark.need_opencor,
    ),  # OpenCOR
    ("CVODE_myokit", {"MaximumStep": 0.0001}),  # Myokit
])
def test_cellml_solvers(model_name, input_param_file, solver, solver_info, generated_cellml_model_factory):
    """
    Test CellML solvers (OpenCOR CVODE_opencor and Myokit CVODE).
    
    Args:
        model_name: Name of the model for test identification
        model_path: Path to the CellML model file
        solver: Solver name ('CVODE_opencor' for OpenCOR, 'CVODE_myokit' for Myokit)
        solver_info: Solver configuration dictionary
    """
    # Skip OpenCOR tests if OpenCOR is not available
    # Check if model file exists
    full_model_path = generated_cellml_model_factory(model_name, input_param_file, solver=solver)
    
    # Simulation parameters
    dt = 0.01
    sim_time = 1.0  # Short simulation for testing
    pre_time = 0.0

    try:
        helper = get_simulation_helper(
            model_path=full_model_path,
            model_type="cellml_only",
            solver=solver,
            dt=dt,
            sim_time=sim_time,
            solver_info=solver_info,
            pre_time=pre_time,
        )
    except RuntimeError as e:
        if solver == "CVODE_opencor":
            pytest.skip(f"{solver} solver not available: {e}")
        pytest.fail(f"{solver} solver not available: {e}")
    
    # Run simulation
    try:
        result = helper.run()
    except RuntimeError as e:
        pytest.fail(f"{solver} simulation failed for {model_name}: {e}")
    assert result, f"{solver} simulation failed for {model_name}"
    
    # Verify results
    results = helper.get_all_results(flatten=False)
    variables = helper.get_all_variable_names()
    
    assert len(variables) > 0, f"No variables returned for {model_name} with {solver}"
    assert len(results) > 0, f"No results returned for {model_name} with {solver}"
    
    # Check that results have expected shape
    for i, var_name in enumerate(variables):
        var_result = results[i][0]
        if isinstance(var_result, np.ndarray):
            assert len(var_result) > 0, f"Empty result for variable {var_name}"


@pytest.mark.parametrize("model_name,input_param_file", [
    ("3compartment", "3compartment_parameters.csv"),
    ("SN_simple", "SN_simple_parameters.csv"),
])
def test_python_BDF_solver(model_name, input_param_file, temp_model_dir, generated_cellml_model_factory):
    """
    Test Python BDF solver on Python models generated from CellML.
    
    Args:
        model_name: Name of the model for test identification
        model_path: Path to the CellML model file
        temp_model_dir: Temporary directory for generated Python models
    """
    # Check if model file exists
    full_model_path = generated_cellml_model_factory(model_name, input_param_file)
    
    # Generate Python model from CellML
    try:
        py_generator = PythonGenerator(
            full_model_path,
            output_dir=temp_model_dir,
            module_name=model_name
        )
        python_model_path = py_generator.generate()
    except Exception as e:
        pytest.fail(f"Failed to generate Python model: {e}")
    
    # Check if Python model was generated
    if not os.path.exists(python_model_path):
        pytest.fail(f"Python model not generated: {python_model_path}")
    
    # Simulation parameters
    dt = 0.01
    sim_time = 1.0  # Short simulation for testing
    pre_time = 0.0
    solver_info = {
        "method": "BDF",  # Use BDF (Backward Differentiation Formula)
        "max_step": dt,  # Maximum step size
    }
    
    # Run simulation with Python BDF solver
    helper = get_simulation_helper(model_path=python_model_path, model_type="python", solver="solve_ivp", dt=dt, sim_time=sim_time, solver_info=solver_info, pre_time=pre_time)
    
    # Run simulation
    result = helper.run()
    assert result, f"Python RK4 simulation failed for {model_name}"
    
    # Verify results
    results = helper.get_all_results(flatten=False)
    variables = helper.get_all_variable_names()
    
    assert len(variables) > 0, f"No variables returned for {model_name} with Python RK4"
    assert len(results) > 0, f"No results returned for {model_name} with Python RK4"
    
    # Check that results have expected shape
    for i, var_name in enumerate(variables):
        var_result = results[i][0]
        if isinstance(var_result, np.ndarray):
            assert len(var_result) > 0, f"Empty result for variable {var_name}"


def _run_all_solvers_and_compare(model_name, full_model_path_cellml, temp_model_dir, dt=0.01, sim_time=1.0,
                                  pre_time=0.0, tolerance=0.01, include_casadi=False, include_aadc=False,
                                  aadc_method="semi_implicit"):
    """
    Run all solvers on a model and compare outputs.

    Backends exercised:
      - CVODE_myokit          (always required)
      - CVODE_opencor         (skipped gracefully if OpenCOR unavailable)
      - solve_ivp_BDF         (Python model, scipy BDF)
      - casadi_integrator_cvodes (Python model, CasADi cvodes; only when include_casadi=True,
                                  skipped gracefully if CasADi unavailable or symbolic eval fails)
      - aadc_semi_implicit    (AADC Python model; only when include_aadc=True, skipped
                                  gracefully if the aadc package is not installed)

    The Python model (.py) is generated once and reused by both Python-family backends.
    The AADC model is generated separately (different codegen — iif/aadc.math etc.) into an
    ``aadc/`` subdir so it does not clobber the plain Python module.

    Returns:
        Tuple of (results dict, comparison_results dict, helpers dict)
    """
    helpers = {}
    results = {}

    python_family_keys = ["solve_ivp_BDF"]
    if include_casadi:
        python_family_keys.append("casadi_integrator_cvodes")

    # Generate Python model once; reused by Python-family backends
    python_model_path = None
    try:
        py_generator = PythonGenerator(
            full_model_path_cellml,
            output_dir=temp_model_dir,
            module_name=model_name,
        )
        python_model_path = py_generator.generate()
    except Exception as e:
        for key in python_family_keys:
            results[key] = {
                "success": False,
                "skipped": True,
                "reason": f"Python model generation failed: {e}",
            }

    # Generate the AADC model (separate codegen) into its own subdir.
    aadc_model_path = None
    if include_aadc:
        try:
            aadc_dir = os.path.join(temp_model_dir, "aadc")
            os.makedirs(aadc_dir, exist_ok=True)
            aadc_generator = PythonGenerator(
                full_model_path_cellml,
                output_dir=aadc_dir,
                module_name=model_name,
                aadc_compat=True,
            )
            aadc_model_path = aadc_generator.generate()
        except Exception as e:
            results["aadc_semi_implicit"] = {
                "success": False,
                "skipped": True,
                "reason": f"AADC model generation failed: {e}",
            }

    # (helper_key, solver_arg, model_type, model_path, solver_info)
    backends = [
        # Tight rtol/atol on the CVODE reference too: myokit's CVODE default is rel_tol=1e-4
        # (== the 0.01% comparison gate), so without this the reference trajectory is only
        # converged to the gate and cross-solver agreement is build-dependent (passes locally,
        # fails on CI's sundials build). Both solvers must be well-converged for a stable compare.
        ("CVODE_opencor",  "CVODE_opencor",    "cellml_only",   full_model_path_cellml, {"MaximumStep": 0.0001, "rtol": 1e-8, "atol": 1e-10}),
        ("CVODE_myokit",   "CVODE_myokit",     "cellml_only",   full_model_path_cellml, {"MaximumStep": 0.0001, "rtol": 1e-8, "atol": 1e-10}),
        ("solve_ivp_BDF",  "solve_ivp",        "python",        python_model_path,      {"method": "BDF", "max_step": 0.0001, "rtol": 1e-8, "atol": 1e-10}),
    ]
    if include_casadi:
        backends.append(
            ("casadi_integrator_cvodes", "casadi_integrator", "casadi_python", python_model_path, {"method": "cvodes"})
        )
    if include_aadc and aadc_model_path is not None:
        backends.append(
            ("aadc_semi_implicit", "aadc_semi_implicit", "aadc_python", aadc_model_path, {"method": aadc_method})
        )

    # Backends where unavailability is a graceful skip rather than a test failure
    skip_on_error = {"CVODE_opencor", "casadi_integrator_cvodes", "aadc_semi_implicit"}

    for helper_key, solver, model_type, model_path, solver_info in backends:
        # Skip Python-family backends if model generation already failed
        if helper_key in results and results[helper_key].get("skipped"):
            continue
        try:
            helper = get_simulation_helper(
                model_path=model_path,
                model_type=model_type,
                solver=solver,
                dt=dt,
                sim_time=sim_time,
                solver_info=solver_info,
                pre_time=pre_time,
            )
            result = helper.run()
            assert result, f"{helper_key} simulation failed"
            helpers[helper_key] = helper
            results[helper_key] = {"success": True, "variables": len(helper.get_all_variable_names())}
        except Exception as e:
            results[helper_key] = {"success": False, "error": str(e)}
            if helper_key in skip_on_error:
                results[helper_key]["skipped"] = True
                results[helper_key]["reason"] = f"{helper_key} backend unavailable: {e}"
                continue
            pytest.fail(f"{model_name} {helper_key} failed: {e}")

    # Print summary
    print(f"\n{'='*80}")
    print(f"SOLVER TEST SUMMARY - {model_name} model")
    print("="*80)
    for solver_name, result in results.items():
        if result.get("success"):
            print(f"✓ {solver_name}: SUCCESS ({result.get('variables', 'N/A')} variables)")
        elif result.get("skipped"):
            print(f"⊘ {solver_name}: SKIPPED ({result.get('reason', 'N/A')})")
        else:
            print(f"✗ {solver_name}: FAILED ({result.get('error', 'N/A')})")

    # Compare all successful helpers against CVODE_myokit as reference
    ref_helper = helpers["CVODE_myokit"]
    comparison_results = {}

    for solver_name, other_helper in helpers.items():
        if solver_name == "CVODE_myokit":
            continue

        print(f"\n{'='*80}")
        print(f"Comparing CVODE_myokit vs {solver_name}")
        print("="*80)

        comp_result = _compare_solver_results(ref_helper, "CVODE_myokit", other_helper, solver_name, tolerance=tolerance)
        comparison_results[solver_name] = comp_result

        print(f"Matched variables: {comp_result['matched_count']}")
        print(f"Compared variables: {comp_result['compared_count']}")
        print(f"Maximum relative error: {comp_result['max_rel_error']:.6f}%")

        if comp_result['failed_vars']:
            print(f"\nVariables exceeding {tolerance}% tolerance ({len(comp_result['failed_vars'])}):")
            for failed in comp_result['failed_vars'][:10]:
                print(f"  {failed['ref_var']} / {failed['other_var']}: {failed['max_rel_error']:.6f}%")
            if len(comp_result['failed_vars']) > 10:
                print(f"  ... and {len(comp_result['failed_vars']) - 10} more")

        sorted_comps = sorted(comp_result['comparisons'], key=lambda x: x['max_rel_error'], reverse=True)
        print(f"\nTop 10 largest differences:")
        print(f"{'Reference Variable':<40} {'Other Variable':<40} {'Max Rel Error %':>15}")
        print("-" * 95)
        for comp in sorted_comps[:10]:
            print(f"{comp['ref_var'][:39]:<40} {comp['other_var'][:39]:<40} {comp['max_rel_error']:>15.6f}")

    print("\n" + "="*80)

    return results, comparison_results, helpers


@pytest.mark.integration
@pytest.mark.slow
# .cellml gets converted to .py for python solvers
@pytest.mark.parametrize("model_name,input_param_file,sim_time,include_casadi", [
    ("3compartment", "3compartment_parameters.csv", 0.1, False),
    ("SN_simple",    "SN_simple_parameters.csv",    1.0, False),
    # Lotka-Volterra has no conditional expressions so CasADi symbolic eval works
    ("Lotka_Volterra", "Lotka_Volterra_parameters.csv", 5.0, True),
])
def test_all_solvers(model_name, input_param_file, sim_time, include_casadi, temp_model_dir, generated_cellml_model_factory):
    """
    Integration test: Run all solvers on a model and compare outputs.

    Backends exercised:
    1. CVODE_myokit          — Myokit CVODE (reference)
    2. CVODE_opencor         — OpenCOR CVODE (skipped if unavailable)
    3. solve_ivp_BDF         — SciPy BDF on generated Python model
    4. casadi_integrator_cvodes — CasADi cvodes (only for Lotka_Volterra; skipped if unavailable)

    Models with conditional expressions (3compartment, SN_simple) cannot use CasADi because
    CasADi requires fully symbolic expressions with no Python-level branching.

    Checks:
    - All active backends agree within 0.01% relative error vs CVODE_myokit.
    - For 3compartment: Myokit↔OpenCOR initial states match (when OpenCOR present).
    - Python-family initial states match the reference (OpenCOR or Myokit).
    """
    cellml_path = generated_cellml_model_factory(model_name, input_param_file)
    results, comparison_results, helpers = _run_all_solvers_and_compare(
        model_name, cellml_path, temp_model_dir, sim_time=sim_time, tolerance=0.01,
        include_casadi=include_casadi,
    )

    # Check Myokit↔OpenCOR initial states when both are available
    if "CVODE_myokit" in helpers and "CVODE_opencor" in helpers:
        init_check = _check_initial_states(
            helpers["CVODE_myokit"],
            helpers["CVODE_opencor"],
            model_name
        )
        mismatches = init_check['mismatches']
        significant_mismatches = [m for m in mismatches if m[4] > 0.01]
        if significant_mismatches:
            mismatch_summary = f"Found {len(significant_mismatches)} significant initial state mismatches (>0.01%):\\n"
            for m in significant_mismatches[:5]:
                mk_var, oc_var, mk_val, oc_val, diff_pct = m
                mismatch_summary += f"  {mk_var} / {oc_var}: {diff_pct:.6f}%\\n"
            if len(significant_mismatches) > 5:
                mismatch_summary += f"  ... and {len(significant_mismatches) - 5} more"
            pytest.fail(f"Initial state validation failed for {model_name}:\\n{mismatch_summary}")
        else:
            print(f"✓ Myokit↔OpenCOR initial states match within 0.01% tolerance ({len(mismatches)} minor differences found)")

    # Check Python-family initial states for all models
    py_init_check = _check_python_family_initial_states(helpers, model_name, tolerance=0.01)
    py_mismatches = py_init_check['mismatches']
    if py_mismatches:
        mismatch_summary = f"Found {len(py_mismatches)} Python-family initial state mismatches (>0.01%):\\n"
        for state_name, helper_key, ref_val, other_val, diff_pct in py_mismatches[:5]:
            mismatch_summary += f"  {state_name} [{helper_key}]: {diff_pct:.6f}%\\n"
        if len(py_mismatches) > 5:
            mismatch_summary += f"  ... and {len(py_mismatches) - 5} more"
        pytest.fail(f"Python-family initial state validation failed for {model_name}:\\n{mismatch_summary}")
    else:
        print(f"✓ Python-family initial states match within 0.01% tolerance")

    # Assert that all comparisons are within tolerance
    for solver_name, comp_result in comparison_results.items():
        if comp_result['failed_vars']:
            failed_msg = f"{solver_name} has {len(comp_result['failed_vars'])} variables exceeding 0.01% tolerance. "
            failed_msg += f"Maximum error: {comp_result['max_rel_error']:.6f}%"
            pytest.fail(failed_msg)


@pytest.mark.integration
@pytest.mark.slow
def test_aadc_vs_cvode_3compartment(aadc_licensed, temp_model_dir,
                                    generated_cellml_model_factory):
    """The stiff 3compartment model integrated with the AADC ``bdf`` solver should track
    the CVODE_myokit reference trajectory.

    Why this test: the PR's own AADC stiff test only asserts the 3compartment run
    produces no NaNs — it does not check the values are *correct*. This compares the
    integrated ODE states against CVODE_myokit, which is what "same outputs with
    CVODE and with AADC" requires.

    This deliberately exercises ``method='bdf'`` (scipy's implicit BDF with an exact dense
    Jacobian supplied by AADC) and not ``method='semi_implicit'``. Semi-implicit Euler is
    first-order with numerical diagonal damping (``y += dt*f/(1+dt*lam)``); that damping
    suppresses the stiff states, which in this cardiovascular model carry physiologically
    important fast dynamics, so it deviates from CVODE by ~35% relative-L2 and cannot meet
    a meaningful tolerance. ``bdf`` is the accurate stiff path and tracks CVODE closely.
    """
    cellml_path = generated_cellml_model_factory("3compartment", "3compartment_parameters.csv")
    results, _, helpers = _run_all_solvers_and_compare(
        "3compartment", cellml_path, temp_model_dir, dt=0.001, sim_time=1.0,
        tolerance=5.0, include_aadc=True, aadc_method="bdf",
    )

    aadc_result = results.get("aadc_semi_implicit", {})
    if aadc_result.get("skipped"):
        pytest.skip(aadc_result.get("reason", "AADC backend unavailable"))

    assert "aadc_semi_implicit" in helpers, "AADC backend did not run"
    assert "CVODE_myokit" in helpers, "CVODE_myokit reference did not run"

    # Compare the integrated ODE states (state names live on the aadc/"python" side).
    aadc_state_names = list(helpers["aadc_semi_implicit"].state_name_to_idx.keys())
    max_pct, worst_var, compared = _max_rel_l2_error(
        helpers["CVODE_myokit"], helpers["aadc_semi_implicit"],
        only_other_vars=aadc_state_names,
    )
    print(f"3compartment CVODE vs AADC: max rel-L2 error {max_pct:.3f}% on "
          f"{worst_var} ({compared} states compared)")

    assert compared > 0, "No ODE states were matched between CVODE and AADC"
    tol_pct = 1.0
    assert max_pct < tol_pct, (
        f"AADC bdf deviates from CVODE by {max_pct:.3f}% (> {tol_pct}%) "
        f"on state {worst_var}"
    )


def _max_rel_l2_error(ref_helper, other_helper, only_other_vars=None):
    """Max over matched time-series variables of the relative L2 error
    ‖ref-other‖ / ‖ref‖, as a percentage. The whole-trajectory L2 norm is robust
    both to variables that pass through zero (which break pointwise relative
    error) and to near-constant variables with a small offset (which break
    range-normalized error). Skips ~zero variables.

    ``only_other_vars`` (optional) restricts the comparison to those "other"-side
    variable names — e.g. the ODE state variables. Auxiliary/algebraic valve
    quantities (e.g. switching valve inductances) are model-representation
    details that legitimately differ between the myokit and casadi backends; the
    integrated states are what the solver actually computes. Returns
    (max_pct, worst_var, compared_count)."""
    ref_vars = ref_helper.get_all_variable_names()
    other_vars = other_helper.get_all_variable_names()
    ref_results = ref_helper.get_all_results(flatten=False)
    other_results = other_helper.get_all_results(flatten=False)
    ref_dict = {v: ref_results[i][0] for i, v in enumerate(ref_vars)}
    other_dict = {v: other_results[i][0] for i, v in enumerate(other_vars)}
    mapping = _match_variables(ref_vars, "myokit", other_vars, "python")

    worst_pct, worst_var, compared = 0.0, None, 0
    for ref_var, other_var in mapping.items():
        if only_other_vars is not None and other_var not in only_other_vars:
            continue
        a = _to_numpy(ref_dict[ref_var])
        b = _to_numpy(other_dict[other_var])
        if len(a) <= 1 or len(b) <= 1:
            continue  # constant / scalar
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        ref_norm = float(np.linalg.norm(a))
        if ref_norm <= 1e-12:
            continue  # ~zero variable throughout
        pct = float(np.linalg.norm(a - b) / ref_norm) * 100.0
        compared += 1
        if pct > worst_pct:
            worst_pct, worst_var = pct, ref_var
    return worst_pct, worst_var, compared


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("model_name,input_param_file,sim_time,dt,tolerance", [
    # casadi_python semi_implicit_euler is a first-order scheme, so it converges to
    # the cellml CVODE_myokit reference as dt -> 0. Lotka_Volterra (smooth) matches
    # tightly at a modest dt; 3compartment_nonstiff has a fast flow state (par/v)
    # that needs a smaller dt — its state error shrinks ~first-order with dt
    # (≈25%/14%/7% at dt=1e-4/3e-5/1e-5), so a small dt + looser bound is used.
    # (The *stiff* 3compartment is excluded: there the diagonal damping trades
    # transient accuracy for stability — see test_param_id
    # .test_3compartment_stiff_casadi_semi_implicit_forward_and_gradient.)
    ("Lotka_Volterra", "Lotka_Volterra_parameters.csv", 2.0, 1e-4, 1.0),
    ("3compartment_nonstiff", "3compartment_nonstiff_parameters.csv", 0.05, 1e-5, 10.0),
])
def test_cvode_myokit_vs_casadi_semi_implicit_euler(
    model_name, input_param_file, sim_time, dt, tolerance,
    temp_model_dir, generated_cellml_model_factory,
):
    """cellml CVODE_myokit and casadi_python semi_implicit_euler integrate the same
    ODE-state trajectories on non-stiff models. Compared with a relative L2
    (whole-trajectory) error so states that pass through zero or are near-constant
    don't spuriously dominate. Only the integrated ODE states are compared —
    auxiliary/algebraic valve quantities differ by model representation between
    the myokit and casadi backends, not by the integration."""
    pytest.importorskip("casadi")

    cellml_path = generated_cellml_model_factory(model_name, input_param_file)
    # CasADi-compatible python model (conditionals rewritten for symbolic exec).
    casadi_model_path = PythonGenerator(
        cellml_path, output_dir=temp_model_dir,
        module_name=f"{model_name}_casadi", casadi_compat=True,
    ).generate()

    ref = get_simulation_helper(
        model_path=cellml_path, model_type="cellml_only", solver="CVODE_myokit",
        dt=dt, sim_time=sim_time, pre_time=0.0,
        solver_info={"MaximumStep": 1e-4, "rtol": 1e-8, "atol": 1e-10},
    )
    assert ref.run(), "CVODE_myokit reference simulation failed"

    sie = get_simulation_helper(
        model_path=casadi_model_path, model_type="casadi_python", solver="casadi_integrator",
        dt=dt, sim_time=sim_time, pre_time=0.0,
        solver_info={"method": "semi_implicit_euler"},
    )
    assert sie.run(), "casadi_python semi_implicit_euler simulation failed"

    # The ODE states are the first len(STATE_INFO) entries of get_all_variable_names
    # (states are listed before algebraic/constant variables).
    state_names = set(sie.get_all_variable_names()[: len(sie.STATE_INFO)])
    worst_pct, worst_var, compared = _max_rel_l2_error(ref, sie, only_other_vars=state_names)
    print(f"\n{model_name}: CVODE_myokit vs casadi semi_implicit_euler (dt={dt}, "
          f"sim_time={sim_time}): {compared} states compared, "
          f"worst {worst_var}={worst_pct:.3f}% (tol {tolerance}%)")

    assert compared > 0, "no state variables were compared"
    assert worst_pct < tolerance, (
        f"{model_name}: {worst_var} differs by {worst_pct:.3f}% (relative L2) "
        f"(> {tolerance}%) between CVODE_myokit and casadi semi_implicit_euler"
    )

@pytest.mark.integration
@pytest.mark.slow
def test_van_der_pol_stiff_semi_implicit_euler_convergence(
    temp_model_dir, generated_cellml_model_factory
):
    """Van der Pol (mu=100) — a classic stiff benchmark.

    casadi_python semi_implicit_euler is a first-order, fixed-step damped scheme:
    at a coarse dt it is wildly inaccurate on this stiff system, but it *converges*
    to the cellml CVODE_myokit reference as dt is refined. This is exactly why a
    convergence study is required before trusting semi_implicit_euler on stiff
    models (see the note in tutorial/docs/parameter-identification.md)."""
    pytest.importorskip("casadi")

    cellml_path = generated_cellml_model_factory("VanDerPol", "VanDerPol_parameters.csv")
    casadi_model_path = PythonGenerator(
        cellml_path, output_dir=temp_model_dir,
        module_name="VanDerPol_casadi", casadi_compat=True,
    ).generate()

    sim_time = 1.0
    dts = (1e-3, 5e-4, 2.5e-4)
    errors = []
    for dt in dts:
        ref = get_simulation_helper(
            model_path=cellml_path, model_type="cellml_only", solver="CVODE_myokit",
            dt=dt, sim_time=sim_time, pre_time=0.0,
            solver_info={"MaximumStep": 1e-4, "rtol": 1e-9, "atol": 1e-11},
        )
        assert ref.run(), "CVODE_myokit reference simulation failed"
        sie = get_simulation_helper(
            model_path=casadi_model_path, model_type="casadi_python", solver="casadi_integrator",
            dt=dt, sim_time=sim_time, pre_time=0.0,
            solver_info={"method": "semi_implicit_euler"},
        )
        assert sie.run(), "casadi_python semi_implicit_euler simulation failed"
        a = np.array(ref.get_results(["VanDerPol/x"], flatten=True)[0], dtype=float)
        b = np.array(sie.get_results(["VanDerPol/x"], flatten=True)[0], dtype=float)
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        errors.append(100.0 * np.linalg.norm(a - b) / np.linalg.norm(a))

    print(f"\nVan der Pol (mu=100) semi_implicit_euler vs CVODE_myokit, x relative-L2 "
          f"error by dt {list(dts)}: {[round(e, 3) for e in errors]}%")

    # First-order convergence: each dt refinement reduces the error.
    assert errors[0] > errors[1] > errors[2], f"not converging with dt: {errors}"
    # And the finest dt is well-converged (so the scheme is correct, just dt-sensitive).
    assert errors[-1] < 2.0, f"finest-dt error {errors[-1]:.3f}% too high"
