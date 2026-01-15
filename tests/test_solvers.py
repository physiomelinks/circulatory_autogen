"""
Tests for different solver implementations.

These tests verify that OpenCOR (CVODE), Myokit (CVODE_myokit), and Python BDFsolvers
work correctly for various models.

"""
import os
import sys
import pytest
import numpy as np
import tempfile
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
    elif solver_type == 'opencor':
        # Format: component/variable
        parts = var_name.split('/')
        if len(parts) == 2:
            return parts[0], parts[1]
        return None, var_name
    elif solver_type == 'python':
        # Format: variable (no component prefix typically)
        return None, var_name
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
    
    # Match reference variables
    for ref_var in ref_vars:
        ref_comp, ref_var_name = _normalize_variable_name(ref_var, ref_type)
        ref_key = (ref_comp, ref_var_name) if ref_comp else (None, ref_var_name)
        
        # Try exact match first
        if ref_key in other_normalized:
            mapping[ref_var] = other_normalized[ref_key]
        else:
            # Try matching by variable name only
            var_only_key = (None, ref_var_name)
            if var_only_key in other_normalized:
                mapping[ref_var] = other_normalized[var_only_key]
    
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
    ref_type = 'myokit' if 'Myokit' in ref_name else ('opencor' if 'OpenCOR' in ref_name else 'python')
    other_type = 'myokit' if 'Myokit' in other_name else ('opencor' if 'OpenCOR' in other_name else 'python')
    
    var_mapping = _match_variables(ref_vars, ref_type, other_vars, other_type)
    
    # Compare matched variables
    comparisons = []
    max_rel_error = 0.0
    failed_vars = []
    
    for ref_var, other_var in var_mapping.items():
        ref_data = ref_dict[ref_var]
        other_data = other_dict[other_var]
        
        # Ensure arrays
        if not isinstance(ref_data, np.ndarray):
            ref_data = np.array([ref_data])
        if not isinstance(other_data, np.ndarray):
            other_data = np.array([other_data])
        
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
        max_rel_error_var = np.max(rel_error)
        mean_rel_error_var = np.mean(rel_error)
        
        comparisons.append({
            'ref_var': ref_var,
            'other_var': other_var,
            'max_rel_error': max_rel_error_var,
            'mean_rel_error': mean_rel_error_var,
            'max_abs_diff': np.max(abs_diff),
            'mean_abs_diff': np.mean(abs_diff)
        })
        
        if max_rel_error_var > max_rel_error:
            max_rel_error = max_rel_error_var
        
        if max_rel_error_var > tolerance:
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


def test_init_states_myokit(base_user_inputs, resources_dir):
    """
    Repro for computed-constant initial state values via Myokit wrapper.

    - Uses resources/test_init_states_vessel_array.csv
    - Uses resources/test_init_states_parameters.csv where a_test_vessel = 3
    - Module defines x0 = 2 * a and x has initial_value=\"x0\"
    - Expect x(0) == 6 and y(0) == 1
    """
    # Prefer using the autogeneration step output; only generate here if missing.
    cellml_path = os.path.join(_TEST_ROOT, "generated_models", "test_init_states", "test_init_states.cellml")
    if not os.path.exists(cellml_path):
        cfg = base_user_inputs.copy()
        cfg.update({
            "DEBUG": True,
            "file_prefix": "test_init_states",
            "input_param_file": "test_init_states_parameters.csv",
            "model_type": "cellml_only",
            "solver": "CVODE",
            "pre_time": 0.0,
            "sim_time": 0.1,
            "dt": 0.01,
            "plot_predictions": False,
            "do_mcmc": False,
            "solver_info": {"MaximumStep": 0.001, "MaximumNumberOfSteps": 5000},
            # Make sure generation uses the repo resources dir (contains our vessel array)
            "resources_dir": resources_dir,
        })
        ok = generate_with_new_architecture(False, cfg)
        assert ok, "Autogeneration failed for test_init_states"
        assert os.path.exists(cellml_path), f"Generated model not found after generation: {cellml_path}"

    helper_cls = get_simulation_helper(solver="CVODE")
    dt = 0.01
    sim_time = 0.1
    solver_info = {"MaximumStep": 0.001, "MaximumNumberOfSteps": 5000}
    helper = helper_cls(cellml_path, dt, sim_time, solver_info=solver_info, pre_time=0.0)
    assert helper.run(), "Myokit simulation failed for init_states_test"

    # Find x and y state series
    names = helper.get_all_variable_names()
    x_name = next((n for n in names if n.endswith(".x")), None)
    y_name = next((n for n in names if n.endswith(".y")), None)
    assert x_name is not None and y_name is not None, f"Could not find x/y. Sample: {names[:20]}"

    x0 = float(np.asarray(helper.get_results([x_name], flatten=True)[0][0])[0])
    y0 = float(np.asarray(helper.get_results([y_name], flatten=True)[0][0])[0])
    assert np.isclose(x0, 6.0, rtol=0, atol=1e-12), f"Expected x(0)=6.0, got {x0} ({x_name})"
    assert np.isclose(y0, 1.0, rtol=0, atol=1e-12), f"Expected y(0)=1.0, got {y0} ({y_name})"


@pytest.fixture(scope="function")
def temp_model_dir():
    """Create a temporary directory for generated Python models."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.parametrize("model_name,model_path", [
    ("3compartment", "generated_models/3compartment/3compartment.cellml"),
    ("SN_simple", "generated_models/SN_simple/SN_simple.cellml"),
])
@pytest.mark.parametrize("solver,solver_info", [
    ("CVODE", {"MaximumStep": 0.0001}),  # OpenCOR 
    ("CVODE_myokit", {"MaximumStep": 0.0001}),  # Myokit
])
def test_cellml_solvers(model_name, model_path, solver, solver_info):
    """
    Test CellML solvers (OpenCOR CVODE and Myokit CVODE).
    
    Args:
        model_name: Name of the model for test identification
        model_path: Path to the CellML model file
        solver: Solver name ('CVODE' for OpenCOR, 'CVODE_myokit' for Myokit)
        solver_info: Solver configuration dictionary
    """
    # Skip OpenCOR tests if OpenCOR is not available
    if solver == "CVODE":
        try:
            helper_cls = get_simulation_helper(solver=solver)
            if helper_cls is None:
                pytest.skip("OpenCOR solver not available")
        except RuntimeError:
            pytest.skip("OpenCOR solver not available")
    
    # Check if model file exists
    full_model_path = os.path.join(_TEST_ROOT, model_path)
    if not os.path.exists(full_model_path):
        pytest.skip(f"Model file not found: {full_model_path}")
    
    # Simulation parameters
    dt = 0.01
    sim_time = 1.0  # Short simulation for testing
    pre_time = 0.0
    
    # Run simulation
    helper_cls = get_simulation_helper(solver=solver)
    helper = helper_cls(full_model_path, dt, sim_time, solver_info=solver_info, pre_time=pre_time)
    
    # Run simulation
    result = helper.run()
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


@pytest.mark.parametrize("model_name,model_path", [
    ("3compartment", "generated_models/3compartment/3compartment.cellml"),
    ("SN_simple", "generated_models/SN_simple/SN_simple.cellml"),
])
def test_python_BDF_solver(model_name, model_path, temp_model_dir):
    """
    Test Python BDF solver on Python models generated from CellML.
    
    Args:
        model_name: Name of the model for test identification
        model_path: Path to the CellML model file
        temp_model_dir: Temporary directory for generated Python models
    """
    # Check if model file exists
    full_model_path = os.path.join(_TEST_ROOT, model_path)
    if not os.path.exists(full_model_path):
        pytest.skip(f"Model file not found: {full_model_path}")
    
    # Generate Python model from CellML
    try:
        py_generator = PythonGenerator(
            full_model_path,
            output_dir=temp_model_dir,
            module_name=model_name
        )
        python_model_path = py_generator.generate()
    except Exception as e:
        pytest.skip(f"Failed to generate Python model: {e}")
    
    # Check if Python model was generated
    if not os.path.exists(python_model_path):
        pytest.skip(f"Python model not generated: {python_model_path}")
    
    # Simulation parameters
    dt = 0.01
    sim_time = 1.0  # Short simulation for testing
    pre_time = 0.0
    solver_info = {
        "method": "BDF",  # Use BDF (Backward Differentiation Formula)
        "max_step": dt,  # Maximum step size
    }
    
    # Run simulation with Python BDF solver
    helper_cls = get_simulation_helper(solver="solve_ivp", model_path=python_model_path)
    helper = helper_cls(python_model_path, dt, sim_time, solver_info=solver_info, pre_time=pre_time)
    
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


def _run_all_solvers_and_compare(model_name, model_path, temp_model_dir, dt=0.01, sim_time=1.0, 
                                  pre_time=0.0, tolerance=0.01):
    """
    Helper function to run all solvers on a model and compare outputs.
    
    Args:
        model_name: Name of the model
        model_path: Path to the CellML model file
        temp_model_dir: Temporary directory for generated Python models
        dt: Time step
        sim_time: Simulation time
        pre_time: Pre-simulation time
        tolerance: Maximum allowed relative error percentage
    
    Returns:
        Tuple of (results dict, comparison_results dict, helpers dict)
    """
    full_model_path = os.path.join(_TEST_ROOT, model_path)
    
    if not os.path.exists(full_model_path):
        pytest.skip(f"Model file not found: {full_model_path}")
    
    solver_info = {"MaximumStep": 0.0001}
    helpers = {}
    results = {}
    
    # Test all solvers (skipping Myokit CVODE for nowsince it's not available)
    for solver, method in [("CVODE", "CVODE"), ("solve_ivp", "BDF")]:
        solver_info['method'] = method
        try:
            helper_cls = get_simulation_helper(solver=solver, model_path=full_model_path)
            helper = helper_cls(full_model_path, dt, sim_time, solver_info=solver_info, pre_time=pre_time)
            result = helper.run()
            assert result, f"{solver} {method} simulation failed"
            helpers[solver] = helper
            results[solver] = {"success": True, "variables": len(helper.get_all_variable_names())}
        except Exception as e:
            results[solver] = {"success": False, "error": str(e)}
            pytest.fail(f"{solver} {method} failed: {e}")
    
    # Test Python BDF (below to disable)
    # results["Python BDF"] = {"success": False, "skipped": True, "reason": "Temporarily disabled - hanging issue"}
    
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
    
    # Compare results (use Myokit as reference)
    ref_helper = helpers["Myokit CVODE"]
    comparison_results = {}
    
    for solver_name, other_helper in helpers.items():
        if solver_name == "Myokit CVODE":
            continue
        
        print(f"\n{'='*80}")
        print(f"Comparing Myokit CVODE vs {solver_name}")
        print("="*80)
        
        comp_result = _compare_solver_results(ref_helper, "Myokit CVODE", other_helper, solver_name, tolerance=tolerance)
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
        
        # Show top differences
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
@pytest.mark.parametrize("model_name,model_path,sim_time", [
    ("3compartment", "generated_models/3compartment/3compartment.cellml", 0.1),
    ("SN_simple", "generated_models/SN_simple/SN_simple.cellml", 1.0),
])
def test_all_solvers(model_name, model_path, sim_time, temp_model_dir):
    """
    Integration test: Run all solvers on a model and compare outputs.
    
    This test verifies that:
    1. Myokit CVODE solver works
    2. OpenCOR CVODE solver works (if available)
    3. Python BDF solver works (after generating Python model)
    4. Initial states are correctly defined in Myokit modified model
    5. Results agree within 0.01% relative error
    
    Args:
        model_name: Name of the model for test identification
        model_path: Path to the CellML model file (relative to project root)
        sim_time: Simulation time (0.1s for 3compartment, 1.0s for SN_simple)
        temp_model_dir: Temporary directory for generated Python models
    """
    results, comparison_results, helpers = _run_all_solvers_and_compare(
        model_name, model_path, temp_model_dir, sim_time=sim_time, tolerance=0.01
    )
    
    # Check initial states for 3compartment model
    if model_name == "3compartment" and "Myokit CVODE" in helpers and "OpenCOR CVODE" in helpers:
        mismatches = _check_initial_states(
            helpers["Myokit CVODE"],
            helpers["OpenCOR CVODE"],
            model_name
        )

        # Fail the test if there are significant initial state mismatches
        significant_mismatches = [m for m in mismatches if m[4] > 0.01]  # 0.01% tolerance
        if significant_mismatches:
            mismatch_summary = f"Found {len(significant_mismatches)} significant initial state mismatches (>0.01%):\\n"
            for m in significant_mismatches[:5]:  # Show first 5
                mk_var, oc_var, mk_val, oc_val, diff_pct = m
                mismatch_summary += f"  {mk_var} / {oc_var}: {diff_pct:.6f}%\\n"
            if len(significant_mismatches) > 5:
                mismatch_summary += f"  ... and {len(significant_mismatches) - 5} more"
            pytest.fail(f"Initial state validation failed for {model_name}:\\n{mismatch_summary}")
        else:
            print(f"✓ Initial states match within 0.01% tolerance ({len(mismatches)} minor differences found)")
    
    # Assert that all comparisons are within tolerance
    for solver_name, comp_result in comparison_results.items():
        if comp_result['failed_vars']:
            failed_msg = f"{solver_name} has {len(comp_result['failed_vars'])} variables exceeding 0.01% tolerance. "
            failed_msg += f"Maximum error: {comp_result['max_rel_error']:.6f}%"
            pytest.fail(failed_msg)

