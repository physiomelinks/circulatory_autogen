"""
Tests that AADC AST transforms correctly handle conditionals in generated code.

Mirrors test_casadi_conditionals.py: verifies that aadc_python code generation
transforms if/else → aadc.iif, and/or → aadc.iand/ior, math → aadc.math,
floor → math.floor(_aadc_passive), pow(x,2) → x*x.

Uses 3compartment model (has valve conditionals with leq_func/geq_func/and_func).
"""
import importlib.util
import os

import pytest

from scripts.script_generate_with_new_architecture import generate_with_new_architecture


def _model_config(model_type, temp_generated_models_dir):
    config = {
        "file_prefix": "3compartment",
        "input_param_file": "3compartment_parameters.csv",
        "model_type": model_type,
        "generated_models_dir": temp_generated_models_dir,
    }
    if model_type == "aadc_python":
        config.update({
            "solver": "aadc_semi_implicit",
            "solver_info": {"method": "semi_implicit"},
        })
    else:
        config.update({
            "solver": "solve_ivp",
            "solver_info": {"method": "RK45"},
        })
    return config


def _load_model(model_path):
    spec = importlib.util.spec_from_file_location("gen_model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.integration
@pytest.mark.slow
def test_aadc_python_has_iif_not_if_else(base_user_inputs, resources_dir, temp_generated_models_dir):
    """aadc_python model should use aadc.iif, not Python if/else."""
    config = base_user_inputs.copy()
    config.update(_model_config("aadc_python", temp_generated_models_dir))

    success = generate_with_new_architecture(False, config)
    assert success, "aadc_python generation should succeed"

    model_path = os.path.join(temp_generated_models_dir, "3compartment", "3compartment.py")
    assert os.path.exists(model_path)

    with open(model_path, encoding="utf-8") as f:
        source = f.read()

    # Should have aadc.iif
    assert "aadc.iif" in source, "AADC model should use aadc.iif for conditionals"
    # Should have aadc.math.cos (not bare cos)
    assert "aadc.math.cos" in source, "AADC model should use aadc.math.cos"
    # Should NOT have ca.if_else
    assert "ca.if_else" not in source, "AADC model should not use CasADI ca.if_else"


@pytest.mark.integration
@pytest.mark.slow
def test_aadc_python_transforms_all_patterns(base_user_inputs, resources_dir, temp_generated_models_dir):
    """All conditional/math patterns should be transformed."""
    config = base_user_inputs.copy()
    config.update(_model_config("aadc_python", temp_generated_models_dir))

    success = generate_with_new_architecture(False, config)
    assert success

    model_path = os.path.join(temp_generated_models_dir, "3compartment", "3compartment.py")
    with open(model_path, encoding="utf-8") as f:
        source = f.read()

    # Count transforms
    assert source.count("aadc.iif") >= 10, "Should have many iif calls (valve logic)"
    assert "aadc.math.cos" in source, "cos should be transformed"
    assert "_aadc_passive" in source, "floor should use _aadc_passive"
    # No bare pow() calls (converted to x*x or x**n)
    import re
    bare_pow = re.findall(r'(?<!\w)pow\s*\(', source)
    assert len(bare_pow) == 0, f"pow() should be converted, found {len(bare_pow)}"


@pytest.mark.integration
@pytest.mark.slow
def test_aadc_python_compute_rates_matches_standard(base_user_inputs, resources_dir, temp_generated_models_dir):
    """compute_rates should give identical results for standard python and aadc_python at same point."""
    import numpy as np

    # Generate standard python model
    config_py = base_user_inputs.copy()
    config_py.update(_model_config("python", temp_generated_models_dir))
    config_py["generated_models_dir"] = os.path.join(temp_generated_models_dir, "py")
    os.makedirs(config_py["generated_models_dir"], exist_ok=True)
    success_py = generate_with_new_architecture(False, config_py)
    assert success_py

    # Generate aadc_python model
    config_aadc = base_user_inputs.copy()
    config_aadc.update(_model_config("aadc_python", temp_generated_models_dir))
    config_aadc["generated_models_dir"] = os.path.join(temp_generated_models_dir, "aadc")
    os.makedirs(config_aadc["generated_models_dir"], exist_ok=True)
    success_aadc = generate_with_new_architecture(False, config_aadc)
    assert success_aadc

    # Load both
    mod_py = _load_model(os.path.join(config_py["generated_models_dir"], "3compartment", "3compartment.py"))
    mod_aadc = _load_model(os.path.join(config_aadc["generated_models_dir"], "3compartment", "3compartment.py"))

    n = mod_py.STATE_COUNT
    assert mod_aadc.STATE_COUNT == n

    # Initialize
    s_py = mod_py.create_states_array()
    r_py = [0.0] * n
    v_py = mod_py.create_variables_array()
    mod_py.initialise_variables(s_py, r_py, v_py)
    mod_py.compute_computed_constants(v_py)

    s_aadc = mod_aadc.create_states_array()
    r_aadc = [0.0] * n
    v_aadc = mod_aadc.create_variables_array()
    mod_aadc.initialise_variables(s_aadc, r_aadc, v_aadc)
    mod_aadc.compute_computed_constants(v_aadc)

    # Compute rates at same point
    mod_py.compute_rates(0.0, list(s_py), r_py, list(v_py))
    mod_aadc.compute_rates(0.0, list(s_aadc), r_aadc, list(v_aadc))

    # Compare
    for i in range(n):
        if abs(r_py[i]) > 1e-10:
            rel = abs(r_py[i] - r_aadc[i]) / abs(r_py[i])
            assert rel < 1e-10, f"Rate[{i}] differs: py={r_py[i]:.6e} aadc={r_aadc[i]:.6e} rel={rel:.2e}"


@pytest.mark.integration
@pytest.mark.slow
def test_aadc_python_idouble_compute_rates_matches_float(base_user_inputs, resources_dir, temp_generated_models_dir):
    """compute_rates with idouble should give same results as with float."""
    aadc = pytest.importorskip("aadc")
    import numpy as np

    config = base_user_inputs.copy()
    config.update(_model_config("aadc_python", temp_generated_models_dir))

    success = generate_with_new_architecture(False, config)
    assert success

    mod = _load_model(os.path.join(temp_generated_models_dir, "3compartment", "3compartment.py"))
    n = mod.STATE_COUNT

    # Float
    s_f = mod.create_states_array()
    r_f = [0.0] * n
    v_f = mod.create_variables_array()
    mod.initialise_variables(s_f, r_f, v_f)
    mod.compute_computed_constants(v_f)
    mod.compute_rates(0.0, list(s_f), r_f, list(v_f))

    # idouble
    import math
    s_id = [aadc.idouble(float(s)) for s in s_f]
    r_id = [aadc.idouble(0.0)] * n
    v_id = [aadc.idouble(float(v) if not math.isnan(float(v)) else 0.0) for v in v_f]
    mod.compute_rates(0.0, s_id, r_id, list(v_id))

    for i in range(n):
        f_val = r_f[i]
        id_val = float(r_id[i]) if hasattr(r_id[i], 'val') else float(r_id[i])
        if abs(f_val) > 1e-10:
            rel = abs(f_val - id_val) / abs(f_val)
            assert rel < 1e-10, f"Rate[{i}] float={f_val:.6e} idouble={id_val:.6e} rel={rel:.2e}"
