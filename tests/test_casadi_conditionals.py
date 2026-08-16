"""
Tests that CasADi symbolic evaluation fails on Python ternaries and succeeds with ca.if_else codegen.

Uses pid_control: a small model whose only piecewise logic is a three-way clamp in
compute_variables (not the full 3compartment heart module).
"""
import importlib.util
import os

import pytest

from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture

CASADI_TRUTH_VALUE_ERROR = (
    "Cannot compute the truth value of a CasADi SXElem symbolic expression"
)


def _pid_control_generation_config(model_type, temp_generated_models_dir):
    config = {
        "file_prefix": "pid_control",
        "input_param_file": "pid_control_parameters.csv",
        "model_type": model_type,
        "generated_models_dir": temp_generated_models_dir,
    }
    if model_type == "casadi_python":
        config.update({
            "solver": "casadi_integrator",
            "solver_info": {
                "method": "cvodes",
                "max_step_size": 0.001,
                "max_num_steps": 5000,
            },
        })
    else:
        config.update({
            "solver": "solve_ivp",
            "solver_info": {
                "method": "RK45",
                "max_step": 0.001,
                "rtol": 1e-8,
                "atol": 1e-10,
            },
        })
    return config


def _load_generated_model(model_path):
    spec = importlib.util.spec_from_file_location("pid_control_model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_casadi_math(module):
    import casadi as ca

    for name, func in {
        "log": ca.log,
        "exp": ca.exp,
        "sin": ca.sin,
        "cos": ca.cos,
        "tan": ca.tan,
        "sqrt": ca.sqrt,
        "floor": ca.floor,
        "pow": ca.power,
    }.items():
        setattr(module, name, func)


def _symbolic_compute_variables(model_path):
    """Evaluate compute_variables with CasADi SX states/variables (AD-relevant path)."""
    import casadi as ca

    module = _load_generated_model(model_path)
    _patch_casadi_math(module)

    states = module.create_states_array()
    rates = module.create_states_array()
    variables = module.create_variables_array()
    module.initialise_variables(states, rates, variables)
    module.compute_computed_constants(variables)

    for idx, info in enumerate(module.STATE_INFO):
        states[idx] = ca.SX.sym(info["name"])
    for idx, info in enumerate(module.VARIABLE_INFO):
        variables[idx] = ca.SX.sym(info["name"])

    module.compute_rates(0.0, states, rates, variables)
    module.compute_variables(0.0, states, rates, variables)
    return variables


@pytest.mark.integration
@pytest.mark.slow
def test_pid_control_casadi_fails_on_python_conditionals(base_user_inputs, resources_dir, temp_generated_models_dir):
    """
    pid_control with model_type=python emits Python if/else for the f_stim clamp.

    Symbolic evaluation of compute_variables must fail with CasADi's truth-value error,
    confirming the failure is due to the case statement rather than 3compartment complexity.
    """
    config = base_user_inputs.copy()
    config.update(_pid_control_generation_config("python", temp_generated_models_dir))

    param_file_path = os.path.join(resources_dir, config["input_param_file"])
    assert os.path.exists(param_file_path), f"Parameter file not found: {param_file_path}"

    success = generate_with_new_architecture(False, config)
    assert success, "pid_control Python model generation should succeed"

    model_path = os.path.join(temp_generated_models_dir, "pid_control", "pid_control.py")
    assert os.path.exists(model_path), f"Generated model not found: {model_path}"

    with open(model_path, encoding="utf-8") as handle:
        source = handle.read()
    assert " if " in source, "Expected Python ternary in non-casadi generated code"
    assert "ca.if_else" not in source, "Python model should not use ca.if_else"

    with pytest.raises(RuntimeError, match=CASADI_TRUTH_VALUE_ERROR):
        _symbolic_compute_variables(model_path)


@pytest.mark.integration
@pytest.mark.slow
def test_pid_control_casadi_succeeds_with_if_else_codegen(base_user_inputs, resources_dir, temp_generated_models_dir):
    """
    pid_control with model_type=casadi_python emits ca.if_else for the same clamp.

    Symbolic evaluation of compute_variables must succeed.
    """
    pytest.importorskip("casadi")

    config = base_user_inputs.copy()
    config.update(_pid_control_generation_config("casadi_python", temp_generated_models_dir))

    param_file_path = os.path.join(resources_dir, config["input_param_file"])
    assert os.path.exists(param_file_path), f"Parameter file not found: {param_file_path}"

    success = generate_with_new_architecture(False, config)
    assert success, "pid_control CasADi Python model generation should succeed"

    model_path = os.path.join(temp_generated_models_dir, "pid_control", "pid_control.py")
    utilities_path = os.path.join(temp_generated_models_dir, "pid_control", "pid_control_utilities.py")
    assert os.path.exists(model_path), f"Generated model not found: {model_path}"
    assert os.path.exists(utilities_path), f"Generated utilities not found: {utilities_path}"

    with open(model_path, encoding="utf-8") as handle:
        source = handle.read()
    with open(utilities_path, encoding="utf-8") as handle:
        utilities_source = handle.read()

    assert "ca.if_else" in source, "CasADi model should transform ternaries to ca.if_else"
    assert "ca.if_else" in utilities_source, "CasADi utilities should use ca.if_else helpers"

    variables = _symbolic_compute_variables(model_path)
    assert variables is not None


@pytest.mark.integration
def test_casadi_solver_uses_full_variables_model_array(
    base_user_inputs, resources_dir, temp_generated_models_dir
):
    """
    CasADi helper keeps a full-length variables_model for model callbacks while
    self.variables remains the constants-only parameter vector for the integrator.
    """
    from libcuflynx.solver_wrappers.casadi_python_solver_helper import SimulationHelper

    config = base_user_inputs.copy()
    config.update(_pid_control_generation_config("casadi_python", temp_generated_models_dir))
    param_file_path = os.path.join(resources_dir, config["input_param_file"])
    assert os.path.exists(param_file_path), f"Parameter file not found: {param_file_path}"

    success = generate_with_new_architecture(False, config)
    assert success, "pid_control CasADi Python model generation should succeed"

    model_path = os.path.join(temp_generated_models_dir, "pid_control", "pid_control.py")
    helper = SimulationHelper(
        model_path,
        dt=0.01,
        sim_time=0.1,
        solver_info={
            "method": "cvodes",
            "max_step_size": 0.001,
            "max_num_steps": 5000,
        },
    )

    assert len(helper.variables) == len(helper.constant_indices)
    assert len(helper.variables_model) == helper.model.VARIABLE_COUNT
    assert len(helper.variables) < len(helper.variables_model)

    helper.reset_states()
    param_name = "parameters/K_p_pid_control"
    helper.set_param_vals([[param_name]], [[2.0]])
    assert helper.variables_model[helper.var_name_to_idx[param_name]] == 2.0
