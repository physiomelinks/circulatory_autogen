"""
Tests for model autogeneration functionality.

These tests verify that models can be generated correctly from parameter files.
"""
import os
import pytest

from scripts.script_generate_with_new_architecture import generate_with_new_architecture


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("file_prefix,input_param_file,model_type,solver", [
    ('ports_test', 'ports_test_parameters.csv', 'cellml_only', 'CVODE'),
    ('test_init_states', 'test_init_states_parameters.csv', 'cellml_only', 'CVODE'),
    ('3compartment', '3compartment_parameters.csv', 'cellml_only', 'CVODE'),
    ('simple_physiological', 'simple_physiological_parameters.csv', 'cellml_only', 'CVODE'),
    ('parasympathetic_model', 'parasympathetic_model_parameters.csv', 'cellml_only', 'CVODE'),
    ('test_fft', 'test_fft_parameters.csv', 'cellml_only', 'CVODE'),
    ('neonatal', 'neonatal_parameters.csv', 'cellml_only', 'CVODE'),
    ('generic_junction_test_closed_loop', 'generic_junction_test_closed_loop_parameters.csv', 'cellml_only', 'CVODE'),
    ('generic_junction_test2_closed_loop', 'generic_junction_test_closed_loop_parameters.csv', 'cellml_only', 'CVODE'),
    ('generic_junction_test_open_loop', 'generic_junction_test_open_loop_parameters.csv', 'cellml_only', 'CVODE'),
    ('generic_junction_test2_open_loop', 'generic_junction_test_open_loop_parameters.csv', 'cellml_only', 'CVODE'),
    ('SN_simple', 'SN_simple_parameters.csv', 'cellml_only', 'CVODE'),
    ('physiological', 'physiological_parameters.csv', 'cellml_only', 'CVODE'),
    ('control_phys', 'control_phys_parameters.csv', 'cellml_only', 'CVODE'),
])
def test_generate_cellml_model_succeeds(file_prefix, input_param_file, model_type, solver, base_user_inputs, resources_dir):
    """
    Test that CellML model generation succeeds for various model configurations.
    
    Args:
        file_prefix: Model file prefix
        input_param_file: Parameter file name
        model_type: Type of model to generate
        solver: Solver to use
        base_user_inputs: Base user inputs configuration fixture
        resources_dir: Resources directory fixture
    """
    # Setup configuration
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': file_prefix,
        'input_param_file': input_param_file,
        'model_type': model_type,
        'solver': solver,
    })
    
    # Verify parameter file exists
    param_file_path = os.path.join(resources_dir, input_param_file)
    assert os.path.exists(param_file_path), f"Parameter file not found: {param_file_path}"
    
    # Generate model
    success = generate_with_new_architecture(False, config)
    
    # Assert generation succeeded
    assert success, f"Model generation failed for {file_prefix} with {input_param_file}"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize(
    "file_prefix,input_param_file,model_type,solver",
    [
        ('3compartment', '3compartment_parameters.csv', 'python', 'solve_ivp'),
        ('SN_simple', 'SN_simple_parameters.csv', 'python', 'solve_ivp'),
        ('pid_control', 'pid_control_parameters.csv', 'python', 'solve_ivp'),
    ],
)
def test_generate_python_model_succeeds(file_prefix, input_param_file, model_type, solver, base_user_inputs, resources_dir):
    """
    Test that Python model generation succeeds for selected configurations.
    """
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': file_prefix,
        'input_param_file': input_param_file,
        'model_type': model_type,
        'solver': solver,
    })

    param_file_path = os.path.join(resources_dir, input_param_file)
    assert os.path.exists(param_file_path), f"Parameter file not found: {param_file_path}"

    success = generate_with_new_architecture(False, config)
    assert success, f"Python model generation failed for {file_prefix} with {input_param_file}"


@pytest.mark.integration
@pytest.mark.slow
def test_generate_cpp_model_succeeds(base_user_inputs, resources_dir, temp_output_dir):
    """
    Test that CPP model generation succeeds for aortic_bif_1d model.
    
    Args:
        base_user_inputs: Base user inputs configuration fixture
        resources_dir: Resources directory fixture
        temp_output_dir: Temporary output directory fixture
    """
    # Setup configuration for CPP model
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': 'aortic_bif_1d',
        'input_param_file': 'aortic_bif_1d_parameters.csv',
        'model_type': 'cpp',
        'solver': 'RK4',
        'couple_to_1d': True,
        'cpp_generated_models_dir': temp_output_dir,
        'cpp_1d_model_config_path': None,
    })
    
    # Verify parameter file exists
    param_file_path = os.path.join(resources_dir, 'aortic_bif_1d_parameters.csv')
    assert os.path.exists(param_file_path), f"Parameter file not found: {param_file_path}"
    
    # Generate model
    success = generate_with_new_architecture(False, config)
    
    # Assert generation succeeded
    assert success, "CPP model generation failed for aortic_bif_1d"


@pytest.mark.integration
@pytest.mark.slow
def test_generate_model_with_invalid_parameters_fails(base_user_inputs, resources_dir):
    """
    Test that model generation fails gracefully with invalid parameters.
    
    When given invalid file paths, the function should raise a FileNotFoundError
    rather than silently failing or crashing.
    
    Args:
        base_user_inputs: Base user inputs configuration fixture
        resources_dir: Resources directory fixture
    """
    # Setup configuration with invalid file prefix
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': 'nonexistent_model',
        'input_param_file': 'nonexistent_parameters.csv',
        'model_type': 'cellml_only',
        'solver': 'CVODE',
    })
    
    # Attempt to generate model - should raise FileNotFoundError
    # The function doesn't catch exceptions, so missing files will raise an error
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        generate_with_new_architecture(False, config)

