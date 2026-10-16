import os  
import sys  
import pytest  
import tempfile  
import shutil  
import numpy as np  
import xml.etree.ElementTree as ET  
import json  
import re
import pandas as pd  
  
  
from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture  
from libcuflynx.solver_wrappers import get_simulation_helper  
  
  
@pytest.fixture  
def temp_model_dir():  
    """Create temporary directory for test models."""  
    temp_dir = tempfile.mkdtemp()  
    yield temp_dir  
    shutil.rmtree(temp_dir, ignore_errors=True)  
  
  
@pytest.fixture
def create_unit_conversion_test_files(temp_model_dir):  
    """
        Create temporary files for unit conversion test with summation.
        - test_modules.cellml defining two components with different units (m3_per_mol and L_per_mol)
    """  
    # Use module_config_user directory instead of resources  
    module_config_dir = os.path.join(temp_model_dir, 'module_config_user')
    os.makedirs(module_config_dir, exist_ok=True)  
      
    # Create test_modules.cellml in module_config_user  
    modules_file = os.path.join(module_config_dir, 'test_modules.cellml')  
    with open(modules_file, 'w') as f:  
        f.write("""<?xml version='1.0' encoding='UTF-8'?>
<model name="modules" xmlns="http://www.cellml.org/cellml/1.1#" xmlns:cellml="http://www.cellml.org/cellml/1.1#">
    <component name="test_type1_type">
        <variable name="t" public_interface="in" units="second"/>
        <variable name="flux" public_interface="out" units="m3_per_mol"/>
        <variable initial_value="0" name="tmp" public_interface="out" units="dimensionless"/>
        <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
                <eq/>
                <apply>
                    <diff/>
                    <bvar>
                        <ci>t</ci>
                    </bvar>
                    <ci>tmp</ci>
                </apply>
                <cn cellml:units="dimensionless">0.0</cn>
            </apply>
            <apply>
                <eq/>
                <ci>flux</ci>
                <cn cellml:units="m3_per_mol">0.001</cn>
            </apply>
        </math>
    </component>
    <component name="test_type2_type">
        <variable name="t" public_interface="in" units="second"/>
        <variable name="flux_in" public_interface="in" units="L_per_mol"/>
        <variable name="flux_local" public_interface="in" units="L_per_mol"/>
        <variable name="flux_total" public_interface="out" units="L_per_mol"/>
        <variable initial_value="0" name="tmp" public_interface="out" units="dimensionless"/>
        <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
                <eq/>
                <apply>
                    <diff/>
                    <bvar>
                        <ci>t</ci>
                    </bvar>
                    <ci>tmp</ci>
                </apply>
                <cn cellml:units="dimensionless">0.0</cn>
            </apply>
            <apply>
                <eq/>
                <ci>flux_total</ci>
                <apply>
                    <plus/>
                    <ci>flux_in</ci>
                    <ci>flux_local</ci>
                </apply>
            </apply>
        </math>
    </component>
</model>
""")  
      
    # Create test_modules_config.json in module_config_user  
    modules_config = os.path.join(module_config_dir, 'test_modules_config.json')  
    with open(modules_config, 'w') as f:  
        json.dump([  
                    {  
                        "module_type": "test_type1",  
                        "module_subtype": "test",  
                        "module_format": "cellml",  
                        "component_file": "test_modules.cellml",  
                        "component_type": "test_type1_type",  
                        "entrance_ports": [],  
                        "exit_ports": [  
                        {  
                            "port_type": "vessel_port",  
                            "variables": ["flux"]  
                        }  
                        ],  
                        "general_ports": [],  
                        "variables_and_units": [  
                        ["flux", "m3_per_mol", "access", "variable"],  
                        ["tmp", "dimensionless", "access", "variable"]  
                        ]  
                    },  
                    {  
                        "module_type": "test_type2",  
                        "module_subtype": "test",  
                        "module_format": "cellml",  
                        "component_file": "test_modules.cellml",  
                        "component_type": "test_type2_type",  
                        "entrance_ports": [  
                        {  
                            "port_type": "vessel_port",  
                            "variables": ["flux_in"]  
                        }  
                        ],  
                        "exit_ports": [],  
                        "general_ports": [],  
                        "variables_and_units": [  
                        ["flux_in", "L_per_mol", "access", "variable"],  
                        ["flux_local", "L_per_mol", "access", "constant"],  
                        ["flux_total", "L_per_mol", "access", "variable"],  
                        ["tmp", "dimensionless", "access", "variable"]  
                        ]  
                    }  
                    ], f, indent=2)  
        
    # Create resources directory for module array and parameters  
    resources_dir = os.path.join(temp_model_dir, 'resources')  
    os.makedirs(resources_dir, exist_ok=True)  
      
    # Create module_array.csv  
    module_array_file = os.path.join(resources_dir, 'unit_test_module_array.csv')  
    with open(module_array_file, 'w') as f:  
        f.write("""name,module_subtype,module_type,inp_instances,out_instances  
vessel1,test,test_type1,,vessel2  
vessel2,test,test_type2,vessel1,  
""")  
      
    # Create parameters.csv  
    parameters_file = os.path.join(resources_dir, 'unit_test_parameters.csv')  
    with open(parameters_file, 'w') as f:    
        f.write("""variable_name,units,value,data_reference  
                flux_local_vessel2,L_per_mol,0.5,test  
    """)
      
    return resources_dir, module_config_dir
  
@pytest.mark.integration  
@pytest.mark.slow  
def test_summation_with_unit_conversion(temp_model_dir, create_unit_conversion_test_files):  
    """  
        Test that unit conversion is correctly applied and results are summed in the generated model.
            - type1 component produces flux in m3_per_mol (0.001 m3_per_mol = 1 L_per_mol)
            - type2 component takes flux_in in L_per_mol, has a local flux_local in L_per_mol, 
              and outputs flux_total as the sum of flux_in and flux_local in L_per_mol.  
    """  
    resources_dir, module_config_dir = create_unit_conversion_test_files  
      
    config = {  
        'file_prefix': 'unit_test',  
        'input_param_file': 'unit_test_parameters.csv',  
        'model_type': 'cellml',  
        'solver': 'CVODE_myokit',  
        'resources_dir': resources_dir,  
        'generated_models_dir': temp_model_dir,  
        'external_modules_dir': module_config_dir,  # Point to module_config_user  
        'DEBUG': False  
    }    
      
    # Generate the model using the same function as test_autogeneration  
    success = generate_with_new_architecture(False, config)  
    assert success, "Model generation should succeed"  
    
    # Verify generated files exist  
    generated_dir = os.path.join(temp_model_dir, 'unit_test')  
    cellml_file = os.path.join(generated_dir, 'unit_test.cellml')  
    assert os.path.exists(cellml_file), "Main CellML file should be generated"  
    
    # flux (m3_per_mol) is mapped directly to flux_in (L_per_mol); the solver converts it
    with open(cellml_file, 'r') as f:
        cellml_content = f.read()
    assert 'unit_converter' not in cellml_content
    assert re.search(r'<map_components component_1="vessel1_module" component_2="vessel2_module"/>\s*'
                     r'<map_variables variable_1="flux" variable_2="flux_in"/>', cellml_content)

    # Run simulation to verify results  
    from libcuflynx.solver_wrappers import get_simulation_helper  
      
    sim_helper = get_simulation_helper(  
        model_path=cellml_file,  
        model_type='cellml',  
        solver='CVODE_myokit',  
        # solver='CVODE',  
        dt=0.01,  
        sim_time=0.1,  
        pre_time=0.0  
    )  
      
    # Run simulation  
    result = sim_helper.run()  
    assert result, "Simulation should run successfully"  

    available_names = sim_helper.get_all_variable_names()  
    if 'vessel2/flux_total' in available_names:  
        flux_total = sim_helper.get_results(['vessel2/flux_total'])[0][0][0]
    else:  
        # Method 2: Access directly from the model after simulation  
        flux_total_var = None
        all_vars = sim_helper.all_vars
        for var in all_vars:  
            if 'flux_total' in var.qname():  
                flux_total_var = var
                flux_total = flux_total_var.eval()  
                break  
        
    # Verify the result is correct (0.5 L_per_mol from local + 1 L_per_mol from conversion)  
    expected_flux_total = 0.5 + 1.0  # 0.5 L_per_mol from local + 1 L_per_mol from conversion (0.001 m3_per_mol = 1 L_per_mol)  
    assert np.isclose(flux_total, expected_flux_total, atol=1e-6), f"Expected flux_total to be {expected_flux_total} L_per_mol, got {flux_total} L_per_mol"

@pytest.mark.unit
def test_scale_factor_rejects_affine_units():
    from libcuflynx.utilities.utility_funcs import AffineUnitsError, UnitConverter
    converter = UnitConverter()
    assert np.isclose(converter.get_scale_factor('m3', 'litre'), 1.0e3)
    for from_units, to_units in [('kelvin', 'celsius'), ('celsius', 'kelvin')]:
        with pytest.raises(AffineUnitsError, match='needs an offset'):
            converter.get_scale_factor(from_units, to_units)
