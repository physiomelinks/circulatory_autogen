import os  
import sys  
import pytest  
import tempfile  
import shutil  
import numpy as np  
import xml.etree.ElementTree as ET  
import json  
import pandas as pd  
  
# Add src to path  
_TEST_ROOT = os.path.join(os.path.dirname(__file__), '..')  
_SRC_DIR = os.path.join(_TEST_ROOT, 'src')  
if _SRC_DIR not in sys.path:  
    sys.path.insert(0, _SRC_DIR)  
  
from scripts.script_generate_with_new_architecture import generate_with_new_architecture  
from solver_wrappers import get_simulation_helper  
  
  
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
                        "vessel_type": "test_type1",  
                        "BC_type": "test",  
                        "module_format": "cellml",  
                        "module_file": "test_modules.cellml",  
                        "module_type": "test_type1_type",  
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
                        "vessel_type": "test_type2",  
                        "BC_type": "test",  
                        "module_format": "cellml",  
                        "module_file": "test_modules.cellml",  
                        "module_type": "test_type2_type",  
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
        
    # Create resources directory for vessel array and parameters  
    resources_dir = os.path.join(temp_model_dir, 'resources')  
    os.makedirs(resources_dir, exist_ok=True)  
      
    # Create vessel_array.csv  
    vessel_array_file = os.path.join(resources_dir, 'unit_test_vessel_array.csv')  
    with open(vessel_array_file, 'w') as f:  
        f.write("""name,BC_type,vessel_type,inp_vessels,out_vessels  
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
        'model_type': 'cellml_only',  
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
    
     # Check if unit converter component is in the generated CellML file  
    with open(cellml_file, 'r') as f:  
        cellml_content = f.read()  
      
    # Parse XML to find unit converter component  
    root = ET.fromstring(cellml_content)

    # Try both possible namespaces  
    cellml_ns_1_1 = "http://www.cellml.org/cellml/1.1#"  
    cellml_ns_2_0 = "http://www.cellml.org/cellml/2.0#"  
    
    # Search for components with namespace  
    components = root.findall(f'.//{{{cellml_ns_1_1}}}component')  
    if not components:  
        components = root.findall(f'.//{{{cellml_ns_2_0}}}component')  

    unit_converter_found = False  
    for comp in components:  
        comp_name = comp.get('name')  
        if comp_name and 'unit_converter' in comp_name:  
            unit_converter_found = True  
            print(f"✓ Found unit converter component: {comp_name}")  
            
            # Verify it has the correct variables  
            vars_found = []  
            for var in comp.findall(f'.//{{{cellml_ns_1_1}}}variable'):  
                vars_found.append(var.get('name'))  
            
            assert 'flux' in vars_found, "Unit converter missing input variable"  
            assert 'flux_in' in vars_found, "Unit converter missing output variable"  
            
            break  
    
    assert unit_converter_found, "Unit converter component not found in generated CellML"

    # Run simulation to verify results  
    from solver_wrappers import get_simulation_helper  
      
    sim_helper = get_simulation_helper(  
        model_path=cellml_file,  
        model_type='cellml_only',  
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