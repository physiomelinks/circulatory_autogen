"""
Converts a CellML model to Circulatory Autogen format and generate files to run Circulatory Autogen.
Inputs: CellML model and the output directory where resources ('parameters.csv' and 'vessel_array.csv') and 'user_inputs.yaml' files are created.
'module_config.json' and 'modules.cellml' files are generated at 'module_config_user' directory.
"""
import argparse
import json
import os.path
import sys
import xml.etree.ElementTree as ET
import re
import yaml

import libcellml as lc
    
root_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir, 'src'))
user_inputs_dir = os.path.join(root_dir, 'user_run_files')
src_dir = os.path.join(os.path.dirname(__file__), '..')
import utilities.libcellml_helper_funcs as cellml
import utilities.libcellml_utilities as libcellml_utils

# TODO:
# - Add support for multiple components in a cellml model
# - Add support for automatically generating the ports in the module_config.json file. Currently the ports are not generated automatically.

# Define file paths
user_units_cellml = os.path.join(root_dir, 'module_config_user/user_units.cellml')
units_cellml = os.path.join(src_dir, 'generators/resources/units.cellml')
user_inputs_yaml = os.path.join(user_inputs_dir, 'user_inputs.yaml')

# Define file_prefix, vessel_name and data_reference for the model
# Specify the time variable and component for which you want to generate files

def _parse_args():
    parser = argparse.ArgumentParser(prog="convert_cellml_for_ca")
    parser.add_argument("-i", "--input-model", help="import CellML model to convert.",
                        default="/home/farg967/Documents/git_projects/cellml_models/Gonzalo_H_SMC/smc_hernandez_one_module.cellml")
    parser.add_argument("-o", "--output-dir", help="output directory for converted model data.",
                        default="/home/farg967/Documents/git_projects/CA_user/smc_hernandez")
    parser.add_argument("--file-prefix", help="file prefix for generated files.",
                        default="smc_hernandez")
    parser.add_argument("--vessel-name", help="vessel name.",
                        default="smc_hernandez")
    parser.add_argument("--data-reference", help="data reference string.",
                        default="hernandezhernandez2024")
    parser.add_argument("--time-variable", help="name of the time variable.",
                        default="t")
    parser.add_argument("--component-name", help="component name to use.",
                        default="smc_hernandez")
    return parser.parse_args()

# Set variables from args or defaults
args = _parse_args()
file_prefix = args.file_prefix
vessel_name = args.vessel_name
data_reference = args.data_reference
time_variable = args.time_variable
component_name = args.component_name
input_model = args.input_model
output_dir = args.output_dir

# Print errors
def _print_errors(l):
    print("Total errors: ", l.errorCount())
    for i in range(l.errorCount()):
        print(f"{i + 1}: {l.error(i).description()}")

# Extract variables, constants and state variables from the model
def _extract_variables_constants_states_from_model(analysed_model):
    variables = []
    constants = {}
    states = []

    # Get variables and constants
    for i in range(analysed_model.variableCount()):
        analysed_model_variable = analysed_model.variable(i)
        variable = analysed_model_variable.variable()
        type_as_string = analysed_model_variable.typeAsString(analysed_model_variable.type())

        if type_as_string == "algebraic":
            variables.append(variable.name())
        elif type_as_string == "constant":
            constants[variable.name()] = variable.initialValue()

    # Get state variables and set initial values
    for i in range(analysed_model.stateCount()):
        analysed_model_state = analysed_model.state(i)
        state = analysed_model_state.variable()
        states.append([state.name(), state.units().name()])

    return variables, constants, states

# Extract variables, constants and state variables of the component
def _extract_variables_constants_states_of_component(component, model_variables, model_constants, model_states):
    variables = []
    constants = []
    states = []

    for i in range(component.variableCount()):
        if component.variable(i).name() in model_variables:
            variables.append([component.variable(i).name(), component.variable(i).units().name(), "variable"])
        elif component.variable(i).name() in model_constants:
            if not component.variable(i).initialValue():
                initial_value = model_constants[component.variable(i).name()]
                constants.append([component.variable(i).name(), component.variable(i).units().name(), "constant", initial_value])
            else:
                constants.append([component.variable(i).name(), component.variable(i).units().name(), "constant", component.variable(i).initialValue()])
        elif component.variable(i).name() in {row[0] for row in model_states}:
            constants.append([f"{component.variable(i).name()}_init", component.variable(i).units().name(), "constant", component.variable(i).initialValue()])
            states.append([component.variable(i).name(), component.variable(i).units().name(), "variable"])

    return variables, constants, states

# Generate module_config.json file
def _generate_module_config(variables, constants, states, file_prefix, component_name):
    config = {
        "vessel_type": file_prefix,
        "BC_type": "nn",
        "module_format": "cellml",
        "module_file": f"{file_prefix}_modules.cellml",
        "module_type": f"{file_prefix}_{component_name}_type",
        "entrance_ports": [],
        "exit_ports": [],
        "general_ports": [],
        "variables_and_units": []
    }

    # Add variables
    for variable in variables:
        info = [variable[0], variable[1], "access", variable[2]]
        config["variables_and_units"].append(info)

    # Add constants
    for constant in constants:
        info = [constant[0], constant[1], "access", constant[2]]
        config["variables_and_units"].append(info)

    # Add state variables
    for state in states:
        info = [state[0], state[1], "access", state[2]]
        config["variables_and_units"].append(info)

    module_config = [config]

    file_path = os.path.join(root_dir, 'module_config_user', f"{file_prefix}_module_config.json")

    with open(file_path, "w") as fh:
        json.dump(module_config, fh, indent=2)

    print(f"Generated module_config.json file: {file_path}")

# Generate vessel_array.csv file
def _generate_vessel_array_csv(output_dir, vessel_name, file_prefix):

    resources_dir = os.path.join(output_dir, "resources")
    os.makedirs(resources_dir, exist_ok=True)

    file_path = os.path.join(resources_dir, f"{file_prefix}_vessel_array.csv")

    with open(file_path, "w") as fh:
        fh.writelines(["name,BC_type,vessel_type,inp_vessels,out_vessels\n"])
        fh.writelines([f"{vessel_name},nn,{file_prefix},,\n"])

    print(f"Generated vessel_array.csv file: {file_prefix}_vessel_array.csv")

# Generate parameters.csv file
def _generate_parameters_csv(output_dir, constants, vessel_name, file_prefix, data_reference):

    resources_dir = os.path.join(output_dir, "resources")
    os.makedirs(resources_dir, exist_ok=True)

    file_path = os.path.join(resources_dir, f"{file_prefix}_parameters.csv")

    with open(file_path, "w") as fh:
        fh.writelines(["variable_name,units,value,data_reference\n"])

        for constant in constants:
            fh.writelines([f"{constant[0]}_{vessel_name},{constant[1]},{constant[3]},{data_reference}\n"])

    print(f"Generated parameters.csv file: {file_prefix}_parameters.csv")

# Generate CellML module
def _generate_cellml_module(input_model, states, file_prefix):

    # Parse the cellml file
    tree = ET.parse(input_model)
    root = tree.getroot()

    # Add xmlns:cellml to the root model element
    root.set("xmlns:cellml", cellml_namespace)

    # Register namespaces
    ET.register_namespace("", cellml_namespace) 
    ET.register_namespace("math", mathml_namespace)

    # Move units to user_units.cellml
    _update_units_file(root)

    # Modify variable elements
    _modify_variables(root)

    # Modify state variable elements
    _modify_state_variables(root, states)

    # Modify component name
    _modify_component_name(root)

    # Ensure math elements are using the proper namespace
    for math in root.findall(f".//{{{mathml_namespace}}}math"):
        math.attrib["xmlns"] = mathml_namespace

    # Modify the 'cn' elements inside math to have 'cellml:' for the 'units' attribute
    for cn in root.findall(f".//{{{mathml_namespace}}}cn"):
        if f"{{{cellml_namespace}}}units" in cn.attrib:
            cn.set("cellml:units", cn.attrib[f"{{{cellml_namespace}}}units"])
            del cn.attrib[f"{{{cellml_namespace}}}units"]

    # Remove math: prefix for all MathML elements
    cellml_str = ET.tostring(root, encoding="UTF-8", xml_declaration=True).decode("utf-8")
    cellml_str = cellml_str.replace('math:', '')

    # Write the modified cellml
    file_path = os.path.join(root_dir, 'module_config_user', f"{file_prefix}_modules.cellml")
    with open(file_path, "w", encoding="UTF-8") as f:
        f.write(cellml_str)

    print(f"Generated cellml module: {file_prefix}_modules.cellml")

# Move units to user_units.cellml
def _update_units_file(root):

    # Extract units from user_units.cellml
    user_units_tree = ET.parse(user_units_cellml)
    user_units_root = user_units_tree.getroot()
    user_units = {unit.get("name"): unit for unit in user_units_root.findall(f".//{{{cellml_namespace}}}units")}

    # Extract units from units.cellml
    units_tree = ET.parse(units_cellml)
    units_root = units_tree.getroot()
    units = {unit.get("name"): unit for unit in units_root.findall(f".//{{{cellml_namespace}}}units")}

    # Extract units from original model
    units_dict = {unit.get("name"): unit for unit in root.findall(f".//{{{cellml_namespace}}}units")}

    # Add units to user_units.cellml
    for name, unit in units_dict.items():
        if name not in user_units and name not in units:
            user_units_root.append(unit)

    # Remove units from the module
    for unit in root.findall(f".//{{{cellml_namespace}}}units"):
        root.remove(unit)

    user_units_tree.write(user_units_cellml, xml_declaration=True, encoding="UTF-8")

    print(f"Updated user_units.cellml")

# Modify variable elements in cellml module component
def _modify_variables(root):
    component = root.find(f".//{{{cellml_namespace}}}component[@name='{component_name}']")

    variables = component.findall(f".//{{{cellml_namespace}}}variable")
    for variable in variables:
        if variable.attrib["name"]=="t":
            variable.set("public_interface", "in")
            if "initial_value" in variable.attrib:
                del variable.attrib["initial_value"]
        elif "initial_value" in variable.attrib:
            del variable.attrib["initial_value"]
            variable.set("public_interface", "in")
        elif "public_interface" in variable.attrib and variable.attrib["public_interface"]=="in":
            pass
        else:
            variable.set("public_interface", "out")

# Modify state variables in cellml module
def _modify_state_variables(root, states):
    for state in states:
        var_name = state[0]
        unit = state[1]
        
        variable = root.find(f".//{{{cellml_namespace}}}variable[@name='{var_name}']")
        
        if variable is not None:
            new_name = f"{var_name}_init"
            variable.set("name", new_name)
            
            new_variable = ET.Element(f"{{{cellml_namespace}}}variable", name=var_name, units=unit, public_interface="out", initial_value=new_name)
            
            root.find(f".//{{{cellml_namespace}}}component").append(new_variable)

# Modify component name
def _modify_component_name(root):
    for component in root.findall(f".//{{{cellml_namespace}}}component[@name='{component_name}']"):
        new_name = f"{file_prefix}_{component.attrib["name"]}_type"
        component.set("name", new_name)

# Generate user_inputs.yaml
def _generate_user_inputs_yaml(output_dir, file_prefix):
    file_path = os.path.join(output_dir, f"{file_prefix}_user_inputs.yaml")

    with open(user_inputs_yaml, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    config["resources_dir"] = f"{output_dir}/resources"
    config["generated_models_dir"] = f"{output_dir}/generated_models"

    config["file_prefix"] = file_prefix
    config["input_param_file"] = f"{file_prefix}_parameters.csv"

    with open(file_path, "w") as file:
        yaml.dump(config, file, sort_keys=False)

    print(f"Generated user_inputs.yaml: {file_prefix}_user_inputs.yaml")

def main():
    # args = _parse_args()
    args = {"input_model": input_model, "output_dir": output_dir}
    if not os.path.isfile(args["input_model"]):
        print(f"Input file '{args["input_model"]}' not found.")
        sys.exit(1)

    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])

    if not os.path.isdir(args["output_dir"]):
        print(f"'{args["output_dir"]}' is not a valid directory")
        sys.exit(2)

    with open(args["input_model"]) as fh:
        content = fh.read()

    # Parse model
    parser = lc.Parser(False)
    model = parser.parseModel(content)

    # Get component
    component = model.component(component_name)

    if component == None:
        print(f"Component '{component_name}' not found.")
        sys.exit(1)

    # Remove initial value of variable time
    variable_t = component.variable(time_variable)

    if variable_t == None:
        print("WARNING: Configure the name of time variable in the component correctly here. If there is no time variable for the provided component, you can ignore this message.")
    else:
        variable_t.removeInitialValue()

    # Validate model
    validator = lc.Validator()
    validator.validateModel(model)

    if validator.errorCount() > 0:
        _print_errors(validator)
        print('continuing with errors')
        # sys.exit(3)

    # Analyse model
    analyser = lc.Analyser()

    importer = cellml.resolve_imports(model, os.path.dirname(args["input_model"]), False)

    flat_model = cellml.flatten_model(model, importer)
    analyser.analyseModel(flat_model)
    analysed_model = analyser.model()

    libcellml_utils.print_issues(analyser)
    print(analysed_model.type())
    if analysed_model.type() != lc.AnalyserModel.Type.ODE:
        print("WARNING model is has some issues, see above. "
            "The model might still run with some of the above issues"
            "but it is recommended to fix them")

    if analyser.errorCount() > 0:
        _print_errors(analyser)
        print('continuing with errors')
        # sys.exit(4)


    default_cellml_namespace = "http://www.cellml.org/cellml/1.1#"

    # Read the CellML file
    with open(args["input_model"], "r", encoding="utf-8") as file:
        cellml_data = file.read()

    # Extract 'xmlns:cellml' using regex
    match = re.search(r'xmlns:cellml="([^"]+)"', cellml_data)
    global cellml_namespace
    cellml_namespace = match.group(1) if match else default_cellml_namespace

    global mathml_namespace
    mathml_namespace = "http://www.w3.org/1998/Math/MathML"

    model_variables, model_constants, model_states = _extract_variables_constants_states_from_model(analysed_model)

    variables, constants, states = _extract_variables_constants_states_of_component(component, model_variables, model_constants, model_states)

    _generate_module_config(variables, constants, states, file_prefix, component_name)

    _generate_vessel_array_csv(args["output_dir"], vessel_name, file_prefix)

    _generate_parameters_csv(args["output_dir"], constants, vessel_name, file_prefix, data_reference)

    _generate_cellml_module(args["input_model"], states, file_prefix)

    _generate_user_inputs_yaml(args["output_dir"], file_prefix)

if __name__ == "__main__":
    main()
