"""
Taking in a CellML model convert it to a module for Circulatory Autogen.
"""
import argparse
import json
import os.path
import sys

import libcellml as lc


def _parse_args():
    parser = argparse.ArgumentParser(prog="convert_cellml_for_ca")
    parser.add_argument("-i", "--input-model", help="import CellML model to convert.", required=True)
    parser.add_argument("-o", "--output-dir", help="output directory for converted model data.", required=True)
    return parser.parse_args()


def _print_errors(l):
    print("Total errors: ", l.errorCount())
    for i in range(l.errorCount()):
        print(f"{i + 1}: {l.error(i).description()}")


def _module_filename(module_type):
    return f"{module_type}_modules.cellml"


def _parameters_filename(module_type):
    return f"{module_type}_parameters.csv"


def _create_module_config(variables, vessel_type, bc_type, module_type):
    config = {
        "vessel_type": vessel_type,
        "BC_type": bc_type,
        "module_format": "cellml",
        "module_file": _module_filename(module_type),
        "module_type": module_type,
        "entrance_ports":
            [
            ],
        "exit_ports":
            [
            ],
        "variables_and_units":
            [
            ]
    }
    for variable in variables:
        info = [variable[0], variable[1], "access", variable[2]]
        config["variables_and_units"].append(info)

    return config


def _write_module_config(output_dir, variables, module_type, vessel_type="NKE_pump", bc_type="nn"):
    config = _create_module_config(variables, vessel_type, bc_type, module_type)
    module_config = [config]

    with open(os.path.join(output_dir, _module_filename(module_type)), "w") as fh:
        json.dump(module_config, fh, indent=2)


def _write_parameters_csv(output_dir, variables, module_type):

    with open(os.path.join(output_dir, _parameters_filename(module_type)), "w") as fh:
        fh.writelines(["variable_name,units,value,data_reference\n"])
        for variable in variables:
            fh.writelines([f"{variable[0]}_{module_type},{variable[1]},{variable[3]},user_defined\n"])


def main():
    args = _parse_args()
    if not os.path.isfile(args.input_model):
        sys.exit(1)

    if not os.path.isdir(args.output_dir):
        sys.exit(2)

    with open(args.input_model) as fh:
        content = fh.read()

    p = lc.Parser(False)
    m = p.parseModel(content)

    model_source = os.path.basename(args.input_model)
    model_name = m.name()
    if model_source == "NKE_pump_orig.cellml" and model_name == "my_model":
        c = m.component("main")
        v = c.variable("t")
        v.removeInitialValue()

    v = lc.Validator()
    v.validateModel(m)

    if v.errorCount() > 0:
        _print_errors(v)
        sys.exit(3)

    a = lc.Analyser()
    a.analyseModel(m)

    if a.errorCount() > 0:
        _print_errors(a)
        sys.exit(4)

    am = a.model()

    # Prepare for module config.
    variables = []
    for i in range(am.variableCount()):
        am_v = am.variable(i)
        v = am_v.variable()
        type_as_string = am_v.typeAsString(am_v.type())
        if type_as_string == "algebraic":
            variables.append([v.name(), v.units().name(), "variable", v.initialValue()])
        elif type_as_string == "constant":
            variables.append([v.name(), v.units().name(), "constant", v.initialValue()])

    module_type = "NKE_pump"
    _write_module_config(args.output_dir, variables, module_type)

    _write_parameters_csv(args.output_dir, variables, module_type)

    # Manipulate CellML 1.0 model to define constants and variables as inputs.


if __name__ == "__main__":
    main()
