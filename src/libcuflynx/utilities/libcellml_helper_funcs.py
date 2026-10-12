import os
import re

try:
    from libcellml import Analyser, AnalyserModel, AnalyserExternalVariable, Component, Generator, GeneratorProfile, \
                          Parser, Importer, Model, Printer, Validator
    LIBCELLML_available = True
except ImportError as e:
    print("Error -> ", e)
    print('continuing without LibCellML, Warning code checks will not be available.'
          'You will need to open generated models in OpenCOR to check for errors.')
    LIBCELLML_available = False
import libcuflynx.utilities.libcellml_utilities as libcellml_utils


# Wrappers for the libCellML python API to give some convenient methods.
#


def _dump_issues(source_method_name, logger):
    if logger.issueCount() > 0:
        print('The method "{}" found {} issues:'.format(source_method_name, logger.issueCount()))
        for i in range(0, logger.issueCount()):
            print('    - {}'.format(logger.issue(i).description()))


def parse_model(filename, strict_mode):
    cellml_file = open(filename)
    parser = Parser(strict_mode)
    model = parser.parseModel(cellml_file.read())
    _dump_issues("parse_model", parser)
    libcellml_utils.print_issues(parser)
    return model

def print_model(model):
    printer = Printer()
    s = printer.printModel(model)
    return s

def validate_model(model):
    validator = Validator()
    validator.validateModel(model)
    _dump_issues("validate_model", validator)
    return validator.issueCount()


def resolve_imports(model, base_dir, strict_mode):
    importer = Importer(strict_mode)
    importer.resolveImports(model, base_dir)
    _dump_issues("resolve_imports", importer)
    if model.hasUnresolvedImports():
        print("unresolved imports?")
    else:
        print("no unresolved imports.")
    return importer

def flatten_model(model, importer):
    flat_model = importer.flattenModel(model)
    return flat_model

def get_analysed_model(analyser):
    """The AnalyserModel produced by analyser.analyseModel(...)."""
    return analyser.analyserModel()


def generate_implementation_code(generator, analysed_model, profile):
    return generator.implementationCode(analysed_model, profile)


def generate_interface_code(generator, analysed_model, profile):
    return generator.interfaceCode(analysed_model, profile)


def analyser_variables(analysed_model):
    """Each non-state analyser variable with its index in one ``variables`` array: constants, computed constants,
    algebraic variables, then external variables (the layout of ``c_code_with_one_variables_array``)."""
    variables = []
    for group in (analysed_model.constants(), analysed_model.computedConstants(),
                  analysed_model.algebraicVariables(), analysed_model.externalVariables()):
        variables.extend(group)
    return list(zip(variables, range(len(variables))))


# libCellML 0.7's C: one array per kind of variable, each function taking all of them
_C_KINDS = (("CONSTANT", "constants", "CONSTANT"), ("COMPUTED_CONSTANT", "computedConstants", "COMPUTED_CONSTANT"),
            ("ALGEBRAIC_VARIABLE", "algebraicVariables", "ALGEBRAIC"), ("EXTERNAL_VARIABLE", "externalVariables", "EXTERNAL"))
_C_ALL_ARRAYS = "double *constants, double *computedConstants, double *algebraicVariables"


def c_code_with_one_variables_array(code, header):
    """libCellML's C (implementation, or the header if ``header``) with its constants, computed constants, algebraic
    and external variables in one ``variables`` array, in that order, and each variable's type in ``VARIABLE_INFO``:
    the layout CVSCppGenerator rewrites into its C++ model."""
    if "RootFindingInfo" in code:
        raise ValueError("C++ generation does not support models with non-linear algebraic equations")
    externals = "EXTERNAL_VARIABLE_COUNT" in code
    if header:
        counts = None
    else:
        counts = {kind: int(re.search(rf"const size_t {kind}_COUNT = (\d+);", code).group(1)) if f"{kind}_COUNT" in code
                  else 0 for kind, _, _ in _C_KINDS}
    ext = ", ExternalVariable externalVariable" if externals else ""
    signatures = {
        "initialiseArrays": (f"void initialiseVariables(double voi, double *states, double *rates, double *variables{ext})"
                             if externals else "void initialiseVariables(double *states, double *rates, double *variables)"),
        # libCellML 0.7 also sets the states whose initial value is a computed constant here
        "computeComputedConstants": "void computeComputedConstants(double voi, double *states, double *rates, double *variables)",
        "computeRates": f"void computeRates(double voi, double *states, double *rates, double *variables{ext})",
        "computeVariables": f"void computeVariables(double voi, double *states, double *rates, double *variables{ext})",
    }
    for name, signature in signatures.items():
        code = re.sub(rf"void {name}\([^)]*\)", signature, code)
    code = re.sub(r"typedef double \(\* ExternalVariable\)\([^)]*\);",
                  "typedef double (* ExternalVariable)(double voi, double *states, double *rates, double *variables, size_t index);",
                  code)

    if header:
        code = re.sub(r"(extern const size_t \w+_COUNT;\n)+", "extern const size_t STATE_COUNT;\nextern const size_t VARIABLE_COUNT;\n",
                      code, count=1)
        types = ["VARIABLE_OF_INTEGRATION", "STATE", "CONSTANT", "COMPUTED_CONSTANT", "ALGEBRAIC"] + (["EXTERNAL"] if externals else [])
        enum = "typedef enum {\n" + ",\n".join(f"    {t}" for t in types) + "\n} VariableType;\n\n"
        code = code.replace("typedef struct {", enum + "typedef struct {", 1)
        code = code.replace("} VariableInfo;", "    VariableType type;\n} VariableInfo;", 1)
        code = re.sub(r"(extern const VariableInfo (CONSTANT|COMPUTED_CONSTANT|ALGEBRAIC_VARIABLE|EXTERNAL_VARIABLE)_INFO\[\];\n)+",
                      "extern const VariableInfo VARIABLE_INFO[];\n", code)
        code = re.sub(r"(double \* create(Constants|ComputedConstants|AlgebraicVariables|ExternalVariables)Array\(\);\n)+",
                      "double * createVariablesArray();\n", code)
        return code

    offsets, total = {}, 0
    for kind, array, _ in _C_KINDS:
        offsets[array] = total
        total += counts[kind]
    code = re.sub(r"(const size_t (CONSTANT|COMPUTED_CONSTANT|ALGEBRAIC_VARIABLE|EXTERNAL_VARIABLE)_COUNT = \d+;\n)+",
                  f"const size_t VARIABLE_COUNT = {total};\n", code)
    code = re.sub(r'(const VariableInfo VOI_INFO = \{[^}]*)\}', r"\1, VARIABLE_OF_INTEGRATION}", code)

    def tagged(block, variable_type):
        return [entry[:-1] + f", {variable_type}}}" for entry in re.findall(r"\{[^}]*\}", block)]

    states = re.search(r"const VariableInfo STATE_INFO\[\] = \{\n(.*?)\n\};", code, re.S)
    code = code.replace(states.group(1), ",\n".join(f"    {e}" for e in tagged(states.group(1), "STATE")))
    entries = []
    for kind, _, variable_type in _C_KINDS:
        # an empty kind (a model with no algebraic variables, say) is an array with no entries: "= {\n};"
        info = re.search(rf"const VariableInfo {kind}_INFO\[\] = \{{\n(.*?)\}};\n\n", code, re.S)
        if info:
            entries += tagged(info.group(1), variable_type)
            code = code.replace(info.group(0), "@VARIABLE_INFO@" if "@VARIABLE_INFO@" not in code else "")
    code = code.replace("@VARIABLE_INFO@", "const VariableInfo VARIABLE_INFO[] = {\n"
                        + ",\n".join(f"    {e}" for e in entries) + "\n};\n\n")
    create = re.search(r"double \* createStatesArray\(\)\n\{.*?\n\}\n\n", code, re.S).group(0)
    code = re.sub(r"(double \* create(Constants|ComputedConstants|AlgebraicVariables|ExternalVariables)Array\(\)\n\{.*?\n\}\n\n)+",
                  create.replace("States", "Variables").replace("STATE", "VARIABLE"), code, count=1, flags=re.S)

    code = re.sub(rf"externalVariable\(voi, states, rates, {_C_ALL_ARRAYS.replace('double *', '')}, externalVariables, (\d+)\)",
                  lambda m: f"externalVariable(voi, states, rates, variables, {offsets['externalVariables'] + int(m.group(1))})",
                  code)
    code = re.sub(r"\b(constants|computedConstants|algebraicVariables|externalVariables)\[(\d+)\]",
                  lambda m: f"variables[{offsets[m.group(1)] + int(m.group(2))}]", code)
    if externals:   # initialisation ends by fetching every external variable
        fetch = "".join(f"    variables[{k}] = externalVariable(voi, states, rates, variables, {k});\n"
                        for k in range(offsets["externalVariables"], total))
        start = code.index(signatures["initialiseArrays"])
        end = code.index("\n}\n", start) + 1
        code = code[:end] + fetch + code[end:]
    return code


def analyse_model(model):
    analyser = Analyser()
    a = analyser.analyseModel(model)
    _dump_issues("analyse_model", analyser)
    return a

def _get_component_node(component):
    node = {
        'id': component.name(),
        'metaid': component.id() if component.id() else "UNIDENTIFIED_COMPONENT",
        'imported': str(component.isImport()),
        'title': 'imported source:'
    }
    return node


def _get_component_hierarchy(root, source_name):
    nodes = []
    edges = []
    current_node = _get_component_node(root)
    nodes.append(current_node)
    destination = current_node['id']
    edges.append({
        'from': source_name,
        'to': destination,
        'type': 'encapsulation'
    })
    for i in range(0, root.componentCount()):
        c = root.component(i)
        child_edges, children = _get_component_hierarchy(c, destination)
        nodes.extend(children)
        edges.extend(child_edges)
    if root.isImport():
        print(root.isResolved())
        imported_model = root.importSource().model()
        imported_component = imported_model.component(root.importReference())
        imported_node = _get_component_node(imported_component)
        nodes.append(imported_node)
        imported_destination = imported_node['id']
        edges.append({
            'from': destination,
            'to': imported_destination,
            'type': 'import'
        })
        child_edges, children = _get_component_hierarchy(imported_component, destination)
        nodes.extend(children)
        edges.extend(child_edges)
    return edges, nodes


def get_model_component_hierarchy(model):
    nodes = []
    edges = []
    model_node = {
        'id': model.name() if model.name() else "UNAMED_MODEL",
        'metaid': model.id() if model.id() else "UNIDENTIFIED_MODEL",
        'imported': 'False',
        'title': 'Model node'
    }
    nodes.append(model_node)
    source = model_node['id']
    for i in range(0, model.componentCount()):
        c = model.component(i)
        child_edges, children = _get_component_hierarchy(c, source)
        nodes.extend(children)
        edges.extend(child_edges)
    return edges, nodes
