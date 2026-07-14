import os

try:
    from libcellml import Analyser, AnalyserModel, AnalyserExternalVariable, Component, Generator, GeneratorProfile, \
                          Parser, Importer, Model, Printer, Validator
    LIBCELLML_available = True
except ImportError as e:
    print("Error -> ", e)
    print('continuing without LibCellML, Warning code checks will not be available.'
          'You will need to open generated models in OpenCOR to check for errors.')
    LIBCELLML_available = False
import utilities.libcellml_utilities as libcellml_utils


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
    """The AnalyserModel produced by analyser.analyseModel(...).

    libCellML renamed this accessor from Analyser.model() to Analyser.analyserModel() in
    0.7.0, so support both rather than pinning users to one side of that release.
    """
    if hasattr(analyser, 'analyserModel'):
        return analyser.analyserModel()
    return analyser.model()


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
