"""Writing supermodule versions for tests.

A test states a supermodule as one dict with its instance records inline under
``submodules`` and, optionally, its instances' values under ``instance_parameters``
(``{instance: {variable: (value, units[, data_reference])}}``). ``split_supermodule`` turns it into what a library
holds: the config entry, with the ``submodule_versions`` its records use, the records of its
module array (``config_schemas.supermodule_array_path``) and its template, the instances' values
its default parameterisation holds. ``write_supermodule(s)`` writes all three; the template goes
into ``parameterisations/<default>/<default>_parameters.json`` (``default`` unless the entry
names another), over what that file holds.
"""
import json
import os

from libcuflynx.utilities import parameters_file


def split_supermodule(entry):
    """``(config entry, array records, template)`` of a supermodule dict with inline
    ``submodules``: the template is ``{instance: {variable: parameter}}``, ``{}`` without
    ``instance_parameters``."""
    entry = dict(entry)
    records = [dict(r) for r in entry.pop('submodules')]
    values = entry.pop('instance_parameters', {})
    versions = dict.fromkeys((r['module_type'], r['module_subtype']) for r in records)
    entry['submodule_versions'] = [{'module_type': t, 'module_subtype': s} for t, s in versions]
    template = {name: {v: dict(zip(('value', 'units', 'data_reference'), given))
                       for v, given in variables.items()}
                for name, variables in values.items()}
    return entry, records, template


def array_file_name(entry):
    return f'{entry["module_type"]}_{entry["module_subtype"]}_module_array.json'


def write_template(directory, entry, template):
    """Put ``template`` (``{instance: {variable: parameter}}``) into the default
    parameterisation of the supermodule ``entry`` whose config is in ``directory``, over what it
    holds; names ``default`` its default when it has none. Returns the entry."""
    entry = dict(entry)
    name = entry.setdefault('default_parameterisation', 'default')
    path = os.path.join(directory, 'parameterisations', name, f'{name}_parameters.json')
    doc = (parameters_file.read_parameters_file(path, kind=parameters_file.MODULE)
           if os.path.isfile(path) else parameters_file.new_document())
    for instance, variables in template.items():
        for variable, parameter in variables.items():
            parameters_file.set_leaf(doc, [instance] + variable.split('/'), parameter)
    parameters_file.write_parameters_file(path, doc)
    return entry


def write_supermodules(directory, entries, config_name):
    """Write ``entries`` (supermodule dicts with inline ``submodules``, or other config
    entries, kept as they are) to ``<directory>/<config_name>``, each supermodule's records to
    its module array beside it and their values to its default parameterisation. Returns the
    config path."""
    os.makedirs(directory, exist_ok=True)
    written = []
    for entry in entries:
        if isinstance(entry.get('submodules'), list):
            entry, records, template = split_supermodule(entry)
            with open(os.path.join(directory, array_file_name(entry)), 'w') as f:
                json.dump(records, f, indent=1)
            if template:
                entry = write_template(directory, entry, template)
        written.append(entry)
    path = os.path.join(str(directory), config_name)
    with open(path, 'w') as f:
        json.dump(written, f, indent=1)
    return path


def write_supermodule(directory, entry, config_name=None):
    """Write one supermodule (``entry`` with inline ``submodules``) to
    ``<directory>/<module_type>_<module_subtype>_modules_config.json`` (or ``config_name``)
    and its module array and template beside it. Returns the config path."""
    name = config_name or f'{entry["module_type"]}_{entry["module_subtype"]}_modules_config.json'
    return write_supermodules(directory, [entry], name)
