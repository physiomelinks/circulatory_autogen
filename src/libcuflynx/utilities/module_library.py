'''
Locates the CellML module files, module config JSON files and units files that a model is
generated from.

Modules come from up to four places, in this order:

1. the shipped (built-in) module library, which is package data;
2. the checkout's ``module_config_user/`` directory;
3. ``external_modules_dir``: one flat directory of ``*modules.cellml``, ``*.json`` and
   ``*units.cellml`` files;
4. ``module_library_dirs``: one or more directories searched recursively, laid out one module
   per subdirectory, e.g. ``modules/<name>/<name>_modules.cellml``,
   ``<name>_modules_config.json`` and ``<name>_units.cellml``. Only JSON files named
   ``*_modules_config.json`` / ``*_module_config.json`` are read as configs, so parameter,
   obs_data or other JSON files can live next to a module.

``use_builtin_modules: false`` switches off (1) and (2) so a module library such as
circulatory-autogen-modules can be the only source of modules.

A (vessel_type, BC_type) defined in more than one source is taken from the most specific:

    external_modules_dir > module_library_dirs (in the order listed) > module_config_user
                         > built-in

The others' config entries and CellML components for it are left out
(``ModuleSources.excluded_entries`` / ``excluded_components``), and one
``ModuleShadowWarning`` lists every type that more than one source defines, and which source
won. Defining a type twice within one source is an error: that is a broken source, not a
choice between two.
'''

import json
import os
import re
import warnings
import xml.etree.ElementTree as ET

from libcuflynx.utilities.package_resources import builtin_modules_dir
from libcuflynx.utilities.paths import default_module_config_user_dir

CELLML_1_1_NS = 'http://www.cellml.org/cellml/1.1#'

_UNITS_BLOCK_RE = re.compile(r'<units\b[^>]*?\bname="([^"]+)"[^>]*?(?:/>|>.*?</units>)', re.S)


def _is_hidden(filename):
    # `._` covers macOS AppleDouble sidecar files, which share the real file's suffix but
    # are binary and break the parsers (issue #83).
    return filename.startswith('.')


def _list_flat(directory, predicate):
    if not directory or not os.path.isdir(directory):
        return []
    return [os.path.join(directory, f) for f in sorted(os.listdir(directory))
            if not _is_hidden(f) and predicate(f)]


def _list_recursive(directory, predicate):
    found = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = sorted(d for d in dirs if not _is_hidden(d))
        found += [os.path.join(root, f) for f in sorted(files) if not _is_hidden(f) and predicate(f)]
    return found


def _is_module_cellml(filename):
    return filename.endswith('modules.cellml')


def _is_any_json(filename):
    return filename.endswith('.json')


def _is_library_config_json(filename):
    return filename.endswith('_modules_config.json') or filename.endswith('_module_config.json')


def _is_units_cellml(filename):
    return filename.endswith('units.cellml')


class ModuleShadowWarning(UserWarning):
    '''More than one module source defines a (vessel_type, BC_type); the most specific one is
    used. ``shadowed`` is ``{key: (winning config file, [the config files it shadows])}``.'''

    def __init__(self, message, shadowed=None):
        super().__init__(message)
        self.shadowed = dict(shadowed or {})


def _config_entries(path):
    '''The normalised entries of a module config file (components and supermodules); none for
    a JSON file that is not a list of entries.'''
    from libcuflynx.utilities.config_schemas import normalise_module_config
    with open(path, encoding='utf-8-sig') as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        return []
    return normalise_module_config(raw, source=str(path))


def _same_file(a, b):
    return os.path.realpath(a) == os.path.realpath(b)


def resolve_duplicate_types(config_files, source_of, precedence):
    '''
    Which config entries and CellML components a type defined in several sources leaves out.

    ``source_of`` maps each of ``config_files`` to its source's label and ``precedence`` lists
    the labels, most specific first. Returns ``(excluded_entries, excluded_components)``:
    ``{(config file, (vessel_type, BC_type))}`` of the shadowed definitions, and
    ``{(realpath of a CellML file, component name)}`` of their components that no definition
    still in use needs. Warns once (``ModuleShadowWarning``) when any type is shadowed; raises
    ValueError for a type defined twice within one source.
    '''
    definitions = {}
    for path in config_files:
        for entry in _config_entries(path):
            key = (entry['vessel_type'], entry['BC_type'])
            definitions.setdefault(key, []).append((str(path), entry))

    excluded, shadowed = set(), {}
    for key, found in definitions.items():
        by_source = {}
        for path, entry in found:
            by_source.setdefault(source_of[path], []).append(path)
        for label, paths in by_source.items():
            if len(paths) > 1:
                raise ValueError(f'{key} is defined {len(paths)} times in {label} ({paths}); a '
                                 f'module source defines each (module_type, module_subtype) '
                                 f'once.')
        if len(found) < 2:
            continue
        ranked = sorted(found, key=lambda pe: precedence.index(source_of[pe[0]]))
        losers = [path for path, _ in ranked[1:]]
        excluded |= {(path, key) for path in losers}
        shadowed[key] = (ranked[0][0], losers)

    def component(path, entry):
        if not entry.get('module_file') or not entry.get('module_type'):
            return None
        cellml = os.path.join(os.path.dirname(os.path.abspath(path)), entry['module_file'])
        return os.path.realpath(cellml), entry['module_type']

    kept = {component(path, entry) for found in definitions.values() for path, entry in found
            if (path, (entry['vessel_type'], entry['BC_type'])) not in excluded}
    excluded_components = {component(path, entry) for found in definitions.values()
                           for path, entry in found
                           if (path, (entry['vessel_type'], entry['BC_type'])) in excluded}
    excluded_components = {c for c in excluded_components if c is not None and c not in kept}

    if shadowed:
        listed = '; '.join(f'{key[0]}/{key[1]}: {source_of[winner]} ({winner}) over '
                           + ', '.join(f'{source_of[p]} ({p})' for p in losers)
                           for key, (winner, losers) in sorted(shadowed.items()))
        warnings.warn(ModuleShadowWarning(
            f'{len(shadowed)} module type(s) are defined in more than one module source; each '
            f'is taken from the most specific (external_modules_dir, then module_library_dirs '
            f'in order, then module_config_user, then the built-in modules): {listed}. Remove '
            f'the copies you do not mean to use, or set use_builtin_modules: false if a library '
            f'replaces the built-in modules.', shadowed), stacklevel=3)
    return excluded, excluded_components


def as_dir_list(dirs):
    '''None, one path, or a list of paths -> list of paths.'''
    if dirs is None:
        return []
    if isinstance(dirs, (str, os.PathLike)):
        return [dirs]
    return list(dirs)


class ModuleSources(object):
    '''The module, config and units files a model is generated from.'''

    def __init__(self, inp_data_dict):
        use_builtin = inp_data_dict.get('use_builtin_modules', True)
        if use_builtin is None:
            use_builtin = True
        external_dir = inp_data_dict.get('external_modules_dir')
        library_dirs = as_dir_list(inp_data_dict.get('module_library_dirs'))

        builtin = builtin_modules_dir()
        # base_script.cellml is the skeleton every generated model starts from, not a module,
        # so it is used whether or not the built-in modules are.
        self.base_script = os.path.join(builtin, 'base_script.cellml')

        self.cellml_files = []
        self.config_files = []
        self.units_files = []
        source_of = {}

        def add(label, configs):
            self.config_files.extend(configs)
            source_of.update((str(path), label) for path in configs)

        if use_builtin:
            self.cellml_files += _list_flat(builtin, _is_module_cellml)
            add('the built-in modules', _list_flat(builtin, _is_any_json))
            self.units_files.append(os.path.join(builtin, 'units.cellml'))
            # module_config_user/ is a checkout directory, absent in a pip install
            # (#431/#432), so a missing one is "no user modules", not an error.
            user_dir = default_module_config_user_dir()
            self.cellml_files += _list_flat(user_dir, _is_module_cellml)
            add('module_config_user', _list_flat(user_dir, _is_any_json))
            self.units_files += _list_flat(user_dir, _is_units_cellml)

        if external_dir is not None:
            self.cellml_files += _list_flat(external_dir, _is_module_cellml)
            add('external_modules_dir', _list_flat(external_dir, _is_any_json))
            self.units_files += _list_flat(external_dir, _is_units_cellml)

        libraries = []
        for i, library_dir in enumerate(library_dirs):
            if not os.path.isdir(library_dir):
                raise FileNotFoundError(f'module_library_dirs entry {library_dir} is not a directory')
            label = f'module_library_dirs[{i}] ({library_dir})'
            libraries.append(label)
            self.cellml_files += _list_recursive(library_dir, _is_module_cellml)
            add(label, _list_recursive(library_dir, _is_library_config_json))
            self.units_files += _list_recursive(library_dir, _is_units_cellml)

        # the most specific source first
        precedence = (['external_modules_dir'] + libraries
                      + ['module_config_user', 'the built-in modules'])
        self.excluded_entries, self.excluded_components = resolve_duplicate_types(
            self.config_files, source_of, precedence)

    def component_registry(self):
        '''``config_schemas.load_component_registry`` of these sources, shadowed types left out.'''
        from libcuflynx.utilities.config_schemas import load_component_registry
        return load_component_registry(self.config_files, exclude=self.excluded_entries)

    def supermodule_registry(self):
        '''``config_schemas.load_supermodule_registry`` of these sources, shadowed types left
        out.'''
        from libcuflynx.utilities.config_schemas import load_supermodule_registry
        return load_supermodule_registry(self.config_files, exclude=self.excluded_entries)

    def uses_component(self, cellml_file, component):
        '''Whether ``component`` of the module file ``cellml_file`` belongs in the model, i.e.
        is not the CellML of a type a more specific source defines.'''
        return (os.path.realpath(cellml_file), component) not in self.excluded_components


def _canonical_units(block):
    '''A comparable form of one <units> definition, independent of formatting.'''
    wrapped = f'<model xmlns="{CELLML_1_1_NS}">{block}</model>'
    units = ET.fromstring(wrapped)[0]
    children = tuple(sorted(tuple(sorted(child.attrib.items())) for child in units))
    return tuple(sorted(units.attrib.items())), children


def collect_units(units_files):
    '''
    Reads every <units> definition from ``units_files``, in order.

    Returns a list of (name, raw_text_block). A unit defined identically in more than one
    file is kept once; a unit defined differently in two files raises ValueError, since
    silently picking one would change the model's numbers.
    '''
    seen = {}
    ordered = []
    for path in units_files:
        with open(path, 'r') as f:
            text = f.read()
        for match in _UNITS_BLOCK_RE.finditer(text):
            name, block = match.group(1), match.group(0)
            canonical = _canonical_units(block)
            if name in seen:
                first_path, first_canonical = seen[name]
                if canonical != first_canonical:
                    raise ValueError(f'units "{name}" is defined differently in {first_path} and {path}')
                continue
            seen[name] = (path, canonical)
            ordered.append((name, block))
    return ordered
