'''
Locates the CellML module files, module config JSON files and units files that a model is
generated from.

Modules come from up to four places, in this order:

1. the shipped (built-in) module library, which is package data;
2. the checkout's ``module_config_user/`` directory;
3. ``external_modules_dir``: one flat directory (or a list of them) of ``*modules.cellml``,
   ``*.json`` and ``*units.cellml`` files;
4. ``module_library_dirs``: one or more directories searched recursively, laid out one module
   per subdirectory, e.g. ``modules/<name>/<name>_modules.cellml``,
   ``<name>_modules_config.json`` and ``<name>_units.cellml``. Only JSON files named
   ``*_modules_config.json`` / ``*_module_config.json`` are read as configs, so parameter,
   obs_data or other JSON files can live next to a module.

``use_builtin_modules: false`` switches off (1) and (2) so a module library such as
circulatory-autogen-modules can be the only source of modules. Without it, redefining a
built-in (vessel_type, BC_type) in a library is rejected as a duplicate.

The environment variable ``CUFLYNX_MODULE_LIBRARY`` (one directory, or several separated by
``os.pathsep``) is the default for a config that sets neither ``module_library_dirs`` nor
``use_builtin_modules``: those modules are used, and the built-in ones are not. It is how a
whole installation (or the test suite) moves onto a module library without editing every
user_inputs.yaml.
'''

import os
import re
import xml.etree.ElementTree as ET

from libcuflynx.utilities.package_resources import builtin_modules_dir
from libcuflynx.utilities.paths import default_module_config_user_dir

MODULE_LIBRARY_ENV = 'CUFLYNX_MODULE_LIBRARY'

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


def as_dir_list(dirs):
    '''None, one path, or a list of paths -> list of paths.'''
    if dirs is None:
        return []
    if isinstance(dirs, (str, os.PathLike)):
        return [dirs]
    return list(dirs)


def module_settings(inp_data_dict):
    '''(use_builtin_modules, module_library_dirs) for ``inp_data_dict``, with the
    CUFLYNX_MODULE_LIBRARY default applied when it sets neither.'''
    use_builtin = inp_data_dict.get('use_builtin_modules')
    library_dirs = as_dir_list(inp_data_dict.get('module_library_dirs'))
    env = os.environ.get(MODULE_LIBRARY_ENV)
    if env and use_builtin is None and not library_dirs:
        return False, [d for d in env.split(os.pathsep) if d]
    return (True if use_builtin is None else bool(use_builtin)), library_dirs


class ModuleSources(object):
    '''The module, config and units files a model is generated from.'''

    def __init__(self, inp_data_dict):
        use_builtin, library_dirs = module_settings(inp_data_dict)
        external_dir = inp_data_dict.get('external_modules_dir')

        builtin = builtin_modules_dir()
        # base_script.cellml is the skeleton every generated model starts from, not a module,
        # so it is used whether or not the built-in modules are.
        self.base_script = os.path.join(builtin, 'base_script.cellml')

        self.cellml_files = []
        self.config_files = []
        self.units_files = []

        if use_builtin:
            self.cellml_files += _list_flat(builtin, _is_module_cellml)
            self.config_files += _list_flat(builtin, _is_any_json)
            self.units_files.append(os.path.join(builtin, 'units.cellml'))
            # module_config_user/ is a checkout directory, absent in a pip install
            # (#431/#432), so a missing one is "no user modules", not an error.
            user_dir = default_module_config_user_dir()
            self.cellml_files += _list_flat(user_dir, _is_module_cellml)
            self.config_files += _list_flat(user_dir, _is_any_json)
            self.units_files += _list_flat(user_dir, _is_units_cellml)

        for ext_dir in as_dir_list(external_dir):
            self.cellml_files += _list_flat(ext_dir, _is_module_cellml)
            self.config_files += _list_flat(ext_dir, _is_any_json)
            self.units_files += _list_flat(ext_dir, _is_units_cellml)

        for library_dir in library_dirs:
            if not os.path.isdir(library_dir):
                raise FileNotFoundError(f'module_library_dirs entry {library_dir} is not a directory')
            self.cellml_files += _list_recursive(library_dir, _is_module_cellml)
            self.config_files += _list_recursive(library_dir, _is_library_config_json)
            self.units_files += _list_recursive(library_dir, _is_units_cellml)


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
