'''
Readable identifiers for a model's states and variables, shared by the generators.

Each state or variable is named ``<component>_<variable>`` (lower case, anything that is not a
letter or digit collapsed to ``_``); a repeated name gets ``_2``, ``_3``, ... in index order.
The Python generator uses the names as attributes (``var.pvn_module_u``) and the C++ generator as
index constants (``variables[V_pvn_module_u]``), so the two match.
'''

import re


def make_identifier(text):
    '''A valid identifier from arbitrary text.'''
    identifier = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")
    identifier = re.sub(r"_+", "_", identifier)
    if not identifier:
        identifier = "unnamed"
    if identifier[0].isdigit():
        identifier = f"N_{identifier}"
    return identifier


def build_symbols(component_name_pairs):
    '''``<component>_<name>`` identifiers for (component, name) pairs given in index order.

    Returns a list with one identifier per pair; repeats get ``_2``, ``_3``, ...
    '''
    symbols = []
    counts = {}
    for component, name in component_name_pairs:
        base_name = make_identifier(f"{component}_{name}").lower()
        count = counts.get(base_name, 0)
        symbols.append(base_name if count == 0 else f"{base_name}_{count + 1}")
        counts[base_name] = count + 1
    return symbols
