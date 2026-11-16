'''
The files of user functions -- operations, cost functions, modifiers -- that the
``operation_funcs_external_path``, ``cost_funcs_external_path`` and
``modifier_funcs_external_path`` settings name.

One layout for every writer (CUFLynx's "Custom funcs" dialog, and
``calibration_workflow.WorkflowBuilder.add_operation``), so either can read and rewrite what
the other wrote::

    """<the kind's header docstring>"""
    import numpy as np
    from libcuflynx... import <the kind's decorators>
    mb = make_math_backend("numpy")         (operations and cost functions)
    # libcuflynx-generated sha256: <hex>     (the checksum of the rest of the file)

    import scipy.signal as signal           (the imports the functions use, if any)

    CUFLYNX_OPERATIONS = ["peak_count", "half_mean"]     (the functions, in order)


    @series_to_constant
    def half_mean(x, series_output=False):
        ...

The decorators are imported, never defined, so libcuflynx registers only the functions below
them (``param_id.external_funcs.register_funcs_from_file`` keeps the callables whose module is
the file). :func:`read_funcs_file` reads such a file back: its functions (each with its
decorators) in order, and the imports beyond the header's.

The checksum line is a comment, so Python and any reader of the functions ignore it. It is the
sha256 of the file without that line: while it matches (:func:`is_unedited`), the file is
exactly what a writer generated and can be generated again without losing anything. A file
holds only the header, imports and functions: :func:`unkept_statements` lists what a rewrite
would drop (constants, classes), so a writer can refuse instead.
'''

import ast
import hashlib
import os

#: the kinds of user function file and the setting each is named by
FUNCS_SETTINGS = {'operation': 'operation_funcs_external_path',
                  'cost': 'cost_funcs_external_path',
                  'modifier': 'modifier_funcs_external_path'}
#: the top-level assignment that lists a file's functions in order
LIST_MARKERS = {'operation': 'CUFLYNX_OPERATIONS', 'cost': 'CUFLYNX_COSTS',
                'modifier': 'CUFLYNX_MODIFIERS'}

_OPERATION_HEADER = '''"""User-defined observable operations.

Each top-level function here is registered as an "operation" that obs_data items can name,
and used by circulatory_autogen during calibration / sensitivity / UQ (loaded from the
operation_funcs_external_path setting). Written by CUFLynx's "Custom funcs" dialog or by
libcuflynx's WorkflowBuilder.add_operation, which rewrite it; edit the functions there.

An operation receives the operand array(s) for a data_item and returns a scalar.
Return the operand series when ``series_output=True`` so the reduction can be
plotted on top of the series -- that returned series is what is drawn with the
feature in the plots. ``np`` (numpy) is available for plain funcs; write the body
against the math backend ``mb`` (and add ``@differentiable``) to use the operation
with AD or FSA gradients. The ``@differentiable`` / ``@series_to_constant`` markers
are libcuflynx's (imported, never redefined).
"""
import numpy as np  # noqa: F401 -- available to user operations

# Imported (not defined) so libcuflynx registers only the user funcs below, never these.
from libcuflynx.param_id.differentiable import differentiable  # noqa: F401
from libcuflynx.param_id.operation_funcs import series_to_constant  # noqa: F401
from libcuflynx.param_id.math_backend import make_math_backend

# Math backend for differentiable ops: use ``mb.<op>`` instead of numpy so libcuflynx can
# rebind it to casadi and take symbolic gradients. It is set per backend.
mb = make_math_backend("numpy")
'''

_COST_HEADER = '''"""User-defined cost functions.

Each top-level function here is registered as a "cost_type" that obs_data items can name,
and used by circulatory_autogen during calibration / sensitivity / UQ (loaded from the
cost_funcs_external_path setting). Written by CUFLynx's "Custom funcs" dialog or by
libcuflynx's WorkflowBuilder.add_cost_func, which rewrite it; edit the functions there.

A cost func compares a model ``output`` to its target and returns a scalar cost
(lower = better fit). It must work for both scalars and arrays. ``np`` (numpy) is
available for plain funcs; write the body against the math backend ``mb`` (and add
``@differentiable``) for AD gradients. The ``@differentiable`` / ``@is_MLE`` /
``@cost_combiner`` markers are libcuflynx's (imported, never redefined).

libcuflynx fills the model output and the ground truth positionally, then supplies ``std``
and ``weight`` only when the signature declares them. Any further keyword argument (give it
a default) is filled per data_item from that item's ``cost_kwargs`` in obs_data.json.
"""
import numpy as np  # noqa: F401 -- available to user cost funcs

# Imported (not defined) so libcuflynx registers only the user funcs below, never these.
from libcuflynx.param_id.differentiable import differentiable  # noqa: F401
from libcuflynx.param_id.math_backend import make_math_backend
from libcuflynx.funcs.cost_funcs_user import is_MLE, cost_combiner  # noqa: F401

# Math backend for differentiable costs: use ``mb.<op>`` instead of numpy so libcuflynx can
# rebind it to casadi and take symbolic gradients. It is set per backend.
mb = make_math_backend("numpy")
'''

_MODIFIER_HEADER = '''"""User-defined parameter modifiers.

Each top-level function here carrying ``@modifier_func`` is registered as a "modifier" that
params_for_id entries can name, and used by circulatory_autogen during calibration /
sensitivity / UQ (loaded from the modifier_funcs_external_path setting). Written by CUFLynx's
"Custom funcs" dialog or by libcuflynx's WorkflowBuilder.add_modifier_func, which rewrite it;
edit the functions there.

A modifier says how one calibrated variable (``theta``) computes each of the model
parameters it governs::

    p_i = fn(theta, baseline_i, **inputs)

``baseline_i`` is that target's model-default value. ``inputs`` are extra model
constants the params_for_id entry names by qname, resolved to their defaults once
at setup. Declare them with ``@modifier_func(inputs={...})``: ``"float"`` for one
qname, ``"list"`` for several.

**Every modifier must be affine in theta** (``a*theta + b`` for fixed inputs): the analytic
gradients apply a constant chain-rule weight ``a = dp_i/dtheta``, and theta's starting value
is derived by inverting the mapping at the baseline. libcuflynx checks this at setup.
"""
import numpy as np  # noqa: F401 -- available to user modifiers

# Imported (not defined) so libcuflynx registers only the decorated funcs below, never this.
from libcuflynx.param_id.modifier_funcs import modifier_func  # noqa: F401
'''

#: the header each kind of file starts with
HEADERS = {'operation': _OPERATION_HEADER, 'cost': _COST_HEADER, 'modifier': _MODIFIER_HEADER}

#: the names each header defines: {name: (module, attribute)}, attribute None for a module,
#: (None, None) for the math backend ``mb``
HEADER_NAMES = {
    'operation': {'np': ('numpy', None),
                  'differentiable': ('libcuflynx.param_id.differentiable', 'differentiable'),
                  'series_to_constant': ('libcuflynx.param_id.operation_funcs',
                                         'series_to_constant'),
                  'make_math_backend': ('libcuflynx.param_id.math_backend', 'make_math_backend'),
                  'mb': (None, None)},
    'cost': {'np': ('numpy', None),
             'differentiable': ('libcuflynx.param_id.differentiable', 'differentiable'),
             'make_math_backend': ('libcuflynx.param_id.math_backend', 'make_math_backend'),
             'is_MLE': ('libcuflynx.funcs.cost_funcs_user', 'is_MLE'),
             'cost_combiner': ('libcuflynx.funcs.cost_funcs_user', 'cost_combiner'),
             'mb': (None, None)},
    'modifier': {'np': ('numpy', None),
                 'modifier_func': ('libcuflynx.param_id.modifier_funcs', 'modifier_func')},
}


#: the start of the comment line that holds a generated file's checksum
CHECKSUM_PREFIX = '# libcuflynx-generated sha256: '


def _split_checksum(text):
    '''``(recorded checksum or None, text without its checksum line)``, line ends as \\n.'''
    lines = text.replace('\r\n', '\n').split('\n')
    recorded = [line[len(CHECKSUM_PREFIX):].strip() for line in lines
                if line.startswith(CHECKSUM_PREFIX)]
    rest = '\n'.join(line for line in lines if not line.startswith(CHECKSUM_PREFIX))
    return (recorded[0] if len(recorded) == 1 else None), rest


def _checksum(rest):
    return hashlib.sha256(rest.encode('utf-8')).hexdigest()


def is_unedited(text):
    '''True when ``text`` (a functions file's content) is as a writer generated it: it
    records a checksum and the rest of it still has that checksum.'''
    recorded, rest = _split_checksum(text)
    return recorded is not None and recorded == _checksum(rest)


def funcs_file_source(kind, sources, imports=()):
    '''The text of a ``kind`` (operation, cost, modifier) functions file, with its checksum
    line after the header.

    Args:
        sources: ``{name: source}``, each one top-level ``def`` with its decorators, in order.
        imports: import statements the functions use, beyond the header's.
    '''
    names = ', '.join(f'"{name}"' for name in sources)
    header = HEADERS[kind].rstrip('\n')
    parts = []
    if imports:
        parts += ['', '# what the functions below use'] + list(dict.fromkeys(imports))
    parts += ['', '', f'{LIST_MARKERS[kind]} = [{names}]']
    for source in sources.values():
        parts += ['', '', source.strip('\n')]
    body = '\n'.join(parts) + '\n'
    return f'{header}\n{CHECKSUM_PREFIX}{_checksum(header + chr(10) + body)}\n{body}'


def write_funcs_file(path, kind, sources, imports=()):
    '''Write :func:`funcs_file_source` to ``path`` (only when its text changes). Returns
    ``path``.'''
    text = funcs_file_source(kind, sources, imports)
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            if f.read() == text:
                return path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(text)
    os.replace(tmp, path)
    return path


def read_funcs_file(path, kind):
    '''``(sources, imports)`` of a functions file in this layout: ``{name: source}`` of its
    top-level functions (with their decorators) in the listed order, and the top-level import
    statements beyond the header's. None when the file has no ``kind`` list marker.'''
    with open(path, encoding='utf-8') as f:
        text = f.read()
    tree = ast.parse(text)
    marker = LIST_MARKERS[kind]
    order = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == marker
                                                for t in node.targets):
            order = [str(x) for x in ast.literal_eval(node.value)]
    if order is None:
        return None
    lines = text.splitlines()
    header = set(HEADERS[kind].splitlines())
    sources, imports = {}, []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            statement = '\n'.join(lines[node.lineno - 1:node.end_lineno])
            if statement not in header:
                imports.append(statement)
        elif isinstance(node, ast.FunctionDef):
            start = min([d.lineno for d in node.decorator_list] + [node.lineno]) - 1
            sources[node.name] = '\n'.join(lines[start:node.end_lineno]) + '\n'
    ordered = {name: sources[name] for name in order if name in sources}
    ordered.update({name: s for name, s in sources.items() if name not in ordered})
    return ordered, imports


def unkept_statements(path, kind):
    '''The top-level statements of the functions file ``path`` that :func:`read_funcs_file`
    does not return, so a rewrite from what it returns would drop them: everything but the
    header (its docstring, imports and ``mb``), the list marker, imports and ``def``
    functions. Each as ``"<first line> (line N)"``.'''
    with open(path, encoding='utf-8') as f:
        text = f.read()
    tree = ast.parse(text)
    lines = text.splitlines()
    header = {ast.dump(node) for node in ast.parse(HEADERS[kind]).body}
    marker = LIST_MARKERS[kind]
    unkept = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)) \
                or ast.dump(node) in header:
            continue
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == marker
                                                for t in node.targets):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and node is tree.body[0]:
            continue                      # another writer's header docstring
        unkept.append(f'{lines[node.lineno - 1].strip()} (line {node.lineno})')
    return unkept
