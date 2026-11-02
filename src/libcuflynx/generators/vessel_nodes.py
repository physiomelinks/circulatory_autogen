"""Nodes: the points where vessel_port ends meet, and the rule each node must follow.

A module array joins modules in pairs (``a`` lists ``b`` in its out_instances), and each pair
joins one exit ``vessel_port`` of ``a`` to one entrance ``vessel_port`` of ``b``. A **node** is a
set of ends joined that way: ``a``'s outlet, the inlets of every daughter it feeds through that
port, the outlets of anything else feeding those inlets, and so on.

**The rule** (issue #616). At a node of three or more ends, the ends lie on two sides (upstream
outlets, downstream inlets), and one side has a single end. That end must be compliant:

* for a vessel, its subtype letter at the node is ``v`` (the second letter for its outlet, the
  first for its inlet), e.g. a ``pv`` parent splitting into ``pv``/``pp`` daughters, or ``vp``/
  ``pp`` vessels merging into a ``vp`` vessel;
* for any other module, its port at the node has a ``"sum"`` entry in its ``multi_port`` (a
  whole-port ``"sum"``, or a list containing ``"sum"``), e.g. a heart chamber's ``[v_in, u]``
  port with ``["sum", "True"]``. Subtype letters only mean ``v``/``p`` for vessels.

"Compliant owner" means the same for every module: the end that takes the sum and shares the
pressure. A ``"Multiply"`` or plain ``"True"`` port broadcasts a value and sums nothing, so it
cannot own a node (decided by Finbar, 2026-10-30).

That end owns the node: its flow is the sum of the flows of every end on the other side, each of
which is its direct neighbour, and its pressure is shared with them. The generator's pairwise
multi-port sums are then the node's exact equations. A node with several ends on both sides
(n-to-m) has no such end, and is refused. Nodes of two ends are plain pairs and are not checked.
Nodes with an ``FV1D_vessel`` end are coupled by the 1D coupler and are not checked either.

Ports are taken as the generator takes them: the n-th neighbour on a side (with a vessel_port
there) takes the n-th vessel_port of that side, a multi_port port takes every later neighbour
too, and neighbours beyond the last port share the last one. General ports are not part of nodes.
"""
from collections import namedtuple

from libcuflynx.generators.multi_port import MULTI_PORT_SUM, is_multi_port, list_multi_port
from libcuflynx.utilities.vessel_bc import is_vessel_module

VESSEL_PORT = 'vessel_port'
EXIT = 'exit'
ENTRANCE = 'entrance'

End = namedtuple('End', ['module', 'side', 'port'])  # port: index into that side's port list


def _ports(row, side):
    ports = row['exit_ports'] if side == EXIT else row['entrance_ports']
    return ports if isinstance(ports, (list, tuple)) else []


def _vessel_port_indices(row, side):
    return [i for i, port in enumerate(_ports(row, side))
            if isinstance(port, dict) and port.get('port_type') == VESSEL_PORT]


def _port_taken(row, side, position):
    """The index of the vessel_port of ``row`` on ``side`` that its ``position``-th neighbour
    there takes."""
    ports = _ports(row, side)
    indices = _vessel_port_indices(row, side)
    for k, index in enumerate(indices):
        if k == position or is_multi_port(ports[index]):
            return index
    return indices[-1]


def find_vessel_nodes(rows):
    """The nodes of the modules in ``rows`` ({name: row}, in array order), each a list of
    ``End``, outlets first, in array order."""
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def has_side(name, side):
        return name in rows and bool(_vessel_port_indices(rows[name], side))

    for name, row in rows.items():
        if not has_side(name, EXIT):
            continue
        outs = [n for n in row['out_instances'] if has_side(n, ENTRANCE)]
        for position, out_name in enumerate(outs):
            inps = [n for n in rows[out_name]['inp_instances'] if has_side(n, EXIT)]
            if name not in inps:
                continue
            upstream = End(name, EXIT, _port_taken(row, EXIT, position))
            downstream = End(out_name, ENTRANCE,
                             _port_taken(rows[out_name], ENTRANCE, inps.index(name)))
            parent[find(upstream)] = find(downstream)

    groups = {}
    for key in list(parent):
        groups.setdefault(find(key), []).append(key)
    order = {name: i for i, name in enumerate(rows)}
    return [sorted(ends, key=lambda e: (e.side != EXIT, order[e.module], e.port))
            for ends in groups.values()]


def _letter(row, end):
    return row['module_subtype'][1 if end.side == EXIT else 0]


def is_compliant(row, end):
    """Whether ``end`` of the module in ``row`` can own its node (see the module docstring)."""
    if is_vessel_module(row):
        return _letter(row, end) == 'v'
    return MULTI_PORT_SUM in (list_multi_port(_ports(row, end.side)[end.port]) or [])


def _describe(rows, end):
    row = rows[end.module]
    where = 'outlet' if end.side == EXIT else 'inlet'
    return f'the {where} of "{end.module}" ({row["module_type"]} {row["module_subtype"]})'


def _swap_hint(rows, lone, others):
    """Fix 1 as concrete subtype changes, or None when the lone end is not a vessel's."""
    row = rows[lone.module]
    if not is_vessel_module(row):
        return None
    changes = []
    for end, letter in [(lone, 'v')] + [(o, 'p') for o in others]:
        r = rows[end.module]
        if not is_vessel_module(r) or _letter(r, end) == letter:
            continue
        old = r['module_subtype'][:2]
        new = old[0] + letter if end.side == EXIT else letter + old[1]
        changes.append(f'"{end.module}" {old} -> {new}')
    return ', '.join(changes)


def check_vessel_nodes(module_df):
    """Raise ValueError for the first node of ``module_df`` that breaks the rule."""
    rows = {row['name']: row for _, row in module_df.iterrows()}
    for ends in find_vessel_nodes(rows):
        if len(ends) < 3 or any(rows[e.module]['module_type'] == 'FV1D_vessel' for e in ends):
            continue
        upstream = [e for e in ends if e.side == EXIT]
        downstream = [e for e in ends if e.side == ENTRANCE]
        node = ', '.join(_describe(rows, e) for e in ends)
        if len(upstream) > 1 and len(downstream) > 1:
            raise ValueError(
                f'{len(upstream)} outlets meet {len(downstream)} inlets at one node ({node}), so '
                f'no single end can own it. Join them through a 2in2out_junction (vv), which has '
                f'a port per connection, or add a lumped vessel (vv) that the upstream vessels '
                f'merge into and the downstream ones split from. See issue #616.')
        lone = upstream[0] if len(upstream) == 1 else downstream[0]
        if is_compliant(rows[lone.module], lone):
            continue
        others = [e for e in ends if e is not lone]
        hint = _swap_hint(rows, lone, others)
        fix_1 = (f'swap subtypes so that end is "v" and the ends on the other side are "p" '
                 f'({hint}), using a module_type whose port there takes several connections'
                 if hint else 'give that end a port with a "sum" entry in its multi_port (e.g. '
                 '["sum", "True"]); a "Multiply" or "True" port broadcasts a value and sums '
                 'nothing, so it cannot own a node')
        raise ValueError(
            f'At the node joining {node}, {_describe(rows, lone)} is the only end on its side, '
            f'so it must be compliant: it owns the node, summing the flows of the other ends and '
            f'sharing its pressure with them. Either {fix_1}; or add a lumped vessel (vv) '
            f'between "{lone.module}" and the other ends, which then owns the node. '
            f'See issue #616.')
