'''
Nodes: the points where vessel ports meet.

A module array connects modules in pairs (``A`` lists ``B`` in its out_vessels), but where several
vessels meet, the physics is a property of the point they share, not of each pair. Take
``aortic_arch_II`` flowing into a node that ``aortic_arch_III`` and ``common_carotid_L`` both leave
from, where ``aortic_arch_III``'s inlet compliance sets the pressure: ``aortic_arch_III``'s net
inflow is ``v(aortic_arch_II) - v(common_carotid_L)``, and ``common_carotid_L`` takes its inlet
pressure from ``aortic_arch_III``, a module it is not connected to in the array.

A **node** is the set of vessel-port ends joined by connections: an exit end of ``A`` and the
entrance ends of everything in ``A``'s out_vessels, closed under the same rule from each of those.
Its **owner** is the one end whose vessel_port has a list-form multi_port with a ``"sum"`` entry
(normally the compliant end of a vessel, ``["sum", "True"]`` on ``[flow, pressure]``). For every
other end, position by position in the port's variables:

* a ``"sum"`` entry of the owner gets that end's variable as a term, added when the end is on the
  other side of the node from the owner (it flows towards the owner) and subtracted when it is on
  the same side (it flows away);
* a ``"True"`` entry of the owner is mapped to that end's variable (one source, many sinks).

A node is only built from module ends that have exactly one vessel_port on that side. Modules with
several vessel_ports on one side (the heart, the legacy fixed-arity junctions) keep the pairwise
port matching, as does every node with no owner and every node with a "Multiply" port.
'''
from collections import namedtuple

from libcuflynx.generators.multi_port import (MULTI_PORT_SHARED, MULTI_PORT_SUM, is_multiply_port,
                                              list_multi_port)

VESSEL_PORT = 'vessel_port'
EXIT = 'exit'
ENTRANCE = 'entrance'

Endpoint = namedtuple('Endpoint', ['module', 'side', 'port'])


class Node(object):
    '''The ends meeting at one point, and (once resolved) the end that owns it.'''

    def __init__(self, endpoints):
        self.endpoints = endpoints
        self.owner = None

    @property
    def description(self):
        return ', '.join(f'{e.module} ({e.side})' for e in self.endpoints)

    def edges(self):
        '''Every (upstream, downstream) module pair joined at this node.'''
        exits = [e.module for e in self.endpoints if e.side == EXIT]
        entrances = [e.module for e in self.endpoints if e.side == ENTRANCE]
        return {(a, b) for a in exits for b in entrances}

    def sign(self, endpoint):
        '''+1 if ``endpoint``'s flow goes towards the owner, -1 if it goes away from it.'''
        return 1. if endpoint.side != self.owner.side else -1.


def _vessel_ports(row, side):
    ports = row['exit_ports'] if side == EXIT else row['entrance_ports']
    return [port for port in ports if port.get('port_type') == VESSEL_PORT]


def _owns(port):
    entries = list_multi_port(port)
    return entries is not None and MULTI_PORT_SUM in entries


def find_nodes(module_df):
    '''The nodes of the module array in ``module_df``, each with its owner set (or None).

    Only nodes that some module could own are returned: nodes of two or more ends, every end of
    which is a single vessel_port of a module. Raises ValueError for a node with more than one
    owner.
    '''
    rows = {row['name']: row for _, row in module_df.iterrows()}

    parent = {}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a, b):
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    single = {}
    for name, row in rows.items():
        for side in (EXIT, ENTRANCE):
            ports = _vessel_ports(row, side)
            if len(ports) == 1:
                single[(name, side)] = ports[0]

    for name, row in rows.items():
        if (name, EXIT) not in single:
            continue
        for out_name in row['out_vessels']:
            if (out_name, ENTRANCE) in single:
                union((name, EXIT), (out_name, ENTRANCE))

    groups = {}
    for key in parent:
        groups.setdefault(find(key), []).append(key)

    nodes = []
    for keys in groups.values():
        keys.sort(key=lambda key: (key[1] != EXIT, list(rows).index(key[0])))
        endpoints = [Endpoint(module, side, single[(module, side)]) for module, side in keys]
        node = Node(endpoints)
        owners = [e for e in endpoints if _owns(e.port)]
        if len(owners) > 1:
            raise ValueError(
                f'The node joining {node.description} has more than one end whose vessel_port '
                f'sums over the node ({", ".join(e.module for e in owners)}). Only one module can '
                f'set the pressure where vessels meet: give the others a port without a "sum" '
                f'multi_port (a pressure-input end, e.g. BC_type "p" on that side).')
        node.owner = owners[0] if owners else None
        nodes.append(node)
    return nodes


def node_variable_pairs(node, module_formats):
    '''How the owner of ``node`` connects to each other end.

    Returns (sum_terms, shared), where ``sum_terms`` is a list of
    (owner_variable, end_module, end_variable, sign) and ``shared`` a list of
    (owner_variable, end_module, end_variable). Raises ValueError for a port pair that cannot be
    matched by position, and NotImplementedError for a node of more than two ends that includes a
    module without CellML (an FV1D vessel or an api provider), whose coupling is one-to-one.
    '''
    owner = node.owner
    owner_variables = owner.port['variables']
    entries = list_multi_port(owner.port)
    if len(node.endpoints) > 2:
        others = [e.module for e in node.endpoints if module_formats.get(e.module) != 'cellml']
        if others:
            raise NotImplementedError(
                f'The node joining {node.description} sums over more than two modules, and '
                f'{", ".join(others)} has no CellML. Coupling a 1D or external module through a '
                f'node is only supported when it meets exactly one other module there.')
    sum_terms, shared = [], []
    for end in node.endpoints:
        if end is owner:
            continue
        if module_formats.get(end.module) != 'cellml':
            # coupled one-to-one (the only case left here): the coupling supplies the owner's
            # variables, so there is nothing to pair; report the owner's sum variables only
            sum_terms += [(v, end.module, None, node.sign(end))
                          for v, entry in zip(owner_variables, entries) if entry == MULTI_PORT_SUM]
            continue
        end_variables = end.port['variables']
        if len(end_variables) != len(owner_variables):
            raise ValueError(
                f'Cannot join "{end.module}" {list(end_variables)} to "{owner.module}" '
                f'{list(owner_variables)} at the node joining {node.description}: their vessel_ports '
                f'need the same number of variables, which are matched by position.')
        for owner_variable, end_variable, entry in zip(owner_variables, end_variables, entries):
            if entry == MULTI_PORT_SUM:
                sum_terms.append((owner_variable, end.module, end_variable, node.sign(end)))
            elif entry == MULTI_PORT_SHARED or len(node.endpoints) == 2:
                shared.append((owner_variable, end.module, end_variable))
            else:
                raise ValueError(
                    f'"{owner.module}" sums over the node joining {node.description}, but its '
                    f'vessel_port variable "{owner_variable}" is neither "sum" nor "True" in its '
                    f'multi_port, so it is not clear which of the other modules it connects to.')
    return sum_terms, shared


def owned_nodes(module_df):
    '''The nodes a list-form "sum" vessel_port owns, excluding any with a "Multiply" port.

    Raises ValueError for a node of three or more ends where none has a multi_port at all:
    nothing there could set the pressure or sum the flows. (A node of legacy "True" ports is left
    to the pairwise matching, which handles it by module type.)
    '''
    nodes = find_nodes(module_df)
    for node in nodes:
        if node.owner is None and len(node.endpoints) > 2 and \
                not any('multi_port' in e.port for e in node.endpoints):
            raise ValueError(
                f'{len(node.endpoints)} modules meet at the node joining {node.description}, but none '
                f'of their vessel_ports sums over it, so nothing there sets the pressure. One of them '
                f'needs a compliant end at that node (a port with "multi_port": ["sum", "True"], e.g. '
                f'BC_type "v" on that side).')
    return [node for node in nodes
            if node.owner is not None and not any(is_multiply_port(e.port) for e in node.endpoints)]
