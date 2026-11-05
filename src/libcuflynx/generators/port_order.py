"""Which of a module's ports a neighbour connects to.

A module's ports of one ``port_type`` are taken by its neighbours in the order its module
array record lists them: the n-th of its ``out_instances`` that has a port of that type
takes the n-th exit port of the type, and the n-th of its ``inp_instances`` that has one
takes the n-th entrance port. A heart with entrance ports ``(v_ivc, u_ra)``,
``(v_svc, u_ra)`` and ``(v_pvn, u_la)``, all ``vessel_port``, connects its first vessel
input to ``v_ivc``, its second to ``v_svc`` and its third to ``v_pvn``, whatever those
inputs are called and wherever their rows are in the array. To leave a port of the type
unconnected while a later one is connected, list a module that sets it (such as
``zero_flow``) in its place. An unconnected port's boundary_condition variables are
parameters, listed in the generator's missing-parameters report.

Ports with ``multi_port`` take any number of neighbours and are not counted.
"""

from libcuflynx.generators.multi_port import is_multi_port


def single_ports_of_type(ports, port_type):
    """Indices into ``ports`` of the ports of ``port_type`` that take one connection."""
    return [i for i, port in enumerate(ports or [])
            if port.get('port_type') == port_type and not is_multi_port(port)]


def neighbour_port_index(ports, neighbours, neighbour, port_type, neighbour_port_types):
    """The index into ``ports`` (a module's entrance or exit ports) of the port of
    ``port_type`` that ``neighbour`` connects to, or None when that is not decided by order:
    the module has fewer than two single ports of the type, ``neighbour`` has no port of the
    type, or more neighbours have the type than the module has ports.

    ``neighbours`` is the module's inp_instances (for entrance ports) or out_instances (exit
    ports), in order; ``neighbour_port_types(name)`` the port types of that neighbour's ports
    facing the module (its exit and general ports for an input, entrance and general ports
    for an output).
    """
    candidates = single_ports_of_type(ports, port_type)
    if len(candidates) < 2:
        return None
    ordered = [n for n in neighbours if port_type in neighbour_port_types(n)]
    if neighbour not in ordered:
        return None
    position = ordered.index(neighbour)
    if position >= len(candidates):
        return None
    return candidates[position]
