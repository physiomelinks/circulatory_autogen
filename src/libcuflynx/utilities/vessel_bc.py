"""Which modules carry a vessel boundary-condition convention in their ``BC_type``.

A vessel module's ``BC_type`` (``module_subtype``) starts with two letters naming what it
takes at its inlet and outlet: ``v`` (flow, it is given a flow) or ``p`` (pressure), e.g.
``vp`` or ``pv_micro``. The generator checks that a vessel ending in ``v`` feeds one
starting with ``p`` and so on, and picks junction and terminal mappings from those letters.

Those two letters only mean something for vessels. Any other module (a cell, an ion
channel, a controller, a heart part, a boundary condition such as ``nn_constant``) can have
any ``BC_type``, so names such as ``lv_test``, ``rv``, ``SN_soma`` or ``cardiomyocyte``
must not be read as a BC pair. ``nn`` (no BC) needs no special case: it is not a BC pair.

A module counts as a vessel when both hold:

* its ``BC_type`` starts with one of ``vv``, ``vp``, ``pv``, ``pp``; and
* its ports are those of a vessel: a ``vessel_port`` among its entrance, exit or general
  ports, or both a ``flow_port`` and a ``pressure_port`` among its entrance/exit ports
  (the microvasculature network modules). When a row carries no port information
  (the raw module array, before the module config is merged in) the prefix alone decides.
"""

VESSEL_BC_PREFIXES = ('vv', 'vp', 'pv', 'pp')

_PORT_KEYS = ('entrance_ports', 'exit_ports', 'general_ports')


def is_vessel_bc(BC_type):
    """True when ``BC_type`` starts with a vessel BC pair (``vv``, ``vp``, ``pv``, ``pp``)."""
    return isinstance(BC_type, str) and BC_type[:2] in VESSEL_BC_PREFIXES


def _get(row, key):
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[key]          # pandas Series
    except (KeyError, TypeError, IndexError):
        return getattr(row, key, None)  # namedtuple from itertuples()


def _port_types(ports):
    if not isinstance(ports, (list, tuple)):
        return []
    return [port.get('port_type') for port in ports if isinstance(port, dict)]


def has_vessel_ports(row):
    """True when ``row`` has the ports of a vessel; None when it carries no port info."""
    port_lists = {key: _get(row, key) for key in _PORT_KEYS}
    if all(not isinstance(ports, (list, tuple)) for ports in port_lists.values()):
        return None
    all_types = [t for ports in port_lists.values() for t in _port_types(ports)]
    if 'vessel_port' in all_types:
        return True
    end_types = _port_types(port_lists['entrance_ports']) + _port_types(port_lists['exit_ports'])
    return 'flow_port' in end_types and 'pressure_port' in end_types


def is_vessel_module(row):
    """True when the module in ``row`` (a dict, pandas Series or itertuples() row of a
    module/vessel dataframe) is a vessel whose ``BC_type`` carries a BC pair."""
    if not is_vessel_bc(_get(row, 'BC_type')):
        return False
    ports = has_vessel_ports(row)
    return True if ports is None else ports
