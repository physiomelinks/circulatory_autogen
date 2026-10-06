"""Helpers for the ``multi_port`` port setting.

A port in a module_config.json entry may carry ``multi_port``. Its values are
case-insensitive (``"sum"``, ``"Sum"`` and ``"SUM"`` are the same):

* ``"True"`` -- the port accepts any number of connections; this module's variables are mapped
  to the corresponding variables of every connected module (one source, many sinks).
* ``"sum"`` on a port with ``port_type`` ``"volume_port"`` -- the legacy blood-volume sum: the
  port's (single) variable is computed in the ``sum_blood_volume`` component.
* ``"sum"`` on any other port -- the port must map exactly one variable, which is the sum, over
  every module connected through the port, of the neighbour's corresponding variable. It is
  read as the list form ``["sum"]`` (below). This is PhLynx's ``"Sum"``.
* ``"Multiply"`` -- the port must map exactly one variable. On the upstream side of a
  connection (an exit or general port of the module listing the neighbour in its
  out_vessels), each connected module's corresponding variable is set to
  ``multiply_factor`` (an optional number on the port, default 1) times this module's
  variable; if the neighbour's port is a "sum", the scaled value is one term of its sum. On
  the downstream side of a connection a Multiply port is mapped like ``"True"``. This is
  PhLynx's ``"Multiply"``, whose factor PhLynx keeps in its UI rather than the config.

The list form gives one entry per port variable, aligned with the port's ``variables``::

    {"port_type": "vessel_port", "variables": ["v_in", "u"], "multi_port": ["sum", "True"]}

* ``"sum"`` -- this module's variable (an input) is the sum, over every module connected
  through this port, of the neighbour's corresponding port variable.
* ``"True"`` -- this module's variable is mapped to the corresponding variable of every
  connected neighbour (normally an output: one source, many sinks).

A list-form port accepts any number of neighbours, on entrance, exit or general ports.
Only one side of a connection can sum.

``normalise_port_multi_port`` puts a port's multi_port in canonical form when the module
config is loaded (utilities/config_schemas.py), so the generator only sees ``"True"``,
``"sum"`` (volume ports), ``"Multiply"`` and list forms of ``"sum"``/``"True"``.
"""

import numbers

MULTI_PORT_SUM = 'sum'
MULTI_PORT_SHARED = 'True'
MULTI_PORT_MULTIPLY = 'Multiply'

# the port_type whose whole-port "sum" is the legacy sum_blood_volume sum
LEGACY_SUM_PORT_TYPE = 'volume_port'

def _canonical_entry(entry):
    """A list-form entry as "sum"/"True" (case-insensitively), or unchanged if unknown."""
    if isinstance(entry, str) and entry.lower() == 'sum':
        return MULTI_PORT_SUM
    if entry is True or entry == 'True':
        return MULTI_PORT_SHARED
    return entry


def list_multi_port(port):
    """The normalised list-form multi_port of ``port``, or None if it has no list form."""
    if not isinstance(port, dict):
        return None
    value = port.get('multi_port')
    if not isinstance(value, (list, tuple)):
        return None
    return [_canonical_entry(entry) for entry in value]


def is_multiply_port(port):
    """Whether ``port`` has a (canonical) "Multiply" multi_port."""
    return isinstance(port, dict) and port.get('multi_port') == MULTI_PORT_MULTIPLY


def multiply_factor(port):
    """The ``multiply_factor`` of a Multiply port, as a float (default 1)."""
    return float(port.get('multiply_factor', 1))


def normalise_port_multi_port(port, module_description):
    """``port`` with its multi_port in canonical form (a new dict if anything changes).

    * a string "sum" in any case: "sum" on a volume_port (the legacy blood-volume sum),
      ``["sum"]`` on any other port, which must then have exactly one variable;
    * a string "multiply" in any case: "Multiply", on a port with exactly one variable, and
      a numeric ``multiply_factor`` if one is given;
    * list-form entries: "sum" in any case becomes "sum";
    * anything else is left as it is.

    Raises ValueError for a Sum/Multiply port that does not have exactly one variable, a
    ``multiply_factor`` that is not a number, or a ``multiply_factor`` on a port that is not
    Multiply.
    """
    if not isinstance(port, dict) or 'multi_port' not in port and 'multiply_factor' not in port:
        return port
    value = port.get('multi_port')
    port_type = port.get('port_type')
    variables = port.get('variables', [])
    if value is None or value is False or (isinstance(value, str) and value.lower() == 'false'):
        # no multi_port (null, false or "False")
        port = {k: v for k, v in port.items() if k != 'multi_port'}
        if 'multiply_factor' in port:
            raise ValueError(f'{module_description}: port "{port_type}" has a multiply_factor but no '
                             f'multi_port; multiply_factor only applies to "Multiply" ports.')
        return port
    if value is True or (isinstance(value, str) and value.lower() == 'true'):
        if value != 'True':
            value = 'True'
            port = dict(port, multi_port=value)
    elif not (isinstance(value, (list, tuple))
              or (isinstance(value, str) and value.lower() in ('sum', 'multiply'))):
        raise ValueError(
            f'{module_description}: port "{port_type}" has multi_port {value!r}. It must be '
            f'"True", "Sum" or "Multiply" (any case), true/false, or a list with one entry per '
            f'port variable.')

    def _single_variable(kind):
        if not isinstance(variables, (list, tuple)) or len(variables) != 1:
            raise ValueError(
                f'{module_description}: port "{port_type}" has multi_port "{value}" but '
                f'variables {variables}. A "{kind}" port must map exactly one variable.')

    new_value = value
    if isinstance(value, str) and value.lower() == 'sum':
        if port_type == LEGACY_SUM_PORT_TYPE:
            new_value = MULTI_PORT_SUM
        else:
            _single_variable('Sum')
            new_value = [MULTI_PORT_SUM]
    elif isinstance(value, str) and value.lower() == 'multiply':
        _single_variable('Multiply')
        new_value = MULTI_PORT_MULTIPLY
    elif isinstance(value, (list, tuple)):
        new_value = [_canonical_entry(entry) for entry in value]

    if 'multiply_factor' in port:
        if new_value != MULTI_PORT_MULTIPLY:
            raise ValueError(
                f'{module_description}: port "{port_type}" has a multiply_factor but its '
                f'multi_port is {value!r}; multiply_factor only applies to "Multiply" ports.')
        factor = port['multiply_factor']
        if isinstance(factor, bool) or not isinstance(factor, (numbers.Real, str)):
            raise ValueError(f'{module_description}: port "{port_type}" has multiply_factor '
                             f'{factor!r}, which is not a number.')
        try:
            float(factor)
        except ValueError:
            raise ValueError(f'{module_description}: port "{port_type}" has multiply_factor '
                             f'{factor!r}, which is not a number.') from None

    if new_value == value and type(new_value) is type(value):
        return port
    port = dict(port)
    port['multi_port'] = new_value
    return port


def is_multi_port(port):
    """Whether ``port`` accepts connections from several modules ("True", "Multiply" or list form)."""
    if not isinstance(port, dict):
        return False
    value = port.get('multi_port')
    return value in ['True', True, MULTI_PORT_MULTIPLY] or isinstance(value, (list, tuple))


def validate_list_multi_port(port, module_description):
    """Raise ValueError if a list-form multi_port on ``port`` is malformed.

    Ports without a list-form multi_port are not checked here.
    """
    value = port.get('multi_port') if isinstance(port, dict) else None
    if not isinstance(value, (list, tuple)):
        return
    variables = port.get('variables', [])
    if len(value) != len(variables):
        raise ValueError(
            f'{module_description}: port "{port.get("port_type")}" has a list-form multi_port '
            f'{list(value)} of length {len(value)}, but {len(variables)} variables '
            f'{list(variables)}. The list must have one entry per port variable.')
    for entry, variable in zip(value, variables):
        if _canonical_entry(entry) not in (MULTI_PORT_SUM, MULTI_PORT_SHARED):
            raise ValueError(
                f'{module_description}: port "{port.get("port_type")}" has multi_port entry '
                f'{entry!r} for variable "{variable}". List-form entries must be "sum" or "True".')


def module_has_list_multi_port(module_row):
    """Whether any entrance, exit or general port of a module row has a list-form multi_port
    or a Multiply multi_port (both are written as generated components)."""
    for key in ('entrance_ports', 'exit_ports', 'general_ports'):
        ports = module_row[key] if key in module_row else []
        if isinstance(ports, (list, tuple)):
            for port in ports:
                if list_multi_port(port) is not None or is_multiply_port(port):
                    return True
    return False
