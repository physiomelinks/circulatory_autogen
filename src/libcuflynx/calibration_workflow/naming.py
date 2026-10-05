'''
How a parameter calibrated in one workflow step is named in another model.

Every step model is generated alone, as a single vessel-array record named ``STEP_VESSEL``
(``mod``). A params_for_id row ``(vessel_name, param_name)`` of a step therefore names
``param_name_for_gen(vessel_name, param_name)`` in that step's model: ``rho_M_mod`` for a
plain module, ``g_leak_mod_i_leak_Na`` for a supermodule's submodule, ``R`` for a global.

When that module is the submodule at ``path`` (``i_M``, or ``soma_i_M`` two levels down) of
a supermodule that is itself generated as ``mod``, supermodule expansion
(``utilities/supermodules.py``) renames its vessel ``mod[_x]`` to ``mod_<path>[_x]``, so the
same parameter is ``param_name_for_gen('mod_<path>[_x]', param_name)`` there. In the
supermodule's own instance parameters CSV the ``mod_`` prefix is absent:
``{var}_{path}[_x]``.

These are the rules circulatory-autogen-modules' ``calibrate apply`` follows by hand.
'''

from libcuflynx.parsers.PrimitiveParsers import param_name_for_gen

STEP_VESSEL = 'mod'
GLOBAL_VESSEL = 'global'


def is_global(vessel):
    return vessel == GLOBAL_VESSEL


def map_vessel(vessel, path, step_vessel=STEP_VESSEL):
    '''``vessel`` of a model generated as ``step_vessel``, renamed for the model in which
    that module is the submodule at ``path`` ('' for the same model).

    ``('mod', 'i_M') -> 'mod_i_M'``; ``('mod_A', 'soma') -> 'mod_soma_A'``; a global is
    unchanged. A vessel that does not belong to ``step_vessel`` is a ValueError.
    '''
    if is_global(vessel) or not path:
        return vessel
    if vessel == step_vessel:
        return f'{step_vessel}_{path}'
    prefix = step_vessel + '_'
    if vessel.startswith(prefix):
        return f'{step_vessel}_{path}_{vessel[len(prefix):]}'
    raise ValueError(f'vessel "{vessel}" is not part of the step model, whose single record '
                     f'is "{step_vessel}"; params_for_id vessel_name must be "{step_vessel}", '
                     f'"{step_vessel}_<submodule>" or "{GLOBAL_VESSEL}".')


def model_name(vessel, param):
    '''The name of ``param`` of ``vessel`` in a generated (flat) model.'''
    return param_name_for_gen(vessel, param)


def instance_name(vessel, param, step_vessel=STEP_VESSEL):
    '''The name of ``param`` of ``vessel`` in the instance parameters CSV of the module
    generated as ``step_vessel``: ``('mod', 'p') -> 'p'``, ``('mod_A', 'p') -> 'p_A'``,
    a global unchanged.'''
    if is_global(vessel) or vessel == step_vessel:
        return param
    prefix = step_vessel + '_'
    if vessel.startswith(prefix):
        return f'{param}_{vessel[len(prefix):]}'
    raise ValueError(f'vessel "{vessel}" is not part of a model generated as "{step_vessel}".')


def relative_path(inner, outer):
    '''The submodule path of a module at ``inner`` relative to the module at ``outer``, both
    paths in the same model ('' is that model itself); None when ``inner`` is not inside
    ``outer``.'''
    if not outer:
        return inner
    if inner == outer:
        return ''
    if inner.startswith(outer + '_'):
        return inner[len(outer) + 1:]
    return None
