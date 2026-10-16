'''
How a parameter calibrated in one workflow step is named in another model.

Every step model is generated alone, as a single module-array record named ``STEP_VESSEL``
(``mod``). A params_for_id row ``(vessel_name, param_name)`` of a step therefore names
``param_name_for_gen(vessel_name, param_name)`` in that step's model: ``rho_M_mod`` for a
plain module, ``g_leak_mod__i_leak_Na`` for a supermodule's submodule, ``R`` for a global.

When that module is the submodule at ``path`` (``i_M``, or ``soma__i_M`` two levels down) of
a supermodule that is itself generated as ``mod``, supermodule expansion
(``utilities/supermodules.py``) renames its vessel ``mod[__x]`` to ``mod__<path>[__x]``, so the
same parameter is ``param_name_for_gen('mod_<path>[_x]', param_name)`` there. In the
supermodule's own parameterisation parameters CSV it is written ``<path>/<var>``
(``i_M/rho_M``, ``soma/i_M/rho_M`` when nested), as ``utilities/parameter_names.py`` reads it.

These are the rules circulatory-autogen-modules' ``calibrate apply`` follows by hand.
'''

from libcuflynx.parsers.PrimitiveParsers import param_name_for_gen
from libcuflynx.utilities.parameter_names import LEVEL_SEPARATOR, flat_name

STEP_VESSEL = 'mod'
GLOBAL_VESSEL = 'global'


def is_global(vessel):
    return vessel == GLOBAL_VESSEL


def map_vessel(vessel, path, step_vessel=STEP_VESSEL):
    '''``vessel`` of a model generated as ``step_vessel``, renamed for the model in which
    that module is the submodule at ``path`` ('' for the same model).

    ``('mod', 'i_M') -> 'mod__i_M'``; ``('mod__A', 'soma') -> 'mod__soma__A'``; a global is
    unchanged. A vessel that does not belong to ``step_vessel`` is a ValueError.
    '''
    if is_global(vessel) or not path:
        return vessel
    if vessel == step_vessel:
        return flat_name((step_vessel, path))
    prefix = step_vessel + LEVEL_SEPARATOR
    if vessel.startswith(prefix):
        return flat_name((step_vessel, path, vessel[len(prefix):]))
    raise ValueError(f'vessel "{vessel}" is not part of the step model, whose single record '
                     f'is "{step_vessel}"; params_for_id vessel_name must be "{step_vessel}", '
                     f'"{step_vessel}{LEVEL_SEPARATOR}<submodule>" or "{GLOBAL_VESSEL}".')


def model_name(vessel, param):
    '''The name of ``param`` of ``vessel`` in a generated (flat) model.'''
    return param_name_for_gen(vessel, param)


def host_row_name(vessel, param):
    '''``param`` of ``vessel`` as a host parameters file names it: ``mod__A/p``, or ``p`` for
    a global.'''
    return param if is_global(vessel) else f'{vessel}/{param}'


def parameterisation_name(vessel, param, submodule_paths=(), step_vessel=STEP_VESSEL):
    '''The name of ``param`` of ``vessel`` in the parameterisation parameters CSV of the module
    generated as ``step_vessel``, whose submodules are ``submodule_paths`` (tuples, as
    ``supermodules.submodule_path_parts`` gives them): ``('mod', 'p') -> 'p'``,
    ``('mod__A', 'p') -> 'A/p'``, ``('mod__soma__i_M', 'p') -> 'soma/i_M/p'``; a global
    unchanged.'''
    if is_global(vessel) or vessel == step_vessel:
        return param
    prefix = step_vessel + LEVEL_SEPARATOR
    if vessel.startswith(prefix):
        joined = vessel[len(prefix):]
        for parts in submodule_paths:
            if flat_name(parts) == joined:
                return '/'.join(tuple(parts) + (param,))
    raise ValueError(f'vessel "{vessel}" is not a submodule of the model generated as '
                     f'"{step_vessel}" (its submodules: '
                     f'{sorted("/".join(p) for p in submodule_paths)}).')


def relative_path(inner, outer):
    '''The submodule path of a module at ``inner`` relative to the module at ``outer``, both
    paths in the same model ('' is that model itself); None when ``inner`` is not inside
    ``outer``.'''
    if not outer:
        return inner
    if inner == outer:
        return ''
    if inner.startswith(outer + LEVEL_SEPARATOR):
        return inner[len(outer) + len(LEVEL_SEPARATOR):]
    return None


def host_model_name(name):
    '''The model name of a row of a module array's parameters file: ``aortic_root/C`` and
    ``C_aortic_root`` are ``C_aortic_root``; ``global/R`` and ``R`` are ``R``.'''
    from libcuflynx.utilities.parameter_names import model_name as path_model_name
    return path_model_name(name)


class VesselMap:
    '''How the vessels of a step's model are named in the workflow's target model, and back.

    ``to_target(vessel)`` names a vessel of the step's model in the target. ``from_target(vessel)``
    names a target vessel in the step's model, or None when the step's model has no such vessel
    (``vessels`` lists the ones it has). A global (``global``) is the same everywhere.'''

    def __init__(self, vessels):
        self.vessels = set(vessels)

    def to_target(self, vessel):
        return vessel if is_global(vessel) else self._to(vessel)

    def from_target(self, vessel):
        if is_global(vessel):
            return vessel
        name = self._from(vessel)
        return name if name in self.vessels else None


class Nested(VesselMap):
    '''A library module generated as ``mod``, inside a library supermodule target (also
    generated as ``mod``) at the submodule ``path`` (``'i_M'``, ``'soma__i_M'``; ``''`` for
    the target module itself).'''

    def __init__(self, path, vessels):
        super().__init__(vessels)
        self.path = path

    def _to(self, vessel):
        return map_vessel(vessel, self.path)

    def _from(self, vessel):
        if not self.path:
            return vessel
        root = flat_name((STEP_VESSEL, self.path))
        if vessel == root:
            return STEP_VESSEL
        if vessel.startswith(root + LEVEL_SEPARATOR):
            return flat_name((STEP_VESSEL, vessel[len(root) + len(LEVEL_SEPARATOR):]))
        return None


class Placed(VesselMap):
    '''A library module generated as ``mod``, that is the record ``vessel`` of a module-array
    target: ``mod`` is ``vessel`` and ``mod__<sub>`` is ``vessel__<sub>``.'''

    def __init__(self, vessel, vessels):
        super().__init__(vessels)
        self.vessel = vessel

    def _to(self, vessel):
        if vessel == STEP_VESSEL:
            return self.vessel
        if vessel.startswith(STEP_VESSEL + LEVEL_SEPARATOR):
            return flat_name((self.vessel, vessel[len(STEP_VESSEL) + len(LEVEL_SEPARATOR):]))
        raise ValueError(f'vessel "{vessel}" is not part of the step model, whose single record '
                         f'is "{STEP_VESSEL}".')

    def _from(self, vessel):
        if vessel == self.vessel:
            return STEP_VESSEL
        if vessel.startswith(self.vessel + LEVEL_SEPARATOR):
            return flat_name((STEP_VESSEL, vessel[len(self.vessel) + len(LEVEL_SEPARATOR):]))
        return None


class Renamed(VesselMap):
    '''A module array's own vessels: named the same in the target unless ``mapping`` (its
    vessel -> the target's) renames them.'''

    def __init__(self, mapping, vessels):
        super().__init__(vessels)
        self.mapping = dict(mapping)
        self.inverse = {v: k for k, v in self.mapping.items()}

    def _to(self, vessel):
        return self.mapping.get(vessel, vessel)

    def _from(self, vessel):
        if vessel in self.inverse:
            return self.inverse[vessel]
        return None if vessel in self.mapping else vessel


def between(source, receiver, vessel):
    '''``vessel`` of the step whose map is ``source``, named in the step whose map is
    ``receiver`` (through the target's naming); None when the receiving model has no such
    vessel.'''
    return receiver.from_target(source.to_target(vessel))
