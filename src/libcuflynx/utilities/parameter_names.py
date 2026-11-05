'''
How parameter names are written in the files a user writes, and the names models use.

A generated model names a module's constant ``{var}_{vessel}`` (``C_aortic_root``) and a global
constant ``{var}`` (``R``). Variable and vessel names both contain ``_``, so such a name cannot
be split back into the two reliably: ``g_leak_Na`` could be ``g_leak`` of ``Na`` or ``g`` of
``leak_Na``. Files people write therefore name a parameter with ``/`` between the vessel and
the variable -- the separator obs_data operands and params_for_id targets already use:

======================================  ==============================  =======================
written                                 where                           model name
======================================  ==============================  =======================
``aortic_root/C``                       host parameters file            ``C_aortic_root``
``heart/lv/E``                          host file, a supermodule's      ``E_heart__lv``
                                        expanded submodule
``global/R`` or ``R``                   any                             ``R``
``lv/E``                                a supermodule's parameter set   ``E_<instance>__lv``
                                                                        (record ``<instance>``)
``soma/i_M/rho_M``                      same, a nested submodule        ``rho_M_<instance>__soma__i_M``
======================================  ==============================  =======================

An expanded submodule's vessel joins the levels of its path with ``__`` (``LEVEL_SEPARATOR``):
``heart/lv`` is the vessel ``heart__lv``. A module-array or submodule name may not contain
``__``, nor begin or end with ``_`` (``check_instance_name``), so a flat name splits back into
its path one way only.

The model names cannot contain ``/`` (or ``.``), because CellML names cannot: only what people
write uses it. The older forms -- ``C_aortic_root`` in a host file, ``E_lv`` in a supermodule's
rows, and nested levels joined with a single ``_`` (``E_heart_lv``) -- are refused, naming
``cuflynx-upgrade parameter-names``, which rewrites them.
'''

from libcuflynx.upgrade import COMMAND as UPGRADE_COMMAND

SEPARATOR = '/'
GLOBAL = 'global'
MIGRATE_COMMAND = f'{UPGRADE_COMMAND} parameter-names'
#: joins the levels of a nested instance's path in its flat (vessel) name: heart/lv -> heart__lv
LEVEL_SEPARATOR = '__'

#: Where another generator may have hoisted the constant ``<comp>/<var>``, as
#: ``(component, variable name template)``, in the order they are tried. CA's own convention
#: (``parameters/{var}_{comp}``, globals in ``parameters_global``) is tried before these.
#: PhLynx before v0.3.0 used ``parameters``/``parameters_global``; from v0.3.0 on it uses
#: ``instance_parameters``/``global_parameters``. A constant only one instance has keeps its
#: bare name; one shared by several instances is prefixed with the instance.
HOISTED_PARAMETER_CONVENTIONS = (
    ('parameters', '{comp}_{var}'),
    ('parameters', '{var}'),
    ('parameters_global', '{var}'),
    ('instance_parameters', '{comp}_{var}'),
    ('instance_parameters', '{var}'),
    ('global_parameters', '{var}'),
)

#: Components that hold a model's parameters, and those of them that hold global constants.
PARAMETER_COMPONENTS = tuple(dict.fromkeys(c for c, _ in HOISTED_PARAMETER_CONVENTIONS))
GLOBAL_PARAMETER_COMPONENTS = tuple(
    c for c in PARAMETER_COMPONENTS if c.endswith('_global') or c.startswith('global_'))


def hoisted_names(component, variable, sep=SEPARATOR):
    '''The names ``<component>/<variable>`` may have after hoisting, in
    HOISTED_PARAMETER_CONVENTIONS order, without repeats. A component ending ``_module`` is
    tried with and without the suffix.'''
    bare = component[:-len('_module')] if component.endswith('_module') else component
    names = []
    for target, template in HOISTED_PARAMETER_CONVENTIONS:
        for comp in (component, bare):
            name = f'{target}{sep}{template.format(comp=comp, var=variable)}'
            if name not in names:
                names.append(name)
    return names


def flat_name(parts):
    '''The vessel name of the path ``parts``: ``('heart', 'lv')`` -> ``heart__lv``.'''
    return LEVEL_SEPARATOR.join(parts)


def check_instance_name(name, where=''):
    '''ValueError unless ``name`` (a module-array or submodule record's) can be a level of a flat
    name -- no ``__``, and no ``_`` at either end -- and a key of a parameters file: not one of
    ``parameters_file.RESERVED_NAMES`` (``global``, ``value``, ``schema_version``).'''
    from libcuflynx.utilities.parameters_file import RESERVED_NAMES
    name = str(name)
    if name in RESERVED_NAMES:
        raise ValueError(
            f'{where}: name "{name}" cannot be used. A parameters file nests values by instance, '
            f'and {", ".join(RESERVED_NAMES)} are reserved there ("global" holds the global '
            f'constants, "value" a parameter\'s value).')
    if LEVEL_SEPARATOR in name or name.startswith('_') or name.endswith('_'):
        raise ValueError(
            f'{where}: name "{name}" cannot be used. A nested instance\'s vessel joins the levels '
            f'of its path with "{LEVEL_SEPARATOR}" (heart/lv is heart{LEVEL_SEPARATOR}lv), so a '
            f'name may not contain "{LEVEL_SEPARATOR}" nor begin or end with "_".')
    return name


def output_model_name(name):
    '''The model's name ``<vessel>/<variable>`` of an output written with its vessel's path:
    ``SN/soma/membrane/V`` -> ``SN__soma__membrane/V``. The path form is the one to write; a
    name with one ``/`` (``aortic_root/u``, or the flat ``SN__soma__membrane/V``) is returned
    unchanged.'''
    name = str(name).strip()
    path, _, variable = name.rpartition(SEPARATOR)
    if SEPARATOR not in path:
        return name
    return f'{flat_name(p.strip() for p in path.split(SEPARATOR))}{SEPARATOR}{variable.strip()}'


def is_path_name(name):
    '''Whether ``name`` is written in the ``/`` form.'''
    return SEPARATOR in str(name)


def split_path_name(name, where=''):
    '''``'heart/lv/E'`` -> ``(['heart', 'lv'], 'E')``; ``'global/R'`` -> ``([], 'R')``.
    ValueError for an empty part.'''
    parts = [p.strip() for p in str(name).split(SEPARATOR)]
    if len(parts) < 2 or any(not p for p in parts):
        raise ValueError(f'{where}parameter "{name}": a "/" name is <vessel>/<variable>, with '
                         f'nothing empty on either side of a "/".')
    *path, variable = parts
    if path == [GLOBAL]:
        return [], variable
    return path, variable


def model_name(name, where=''):
    '''The model's name for a parameter as written in a host parameters file:
    ``aortic_root/C`` -> ``C_aortic_root``, ``heart/lv/E`` -> ``E_heart__lv``,
    ``global/R`` -> ``R``; a name without ``/`` is already one and is returned unchanged.'''
    if not is_path_name(name):
        return name
    path, variable = split_path_name(name, where)
    return f'{variable}_{flat_name(path)}' if path else variable


def supermodule_row_name(name, instance, submodule_paths, where=''):
    '''The model name of a row of a supermodule's parameterisation or default_parameters
    file, for the supermodule record ``instance``. ``submodule_paths`` are the supermodule's
    submodules as tuples of names (``('lv',)``, and ``('soma', 'i_M')`` for one nested in a
    submodule).

    ``lv/E`` -> ``E_<instance>__lv``; ``soma/i_M/rho_M`` -> ``rho_M_<instance>__soma__i_M``
    (the path must be one of ``submodule_paths``); ``global/R`` and ``R`` -> ``R``. A name in
    the older ``E_lv`` form (ending ``_<submodule path>``) is a ValueError naming the converter.
    '''
    paths = [tuple(p) for p in submodule_paths]
    if is_path_name(name):
        path, variable = split_path_name(name, where)
        if not path:
            return variable
        if tuple(path) not in paths:
            known = sorted('/'.join(p) for p in paths)
            raise ValueError(
                f'{where}parameter "{name}": "{"/".join(path)}" is not a submodule of this '
                f'supermodule (its submodules: {known}; a nested one is written '
                f'<submodule>/<its submodule>).')
        return f'{variable}_{flat_name((instance,) + tuple(path))}'
    for path in paths:
        for joined in {'_'.join(path), flat_name(path)}:
            if name.endswith('_' + joined) and len(name) > len(joined) + 1:
                raise ValueError(
                    f'{where}parameter "{name}" is written {{variable}}_{{submodule}}; libcuflynx '
                    f'reads <submodule>/<variable> ({"/".join(path)}/{name[:-len(joined) - 1]}). '
                    f'Convert the file once with: {MIGRATE_COMMAND} --supermodule-config '
                    f'<its modules_config.json>')
    return name


def host_model_names(names, vessels, where, check_globals=True):
    '''The model names of a host parameters file's ``names``, checked against the model.

    ``vessels`` is ``{vessel name: {variable name: kind}}`` for the model's (expanded) module
    array, ``kind`` as in a module config's ``variables_and_units`` (``constant``,
    ``global_constant`` ...). A ``/`` name whose vessel is not in the model, whose variable its
    module does not have, or that addresses a global constant through a vessel would not be
    used, nor would a global's name (``R``, ``global/R``) that is no module's global constant
    (checked unless ``check_globals`` is false): all of them are reported in one ValueError. One written ``{variable}_{vessel}``
    (``C_aortic_root``), as before, is a ValueError naming the converter.
    '''
    model_names, seen, unused, old = [], {}, [], []
    globals_ = {var for variables in vessels.values() for var, kind in variables.items()
                if kind == 'global_constant'}
    for name in names:
        name = str(name)
        path = vessel = None
        if is_path_name(name):
            path, variable = split_path_name(name, where=f'{where}: ')
            if path:
                vessel = flat_name(path)
                translated = f'{variable}_{vessel}'
            else:
                translated = variable
        else:
            translated = name
            if any(name.endswith('_' + joined)
                   and variables.get(name[:-len(joined) - 1]) not in (None, 'global_constant')
                   for v, variables in vessels.items()
                   for joined in {v, v.replace(LEVEL_SEPARATOR, '_')}):
                old.append(name)
        if translated in seen and seen[translated] != name:
            raise ValueError(f'{where}: "{seen[translated]}" and "{name}" both set parameter '
                             f'{translated}; keep one.')
        seen.setdefault(translated, name)
        model_names.append(translated)
        if not path:
            if check_globals and name not in old and translated not in globals_:
                unused.append(f'{name} (no module in the model has a global constant "{translated}")')
            continue
        if vessel not in vessels:
            unused.append(f'{name} (no vessel "{vessel}" in the module array)')
        elif variable not in vessels[vessel]:
            unused.append(f'{name} (the module of "{vessel}" has no variable "{variable}")')
        elif vessels[vessel][variable] == 'global_constant':
            unused.append(f'{name} ("{variable}" is a global constant: write global/{variable} '
                          f'or {variable})')
    if old:
        raise ValueError(f'{where}: {len(old)} parameter(s) are written {{variable}}_{{vessel}} '
                         f'(e.g. "{old[0]}"); libcuflynx reads <vessel>/<variable>. Convert the '
                         f'file once with: {MIGRATE_COMMAND} --module-array <its module array> '
                         f'--module-library-dir <library>')
    if unused:
        raise ValueError(f'{where}: {len(unused)} parameter(s) name something the model does '
                         f'not have: ' + '; '.join(unused))
    return model_names
