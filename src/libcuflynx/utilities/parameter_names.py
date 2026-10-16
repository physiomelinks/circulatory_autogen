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
                                        or default_parameters           (record ``<instance>``)
``soma/i_M/rho_M``                      same, a nested submodule        ``rho_M_<instance>__soma__i_M``
======================================  ==============================  =======================

An expanded submodule's vessel joins the levels of its path with ``__`` (``LEVEL_SEPARATOR``):
``heart/lv`` is the vessel ``heart__lv``. A module-array or submodule name may not contain
``__``, nor begin or end with ``_`` (``check_instance_name``), so a flat name splits back into
its path one way only. Names joined with a single ``_``, as before #582 (``heart_lv``,
``E_heart_lv``), are still read with a FutureWarning when they name one vessel; when they could
name more than one they are an error (``legacy_flat_name``).

The model names cannot contain ``/`` (or ``.``), because CellML names cannot: only what people
write uses it. The older forms are still read: ``C_aortic_root`` in a host file (unambiguous
there, as it is looked up rather than split), and ``E_lv`` in a supermodule's rows, which has to
be split -- longest submodule suffix first, the only reading available -- and is warned about.
``cuflynx-migrate-parameter-names`` rewrites old names in either kind of file.
'''

import warnings

SEPARATOR = '/'
GLOBAL = 'global'
MIGRATE_COMMAND = 'cuflynx-migrate-parameter-names'
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
    name: no ``__``, and no ``_`` at either end.'''
    name = str(name)
    if LEVEL_SEPARATOR in name or name.startswith('_') or name.endswith('_'):
        raise ValueError(
            f'{where}: name "{name}" cannot be used. A nested instance\'s vessel joins the levels '
            f'of its path with "{LEVEL_SEPARATOR}" (heart/lv is heart{LEVEL_SEPARATOR}lv), so a '
            f'name may not contain "{LEVEL_SEPARATOR}" nor begin or end with "_".')
    return name


def legacy_flat_name(name, known, where=''):
    '''``name`` as one of the ``known`` vessel names, reading the single-``_`` joins used
    before #582 (``heart_lv`` for ``heart__lv``).

    A known name is returned as it is. A name that is one known name with each ``__`` written
    ``_`` is returned as that name, with a FutureWarning; one that could be several is a
    ValueError. Anything else is returned unchanged, for the caller to report.'''
    known = set(known)
    if name in known:
        return name
    matches = sorted(k for k in known
                     if LEVEL_SEPARATOR in k and k.replace(LEVEL_SEPARATOR, '_') == name)
    if len(matches) > 1:
        raise ValueError(f'{where}: "{name}" could be any of {matches}; nested levels are '
                         f'joined with "{LEVEL_SEPARATOR}" now, so write one of those.')
    if matches:
        warnings.warn(f'{where}: "{name}" joins nested levels with "_"; write "{matches[0]}" '
                      f'(levels are joined with "{LEVEL_SEPARATOR}").', FutureWarning,
                      stacklevel=2)
        return matches[0]
    return name


def legacy_model_name(name, vessels, where=''):
    '''A parameter's model name ``{var}_{vessel}`` whose vessel is written with the
    single-``_`` joins used before #582 (``E_heart_lv``), as the current name
    (``E_heart__lv``) with a FutureWarning. ``vessels`` are the model's vessel names; a name
    that could be read for two of them is a ValueError, and one that is not in the older form
    is returned unchanged.'''
    matches = sorted({f'{name[:-len(old) - 1]}_{v}' for v in vessels if LEVEL_SEPARATOR in v
                      for old in [v.replace(LEVEL_SEPARATOR, '_')]
                      if name.endswith('_' + old) and len(name) > len(old) + 1})
    if len(matches) > 1:
        raise ValueError(f'{where}: parameter "{name}" could be any of {matches}; nested levels '
                         f'are joined with "{LEVEL_SEPARATOR}" now, so write one of those.')
    if matches:
        warnings.warn(f'{where}: parameter "{name}" joins nested levels with "_"; write '
                      f'"{matches[0]}" (levels are joined with "{LEVEL_SEPARATOR}").',
                      FutureWarning, stacklevel=2)
        return matches[0]
    return name


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
    file, for the supermodule record ``instance``. ``submodule_paths`` are the supermodule's submodules as
    tuples of names (``('lv',)``, and ``('soma', 'i_M')`` for one nested in a submodule).
    Returns ``(model name, old_form)``.

    ``lv/E`` -> ``E_<instance>__lv``; ``soma/i_M/rho_M`` -> ``rho_M_<instance>__soma__i_M``
    (the path must be one of ``submodule_paths``); ``global/R`` and ``R`` -> ``R``. A name in
    the older ``E_lv`` form ends with ``_<submodule path>`` (levels joined with ``_`` or
    ``__``): it is split at the longest such suffix and ``old_form`` is True.
    '''
    paths = [tuple(p) for p in submodule_paths]
    if is_path_name(name):
        path, variable = split_path_name(name, where)
        if not path:
            return variable, False
        if tuple(path) not in paths:
            known = sorted('/'.join(p) for p in paths)
            raise ValueError(
                f'{where}parameter "{name}": "{"/".join(path)}" is not a submodule of this '
                f'supermodule (its submodules: {known}; a nested one is written '
                f'<submodule>/<its submodule>).')
        return f'{variable}_{flat_name((instance,) + tuple(path))}', False
    joins = sorted(((joined, p) for p in paths
                    for joined in {'_'.join(p), flat_name(p)}), key=lambda j: -len(j[0]))
    for joined, path in joins:
        suffix = '_' + joined
        if name.endswith(suffix) and len(name) > len(suffix):
            return f'{name[:-len(suffix)]}_{flat_name((instance,) + path)}', True
    return name, False


def warn_old_supermodule_names(old_names, where):
    '''One FutureWarning for a file whose rows use the older ``{var}_{submodule}`` form.'''
    if not old_names:
        return
    example = old_names[0]
    warnings.warn(
        f'{where}: {len(old_names)} parameter name(s) are written {{variable}}_{{submodule}} '
        f'(e.g. "{example}"), which has to be guessed apart where a variable or submodule name '
        f'contains "_". Write <submodule>/<variable> instead (nested: '
        f'<submodule>/<its submodule>/<variable>); {MIGRATE_COMMAND} --supermodule-config '
        f'<config> rewrites the file.', FutureWarning, stacklevel=3)


def host_model_names(names, vessels, where):
    '''The model names of a host parameters file's ``names``, checked against the model.

    ``vessels`` is ``{vessel name: {variable name: kind}}`` for the model's (expanded) module
    array, ``kind`` as in a module config's ``variables_and_units`` (``constant``,
    ``global_constant`` ...). A ``/`` name whose vessel is not in the model, whose variable its
    module does not have, or that addresses a global constant through a vessel would not be
    used: all of them are reported in one ValueError. So are two rows naming one parameter
    (``aortic_root/C`` and ``C_aortic_root``). A vessel written with the single-``_`` joins used
    before #582 (``heart_lv/E``, ``E_heart_lv``) is read as the current one, with a
    FutureWarning (``legacy_flat_name``, ``legacy_model_name``).
    '''
    model_names, seen, unused = [], {}, []
    for name in names:
        name = str(name)
        path = vessel = None
        if is_path_name(name):
            path, variable = split_path_name(name, where=f'{where}: ')
            if path:
                vessel = legacy_flat_name(flat_name(path), vessels, where)
                translated = f'{variable}_{vessel}'
            else:
                translated = variable
        elif any(name.endswith('_' + v) and name[:-len(v) - 1] in variables
                 for v, variables in vessels.items()):
            translated = name
        else:
            translated = legacy_model_name(name, vessels, where)
        if translated in seen and seen[translated] != name:
            raise ValueError(f'{where}: "{seen[translated]}" and "{name}" both set parameter '
                             f'{translated}; keep one.')
        seen.setdefault(translated, name)
        model_names.append(translated)
        if not path:
            continue
        if vessel not in vessels:
            unused.append(f'{name} (no vessel "{vessel}" in the module array)')
        elif variable not in vessels[vessel]:
            unused.append(f'{name} (the module of "{vessel}" has no variable "{variable}")')
        elif vessels[vessel][variable] == 'global_constant':
            unused.append(f'{name} ("{variable}" is a global constant: write global/{variable} '
                          f'or {variable})')
    if unused:
        raise ValueError(f'{where}: {len(unused)} parameter(s) name something the model does '
                         f'not have: ' + '; '.join(unused))
    return model_names
