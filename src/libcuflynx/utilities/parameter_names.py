'''
How parameter names are written in the files a user writes, and the names models use.

A generated model names a module's constant ``{var}_{vessel}`` (``C_aortic_root``) and a global
constant ``{var}`` (``R``). Variable and vessel names both contain ``_``, so such a name cannot
be split back into the two reliably: ``g_leak_Na`` could be ``g_leak`` of ``Na`` or ``g`` of
``leak_Na``. Files people write therefore name a parameter with ``/`` between the vessel and
the variable -- the separator obs_data operands and params_for_id targets already use:

======================================  ==============================  =====================
written                                 where                           model name
======================================  ==============================  =====================
``aortic_root/C``                       host parameters file            ``C_aortic_root``
``heart/lv/E``                          host file, a supermodule's      ``E_heart_lv``
                                        expanded submodule
``global/R`` or ``R``                   any                             ``R``
``lv/E``                                a supermodule's instance or     ``E_<instance>_lv``
                                        default_parameters              (record ``<instance>``)
``soma/i_M/rho_M``                      same, a nested submodule        ``rho_M_<instance>_soma_i_M``
======================================  ==============================  =====================

The model names stay what they were, because a CellML name cannot contain ``/`` (or ``.``):
only what people write changes. The older forms are still read: ``C_aortic_root`` in a host
file (unambiguous there, as it is looked up rather than split), and ``E_lv`` in a supermodule's
rows, which has to be split -- longest submodule suffix first, the only reading available -- and
is warned about. ``cuflynx-migrate-parameter-names`` rewrites old names in either kind of file.
'''

import warnings

SEPARATOR = '/'
GLOBAL = 'global'
MIGRATE_COMMAND = 'cuflynx-migrate-parameter-names'


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
    ``aortic_root/C`` -> ``C_aortic_root``, ``heart/lv/E`` -> ``E_heart_lv``,
    ``global/R`` -> ``R``; a name without ``/`` is already one and is returned unchanged.'''
    if not is_path_name(name):
        return name
    path, variable = split_path_name(name, where)
    return f'{variable}_{"_".join(path)}' if path else variable


def supermodule_row_name(name, instance, submodule_paths, where=''):
    '''The model name of a row of a supermodule's instance or default_parameters file, for the
    supermodule record ``instance``. ``submodule_paths`` are the supermodule's submodules as
    tuples of names (``('lv',)``, and ``('soma', 'i_M')`` for one nested in a submodule).
    Returns ``(model name, old_form)``.

    ``lv/E`` -> ``E_<instance>_lv``; ``soma/i_M/rho_M`` -> ``rho_M_<instance>_soma_i_M`` (the
    path must be one of ``submodule_paths``); ``global/R`` and ``R`` -> ``R``. A name in the
    older ``E_lv`` form ends with ``_<submodule path>``: it is split at the longest such
    suffix and ``old_form`` is True.
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
        return f'{variable}_{instance}_{"_".join(path)}', False
    for joined in sorted(('_'.join(p) for p in paths), key=len, reverse=True):
        suffix = '_' + joined
        if name.endswith(suffix) and len(name) > len(suffix):
            return f'{name[:-len(suffix)]}_{instance}_{joined}', True
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
    module does not have, or that addresses a global constant through a vessel, is collected in
    one UserWarning -- such a row is not used, which used to happen without a word. Two rows
    naming one parameter (``aortic_root/C`` and ``C_aortic_root``) are a ValueError.
    '''
    model_names, seen, unused = [], {}, []
    for name in names:
        name = str(name)
        translated = model_name(name, where=f'{where}: ')
        if translated in seen and seen[translated] != name:
            raise ValueError(f'{where}: "{seen[translated]}" and "{name}" both set parameter '
                             f'{translated}; keep one.')
        seen.setdefault(translated, name)
        model_names.append(translated)
        if not is_path_name(name):
            continue
        path, variable = split_path_name(name, where=f'{where}: ')
        if not path:
            continue
        vessel = '_'.join(path)
        if vessel not in vessels:
            unused.append(f'{name} (no vessel "{vessel}" in the module array)')
        elif variable not in vessels[vessel]:
            unused.append(f'{name} (the module of "{vessel}" has no variable "{variable}")')
        elif vessels[vessel][variable] == 'global_constant':
            unused.append(f'{name} ("{variable}" is a global constant: write global/{variable} '
                          f'or {variable})')
    if unused:
        warnings.warn(f'{where}: {len(unused)} parameter(s) are not used by the model: '
                      + '; '.join(unused), UserWarning, stacklevel=2)
    return model_names
