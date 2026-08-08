"""Pairing a params_for_id row's names with its value(s).

A ``params_for_id`` row may name several vessels, which means "one calibrated value drives all
of these" -- the grouped-parameter feature. Every backend's ``set_param_vals`` therefore receives
entries that are either a single name or a list of names, against a single shared value.

The obvious pairing, ``zip(names, vals)``, is wrong for exactly that case: with two names and one
value, ``zip`` stops at the shorter sequence and silently sets only the first name. That is what
made a grouped row calibrate its first vessel and leave the rest at their defaults, on every
backend, with no error and a cost curve identical to the ungrouped one.

``pair_names_with_values`` is the single place that decides the pairing, so the broadcast rule
cannot drift between backends, and a genuine length mismatch raises instead of truncating.
"""


def as_name_list(name_or_list):
    """Normalise a params_for_id entry to a list of names."""
    if isinstance(name_or_list, (list, tuple)):
        return list(name_or_list)
    return [name_or_list]


def as_value_list(value_or_list):
    """Normalise a value entry to a list, without splitting strings (a protocol trace key)."""
    if isinstance(value_or_list, str):
        return [value_or_list]
    if isinstance(value_or_list, (list, tuple)):
        return list(value_or_list)
    if hasattr(value_or_list, 'tolist') and getattr(value_or_list, 'ndim', 1) > 0:
        return list(value_or_list.tolist())
    return [value_or_list]


def pair_names_with_values(name_or_list, value_or_list, context=''):
    """Yield ``(name, value)`` for one params_for_id entry, broadcasting a shared value.

    - N names, 1 value  -> the value is applied to all N. This is the grouped-parameter case
      and the reason this helper exists.
    - N names, N values -> paired positionally.
    - anything else     -> ValueError, rather than the silent truncation ``zip`` would do.
    """
    names = as_name_list(name_or_list)
    values = as_value_list(value_or_list)

    if len(values) == 1 and len(names) > 1:
        values = values * len(names)

    if len(names) != len(values):
        where = f' ({context})' if context else ''
        raise ValueError(
            f'params_for_id entry{where} has {len(names)} name(s) {names} but {len(values)} '
            f'value(s). A grouped entry takes one shared value for all its names, or one value '
            f'per name.')

    return list(zip(names, values))
