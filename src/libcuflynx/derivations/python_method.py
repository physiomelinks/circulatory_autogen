'''
The ``python`` method of a derivation: a Python function, for a derivation a CellML model cannot
state (an implicit equation, a network solve).

A derivation names ``file.py:function``, the file relative to the derivations file, or the bare
name of a function libcuflynx ships (``libcuflynx.funcs.derivation_funcs_user``). The registry is
built as the operation funcs' is (``param_id.operation_funcs.build_operation_funcs_dict``): the
shipped functions, then every top-level function of the external file, which may override them.

A function is called ``f(inputs, units)``: ``inputs`` maps each input symbol to its value in its
source's units, ``units`` maps every input and output symbol to its units. It returns
``{output symbol: value}`` in the output parameters' units.
'''

import math
import os


class DerivationFunctionError(ValueError):
    '''A derivation function that cannot be found, or returns something other than its
    outputs.'''


def build_derivation_funcs_dict(external_path=None):
    '''``{name: function}``: the functions libcuflynx ships, then (if given) every top-level
    function of the file ``external_path``.'''
    from libcuflynx.funcs import derivation_funcs_user as shipped
    from libcuflynx.param_id.external_funcs import register_funcs_from_file
    from libcuflynx.param_id.math_backend import make_math_backend
    registry = {name: obj for name, obj in vars(shipped).items()
                if not name.startswith('_') and callable(obj) and not isinstance(obj, type)
                and getattr(obj, '__module__', None) == shipped.__name__}
    if external_path:
        register_funcs_from_file(external_path, registry, make_math_backend('numpy'))
    return registry


def split_python_spec(spec, base_dir):
    '''``(file or None, function name)`` of a derivation's ``python`` value, the file made
    absolute against ``base_dir``.'''
    spec = spec.strip()
    if spec.endswith('.py') or ':' not in spec:
        if spec.endswith('.py'):
            raise DerivationFunctionError(f"python '{spec}' names a file but no function; write "
                                          f"'{spec}:<function>'.")
        return None, spec
    file, name = spec.rsplit(':', 1)
    return os.path.normpath(os.path.join(base_dir, file)), name.strip()


def derivation_function(spec, base_dir):
    '''The function a derivation's ``python`` value names.'''
    file, name = split_python_spec(spec, base_dir)
    registry = build_derivation_funcs_dict(file)
    if name not in registry:
        where = file or 'libcuflynx.funcs.derivation_funcs_user'
        raise DerivationFunctionError(f"python '{spec}': {where} has no function '{name}' "
                                      f"(it has {sorted(registry) or 'none'}).")
    return registry[name]


def evaluate_python(spec, base_dir, inputs, units, outputs):
    '''Call the function ``spec`` names with ``inputs`` (``{symbol: value}``) and ``units``
    (``{symbol: units}`` of inputs and outputs). Returns ``{symbol: value}`` for exactly the
    symbols in ``outputs``.'''
    result = derivation_function(spec, base_dir)(dict(inputs), dict(units))
    if not isinstance(result, dict):
        raise DerivationFunctionError(f"python '{spec}' returned {type(result).__name__}, not "
                                      f"a dict of its outputs.")
    missing, extra = set(outputs) - set(result), set(result) - set(outputs)
    if missing or extra:
        raise DerivationFunctionError(
            f"python '{spec}' returned {sorted(result)}; its outputs are {sorted(outputs)}"
            + (f' (missing {sorted(missing)})' if missing else '')
            + (f' (not outputs: {sorted(extra)})' if extra else '') + '.')
    values = {}
    for symbol in outputs:
        try:
            values[symbol] = float(result[symbol])
        except (TypeError, ValueError):
            raise DerivationFunctionError(f"python '{spec}': output '{symbol}' is "
                                          f"{result[symbol]!r}, not a number.") from None
        if not math.isfinite(values[symbol]):
            raise DerivationFunctionError(f"python '{spec}': output '{symbol}' is "
                                          f"{values[symbol]}.")
    return values
