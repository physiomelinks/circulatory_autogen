"""Shipped derivation functions (the ``python`` method of a parameter derivation,
libcuflynx.derivations).

A derivation function calculates parameters from experimental inputs when a CellML derivation
model cannot state the calculation (an implicit equation, a network solve)::

    def f(inputs, units):
        '''inputs: {symbol: value}; units: {symbol: units} of the inputs and outputs.'''
        return {'R': (inputs['MAP'] - inputs['CVP']) / inputs['Q']}

It returns ``{output symbol: value}`` in the output parameters' units. A derivation names one as
``"python": "file.py:function"`` (the file relative to the derivations file) or, for a function
here, by its bare name.

This module is *library* code (issue #433): do not add your own functions here, because an
upgrade of libcuflynx replaces the file. Put them in your own file beside the derivations file;
see ``funcs_user/derivation_funcs_example.py``. It currently ships no functions; it exists as
the first tier of the registry (``derivations.python_method.build_derivation_funcs_dict``).
Every top-level function defined here, except private names, is registered.
"""
