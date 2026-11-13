"""Template for your own derivation functions (libcuflynx.derivations).

Copy this file beside your ``<name>_derivations.json``, rename it, and name a function in a
derivation::

    {"name": "Q_T0",
     "inputs": {"CO": {"input_item": "CO_Rowell1986"}, "frac": {"parameter": "Systemic_T/frac"}},
     "outputs": {"Q": "Systemic_T/Q_T0"},
     "python": "my_derivation_funcs.py:flow_share"}

A function is called ``f(inputs, units)``: ``inputs`` is ``{symbol: value}`` in each source's
units, ``units`` is ``{symbol: units}`` for the inputs and outputs. It returns
``{output symbol: value}`` in the output parameters' units, converting units itself.

Prefer a CellML derivation model (``"cellml": ...``) where the calculation is explicit: its units
are checked and its equation can be shown on a review page. Use a function for what CellML cannot
state, such as an implicit equation or a network solve. Every top-level function of the file is
registered; private names (leading ``_``) are not.
"""


def flow_share(inputs, units):
    """The flow through a bed: its share of the cardiac output."""
    return {'Q': inputs['CO'] * inputs['frac']}
