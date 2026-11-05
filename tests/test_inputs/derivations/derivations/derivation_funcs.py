"""Derivation functions of the toy derivations (tests/test_derivations.py)."""


def flow_share(inputs, units):
    """The flow through a bed: its share of the cardiac output."""
    return {'Q': inputs['CO'] * inputs['frac']}
