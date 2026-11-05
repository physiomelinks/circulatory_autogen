"""The built-in pulmonary_GE modules' arterial concentrations C = max((v_p C_p + v_sh C_v) v /
(v^2 + v_eps^2), 0): defined at v = 0 (where num/v is 0/0, the capillary flow starting from rest)
and equal to num/v away from 0, for both signs of v. The guarded form uses only products, sums and
1/D, so a smooth copy of the model lifts exactly to quadratic form."""
import math
import os
import xml.etree.ElementTree as ET

import pytest

import libcuflynx

M = '{http://www.w3.org/1998/Math/MathML}'
C11 = '{http://www.cellml.org/cellml/1.1#}'
MODULES = os.path.join(os.path.dirname(libcuflynx.__file__), 'generators', 'resources', 'gas_exchange_modules.cellml')


def _eval(e, env):
    tag = e.tag.split('}')[-1]
    if tag == 'ci':
        return env[e.text.strip()]
    if tag == 'cn':
        return float(e.text)
    op, args = e[0].tag.split('}')[-1], [_eval(a, env) for a in list(e)[1:]]
    if op == 'plus':
        return sum(args)
    if op == 'minus':
        return args[0] - args[1] if len(args) == 2 else -args[0]
    if op == 'times':
        return math.prod(args)
    if op == 'divide':
        return args[0] / args[1]
    if op == 'power':
        return args[0] ** args[1]
    if op == 'max':
        return max(args)
    raise ValueError(op)


def _equation(component, target):
    for eq in component.iter(M + 'apply'):
        args = list(eq)
        if args and args[0].tag == M + 'eq' and args[1].tag == M + 'ci' and args[1].text.strip() == target:
            return args[2]
    raise KeyError(target)


@pytest.mark.parametrize('name', ['pulmonary_GE_type', 'pulmonary_GE_5_lobe_type'])
def test_arterial_concentrations_are_defined_at_zero_flow_and_unchanged_elsewhere(name):
    component = next(c for c in ET.parse(MODULES).getroot().iter(C11 + 'component') if c.get('name') == name)
    v_eps = float(next(v for v in component.iter(C11 + 'variable') if v.get('name') == 'v_eps').get('initial_value'))
    assert v_eps == 1e-10
    sf, c_p, c_v = 0.02, 0.2, 0.15
    for target, (p, ven) in {'C_O2_a': ('C_O2_p', 'C_O2_v'), 'C_CO2_a': ('C_CO2_p', 'C_CO2_v')}.items():
        rhs = _equation(component, target)
        for v in (5.7e-5, 2.2e-4, -3.9e-5, 1e-7):
            env = {'v': v, 'v_eps': v_eps, 'v_p': v * (1 - sf), 'v_sh': v * sf, p: c_p, ven: c_v}
            assert _eval(rhs, env) == pytest.approx(max((1 - sf) * c_p + sf * c_v, 0), rel=1e-5)
        env = {'v': 0.0, 'v_eps': v_eps, 'v_p': 0.0, 'v_sh': 0.0, p: c_p, ven: c_v}
        assert _eval(rhs, env) == 0.0
