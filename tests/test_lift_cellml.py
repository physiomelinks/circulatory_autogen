"""lift_cellml: the quadratic lifting of a smooth flat CellML model, for Operator Inference.

The models are written by hand, so each test runs in seconds. Each lifting is checked the same way: the lifted
model passes libCellML's Validator and Analyser, simulates with CVODE_myokit as the original does over a short
run, stays on the manifold w = T(s), and J(s) f(s) equals the lifted right-hand side at sampled states.
"""
import json

import numpy as np
import pytest

sp = pytest.importorskip('sympy')
pytest.importorskip('qbee')

from libcuflynx.utilities import lift_cellml as lc  # noqa: E402
from libcuflynx.utilities import smooth_cellml as sc  # noqa: E402

C2 = 'http://www.cellml.org/cellml/2.0#'
MML = 'http://www.w3.org/1998/Math/MathML'


def _model(rhs, init, params, alg=None, extra_math=''):
    """A one-component CellML 2.0 model; rhs {state: sympy expr}, alg {name: sympy expr}."""
    vs = ['<variable name="t" units="dimensionless"/>']
    vs += [f'<variable name="{p}" units="dimensionless" initial_value="{v}"/>' for p, v in params.items()]
    vs += [f'<variable name="{s}" units="dimensionless" initial_value="{v}"/>' for s, v in init.items()]
    vs += [f'<variable name="{a}" units="dimensionless"/>' for a in (alg or {})]
    eqs = [f'<apply><eq/><ci>{a}</ci>{e if isinstance(e, str) else lc._mml(e)}</apply>' for a, e in (alg or {}).items()]
    eqs += [f'<apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>{s}</ci></apply>{lc._mml(e)}</apply>'
            for s, e in rhs.items()]
    return (f'<?xml version="1.0"?>\n<model xmlns="{C2}" name="m"><component name="c">' + ''.join(vs) +
            f'<math xmlns="{MML}" xmlns:cellml="{C2}">' + ''.join(eqs) + extra_math + '</math></component></model>')


def _lift(tmp_path, text, sim_time=2.0, **kw):
    path = tmp_path / 'm.cellml'
    path.write_text(text)
    kw.setdefault('time_budget', 5.0)
    doc = lc.lift_cellml_file(str(path), str(tmp_path / 'out'), sim_time=sim_time, dt=0.01, **kw)
    assert doc['libcellml'] == {'validator': [], 'analyser': []}
    v = doc['verification']
    assert v['jf']['max'] < 1e-9, v['jf']
    assert doc['check']['max_relative_residual'] < 1e-9, doc['check']
    # solver accuracy (rtol 1e-10), amplified along a manifold that need not attract
    assert v['max_state_error'] < 1e-5, v['states']
    assert v['max_manifold_defect'] < 1e-5, v['manifold']
    return doc


x, y, z, k, K, u = sp.symbols('x y z k K u')


def test_fitzhugh_nagumo_lifts_with_w_equal_to_v_squared(tmp_path):
    """Lift & Learn eq. 59-60: the cubic term of FitzHugh-Nagumo needs only w = V^2."""
    V, R, a, b, c = sp.symbols('V R a b c')
    doc = _lift(tmp_path, _model({V: c * (V - V ** 3 / 3 + R), R: -(V - a + b * R) / c}, {V: -1, R: 1},
                                 {'a': 0.2, 'b': 0.2, 'c': 3}))
    assert doc['lifted'][:2] == ['V', 'R']
    assert [a['expr'] for a in doc['auxiliaries']] == ['V**2']
    assert doc['auxiliaries'][0]['jac'] == ['2*V', '0']
    assert doc['operators']['present'] == {'A': True, 'H': True, 'N': False, 'B': False, 'C': True}
    # dV/dt = c V + c R - (c/3) V w: the V w pair of H
    H = doc['operators']['symbolic']['H']
    assert [0, 0, 2, 2, '-c/3'] in H
    # the defect's growth rate off the manifold is reported (it is positive: w = V^2 does not attract)
    assert doc['verification']['defect_growth']['defect']


@pytest.mark.parametrize('name, rhs, init, params, expected_aux', [
    ('exp', {x: -k * sp.exp(x) + 1}, {x: 0.5}, {'k': 2}, ['exp(x)']),
    ('ln', {x: -sp.log(x)}, {x: 2.0}, {}, None),
    ('power', {x: 1 - x ** sp.Rational(3, 2)}, {x: 0.2}, {}, None),
    ('irrational_power', {x: 1 - x ** sp.Float(0.3836)}, {x: 0.5}, {}, ['x**0.3836', '1/x']),
    ('hill', {x: 1 - x - K * x ** 2 / (K + x ** 2)}, {x: 0.5}, {'K': 0.5}, None),
    ('sqrt', {x: 2 - sp.sqrt(x ** 2 + K)}, {x: 0.0}, {'K': 1}, None),
    ('reciprocal', {x: 1 / (1 + x ** 2) - x}, {x: 0.3}, {}, ['1/(x**2 + 1)']),
    ('product3', {x: -y, y: x - x * y * z, z: -z}, {x: 1, y: 0, z: 0.5}, {}, None),
])
def test_each_kind_lifts_to_quadratic_form_and_verifies(tmp_path, name, rhs, init, params, expected_aux):
    doc = _lift(tmp_path, _model(rhs, init, params))
    if expected_aux is not None:
        assert set(expected_aux) <= {a['expr'] for a in doc['auxiliaries']}
    assert all(a['exact'] for a in doc['auxiliaries'])
    assert doc['size']['n_w'] == len(doc['lifted'])


def test_rules_engine_gives_a_lifting_that_verifies_when_qbee_is_not_used(tmp_path):
    doc = _lift(tmp_path, _model({x: -sp.log(x) + sp.sqrt(x)}, {x: 2.0}, {}), engine='rules')
    assert doc['engine']['polynomialize'] == 'rules' and doc['engine']['quadratize'] == 'greedy'


def test_a_qbee_failure_falls_back_and_is_reported(tmp_path):
    """QBee's polynomialize fails on sqrt (sympy 1.14); the rule table takes over and says so."""
    doc = _lift(tmp_path, _model({x: 2 - sp.sqrt(x ** 2 + 1)}, {x: 0.0}, {}))
    assert doc['engine']['polynomialize'] == 'rules'
    assert any(e.startswith('polynomialize') for e in doc['engine']['qbee_errors'])


def test_a_named_input_enters_through_n_and_b(tmp_path):
    """dx/dt = -x^3 + u x + u: input-free with w = x^2; dw/dt = -2 w^2 + 2 u w + 2 u x (u w, u x: N terms)."""
    doc = _lift(tmp_path, _model({x: -x ** 3 + u * x + u}, {x: 0.1}, {'u': 0.3}), inputs=['u'])
    assert [i['name'] for i in doc['inputs']] == ['u']
    assert doc['engine']['input_free']
    present = doc['operators']['present']
    assert present['N'] and present['B'] and present['H'] and not present['A']
    assert [a['expr'] for a in doc['auxiliaries']] == ['x**2']
    inp = [i for i in doc['structure']['inputs'] if i['name'] == 'u'][0]
    assert inp['kind'] == 'held' and inp['multiplies_state']


def test_an_input_times_a_square_has_no_input_affine_lift_and_the_search_says_so(tmp_path):
    """u x^2 needs x^3, whose derivative needs u x^4, and so on: no finite lift with only N w u terms."""
    path = tmp_path / 'm.cellml'
    path.write_text(_model({x: -x + u * x ** 2}, {x: 0.1}, {'u': 0.3}))
    with pytest.raises(lc.LiftError, match='degree > 2 left'):
        lc.lift(str(path), inputs=['u'], engine='rules', max_aux=12)


def test_a_polynomial_system_over_max_terms_stops_with_its_size():
    """(1 + x + y)^30 has 496 terms: over max_terms, the lift stops before any quadratization and says why."""
    with pytest.raises(lc.LiftError, match='too large to quadratize'):
        lc.lift(_model({x: -x + (1 + x + y) ** 30 * 1e-30, y: -y}, {x: 0.1, y: 0.1}, {}), engine='rules', max_terms=400)


def test_a_right_hand_side_that_stays_at_zero_is_judged_on_its_terms_not_on_rounding(tmp_path):
    """dz/dt = 10 (x - y/3)(x + y/3) with y = 3 x throughout is 0, but the operator form's terms 10 x^2 and
    -(10/9) y^2 are not, and round differently from the model's product: the checks are relative to the terms, so
    that rounding does not fail them (Burgers' middle node)."""
    path = tmp_path / 'm.cellml'
    path.write_text(_model({x: -x, y: -y, z: 10 * (x - y / 3) * (x + y / 3)}, {x: 0.1, y: 0.3, z: 0.0}, {}))
    doc = lc.lift_cellml_file(str(path), str(tmp_path / 'out'), sim_time=2.0, dt=0.01, time_budget=5.0)
    assert doc['check']['per_equation']['z'] < 1e-12
    assert doc['verification']['jf']['max'] < 1e-12


def test_auxiliaries_identically_constant_or_equal_are_removed():
    """QBee may add w = sqrt(g) * (1/sqrt(g)) = 1 or two monomials with one T: the lift drops them."""
    L = lc.lift(_model({x: 2 - sp.sqrt(x ** 2 + 1)}, {x: 0.0}, {}), time_budget=5.0)
    values = [lc._value(T, {sp.Symbol('x', real=True): 0.7}) for T in L.T]
    assert len(set(np.round(values, 12))) == len(values)
    assert all(T.free_symbols for T in L.T)


def test_a_time_input_inside_a_root_with_a_state_takes_its_derivative_as_an_input():
    """dx/dt = 1 - sqrt(1 + (E x)^2), E = 1 + sin(t)/2 (an elastance times a volume inside a valve law). The rule
    table stops at w = sqrt(.) and z = 1/w, with dE/dt an input (not input-free). The input-affine form has no finite
    lift of it (dw/dt has E dE/dt x^2 z), so the search stops at max_aux and says so."""
    tt = sp.Symbol('t')
    m = lc.Model(_model({x: 1 - sp.sqrt(1 + (sp.Symbol('E') * x) ** 2)}, {x: 0.5}, {}, alg={'E': 1 + sp.sin(tt) / 2}))
    p = lc._Polynomializer(m, max_aux=20)
    p.run()
    assert sorted(a.rule for a in p.aux) == ['reciprocal root', 'root']
    assert not p.input_free and [i['kind'] for i in m.inputs] == ['time', 'derivative']
    with pytest.raises(lc.LiftError, match='degree > 2 left'):
        lc.lift(_model({x: 1 - sp.sqrt(1 + (sp.Symbol('E') * x) ** 2)}, {x: 0.5}, {}, alg={'E': 1 + sp.sin(tt) / 2}),
                engine='rules', max_aux=20)


def test_a_function_of_time_becomes_an_input(tmp_path):
    doc = _lift(tmp_path, _model({x: -x + sp.sin(2 * sp.Symbol('t'))}, {x: 0.0}, {}))
    assert [(i['kind'], i['expr']) for i in doc['inputs']] == [('time', 'sin(2*t)')]
    assert doc['size']['n_aux'] == 0
    assert doc['structure']['inputs'][0]['kind'] == 'known function of time'


def test_a_smoothed_rectifier_lifts_with_its_eps_and_is_not_exact(tmp_path):
    """z' = max(x, 0) - z on a harmonic oscillator, smoothed by cuflynx-smooth, lifts exactly."""
    text = _model({x: -y, y: x}, {x: 1, y: 0}, {}, alg={'r': '<apply><max/><ci>x</ci><cn cellml:units="dimensionless">0</cn></apply>'},
                  extra_math='<apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>z</ci></apply>'
                             '<apply><minus/><ci>r</ci><ci>z</ci></apply></apply>')
    text = text.replace('<variable name="r"', '<variable name="z" units="dimensionless" initial_value="0"/><variable name="r"')
    smoothed, report = sc.smooth(text, {'sm1_x': {'min': -1.0, 'max': 1.0, 'min_abs': 0.0}}, rel=1e-2)
    assert report['sites'][0]['rule'] == 'rectifier'
    doc = _lift(tmp_path, smoothed, smoothing_report=report)
    assert not doc['engine']['qbee_errors'] or doc['engine']['polynomialize'] == 'rules'
    inexact = [a for a in doc['auxiliaries'] if not a['exact']]
    assert inexact and all(a['epsilon'] == {'eps_sm1': pytest.approx(0.02)} for a in inexact)
    assert doc['inputs'] == []
    assert doc['smoothing_sites'][0]['original_class'].startswith('C0')
    # the harmonic oscillator is a clock: x^2 + y^2 conserved at a constant rate
    clocks = doc['structure']['clocks']
    assert clocks and clocks[0]['rate_kind'] == 'constant'


def test_an_exempt_site_becomes_a_state_dependent_input(tmp_path):
    text = _model({x: -y, y: x}, {x: 1, y: 0}, {}, alg={'r': '<apply><max/><ci>x</ci><cn cellml:units="dimensionless">0</cn></apply>'},
                  extra_math='<apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>z</ci></apply>'
                             '<apply><minus/><ci>r</ci><ci>z</ci></apply></apply>')
    text = text.replace('<variable name="r"', '<variable name="z" units="dimensionless" initial_value="0"/><variable name="r"')
    smoothed, _ = sc.smooth(text, {'sm1_x': {'min': -1.0, 'max': 1.0, 'min_abs': 0.0}}, exempt=[{'variable': 'r', 'reason': 'test'}])
    doc = _lift(tmp_path, smoothed)
    assert [(i['kind'], i['switch_variables']) for i in doc['inputs']] == [('nonsmooth', ['x'])]
    assert doc['structure']['inputs'][0]['kind'] == 'state-dependent'


def test_block_triangular_pattern_and_a_conserved_functional(tmp_path):
    """y1, y2 rotate on their own and drive v through y1^2 + y1 y2 (test 4 with b = 0); x1 + x2 is conserved."""
    y1, y2, v, x1, x2 = sp.symbols('y1 y2 v x1 x2')
    doc = _lift(tmp_path, _model({y1: -0.1 * y1 - 2 * sp.pi * y2, y2: 2 * sp.pi * y1 - 0.1 * y2,
                                  v: -v + y1 ** 2 + y1 * y2, x1: -x1 + x2, x2: x1 - x2},
                                 {y1: 1, y2: 0, v: 0, x1: 1, x2: 0}, {}))
    assert doc['size']['n_aux'] == 0
    blocks = doc['structure']['blocks']
    assert blocks['triangular']
    vb = [i for i, b in enumerate(blocks['blocks']) if 'v' in b][0]
    assert any(c['to'] == vb and c['quadratic'] and not c['linear'] for c in blocks['coupling'])
    cons = [c for c in doc['structure']['conserved'] if set(c['l']) == {'x1', 'x2'}]
    assert cons and cons[0]['drift'] < 1e-8


def test_python_functions_have_the_documented_shapes(tmp_path):
    doc = _lift(tmp_path, _model({x: -k * sp.exp(x) + y, y: -y}, {x: 0.1, y: 1.0}, {'k': 1}))
    f = lc.load_functions(doc['python'])
    X = np.random.default_rng(0).uniform(0, 1, (7, 2))
    assert f.T(X).shape == (7, doc['size']['n_w'])
    assert f.J(X).shape == (7, doc['size']['n_w'], 2)
    assert f.rhs_lifted(f.T(X), np.zeros((7, 0))).shape == (7, doc['size']['n_w'])
    # the json is parsable as documented
    syms = {n: sp.Symbol(n) for n in list(doc['symbols']) + doc['lifted']}
    T = [sp.sympify(a['expr'], locals=syms) for a in doc['auxiliaries']]
    assert T == [sp.exp(sp.Symbol('x'))]
    json.dumps(doc)


def test_cli_writes_the_three_files(tmp_path):
    path = tmp_path / 'm.cellml'
    path.write_text(_model({x: -x ** 3}, {x: 1.0}, {}))
    assert lc.main([str(path), '--output-dir', str(tmp_path / 'o'), '--sim-time', '1']) == 0
    assert {p.name for p in (tmp_path / 'o').iterdir()} >= {'m_lifted.cellml', 'lifting.json', 'lifting_functions.py'}
