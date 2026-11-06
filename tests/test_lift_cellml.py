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


def test_a_reciprocal_of_a_rational_expression_lifts_in_its_natural_form(tmp_path):
    """1/(1 + x/(K + x)) = (K + x)/(K + 2x): one reciprocal of a polynomial, not 1/(K + x) inside another reciprocal
    (the drug combiners 1/(1 + Delta_R))."""
    doc = _lift(tmp_path, _model({x: 1 - x, y: -y / (1 + x / (K + x))}, {x: 0.2, y: 1.0}, {'K': 0.5}), engine='rules')
    assert [a['expr'] for a in doc['auxiliaries']][0] == '1/(K + 2*x)'


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


def test_variables_equal_near_the_initial_state_are_not_merged_unless_equal_symbolically():
    """tanh(50 x) and tanh(60 x) are both -1 to rounding at x0 = -1 (saturated valves), but not elsewhere."""
    m = lc.Model(_model({x: -x}, {x: -1.0}, {}), myokit_order=False)
    xs = m.states[0]
    T = [xs, sp.tanh(50 * xs), sp.tanh(60 * xs)]
    names, T2, _, _, kept, sub = lc._reduce(m, m.states, ['x', 'a', 'b'], T, [None] * 3, [[], [], []])
    assert names == ['x', 'a', 'b'] and sub == {}


def test_the_complete_quadratization_closes_a_system_the_greedy_search_does_not():
    """With no inputs every polynomial system quadratizes with the monomials of degree up to d - 1; the search
    falls back to them (pruned) when the greedy one stops."""
    q = lc._Quadratic(2, 0, [lc._to_dict(-x ** 3 + x * y ** 2, [x, y]), lc._to_dict(-y ** 3 + x ** 2 * y, [x, y])])
    assert not q.greedy(0)
    assert q.complete(10) and not q.bad()
    assert 1 <= len(q.W) - 2 <= 3


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


def test_qbee_is_not_tried_on_a_system_over_its_size(monkeypatch):
    monkeypatch.setattr(lc, 'QBEE_MAX_TERMS', 0)
    L = lc.lift(_model({x: -x ** 3}, {x: 1.0}, {}), time_budget=5.0)
    assert L.engine['quadratize'] == 'greedy'
    assert any('not tried' in e for e in L.engine['qbee_errors'])


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


def test_a_site_whose_argument_is_a_sum_lifts_the_argument_too(tmp_path):
    """z' = |x - y| - z: the site's argument x - y is a lifted variable, so w' = x z x' has as many terms as x'."""
    text = _model({x: -y, y: x}, {x: 1, y: 0}, {}, alg={'r': '<apply><abs/><apply><minus/><ci>x</ci><ci>y</ci></apply></apply>'},
                  extra_math='<apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>z</ci></apply>'
                             '<apply><minus/><ci>r</ci><ci>z</ci></apply></apply>')
    text = text.replace('<variable name="r"', '<variable name="z" units="dimensionless" initial_value="0"/><variable name="r"')
    smoothed, report = sc.smooth(text, {'sm1_x': {'min': -1.5, 'max': 1.5, 'min_abs': 0.0}}, rel=1e-2)
    doc = _lift(tmp_path, smoothed)
    kinds = {s['name']: s['kind'] for s in doc['sites']}
    assert kinds.get('sm1_x') == 'x' and 'sm1_w' in kinds
    defs = {d['variable']: d['expr'] for d in doc['invariants'] if d['origin'] == 'definition' and 'expr' in d}
    assert sp.sympify(defs['sm1_x']) == sp.sympify('sm1_x - x + y')


def test_held_inputs_lift_an_input_inside_tanh_and_are_marked(tmp_path):
    """dx/dt = -x + tanh(k (u - x)) with u held (du/dt = 0 within a chunk) as a state: t = tanh(.),
    dt/dt = (1 - t^2) k (u' - x') with u' = 0; u is marked as a held input at its index in w."""
    k_ = sp.Symbol('k')
    text = _model({x: -x + sp.tanh(k_ * (u - x))}, {x: 0.1}, {'u': 0.5, 'k': 2})
    doc = _lift(tmp_path, text, inputs=['u'], held_inputs=True)
    assert doc['lifted'][:2] == ['x', 'u'] and doc['inputs'] == []
    assert doc['held_inputs'][0]['name'] == 'u' and doc['held_inputs'][0]['index'] == 1
    assert doc['reverse'] == 'w[:1]' and doc['size']['n_held'] == 1


def test_stabilize_adds_minus_kappa_times_each_quadratic_identity(tmp_path):
    """x' = 1 - x - x/(1 + x): w = 1/(1 + x), identity w (1 + x) - 1; with kappa the equation of w gets
    -kappa (w + w x - 1), which is zero on the manifold, so the checks still hold."""
    text = _model({x: 1 - x - x / (1 + x)}, {x: 0.2}, {})
    doc = _lift(tmp_path, text, engine='rules', stabilize=1e3)
    st = doc['stabilization']
    assert st['kappa'] == 1e3 and 'w_1' in [i['variable'] for i in st['identities']]
    i = doc['lifted'].index('w_1')
    assert doc['operators']['C'][i] == pytest.approx(1e3)
    assert doc['operators']['A'][i][i] <= -1e3 + 1e-9


def test_tanh_in_exp_form_has_an_identity_to_stabilize(tmp_path):
    doc = _lift(tmp_path, _model({x: -x + sp.tanh(2 * x)}, {x: 0.3}, {}), tanh_form='exp', stabilize=10.0)
    exprs = [a['expr'] for a in doc['auxiliaries']]
    assert 'exp(4*x)' in exprs and any(e.startswith('1/(exp(4*x) + 1)') for e in exprs)
    assert any(d['variable'] for d in doc['stabilization']['identities'])


def test_held_inputs_as_inputs_give_b_and_n_terms_without_products(tmp_path):
    """A dose rate d (a parameter driven as an input) into C' = d - C, an effect compartment Ce' = C - Ce, and an Emax
    output Ce/(K + Ce): held as an input, d enters through B (the C row) and N (d times auxiliaries), with no monomials
    of it."""
    C_, Ce_, d_, K_ = sp.symbols('C Ce d K')
    path = tmp_path / 'm.cellml'
    path.write_text(_model({C_: d_ - C_, Ce_: C_ - Ce_}, {C_: 0.0, Ce_: 0.0}, {'d': 0.5, 'K': 1.0},
                           alg={'E': Ce_ / (K_ + Ce_)}))
    doc = lc.lift_cellml_file(str(path), str(tmp_path / 'out'), inputs=['d'], held_inputs='inputs', outputs=['E'],
                              sim_time=2.0, dt=0.01, time_budget=5.0, engine='rules')
    assert [i['name'] for i in doc['inputs']] == ['d'] and doc['held_inputs'] == []
    assert doc['operators']['present']['B'] and doc['operators']['present']['N']
    assert doc['verification']['jf']['max'] < 1e-12


def test_spectrum_and_simulation_show_fitzhugh_nagumos_transverse_instability(tmp_path):
    """w = V^2 is not attracting: the leading eigenvalue at states on the manifold is positive somewhere along the run,
    and simulate_lifted reports the defect off the manifold (smaller with re-lifting)."""
    V, R, a, b, c = sp.symbols('V R a b c')
    doc = _lift(tmp_path, _model({V: c * (V - V ** 3 / 3 + R), R: -(V - a + b * R) / c}, {V: -1, R: 1},
                                 {'a': 0.2, 'b': 0.2, 'c': 3}))
    sp_ = doc['structure']['spectrum']
    assert sp_['n_points'] > 10 and sp_['leading_re_max'] > 0
    f = lc.load_functions(doc['python'])
    free = lc.simulate_lifted(f, [-1.0, 1.0], np.zeros((1, 0)), 10.0)
    relifted = lc.simulate_lifted(f, [-1.0, 1.0], np.zeros((1, 0)), 10.0, relift_every=0.5)
    assert free['success'] and relifted['success'] and relifted['max_defect'] < free['max_defect']
    cut_short = lc.simulate_lifted(f, [-1.0, 1.0], np.zeros((1, 0)), 1e6, max_wall=0.5)
    assert not cut_short['success'] and 'max_wall' in cut_short['message']


def test_keep_cuts_the_other_states_as_inputs_from_data(tmp_path):
    """Two components; keeping one makes the other's state an input whose values come from the model run."""
    text = f"""<?xml version="1.0"?><model xmlns="{C2}" name="m">
<component name="a"><variable name="t" units="dimensionless" interface="public_and_private"/>
<variable name="x" units="dimensionless" initial_value="1" interface="public_and_private"/>
<variable name="y" units="dimensionless" interface="public_and_private"/>
<math xmlns="{MML}"><apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>x</ci></apply>
<apply><minus/><apply><power/><ci>y</ci><cn xmlns:cellml="{C2}" cellml:units="dimensionless">3</cn></apply><ci>x</ci></apply></apply></math></component>
<component name="b"><variable name="t" units="dimensionless" interface="public_and_private"/>
<variable name="y" units="dimensionless" initial_value="0.5" interface="public_and_private"/>
<variable name="x" units="dimensionless" interface="public_and_private"/>
<math xmlns="{MML}"><apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>y</ci></apply><apply><minus/><ci>x</ci></apply></apply></math></component>
<connection component_1="a" component_2="b"><map_variables variable_1="t" variable_2="t"/>
<map_variables variable_1="x" variable_2="x"/><map_variables variable_1="y" variable_2="y"/></connection></model>"""
    path = tmp_path / 'm.cellml'
    path.write_text(text)
    doc = lc.lift_cellml_file(str(path), str(tmp_path / 'out'), keep=['a'], sim_time=1.0, dt=0.01, time_budget=5.0)
    assert doc['states'] == ['a.x'] and [(i['name'], i['kind']) for i in doc['inputs']] == [('y', 'cut')]
    assert not doc['verification']['simulated'] and doc['verification']['jf']['max'] < 1e-12
    assert doc['check']['max_relative_residual'] < 1e-12
    assert doc['operators']['present']['B']


def test_an_output_is_lifted_with_the_states(tmp_path):
    """x (linear PK) feeds no rate, but its Emax effect E = x/(1 + x) is an output: w = 1/(1 + x) is lifted, and
    dw/dt = -w^2 (1 - x) needs x w, which is E itself: E is one lifted variable."""
    path = tmp_path / 'm.cellml'
    path.write_text(_model({x: 1 - x}, {x: 0.0}, {}, alg={'E': x / (1 + x)}))
    doc = lc.lift_cellml_file(str(path), str(tmp_path / 'out'), outputs=['E'], sim_time=2.0, dt=0.01, time_budget=5.0)
    assert [a['expr'] for a in doc['auxiliaries']] == ['1/(x + 1)', 'x/(x + 1)']
    out = doc['outputs'][0]
    assert out['name'] == 'E_' and out['degree'] == 1 and out['expr'] == 'w_2' 
    assert doc['verification']['jf']['max'] < 1e-12
    assert lc.check_lifting_json(json.loads(json.dumps(doc, default=str))) == []


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
    # the site's w and z are lifted variables in closed form, with their identities
    assert doc['engine']['polynomialize'] == 'sites+rules'
    assert {'sm1_w', 'sm1_z'} <= set(doc['lifted']) and [s['kind'] for s in doc['sites']] == ['w', 'z']
    defs = {d['variable']: d for d in doc['invariants'] if d['origin'] == 'definition' and 'expr' in d}
    assert sp.sympify(defs['sm1_w']['expr']) == sp.sympify('sm1_w**2 - x**2 - eps_sm1**2')
    assert sp.sympify(defs['sm1_z']['expr']) == sp.sympify('sm1_w*sm1_z - 1')
    assert all(d['relative_reference'] < 1e-12 for d in defs.values())
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


def test_the_quadratic_heart_has_a_reflex_clock_a_drug_n_term_and_conserved_volume(tmp_path):
    """ICUHealthy's QHEART (CAM benchmarks/quadratic_heart): quadratic already, so no auxiliaries; the clock (c, s) is
    found with a rate that depends on V_a, the drug u enters through N, and V_lv + V_a + V_v is conserved."""
    lv, a, v, c, s_, u_ = sp.symbols('V_lv V_a V_v c s u')
    Plv = (sp.Float(0.06) + sp.Float(2.44) * (1 - c) / 2) * (lv - 10)
    Pa, Pv = (a - 700) / sp.Float(1.5), (v - 2400) / sp.Float(60)
    w = sp.Float(7.85) * (1 + sp.Float(0.3) * (Pa - 90) / 90)
    qin, qout, qs = (Pv - Plv) / sp.Float(0.05), (Plv - Pa) / sp.Float(0.05), (1 + u_) * (Pa - Pv)
    doc = _lift(tmp_path, _model({lv: qin - qout, a: qout - qs, v: qs - qin, c: -w * s_, s_: w * c},
                                 {lv: 120, a: 760, v: 2620, c: 1, s_: 0}, {'u': 0.1}), inputs=['u'])
    assert doc['size']['n_aux'] == 0 and doc['operators']['present']['N']
    clock = doc['structure']['clocks'][0]
    assert (clock['c'], clock['s'], clock['rate_kind']) == ('c', 's', 'depends on V_a')
    assert {'V_lv': 1.0, 'V_a': 1.0, 'V_v': 1.0} in [c['l'] for c in doc['structure']['conserved']]
    assert doc['structure']['inputs'][0]['multiplies_state']


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


def test_lifting_json_conforms_to_its_schema_and_the_docs_describe_every_field(tmp_path):
    """lifting.json is read by OpInf and by cuflynx-rom: its fields are only ever added, never renamed."""
    import pathlib
    doc = _lift(tmp_path, _model({x: -k * sp.exp(x) + y, y: -y}, {x: 0.1, y: 1.0}, {'k': 1}))
    assert doc['schema_version'] == lc.SCHEMA_VERSION == 1
    assert lc.check_lifting_json(json.loads(json.dumps(doc, default=str))) == []
    docs = (pathlib.Path(__file__).resolve().parents[1] / 'tutorial' / 'docs' / 'lifting.md').read_text()
    assert [f for f, *_ in lc.LIFTING_JSON_FIELDS if f'`{f}`' not in docs] == []
    bad = dict(doc)
    del bad['lifted']
    assert 'missing lifted' in lc.check_lifting_json(bad)


def test_cli_writes_the_three_files(tmp_path):
    path = tmp_path / 'm.cellml'
    path.write_text(_model({x: -x ** 3}, {x: 1.0}, {}))
    assert lc.main([str(path), '--output-dir', str(tmp_path / 'o'), '--sim-time', '1']) == 0
    assert {p.name for p in (tmp_path / 'o').iterdir()} >= {'m_lifted.cellml', 'lifting.json', 'lifting_functions.py'}
