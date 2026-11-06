"""smooth_cellml: the smooth replacements of abs, max, min and piecewise in a flat CellML model.

The models here are written by hand, so each test runs in seconds. One integration test smooths
the generated 3compartment model and compares its simulation with the original's.
"""
import json
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from libcuflynx.utilities import smooth_cellml as sc

_M = sc._M
_C = sc._C

UNITS = """
  <units name="mmHg"><unit units="metre"/></units>
  <units name="mmHg_per_s"><unit units="mmHg"/><unit units="second" exponent="-1"/></units>
"""


def _model(equations, variables=''):
    """A one-component model with a ramp v = -1 + t (mmHg) and the given equations."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<model xmlns="http://www.cellml.org/cellml/2.0#" xmlns:cellml="http://www.cellml.org/cellml/2.0#" name="m">
{UNITS}
  <component name="c">
    <variable name="t" units="second"/>
    <variable name="v" units="mmHg" initial_value="-1"/>
    <variable name="r" units="mmHg_per_s" initial_value="1"/>
    <variable name="zero" units="mmHg" initial_value="0"/>
    <variable name="half" units="mmHg" initial_value="0.5"/>
    <variable name="y" units="mmHg"/>
    {variables}
    <math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>v</ci></apply><ci>r</ci></apply>
      {equations}
    </math>
  </component>
</model>"""


def _y(rhs):
    return f'<apply><eq/><ci>y</ci>{rhs}</apply>'


ABS = _y('<apply><abs/><ci>v</ci></apply>')
MAX = _y('<apply><max/><ci>v</ci><ci>zero</ci></apply>')
MIN = _y('<apply><min/><ci>v</ci><ci>half</ci></apply>')
CLAMP = _y('<apply><max/><apply><min/><ci>v</ci><ci>half</ci></apply><ci>zero</ci></apply>')
PIECEWISE_GEQ = _y('<piecewise><piece><ci>v</ci><apply><geq/><ci>v</ci><ci>zero</ci></apply></piece>'
                   '<otherwise><ci>half</ci></otherwise></piecewise>')
PIECEWISE_AND = _y('<piecewise><piece><ci>v</ci><apply><and/><apply><gt/><ci>v</ci><ci>zero</ci></apply>'
                   '<apply><lt/><ci>v</ci><ci>half</ci></apply></apply></piece>'
                   '<otherwise><ci>zero</ci></otherwise></piecewise>')
PIECEWISE_OR_NOT = _y('<piecewise><piece><ci>v</ci><apply><or/><apply><leq/><ci>v</ci><ci>zero</ci></apply>'
                      '<apply><not/><apply><lt/><ci>v</ci><ci>half</ci></apply></apply></apply></piece>'
                      '<otherwise><ci>zero</ci></otherwise></piecewise>')
PIECEWISE_THREE = _y('<piecewise>'
                     '<piece><ci>zero</ci><apply><lt/><ci>v</ci><ci>zero</ci></apply></piece>'
                     '<piece><ci>v</ci><apply><lt/><ci>v</ci><ci>half</ci></apply></piece>'
                     '<otherwise><ci>half</ci></otherwise></piecewise>')


def _root(text):
    return ET.fromstring(text.encode())


def _equation(text, name):
    """The right-hand side of ``name = ...`` in the model text."""
    for eq in _root(text).iter(_M + 'apply'):
        if sc._op(eq) != 'eq':
            continue
        lhs = sc._args(eq)[0]
        if sc._op(lhs) == 'diff':
            lhs = sc._args(lhs)[0]
        if sc._tag(lhs) == 'ci' and lhs.text.strip() == name:
            return sc.infix(sc._args(eq)[1])
    raise KeyError(name)


def _ranges(lo=-1.0, hi=1.0, n=20):
    return {f'sm{k}_x': {'min': lo, 'max': hi} for k in range(1, n + 1)}


def _no_new_issues(original, smoothed):
    before, after = sc.libcellml_issues(original), sc.libcellml_issues(smoothed)
    assert after == before == {'validator': [], 'analyser': []}


# ---------------------------------------------------------------------------------------------
# each kind

@pytest.mark.parametrize('equations, replacement, x', [
    (ABS, 'sm1_w', 'v'),
    (MAX, '(0.5*((v+zero)+sm1_w))', '(v-zero)'),
    (MIN, '(0.5*((v+half)-sm1_w))', '(v-half)'),
    (PIECEWISE_GEQ, '(half+(sm1_H*(v-half)))', '(v-zero)'),
])
def test_each_kind_is_rewritten_as_in_the_table(equations, replacement, x):
    text = _model(equations)
    smoothed, report = sc.smooth(text, _ranges())
    assert _equation(smoothed, 'y') == replacement
    assert _equation(smoothed, 'sm1_x') == x
    assert _equation(smoothed, 'sm1_w') == 'sqrt(((sm1_x^2)+(eps_sm1^2)))'
    assert _equation(smoothed, 'sm1_z') == '(1/sm1_w)'
    if 'piecewise' in equations:
        assert _equation(smoothed, 'sm1_H') == '(0.5*(1+(sm1_x*sm1_z)))'
    assert [s['k'] for s in report['sites']] == [1]
    _no_new_issues(text, smoothed)


def _value(text, name, **values):
    """``name``'s right-hand side in the model text, evaluated at ``values``."""
    return eval(_equation(text, name).replace('^', '**'), {}, values)      # noqa: S307 - our own infix


def test_the_rectifier_in_xh_form_is_b_plus_x_times_h():
    smoothed, report = sc.smooth(_model(MAX + MIN.replace('<ci>y</ci>', '<ci>q</ci>'), '<variable name="q" units="mmHg"/>'),
                                 _ranges(), maxmin_form='xH')
    assert _equation(smoothed, 'y') == '(zero+(sm1_x*sm1_H))'
    assert _equation(smoothed, 'q') == '(v-(sm2_x*sm2_H))'
    assert _equation(smoothed, 'sm1_H') == '(0.5*(1+(sm1_x*sm1_z)))'
    assert [s['rule'] for s in report['sites']] == ['maxmin', 'maxmin']
    assert [s['lifted'] for s in report['sites']] == [4, 4] and report['lifted_estimate'] == 8
    for h, v in ((1.0, 2.0), (0.0, -2.0), (0.5, 0.0)):
        assert _value(smoothed, 'y', sm1_x=v, sm1_H=h, zero=0.0) == pytest.approx(v * h)


def test_every_record_names_its_rule_and_lifted_count():
    text = _model(ABS + PIECEWISE_GEQ.replace('<ci>y</ci>', '<ci>q</ci>')
                  + _y('<apply><max/><ci>v</ci><cn cellml:units="mmHg">0</cn></apply>').replace('<ci>y</ci>', '<ci>p</ci>'),
                  '<variable name="q" units="mmHg"/><variable name="p" units="mmHg"/>')
    _, report = sc.smooth(text, _ranges())
    assert [(s['rule'], s['lifted']) for s in report['sites']] == [('abs', 2), ('piecewise', 2), ('rectifier', 2)]
    assert report['lifted_estimate'] == 6
    # auto: xH for a rectifier only
    assert _equation(sc.smooth(text, _ranges(), maxmin_form='auto')[0], 'p') == '(0+(sm3_x*sm3_H))'
    _, report = sc.smooth(text, _ranges(), exempt=[{'variable': 'q'}])
    assert [s['rule'] for s in report['sites']] == ['abs', 'exempt', 'rectifier']


def test_eps_of_an_argument_that_keeps_one_sign_stays_below_its_smallest_value():
    """abs(tau_h_Na1_6/1000) in SN_simple: x in [1.5e-4, 0.245]. rel x range (2.45e-4) would exceed the
    smallest x and slow inactivation 2x during each spike; rel x min(range, smallest |x|) is 1.5e-7."""
    _, report = sc.smooth(_model(ABS), {'sm1_x': {'min': 1.5e-4, 'max': 0.245, 'min_abs': 1.5e-4}})
    assert report['sites'][0]['eps'] == pytest.approx(1.5e-7)
    _, report = sc.smooth(_model(ABS), {'sm1_x': {'min': -1.0, 'max': 1.0, 'min_abs': 0.0}})
    assert report['sites'][0]['eps'] == pytest.approx(2e-3)
    # a comparison keeps rel x range: never seen to switch, it may switch between samples
    _, report = sc.smooth(_model(PIECEWISE_GEQ), {'sm1_x': {'min': 1.5e-4, 'max': 0.245, 'min_abs': 1.5e-4}})
    assert report['sites'][0]['eps'] == pytest.approx(1e-3 * (0.245 - 1.5e-4))


def test_less_than_swaps_the_sign_of_x_and_and_is_a_product():
    smoothed, report = sc.smooth(_model(PIECEWISE_AND), _ranges())
    assert _equation(smoothed, 'sm1_x') == '(v-zero)'
    assert _equation(smoothed, 'sm2_x') == '(half-v)'
    assert _value(smoothed, 'y', sm1_H=0.3, sm2_H=0.6, v=1.0, zero=0.0) == pytest.approx(0.3 * 0.6)
    assert [s['kind'] for s in report['sites']] == ['piecewise', 'piecewise']


def test_or_and_not():
    smoothed, _ = sc.smooth(_model(PIECEWISE_OR_NOT), _ranges())
    # (v <= zero) or not (v < half): H1 for zero - v >= 0, H2 for half - v > 0
    assert _value(smoothed, 'y', sm1_H=0.3, sm2_H=0.6, v=1.0, zero=0.0) == pytest.approx(1 - 0.7 * 0.6)


COMPLEMENTARY_RATE = ('<apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>z</ci></apply><piecewise>'
                      '<piece><ci>p1</ci><apply><leq/><ci>v</ci><ci>zero</ci></apply></piece>'
                      '<piece><ci>p2</ci><apply><and/><apply><gt/><ci>v</ci><ci>zero</ci></apply>'
                      '<apply><lt/><ci>v</ci><ci>big</ci></apply></apply></piece>'
                      '<otherwise><cn cellml:units="per_s">0</cn></otherwise></piecewise></apply>')
COMPLEMENTARY_VARIABLES = ('<variable name="z" units="dimensionless" initial_value="0"/>'
                           '<variable name="p1" units="per_s" initial_value="1"/>'
                           '<variable name="p2" units="per_s" initial_value="1"/>'
                           '<variable name="big" units="mmHg" initial_value="100"/>')


def _complementary_model():
    text = _model(COMPLEMENTARY_RATE, COMPLEMENTARY_VARIABLES)
    return text.replace('<units name="mmHg">', '<units name="per_s"><unit units="second" exponent="-1"/></units>\n'
                                               '  <units name="mmHg">')


def test_complementary_conditions_share_one_site_and_the_weights_sum_to_1():
    """f1 if v <= zero, f2 if v > zero and v < big, else 0: v > zero is the complement of v <= zero, so
    it is the same site, and at the boundary the weights are H and 1 - H (not H and (1 - H)^2)."""
    smoothed, report = sc.smooth(_complementary_model(), _ranges())
    assert [s['original'] for s in report['sites']] == ['(v<=zero)', '(v<big)']
    for h1 in (0.0, 0.25, 0.5, 0.9):
        assert _value(smoothed, 'z', sm1_H=h1, sm2_H=1.0, p1=1.0, p2=1.0) == pytest.approx(1.0)
    assert _value(smoothed, 'z', sm1_H=0.5, sm2_H=1.0, p1=1.0, p2=0.0) == pytest.approx(0.5)


def test_complementary_conditions_reproduce_the_rate_across_the_boundary(tmp_path):
    """The rate is p1 = p2 = 1 on both sides of v = 0 (crossed at t = 1), so z = t exactly."""
    text = _complementary_model()
    smoothed, _ = sc.smooth(text, {'sm1_x': {'min': -1.0, 'max': 1.0}, 'sm2_x': {'min': 99.9, 'max': 100.1}}, rel=0.1)
    z0, = _simulate(text, tmp_path, 'o', ['z'])
    z1, = _simulate(smoothed, tmp_path, 's', ['z'])
    assert np.max(np.abs(z1 - z0)) < 1e-6
    assert z0[-1] == pytest.approx(2.0, rel=1e-6)


def test_several_pieces_blend_from_the_last_back():
    smoothed, _ = sc.smooth(_model(PIECEWISE_THREE), _ranges())
    assert _equation(smoothed, 'y') == '((half+(sm2_H*(v-half)))+(sm1_H*(zero-(half+(sm2_H*(v-half))))))'


def test_a_clamp_is_two_sites_rewritten_bottom_up():
    smoothed, report = sc.smooth(_model(CLAMP), _ranges())
    assert _equation(smoothed, 'sm1_x') == '(v-half)'
    assert _equation(smoothed, 'sm2_x') == '((0.5*((v+half)-sm1_w))-zero)'
    assert _equation(smoothed, 'y') == '(0.5*(((0.5*((v+half)-sm1_w))+zero)+sm2_w))'
    assert [s['original'] for s in report['sites']] == ['min(v, half)', 'max(min(v, half), zero)']


def test_auxiliaries_are_algebraic_variables_in_the_site_component():
    from libcellml import Analyser, AnalyserVariable, Parser
    smoothed, _ = sc.smooth(_model(PIECEWISE_GEQ), _ranges())
    analyser = Analyser()
    analyser.analyseModel(Parser().parseModel(smoothed))
    model = analyser.analyserModel()
    assert model.stateCount() == 1
    kinds = {}
    for v in list(model.constants()) + list(model.algebraicVariables()):
        kinds[v.variable().name()] = AnalyserVariable.typeAsString(v.type())
    assert kinds['eps_sm1'] == 'constant'
    for name in ('sm1_x', 'sm1_w', 'sm1_z', 'sm1_H'):
        assert kinds[name] == 'algebraic_variable'


# ---------------------------------------------------------------------------------------------
# units

def test_units_of_x_are_inferred_with_composite_units_added():
    equations = _y('<apply><divide/><apply><abs/><apply><times/><ci>v</ci><ci>r</ci></apply></apply><ci>r</ci></apply>')
    text = _model(equations)
    smoothed, report = sc.smooth(text, _ranges())
    site = report['sites'][0]
    assert site['units_x'] == 'smu_mmHg_e1_mmHg_per_s_e1'
    root = _root(smoothed)
    units = {u.get('name'): {x.get('units'): x.get('exponent', '1') for x in u} for u in root.iter(_C + 'units')}
    assert units['smu_mmHg_e1_mmHg_per_s_e1'] == {'mmHg': '1', 'mmHg_per_s': '1'}
    assert units['smu_mmHg_em1_mmHg_per_s_em1'] == {'mmHg': '-1', 'mmHg_per_s': '-1'}
    _no_new_issues(text, smoothed)


@pytest.mark.parametrize('rhs, expected', [
    ('<cn cellml:units="mmHg">2</cn>', {'mmHg': 1.0}),
    ('<apply><power/><ci>v</ci><cn cellml:units="dimensionless">2</cn></apply>', {'mmHg': 2.0}),
    ('<apply><root/><apply><times/><ci>v</ci><ci>v</ci></apply></apply>', {'mmHg': 1.0}),
    ('<apply><exp/><ci>k</ci></apply>', {}),
    ('<apply><ln/><ci>k</ci></apply>', {}),
    ('<apply><minus/><ci>v</ci><cn cellml:units="mmHg">1</cn></apply>', {'mmHg': 1.0}),
    ('<apply><divide/><ci>v</ci><ci>r</ci></apply>', {'mmHg': 1.0, 'mmHg_per_s': -1.0}),
    ('<apply><power/><ci>v</ci><ci>k</ci></apply>', None),
])
def test_units_inference(rhs, expected):
    e = ET.fromstring(f'<x xmlns="{sc.MATHML_NS}" xmlns:cellml="{sc.CELLML_NS}">{rhs}</x>')[0]
    var_units = {'v': {'mmHg': 1.0}, 'r': {'mmHg_per_s': 1.0}, 'k': {}}
    assert sc._units_of(e, var_units) == expected


def test_a_site_whose_units_cannot_be_inferred_is_skipped_and_reported():
    variables = '<variable name="n" units="dimensionless" initial_value="1"/>'
    text = _model(_y('<apply><abs/><apply><power/><ci>v</ci><ci>n</ci></apply></apply>'), variables)
    smoothed, report = sc.smooth(text, _ranges())
    site = report['sites'][0]
    assert site['status'] == 'skipped' and 'units' in site['reason']
    assert report['counts'] == {'abs_skipped': 1}
    assert _equation(smoothed, 'y') == 'abs((v^n))'
    assert sc.libcellml_issues(smoothed) == sc.libcellml_issues(text)


# ---------------------------------------------------------------------------------------------
# eps, deviation and collapse

def test_eps_is_rel_times_the_range_with_an_absolute_fallback():
    ranges = {'sm1_x': {'min': -2.0, 'max': 6.0}}
    _, report = sc.smooth(_model(ABS), ranges, rel={'abs': 1e-2})
    assert report['sites'][0]['eps'] == pytest.approx(0.08)
    assert report['sites'][0]['rel'] == 1e-2
    # constant in the nominal run, moved in another (a swept input at rest): rel x its range over all runs
    _, report = sc.smooth(_model(ABS), {'sm1_x': {'min': 0.0, 'max': 0.0, 'any_min': -1.0, 'any_max': 3.0}}, rel=1e-2)
    assert report['sites'][0]['eps'] == pytest.approx(0.04) and 'zero_range' in report['sites'][0]['flags']
    _, report = sc.smooth(_model(ABS), {}, abs_eps=1e-5)
    assert report['sites'][0]['eps'] == 1e-5 and 'no_range' in report['sites'][0]['flags']
    _, report = sc.smooth(_model(ABS + MAX.replace('<ci>y</ci>', '<ci>q</ci>'), '<variable name="q" units="mmHg"/>'),
                          {'sm1_x': {'min': -1.0, 'max': 1.0}, 'sm2_x': {'min': -1.0, 'max': 1.0}},
                          rel={'abs': 1e-2, 'variables': {'c/q': 1e-4}})
    assert [s['eps'] for s in report['sites']] == [pytest.approx(0.02), pytest.approx(2e-4)]


def test_a_site_whose_argument_is_constant_in_every_run_is_left_as_written_and_flagged():
    # max(x, 0) with x = 0 in every reference run (an afferent input at rest): eps from abs_eps would
    # offset the output by eps/2 for nothing, and no run says how wide the switch should be
    for r in ({'min': 0.0, 'max': 0.0, 'any_min': 0.0, 'any_max': 0.0}, [0.0, 0.0]):
        smoothed, report = sc.smooth(_model(MAX), {'sm1_x': r})
        site = report['sites'][0]
        assert site['status'] == 'not_needed' and 'not_exercised' in site['flags'], r
        assert site['constant_value'] == 0.0
        assert _equation(smoothed, 'y') == _equation(_model(MAX), 'y')
        assert report['counts'] == {} and report['lifted_estimate'] == 0
    _, report = sc.smooth(_model(MAX), {'sm1_x': {'min': 0.48, 'max': 0.48, 'any_min': 0.48, 'any_max': 0.48}})
    assert report['sites'][0]['constant_value'] == 0.48 and report['sites'][0]['status'] == 'not_needed'


def test_a_comparison_constant_in_every_run_takes_its_outcome_and_the_piecewise_is_still_smoothed():
    # v >= 0 and w >= 0, with w at 2 throughout: the piecewise is smoothed on v alone
    text = _model(_y('<piecewise><piece><ci>v</ci><apply><and/><apply><geq/><ci>v</ci><ci>zero</ci></apply>'
                     '<apply><geq/><ci>w</ci><ci>zero</ci></apply></apply></piece><otherwise><ci>zero</ci></otherwise></piecewise>'),
                  '<variable name="w" units="mmHg" initial_value="2"/>')
    smoothed, report = sc.smooth(text, {'sm1_x': {'min': -1.0, 'max': 1.0},
                                        'sm2_x': {'min': 2.0, 'max': 2.0, 'any_min': 2.0, 'any_max': 2.0}})
    v_site, w_site = report['sites']
    assert v_site['status'] == 'smoothed' and not v_site['collapsed']
    assert w_site['collapsed'] and 'not_exercised' in w_site['flags'] and w_site['constant_value'] == 2.0
    assert 'piecewise' not in _equation(smoothed, 'y') and 'sm1_H' in _equation(smoothed, 'y')


def _simulate(text, tmp_path, name, variables, sim_time=2.0):
    from libcuflynx.solver_wrappers import get_simulation_helper
    path = tmp_path / f'{name}.cellml'
    path.write_text(text)
    helper = get_simulation_helper(model_path=str(path), solver='CVODE_myokit', model_type='cellml',
                                   dt=0.01, sim_time=sim_time, solver_info={'MaximumStep': 0.01})
    assert helper.run()
    out = helper.get_results([[f'c/{v}'] for v in variables], flatten=True)
    helper.close_simulation()
    return [np.asarray(o, dtype=float).ravel() for o in out]


@pytest.mark.parametrize('equations, kind', [(ABS, 'abs'), (MAX, 'maxmin'), (PIECEWISE_GEQ, 'piecewise')])
def test_smoothed_is_within_the_bound_away_from_the_switch(tmp_path, equations, kind):
    """|H - step| <= eps^2/(4x^2); for abs the value is off by <= eps^2/(2|x|), for max by half that."""
    text = _model(equations)
    probed = sc.probe(text)
    (tmp_path / 'p.cellml').write_text(probed)
    ranges = sc.reference_ranges(str(tmp_path / 'p.cellml'), [{'sim_time': 2.0, 'dt': 0.01}])
    rel = 0.05
    smoothed, report = sc.smooth(text, ranges, rel=rel)
    eps = report['sites'][0]['eps']
    assert eps == pytest.approx(rel * 2.0)
    y0, x = _simulate(text, tmp_path, 'o', ['y', 'v'])
    y1, = _simulate(smoothed, tmp_path, 's', ['y'])
    away = np.abs(x) > 4 * eps
    err = np.abs(y1 - y0)[away]
    xa = np.abs(x)[away]
    bound = {'abs': eps ** 2 / (2 * xa), 'maxmin': eps ** 2 / (4 * xa),
             'piecewise': eps ** 2 / (4 * xa ** 2) * np.abs(x[away] - 0.5)}[kind]
    assert np.all(err <= bound * (1 + 1e-6) + 1e-12)
    assert np.max(np.abs(y1 - y0)) <= report['sites'][0]['max_deviation'] * (
        np.max(np.abs(x - 0.5)) if kind == 'piecewise' else 1) * (1 + 1e-6) + 1e-9
    assert report['sites'][0]['margin'] == pytest.approx(0.0, abs=1e-12)


def test_collapse_replaces_a_site_that_never_switches_and_records_its_margin():
    text = _model(MAX + PIECEWISE_THREE.replace('<ci>y</ci>', '<ci>q</ci>'),
                  '<variable name="q" units="mmHg"/>')
    ranges = {'sm1_x': {'min': 0.2, 'max': 3.0, 'min_abs': 0.2},    # max: v always above zero
              'sm2_x': {'min': -3.0, 'max': -0.2},                  # v < zero never true
              'sm3_x': {'min': -1.0, 'max': 1.0}}                   # v < half switches
    smoothed, report = sc.smooth(text, ranges, collapse=True)
    sites = {s['k']: s for s in report['sites']}
    assert sites[1]['collapsed'] and sites[1]['margin'] == 0.2 and sites[1]['max_deviation'] == 0.0
    assert _equation(smoothed, 'y') == 'v'
    assert sites[2]['collapsed'] and sites[2]['margin'] == pytest.approx(0.2)
    assert not sites[3]['collapsed']
    assert _equation(smoothed, 'q') == '(half+(sm3_H*(v-half)))'
    assert 'sm1_x' not in smoothed and 'sm3_H' in smoothed
    _no_new_issues(text, smoothed)
    smoothed_off, report_off = sc.smooth(text, ranges)
    assert not any(s['collapsed'] for s in report_off['sites'])
    _, report_one = sc.smooth(text, ranges, collapse=[2])
    assert [s['collapsed'] for s in report_one['sites']] == [False, True, False]


def test_a_site_that_reaches_its_switch_is_one_sided_and_collapsed_only_when_named():
    """x >= 0 with x = 0 reached (e.g. a phase where a floor wraps): v >= zero always holds, but
    abs(v) at v = 0 is where a perturbed run changes branch, so collapse=True leaves it."""
    text = _model(PIECEWISE_GEQ + ABS.replace('<ci>y</ci>', '<ci>q</ci>'), '<variable name="q" units="mmHg"/>')
    ranges = {'sm1_x': {'min': 0.0, 'max': 1.0}, 'sm2_x': {'min': 0.0, 'max': 1.0}}
    _, report = sc.smooth(text, ranges, collapse=True)
    assert [s['flags'] for s in report['sites']] == [['one_sided'], ['one_sided']]
    assert [s['collapsed'] for s in report['sites']] == [False, False]
    smoothed, report = sc.smooth(text, ranges, collapse=[1, 2])
    assert [s['collapsed'] for s in report['sites']] == [True, True]
    assert _equation(smoothed, 'y') == 'v' and _equation(smoothed, 'q') == 'v'


@pytest.mark.parametrize('op, lo, collapsed', [('geq', 0.0, True), ('gt', 0.0, False), ('gt', 0.1, True)])
def test_a_strict_comparison_switches_at_zero(op, lo, collapsed):
    text = _model(PIECEWISE_GEQ.replace('geq', op))
    _, report = sc.smooth(text, {'sm1_x': {'min': lo, 'max': 1.0}}, collapse=[1])
    assert report['sites'][0]['collapsed'] is collapsed


# ---------------------------------------------------------------------------------------------
# left as written, and flags

def test_floor_and_eq_conditions_are_reported_not_smoothed_and_floor_upstream_is_flagged():
    variables = '<variable name="ph" units="dimensionless"/><variable name="s" units="dimensionless"/>' \
                '<variable name="q" units="mmHg"/>'
    equations = (
        '<apply><eq/><ci>s</ci><apply><divide/><ci>v</ci><ci>half</ci></apply></apply>'
        '<apply><eq/><ci>ph</ci><apply><minus/><ci>s</ci><apply><floor/><ci>s</ci></apply></apply></apply>'
        + _y('<piecewise><piece><ci>v</ci><apply><gt/><ci>ph</ci><cn cellml:units="dimensionless">0.5</cn>'
             '</apply></piece><otherwise><ci>zero</ci></otherwise></piecewise>')
        + '<apply><eq/><ci>q</ci><piecewise><piece><ci>v</ci><apply><eq/><ci>v</ci><ci>zero</ci></apply></piece>'
          '<otherwise><ci>zero</ci></otherwise></piecewise></apply>')
    text = _model(equations, variables)
    smoothed, report = sc.smooth(text, _ranges())
    kinds = sorted((n['kind'], n['variable']) for n in report['not_smoothed'])
    assert kinds == [('floor', 'ph'), ('piecewise', 'q')]
    assert report['sites'][0]['flags'] == ['floor_upstream']
    assert 'floor(s)' in _equation(smoothed, 'ph')
    assert _equation(smoothed, 'q').startswith('{v if (v==zero)')
    assert report['counts'] == {'piecewise': 1, 'floor_not_smoothed': 1, 'piecewise_not_smoothed': 1}
    _no_new_issues(text, smoothed)


def test_floor_upstream_follows_connections_between_components():
    text = f"""<?xml version="1.0" encoding="UTF-8"?>
<model xmlns="http://www.cellml.org/cellml/2.0#" xmlns:cellml="http://www.cellml.org/cellml/2.0#" name="m">
  <component name="a">
    <variable name="t" units="second" interface="public"/>
    <variable name="ph" units="dimensionless" interface="public"/>
    <math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><eq/><ci>ph</ci><apply><floor/><apply><divide/><ci>t</ci><cn cellml:units="second">1</cn></apply></apply></apply>
    </math>
  </component>
  <component name="b">
    <variable name="ph" units="dimensionless" interface="public"/>
    <variable name="y" units="dimensionless"/>
    <math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><eq/><ci>y</ci><apply><abs/><ci>ph</ci></apply></apply>
    </math>
  </component>
  <connection component_1="a" component_2="b"><map_variables variable_1="ph" variable_2="ph"/></connection>
</model>"""
    _, report = sc.smooth(text, _ranges())
    assert report['sites'][0]['flags'] == ['floor_upstream']
    assert report['sites'][0]['component'] == 'b'


def test_a_branch_that_may_be_undefined_off_its_condition_is_flagged():
    variables = '<variable name="f" units="dimensionless"/>'
    equations = ('<apply><eq/><ci>f</ci><piecewise><piece><apply><ln/><apply><plus/><apply><divide/><ci>v</ci>'
                 '<ci>half</ci></apply><cn cellml:units="dimensionless">1</cn></apply></apply>'
                 '<apply><geq/><ci>v</ci><ci>zero</ci></apply></piece>'
                 '<otherwise><cn cellml:units="dimensionless">0</cn></otherwise></piecewise></apply>')
    _, report = sc.smooth(_model(equations, variables), _ranges())
    assert report['sites'][0]['flags'] == ['branch_domain']
    _, report = sc.smooth(_model(PIECEWISE_GEQ), _ranges())
    assert report['sites'][0]['flags'] == []


PHASE_VARIABLES = ('<variable name="s" units="dimensionless" initial_value="0"/>'
                   '<variable name="mt" units="dimensionless"/><variable name="act" units="dimensionless"/>'
                   '<variable name="e" units="dimensionless"/><variable name="c" units="dimensionless" initial_value="0"/>')
PHASE_EQUATIONS = (
    '<apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>s</ci></apply><cn cellml:units="per_s">1</cn></apply>'
    '<apply><eq/><ci>mt</ci><apply><minus/><ci>s</ci><apply><floor/><ci>s</ci></apply></apply></apply>'
    '<apply><eq/><ci>act</ci><piecewise><piece><apply><times/><cn cellml:units="dimensionless">2</cn><ci>mt</ci></apply>'
    '<apply><leq/><ci>mt</ci><cn cellml:units="dimensionless">0.5</cn></apply></piece>'
    '<otherwise><cn cellml:units="dimensionless">0</cn></otherwise></piecewise></apply>'
    '<apply><eq/><ci>e</ci><apply><times/><cn cellml:units="dimensionless">0.5</cn><apply><minus/>'
    '<cn cellml:units="dimensionless">1</cn><apply><cos/><apply><times/><cn cellml:units="dimensionless">2</cn><pi/>'
    '<ci>act</ci></apply></apply></apply></apply></apply>'
    '<apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>c</ci></apply><piecewise>'
    '<piece><cn cellml:units="per_s">1</cn><apply><lt/><ci>mt</ci><cn cellml:units="dimensionless">0.3</cn></apply></piece>'
    '<otherwise><cn cellml:units="per_s">0</cn></otherwise></piecewise></apply>' + PIECEWISE_GEQ)


def _phase_model():
    return _model(PHASE_EQUATIONS, PHASE_VARIABLES).replace(
        '<units name="mmHg">', '<units name="per_s"><unit units="second" exponent="-1"/></units>\n  <units name="mmHg">')


def test_jump_candidates_are_reported_with_their_switch_variables():
    """act jumps from 1 to 0 but reaches only cos(2 pi act); c's rate is on/off; y has no constant branch."""
    text = _phase_model()
    smoothed, report = sc.smooth(text, _ranges())
    found = {j['variable']: j for j in report['jump_candidates']}
    assert set(found) == {'act', 'c'}
    assert 'cos/sin' in found['act']['reason'] and 'state derivative' in found['c']['reason']
    assert found['act']['switch_variables'] == [{'variable': 'mt', 'phase': True, 'state': 's'}]
    assert not found['act']['exempt']
    y_site = next(s for s in report['sites'] if s['variable'] == 'y')
    assert 'jump_candidate' not in y_site['flags']
    assert y_site['status'] == 'smoothed'
    assert 'sm1_H' in _equation(smoothed, 'act')            # reported only, still smoothed
    _no_new_issues(text, smoothed)


def test_exempt_sites_are_left_as_written_with_their_reason():
    text = _phase_model()
    exempt = [{'variable': 'act', 'reason': 'continuous through cos'}, {'variable': 'c/c', 'reason': 'on/off rate'}]
    smoothed, report = sc.smooth(text, _ranges(), exempt=exempt)
    by_variable = {s['variable']: s for s in report['sites']}
    assert by_variable['act']['status'] == 'exempt' and by_variable['act']['reason'] == 'continuous through cos'
    assert by_variable['act']['switch_variables'] == [{'variable': 'mt', 'phase': True, 'state': 's'}]
    assert by_variable['c']['status'] == 'exempt' and by_variable['y']['status'] == 'smoothed'
    assert _equation(smoothed, 'act').startswith('{(2*mt) if (mt<=0.5)')
    assert report['counts'] == {'piecewise_exempt': 2, 'piecewise': 1, 'floor_not_smoothed': 1}
    assert report['exempt'] == exempt and all(j['exempt'] for j in report['jump_candidates'])
    _no_new_issues(text, smoothed)


def test_switch_variables_follow_algebraic_equations_to_a_state():
    variables = '<variable name="d" units="mmHg"/>'
    equations = ('<apply><eq/><ci>d</ci><apply><minus/><ci>v</ci><ci>half</ci></apply></apply>'
                 + _y('<apply><abs/><ci>d</ci></apply>'))
    smoothed, report = sc.smooth(_model(equations, variables), _ranges(), exempt=[{'variable': 'y'}])
    assert report['sites'][0]['switch_variables'] == [{'variable': 'v', 'phase': False, 'state': 'v'}]
    assert report['sites'][0]['reason'] == 'exempt'


# ---------------------------------------------------------------------------------------------
# probe, the file helper and the command

def test_probe_adds_only_the_x_variables(tmp_path):
    text = _model(CLAMP)
    probed = sc.probe(text)
    assert sc.probe_variables(probed) == [('c', 'sm1_x'), ('c', 'sm2_x')]
    assert _equation(probed, 'y') == _equation(text, 'y')
    assert _equation(probed, 'sm2_x') == '(min(v, half)-zero)'
    assert 'sm1_w' not in probed and 'eps_sm1' not in probed
    _no_new_issues(text, probed)


def test_smooth_cellml_file_writes_the_model_and_report(tmp_path):
    path = tmp_path / 'clamp.cellml'
    path.write_text(_model(CLAMP))
    smooth_path, report_path = sc.smooth_cellml_file(str(path), runs=[{'sim_time': 2.0, 'dt': 0.01}],
                                                     output_dir=str(tmp_path / 'out'))
    assert os.path.basename(smooth_path) == 'clamp_smooth.cellml'
    report = json.loads(open(report_path).read())
    assert report['new_issues'] == {'validator': [], 'analyser': []}
    assert report['sites'][0]['range'] == pytest.approx([-1.5, 0.5])
    assert report['sites'][1]['range'][0] == pytest.approx(-1.0, abs=1e-2)
    ranges = json.loads((tmp_path / 'out' / 'smoothing_ranges.json').read_text())
    assert ranges['sm1_x']['min'] == pytest.approx(-1.5) and ranges['sm1_x']['nan'] is False


def test_a_nan_in_the_runs_is_flagged():
    """max(NaN, 0) is 0 in the original, but NaN in the smooth form (e.g. 0/0 at t = 0)."""
    _, report = sc.smooth(_model(MAX), {'sm1_x': {'min': -1.0, 'max': 1.0, 'min_abs': 0.0, 'nan': True}})
    assert report['sites'][0]['flags'] == ['nan_in_runs']


def test_the_command_smooths_from_a_ranges_file(tmp_path):
    path = tmp_path / 'abs.cellml'
    path.write_text(_model(ABS))
    ranges = tmp_path / 'ranges.json'
    ranges.write_text(json.dumps({'sm1_x': [-1.0, 1.0]}))
    assert sc.main([str(path), '--ranges', str(ranges), '--rel', '0.01']) == 0
    report = json.loads((tmp_path / 'smoothing_report.json').read_text())
    assert report['sites'][0]['eps'] == pytest.approx(0.02)
    assert (tmp_path / 'abs_smooth.cellml').exists()


def test_reference_ranges_start_at_t0_and_include_pre_time(tmp_path):
    path = tmp_path / 'p.cellml'
    path.write_text(sc.probe(_model(ABS)))
    ranges, each = sc.reference_ranges(str(path), [{'pre_time': 2.0, 'sim_time': 1.0, 'dt': 0.01},
                                                   {'pre_time': 0.0, 'sim_time': 1.0, 'dt': 0.01,
                                                    'param_vals': {'c/r': 3.0}}], per_run=True)
    assert each[0]['sm1_x']['min'] == pytest.approx(-1.0) and each[0]['sm1_x']['max'] == pytest.approx(2.0)
    assert each[1]['sm1_x']['max'] == pytest.approx(2.0)          # r = 3 for 1 s from v = -1
    assert ranges['sm1_x'] == {'min': pytest.approx(-1.0), 'max': pytest.approx(2.0), 'min_abs': pytest.approx(0.0, abs=1e-12),
                               'nan': False, 'any_min_abs': pytest.approx(0.0, abs=1e-12), 'any_sign_change': True,
                               'any_min': pytest.approx(-1.0), 'any_max': pytest.approx(2.0)}


def test_eps_comes_from_the_nominal_run_unless_union_is_asked_for(tmp_path):
    """An outlying run (here r = 1000, as one chemoreceptor sweep point drove its x to -3.8e9) does
    not set eps for every point: the default range is the nominal (first) run's."""
    path = tmp_path / 'p.cellml'
    path.write_text(sc.probe(_model(ABS)))
    runs = [{'sim_time': 2.0, 'dt': 0.01}, {'sim_time': 2.0, 'dt': 0.01, 'param_vals': {'c/r': 1000.0}}]
    nominal = sc.reference_ranges(str(path), runs)
    union = sc.reference_ranges(str(path), runs, over='union')
    assert nominal['sm1_x']['max'] == pytest.approx(1.0) and union['sm1_x']['max'] == pytest.approx(1999.0)
    _, report = sc.smooth(_model(ABS), nominal)
    assert report['sites'][0]['eps'] == pytest.approx(2e-3)


def test_a_division_by_zero_at_t0_is_guarded_and_the_nan_it_hid_is_flagged(tmp_path):
    """max(v/(w r), 0) with w(0) = 0: the original gives 0 at t = 0 (max hides the -Inf), the probe records
    x = -Inf there (nan_in_runs), and the division is guarded, num den/(den^2 + eps_d^2), so the smoothed
    model is finite at t = 0 (no nan_at_init). The CVS-ANS C_O2_a case."""
    variables = '<variable name="w" units="second" initial_value="0"/><variable name="q" units="dimensionless"/>'
    equations = ('<apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>w</ci></apply>'
                 '<cn cellml:units="dimensionless">1</cn></apply>'
                 '<apply><eq/><ci>q</ci><apply><max/><apply><divide/><ci>v</ci><apply><times/><ci>w</ci>'
                 '<ci>r</ci></apply></apply><cn cellml:units="dimensionless">0</cn></apply></apply>')
    path = tmp_path / 'm.cellml'
    path.write_text(_model(equations, variables))
    _, report_path = sc.smooth_cellml_file(str(path), runs=[{'pre_time': 1.0, 'sim_time': 1.0, 'dt': 0.01}],
                                           output_dir=str(tmp_path / 'out'))
    report = json.loads(open(report_path).read())
    division, rectifier = report['sites']
    assert division['rule'] == 'division' and division['status'] == 'smoothed'
    assert division['eps'] == pytest.approx(1e-3 * 2.0)              # rel_div x the range of w r over [0, 2]
    assert rectifier['rule'] == 'rectifier' and 'nan_in_runs' in rectifier['flags']
    assert report['nan_at_init'] == [{'k': 2, 'probed': ['sm2_x']}]


def test_a_division_whose_denominator_stays_away_from_zero_is_left():
    equations = _y('<apply><divide/><ci>half</ci><apply><plus/><ci>v</ci><cn cellml:units="mmHg">3</cn></apply></apply>'
                   '').replace('<ci>half</ci>', '<apply><times/><ci>half</ci><ci>half</ci></apply>')
    text = _model(equations)
    smoothed, report = sc.smooth(text, {'sm1_x': {'min': 2.0, 'max': 4.0, 'min_abs': 2.0}})
    assert report['sites'][0]['status'] == 'not_needed'
    assert report['counts'] == {}
    assert 'sm1' not in smoothed


def test_only_smooths_the_named_sites():
    smoothed, report = sc.smooth(_model(CLAMP), _ranges(), only=[1])
    assert _equation(smoothed, 'y') == 'max((0.5*((v+half)-sm1_w)), zero)'
    assert [s['status'] for s in report['sites']] == ['smoothed', 'left']
    assert report['counts'] == {'maxmin': 1}
    smoothed, report = sc.smooth(_model(PIECEWISE_AND), _ranges(), only=[1])
    assert _equation(smoothed, 'y').startswith('{v if')
    assert [s['piecewise_k'] for s in report['sites']] == [1, 1]


# ---------------------------------------------------------------------------------------------

def test_analyse_model_returns_the_analyser_model():
    from libcellml import AnalyserModel, Parser
    from libcuflynx.utilities.libcellml_helper_funcs import analyse_model
    analysed = analyse_model(Parser().parseModel(_model(ABS)))
    assert analysed is not None
    assert analysed.type() == AnalyserModel.Type.ODE
    assert analysed.stateCount() == 1


@pytest.mark.integration
def test_3compartment_smoothed_simulates_close_to_the_original(generated_cellml_model_factory, tmp_path):
    """The error falls with rel. Volumes are compared in relative L2: the valve flows move with
    the timing of each valve switch, so their pointwise error is large at any eps."""
    from libcuflynx.solver_wrappers import get_simulation_helper
    model_path = generated_cellml_model_factory('3compartment')
    flat = os.path.join(os.path.dirname(model_path), '3compartment_flat.cellml')
    run = {'pre_time': 2.0, 'sim_time': 2.0, 'dt': 0.01, 'solver_info': {'MaximumStep': 0.001}}

    def states(path):
        helper = get_simulation_helper(model_path=path, solver='CVODE_myokit', model_type='cellml',
                                       dt=run['dt'], sim_time=run['sim_time'], pre_time=run['pre_time'],
                                       solver_info=run['solver_info'])
        assert helper.run()
        names = sorted(helper.state_index)
        out = dict(zip(names, (np.asarray(r, dtype=float).ravel()
                               for r in helper.get_results([[n] for n in names], flatten=True))))
        helper.close_simulation()
        return out

    original = states(flat)
    smooth_path, report_path = sc.smooth_cellml_file(flat, runs=[run], output_dir=str(tmp_path), rel=1e-4)
    report = json.loads(open(report_path).read())
    assert report['new_issues'] == {'validator': [], 'analyser': []}
    assert all(s['status'] in ('smoothed', 'not_needed') for s in report['sites'])
    assert {'abs', 'maxmin', 'piecewise', 'floor_not_smoothed'} <= set(report['counts'])
    ranges = json.loads((tmp_path / 'smoothing_ranges.json').read_text())

    def volume_error(smoothed_text):
        (tmp_path / 's.cellml').write_text(smoothed_text)
        smoothed = states(str(tmp_path / 's.cellml'))
        assert set(smoothed) == set(original)
        return max(np.linalg.norm(smoothed[k] - original[k]) / np.linalg.norm(original[k])
                   for k in original if '.q' in k)

    with open(smooth_path) as f:
        fine = volume_error(f.read())
    coarse = volume_error(sc.smooth(open(flat).read(), ranges, rel=1e-2)[0])
    assert fine < 2e-2
    assert fine < coarse / 10
