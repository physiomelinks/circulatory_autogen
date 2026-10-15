"""BE_batched_c: kernels from libCellML's analysis (kernel_ir), their C (generators/cpp/batched) and the solver.

The models are small CellML 2.0 files written here, so every test is quick:
- ``OPS``: one block of two states through most MathML operations and algebraic variables, for the exact Jacobian;
- ``CELLS``: two unconnected cells (one kernel, two blocks) and a coupled oscillator (another kernel, one block),
  with known solutions x = x0 exp(-k t) and p = cos(w t).
"""
import math
import os
import shutil

import numpy as np
import pytest

from libcuflynx.generators import kernel_ir

MATHML = 'xmlns="http://www.w3.org/1998/Math/MathML" xmlns:cellml="http://www.cellml.org/cellml/2.0#"'


def _cellml(components, units=''):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<model xmlns="http://www.cellml.org/cellml/2.0#" xmlns:cellml="http://www.cellml.org/cellml/2.0#" name="m">
  <units name="per_s"><unit units="second" exponent="-1"/></units>
  {units}
  {components}
</model>
"""


def _ode(state, rhs):
    return (f'<apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>{state}</ci></apply>{rhs}</apply>')


def _cn(v, units='dimensionless'):
    return f'<cn cellml:units="{units}">{v}</cn>'


def _cell(name, k, x0):
    return f"""
  <component name="{name}">
    <variable name="t" units="second" interface="public"/>
    <variable name="x0" units="dimensionless" initial_value="{x0}"/>
    <variable name="x" units="dimensionless" initial_value="x0"/>
    <variable name="k" units="per_s" initial_value="{k}"/>
    <variable name="flux" units="per_s"/>
    <math {MATHML}>
      <apply><eq/><ci>flux</ci><apply><times/><ci>k</ci><ci>x</ci></apply></apply>
      {_ode('x', '<apply><minus/><ci>flux</ci></apply>')}
    </math>
  </component>"""


CELLS = _cellml(_cell('cell_a', 0.5, 2.0) + _cell('cell_b', 0.2, 1.0) + f"""
  <component name="pair">
    <variable name="t" units="second" interface="public"/>
    <variable name="p" units="dimensionless" initial_value="1"/>
    <variable name="q" units="per_s" initial_value="0"/>
    <variable name="w" units="per_s" initial_value="3"/>
    <math {MATHML}>
      {_ode('p', '<ci>q</ci>')}
      {_ode('q', '<apply><times/><apply><minus/><ci>w</ci></apply><ci>w</ci><ci>p</ci></apply>')}
    </math>
  </component>
  <component name="environment"><variable name="t" units="second" interface="public"/></component>
  <connection component_1="environment" component_2="cell_a"><map_variables variable_1="t" variable_2="t"/></connection>
  <connection component_1="environment" component_2="cell_b"><map_variables variable_1="t" variable_2="t"/></connection>
  <connection component_1="environment" component_2="pair"><map_variables variable_1="t" variable_2="t"/></connection>
""", units='<units name="per_s2"><unit units="second" exponent="-2"/></units>')

# Dimensionless throughout (time too), so every operation is allowed on every variable. The piecewise w is
# continuous at x = 1, and the rates keep x above -2 (where sqrt(x + 2) would end the solution).
OPS = _cellml(f"""
  <component name="ops">
    <variable name="t" units="dimensionless"/>
    <variable name="x" units="dimensionless" initial_value="0.7"/>
    <variable name="y" units="dimensionless" initial_value="1.3"/>
    <variable name="a" units="dimensionless" initial_value="0.8"/>
    <variable name="b" units="dimensionless" initial_value="1.5"/>
    <variable name="u" units="dimensionless"/>
    <variable name="v" units="dimensionless"/>
    <variable name="w" units="dimensionless"/>
    <math {MATHML}>
      <apply><eq/><ci>u</ci><apply><divide/><apply><times/><ci>a</ci><ci>x</ci><ci>y</ci></apply>
        <apply><plus/>{_cn(1)}<apply><power/><ci>y</ci>{_cn(2)}</apply></apply></apply></apply>
      <apply><eq/><ci>v</ci><apply><plus/><apply><exp/><apply><minus/><ci>u</ci></apply></apply>
        <apply><root/><apply><plus/><ci>x</ci>{_cn(2)}</apply></apply>
        <apply><ln/><apply><plus/>{_cn(1)}<apply><times/><ci>y</ci><ci>y</ci></apply></apply></apply>
        <apply><power/><ci>y</ci><ci>b</ci></apply></apply></apply>
      <apply><eq/><ci>w</ci><piecewise>
        <piece><apply><minus/>{_cn(2)}<apply><divide/>{_cn(1)}<ci>x</ci></apply></apply><apply><gt/><ci>x</ci>{_cn(1)}</apply></piece>
        <otherwise><apply><plus/><ci>x</ci><apply><sin/><apply><minus/><ci>x</ci>{_cn(1)}</apply></apply></apply></otherwise>
      </piecewise></apply>
      {_ode('x', f'<apply><minus/><apply><plus/><apply><tanh/><ci>v</ci></apply><apply><times/>{_cn(0.3)}<ci>w</ci></apply><apply><times/>{_cn(0.2)}<apply><max/><ci>x</ci>{_cn(0.2)}</apply></apply></apply><apply><times/>{_cn(1.5)}<ci>x</ci></apply></apply>')}
      {_ode('y', f'<apply><minus/><apply><plus/><apply><times/><ci>u</ci><apply><cos/><ci>y</ci></apply></apply><apply><times/>{_cn(0.5)}<apply><min/><ci>y</ci>{_cn(2)}</apply></apply><apply><arctan/><ci>x</ci></apply></apply><apply><plus/><apply><divide/><ci>y</ci><apply><plus/><ci>b</ci><apply><abs/><apply><minus/><ci>x</ci><ci>y</ci></apply></apply></apply></apply><apply><times/>{_cn(0.5)}<ci>y</ci></apply></apply></apply>')}
    </math>
  </component>""")

NLA = _cellml(f"""
  <component name="c">
    <variable name="t" units="dimensionless"/>
    <variable name="x" units="dimensionless" initial_value="1"/>
    <variable name="z" units="dimensionless" initial_value="1"/>
    <math {MATHML}>
      <apply><eq/><apply><plus/><ci>z</ci><apply><exp/><ci>z</ci></apply></apply><ci>x</ci></apply>
      {_ode('x', '<apply><minus/><ci>z</ci></apply>')}
    </math>
  </component>""")


def _write(tmp_path, name, text):
    path = tmp_path / f'{name}.cellml'
    path.write_text(text)
    return str(path)


def _kernel_model(tmp_path, name, text):
    _, am = kernel_ir.analyse_cellml(_write(tmp_path, name, text))
    return am, kernel_ir.build_kernel_model(am)


needs_c = pytest.mark.skipif(shutil.which('cmake') is None or (shutil.which('cc') or shutil.which('gcc')) is None,
                             reason='needs CMake and a C compiler')


def _helper(tmp_path, text, sim_time=1.0, dt=0.1, **solver_info):
    from libcuflynx.solver_wrappers import get_simulation_helper
    solver_info.setdefault('build_dir', str(tmp_path / 'build'))
    return get_simulation_helper(_write(tmp_path, 'model', text), 'BE_batched_c', 'cellml', dt=dt,
                                 sim_time=sim_time, solver_info=solver_info, pre_time=0.0)


@pytest.mark.unit
def test_unconnected_cells_are_blocks_of_one_kernel_and_coupled_states_one_block(tmp_path):
    _, km = _kernel_model(tmp_path, 'cells', CELLS)
    by_names = {tuple(km.states[i].component for i in b.states): b for b in km.blocks}
    assert set(by_names) == {('cell_a',), ('cell_b',), ('pair', 'pair')}
    assert by_names[('cell_a',)].kernel == by_names[('cell_b',)].kernel != by_names[('pair', 'pair')].kernel
    assert len(km.kernels) == 2
    cell = km.kernels[by_names[('cell_a',)].kernel]
    # the rate reads k through the algebraic flux, a temporary; the Jacobian is d(-k x)/dx = -k
    assert cell.rates.assignments == [('a0', ('mul', ('in', 0), ('y', 0)))]
    assert [(r, c) for r, c, _ in cell.jacobian.entries] == [(0, 0)]
    # each cell reads its own k, so no input of the cells' kernel is shared
    assert cell.shared_inputs == ()


@pytest.mark.unit
def test_algebraic_loop_is_refused(tmp_path):
    with pytest.raises(kernel_ir.KernelIRError, match='algebraic loop'):
        _kernel_model(tmp_path, 'nla', NLA)


@pytest.mark.unit
def test_derivative_rules_of_the_intermediate_form():
    tangent = lambda name, j: None                                # noqa: E731
    x = ('y', 0)
    assert kernel_ir.derivative(('mul', ('in', 0), x), 0, tangent) == ('in', 0)
    assert kernel_ir.derivative(('mul', ('in', 0), x), 1, tangent) is None
    # the quotient rule as (du - (u/v) dv) / v
    assert kernel_ir.derivative(('div', ('num', 1.0), x), 0, tangent) == \
        ('div', ('neg', ('mul', ('div', ('num', 1.0), x), ('num', 1.0))), x)


@needs_c
@pytest.mark.solver
def test_exact_jacobian_matches_central_differences(tmp_path):
    from libcuflynx.generators.cpp import batched
    am, km = _kernel_model(tmp_path, 'ops', OPS)
    assert len(km.blocks) == 1
    path, _ = batched.build(batched.sources(am, km), str(tmp_path))
    lib = batched.BatchedLibrary(path)
    states, rates, variables = np.zeros(km.state_count), np.zeros(km.state_count), np.zeros(km.variable_count)
    lib.bk_initialise(batched.ptr(states), batched.ptr(rates), batched.ptr(variables))
    lib.bk_compute_computed_constants(batched.ptr(states), batched.ptr(rates), batched.ptr(variables))
    block = km.blocks[0]
    inputs = variables[block.inputs].copy()
    n = len(block.states)

    def f(y):
        out = np.zeros(n)
        lib.bk_kernel_rates(block.kernel, 0.0, batched.ptr(y), batched.ptr(inputs), batched.ptr(out))
        return out

    rng = np.random.default_rng(1)
    for y in [states[block.states].copy()] + [rng.uniform(0.2, 2.0, n) for _ in range(5)]:
        exact = np.zeros(n * n)
        lib.bk_kernel_jacobian(block.kernel, 0.0, batched.ptr(y), batched.ptr(inputs), batched.ptr(exact))
        central = np.zeros((n, n))
        for j in range(n):
            d = 1e-6 * max(abs(y[j]), 1.0)
            up, down = y.copy(), y.copy()
            up[j] += d
            down[j] -= d
            central[:, j] = (f(up) - f(down)) / (2 * d)
        assert np.allclose(exact.reshape(n, n), central, rtol=1e-6, atol=1e-8), (y, exact, central)


@needs_c
@pytest.mark.solver
def test_solutions_converge_onto_the_analytic_ones(tmp_path):
    t = np.linspace(0.0, 3.0, 31)
    previous = None
    for rtol in (1e-4, 1e-6, 1e-8):
        helper = _helper(tmp_path, CELLS, sim_time=3.0, rtol=rtol, atol=1e-12)
        assert helper.run()
        a, b, p = (helper.get_results([[n]])[0][0] for n in ('cell_a/x', 'cell_b/x', 'pair/p'))
        worst = max(np.max(np.abs(a - 2.0 * np.exp(-0.5 * t))) / 2.0, np.max(np.abs(b - np.exp(-0.2 * t))),
                    np.max(np.abs(p - np.cos(3.0 * t))))
        if previous is not None:
            assert worst < 0.3 * previous
        previous = worst
    assert worst < 1e-3
    # algebraic variables come from libCellML's computeVariables at each output
    flux = helper.get_results([['cell_a/flux']])[0][0]
    assert np.allclose(flux, 0.5 * a, rtol=1e-12)


@needs_c
@pytest.mark.solver
def test_results_are_the_same_on_any_number_of_threads(tmp_path):
    results = []
    for threads in (1, 3):
        helper = _helper(tmp_path, CELLS, sim_time=5.0, rtol=1e-5, threads=threads)
        assert helper.run()
        results.append(np.array(helper.get_all_results(flatten=True)))
    assert np.array_equal(results[0], results[1])


@needs_c
@pytest.mark.solver
def test_a_block_whose_inputs_did_not_change_continues_its_steps(tmp_path):
    """Changing cell_b's rate constant between sub-experiments restarts cell_b only: cell_a's outputs are the
    same bits as in one unbroken run, and cell_b follows the new constant from where it was."""
    from libcuflynx.protocol_runners.protocol_executor import ProtocolExecutor
    whole = _helper(tmp_path, CELLS, sim_time=4.0, dt=0.5, rtol=1e-6)
    assert whole.run()
    unbroken = whole.get_results([['cell_a/x']])[0][0]

    helper = _helper(tmp_path, CELLS, sim_time=2.0, dt=0.5, rtol=1e-6)
    protocol = {'pre_times': [0.0], 'sim_times': [[2.0, 2.0]],
                'params_to_change': {'cell_b/k': [[0.2, 1.0]]}}
    steps_before = helper.stats()['steps']
    ok, by_sub, _, t_by_exp = ProtocolExecutor(helper).run_protocol(protocol, result_variables=[['cell_a/x',
                                                                                                   'cell_b/x']])
    assert ok
    a = np.concatenate([by_sub[(0, 0)][0][0], by_sub[(0, 1)][0][0][1:]])
    b = np.concatenate([by_sub[(0, 0)][0][1], by_sub[(0, 1)][0][1][1:]])
    assert np.array_equal(a, unbroken)
    t = t_by_exp[0]
    expected = np.where(t <= 2.0, np.exp(-0.2 * t), np.exp(-0.4) * np.exp(-1.0 * (t - 2.0)))
    assert np.max(np.abs(b - expected)) < 2e-3
    assert helper.stats()['steps'] > steps_before


@needs_c
@pytest.mark.solver
def test_a_state_initialised_by_a_parameter_follows_it(tmp_path):
    helper = _helper(tmp_path, CELLS, sim_time=1.0, rtol=1e-6)
    helper.set_param_vals(['cell_a/x0'], [5.0])
    assert helper.run()
    x = helper.get_results([['cell_a/x']])[0][0]
    assert x[0] == 5.0
    assert abs(x[-1] - 5.0 * math.exp(-0.5)) < 5e-3 * 5.0
    with pytest.raises(ValueError, match='cannot set states'):
        helper.set_param_vals(['cell_a/x'], [1.0], change_states=False)


@needs_c
@pytest.mark.solver
def test_agrees_with_cvode_myokit(tmp_path):
    from libcuflynx.solver_wrappers import get_simulation_helper
    path = _write(tmp_path, 'ops_model', OPS)
    names = [['ops/x', 'ops/y', 'ops/v']]
    out = {}
    for solver, info in (('CVODE_myokit', {'rtol': 1e-10, 'atol': 1e-12}),
                         ('BE_batched_c', {'rtol': 1e-7, 'build_dir': str(tmp_path / 'build')})):
        helper = get_simulation_helper(path, solver, 'cellml', dt=0.05, sim_time=4.0, solver_info=info)
        assert helper.run()
        out[solver] = np.array(helper.get_results(names)[0])
    scale = np.max(np.abs(out['CVODE_myokit']), axis=1, keepdims=True)
    assert np.max(np.abs(out['BE_batched_c'] - out['CVODE_myokit']) / scale) < 2e-3


@pytest.mark.unit
def test_be_batched_c_is_a_cellml_solver_in_the_schema():
    from libcuflynx.parsers.PrimitiveParsers import SOLVER_INFO_FIELDS, SOLVER_SCHEMA
    assert 'BE_batched_c' in SOLVER_SCHEMA['solvers_by_model_type']['cellml']
    assert SOLVER_SCHEMA['stiff_suitable_methods']['BE_batched_c'] == ['backward_euler', 'trbdf2']
    from libcuflynx.generators.cpp.batched import METHODS
    assert SOLVER_SCHEMA['methods_by_solver']['BE_batched_c'] == list(METHODS)
    from libcuflynx.solver_wrappers import batched_c_helper
    defaults = {f['name']: f['default'] for f in SOLVER_INFO_FIELDS['BE_batched_c']}
    assert defaults['rtol'] == batched_c_helper.DEFAULT_RTOL
    assert defaults['atol'] == batched_c_helper.DEFAULT_ATOL
    assert defaults['MaximumNumberOfSteps'] == batched_c_helper.DEFAULT_MAX_STEPS


@needs_c
@pytest.mark.solver
def test_a_built_model_is_loaded_from_the_cache_until_its_cellml_changes(tmp_path):
    first = _helper(tmp_path, CELLS, rtol=1e-6)
    again = _helper(tmp_path, CELLS, rtol=1e-6)
    assert not first.cached and again.cached
    assert again.build_seconds == 0.0
    assert first.run() and again.run()
    assert np.array_equal(first.get_all_results(flatten=True), again.get_all_results(flatten=True))
    changed = _helper(tmp_path, CELLS.replace('initial_value="0.5"', 'initial_value="0.6"'), rtol=1e-6)
    assert not changed.cached
    assert changed.run()
    assert changed.get_results([['cell_a/k']])[0][0][0] == 0.6


@needs_c
@pytest.mark.solver
def test_a_state_initialised_by_a_computed_constant_moves_only_when_states_may_change(tmp_path):
    """cell_a's x starts at x0 = 4 k: setting k re-initialises x at an experiment's start
    (change_states=True), and leaves the evolved x alone mid-protocol (change_states=False)."""
    model = CELLS.replace(
        '<variable name="x0" units="dimensionless" initial_value="2.0"/>',
        '<variable name="x0" units="dimensionless"/>').replace(
        '<apply><eq/><ci>flux</ci>',
        f'<apply><eq/><ci>x0</ci><apply><times/>{_cn(4, "second")}<ci>k</ci></apply></apply><apply><eq/><ci>flux</ci>', 1)
    helper = _helper(tmp_path, model, sim_time=1.0, rtol=1e-6)
    assert helper.run()
    assert helper.get_results([['cell_a/x']])[0][0][0] == 2.0           # 4 * 0.5
    helper.reset_states()
    helper.set_param_vals(['cell_a/k'], [1.0])
    assert helper.run()
    x = helper.get_results([['cell_a/x']])[0][0]
    assert x[0] == 4.0
    helper.set_param_vals(['cell_a/k'], [0.25], change_states=False)
    helper.update_times(helper.dt, 1.0, 1.0, 0.0)
    assert helper.run()
    assert helper.get_results([['cell_a/x']])[0][0][0] == pytest.approx(x[-1], rel=1e-12)


FAST_START = _cellml(f"""
  <component name="fast">
    <variable name="t" units="dimensionless"/>
    <variable name="v" units="dimensionless" initial_value="0"/>
    <variable name="y" units="dimensionless" initial_value="1"/>
    <math {MATHML}>
      {_ode('v', f'<apply><times/>{_cn(-1e12)}<apply><minus/><ci>v</ci><ci>y</ci></apply></apply>')}
      {_ode('y', '<apply><minus/><ci>y</ci></apply>')}
    </math>
  </component>""")


@needs_c
@pytest.mark.solver
def test_a_fast_transient_from_rest_is_stepped_through(tmp_path):
    """v relaxes onto y in ~1e-12 time units from rest, as a flow through a tiny inertance does: the first steps are
    far below 1e-14, and the error of the damped fast mode must not keep them there."""
    helper = _helper(tmp_path, FAST_START, sim_time=2.0, dt=0.5, rtol=1e-5)
    assert helper.run()
    v, y = (helper.get_results([[n]])[0][0] for n in ('fast/v', 'fast/y'))
    t = np.linspace(0.0, 2.0, 5)
    assert np.max(np.abs(y - np.exp(-t))) < 1e-2
    assert np.max(np.abs(v[1:] - y[1:])) < 1e-6
    assert helper.stats()['steps'] < 2000


@needs_c
@pytest.mark.solver
def test_trbdf2_is_second_order(tmp_path):
    """On the analytic cells, TR-BDF2's error falls about a hundredfold over two decades of rtol (second order,
    rtol^(2/3)), backward Euler's about tenfold (first order, rtol^(1/2))."""
    t = np.linspace(0.0, 3.0, 31)
    worst = {}
    for method in ('backward_euler', 'trbdf2'):
        for rtol in (1e-4, 1e-6):
            helper = _helper(tmp_path, CELLS, sim_time=3.0, rtol=rtol, atol=1e-12, method=method)
            assert helper.run()
            a, p = (helper.get_results([[n]])[0][0] for n in ('cell_a/x', 'pair/p'))
            worst[method, rtol] = max(np.max(np.abs(a - 2.0 * np.exp(-0.5 * t))) / 2.0,
                                      np.max(np.abs(p - np.cos(3.0 * t))))
    assert worst['trbdf2', 1e-6] < worst['trbdf2', 1e-4] / 15
    assert worst['backward_euler', 1e-6] > worst['backward_euler', 1e-4] / 15
    assert worst['trbdf2', 1e-6] < worst['backward_euler', 1e-6] / 10


# v follows u = 1e4 y through a factor of 1e11: its rate is the difference of terms near 1e15, whose rounding
# (~0.1) times a step is far above v's tolerance, as in a flow through a tiny inertance.
CANCELLING = _cellml(f"""
  <component name="c">
    <variable name="t" units="dimensionless"/>
    <variable name="v" units="dimensionless" initial_value="1e4"/>
    <variable name="y" units="dimensionless" initial_value="1"/>
    <math {MATHML}>
      {_ode('v', f'<apply><times/>{_cn(1e11)}<apply><minus/><apply><times/>{_cn(1e4)}<ci>y</ci></apply><ci>v</ci></apply></apply>')}
      {_ode('y', f'<apply><times/>{_cn(-0.5)}<ci>y</ci></apply>')}
    </math>
  </component>""")


@needs_c
@pytest.mark.solver
@pytest.mark.parametrize('method', ['backward_euler', 'trbdf2'])
def test_newton_converges_where_the_residual_is_all_rounding(tmp_path, method):
    helper = _helper(tmp_path, CANCELLING, sim_time=2.0, dt=0.5, rtol=1e-6, method=method)
    assert helper.run()
    v, y = (helper.get_results([[n]])[0][0] for n in ('c/v', 'c/y'))
    assert np.max(np.abs(y - np.exp(-0.5 * np.linspace(0.0, 2.0, 5)))) < 1e-3
    assert np.max(np.abs(v - 1e4 * y)) < 1e-2
    stats = helper.stats()
    assert stats['rejected_steps'] < 10 and stats['steps'] < 5000, stats


class _Instances:
    """The instance API of a built library, through ctypes."""

    def __init__(self, tmp_path, text, outputs=()):
        import ctypes
        from libcuflynx.generators.cpp import batched
        _, path, _, _ = batched.load_or_build(_write(tmp_path, 'model', text), str(tmp_path / 'build'),
                                              outputs=outputs)
        lib = ctypes.CDLL(path)
        vp, d, i, s = ctypes.c_void_p, ctypes.c_double, ctypes.c_int, ctypes.c_char_p
        dp = ctypes.POINTER(d)
        for name, res, args in [
                ('bk_instance_create', vp, []), ('bk_instance_copy', vp, [vp]), ('bk_instance_destroy', None, [vp]),
                ('bk_instance_states', dp, [vp]), ('bk_instance_variables', dp, [vp]),
                ('bk_instance_time', d, [vp]), ('bk_instance_sim', vp, [vp]), ('bk_instance_restart', None, [vp]),
                ('bk_instance_parameters_changed', None, [vp]), ('bk_instance_advance', i, [vp, d]),
                ('bk_instance_compute_variables', None, [vp]), ('bk_state_count', i, []), ('bk_variable_count', i, []),
                ('bk_instance_outputs', None, [vp]), ('bk_output_count', i, []), ('bk_output_index', i, [i]),
                ('bk_instance_create_blocks', vp, [i, ctypes.POINTER(i)]), ('bk_state_block', i, [i]),
                ('bk_instance_initialise_states', None, [vp]),
                ('bk_state_info', None, [i, ctypes.POINTER(s)] * 1 + [ctypes.POINTER(s)] * 2),
                ('bk_variable_info', i, [i] + [ctypes.POINTER(s)] * 3),
                ('bk_set_options', None, [vp, d, d, d, d, d, i, i, ctypes.c_long, i, i])]:
            f = getattr(lib, name)
            f.restype, f.argtypes = res, args
        self.lib = lib

        def names(count, info):
            out = {}
            for k in range(count):
                n, u, c = s(), s(), s()
                info(k, ctypes.byref(n), ctypes.byref(u), ctypes.byref(c))
                out[f'{c.value.decode()}/{n.value.decode()}'] = k
            return out
        self.state = names(lib.bk_state_count(), lib.bk_state_info)
        self.variable = names(lib.bk_variable_count(), lib.bk_variable_info)

    def create(self, rtol=1e-8, method=1):
        inst = self.lib.bk_instance_create()
        self.lib.bk_set_options(self.lib.bk_instance_sim(inst), rtol, 1e-12, 0.0, 0.0, 0.01, 6, 50, 10_000_000, 0,
                                method)
        return inst

    def states(self, inst):
        return np.ctypeslib.as_array(self.lib.bk_instance_states(inst), shape=(self.lib.bk_state_count(),))

    def variables(self, inst):
        return np.ctypeslib.as_array(self.lib.bk_instance_variables(inst), shape=(self.lib.bk_variable_count(),))


@needs_c
@pytest.mark.solver
def test_an_instance_stepped_by_its_caller_follows_the_analytic_solutions(tmp_path):
    """Ten world steps of 0.1 s: cell_a's x = 2 exp(-0.5 t), and the oscillator's p = cos(3 t)."""
    api = _Instances(tmp_path, CELLS)
    inst = api.create()
    for _ in range(10):
        assert api.lib.bk_instance_advance(inst, 0.1) == 0
    y = api.states(inst)
    assert api.lib.bk_instance_time(inst) == pytest.approx(1.0)
    assert y[api.state['cell_a/x']] == pytest.approx(2.0 * math.exp(-0.5), rel=1e-6)
    assert y[api.state['pair/p']] == pytest.approx(math.cos(3.0), rel=1e-5)
    api.lib.bk_instance_destroy(inst)


@needs_c
@pytest.mark.solver
def test_a_changed_parameter_takes_effect_from_the_next_step_and_keeps_the_states(tmp_path):
    """cell_a's x starts at x0 = 4 k. Setting k to 0.25 at t = 1 changes x's rate from there on, and leaves x
    where it was (the computed constant x0 changes, the state does not)."""
    model = CELLS.replace(
        '<variable name="x0" units="dimensionless" initial_value="2.0"/>',
        '<variable name="x0" units="dimensionless"/>').replace(
        '<apply><eq/><ci>flux</ci>',
        f'<apply><eq/><ci>x0</ci><apply><times/>{_cn(4, "second")}<ci>k</ci></apply></apply><apply><eq/><ci>flux</ci>', 1)
    api = _Instances(tmp_path, model)
    inst = api.create()
    x, k, x0 = api.state['cell_a/x'], api.variable['cell_a/k'], api.variable['cell_a/x0']
    assert api.states(inst)[x] == 2.0
    for _ in range(10):
        api.lib.bk_instance_advance(inst, 0.1)
    x1 = api.states(inst)[x]
    api.variables(inst)[k] = 0.25
    api.lib.bk_instance_parameters_changed(inst)
    assert api.variables(inst)[x0] == 1.0 and api.states(inst)[x] == x1
    for _ in range(10):
        api.lib.bk_instance_advance(inst, 0.1)
    assert api.states(inst)[x] == pytest.approx(x1 * math.exp(-0.25), rel=1e-6)
    api.lib.bk_instance_destroy(inst)


@needs_c
@pytest.mark.solver
def test_instances_stepped_on_threads_match_one_thread_bit_for_bit(tmp_path):
    """Eight instances with different rate constants, stepped one by one and then from eight threads at once."""
    from concurrent.futures import ThreadPoolExecutor
    api = _Instances(tmp_path, CELLS)
    k = api.variable['cell_a/k']

    def run(threads):
        instances = [api.create(rtol=1e-4) for _ in range(8)]
        for j, inst in enumerate(instances):
            api.variables(inst)[k] = 0.1 * (j + 1)
            api.lib.bk_instance_parameters_changed(inst)

        def step(inst):
            for _ in range(50):
                api.lib.bk_instance_advance(inst, 0.1)
            return api.states(inst).copy()
        if threads == 1:
            out = [step(inst) for inst in instances]
        else:
            with ThreadPoolExecutor(threads) as pool:
                out = list(pool.map(step, instances))
        for inst in instances:
            api.lib.bk_instance_destroy(inst)
        return np.array(out)
    one, many = run(1), run(8)
    assert np.array_equal(one, many)
    assert len(set(one[:, api.state['cell_a/x']])) == 8


@needs_c
@pytest.mark.solver
def test_a_copy_continues_from_its_original(tmp_path):
    """A daughter: a copy at t = 0.5 carries on exactly as its original (same steps, bit for bit)."""
    api = _Instances(tmp_path, CELLS)
    inst = api.create()
    for _ in range(5):
        api.lib.bk_instance_advance(inst, 0.1)
    copy = api.lib.bk_instance_copy(inst)
    assert api.lib.bk_instance_time(copy) == api.lib.bk_instance_time(inst)
    assert np.array_equal(api.states(copy), api.states(inst))
    for _ in range(5):
        api.lib.bk_instance_advance(inst, 0.1)
        api.lib.bk_instance_advance(copy, 0.1)
    assert np.array_equal(api.states(copy), api.states(inst))
    api.lib.bk_instance_destroy(copy)
    api.lib.bk_instance_destroy(inst)


@needs_c
@pytest.mark.solver
def test_outputs_are_computed_alone_as_libcellml_computes_them(tmp_path):
    """Built with cell_b/flux as its output, an instance computes that flux (and only what it reads) after a step,
    exactly as libCellML's computeVariables does; cell_a's flux is left alone."""
    api = _Instances(tmp_path, CELLS, outputs=('cell_b/flux',))
    assert api.lib.bk_output_count() == 1
    flux_b, flux_a = api.variable['cell_b/flux'], api.variable['cell_a/flux']
    assert api.lib.bk_output_index(0) == flux_b
    inst = api.create()
    for _ in range(3):
        api.lib.bk_instance_advance(inst, 0.1)
    v = api.variables(inst)
    v[flux_a] = v[flux_b] = -1.0
    api.lib.bk_instance_outputs(inst)
    assert v[flux_a] == -1.0
    computed = v[flux_b]
    api.lib.bk_instance_compute_variables(inst)
    assert computed == v[flux_b] == 0.2 * api.states(inst)[api.state['cell_b/x']]
    api.lib.bk_instance_destroy(inst)


@pytest.mark.unit
def test_an_output_must_name_one_algebraic_variable(tmp_path):
    am, _ = _kernel_model(tmp_path, 'cells', CELLS)
    with pytest.raises(kernel_ir.KernelIRError, match='names 2 algebraic variables or computed constants'):
        kernel_ir.outputs_program(am, ['flux'])
    assert len(kernel_ir.outputs_program(am, ['cell_a/flux'])) == 1


TRACED = _cellml(f"""
  <component name="drive">
    <variable name="t" units="second"/>
    <variable name="u" units="per_s" initial_value="0"/>
    <variable name="x" units="dimensionless" initial_value="0"/>
    <variable name="flux" units="per_s"/>
    <math {MATHML}>
      <apply><eq/><ci>flux</ci><ci>u</ci></apply>
      {_ode('x', '<ci>flux</ci>')}
    </math>
  </component>""")


def _ramp_protocol():
    """u ramps 0 -> 2 over 0.5 s, then holds; x = integral of u."""
    return {'pre_times': [0.0], 'sim_times': [[1.0]], 'params_to_change': {'drive/u': [['ramp']]},
            'protocol_traces': {'ramp': {'t': [0.0, 0.5], 'values': [0.0, 2.0]}}}


@needs_c
@pytest.mark.solver
@pytest.mark.parametrize('method', ['backward_euler', 'trbdf2'])
def test_a_constant_follows_its_protocol_trace(tmp_path, method):
    """x' = u with u a ramp to 2 over 0.5 s then held: x(t) = 2 t^2 to 0.5 s, then 0.5 + 2 (t - 0.5). Steps
    end on the breakpoint, so even first-order backward Euler gets the kink right; flux (algebraic) and u
    report the trace at every output time."""
    helper = _helper(tmp_path, TRACED, sim_time=1.0, dt=0.05, rtol=1e-8, atol=1e-12, method=method)
    protocol = _ramp_protocol()
    helper.set_protocol_info(protocol)
    helper.set_param_vals(['drive/u'], ['ramp'])
    assert helper.run()
    t = helper.get_time()
    x, flux, u = helper.get_results([['drive/x', 'drive/flux', 'drive/u']])[0]
    expected = np.where(t <= 0.5, 2 * t ** 2, 0.5 + 2 * (t - 0.5))
    assert x == pytest.approx(expected, abs=2e-4 if method == 'backward_euler' else 1e-6)
    assert flux == pytest.approx(np.interp(t, [0.0, 0.5], [0.0, 2.0]))
    assert u == pytest.approx(flux)
    assert 'drive.u' in helper.get_all_variable_names()                # a traced parameter is an output
    helper.set_param_vals(['drive/u'], [0.0], change_states=False)     # a number again ends the trace
    helper.update_times(helper.dt, 1.0, 1.0, 0.0)
    assert helper.run()
    x, u = helper.get_results([['drive/x', 'drive/u']])[0]
    assert x == pytest.approx(np.full(len(t), x[-1])) and np.all(u == 0.0)


@needs_c
@pytest.mark.solver
def test_a_trace_runs_through_the_protocol_executor_as_on_cvode_myokit(tmp_path):
    """The same ramp protocol through ProtocolExecutor on BE_batched_c and on CVODE_myokit."""
    from libcuflynx.protocol_runners.protocol_executor import ProtocolExecutor
    from libcuflynx.solver_wrappers import get_simulation_helper
    path = _write(tmp_path, 'model', TRACED)
    out = {}
    for solver, info in [('BE_batched_c', {'build_dir': str(tmp_path / 'build'), 'rtol': 1e-8, 'atol': 1e-12,
                                           'method': 'trbdf2'}),
                         ('CVODE_myokit', {'rtol': 1e-10, 'atol': 1e-12, 'MaximumStep': 0.01})]:
        helper = get_simulation_helper(path, solver, 'cellml', dt=0.05, sim_time=1.0, solver_info=info,
                                       pre_time=0.0)
        helper.set_protocol_info(_ramp_protocol())          # as ProtocolRunner hands it over
        ok, by_sub, _, _ = ProtocolExecutor(helper).run_protocol(_ramp_protocol(),
                                                                 result_variables=[['drive/x']])
        assert ok
        out[solver] = np.asarray(by_sub[(0, 0)][0][0] if len(by_sub[(0, 0)]) == 1 else by_sub[(0, 0)][0],
                                 dtype=float).ravel()
    assert out['BE_batched_c'] == pytest.approx(out['CVODE_myokit'], abs=1e-6)


@needs_c
@pytest.mark.solver
def test_a_trace_reaches_the_computed_constants_it_feeds(tmp_path):
    """k = 2 u is a computed constant (it reads only constants), and x' = k: with u traced, the kernel computes k
    along the trace, so x = 2 * integral of u."""
    model = TRACED.replace('<variable name="flux" units="per_s"/>',
                           '<variable name="flux" units="per_s"/><variable name="k" units="per_s"/>').replace(
        '<apply><eq/><ci>flux</ci><ci>u</ci></apply>',
        f'<apply><eq/><ci>k</ci><apply><times/>{_cn(2)}<ci>u</ci></apply></apply>'
        '<apply><eq/><ci>flux</ci><ci>k</ci></apply>')
    helper = _helper(tmp_path, model, sim_time=1.0, dt=0.05, rtol=1e-8, atol=1e-12, method='trbdf2')
    helper.set_protocol_info(_ramp_protocol())
    helper.set_param_vals(['drive/u'], ['ramp'])
    assert helper.run()
    t = helper.get_time()
    x, k = helper.get_results([['drive/x', 'drive/k']])[0]
    assert x == pytest.approx(2 * np.where(t <= 0.5, 2 * t ** 2, 0.5 + 2 * (t - 0.5)), abs=1e-6)
    assert k == pytest.approx(2 * np.interp(t, [0.0, 0.5], [0.0, 2.0]))


@pytest.mark.unit
def test_a_variable_set_by_a_literal_equation_reaches_the_rates(tmp_path):
    """c = 0.5 written as an equation (libCellML: a computed constant, defined in an equation of type 'constant')
    is computed in the kernel like any other computed constant: x' = -c x."""
    model = TRACED.replace('<variable name="flux" units="per_s"/>',
                           '<variable name="flux" units="per_s"/><variable name="c" units="per_s"/>').replace(
        '<apply><eq/><ci>flux</ci><ci>u</ci></apply>',
        f'<apply><eq/><ci>c</ci>{_cn(0.5, "per_s")}</apply>'
        '<apply><eq/><ci>flux</ci><apply><times/><apply><minus/><ci>c</ci></apply><ci>x</ci></apply></apply>')
    am, km = _kernel_model(tmp_path, 'literal', model)
    kernel = km.kernels[km.blocks[0].kernel]
    assert any(e == ('num', 0.5) for _, e in kernel.rates.assignments)


@pytest.mark.unit
def test_its_own_defaults_not_cvodes_fill_missing_settings():
    """The cellml defaults are CVODE_myokit's: BE_batched_c gets its own (no step cap, 10M steps per block), so
    a protocol run is not stopped by CVODE's 5000 steps; CVODE_myokit keeps its defaults."""
    from libcuflynx.parsers.PrimitiveParsers import _solver_info_default_for, solver_field_defaults
    be = _solver_info_default_for('cellml', 'BE_batched_c')
    assert be['MaximumNumberOfSteps'] == 10_000_000 and 'MaximumStep' not in be and be['rtol'] == 1e-4
    assert _solver_info_default_for('cellml', 'CVODE_myokit')['MaximumStep'] == 0.001
    assert solver_field_defaults('BE_batched_c')['MaximumStep'] is None


@pytest.mark.solver
def test_a_protocol_runner_gives_the_solver_its_own_step_defaults(tmp_path, monkeypatch):
    from libcuflynx.protocol_runners import protocol_runner
    seen = {}

    def fake_helper(**kwargs):
        seen.update(kwargs['solver_info'])
        raise RuntimeError('stop here')
    monkeypatch.setattr(protocol_runner, 'get_simulation_helper', fake_helper)
    with pytest.raises(RuntimeError, match='stop here'):
        protocol_runner.ProtocolRunner(_write(tmp_path, 'm', CELLS), inp_data_dict={'dt': 0.01, 'solver_info': {}},
                                       solver='BE_batched_c', model_type='cellml')
    assert seen['MaximumNumberOfSteps'] == 10_000_000 and seen['MaximumStep'] is None


@pytest.mark.unit
def test_build_dir_is_a_setting_configs_may_give():
    from libcuflynx.parsers.PrimitiveParsers import validate_solver_info
    validate_solver_info('BE_batched_c', {'rtol': 1e-4, 'build_dir': '/tmp/x'})


@pytest.mark.unit
def test_the_build_digest_needs_no_source_files_on_disk(tmp_path, monkeypatch):
    """A frozen application ships the templates but not the generator's .py files: the digest then uses the
    installed version instead of failing."""
    from libcuflynx.generators.cpp import batched
    path = _write(tmp_path, 'm', CELLS)
    on_disk = batched.source_digest(path)
    monkeypatch.setattr(batched.kernel_ir, '__file__', str(tmp_path / 'not_shipped' / 'kernel_ir.py'))
    assert batched.source_digest(path) != on_disk


@needs_c
@pytest.mark.solver
def test_initial_values_follow_the_parameters_they_name(tmp_path):
    """cell_a's x starts at x0 (a constant) and, in a second model, at x0 = 4 k (a computed constant): after the
    parameter changes, bk_instance_initialise_states starts x from the new value."""
    api = _Instances(tmp_path, CELLS)
    inst = api.create()
    api.variables(inst)[api.variable['cell_a/x0']] = 5.0
    api.lib.bk_instance_initialise_states(inst)
    assert api.states(inst)[api.state['cell_a/x']] == 5.0
    assert api.states(inst)[api.state['cell_b/x']] == 1.0
    api.lib.bk_instance_destroy(inst)

    model = CELLS.replace(
        '<variable name="x0" units="dimensionless" initial_value="2.0"/>',
        '<variable name="x0" units="dimensionless"/>').replace(
        '<apply><eq/><ci>flux</ci>',
        f'<apply><eq/><ci>x0</ci><apply><times/>{_cn(4, "second")}<ci>k</ci></apply></apply><apply><eq/><ci>flux</ci>', 1)
    (tmp_path / 'computed').mkdir()
    api = _Instances(tmp_path / 'computed', model)
    inst = api.create()
    api.variables(inst)[api.variable['cell_a/k']] = 0.25
    api.lib.bk_instance_initialise_states(inst)
    assert api.states(inst)[api.state['cell_a/x']] == 1.0
    api.lib.bk_instance_destroy(inst)


@needs_c
@pytest.mark.solver
def test_an_instance_of_some_blocks_steps_only_those(tmp_path):
    """An instance of cell_a's block alone: its x decays, the other blocks' states stay where they were; a copy
    keeps the same blocks."""
    import ctypes
    api = _Instances(tmp_path, CELLS)
    b = api.lib.bk_state_block(api.state['cell_a/x'])
    assert b >= 0 and api.lib.bk_state_block(api.state['pair/p']) != b
    blocks = (ctypes.c_int * 1)(b)
    inst = api.lib.bk_instance_create_blocks(1, blocks)
    api.lib.bk_set_options(api.lib.bk_instance_sim(inst), 1e-8, 1e-12, 0.0, 0.0, 0.01, 6, 50, 10_000_000, 0, 1)
    copy = None
    for step in range(10):
        if step == 5:
            copy = api.lib.bk_instance_copy(inst)
        api.lib.bk_instance_advance(inst, 0.1)
        if copy:
            api.lib.bk_instance_advance(copy, 0.1)
    for one in (inst, copy):
        y = api.states(one)
        assert y[api.state['cell_a/x']] == pytest.approx(2.0 * math.exp(-0.5), rel=1e-6)
        assert y[api.state['cell_b/x']] == 1.0 and y[api.state['pair/p']] == 1.0
        api.lib.bk_instance_destroy(one)


@needs_c
@pytest.mark.solver
def test_outputs_read_computed_constants_as_libcellml_computes_them(tmp_path):
    """An output that reads a computed constant (cell_b/flux = k x with k from k0 = 2 kk) uses the value
    libCellML's computeComputedConstants keeps, bit for bit the same as a full computeVariables."""
    head, cell_b = CELLS.split('<component name="cell_b">')
    cell_b = cell_b.replace('<variable name="k" units="per_s" initial_value="0.2"/>',
                            '<variable name="k" units="per_s"/><variable name="kk" units="per_s" initial_value="0.1"/>').replace(
        '<apply><eq/><ci>flux</ci><apply><times/><ci>k</ci><ci>x</ci></apply></apply>',
        f'<apply><eq/><ci>k</ci><apply><times/>{_cn(2)}<ci>kk</ci></apply></apply>'
        '<apply><eq/><ci>flux</ci><apply><times/><ci>k</ci><ci>x</ci></apply></apply>', 1)
    model = head + '<component name="cell_b">' + cell_b
    api = _Instances(tmp_path, model, outputs=('cell_b/flux',))
    inst = api.create()
    api.lib.bk_instance_advance(inst, 0.3)
    api.lib.bk_instance_outputs(inst)
    flux = api.variables(inst)[api.variable['cell_b/flux']]
    api.lib.bk_instance_compute_variables(inst)
    assert flux == api.variables(inst)[api.variable['cell_b/flux']]
    assert all(k != api.variable['cell_b/k'] for k, _ in kernel_ir.outputs_program(
        kernel_ir.analyse_cellml(str(tmp_path / 'model.cellml'))[1], ['cell_b/flux']))


@pytest.mark.unit
@pytest.mark.parametrize('method, expected', [(None, 'backward_euler'), ('trbdf2', 'trbdf2'),
                                              ('backward_euler', 'backward_euler')])
def test_a_user_inputs_config_selects_it(base_user_inputs, temp_output_dir, temp_generated_models_dir,
                                         method, expected):
    """solver: BE_batched_c in user_inputs passes validation with each of its methods, defaulting to its first,
    and gets its own step limit."""
    from libcuflynx.parsers.PrimitiveParsers import YamlFileParser
    solver_info = {'rtol': 1e-5, 'threads': 2, **({'method': method} if method else {})}
    config = dict(base_user_inputs, file_prefix='3compartment', input_param_file='3compartment_parameters.csv',
                  model_type='cellml', solver='BE_batched_c', solver_info=solver_info,
                  generated_models_dir=temp_generated_models_dir, param_id_output_dir=temp_output_dir)
    info = YamlFileParser().parse_user_inputs_file(config, obs_path_needed=False)['solver_info']
    assert info['solver'] == 'BE_batched_c' and info['method'] == expected
    assert info['rtol'] == 1e-5 and info['threads'] == 2 and info['MaximumNumberOfSteps'] == 10_000_000


@pytest.mark.unit
def test_a_user_inputs_config_with_another_solvers_method_is_refused(base_user_inputs, temp_output_dir,
                                                                     temp_generated_models_dir):
    from libcuflynx.parsers.PrimitiveParsers import YamlFileParser
    config = dict(base_user_inputs, file_prefix='3compartment', input_param_file='3compartment_parameters.csv',
                  model_type='cellml', solver='BE_batched_c', solver_info={'method': 'CVODE'},
                  generated_models_dir=temp_generated_models_dir, param_id_output_dir=temp_output_dir)
    with pytest.raises(SystemExit):
        YamlFileParser().parse_user_inputs_file(config, obs_path_needed=False)


# Two unconnected decays, slow enough for long steps: x = x0 exp(-k t).
DECAYS = _cellml(_cell('cell_a', 0.05, 2.0) + _cell('cell_b', 0.2, 1.0) + """
  <component name="environment"><variable name="t" units="second" interface="public"/></component>
  <connection component_1="environment" component_2="cell_a"><map_variables variable_1="t" variable_2="t"/></connection>
  <connection component_1="environment" component_2="cell_b"><map_variables variable_1="t" variable_2="t"/></connection>
""")


def _stepper(tmp_path, model, stop_at_outputs, rtol=1e-6):
    """A BE_batched_c (TR-BDF2) helper whose BatchedSimulation the test runs itself, a call at a time."""
    helper = _helper(tmp_path, model, rtol=rtol, atol=1e-12, method='trbdf2')
    helper.sim.set_stop_at_outputs(stop_at_outputs)
    return helper


def _index(helper, name):
    return helper._index(name)[1]


@needs_c
@pytest.mark.solver
def test_the_step_size_holds_while_the_error_control_asks_for_little_growth(tmp_path):
    """The step grows only when the error control allows more than 20%, so most steps reuse the last LU
    factors (one factorisation per step without it), and the solutions keep their accuracy."""
    t = np.linspace(0.0, 20.0, 41)
    helper = _helper(tmp_path, CELLS, sim_time=20.0, dt=0.5, rtol=1e-6, atol=1e-12, method='trbdf2')
    assert helper.run()
    stats = helper.stats()
    assert stats['factorisations'] < stats['steps'] / 4
    a, b, p = (helper.get_results([[n]])[0][0] for n in ('cell_a/x', 'cell_b/x', 'pair/p'))
    assert np.max(np.abs(a - 2.0 * np.exp(-0.5 * t))) / 2.0 < 2e-5     # 1.6e-5 with a new LU every step
    assert np.max(np.abs(b - np.exp(-0.2 * t))) < 2e-5
    assert np.max(np.abs(p - np.cos(3.0 * t))) < 2e-3


@needs_c
@pytest.mark.solver
def test_with_stop_at_outputs_each_run_ends_on_a_step(tmp_path):
    """Twenty runs of 0.1 s: by default steps far longer than 0.1 s pass the runs' ends; with stop_at_outputs every
    run ends on a step of its own. The solutions are as accurate either way."""
    steps = {}
    for stop in (False, True):
        helper = _stepper(tmp_path, DECAYS, stop, rtol=1e-4)
        for k in range(20):
            assert helper.sim.run(helper.states, helper.variables, 0.1 * k, [0.1 * (k + 1)]) == []
        steps[stop] = helper.stats()['steps']
        assert helper.states[_index(helper, 'cell_a/x')] == pytest.approx(2.0 * math.exp(-0.1), rel=1e-3)
    assert steps[False] < 20 and steps[True] >= 2 * 20


@needs_c
@pytest.mark.solver
def test_an_input_changed_where_a_run_stopped_keeps_the_jacobian_and_factors(tmp_path):
    """cell_b's k changes from 0.2 to 0.21 at t = 1. With stop_at_outputs the block restarts softly from the end of
    its own step (its Jacobian kept, its factors redone only as its step size changes); by default it restarts
    afresh. Both follow the new k."""
    new = {}
    for stop in (False, True):
        helper = _stepper(tmp_path, DECAYS, stop)
        x, k = _index(helper, 'cell_b/x'), _index(helper, 'cell_b/k')
        assert helper.sim.run(helper.states, helper.variables, 0.0, [1.0]) == []
        before = helper.stats()
        helper.variables[k] = 0.21
        assert helper.sim.run(helper.states, helper.variables, 1.0, [2.0]) == []
        after = helper.stats()
        new[stop] = {s: after[s] - before[s] for s in ('jacobians', 'factorisations')}
        assert helper.states[x] == pytest.approx(math.exp(-0.2 - 0.21), rel=1e-4)
    assert new[True]['jacobians'] == 0 and new[False]['jacobians'] >= 1
    assert new[True]['factorisations'] < new[False]['factorisations']


@needs_c
@pytest.mark.solver
def test_a_restart_from_an_interpolated_state_is_never_soft(tmp_path):
    """A run from t = 0.95, inside the block's last step to 1.0, starts from interpolated states: with an input
    changed there it restarts afresh, with a new Jacobian, even with stop_at_outputs."""
    helper = _stepper(tmp_path, DECAYS, True)
    x, k = _index(helper, 'cell_b/x'), _index(helper, 'cell_b/k')
    out = np.empty((2, helper.STATE_COUNT))
    assert helper.sim.run(helper.states, helper.variables, 0.0, [0.95, 1.0], out) == []
    before = helper.stats()
    states = out[0].copy()
    helper.variables[k] = 0.21
    assert helper.sim.run(states, helper.variables, 0.95, [1.95]) == []
    assert helper.stats()['jacobians'] > before['jacobians']
    assert states[x] == pytest.approx(math.exp(-0.2 * 0.95 - 0.21), rel=1e-4)


@needs_c
@pytest.mark.solver
def test_soft_restarts_give_the_same_bits_on_any_number_of_threads(tmp_path):
    """With stop_at_outputs and both cells' k changed at every 0.1 s run, results do not depend on the threads."""
    results = []
    for threads in (1, 3):
        helper = _stepper(tmp_path, CELLS, True, rtol=1e-5)
        ka, kb = _index(helper, 'cell_a/k'), _index(helper, 'cell_b/k')
        for j in range(30):
            helper.variables[ka] = 0.5 + 0.05 * math.sin(j)
            helper.variables[kb] = 0.2 + 0.02 * math.cos(j)
            assert helper.sim.run(helper.states, helper.variables, 0.1 * j, [0.1 * (j + 1)], threads=threads) == []
        results.append(helper.states.copy())
    assert np.array_equal(results[0], results[1])


# A motor's activation a rises from 0 towards m with a 0.3 s time constant, beside a slow state c: a restart with a
# at zero (weight 1/atol) must still take a step that moves t, however late the restart.
MOTOR = _cellml(f"""
  <component name="motor">
    <variable name="t" units="second"/>
    <variable name="a" units="dimensionless" initial_value="0"/>
    <variable name="c" units="dimensionless" initial_value="1"/>
    <variable name="m" units="dimensionless" initial_value="1"/>
    <variable name="tau" units="second" initial_value="0.3"/>
    <variable name="k" units="per_s" initial_value="0.01"/>
    <math {MATHML}>
      {_ode('a', '<apply><divide/><apply><minus/><ci>m</ci><ci>a</ci></apply><ci>tau</ci></apply>')}
      {_ode('c', '<apply><minus/><apply><times/><ci>k</ci><ci>c</ci></apply></apply>')}
    </math>
  </component>""")


@needs_c
@pytest.mark.solver
@pytest.mark.parametrize('t0', [1.0, 2040.0, 1e6])
def test_a_restart_with_a_state_at_zero_takes_a_step_that_moves_time(tmp_path, t0):
    """The first step after a restart comes from the norms over all the block's states (Hairer, Norsett & Wanner),
    not from the state at zero alone: that gave ~3e-14 s, which t + h no longer resolves beyond t ~ 1000 s."""
    helper = _stepper(tmp_path, MOTOR, False, rtol=1e-4)
    a, c = _index(helper, 'motor/a'), _index(helper, 'motor/c')
    helper.states[a] = 0.0
    assert helper.sim.run(helper.states, helper.variables, t0, [t0 + 0.1]) == []
    assert helper.states[a] == pytest.approx(1.0 - math.exp(-0.1 / 0.3), rel=1e-3)
    assert helper.states[c] == pytest.approx(math.exp(-0.001), rel=1e-6)
